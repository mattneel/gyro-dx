#!/usr/bin/env python3
"""Generate finite-scale decoder fixtures from the pinned AgentionAI C oracle."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "third_party" / "agention-gyro"
TYPES = [(144, 34), (145, 26), (146, 30), (147, 42),
         (148, 50), (149, 58), (150, 22), (151, 18)]
SEED = 0x4759524F


def generate():
    rng = random.Random(SEED)
    output = bytearray(b"GYROV1\0\0" + struct.pack("<I", len(TYPES) * 64))
    compiler = os.environ.get("CC", "cc")
    with tempfile.TemporaryDirectory(prefix="gyro-oracle-", dir=ROOT) as tmp:
        library = Path(tmp) / "libtq_oracle.so"
        command = [compiler, "-std=c11", "-O2", "-ffp-contract=off", "-fPIC", "-shared",
                   str(ORACLE / "src/tq_ref.c"), str(ORACLE / "src/tq_codebook.c"),
                   "-lm", "-o", str(library)]
        subprocess.run(command, check=True)
        reference = ctypes.CDLL(str(library))
        decode = reference.tq_dequantize_row
        decode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        decode.restype = ctypes.c_int
        rotate = reference.tq_hadamard_rotate
        rotate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int, ctypes.POINTER(ctypes.c_int8)]
        rotate.restype = ctypes.c_int
        unrotate = reference.tq_hadamard_unrotate
        unrotate.argtypes = rotate.argtypes
        unrotate.restype = ctypes.c_int
        edge_scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF,
                       0x0400, 0x8400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF]
        for type_id, size in TYPES:
            for case in range(64):
                scale = edge_scales[case] if case < len(edge_scales) else rng.randrange(65536)
                while scale & 0x7C00 == 0x7C00:
                    scale = rng.randrange(65536)
                raw = struct.pack("<H", scale) + bytes(rng.randrange(256) for _ in range(size - 2))
                packed = ctypes.create_string_buffer(raw)
                decoded = (ctypes.c_float * 128)()
                if decode(type_id, packed, decoded, 128) != 0:
                    raise RuntimeError("The C oracle rejected a valid fixture.")
                output += struct.pack("<II", type_id, size) + raw
                output += struct.pack("<128f", *decoded)
        output += struct.pack("<I", 2)
        for width in [640, 2560]:
            source = [rng.uniform(-2.0, 2.0) for _ in range(width)]
            values = (ctypes.c_float * width)(*source)
            signs = (ctypes.c_int8 * width)(*(rng.choice([-1, 1]) for _ in range(width)))
            output += struct.pack("<II", width, 128)
            output += struct.pack(f"<{width}f", *values)
            output += bytes(sign & 255 for sign in signs)
            if rotate(values, width, 128, signs) != 0:
                raise RuntimeError("The C oracle rejected a valid rotation.")
            output += struct.pack(f"<{width}f", *values)
            if unrotate(values, width, 128, signs) != 0:
                raise RuntimeError("The C oracle rejected a valid inverse rotation.")
            output += struct.pack(f"<{width}f", *values)
    destination = ROOT / "tests/gyro_vectors.bin"
    destination.write_bytes(output)
    summary = {
        "upstream": "https://gitlab.com/agentionai-group/agention-gyro",
        "revision": "587a5269848eb07996c9c1433f9ad638604eb4d8",
        "oracle": "src/tq_ref.c:tq_dequantize_row",
        "seed": SEED,
        "decode_cases": 512,
        "decoded_values": 65536,
        "rotation_widths": [640, 2560],
        "finite_scale_bits": "All signs, zero, subnormal, normal, maximum finite, and deterministic random values.",
        "sha256": hashlib.sha256(output).hexdigest(),
    }
    (ROOT / "tests/gyro_vectors.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate()
