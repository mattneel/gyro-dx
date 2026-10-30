#!/usr/bin/env python3
"""Test the compiled CLI with independently encoded miniature GGUF files.

Usage: python3 tests/integration.py ./zig-out/bin/gyro-dx
The test uses Python's standard library. It requires no model download.
"""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


U32 = 4
STRING = 8
ARRAY = 9
U64 = 10
I32 = 5


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def scalar(kind, value):
    if kind == STRING:
        return string(value)
    return struct.pack({U32: "<I", U64: "<Q", I32: "<i"}[kind], value)


def entry(name, kind, value):
    if kind == ARRAY:
        element_type, values = value
        body = struct.pack("<IQ", element_type, len(values))
        body += b"".join(scalar(element_type, item) for item in values)
    else:
        body = scalar(kind, value)
    return string(name) + struct.pack("<I", kind) + body


def aligned(value, alignment=32):
    return (value + alignment - 1) // alignment * alignment


def fixture():
    """Encode GGUF v3 without a GGUF library or application imports."""
    names = [
        f"blk.{layer}.ffn_{projection}_exps.weight"
        for layer in range(2)
        for projection in ("gate", "up", "down")
    ]
    metadata = [
        ("general.architecture", STRING, "qwen4exp"),
        ("general.alignment", U32, 32),
        ("qwen4exp.block_count", U32, 2),
        ("qwen4exp.expert_count", U32, 4),
        ("qwen4exp.expert_used_count", U32, 2),
        ("qwen4exp.embedding_length", U32, 128),
        ("qwen4exp.expert_feed_forward_length", U32, 128),
        ("qwen4exp.embedding_length_per_layer_input", U32, 160),
        ("qwen4exp.ple.ngram_size", U32, 3),
        ("qwen4exp.ple.heads_per_ngram", U32, 8),
        ("qwen4exp.ple.eos_token_id", U32, 15),
        ("qwen4exp.ple.layers", ARRAY, (I32, [1])),
        ("qwen4exp.ple.layer_multipliers", ARRAY, (U64, [1, 3, 7])),
        ("qwen4exp.ple.head_offsets", ARRAY, (U64, list(range(0, 64, 4)))),
        ("qwen4exp.ple.head_vocab_sizes", ARRAY, (U64, [4] * 16)),
        ("agention.trellis.state_bits", U32, 15),
        ("prism.hadamard.version", U32, 1),
        ("prism.hadamard.block_size", U32, 128),
        ("prism.hadamard.transform", STRING, "normalized-sylvester-walsh-hadamard"),
        ("prism.hadamard.axis", STRING, "input-last-dimension"),
        ("prism.hadamard.sign_mode", STRING, "identity"),
        ("prism.hadamard.weight_names", ARRAY, (STRING, names)),
    ]
    specs = [
        ("token_embd.weight", [128, 16], 0, 128 * 16 * 4),
        ("per_layer_token_embd.weight", [160, 64], 8, 160 // 32 * 34 * 64),
    ]
    specs += [(name, [128, 128, 4], 144, 128 * 128 * 4 // 128 * 34) for name in names]
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(specs), len(metadata)))
    for item in metadata:
        data.extend(entry(*item))
    tensors = {}
    mutations = {}
    payload_bytes = 0
    for name, dimensions, kind, size in specs:
        offset = aligned(payload_bytes)
        data.extend(string(name))
        data.extend(struct.pack("<I", len(dimensions)))
        dimension_position = len(data)
        data.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        type_position = len(data)
        data.extend(struct.pack("<I", kind))
        offset_position = len(data)
        data.extend(struct.pack("<Q", offset))
        tensors[name] = {
            "rank": len(dimensions),
            "dims": dimensions + [1] * (4 - len(dimensions)),
            "type_id": kind,
            "relative_offset": offset,
            "byte_length": size,
        }
        mutations[name] = {
            "dimension": dimension_position,
            "type": type_position,
            "offset": offset_position,
        }
        payload_bytes = offset + size
    data_offset = aligned(len(data))
    data.extend(bytes(data_offset - len(data)))
    for tensor in tensors.values():
        tensor["file_offset"] = data_offset + tensor["relative_offset"]
    data.extend(bytes(payload_bytes))
    return bytes(data), tensors, mutations, data_offset, len(metadata)


def documents(text):
    """Decode consecutive JSON documents from the CLI output."""
    decoder = json.JSONDecoder()
    result = []
    remainder = text.lstrip()
    while remainder:
        value, end = decoder.raw_decode(remainder)
        result.append(value)
        remainder = remainder[end:].lstrip()
    return result


class Runner:
    def __init__(self, executable):
        self.executable = executable
        self.calls = 0
        self.failures = 0

    def invoke(self, *arguments):
        self.calls += 1
        return subprocess.run(
            [str(self.executable), *map(str, arguments)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def json(self, *arguments, multiple=False):
        result = self.invoke(*arguments)
        assert result.returncode == 0, (arguments, result.returncode, result.stderr)
        decoded = documents(result.stdout)
        if multiple:
            return decoded
        assert len(decoded) == 1, result.stdout
        return decoded[0]

    def error(self, expected, *arguments):
        result = self.invoke(*arguments)
        assert result.returncode != 0, (arguments, result.stdout)
        assert f"gyro-dx: {expected}" in result.stderr, (arguments, result.stderr)
        self.failures += 1


def exercise(executable, directory):
    raw, tensors, positions, header_bytes, metadata_count = fixture()
    model = directory / "miniature.gguf"
    model.write_bytes(raw)
    runner = Runner(executable)

    report = runner.json("inspect", model)
    expected_summary = {
        "total_tensor_bytes": 123_520,
        "expert_bytes": 104_448,
        "resident_expert_allocation_bytes": 524_288,
        "disk_table_bytes": 10_880,
        "fixed_weight_bytes": 8_192,
        "max_expert_bundle_bytes": 13_056,
        "min_expert_bundle_bytes": 13_056,
        "expert_slots": 8,
        "layers": 2,
        "experts_per_layer": 4,
        "active_experts": 2,
        "cold_bytes_per_token": 52_224,
    }
    assert report["summary"] == expected_summary, report["summary"]
    assert report["architecture"] == "qwen4exp"
    assert report["header_only"] is False
    assert report["tensor_count"] == 8
    assert report["metadata_count"] == metadata_count
    assert report["file_bytes"] == len(raw)
    assert report["header_bytes"] == header_bytes
    assert report["header_sha256"] == hashlib.sha256(raw[:header_bytes]).hexdigest()
    prefix = directory / "header.bin"
    prefix.write_bytes(raw[:header_bytes])
    prefix_report = runner.json("inspect-prefix", prefix, len(raw))
    assert prefix_report["header_only"] is True
    assert prefix_report["summary"] == expected_summary
    print("PASS inspect and inspect-prefix")

    manifest = runner.json("manifest", model)
    assert manifest["data_offset"] == header_bytes
    assert manifest["file_bytes"] == len(raw)
    assert len(manifest["tensors"]) == len(tensors)
    for tensor in manifest["tensors"]:
        expected = tensors[tensor["name"]]
        for key, value in expected.items():
            assert tensor[key] == value, (tensor["name"], key, tensor[key], value)
    print("PASS manifest")

    resident = runner.json("plan", model, "0.0009765625", "0.00006103515625")
    assert resident == {
        "available_bytes": 1_048_576,
        "reserve_bytes": 65_536,
        "fixed_weight_bytes": 8_192,
        "resident_weight_bytes": 112_640,
        "all_experts_fit": True,
        "allocation_strategy": "sized_resident_buffers",
        "cache_slots": 8,
        "cache_slot_bytes": None,
        "cache_allocated_bytes": 524_288,
        "minimum_layer_bytes": 131_072,
        "remaining_bytes": 450_560,
    }, resident
    spill = runner.json("plan", model, "0.0003662109375", "0.00006103515625")
    assert spill["all_experts_fit"] is False
    assert spill["allocation_strategy"] == "uniform_cache_slots"
    assert spill["cache_slots"] == 4
    assert spill["cache_slot_bytes"] == 65_536
    assert spill["cache_allocated_bytes"] == 262_144
    assert spill["remaining_bytes"] == 57_344
    runner.error("InsufficientBudget", "plan", model, "0.00018310546875", "0.00006103515625")
    print("PASS plan resident and spill allocation")

    ranges = runner.json("ranges", model, 1, 3)
    assert len(ranges) == 3
    for projection, kind in zip(ranges, ("gate", "up", "down")):
        name = f"blk.1.ffn_{kind}_exps.weight"
        assert projection == {
            "name": name,
            "type_id": 144,
            "input_width": 128,
            "output_width": 128,
            "file_offset": tensors[name]["file_offset"] + 3 * 4_352,
            "bytes": 4_352,
        }, projection
    assert sum(projection["bytes"] for projection in ranges) == 13_056
    runner.error("InvalidExpert", "ranges", model, 2, 0)
    runner.error("InvalidExpert", "ranges", model, 0, 4)
    print("PASS expert ranges")

    trace = directory / "routes.tsv"
    trace.write_text("# token layer experts\n0\t0\t0\t1\n0 1 0 1\n1 0 0 1\n", encoding="utf-8")
    replay = runner.json("replay", model, trace, 4)
    assert replay == {"requests": 6, "hits": 2, "misses": 4, "loaded_bytes": 52_224, "steps": 3}, replay
    runner.error("CacheFull", "replay", model, trace, 1)
    trace.write_text("0 0 1 1\n", encoding="utf-8")
    runner.error("DuplicateRoute", "replay", model, trace, 4)
    print("PASS cache trace replay")

    token_ids = [1, 2, 15, 4]
    rows = runner.json("ple-ranges", model, *token_ids, multiple=True)
    assert len(rows) == len(token_ids)
    # Residues follow the independent XOR formulas for this fixture's token sequence.
    residues = [(0, 1), (1, 0), (1, 2), (1, 0)]
    table_offset = tensors["per_layer_token_embd.weight"]["file_offset"]
    for report, token, (two_gram, three_gram) in zip(rows, token_ids, residues):
        assert report["token"] == token
        assert len(report["rows"]) == 16
        for head, row in enumerate(report["rows"]):
            expected_row = 4 * head + (two_gram if head < 8 else three_gram)
            expected_start = table_offset + expected_row * 170
            assert row == {
                "head": head,
                "row": expected_row,
                "range": {"file_offset": expected_start, "bytes": 170},
            }, row
            assert any(
                page["file_offset"] <= expected_start
                and page["file_offset"] + page["bytes"] >= expected_start + 170
                for page in report["pages"]
            ), (row, report["pages"])
        previous_end = None
        for page in report["pages"]:
            start, size = page["file_offset"], page["bytes"]
            end = start + size
            assert start % 4096 == 0 and 0 < size and end <= len(raw)
            assert end % 4096 == 0 or end == len(raw)
            assert previous_end is None or previous_end < start
            previous_end = end
    runner.error("InvalidToken", "ple-ranges", model, 16)
    print("PASS PLE addresses, history, and page coverage")

    malformed = directory / "malformed.gguf"
    variants = [
        ("BadMagic", b"NOPE" + raw[4:]),
        ("UnsupportedVersion", raw[:4] + struct.pack("<I", 2) + raw[8:]),
        ("UnexpectedEnd", raw[:20]),
        ("InvalidTensorRange", raw[:-1]),
    ]
    name = "blk.0.ffn_gate_exps.weight"
    for error, field, value, encoding in (
        ("InvalidAlignment", "offset", 1, "<Q"),
        ("OverlappingTensors", "offset", 0, "<Q"),
        ("InvalidDimensions", "dimension", 130, "<Q"),
        ("UnsupportedTensorType", "type", 999, "<I"),
    ):
        changed = bytearray(raw)
        struct.pack_into(encoding, changed, positions[name][field], value)
        variants.append((error, changed))
    for error, payload in variants:
        malformed.write_bytes(payload)
        runner.error(error, "inspect", malformed)
    print("PASS malformed files fail with specific errors")

    # Inference commands validate arguments before loading weights, and the synthetic fixture
    # lacks the Qwen4Exp graph metadata, so binding fails with a metadata error.
    runner.error("MissingPrompt", "generate", model)
    runner.error("UnknownBackend", "generate", model, "--prompt", "hi", "--backend", "tpu")
    runner.error("InvalidArguments", "generate", "--prompt", "hi")
    runner.error("InvalidMetadata", "generate", model, "--prompt", "hi")
    runner.error("InvalidArguments", "serve")
    print("PASS inference commands reject invalid requests")
    print(f"PASS {runner.calls} CLI invocations, including {runner.failures} expected failures")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 tests/integration.py PATH_TO_GYRO_DX")
    executable = Path(sys.argv[1]).resolve()
    if not executable.is_file():
        raise SystemExit(f"Executable does not exist: {executable}")
    with tempfile.TemporaryDirectory(prefix="gyro-integration-", dir=Path.cwd()) as temporary:
        exercise(executable, Path(temporary))


if __name__ == "__main__":
    main()
