#!/usr/bin/env python3
"""Fetch one pinned metadata prefix. Never fetch a complete model file."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION = "b7d23f5036e9134b3b84e20e7040ba47bc5e0b4c"
REPO = "agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF"
FILES = {
    "coder": ("Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf", 11042656, 75796732256,
              "586e30b73a4e362ae5d15c72d30796303253b116d88c1dca7b970b9d115f9eb8"),
    "coder-l": ("Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf", 11043008, 83975625408,
                "5b14b64c3a371f0a45ac2fc90125a1488fa13a5723e0ce162a3f3ab54da4e997"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=FILES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    filename, length, full_size, checksum = FILES[args.variant]
    if args.output.exists():
        raise SystemExit("The output path already exists.")
    request = urllib.request.Request(
        f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}",
        headers={"Range": f"bytes=0-{length - 1}", "Accept-Encoding": "identity",
                 "User-Agent": "gyro-dx-header/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise SystemExit("The server did not honor the bounded range request.")
        expected_range = f"bytes 0-{length - 1}/{full_size}"
        if response.headers.get("Content-Range") != expected_range:
            raise SystemExit("The returned content range differs from the pinned file.")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != length:
            raise SystemExit("The response length differs from the pinned header size.")
        data = response.read(length + 1)
    if len(data) != length or hashlib.sha256(data).hexdigest() != checksum:
        raise SystemExit("The header failed size or SHA-256 verification.")
    with args.output.open("xb") as output:
        output.write(data)
    print(f"Header bytes: {length}")
    print(f"Full model bytes: {full_size}")
    print(f"Header SHA-256: {checksum}")
    print(f"Inspection: gyro-dx inspect-prefix {args.output} {full_size}")


if __name__ == "__main__":
    main()
