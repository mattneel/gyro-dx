#!/usr/bin/env python3
"""Download and verify the pinned Zig master into this project's .tools directory."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

VERSION = "0.18.0-dev.738+ae9a8acc7"
PACKAGES = {
    ("Linux", "x86_64"): (f"zig-x86_64-linux-{VERSION}.tar.xz", "022b2a6b7eecd95a786eba8c628707ce3c63dd56140523c29a76156e7d26b174"),
    ("Windows", "AMD64"): (f"zig-x86_64-windows-{VERSION}.zip", "67f5405920914144f8218a538055eb26589c85431ce31e3319cfd54d306c0b0d"),
    ("Windows", "x86_64"): (f"zig-x86_64-windows-{VERSION}.zip", "67f5405920914144f8218a538055eb26589c85431ce31e3319cfd54d306c0b0d"),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    key = (platform.system(), platform.machine())
    if key not in PACKAGES:
        raise SystemExit(f"Unsupported bootstrap host: {key}. Use the pinned compiler from ziglang.org.")
    filename, digest = PACKAGES[key]
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    destination = tools / "zig"
    if destination.exists():
        raise SystemExit(f"The destination already exists: {destination}. Verify its version before use.")
    with tempfile.TemporaryDirectory(prefix="zig-download-", dir=tools) as temp:
        work = Path(temp)
        archive = work / filename
        checksum = hashlib.sha256()
        request = urllib.request.Request("https://ziglang.org/builds/" + filename,
                                         headers={"User-Agent": "gyro-dx-bootstrap/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise SystemExit("The Zig archive failed SHA-256 verification.")
        unpack = work / "unpack"
        unpack.mkdir()
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    resolved = (unpack / member.filename).resolve()
                    if unpack.resolve() not in resolved.parents:
                        raise SystemExit("The archive contains an invalid path.")
                package.extractall(unpack)
        else:
            with tarfile.open(archive, "r:xz") as package:
                package.extractall(unpack, filter="data")
        roots = list(unpack.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise SystemExit("The archive has an unexpected directory layout.")
        shutil.move(str(roots[0]), destination)
    executable = destination / ("zig.exe" if os.name == "nt" else "zig")
    print(executable)


if __name__ == "__main__":
    main()
