#!/usr/bin/env python3
"""Verify the separately distributed raw CSV ZIP without extracting it."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    expected = json.loads((ROOT / "provenance/raw-release.json").read_text())
    digest = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        raise SystemExit("Raw ZIP SHA-256 mismatch")
    prefix = "data/raw/ctaea-sensitivity/"
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or not name.startswith(prefix):
                raise SystemExit("Unsafe or unexpected archive member: " + name)
        index = json.loads(archive.read(prefix + "index.json"))
        if len(index["runs"]) != 660:
            raise SystemExit("Expected 660 selected runs")
        checksums = archive.read(prefix + "SHA256SUMS.txt").decode().splitlines()
        for line in checksums:
            digest, name = line.split("  ", 1)
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != digest:
                raise SystemExit("CSV hash mismatch: " + name)
    print("PASS: raw ZIP identity, 660 selected runs, and %d payload checksums." % len(checksums))
    return 0


if __name__ == "__main__":
    sys.exit(main())
