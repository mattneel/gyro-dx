"""Compare Gyro DX tensor dumps against reference llama-gyro-dump output.

Usage: python tests/parity.py REFERENCE_DIR CANDIDATE_DIR [--max-rel 0.05] [--json OUT]

Both tools write files named sNNN.<tensor>-<layer>.bin: four little-endian int64 dims followed by
float32 values. The script compares every file present in both directories, plus greedy.txt
token traces when both exist, and exits nonzero when a tensor exceeds the relative tolerance or
the greedy traces differ.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
from array import array


def load(path):
    with open(path, "rb") as f:
        dims = struct.unpack("<4q", f.read(32))
        values = array("f")
        values.frombytes(f.read())
    return dims, values


def compare(a, b):
    n = min(len(a), len(b))
    diff2 = ref2 = 0.0
    max_abs = 0.0
    dot = na = nb = 0.0
    for i in range(n):
        x, y = a[i], b[i]
        d = x - y
        diff2 += d * d
        ref2 += x * x
        max_abs = max(max_abs, abs(d))
        dot += x * y
        na += x * x
        nb += y * y
    rel = math.sqrt(diff2 / ref2) if ref2 > 0 else math.sqrt(diff2)
    cos = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else 1.0
    return {"elements": n, "rel_rms": rel, "max_abs": max_abs, "cosine": cos}


def layer_key(name):
    m = re.match(r"s(\d+)\.(.*?)(?:-(\d+))?\.bin$", name)
    if not m:
        return (0, name, -1)
    return (int(m.group(1)), m.group(2), int(m.group(3)) if m.group(3) else -1)


def read_tokens(path):
    with open(path) as f:
        return [int(line) for line in f.read().split() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--max-rel", type=float, default=0.05)
    parser.add_argument("--json")
    args = parser.parse_args()

    names = sorted(set(os.listdir(args.reference)) & set(os.listdir(args.candidate)), key=layer_key)
    results = {}
    failed = False
    for name in names:
        if not name.endswith(".bin"):
            continue
        ref_dims, ref = load(os.path.join(args.reference, name))
        cand_dims, cand = load(os.path.join(args.candidate, name))
        if len(ref) != len(cand):
            print(f"{name}: element count differs ({len(ref)} vs {len(cand)})")
            failed = True
            continue
        r = compare(ref, cand)
        # Expert IDs are compared as sets per token: the order within the top-k is not meaningful.
        if ".ffn_moe_topk" in name:
            k = ref_dims[0]
            same = sum(set(ref[i:i + k]) == set(cand[i:i + k]) for i in range(0, len(ref), k))
            r = {"tokens": len(ref) // k, "identical_sets": same}
            ok = same == len(ref) // k
        else:
            ok = r["rel_rms"] <= args.max_rel
        results[name] = r
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {r}")

    for trace in ("greedy.txt",):
        a, b = os.path.join(args.reference, trace), os.path.join(args.candidate, trace)
        if os.path.exists(a) and os.path.exists(b):
            ta, tb = read_tokens(a), read_tokens(b)
            n = min(len(ta), len(tb))
            first = next((i for i in range(n) if ta[i] != tb[i]), None)
            same = first is None and len(ta) == len(tb)
            results[trace] = {"reference": len(ta), "candidate": len(tb), "first_difference": first}
            failed |= not same
            print(f"{'ok  ' if same else 'FAIL'} {trace}: {results[trace]}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
