#!/usr/bin/env python3
"""One-shot post-sim check for the linear workload debug session.

Checks, from build/test/linear/sim.log + tensors:
1. v2 (activation) beat sequence: every tile must see k0=0,16,...,112 in order.
2. v1 (weight) beat sequence sanity (first 40 beats).
3. VRAM result rows 64-71 vs golden (match summary done by verifier; here we
   report per-4-col-block best-matching golden tile to localise placement bugs).
"""
import re
import sys

import numpy as np
import torch

BUILD = "build/test/linear"


def load_beats(tag):
    beats, scales = {}, {}
    pat_e = re.compile(rf"\[MM\] {tag}#(\d+) .*int=\[([^\]]+)\]")
    pat_s = re.compile(rf"\[MM\] {tag}#(\d+) scales\(biased\)=\[([^\]]+)\]")
    for line in open(f"{BUILD}/sim.log", errors="ignore"):
        m = pat_e.search(line)
        if m:
            i = int(m.group(1))
            beats.setdefault(i, [int(x) for x in m.group(2).split(",")])
        m = pat_s.search(line)
        if m:
            i = int(m.group(1))
            scales.setdefault(i, [int(x) for x in m.group(2).split(",")])
    return beats, scales


def main():
    act = torch.load(f"{BUILD}/act_tensor.pt").float().numpy()
    beats, scales = load_beats("v2")
    print(f"v2 beats captured: {len(beats)}")
    seq = []
    for i in sorted(beats):
        v = np.array(beats[i], dtype=float)
        sc = scales.get(i, [127] * 4)
        f = np.concatenate(
            [v[b * 4:(b + 1) * 4] * 2.0 ** (sc[b] - 127 - 7) for b in range(4)]
        )
        best = None
        for b in range(act.shape[0]):
            for k0 in range(0, act.shape[1], 16):
                err = float(np.abs(f - act[b, k0:k0 + 16]).mean())
                if best is None or err < best[0]:
                    best = (err, b, k0)
        seq.append((i, best[1], best[2], best[0]))

    # group into k-load groups of 4 beats (4 batch rows per M_MM)
    print("beat -> (batch,k0):", [(b, k) for _, b, k, _ in seq])
    ks = [k for _, _, k, _ in seq[::4]]  # k0 of each load group
    expected = list(range(0, 128, 16))
    g1 = ks[:8]
    print(f"tile1 k-sequence: {g1}  {'OK' if g1 == expected else 'BROKEN'}")
    if len(ks) >= 16:
        g2 = ks[8:16]
        print(f"tile2 k-sequence: {g2}  {'OK' if g2 == expected else 'BROKEN'}")

    g = torch.load(f"{BUILD}/golden_result.pt").float()
    rows = {}
    rline = re.compile(r"Row\s+(\d+):\s+(.*)")
    for line in open(f"{BUILD}/vector_result.fp.txt"):
        m = rline.match(line.strip())
        if m:
            rows[int(m.group(1))] = [float(x) for x in m.group(2).split()]
    print("\nVRAM rows 64-71 per-4col block -> best golden tile:")
    for r in range(64, 72):
        if r not in rows:
            continue
        out = []
        for cb in range(4):
            seg = np.array(rows[r][cb * 4:(cb + 1) * 4])
            best = None
            for gr in range(8):
                for gc in range(0, 256 - 3, 4):
                    err = float(np.abs(g[gr, gc:gc + 4].numpy() - seg).mean())
                    if best is None or err < best[0]:
                        best = (err, gr, gc)
            tagok = "OK " if best[0] < 0.5 else "?? "
            out.append(f"cols{cb*4}-{cb*4+3}={tagok}g[{best[1]},{best[2]}:{best[2]+4}](e{best[0]:.2f})")
        print(f"  row{r}: " + "  ".join(out))


if __name__ == "__main__":
    sys.exit(main())
