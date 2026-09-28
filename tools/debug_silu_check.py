#!/usr/bin/env python3
"""Stage-by-stage check for the silu workload.

If the VRAM result is wrong, this pinpoints which SiLU sub-step the HW value
matches (or fails) by reverse-mapping the observed output against every
intermediate of x -> -x -> exp -> 1+ -> 1/ -> *x. A uniform-ratio match to a
specific stage tells you the chain broke right after that op.
"""
import numpy as np
import torch
import torch.nn.functional as F

BUILD = "build/test/silu"
FMT_EXP, FMT_MAN = 6, 5  # FP12 in VRAM


def dfp(bits):
    s = (bits >> (FMT_EXP + FMT_MAN)) & 1
    e = (bits >> FMT_MAN) & ((1 << FMT_EXP) - 1)
    m = bits & ((1 << FMT_MAN) - 1)
    bias = (1 << (FMT_EXP - 1)) - 1
    v = (m / 2**FMT_MAN) * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2**FMT_MAN) * 2.0 ** (e - bias)
    return -v if s else v


def main():
    act = torch.load(f"{BUILD}/act_tensor.pt").float().numpy()
    B, H = act.shape
    VLEN = 16

    rows = []
    for line in open(f"{BUILD}/vector_result.mem"):
        line = line.strip()
        if line:
            w = int(line, 16)
            rows.append([dfp((w >> (12 * i)) & 0xFFF) for i in range(VLEN)])
    rows = np.array(rows)
    print(f"vram rows parsed: {len(rows)}")

    # golden silu, strided layout: row = seg*B + b  (seg = h//VLEN)
    stages = {
        "x": act,
        "-x": -act,
        "exp(-x)": np.exp(-act),
        "1+exp(-x)": 1 + np.exp(-act),
        "sigmoid": 1 / (1 + np.exp(-act)),
        "silu": (act / (1 + np.exp(-act))),
    }

    # Compare HW row for (seg=0, b=0) = vram row 0 against batch0 seg0 of each stage
    hw = rows[0]
    print("\nHW vram row0:", np.round(hw, 3))
    for name, st in stages.items():
        ref = st[0, :VLEN]
        mae = float(np.abs(hw - ref).mean())
        print(f"  vs {name:11s} mae={mae:.3f}  ref={np.round(ref,3)[:6]}")

    # Full region MAE against silu golden
    g = torch.load(f"{BUILD}/golden_vram_result.pt").float().numpy()
    n = min(len(g), len(rows))
    err = np.abs(rows[:n] - g[:n]).mean(axis=1)
    print(f"\nfull region vs silu golden: rows={n} MAE={err.mean():.4f} "
          f"max-row={err.max():.3f}@{int(err.argmax())} bad(>0.3)={int((err>0.3).sum())}")


if __name__ == "__main__":
    main()
