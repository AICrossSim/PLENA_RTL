"""Layer test-case registry + suite generator for the looped SimTop harness.

A "test case" is a Python-defined layer (op + shapes), reusing the existing
per-op workload generators. The suite generator lays every case out at a SINGLE
fixed instruction-storage offset so one Verilator build can serve all cases
(SimTop_suite_tb.py backdoor-reloads each case's HBM/SRAM between runs).

Add a case by appending to `default_suite()` — no JSON, just a factory that
builds the existing generator.

Usage (generate all case artifacts under build/test/suite/<name>/):
    python -m tools.testworkloads.test_suite --root build/test/suite
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJECT_PATH / "tools", _PROJECT_PATH / "PLENA_Tools", _PROJECT_PATH / "PLENA_Compiler"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plena_utils.load_config import load_precision_from_svh
from plena_utils.config import (
    calculate_instr_storage_offset_from_shapes,
    update_instruction_storage_offset,
)
from cfl_tools import SRC_PATH

from .linear import LinearWorkload
from .rms_norm import RMSNormWorkload
from .silu import SiLUWorkload


# A case = (name, op, factory(build_dir) -> WorkloadGenerator, data_tensor_shapes).
# data_tensor_shapes is used only to size the shared fixed instruction offset.
def default_suite():
    """The default layer test suite. Append cases here."""
    return [
        Case(
            "linear_8x128x256",
            lambda d: LinearWorkload(batch_size=8, in_features=128, out_features=256, build_dir=d, seed=42),
            [(8, 128), (128, 256)],
        ),
        Case(
            "rms_norm_8x128",
            lambda d: RMSNormWorkload(batch_size=8, hidden_size=128, eps=1e-6, build_dir=d, seed=42),
            [(8, 128)],
        ),
        Case(
            "silu_8x128",
            lambda d: SiLUWorkload(batch_size=8, hidden_size=128, build_dir=d, seed=42),
            [(8, 128)],
        ),
    ]


class Case:
    def __init__(self, name, factory, data_shapes):
        self.name = name
        self.factory = factory
        self.data_shapes = data_shapes


def compute_shared_offset(cases, hbm_row_width=256, headroom_rows=8):
    """Pick one instruction-storage offset large enough for EVERY case's data.

    Returns a byte offset = max over cases of (that case's data size) rounded up
    to a row boundary, plus a few rows of headroom. All cases place their
    instructions here so a single build's INSTRUCTION_STORAGE_OFFSET is valid.
    """
    precision_settings, _ = load_precision_from_svh(SRC_PATH / "definitions")
    bytes_per_row = hbm_row_width // 8
    max_off = 0
    for c in cases:
        off = calculate_instr_storage_offset_from_shapes(
            c.data_shapes, precision_settings, hbm_row_width
        )
        max_off = max(max_off, off)
    # add headroom and align to row
    max_off += headroom_rows * bytes_per_row
    # round up to a 256-byte boundary for cleanliness
    max_off = ((max_off + 255) // 256) * 256
    return max_off


def generate_suite(cases, root, shared_offset=None):
    """Generate all case artifacts under root/<name>/.

    generate_hbm always packs a case's instructions right after its data (at the
    case's NATURAL offset; the instr_storage_offset param is decorative). So each
    case is generated normally, we record its natural offset, and the single
    build uses a FIXED shared offset >= every natural offset. The suite tb then
    backdoor-RELOCATES each case's instructions from its natural offset to the
    fixed offset when loading.

    Returns (shared_offset, [(name, case_dir, natural_offset)]).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    precision_settings, _ = load_precision_from_svh(SRC_PATH / "definitions")
    if shared_offset is None:
        shared_offset = compute_shared_offset(cases)

    case_info = []
    for c in cases:
        case_dir = root / c.name
        case_dir.mkdir(parents=True, exist_ok=True)
        # natural offset = byte addr where generate_hbm places this case's
        # instructions (right after data).
        nat = calculate_instr_storage_offset_from_shapes(
            c.data_shapes, precision_settings, hbm_row_width=256
        )
        gen = c.factory(str(case_dir))
        gen.generate()  # natural layout (no force)
        case_info.append((c.name, str(case_dir), nat))
        print(f"  generated {c.name} -> {case_dir}  (natural_offset={nat})")

    # Set the FIXED shared offset in configuration.svh for the single build.
    update_instruction_storage_offset(shared_offset, SRC_PATH / "definitions")

    manifest = {
        "shared_instr_offset": shared_offset,
        "cases": [{"name": n, "dir": d, "natural_offset": off} for n, d, off in case_info],
    }
    (root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2))
    return shared_offset, case_info


def main():
    parser = argparse.ArgumentParser(description="Generate the layer test suite")
    parser.add_argument("--root", type=str, default="build/test/suite",
                        help="Root dir for per-case artifacts")
    args = parser.parse_args()

    cases = default_suite()
    print(f"Generating {len(cases)} cases under {args.root} ...")
    offset, case_info = generate_suite(cases, args.root)
    print(f"\nFixed build INSTRUCTION_STORAGE_OFFSET = 0x{offset:X} ({offset} bytes)")
    print(f"Manifest: {Path(args.root) / 'suite_manifest.json'}")
    for n, d, off in case_info:
        print(f"  {n}: {d} (natural_offset={off})")


if __name__ == "__main__":
    main()
