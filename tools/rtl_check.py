#!/usr/bin/env python3
"""RTL compilation check using Verilator (no simulation)."""

import argparse
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
TOOLS_PATH = PROJECT_ROOT / "tools"

if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

from cfl_cocotb.runner import get_runner, _verilator_args


def check_simtop(trace: bool = False, jobs: int = None):
    """Check SimTop module compilation."""
    import os
    if jobs is None:
        jobs = min(os.cpu_count() or 8, 32)

    module = "SimTop"
    group_path = SRC_PATH / "system"
    module_path = group_path / "rtl" / f"{module}.sv"

    additional_include_paths = [
        SRC_PATH / "basic_components/common",
        SRC_PATH / "basic_components/mx_fp_operation",
        SRC_PATH / "basic_components/fp_operation",
        SRC_PATH / "basic_components/conversion",
        SRC_PATH / "basic_components/buffer",
        SRC_PATH / "basic_components/fixed_operation",
        SRC_PATH / "basic_components/int_operation",
        SRC_PATH / "basic_components/cast",
        SRC_PATH / "basic_components/systolic_gemm_mx",
        SRC_PATH / "basic_components/systolic_gemm_mxint",
        SRC_PATH / "basic_components/gemv",
        SRC_PATH / "basic_components/synopsis_ip_inst",
        SRC_PATH / "basic_components/hadamard_transform",
        SRC_PATH / "frontend",
        SRC_PATH / "control",
        SRC_PATH / "matrix_machine",
        SRC_PATH / "vector_machine",
        SRC_PATH / "scalar_machine",
        SRC_PATH / "memory/matrix_sram",
        SRC_PATH / "memory/vector_sram",
        SRC_PATH / "memory/scalar_sram",
        SRC_PATH / "memory/HBM",
        SRC_PATH / "core",
    ]

    definitions_path = [
        SRC_PATH / "definitions",
        SRC_PATH / "memory/HBM/TileLink_Lib",
    ]

    extra_build_args = ["-DSIMULATION"]

    # Build includes
    includes = [str(group_path / "rtl")]
    for p in additional_include_paths:
        includes.append(str(p / "rtl"))
    for p in definitions_path:
        includes.append(str(p))

    # Get runner
    runner = get_runner("verilator")
    tool_args = _verilator_args(hierarchical=False, trace=trace, jobs=jobs)

    # Build directory
    build_dir = group_path / "test" / "build" / module / "check"

    print(f"Building {module}...")
    print(f"Build dir: {build_dir}")
    print(f"Trace: {trace}")
    print(f"Jobs: {jobs}")
    print()

    runner.build(
        verilog_sources=[module_path],
        includes=includes,
        hdl_toplevel=module,
        build_args=[*tool_args, *extra_build_args],
        parameters={},
        build_dir=build_dir,
        waves=trace,
    )

    print()
    print("========================================")
    print("RTL Check PASSED")
    print("========================================")


def main():
    parser = argparse.ArgumentParser(description="RTL compilation check")
    parser.add_argument(
        "module",
        nargs="?",
        default="SimTop",
        help="Module to check (default: SimTop)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable trace generation (slower)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel compilation jobs (default: auto-detect, max 32)",
    )
    args = parser.parse_args()

    if args.module == "SimTop":
        check_simtop(trace=args.trace, jobs=args.jobs)
    else:
        print(f"Module {args.module} not configured.")
        print("Available modules: SimTop")
        sys.exit(1)


if __name__ == "__main__":
    main()
