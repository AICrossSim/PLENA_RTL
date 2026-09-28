#!/usr/bin/env python3
"""View and compare RTL simulation memory results.

This tool parses HBM result files from RTL simulation and compares them
with golden results, similar to view_mem.py for the transactional emulator.

Usage:
    python tools/view_rtl_mem.py /path/to/build_dir
    python tools/view_rtl_mem.py /path/to/build_dir --compare
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project paths
_PROJECT_PATH = Path(__file__).resolve().parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"

for p in [str(_TOOLS_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def mx_to_float(element: int, scale: int, exp_width: int = 4, man_width: int = 3) -> float:
    """Convert MX format element + scale to float.

    Args:
        element: MX element value (sign + exponent + mantissa)
        scale: Shared scaling factor (8-bit exponent bias)
        exp_width: Exponent bit width
        man_width: Mantissa bit width

    Returns:
        Floating point value
    """
    total_width = 1 + exp_width + man_width

    # Extract fields
    sign = (element >> (exp_width + man_width)) & 1
    exponent = (element >> man_width) & ((1 << exp_width) - 1)
    mantissa = element & ((1 << man_width) - 1)

    # MX format: value = (-1)^sign * 2^(scale - bias) * 2^(exp - bias) * (1 + mantissa/2^man_width)
    # Simplified: value = (-1)^sign * mantissa_val * 2^(scale + exp - 2*bias)

    bias = (1 << (exp_width - 1)) - 1 if exp_width > 0 else 0
    scale_bias = 127  # FP8 E5M2 scale bias

    if exponent == 0:
        # Subnormal or zero
        if mantissa == 0:
            return 0.0
        # Subnormal: implicit leading 0
        mantissa_val = mantissa / (1 << man_width)
        exp_val = 1 - bias
    elif exponent == (1 << exp_width) - 1:
        # Inf/NaN
        if mantissa == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        # Normal: implicit leading 1
        mantissa_val = 1 + mantissa / (1 << man_width)
        exp_val = exponent - bias

    # Apply scale (scale is the shared exponent bias)
    # For MX format: actual_exp = element_exp + scale - element_bias - scale_bias
    total_exp = exp_val + (scale - scale_bias)

    result = mantissa_val * (2 ** total_exp)
    return -result if sign else result


def parse_hbm_result_file(result_file: Path) -> Dict[str, List[Dict[str, int]]]:
    """Parse the combined HBM result file from RTL simulation.

    Args:
        result_file: Path to hbm_result.mem

    Returns:
        Dictionary with sections 'm_element', 'v_element', 'm_scale', 'v_scale'
    """
    result = {
        "m_element": [],
        "v_element": [],
        "m_scale": [],
        "v_scale": [],
    }

    if not result_file.exists():
        return result

    current_section = None
    section_map = {
        "M_ELEMENT": "m_element",
        "V_ELEMENT": "v_element",
        "M_SCALE": "m_scale",
        "V_SCALE": "v_scale",
    }

    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for section header
            if line.startswith("// =========="):
                for section_key, section_name in section_map.items():
                    if section_key in line:
                        current_section = section_name
                        break
                continue

            # Skip other comments
            if line.startswith("//"):
                continue

            # Parse data line: @addr data
            if line.startswith("@") and current_section:
                parts = line[1:].split()
                if len(parts) >= 2:
                    addr = int(parts[0], 16)
                    data = int(parts[1], 16)
                    result[current_section].append({"addr": addr, "data": data})

    return result


def parse_vector_result_file(result_file: Path, data_width: int = 128) -> List[Dict[str, int]]:
    """Parse the vector SRAM result file.

    Args:
        result_file: Path to vector_result.mem
        data_width: Width of each data word in bits

    Returns:
        List of dicts with 'addr' and 'data' keys
    """
    result = []

    if not result_file.exists():
        return result

    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("@"):
                parts = line[1:].split()
                if len(parts) >= 2:
                    addr = int(parts[0], 16)
                    data = int(parts[1], 16)
                    result.append({"addr": addr, "data": data})

    return result


def view_hbm_results(
    build_dir: Path,
    exp_width: int = 4,
    man_width: int = 3,
    block_size: int = 8,
    row_dim: int = 8,
    max_rows: int = 10,
) -> None:
    """View HBM memory results from RTL simulation.

    Args:
        build_dir: Path to build directory
        exp_width: MX format exponent width
        man_width: MX format mantissa width
        block_size: Number of elements per block (sharing one scale)
        row_dim: Number of values to display per row
        max_rows: Maximum number of rows to display
    """
    hbm_result_file = build_dir / "hbm_result.mem"

    print("=" * 80)
    print(f"HBM Results from: {hbm_result_file}")
    print("=" * 80)

    if not hbm_result_file.exists():
        print(f"ERROR: File not found: {hbm_result_file}")
        return

    # Check if file is empty
    if hbm_result_file.stat().st_size == 0:
        print("WARNING: HBM result file is empty!")
        print("This usually means the simulation didn't write any data to HBM.")
        print("Possible reasons:")
        print("  1. Simulation failed before completing")
        print("  2. No store operations executed")
        print("  3. The 'final' block didn't execute properly")
        return

    data = parse_hbm_result_file(hbm_result_file)

    for section_name, entries in data.items():
        print(f"\n--- {section_name.upper()} ({len(entries)} entries) ---")
        if not entries:
            print("  (empty)")
            continue

        # Sort by address
        entries = sorted(entries, key=lambda x: x["addr"])

        # Display first few rows
        displayed = 0
        for i, entry in enumerate(entries):
            if displayed >= max_rows * row_dim:
                print(f"  ... ({len(entries) - displayed} more entries)")
                break

            if i % row_dim == 0:
                print(f"  @{entry['addr']:04x}: ", end="")

            print(f"{entry['data']:016x} ", end="")
            displayed += 1

            if (i + 1) % row_dim == 0:
                print()

        if displayed % row_dim != 0:
            print()


def view_hbm_as_floats(
    build_dir: Path,
    exp_width: int = 4,
    man_width: int = 3,
    block_size: int = 8,
    row_dim: int = 8,
    max_rows: int = 10,
    section: str = "v_element",
) -> List[float]:
    """View HBM memory results as floating point values.

    Args:
        build_dir: Path to build directory
        exp_width: MX format exponent width
        man_width: MX format mantissa width
        block_size: Number of elements per block
        row_dim: Number of values per row
        max_rows: Maximum rows to display
        section: Which section to view ('m_element', 'v_element', etc.)

    Returns:
        List of converted float values
    """
    hbm_result_file = build_dir / "hbm_result.mem"

    if not hbm_result_file.exists() or hbm_result_file.stat().st_size == 0:
        print(f"WARNING: HBM result file is empty or missing")
        return []

    data = parse_hbm_result_file(hbm_result_file)

    element_section = section
    scale_section = section.replace("element", "scale")

    elements = {e["addr"]: e["data"] for e in data.get(element_section, [])}
    scales = {s["addr"]: s["data"] for s in data.get(scale_section, [])}

    if not elements:
        print(f"WARNING: No data in {element_section} section")
        return []

    print(f"\n--- {element_section.upper()} as Floats ---")

    # Convert to floats
    floats = []
    element_width = 1 + exp_width + man_width
    elements_per_word = 64 // element_width  # Assuming 64-bit data width

    for addr in sorted(elements.keys()):
        word = elements[addr]
        scale_word = scales.get(addr, 127)  # Default scale

        # Extract individual elements from the word
        for i in range(elements_per_word):
            elem = (word >> (i * element_width)) & ((1 << element_width) - 1)
            scale = (scale_word >> (i * 8)) & 0xFF if isinstance(scale_word, int) else 127
            fval = mx_to_float(elem, scale, exp_width, man_width)
            floats.append(fval)

    # Display
    for i in range(0, min(len(floats), max_rows * row_dim), row_dim):
        print(f"  [{i:4d}]: ", end="")
        for j in range(row_dim):
            if i + j < len(floats):
                print(f"{floats[i + j]:8.4f} ", end="")
        print()

    if len(floats) > max_rows * row_dim:
        print(f"  ... ({len(floats) - max_rows * row_dim} more values)")

    return floats


def compare_with_golden(
    build_dir: Path,
    tolerance: float = 0.1,
    atol: float = 0.03,
    rtol: float = 0.1,
) -> bool:
    """Compare HBM results with golden output.

    Args:
        build_dir: Path to build directory
        tolerance: Maximum allowed absolute difference
        atol: Absolute tolerance for np.allclose
        rtol: Relative tolerance for np.allclose

    Returns:
        True if results match within tolerance
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: torch and numpy required for golden comparison")
        return False

    golden_file = build_dir / "golden_result.pt"
    if not golden_file.exists():
        print(f"WARNING: Golden file not found: {golden_file}")
        return False

    print("\n" + "=" * 80)
    print("Comparison with Golden Output")
    print("=" * 80)

    golden = torch.load(golden_file)
    if isinstance(golden, dict):
        golden_tensor = golden.get("output", golden.get("golden", None))
        if golden_tensor is None:
            print(f"WARNING: Could not find output tensor in golden file")
            print(f"Available keys: {golden.keys()}")
            return False
    else:
        golden_tensor = golden

    golden_np = golden_tensor.flatten().numpy()
    print(f"Golden shape: {golden_tensor.shape}, total elements: {len(golden_np)}")

    # Get simulation results
    sim_floats = view_hbm_as_floats(build_dir, max_rows=0)
    if not sim_floats:
        print("WARNING: No simulation results to compare")
        return False

    sim_np = np.array(sim_floats[:len(golden_np)])

    if len(sim_np) != len(golden_np):
        print(f"WARNING: Size mismatch - sim: {len(sim_np)}, golden: {len(golden_np)}")

    # Compare
    diff = np.abs(sim_np - golden_np[:len(sim_np)])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    matches = np.allclose(sim_np, golden_np[:len(sim_np)], atol=atol, rtol=rtol)

    print(f"\nResults:")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Match (atol={atol}, rtol={rtol}): {'PASS' if matches else 'FAIL'}")

    if not matches:
        # Show first few mismatches
        mismatch_idx = np.where(diff > atol)[0][:5]
        print(f"\nFirst mismatches:")
        for idx in mismatch_idx:
            print(f"  [{idx}]: sim={sim_np[idx]:.6f}, golden={golden_np[idx]:.6f}, diff={diff[idx]:.6f}")

    return matches


def main():
    parser = argparse.ArgumentParser(description="View RTL simulation memory results")
    parser.add_argument(
        "build_dir",
        type=str,
        help="Path to build directory containing result files",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with golden results",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="v_element",
        choices=["m_element", "v_element", "m_scale", "v_scale"],
        help="Which HBM section to view",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Maximum number of rows to display",
    )
    parser.add_argument(
        "--as-float",
        action="store_true",
        help="Convert and display as floating point values",
    )

    args = parser.parse_args()
    build_dir = Path(args.build_dir)

    if not build_dir.exists():
        print(f"ERROR: Build directory not found: {build_dir}")
        sys.exit(1)

    # Always show raw HBM data first
    view_hbm_results(build_dir, max_rows=args.max_rows)

    # Show as floats if requested
    if args.as_float:
        view_hbm_as_floats(build_dir, section=args.section, max_rows=args.max_rows)

    # Compare with golden if requested
    if args.compare:
        success = compare_with_golden(build_dir)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
