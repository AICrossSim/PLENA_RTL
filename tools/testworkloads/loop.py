"""Simple loop test workload for PLENA.

Tests the loop controller with simple scalar operations:
- Single loop test
- Nested loop test

No HBM/prefetch operations - just pure loop behavior verification.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add necessary paths
_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
if str(_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_TOOLS_PATH))

_PLENA_TOOLS_PATH = _PROJECT_PATH / "PLENA_Tools"
if str(_PLENA_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_PLENA_TOOLS_PATH))

_COMPILER_PATH = _PROJECT_PATH / "PLENA_Compiler"
if str(_COMPILER_PATH) not in sys.path:
    sys.path.insert(0, str(_COMPILER_PATH))

import torch

from .base import WorkloadGenerator
from cfl_tools import SRC_PATH
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset


class LoopWorkload(WorkloadGenerator):
    """Simple loop test workload.

    Generates minimal assembly to test loop controller behavior.
    Uses only scalar integer operations inside loops.
    """

    def __init__(
        self,
        outer_count: int = 3,
        inner_count: int = 2,
        test_type: str = "nested",
        many_count: int = 300,
        **kwargs,
    ):
        """Initialize loop test workload.

        Args:
            outer_count: Number of outer loop iterations
            inner_count: Number of inner loop iterations
            test_type: "single", "nested", or "many" (many sequential loops)
            many_count: Number of sequential loops for the "many" stress test
            **kwargs: Additional arguments passed to WorkloadGenerator
        """
        super().__init__(**kwargs)
        self.outer_count = outer_count
        self.inner_count = inner_count
        self.test_type = test_type
        self.many_count = many_count

    def generate(self) -> dict:
        """Generate loop test workload files.

        Returns:
            Dictionary with paths to all generated files
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Load precision settings
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # Generate assembly code (must be done BEFORE create_mem_for_sim)
        asm_code = self._generate_assembly()
        asm_path = self.build_dir / "generated_asm_code.asm"
        asm_path.write_text(asm_code)
        paths["asm"] = asm_path

        # Create minimal memory files (empty but required)
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex

        # FP SRAM with some default values
        fp_preload = [0.0] * 16
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)

        # INT SRAM - empty
        paths["int_sram"] = write_int_sram_hex([0] * 16, self.build_dir)

        # Create minimal placeholder tensor for HBM (consistent with prefetch.py)
        # This ensures instructions are placed after data, not at offset 0
        placeholder_shape = (8, 64)  # Minimal tensor: 8 rows x 64 elements
        placeholder_tensor = torch.zeros(placeholder_shape, dtype=torch.bfloat16)
        placeholder_path = self.build_dir / "placeholder.pt"
        torch.save(placeholder_tensor, placeholder_path)

        # Calculate instruction offset based on placeholder data (like prefetch.py)
        tensor_shapes = [placeholder_shape]
        instr_offset = calculate_instr_storage_offset_from_shapes(
            tensor_shapes, precision_settings, hbm_row_width
        )
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        # Use create_mem_for_sim to properly integrate instructions into HBM
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="loop",
            data=None,
            specified_data_order=["placeholder"],  # Include placeholder data
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=None,
            instr_storage_offset=instr_offset,
        )

        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # Calculate expected result
        if self.test_type == "single":
            expected_gp1 = self.outer_count
        elif self.test_type == "many":
            expected_gp1 = self.many_count * self.inner_count
        else:  # nested
            expected_gp1 = self.outer_count * self.inner_count

        # Save test parameters
        test_params = {
            "test_type": self.test_type,
            "outer_count": self.outer_count,
            "inner_count": self.inner_count,
            "expected_gp1": expected_gp1,
            "workload_type": "loop",
        }
        params_path = self.build_dir / "test_params.json"
        with open(params_path, "w") as f:
            json.dump(test_params, f, indent=2)
        paths["test_params"] = params_path

        # Set environment variables
        self._set_env_vars(paths)

        print(f"Generated loop test: {self.test_type}")
        print(f"  Outer iterations: {self.outer_count}")
        if self.test_type == "nested":
            print(f"  Inner iterations: {self.inner_count}")
        print(f"  Expected gp1 value: {expected_gp1}")

        return paths

    def _generate_assembly(self) -> str:
        """Generate assembly code for loop test.

        Returns:
            Assembly code string
        """
        asm = []
        asm.append("; Simple Loop Test")
        asm.append(f"; Test type: {self.test_type}")
        asm.append(f"; Outer count: {self.outer_count}")
        if self.test_type == "nested":
            asm.append(f"; Inner count: {self.inner_count}")
        asm.append("")

        if self.test_type == "single":
            # Single loop test
            # gp1 = accumulator (starts at 0)
            # gp2 = loop counter
            asm.append("; Initialize gp1 = 0 (accumulator)")
            asm.append("S_ADDI_INT gp1, gp0, 0")
            asm.append("")
            asm.append(f"; Simple loop: {self.outer_count} iterations")
            asm.append(f"C_LOOP_START gp2, {self.outer_count}")
            asm.append("S_ADDI_INT gp1, gp1, 1")
            asm.append("C_LOOP_END gp2")
            asm.append("")
            asm.append(f"; After loop, gp1 should be {self.outer_count}")
        elif self.test_type == "many":
            # Many sequential loops (static loop-count stress test).
            # Each loop reuses gp2 (popped before the next is pushed), mirroring the
            # ATen codegen's register reuse. Tests whether the hang scales with the
            # NUMBER of distinct C_LOOPs executed (fetch<->decode drift per loop-end).
            asm.append("; Initialize gp1 = 0 (accumulator)")
            asm.append("S_ADDI_INT gp1, gp0, 0")
            asm.append("")
            asm.append(f"; {self.many_count} sequential loops x {self.inner_count} iters each")
            for _ in range(self.many_count):
                asm.append(f"C_LOOP_START gp2, {self.inner_count}")
                asm.append("S_ADDI_INT gp1, gp1, 1")
                asm.append("C_LOOP_END gp2")
            expected = self.many_count * self.inner_count
            asm.append(f"; After {self.many_count} loops, gp1 should be {expected}")
        else:
            # Nested loop test
            # gp1 = accumulator (starts at 0)
            # gp2 = outer loop counter
            # gp3 = inner loop counter
            asm.append("; Initialize gp1 = 0 (accumulator)")
            asm.append("S_ADDI_INT gp1, gp0, 0")
            asm.append("")
            asm.append(f"; Outer loop: {self.outer_count} iterations")
            asm.append(f"C_LOOP_START gp2, {self.outer_count}")
            asm.append(f"; Inner loop: {self.inner_count} iterations")
            asm.append(f"C_LOOP_START gp3, {self.inner_count}")
            asm.append("S_ADDI_INT gp1, gp1, 1")
            asm.append("C_LOOP_END gp3")
            asm.append("C_LOOP_END gp2")
            asm.append("")
            expected = self.outer_count * self.inner_count
            asm.append(f"; After nested loop, gp1 should be {self.outer_count} * {self.inner_count} = {expected}")

        # Add break at end
        asm.append("")
        asm.append("; End of test")
        asm.append("C_BREAK")
        asm.append("")

        return "\n".join(asm)

    def get_config(self) -> dict:
        """Get the workload configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            "workload_type": "loop",
            "test_type": self.test_type,
            "outer_count": self.outer_count,
            "inner_count": self.inner_count,
            "seed": self.seed,
        }


def main():
    """CLI entry point for loop test workload generation."""
    parser = argparse.ArgumentParser(
        description="Generate simple loop test workload for PLENA RTL simulation"
    )
    parser.add_argument(
        "--outer", type=int, default=3,
        help="Outer loop iteration count"
    )
    parser.add_argument(
        "--inner", type=int, default=2,
        help="Inner loop iteration count (for nested test)"
    )
    parser.add_argument(
        "--type", type=str, default="nested", choices=["single", "nested", "many"],
        help="Test type: 'single', 'nested', or 'many' (sequential-loop stress)"
    )
    parser.add_argument(
        "--count", type=int, default=300,
        help="Number of sequential loops for the 'many' stress test"
    )
    parser.add_argument(
        "--build-dir", type=str, default=None,
        help="Output directory for generated files"
    )
    args = parser.parse_args()

    # Create workload generator
    workload = LoopWorkload(
        outer_count=args.outer,
        inner_count=args.inner,
        test_type=args.type,
        many_count=args.count,
        build_dir=args.build_dir,
    )

    # Generate files
    print(f"Generating loop test workload:")
    print(f"  Type: {args.type}")
    print(f"  Build dir: {workload.build_dir}")

    paths = workload.generate()

    print(f"\nGenerated files:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    print(f"\nConfiguration:")
    config = workload.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
