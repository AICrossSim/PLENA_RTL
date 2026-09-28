"""Base workload generator class for PLENA test workloads.

Provides the foundation for generating test workloads that can be used
with RTL simulation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# Add tools to path for imports
_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
if str(_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_TOOLS_PATH))

# Add PLENA_Tools to path for quant/utils/cfl imports
_PLENA_TOOLS_PATH = _PROJECT_PATH / "PLENA_Tools"
if str(_PLENA_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_PLENA_TOOLS_PATH))

# Add PLENA_Compiler to path for assembler imports
_COMPILER_PATH = _PROJECT_PATH / "PLENA_Compiler"
if str(_COMPILER_PATH) not in sys.path:
    sys.path.insert(0, str(_COMPILER_PATH))

from plena_quant.mxfp import _mx_fp_quantize_hardware
from plena_quant.mxint import _mx_int_quantize_hardware
from cfl_cocotb.torch_fp_conversion import pack_fp_to_bin
from cfl_tools import PROJECT_PATH, SRC_PATH

from assembler.assembly_to_binary import AssemblyToBinary

from .utils.memory_format import (
    tensor_to_mxfp_blocks,
    write_fp_sram_hex,
    write_int_sram_hex,
)


class WorkloadGenerator:
    """Base class for test workload generators.

    This class provides the infrastructure for generating test workloads
    including tensor generation, quantization, memory file generation,
    and assembly compilation.

    Attributes:
        build_dir: Directory where generated files will be saved
        seed: Random seed for reproducibility
        quant_config: Quantization configuration dictionary
        data_config: Data configuration dictionary
    """

    # Default quantization configuration
    DEFAULT_QUANT_CONFIG = {
        "exp_width": 4,        # Exponent width
        "man_width": 3,        # Mantissa width
        "exp_bias_width": 8,   # Block scaling factor width
        "block_size": [1, 8],  # Block dimensions [h, w]
        "skip_first_dim": False,
    }

    # Default hardware configuration
    DEFAULT_HBM_ROW_WIDTH = 256

    def __init__(
        self,
        build_dir: Optional[str] = None,
        seed: int = 42,
        quant_config: Optional[dict] = None,
    ):
        """Initialize the workload generator.

        Args:
            build_dir: Output directory for generated files. If None,
                uses a default 'build' directory under testworkloads.
            seed: Random seed for tensor generation
            quant_config: Optional quantization configuration override
        """
        self.build_dir = Path(build_dir) if build_dir else Path(__file__).parent / "build"
        self.seed = seed
        torch.manual_seed(seed)

        # Merge with default config
        self.quant_config = self.DEFAULT_QUANT_CONFIG.copy()
        if quant_config:
            self.quant_config.update(quant_config)

    def generate(self) -> dict:
        """Generate workload files.

        Returns:
            Dictionary with paths to generated files:
                - machine_code: Path to compiled machine code
                - asm: Path to assembly source
                - fp_sram: Path to FP SRAM preload (if applicable)
                - int_sram: Path to INT SRAM preload (if applicable)
                - golden: Path to golden result file
                - tensors: Dict of paths to saved tensors
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def _quantize_to_mxfp(self, tensor: Tensor) -> Tensor:
        """Quantize tensor to MXFP format.

        Uses the hardware quantization function to simulate actual
        hardware precision.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized tensor in MXFP format
        """
        bm_x, _, _, _ = _mx_fp_quantize_hardware(
            tensor,
            width=self.quant_config["exp_width"] + self.quant_config["man_width"] + 1,
            exponent_width=self.quant_config["exp_width"],
            exponent_bias_width=self.quant_config["exp_bias_width"],
            block_size=self.quant_config["block_size"],
            skip_first_dim=self.quant_config.get("skip_first_dim", False),
        )
        return bm_x

    def _quantize_to_mxint(self, tensor: Tensor) -> Tensor:
        """Quantize tensor to MXINT format.

        Uses the hardware quantization function to simulate actual
        hardware precision for MXINT (microscaling integer) format.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized tensor in MXINT format (dequantized back to float)
        """
        bm_x, _, _ = _mx_int_quantize_hardware(
            tensor,
            width=self.quant_config["man_width"],  # Total integer width including sign
            exponent_width=self.quant_config["exp_width"],  # Shared scale width
            block_size=self.quant_config["block_size"],
            skip_first_dim=self.quant_config.get("skip_first_dim", False),
        )
        return bm_x

    def _quantize(self, tensor: Tensor) -> Tensor:
        """Quantize tensor using the appropriate format based on quant_config.

        Automatically selects MXFP or MXINT quantization based on the
        'format' field in quant_config.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized tensor (dequantized back to float for comparison)
        """
        fmt = self.quant_config.get("format", "mxfp").lower()
        if fmt == "mxint":
            return self._quantize_to_mxint(tensor)
        else:
            return self._quantize_to_mxfp(tensor)

    def _get_mxfp_blocks(self, tensor: Tensor) -> Tuple[List[List[int]], List[int]]:
        """Convert tensor to MXFP blocks and bias.

        Args:
            tensor: Input tensor to convert

        Returns:
            Tuple of (blocks_list, scaling_list)
        """
        return tensor_to_mxfp_blocks(tensor, self.quant_config)

    def _save_tensors(self, tensors: Dict[str, Tensor]) -> Dict[str, Path]:
        """Save tensors as .pt files.

        Args:
            tensors: Dictionary mapping names to tensors

        Returns:
            Dictionary mapping names to saved file paths
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, tensor in tensors.items():
            path = self.build_dir / f"{name}.pt"
            torch.save(tensor, path)
            paths[name] = path
        return paths

    def _save_fp_sram_hex(
        self,
        data: Union[List[float], Tensor],
        filename: str = "fp_sram.mem",
    ) -> Path:
        """Save fp preload as hex format fp_sram.mem.

        Args:
            data: Float data to save
            filename: Output filename

        Returns:
            Path to generated file
        """
        return write_fp_sram_hex(data, self.build_dir, filename)

    def _save_int_sram_hex(
        self,
        data: Union[List[int], Tensor],
        filename: str = "int_sram.mem",
    ) -> Path:
        """Save int preload as hex format int_sram.mem.

        Args:
            data: Integer data to save
            filename: Output filename

        Returns:
            Path to generated file
        """
        return write_int_sram_hex(data, self.build_dir, filename)

    def _save_assembly(self, asm_code: str, filename: str = "generated_asm_code.asm") -> Path:
        """Save assembly code and compile to machine code.

        Args:
            asm_code: Assembly code string
            filename: Output filename for assembly

        Returns:
            Path to compiled machine code file
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Save assembly source
        asm_path = self.build_dir / filename
        asm_path.write_text(asm_code)

        # Compile to machine code
        isa_file = PROJECT_PATH / "PLENA_Compiler" / "doc" / "operation.svh"
        config_file = PROJECT_PATH / "PLENA_Compiler" / "doc" / "configuration.svh"

        assembler = AssemblyToBinary(str(isa_file), str(config_file))

        machine_code_filename = filename.replace(".asm", ".mem")
        machine_code_path = self.build_dir / machine_code_filename

        assembler.generate_binary(str(asm_path), str(machine_code_path))

        return machine_code_path

    def _save_golden(
        self,
        result: Tensor,
        filename: str = "golden_result.pt",
        save_text: bool = True,
    ) -> Dict[str, Path]:
        """Save golden result for verification.

        Args:
            result: Golden result tensor
            filename: Output filename for tensor
            save_text: Whether to also save as text file

        Returns:
            Dictionary with paths to saved files
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save as PyTorch tensor
        pt_path = self.build_dir / filename
        torch.save(result, pt_path)
        paths["pt"] = pt_path

        # Save as text file
        if save_text:
            txt_path = self.build_dir / filename.replace(".pt", ".txt")
            with open(txt_path, "w") as f:
                f.write(f"# Golden result shape: {list(result.shape)}\n")
                f.write(f"# dtype: {result.dtype}\n")
                flat = result.flatten()
                for i, val in enumerate(flat):
                    f.write(f"{val.item():.8f}\n")
            paths["txt"] = txt_path

        return paths

    def _init_memory_files(self) -> Dict[str, Path]:
        """Initialize empty memory files for RTL simulation.

        Creates the necessary empty files and returns their paths.

        Returns:
            Dictionary of file paths
        """
        self.build_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "hbm": self.build_dir / "hbm.mem",  # Combined binary HBM file
            "fp_sram": self.build_dir / "fp_sram.mem",
            "int_sram": self.build_dir / "int_sram.mem",
            "vector_result": self.build_dir / "vector_result.mem",
        }

        # Touch all files
        for path in files.values():
            path.touch()

        return files

    def _set_env_vars(self, paths: Dict[str, Path]) -> None:
        """Set environment variables for RTL simulation.

        Args:
            paths: Dictionary of file paths
        """
        env_mapping = {
            "hbm": "FAKE_HBM_INIT_FILE",  # Combined binary HBM file
            "machine_code": "INSTR_FILE",
            "fp_sram": "FP_MEM_INIT_FILE",
            "int_sram": "INT_MEM_INIT_FILE",
            "vector_result": "VECTOR_MEM_RESULT_FILE",
        }

        for key, env_var in env_mapping.items():
            if key in paths:
                os.environ[env_var] = str(paths[key])
