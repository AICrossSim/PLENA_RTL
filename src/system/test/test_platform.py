"""RTL simulation platform for PLENA.

This module provides the PLENATestPlatform class that handles the complete
setup for RTL simulation, including:
- Loading and quantizing tensor data from .pt files
- Compiling assembly code to machine code
- Generating hex-format memory files for RTL simulation
- Setting up environment variables for cocotb
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch

# Add tools to path
_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
_PLENA_TOOLS_PATH = _PROJECT_PATH / "PLENA_Tools"
_COMPILER_PATH = _PROJECT_PATH / "PLENA_Compiler"

if str(_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_TOOLS_PATH))
if str(_PLENA_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(_PLENA_TOOLS_PATH))
if str(_COMPILER_PATH) not in sys.path:
    sys.path.insert(0, str(_COMPILER_PATH))

from cfl_tools import PROJECT_PATH, SRC_PATH
from assembler.assembly_to_binary import AssemblyToBinary
from plena_utils.load_config import load_svh_settings
from verification.view_vector_result import float_to_fp


class PLENATestPlatform:
    """RTL simulation platform for PLENA.

    This class provides a clean interface for setting up RTL simulation tests.
    It takes raw .pt tensor files, handles quantization to MXFP format, compiles
    assembly code, and generates all necessary memory files.

    Example usage:
        platform = PLENATestPlatform(
            asm_file="test/projection.asm",
            data_files=["weights.pt", "activations.pt"],
        )
        paths = platform.prepare()
        # Now ready to run cocotb simulation

    Attributes:
        asm_file: Path to the assembly source file
        data_files: List of paths to .pt tensor files (in HBM order)
        fp_sram_file: Optional path to FP SRAM preload file
        int_sram_file: Optional path to INT SRAM preload file
        build_dir: Directory for generated output files
    """

    # Default quantization configuration
    DEFAULT_QUANT_CONFIG = {
        "exp_width": 4,
        "man_width": 3,
        "exp_bias_width": 8,
        "block_size": [1, 8],
        "skip_first_dim": False,
        "format": "mxfp",  # "mxfp" or "mxint"
    }

    DEFAULT_HBM_ROW_WIDTH = 256

    def __init__(
        self,
        asm_file: Union[str, Path],
        data_files: List[Union[str, Path]],
        fp_sram_file: Optional[Union[str, Path, List[float]]] = None,
        int_sram_file: Optional[Union[str, Path, List[int]]] = None,
        build_dir: Optional[Union[str, Path]] = None,
        quant_config: Optional[dict] = None,
        instr_storage_offset: Optional[Union[int, str]] = "auto",
    ):
        """Initialize the test platform.

        Args:
            asm_file: Path to .asm assembly file
            data_files: Paths to .pt tensor files (loaded in order for HBM)
            fp_sram_file: Optional FP SRAM preload data (path to .pt file or list)
            int_sram_file: Optional INT SRAM preload data (path to .pt file or list)
            build_dir: Output directory for generated files. Defaults to
                asm_file.parent / "build"
            quant_config: Optional quantization configuration override
            instr_storage_offset: Byte offset for instruction storage.
                "auto" (default) places instructions right after data.
                An integer specifies an explicit byte offset.
        """
        self.asm_file = Path(asm_file)
        self.data_files = [Path(f) for f in data_files]
        self.fp_sram_file = fp_sram_file
        self.int_sram_file = int_sram_file
        self.instr_storage_offset = instr_storage_offset

        if build_dir:
            self.build_dir = Path(build_dir)
        else:
            self.build_dir = self.asm_file.parent / "build"

        # Load precision settings from hardware definitions
        precision_file = SRC_PATH / "definitions" / "precision.svh"
        if precision_file.exists():
            self.precision_settings = load_svh_settings(str(precision_file))
        else:
            self.precision_settings = {}

        # Setup quantization config
        self.quant_config = self.DEFAULT_QUANT_CONFIG.copy()
        if quant_config:
            self.quant_config.update(quant_config)

        # Override with precision settings if available
        if "ACT_MXFP_EXP_WIDTH" in self.precision_settings:
            self.quant_config["exp_width"] = self.precision_settings["ACT_MXFP_EXP_WIDTH"]
        if "ACT_MXFP_MANT_WIDTH" in self.precision_settings:
            self.quant_config["man_width"] = self.precision_settings["ACT_MXFP_MANT_WIDTH"]
        if "MX_SCALE_WIDTH" in self.precision_settings:
            self.quant_config["exp_bias_width"] = self.precision_settings["MX_SCALE_WIDTH"]

    def prepare(self) -> dict:
        """Prepare simulation environment, reusing files from workload generator if available.

        If hbm.mem already exists (generated by workload generator with correct interleaved
        format), this method will NOT regenerate it. It only sets up environment variables.

        For fresh builds without pre-generated files, it performs:
        1. Creates build directory
        2. Loads and quantizes tensors using shared PLENA_Tools functions
        3. Generates hbm.mem with interleaved format
        4. Compiles ASM -> generated_machine_code.mem
        5. Processes SRAM files
        6. Sets environment variables for cocotb

        Returns:
            Dictionary with paths to all generated files
        """
        # 1. Create build directory
        self.build_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "hbm_init": self.build_dir / "hbm.mem",
            "fp_sram": self.build_dir / "fp_sram.mem",
            "int_sram": self.build_dir / "int_sram.mem",
            "vector_result": self.build_dir / "vector_result.mem",
            "fp_reg_result": self.build_dir / "fp_reg_result.mem",
            "machine_code": self.build_dir / "generated_machine_code.mem",
        }

        # Check if files were already generated by workload generator
        hbm_exists = paths["hbm_init"].exists() and paths["hbm_init"].stat().st_size > 0
        machine_code_exists = paths["machine_code"].exists() and paths["machine_code"].stat().st_size > 0

        if hbm_exists and machine_code_exists:
            # Files already generated by workload generator - just set env vars
            print("[PLENATestPlatform] Using pre-generated files from workload generator")
        else:
            # Need to generate files - use shared functions from PLENA_Tools
            print("[PLENATestPlatform] Generating files using PLENA_Tools")
            self._generate_all_files(paths)

        # Touch result file
        paths["vector_result"].touch()
        paths["fp_reg_result"].touch()

        # Set environment variables
        self._set_env_vars(paths)

        return paths

    def _generate_all_files(self, paths: dict) -> None:
        """Generate all simulation files using shared PLENA_Tools functions.

        Args:
            paths: Dictionary of output file paths
        """
        from plena_utils import Random_MXFP_Tensor_Generator, Random_MXINT_Tensor_Generator
        from sim_env_utils.build_sys_tools import env_setup, read_instructions_from_mem
        from sim_env_utils.build_env import MemoryDataManager

        # Determine format from precision settings or quant_config
        use_mxint = self.quant_config.get("format", "mxfp") == "mxint"
        if "ACT_MX_INT_ENABLE" in self.precision_settings:
            use_mxint = self.precision_settings["ACT_MX_INT_ENABLE"] == 1

        # Create memory data manager
        memory_data_manager = MemoryDataManager()

        # Load and quantize each tensor file
        for data_file in self.data_files:
            if use_mxint:
                generator = Random_MXINT_Tensor_Generator(
                    shape=(1, 1),  # Shape not used for loading
                    quant_config=self.quant_config,
                    directory=data_file.parent,
                    filename=data_file.name,
                )
            else:
                generator = Random_MXFP_Tensor_Generator(
                    shape=(1, 1),  # Shape not used for loading
                    quant_config=self.quant_config,
                    config_settings={},
                    directory=data_file.parent,
                    filename=data_file.name,
                )

            tensor = generator.tensor_load()
            if tensor is not None:
                blocks, bias = generator.quantize_tensor(tensor)
                memory_data_manager.add_mx_file(data_file.name, blocks, bias)

        # Compile ASM
        machine_code_path = self._compile_asm()

        # Read instructions
        instructions = read_instructions_from_mem(machine_code_path)

        # Generate HBM using env_setup (which uses interleaved format)
        data_config = {
            "tensor_size": [1, 256],  # Not used for actual quantization
            "block_size": self.quant_config["block_size"],
        }

        env_setup(
            memory_data_manager,
            self.build_dir,
            data_config,
            self.quant_config,
            hbm_row_width=self.DEFAULT_HBM_ROW_WIDTH,
        )

    def _compile_asm(self) -> Path:
        """Compile ASM file to machine code.

        Returns:
            Path to compiled machine code file
        """
        isa_file = SRC_PATH / "definitions" / "operation.svh"
        config_file = SRC_PATH / "definitions" / "configuration.svh"

        assembler = AssemblyToBinary(str(isa_file), str(config_file))

        # Output file name based on ASM file
        machine_code_path = self.build_dir / "generated_machine_code.mem"

        assembler.generate_binary(str(self.asm_file), str(machine_code_path))

        return machine_code_path

    def _generate_sram_hex(
        self,
        data: Optional[Union[str, Path, List]],
        dtype: str,
    ) -> Path:
        """Generate hex-format SRAM file.

        Args:
            data: Data source - path to .pt file, list of values, or None
            dtype: Data type - "fp" or "int"

        Returns:
            Path to generated file
        """
        filename = f"{dtype}_sram.mem"
        output_path = self.build_dir / filename

        # Scalar FP register width comes straight from precision.svh so the
        # encoded constants always match the RTL fp_reg_file / fp_scalar_sram
        # width (1 + S_FP_EXP_WIDTH + S_FP_MANT_WIDTH). Defaults match the
        # FP12 (6e + 5m) format in precision.svh.
        fp_exp_width = int(self.precision_settings.get("S_FP_EXP_WIDTH", 6))
        fp_man_width = int(self.precision_settings.get("S_FP_MANT_WIDTH", 5))
        fp_total_width = 1 + fp_exp_width + fp_man_width
        fp_hex_digits = (fp_total_width + 3) // 4

        if data is None:
            # Create empty file with minimal placeholder
            with open(output_path, "w") as f:
                if dtype == "fp":
                    f.write(f"0x{0:0{fp_hex_digits}X}\n")  # scalar FP zero
                else:
                    f.write("0x00000000\n")  # INT32 zero
            return output_path

        # Load data if it's a file path
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == ".pt":
                tensor = torch.load(data_path)
                data = tensor.flatten().tolist()
            else:
                # Assume text file with one value per line
                with open(data_path) as f:
                    data = [float(line.strip()) if dtype == "fp" else int(line.strip())
                            for line in f if line.strip()]

        # Write hex format
        with open(output_path, "w") as f:
            for value in data:
                if dtype == "fp":
                    # Encode in the scalar FP format the RTL actually uses
                    # (1 + S_FP_EXP_WIDTH + S_FP_MANT_WIDTH), not IEEE FP16.
                    int_val = float_to_fp(float(value), fp_exp_width, fp_man_width)
                    f.write(f"0x{int_val:0{fp_hex_digits}X}\n")
                else:
                    # Convert to INT32 hex
                    int_val = int(value)
                    if int_val < 0:
                        int_val = (1 << 32) + int_val
                    f.write(f"0x{int_val:08X}\n")

        return output_path

    def _set_env_vars(self, paths: dict) -> None:
        """Set environment variables for RTL simulation.

        Args:
            paths: Dictionary of file paths (and instr_storage_offset)
        """
        env_mapping = {
            "hbm_init": "FAKE_HBM_INIT_FILE",  # Combined HBM init file
            "machine_code": "INSTR_FILE",
            "fp_sram": "FP_MEM_INIT_FILE",
            "int_sram": "INT_MEM_INIT_FILE",
            "vector_result": "VECTOR_MEM_RESULT_FILE",
            "fp_reg_result": "FP_REG_RESULT_FILE",
        }

        for key, env_var in env_mapping.items():
            if key in paths:
                os.environ[env_var] = str(paths[key])
                
    @classmethod
    def from_workload(
        cls,
        workload_dir: Union[str, Path],
        asm_filename: str = "generated_asm_code.asm",
    ) -> "PLENATestPlatform":
        """Create platform from a generated workload directory.

        This is a convenience method for using output from WorkloadGenerator.

        Args:
            workload_dir: Path to workload build directory
            asm_filename: Name of assembly file in the directory

        Returns:
            Configured PLENATestPlatform instance
        """
        workload_dir = Path(workload_dir)

        # Find all .pt files in the directory
        pt_files = sorted(workload_dir.glob("*.pt"))

        # Filter out golden and quantized files
        data_files = [
            f for f in pt_files
            if not f.stem.startswith("golden")
            and not f.stem.startswith("q_")
        ]

        return cls(
            asm_file=workload_dir / asm_filename,
            data_files=data_files,
            build_dir=workload_dir,
        )
