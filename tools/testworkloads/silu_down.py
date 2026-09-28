"""silu+down workload generator for PLENA: silu(X @ W_up) @ W_down.

This is the FFN "down-projection handoff" building block. It chains the proven
linear+SiLU (up projection + SiLU) with a SECOND projection that reads the SiLU
output region back as its activation. The genuinely new integration point is the
vector-produced VRAM region -> matmul activation handoff.

Pipeline:
  1. preload activation X (batch, in_features) HBM -> VRAM @ addr 0
  2. up projection: U = X @ W_up -> up_result region
  3. silu_asm: in-place SiLU on U (one VLEN scratch row after U)
  4. down projection: Z = silu(U) @ W_down -> down_result region
  5. golden = silu(qX @ qW_up) @ qW_down, strided VRAM layout; verify Z region

Layout note: a projection's OUTPUT layout (out_tile*batch*VLEN + b*VLEN + elem)
is identical to its ACTIVATION-INPUT layout, and MLEN==VLEN==16, so the SiLU
output is directly consumable as the down projection's activation with NO
re-layout. Confirmed against projection_asm + ffn_asm.

W_up scaled by 1/sqrt(in_features) so X@W_up ~ N(0,1) stays in the FP12 exp
unit's calibrated range; W_down scaled by 1/sqrt(out_features) to keep Z sane.

Modeled on linear_silu.py (which proved the matmul -> SiLU handoff).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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

from .base import WorkloadGenerator
from cfl_tools import SRC_PATH
from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_svh, load_hardware_tile_sizes, get_quant_config_for_format
from plena_utils.config import calculate_instr_storage_offset_from_shapes, update_instruction_storage_offset
from verification import save_golden_vram

from asm_templates import (
    preload_act_asm,
    preload_addr_reg_asm,
    reset_reg_asm,
    projection_asm,
    silu_asm,
)

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")

DEFAULT_MX_FORMAT = None  # None means use precision.svh setting

# HBM packing ratio: one shared scale per block of 8 elements -> 9/8 = 1.125.
# Same literal as linear_silu.py / the ATen FFN path (program_tensors.py).
_REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


class SiLUDownWorkload(WorkloadGenerator):
    """silu(X @ W_up) @ W_down: up projection, in-place SiLU, then down projection."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    def __init__(
        self,
        batch_size: int = 8,
        in_features: int = 128,
        out_features: int = 256,
        down_features: int = None,
        mx_format: str = None,
        skip_asm_gen: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # down_features defaults to in_features (FFN projects back to hidden_size).
        if down_features is None:
            down_features = in_features
        assert batch_size % self.BLEN == 0, f"batch_size must be divisible by {self.BLEN}"
        assert in_features % self.MLEN == 0, f"in_features must be divisible by {self.MLEN}"
        assert out_features % self.MLEN == 0, f"out_features must be divisible by {self.MLEN}"
        assert down_features % self.MLEN == 0, f"down_features must be divisible by {self.MLEN}"
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.down_features = down_features
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        actual_quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        self.quant_config = actual_quant_config
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        # 1. Random tensors. Scale weights so each matmul output stays ~N(0,1),
        #    inside the FP12 exp unit's calibrated range (|x|>~3 saturates).
        activation = torch.randn(self.batch_size, self.in_features, dtype=torch.bfloat16)
        weight_up = (torch.randn(self.in_features, self.out_features, dtype=torch.float32)
                     / (self.in_features ** 0.5)).to(torch.bfloat16)
        weight_down = (torch.randn(self.out_features, self.down_features, dtype=torch.float32)
                       / (self.out_features ** 0.5)).to(torch.bfloat16)

        # 2. Quantize
        q_activation = self._quantize(activation)
        q_weight_up = self._quantize(weight_up)
        q_weight_down = self._quantize(weight_down)

        # 3. Golden: silu(X @ W_up) @ W_down on quantized values
        up = torch.matmul(q_activation.float(), q_weight_up.float())
        hidden = F.silu(up)
        golden_result = torch.matmul(hidden.float(), q_weight_down.float())

        # 4. Memory layout: HBM = [activation | W_up | W_down]
        input_tensors = {
            "act_tensor": q_activation.to(torch.bfloat16),
            "weights_up": q_weight_up.to(torch.bfloat16),
            "weights_down": q_weight_down.to(torch.bfloat16),
        }
        tensor_paths = self._save_tensors(input_tensors)
        paths["tensors"] = tensor_paths

        # 5. Instruction offset (forced in suite mode)
        tensor_shapes = [
            (self.batch_size, self.in_features),    # activation
            (self.in_features, self.out_features),  # W_up
            (self.out_features, self.down_features),  # W_down
        ]
        auto_offset = calculate_instr_storage_offset_from_shapes(
            tensor_shapes, precision_settings, hbm_row_width
        )
        forced = getattr(self, "force_instr_offset", None)
        if forced is not None:
            assert forced >= auto_offset, (
                f"force_instr_offset {forced} < data size {auto_offset} for this case"
            )
            instr_offset = forced
        else:
            instr_offset = auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        # 6. Assembly
        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(
                    f"--skip-asm-gen was set but no existing assembly was found at {asm_path}."
                )
            print(f"Skipping assembly generation; reusing existing {asm_path}")
        else:
            asm_path.write_text(self._generate_assembly_with_template())
        paths["asm"] = asm_path

        # 7. FP SRAM preload. f0=0.0 (silu reverse-sub + projection); f3=1.0 is
        #    silu's const-one (loaded into f1 by silu_asm).
        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        fp_preload = [0.0, 1e-6, 1.0 / self.in_features, 1.0]
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        # 8. HBM mem files (order must match input_tensors / the address registers)
        create_mem_for_sim(
            precision_settings=precision_settings,
            data_size=256,
            mode="behave_sim",
            asm="linear",
            data=None,
            specified_data_order=["act_tensor", "weights_up", "weights_down"],
            build_path=self.build_dir,
            hbm_row_width=hbm_row_width,
            mx_format=self.mx_format,
            instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        # 9. Golden VRAM (strided layout, same as any projection output: the down
        #    result Z in the down_result region).
        down_result_base = self._vram_layout()[3]
        golden_2d = golden_result.reshape(self.batch_size, self.down_features).to(torch.float32)
        golden_3d = golden_2d.reshape(self.batch_size, self.down_features // self.VLEN, self.VLEN)
        golden_transposed = golden_3d.permute(1, 0, 2)
        down_vram_rows = self.batch_size * (self.down_features // self.VLEN)
        golden_vram = golden_transposed.reshape(down_vram_rows, self.VLEN)
        vram_golden_paths = save_golden_vram(
            golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN
        )
        paths["golden_vram"] = vram_golden_paths

        golden_paths = self._save_golden(golden_result, filename="golden_result.pt")
        paths["golden"] = golden_paths

        # 10. Verification params: check the down result region.
        result_start_row = down_result_base // self.VLEN
        actual_format = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False,
            "check_vram": True,
            "vram_start_row_idx": result_start_row,
            "vram_num_rows": down_vram_rows,
            "row_dim": self.VLEN,
            "vram_compare_start_row": 0,
            "vram_compare_num_rows": down_vram_rows,
            "vram_total_rows": down_vram_rows,
            "golden_vram_file": "golden_vram_result.pt",
            "workload_type": "silu_down",
            "batch_size": self.batch_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "down_features": self.down_features,
            "output_shape": list(golden_result.shape),
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        # 11. Test params
        test_params = {
            "workload_type": "silu_down",
            "batch_size": self.batch_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "down_features": self.down_features,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "mx_format": actual_format,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
            "seed": self.seed,
        }
        with open(self.build_dir / "test_params.json", "w") as f:
            json.dump(test_params, f, indent=2)
        paths["test_params"] = self.build_dir / "test_params.json"

        # 12. Env vars
        self._set_env_vars(paths)
        return paths

    def _vram_layout(self):
        """Returns (act_base, up_result_base, silu_scratch_base, down_result_base)."""
        act_base = 0
        up_result_base = self.batch_size * self.in_features
        silu_scratch_base = up_result_base + self.batch_size * self.out_features
        # SiLU uses exactly one VLEN scratch row; down result starts right after.
        down_result_base = silu_scratch_base + self.VLEN
        return act_base, up_result_base, silu_scratch_base, down_result_base

    def _generate_assembly_with_template(self) -> str:
        vlen = self.VLEN
        mlen = self.MLEN
        blen = self.BLEN
        preload_len = self.HBM_V_Prefetch_Amount

        act_base, up_result_base, silu_scratch_base, down_result_base = self._vram_layout()

        code = "; silu+down Test: silu(X @ W_up) @ W_down\n"
        code += (f"; Shape: ({self.batch_size}, {self.in_features}) @ "
                 f"({self.in_features}, {self.out_features}) -> SiLU -> @ "
                 f"({self.out_features}, {self.down_features}) -> ({self.batch_size}, {self.down_features})\n")
        code += f"; MLEN={mlen}, BLEN={blen}, VLEN={vlen}\n\n"

        # HBM offsets: each tensor base is the hbm_row_width-aligned cumulative
        # size of the tensors that PRECEDE it (elements + scales on separate
        # aligned rows), matching the stager. The old packed estimate
        # int(logical * ratio) under-counted the scale-row padding at MLEN=8,
        # landing weight bases a few rows too low so the reader fetched zeros.
        #   a1 = W_up   base = aligned size of [activation]
        #   a2 = W_down base = aligned size of [activation, W_up]
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        w_up_base = calculate_instr_storage_offset_from_shapes(
            [(self.batch_size, self.in_features)],
            precision_settings,
            hbm_row_width,
        )
        w_down_base = calculate_instr_storage_offset_from_shapes(
            [(self.batch_size, self.in_features), (self.in_features, self.out_features)],
            precision_settings,
            hbm_row_width,
        )

        code += preload_addr_reg_asm(
            addr_reg_to_set=[1, 2],
            available_registers=[1, 2],
            addr_reg_val=[w_up_base, w_down_base],
        )
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

        # Activation X -> VRAM @ 0 (HBM activation base = address register a0 = 0)
        code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=self.batch_size,
            hidden_size=self.in_features,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=act_base,
            activation_offset_reg=0,
            stride_size=self.in_features,
        )

        # Up projection: U = X @ W_up  (weights at a1)
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=self.batch_size,
            hidden_size=self.in_features,
            alive_registers=[1, 2, 3, 4, 5, 6],
            w_base_hbm_offset_reg=1,
            activation_base_address=act_base,
            result_base_address=up_result_base,
            out_features=self.out_features,
        )

        # SiLU in-place on U (FP slot 3 holds 1.0)
        code += silu_asm(
            const_one_fp_address=3,
            alive_registers=[1, 2, 3],
            activation_base_address=up_result_base,
            scratchpad_base_address=silu_scratch_base,
            vlen=vlen,
            batch_size=self.batch_size,
            hidden_dim=self.out_features,
        )

        # Down projection: Z = silu(U) @ W_down  (activation = silu output region,
        # K = out_features; weights at a2)
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=self.batch_size,
            hidden_size=self.out_features,
            alive_registers=[1, 2, 3, 4, 5, 6],
            w_base_hbm_offset_reg=2,
            activation_base_address=up_result_base,
            result_base_address=down_result_base,
            out_features=self.down_features,
        )

        code += "\n; End of silu+down test\n"
        code += "C_BREAK\n"
        return code

    def get_config(self) -> dict:
        return {
            "workload_type": "silu_down",
            "batch_size": self.batch_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "down_features": self.down_features,
            "mlen": self.MLEN,
            "blen": self.BLEN,
            "vlen": self.VLEN,
            "quant_config": self.quant_config,
            "seed": self.seed,
            "mx_format": self.mx_format,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate silu+down workload for PLENA RTL simulation")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (divisible by BLEN)")
    parser.add_argument("--in-features", type=int, default=128, help="Input/hidden dim (divisible by MLEN)")
    parser.add_argument("--out-features", type=int, default=256, help="Intermediate dim (divisible by MLEN)")
    parser.add_argument("--down-features", type=int, default=None,
                        help="Down-projection output dim (default = in-features)")
    parser.add_argument("--build-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"],
                        help="MX format (default: use precision.svh)")
    parser.add_argument("--skip-asm-gen", action="store_true",
                        help="Reuse existing generated_asm_code.asm, only re-assemble")
    args = parser.parse_args()

    workload = SiLUDownWorkload(
        batch_size=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        down_features=args.down_features,
        build_dir=args.build_dir,
        seed=args.seed,
        mx_format=args.mx_format,
        skip_asm_gen=args.skip_asm_gen,
    )

    df = workload.down_features
    print("Generating silu+down workload:")
    print(f"  Shape: ({args.batch}, {args.in_features}) @ ({args.in_features}, {args.out_features})"
          f" -> SiLU -> @ ({args.out_features}, {df})")
    print(f"  Build dir: {workload.build_dir}")

    paths = workload.generate()
    print("\nGenerated files:")
    for key, value in paths.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {key}/{subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
