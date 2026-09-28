"""Full FFN workload generator for PLENA:
    out = (silu(X @ W_up) * (X @ W_gate)) @ W_down

Mirrors PLENA_Compiler/aten/ops/cpu/ffn_ops.py (SiLU on the UP projection, then
elementwise gate, then down projection). Composes the proven building blocks:
  - projection_asm        (matmul)               -- proven by linear
  - silu_asm              (SiLU on up)           -- proven by silu
  - V_MUL_VV gate loop    (silu(up) * gate)      -- elementwise, in-place
  - projection_asm        (down, reads VRAM act) -- proven by silu_down (out<=128)

Pipeline (all in VRAM scratchpad):
  1. preload X (batch, hidden) HBM -> VRAM
  2. up   = X @ W_up   -> up_region
  3. gate = X @ W_gate -> gate_region
  4. silu in-place on up_region
  5. up_region *= gate_region   (elementwise V_MUL_VV)
  6. down = up_region @ W_down -> down_region
  7. C_BREAK; verify down_region against golden

NOTE: the on-chip PC is 16-bit (PC_ADDR_WIDTH=16) => max 16384 instructions. The
unrolled projections are large, so keep dims small (this default ~? instr). Use
--print-instr-count via the generator output / check generated_machine_code.mem.

All projection outputs share the same strided layout (out_tile*batch*VLEN + ...),
which equals the activation-input layout (MLEN==VLEN==16), so up/gate/silu/gate-mul
are elementwise-consistent and the down projection reads the gated region directly.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJECT_PATH / "tools", _PROJECT_PATH / "PLENA_Tools", _PROJECT_PATH / "PLENA_Compiler"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
from asm_templates._imm import load_large_int_str as _load_large_int

# Import PlenaCompiler from the aten compiler (used when use_aten_compiler=True)
from compiler.aten.plena import PlenaCompiler

_HW_TILE_SIZES = load_hardware_tile_sizes(SRC_PATH / "definitions")
DEFAULT_MX_FORMAT = None
_REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)  # 1.125, one scale per block of 8


def _gate_mul_asm(up_base, gate_base, vlen, batch, inter, up_reg, gate_reg, loop_reg):
    """Elementwise in-place: up_region[i] *= gate_region[i], over batch*inter elems.

    UNROLLED with ABSOLUTE addresses (gp = const each iter). A C_LOOP version
    hangs: the in-loop INCREMENTAL S_ADDI (gp += vlen) drifts under the
    scalar<->vector RAW (the scalar pointer-advance runs ahead of the serialized
    element op), and the C_LOOP_END deadlocks against the interlock — the same
    failure as the original SiLU residual. Unrolling removes the loop entirely;
    absolute address setup can't drift (the interlock may hold these S_ADDIs but
    they load a fixed value, not an increment). V_MUL_VV uses rd==rs2 (in-place
    on port B), the PROVEN SiLU mul operand order.
    """
    num_vectors = (batch * inter) // vlen
    code = "; Gate multiply: silu(up) *= gate (elementwise, in-place, UNROLLED)\n"
    for i in range(num_vectors):
        code += _load_large_int(up_reg, up_base + i * vlen)
        code += _load_large_int(gate_reg, gate_base + i * vlen)
        code += f"V_MUL_VV gp{up_reg}, gp{gate_reg}, gp{up_reg}, 0\n"
    return code


class FFNWorkload(WorkloadGenerator):
    """Full FFN: (silu(X@W_up) * (X@W_gate)) @ W_down."""

    MLEN = _HW_TILE_SIZES["MLEN"]
    BLEN = _HW_TILE_SIZES["BLEN"]
    VLEN = _HW_TILE_SIZES["VLEN"]
    HBM_V_Prefetch_Amount = _HW_TILE_SIZES["HBM_V_Prefetch_Amount"]

    def __init__(self, batch_size=8, hidden_size=64, inter_size=64,
                 mx_format=None, skip_asm_gen=False, stage="full",
                 use_aten_compiler=False, **kwargs):
        super().__init__(**kwargs)
        assert batch_size % self.BLEN == 0
        assert hidden_size % self.MLEN == 0
        assert inter_size % self.MLEN == 0
        assert stage in ("full", "up", "upgate", "upsilu", "silu", "gatemul")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.stage = stage  # bring-up isolation: stop after silu/gatemul, verify up region
        self.mx_format = mx_format if mx_format is not None else DEFAULT_MX_FORMAT
        self.skip_asm_gen = skip_asm_gen
        # When True, emit ISA via the ATen PlenaCompiler (prog.ffn) instead of the
        # projection_asm/silu_asm templates. The ATen ffn writes IN-PLACE at X's
        # VRAM address (row 0), so verification targets X's row, not down_base.
        self.use_aten_compiler = use_aten_compiler

    def _vram_layout(self):
        act = 0
        up = self.batch_size * self.hidden_size
        gate = up + self.batch_size * self.inter_size
        scratch = gate + self.batch_size * self.inter_size
        down = scratch + self.VLEN
        return act, up, gate, scratch, down

    def generate(self) -> dict:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        paths = self._init_memory_files()

        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        self.quant_config = get_quant_config_for_format(self.mx_format, precision_settings)
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)

        B, H, I = self.batch_size, self.hidden_size, self.inter_size
        # Scale weights so each matmul output stays ~N(0,1) (FP12 exp range).
        X = torch.randn(B, H, dtype=torch.bfloat16)
        W_up = (torch.randn(H, I, dtype=torch.float32) / (H ** 0.5)).to(torch.bfloat16)
        W_gate = (torch.randn(H, I, dtype=torch.float32) / (H ** 0.5)).to(torch.bfloat16)
        W_down = (torch.randn(I, H, dtype=torch.float32) / (I ** 0.5)).to(torch.bfloat16)

        qX = self._quantize(X)
        qW_up = self._quantize(W_up)
        qW_gate = self._quantize(W_gate)
        qW_down = self._quantize(W_down)

        up = torch.matmul(qX.float(), qW_up.float())
        gate = torch.matmul(qX.float(), qW_gate.float())
        hidden = F.silu(up) * gate
        golden_result = torch.matmul(hidden.float(), qW_down.float())

        input_tensors = {
            "act_tensor": qX.to(torch.bfloat16),
            "weights_up": qW_up.to(torch.bfloat16),
            "weights_gate": qW_gate.to(torch.bfloat16),
            "weights_down": qW_down.to(torch.bfloat16),
        }
        paths["tensors"] = self._save_tensors(input_tensors)

        tensor_shapes = [(B, H), (H, I), (H, I), (I, H)]
        auto_offset = calculate_instr_storage_offset_from_shapes(
            tensor_shapes, precision_settings, hbm_row_width)
        forced = getattr(self, "force_instr_offset", None)
        instr_offset = forced if forced is not None else auto_offset
        update_instruction_storage_offset(instr_offset, SRC_PATH / "definitions")

        if self.use_aten_compiler and self.stage != "full":
            raise ValueError(
                "use_aten_compiler only supports stage='full' (the ATen prog.ffn "
                f"emits the fused kernel end-to-end); got stage='{self.stage}'."
            )

        asm_path = self.build_dir / "generated_asm_code.asm"
        if self.skip_asm_gen:
            if not asm_path.exists():
                raise FileNotFoundError(f"--skip-asm-gen set but no asm at {asm_path}")
        elif self.use_aten_compiler:
            asm_path.write_text(self._generate_assembly_with_aten_compiler())
        else:
            asm_path.write_text(self._generate_assembly())
        paths["asm"] = asm_path

        from .utils.memory_format import write_fp_sram_hex, write_int_sram_hex
        if self.use_aten_compiler:
            # ATen silu reads f0 (0.0) and f1<-slot const_one_fp_address=5 (1.0).
            # Mirror the sliced_layer_test_builder preload: slot5=1.0 (sigmoid one).
            fp_preload = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        else:
            fp_preload = [0.0, 1e-6, 1.0 / H, 1.0]  # f0=0, f3=1.0 (silu const-one)
        paths["fp_sram"] = write_fp_sram_hex(fp_preload, self.build_dir)
        paths["int_sram"] = write_int_sram_hex([0] * 10, self.build_dir)

        create_mem_for_sim(
            precision_settings=precision_settings, data_size=256, mode="behave_sim",
            asm="linear", data=None,
            specified_data_order=["act_tensor", "weights_up", "weights_gate", "weights_down"],
            build_path=self.build_dir, hbm_row_width=hbm_row_width,
            mx_format=self.mx_format, instr_storage_offset=instr_offset,
        )
        paths["hbm"] = self.build_dir / "hbm.mem"
        paths["machine_code"] = self.build_dir / "generated_machine_code.mem"

        act_base, up_base, _, _, down_base = self._vram_layout()
        # Stage-dependent verify target (bring-up isolation).
        if self.stage == "full" and self.use_aten_compiler:
            # ATen prog.ffn writes the (B,H) output IN-PLACE at X's VRAM address.
            verify_t, verify_base, cols = golden_result, act_base, H
        elif self.stage == "full":
            verify_t, verify_base, cols = golden_result, down_base, H
        elif self.stage in ("up", "upgate"):
            verify_t, verify_base, cols = up, up_base, I  # up region should be raw up
        elif self.stage in ("silu", "upsilu"):
            verify_t, verify_base, cols = F.silu(up), up_base, I
        else:  # gatemul
            verify_t, verify_base, cols = F.silu(up) * gate, up_base, I
        golden_2d = verify_t.reshape(B, cols).to(torch.float32)
        golden_3d = golden_2d.reshape(B, cols // self.VLEN, self.VLEN).permute(1, 0, 2)
        down_vram_rows = B * (cols // self.VLEN)
        golden_vram = golden_3d.reshape(down_vram_rows, self.VLEN)
        paths["golden_vram"] = save_golden_vram(
            golden_vram, output_dir=self.build_dir, filename="golden_vram_result", vlen=self.VLEN)
        paths["golden"] = self._save_golden(golden_result, filename="golden_result.pt")

        # Check ALL rows (every block, every batch) to catch drift in any region,
        # not just the first debug window of 8 rows.
        n_check = down_vram_rows
        fmt = self.quant_config.get("format", "mxfp")
        verification_params = {
            "check_hbm": False, "check_vram": True,
            "vram_start_row_idx": verify_base // self.VLEN,
            "vram_num_rows": n_check,
            "row_dim": self.VLEN, "vram_compare_start_row": 0,
            "vram_compare_num_rows": n_check,
            "vram_total_rows": down_vram_rows,
            "golden_vram_file": "golden_vram_result.pt", "workload_type": "ffn",
            "batch_size": B, "hidden_size": H, "inter_size": I,
            "output_shape": list(golden_result.shape), "mx_format": fmt,
            "exp_width": self.quant_config["exp_width"],
            "man_width": self.quant_config["man_width"],
            "scale_width": self.quant_config["exp_bias_width"],
        }
        with open(self.build_dir / "verification_params.json", "w") as f:
            json.dump(verification_params, f, indent=2)
        paths["verification_params"] = self.build_dir / "verification_params.json"

        with open(self.build_dir / "test_params.json", "w") as f:
            json.dump({"workload_type": "ffn", "batch_size": B, "hidden_size": H,
                       "inter_size": I, "mlen": self.MLEN, "blen": self.BLEN,
                       "vlen": self.VLEN, "mx_format": fmt, "seed": self.seed}, f, indent=2)
        paths["test_params"] = self.build_dir / "test_params.json"

        self._set_env_vars(paths)
        return paths

    def _generate_assembly(self) -> str:
        vlen, mlen, blen = self.VLEN, self.MLEN, self.BLEN
        B, H, I = self.batch_size, self.hidden_size, self.inter_size
        act_base, up_base, gate_base, scratch_base, down_base = self._vram_layout()

        code = "; FFN Test: (silu(X@W_up) * (X@W_gate)) @ W_down\n"
        code += f"; B={B} H={H} I={I}  MLEN={mlen} BLEN={blen} VLEN={vlen}\n\n"

        # HBM offsets (cumulative): a1=W_up, a2=W_gate, a3=W_down. Each weight base
        # must be the hbm_row_width-aligned cumulative size of the PRECEDING tensors
        # (elements + scales on separate aligned rows), matching the stager; the old
        # packed estimate int(dim*batch*ratio) under-counted the scale-row padding at
        # MLEN=8 so a following tensor's base landed low and the reader fetched zeros.
        precision_settings, config_settings = load_precision_from_svh(SRC_PATH / "definitions")
        hbm_row_width = config_settings.get("HBM_WIDTH", 256)
        w_up_base = calculate_instr_storage_offset_from_shapes(
            [(B, H)], precision_settings, hbm_row_width)
        w_gate_base = calculate_instr_storage_offset_from_shapes(
            [(B, H), (H, I)], precision_settings, hbm_row_width)
        w_down_base = calculate_instr_storage_offset_from_shapes(
            [(B, H), (H, I), (H, I)], precision_settings, hbm_row_width)

        code += preload_addr_reg_asm(
            addr_reg_to_set=[1, 2, 3], available_registers=[1, 2, 3],
            addr_reg_val=[w_up_base, w_gate_base, w_down_base])
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

        code += preload_act_asm(
            vlen=vlen, preload_len=self.HBM_V_Prefetch_Amount, batch=B, hidden_size=H,
            alive_registers=[1, 2, 3, 4, 5], act_vram_offset=act_base,
            activation_offset_reg=0, stride_size=H)

        # up = X @ W_up
        code += projection_asm(
            mlen=mlen, blen=blen, batch=B, hidden_size=H,
            alive_registers=[1, 2, 3, 4, 5, 6], w_base_hbm_offset_reg=1,
            activation_base_address=act_base, result_base_address=up_base, out_features=I)

        if self.stage == "up":
            return code + "\n; End (stage=up)\nC_BREAK\n"

        if self.stage == "upsilu":
            # silu right after up proj (no gate proj between) — proven silu_down position
            code += silu_asm(
                const_one_fp_address=3, alive_registers=[1, 2, 3],
                activation_base_address=up_base, scratchpad_base_address=scratch_base,
                vlen=vlen, batch_size=B, hidden_dim=I)
            return code + "\n; End (stage=upsilu)\nC_BREAK\n"

        # gate = X @ W_gate
        code += projection_asm(
            mlen=mlen, blen=blen, batch=B, hidden_size=H,
            alive_registers=[1, 2, 3, 4, 5, 6], w_base_hbm_offset_reg=2,
            activation_base_address=act_base, result_base_address=gate_base, out_features=I)

        if self.stage == "upgate":
            return code + "\n; End (stage=upgate)\nC_BREAK\n"

        # silu in-place on up
        code += silu_asm(
            const_one_fp_address=3, alive_registers=[1, 2, 3],
            activation_base_address=up_base, scratchpad_base_address=scratch_base,
            vlen=vlen, batch_size=B, hidden_dim=I)

        if self.stage == "silu":
            return code + "\n; End (stage=silu)\nC_BREAK\n"

        # up *= gate  (elementwise gate)
        code += _gate_mul_asm(up_base, gate_base, vlen, B, I, up_reg=1, gate_reg=2, loop_reg=3)

        if self.stage == "gatemul":
            return code + "\n; End (stage=gatemul)\nC_BREAK\n"

        # down = (gated) @ W_down  -> reads up_base region as activation, K=I
        code += projection_asm(
            mlen=mlen, blen=blen, batch=B, hidden_size=I,
            alive_registers=[1, 2, 3, 4, 5, 6], w_base_hbm_offset_reg=3,
            activation_base_address=up_base, result_base_address=down_base, out_features=H)

        code += "\n; End of FFN test\nC_BREAK\n"
        return code

    def _generate_assembly_with_aten_compiler(self) -> str:
        """Generate FFN ISA via the ATen PlenaCompiler (fused prog.ffn kernel).

        Mirrors linear.py's _generate_assembly_with_aten_compiler. prog.ffn emits
        the fused SwiGLU FFN (up + gate + SiLU + down) and returns the activation
        var IN-PLACE, i.e. the (B,H) output OVERWRITES X at X's VRAM address.

        Inputs are declared in HBM order (X, W_up, W_gate, W_down) so their
        auto-allocated hbm_addr matches specified_data_order. prog.ffn's signature
        is (input, w_gate, w_up, w_down), so we pass the vars by their roles.
        """
        mlen = self.MLEN
        blen = self.BLEN
        B, H, I = self.batch_size, self.hidden_size, self.inter_size
        real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MXINT8 format ratio

        # prog.ffn selects its projection form from mram_tile_capacity, NOT from
        # unroll_loops (which only affects linear_projection/attention helpers):
        #   use_loop_instructions = max_k_tiles <= mram_tile_capacity
        # The loop form emits tight 3-level nested C_LOOPs that risk the RTL decoder
        # early_loop_end_stall deadlock (see memory/aten-compiler-hang). We force the
        # UNROLLED FFN projection (flat, single K-loop) by making mram_tile_capacity <
        # max_k_tiles, while keeping it large enough that the unrolled path's K-split
        # threshold (MAX_K_TILES = mram_tile_capacity * mlen) is NOT crossed (so no
        # V_ADD_VV partial-sum accumulation). The unrolled projection matches the
        # proven projection_asm column strides.
        max_k_tiles = max(H // mlen, I // mlen)
        if max_k_tiles > 1:
            mram_tile_capacity = max_k_tiles - 1  # forces unrolled; MAX_K_TILES = cap*mlen >> max_k_tiles
        else:
            _, _cfg = load_precision_from_svh(SRC_PATH / "definitions")
            mram_tile_capacity = _cfg.get("MATRIX_SRAM_DEPTH", 1024) // mlen

        prog = PlenaCompiler(
            mlen=mlen,
            blen=blen,
            real_data_ratio=real_data_ratio,
            unroll_loops=True,
            mram_tile_capacity=mram_tile_capacity,
        )

        prog.emit("; FFN Test using ATen PlenaCompiler\n")
        prog.emit("; (silu(X@W_up) * (X@W_gate)) @ W_down\n")
        prog.emit(f"; B={B} H={H} I={I}  MLEN={mlen} BLEN={blen}\n\n")

        # Declare inputs in HBM order (matches specified_data_order):
        #   X, W_up, W_gate, W_down
        act_input = prog.input("X", shape=(B, H))
        w_up_input = prog.input("W_up", shape=(H, I))
        w_gate_input = prog.input("W_gate", shape=(H, I))
        w_down_input = prog.input("W_down", shape=(I, H))

        # Load X from HBM to VRAM (row 0).
        x = prog.load_batch(act_input, name="X")

        # Fused FFN, in-place at X's VRAM address. Signature: (input, w_gate, w_up, w_down)
        prog.ffn(x, w_gate_input, w_up_input, w_down_input)

        prog.emit("\n; End of FFN test\n")
        prog.emit("C_BREAK\n")

        return prog.compile()

    def get_config(self) -> dict:
        return {"workload_type": "ffn", "batch_size": self.batch_size,
                "hidden_size": self.hidden_size, "inter_size": self.inter_size,
                "mlen": self.MLEN, "blen": self.BLEN, "vlen": self.VLEN,
                "quant_config": self.quant_config, "seed": self.seed,
                "mx_format": self.mx_format,
                "use_aten_compiler": self.use_aten_compiler}


def main():
    p = argparse.ArgumentParser(description="Generate full FFN workload for PLENA RTL sim")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--inter-size", type=int, default=64)
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mx-format", type=str, default=None, choices=["mxfp", "mxint"])
    p.add_argument("--skip-asm-gen", action="store_true")
    p.add_argument("--stage", type=str, default="full", choices=["full", "up", "upgate", "upsilu", "silu", "gatemul"],
                   help="Bring-up isolation: stop after silu/gatemul, verify the up region")
    p.add_argument("--use-aten-compiler", action="store_true",
                   help="Use ATen PlenaCompiler (fused prog.ffn) instead of the asm templates (stage=full only)")
    args = p.parse_args()

    w = FFNWorkload(batch_size=args.batch, hidden_size=args.hidden_size,
                    inter_size=args.inter_size, build_dir=args.build_dir,
                    seed=args.seed, mx_format=args.mx_format, skip_asm_gen=args.skip_asm_gen,
                    stage=args.stage, use_aten_compiler=args.use_aten_compiler)
    print(f"Generating FFN: (silu(X@W_up)*(X@W_gate))@W_down  "
          f"B={args.batch} H={args.hidden_size} I={args.inter_size}")
    print(f"  Build dir: {w.build_dir}")
    paths = w.generate()
    mc = Path(paths["machine_code"])
    n_instr = sum(1 for _ in open(mc)) if mc.exists() else -1
    print(f"  Instruction count: {n_instr}  (16-bit PC limit = 16384)")
    if n_instr > 16384:
        print("  *** WARNING: exceeds 16-bit PC limit; reduce dims ***")


if __name__ == "__main__":
    main()
