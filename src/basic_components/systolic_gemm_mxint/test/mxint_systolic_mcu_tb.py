#!/usr/bin/env python3

import logging
import pytest
import cocotb
import torch

from cocotb.triggers import RisingEdge
from cocotb.log import SimLog

from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.testbench import Testbench

from plena_quant.mxint import _mx_int_quantize_hardware
from plena_quant.common.utils import block


# M_OP enum values (src/definitions/operation.svh)
STALL_M = 0x0
MM_IC   = 0x5   # load operands
MM_WO   = 0x7   # write result out


def quantize_line(values, element_width, scale_width, block_size):
    """Quantize one 1-D line (the full K vector) to MXINT (two's-complement elements).

    The line is split into K/block_size MXINT blocks, each with its own shared scale.

    Returns:
        real_values : list[K] dequantized real value of each element (golden ref)
        block_scales: list[K/block_size] biased block scales
        element_bus : all K elements concatenated (two's complement) into one bus word
    """
    block_length = values.shape[0]
    num_blocks = block_length // block_size
    mantissa_bits = element_width - 1
    scale_bias = (1 << (scale_width - 1)) - 1
    element_mask = (1 << element_width) - 1

    real_values = [0.0] * block_length
    block_scales = [0] * num_blocks
    element_bus = 0
    for b in range(num_blocks):
        segment = values[b * block_size:(b + 1) * block_size]
        _, mantissa, scaling = _mx_int_quantize_hardware(
            segment.unsqueeze(0), width=element_width, exponent_width=scale_width,
            block_size=[block_size], skip_first_dim=True,
        )
        blocked, _, _, _ = block(segment.unsqueeze(0), block_shape=[block_size], skip_first_dim=True)
        signed_mantissa = torch.sign(blocked + 1e-9) * mantissa
        signed_ints = (signed_mantissa * (1 << mantissa_bits)).round().int().flatten().tolist()
        biased_scale = int((scaling + scale_bias)[0, 0].item())
        block_scales[b] = biased_scale
        exponent = biased_scale - scale_bias - mantissa_bits
        for e in range(block_size):
            k = b * block_size + e
            real_values[k] = signed_ints[e] * 2.0 ** exponent
            element_bus |= (signed_ints[e] & element_mask) << (k * element_width)
    return real_values, block_scales, element_bus


def pack_bus(values, width):
    """Concatenate width-bit values (lsb first) into one bus word."""
    mask = (1 << width) - 1
    bus = 0
    for i, v in enumerate(values):
        bus |= (v & mask) << (i * width)
    return bus


def decode_fp(bits, exp_width, mant_width):
    sign = (bits >> (exp_width + mant_width)) & 1
    exponent = (bits >> mant_width) & ((1 << exp_width) - 1)
    mantissa = bits & ((1 << mant_width) - 1)
    bias = (1 << (exp_width - 1)) - 1
    if exponent == 0:
        value = (mantissa / 2**mant_width) * 2.0 ** (1 - bias)
    else:
        value = (1.0 + mantissa / 2**mant_width) * 2.0 ** (exponent - bias)
    return -value if sign else value


class MXINTSystolicMCUTB(Testbench):
    def __init__(self, dut) -> None:
        super().__init__(dut, dut.clk, dut.rst)
        if not hasattr(self, "log"):
            self.log = SimLog("%s" % (type(self).__qualname__))
            self.log.setLevel(logging.DEBUG)
        self.param = self.read_dut_param()

    def read_dut_param(self):
        """Read the actual elaborated parameters back from the DUT, then derive the rest.
        The literal values live in module_param_list (at the veri_runner call)."""
        param = {
            "FP_EXP_WIDTH":      int(self.dut.FP_EXP_WIDTH.value),
            "FP_MANT_WIDTH":     int(self.dut.FP_MANT_WIDTH.value),
            "MX_T_INT_WIDTH":    int(self.dut.MX_T_INT_WIDTH.value),
            "MX_L_INT_WIDTH":    int(self.dut.MX_L_INT_WIDTH.value),
            "MXINT_SCALE_WIDTH": int(self.dut.MXINT_SCALE_WIDTH.value),
            "KLEN":              int(self.dut.KLEN.value),
            "BLEN":              int(self.dut.BLEN.value),
            "MLEN":              int(self.dut.MLEN.value),
        }
        param["ROW_BLOCK_NUM"] = param["MLEN"] // param["KLEN"]
        param["FP_W"]          = param["FP_EXP_WIDTH"] + param["FP_MANT_WIDTH"] + 1
        param["SCALE_BIAS"]    = (1 << (param["MXINT_SCALE_WIDTH"] - 1)) - 1
        return param

    async def init_signals(self):
        self.dut.control.value = STALL_M
        self.dut.per_head.value = 0
        self.dut.v1_element.value = 0
        self.dut.v1_scale.value = 0
        self.dut.v1_in_valid.value = 0
        self.dut.v2_element.value = 0
        self.dut.v2_scale.value = 0
        self.dut.v2_in_valid.value = 0

    def generate_inputs(self, seed):
        """C[BLEN][BLEN] = A[BLEN][MLEN] . B[MLEN][BLEN], full MLEN reduction.
        v2 = LEFT = A (rows), v1 = TOP = B (cols); both carry per-block scales (one per
        K-segment) presented per cycle alongside that cycle's row/column."""
        param = self.param
        BLEN, MLEN = param["BLEN"], param["MLEN"]
        block_size  = param["KLEN"]
        scale_width = param["MXINT_SCALE_WIDTH"]
        left_width  = param["MX_L_INT_WIDTH"]
        top_width   = param["MX_T_INT_WIDTH"]

        torch.manual_seed(seed)
        a_float = torch.randn(BLEN, MLEN)
        b_float = torch.randn(MLEN, BLEN)

        # v2 (LEFT = A): one bus word per row + that row's per-block scales.
        a_qdata = [[0.0] * MLEN for _ in range(BLEN)]
        self.v2_element_bus = [0] * BLEN
        self.v2_scale_bus   = [0] * BLEN
        for row in range(BLEN):
            real_values, block_scales, element_bus = quantize_line(
                a_float[row], left_width, scale_width, block_size)
            a_qdata[row] = real_values
            self.v2_element_bus[row] = element_bus
            self.v2_scale_bus[row]   = pack_bus(block_scales, scale_width)

        # v1 (TOP = B): one bus word per column + that column's per-block scales.
        b_qdata = [[0.0] * BLEN for _ in range(MLEN)]
        self.v1_element_bus = [0] * BLEN
        self.v1_scale_bus   = [0] * BLEN
        for col in range(BLEN):
            real_values, block_scales, element_bus = quantize_line(
                b_float[:, col], top_width, scale_width, block_size)
            for k in range(MLEN):
                b_qdata[k][col] = real_values[k]
            self.v1_element_bus[col] = element_bus
            # v1 scale bus is MLEN-wide (per-element): broadcast each block's scale
            # across its KLEN elements.
            per_element_scales = [block_scales[k // block_size] for k in range(MLEN)]
            self.v1_scale_bus[col] = pack_bus(per_element_scales, scale_width)

        # Golden real-valued matmul over the quantized inputs.
        self.expected_real = [[0.0] * BLEN for _ in range(BLEN)]
        for i in range(BLEN):
            for j in range(BLEN):
                self.expected_real[i][j] = sum(a_qdata[i][k] * b_qdata[k][j] for k in range(MLEN))

    async def in_driver(self):
        """MM_IC load phase: BLEN cycles, cycle t presents A row t (v2) and B col t (v1)."""
        BLEN = self.param["BLEN"]
        for t in range(BLEN):
            await RisingEdge(self.dut.clk)
            self.dut.control.value     = MM_IC
            self.dut.v2_element.value   = self.v2_element_bus[t]
            self.dut.v2_scale.value     = self.v2_scale_bus[t]
            self.dut.v2_in_valid.value  = 1
            self.dut.v1_element.value   = self.v1_element_bus[t]
            self.dut.v1_scale.value     = self.v1_scale_bus[t]
            self.dut.v1_in_valid.value  = 1

        await RisingEdge(self.dut.clk)
        self.dut.v1_in_valid.value = 0
        self.dut.v2_in_valid.value = 0
        self.dut.control.value     = STALL_M

    async def mm_ic(self, seed, settle_cycles=80, gap_cycles=4):
        """One MM_IC: load A/B (seed), compute one tile, let it accumulate into the FP
        bank, then idle (so the mini arrays soft-reset before the next tile). Returns
        this tile's golden [BLEN][BLEN].

        settle_cycles=0 returns right after the load (compute still in flight) — used to
        fire an MM_WO while compute_busy is still high and check it gets deferred."""
        self.generate_inputs(seed)
        await self.in_driver()
        for _ in range(settle_cycles):              # compute + accumulate_pulse
            await RisingEdge(self.dut.clk)
        if settle_cycles > 0:
            await self.init_signals()               # idle gap (lets mini_clear fire)
            for _ in range(gap_cycles):
                await RisingEdge(self.dut.clk)
        return self.expected_real

    async def mm_wo(self, golden, label, timeout_cycles=300):
        """One MM_WO: drain the FP accumulator bank to v_result, one row per cycle.

        The drain asserts v_result_write_req for BLEN consecutive cycles; each
        cycle's v_result row is compared against the expected (golden) row.

        Deferred-drain correctness is validated IMPLICITLY by the values: if MM_WO
        is issued while a tile is still computing, the drain FSM (start_drain needs
        !compute_busy) holds off until the accumulate lands, so v_result_write_req
        simply stays low until then; the loop below waits through that. If the
        drain ever read a half-accumulated bank, the rows would not match golden.

        Note: by design v_result_write_req == draining, and mcu_active ==
        (compute_busy || draining), so write-req and mcu_active are high TOGETHER
        during the drain — that overlap is expected (draining IS "array in use"),
        not a violation. The checkable timing properties are: (a) the BLEN drain
        rows are contiguous, and (b) mcu_active drops once the drain finishes."""
        param = self.param
        BLEN, FP_W = param["BLEN"], param["FP_W"]

        self.dut.control.value = MM_WO
        busy_at_issue = int(self.dut.mcu_active.value)
        mismatches = 0
        row_idx = 0
        for cycle in range(timeout_cycles):
            await RisingEdge(self.dut.clk)
            wreq = int(self.dut.v_result_write_req.value)
            if wreq != 1:
                if row_idx > 0:        # drain must be contiguous once it starts
                    self.log.info(f"  [BAD] {label}: drain gap after row{row_idx-1} @ cyc{cycle}")
                    mismatches += 1
                    break
                continue
            if row_idx == 0:
                self.log.info(f"  {label}: drain starts @ cyc{cycle} "
                              f"(busy_at_issue={busy_at_issue})")
            word = int(self.dut.v_result.value)
            got_row = [decode_fp((word >> (j * FP_W)) & ((1 << FP_W) - 1),
                                 param["FP_EXP_WIDTH"], param["FP_MANT_WIDTH"]) for j in range(BLEN)]
            exp_row = golden[row_idx]
            row_ok = all(abs(got_row[j] - exp_row[j]) <= abs(exp_row[j]) * 2.0 ** (-5) + 2e-2
                         for j in range(BLEN))
            tag = "OK " if row_ok else "BAD"
            got_s = " ".join(f"{v:8.4f}" for v in got_row)
            exp_s = " ".join(f"{v:8.4f}" for v in exp_row)
            self.log.info(f"  [{tag}] {label} drain cyc{cycle} row{row_idx}: "
                          f"got=[{got_s}]  exp=[{exp_s}]")
            if not row_ok:
                mismatches += 1
            row_idx += 1
            if row_idx == BLEN:
                break

        self.dut.control.value = STALL_M
        assert row_idx == BLEN, f"expected {BLEN} drained rows, got {row_idx}"

        # after the drain the array must report idle (mcu_active low) shortly
        for _ in range(4):
            await RisingEdge(self.dut.clk)
            if int(self.dut.mcu_active.value) == 0:
                break
        else:
            self.log.info(f"  [BAD] {label}: mcu_active still high after drain")
            mismatches += 1

        return mismatches

    async def run_per_head(self, seed=100):
        """PER-HEAD (M_BTMM): ROW_BLOCK_NUM independent heads, each computing
        S_h = Q_h @ K^T over KLEN=head_dim, with K SHARED (broadcast to all minis).
        v2 (LEFT/a) row t = the heads' query-row-t slices concatenated (MLEN wide);
        v1 (TOP/b) row t = shared K's row t in the FIRST KLEN lanes (broadcast).
        Drains ROW_BLOCK_NUM*BLEN rows head-major (head=row//BLEN, q=row%BLEN)."""
        p = self.param
        BLEN, KLEN, MLEN, RBN = p["BLEN"], p["KLEN"], p["MLEN"], p["ROW_BLOCK_NUM"]
        lew, tew, sw = p["MX_L_INT_WIDTH"], p["MX_T_INT_WIDTH"], p["MXINT_SCALE_WIDTH"]
        FP_W = p["FP_W"]
        await self.reset()
        await self.init_signals()
        for _ in range(4):
            await RisingEdge(self.dut.clk)

        torch.manual_seed(seed)
        Q = torch.randn(RBN, BLEN, KLEN)   # per-head query [head][q_row][head_dim]
        K = torch.randn(BLEN, KLEN)        # SHARED kv [kv_row][head_dim]

        q_real = [[None] * BLEN for _ in range(RBN)]
        q_scale = [[0] * BLEN for _ in range(RBN)]
        q_elem = [[0] * BLEN for _ in range(RBN)]
        for h in range(RBN):
            for r in range(BLEN):
                rv, bs, eb = quantize_line(Q[h, r], lew, sw, KLEN)
                q_real[h][r] = rv; q_scale[h][r] = bs[0]; q_elem[h][r] = eb
        k_real = [None] * BLEN; k_scale = [0] * BLEN; k_elem = [0] * BLEN
        for r in range(BLEN):
            rv, bs, eb = quantize_line(K[r], tew, sw, KLEN)
            k_real[r] = rv; k_scale[r] = bs[0]; k_elem[r] = eb

        golden = [[[0.0] * BLEN for _ in range(BLEN)] for _ in range(RBN)]
        for h in range(RBN):
            for i in range(BLEN):
                for j in range(BLEN):
                    golden[h][i][j] = sum(q_real[h][i][k] * k_real[j][k] for k in range(KLEN))

        self.dut.per_head.value = 1
        lmask = (1 << (KLEN * lew)) - 1
        tmask = (1 << (KLEN * tew)) - 1
        # Drive control=MM_IC and the operand stream SIMULTANEOUSLY (worst case): the
        # MCU's ph_eff must make the K-broadcast active from the first load cycle even
        # though per_head_exe latches one cycle later.
        for t in range(BLEN):
            await RisingEdge(self.dut.clk)
            self.dut.control.value = MM_IC
            v2e = 0
            for h in range(RBN):
                v2e |= (q_elem[h][t] & lmask) << (h * KLEN * lew)
            self.dut.v2_element.value = v2e
            self.dut.v2_scale.value = pack_bus([q_scale[h][t] for h in range(RBN)], sw)
            self.dut.v2_in_valid.value = 1
            self.dut.v1_element.value = k_elem[t] & tmask          # K row t in lanes [0:KLEN]
            self.dut.v1_scale.value = pack_bus([k_scale[t]] * MLEN, sw)  # broadcast; [0] used
            self.dut.v1_in_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.v1_in_valid.value = 0
        self.dut.v2_in_valid.value = 0
        self.dut.control.value = STALL_M
        for _ in range(80):
            await RisingEdge(self.dut.clk)

        self.dut.control.value = MM_WO
        mism = 0; ri = 0
        for cyc in range(400):
            await RisingEdge(self.dut.clk)
            if int(self.dut.v_result_write_req.value) != 1:
                if ri > 0:
                    break
                continue
            word = int(self.dut.v_result.value)
            got = [decode_fp((word >> (j * FP_W)) & ((1 << FP_W) - 1),
                             p["FP_EXP_WIDTH"], p["FP_MANT_WIDTH"]) for j in range(BLEN)]
            h = ri // BLEN; qi = ri % BLEN
            exp = golden[h][qi]
            ok = all(abs(got[j] - exp[j]) <= abs(exp[j]) * 2.0 ** (-5) + 3e-2 for j in range(BLEN))
            self.log.info(f"  [{'OK ' if ok else 'BAD'}] per_head row{ri} head{h} q{qi}: "
                          f"got={[f'{v:.3f}' for v in got]} exp={[f'{v:.3f}' for v in exp]}")
            if not ok:
                mism += 1
            ri += 1
            if ri == RBN * BLEN:
                break
        self.dut.control.value = STALL_M
        self.log.info(f"per_head: {ri}/{RBN*BLEN} rows drained, {mism} mismatch(es)")
        assert ri == RBN * BLEN, f"per_head drained {ri} rows, expected {RBN*BLEN}"
        assert mism == 0, f"per_head: {mism} mismatch(es)"

    async def run_test(self, combos=((5, False), (3, False), (1, False),
                                     (2, True), (4, True))):
        """Exercise different MM_IC / MM_WO combinations. Each entry is (n, early):
        run n MM_IC tiles (each accumulates into the FP bank), then ONE MM_WO that
        drains the SUM and clears. If `early`, the LAST MM_IC does not settle — MM_WO is
        issued while compute_busy is still high, so the drain must wait for compute to
        finish (deferred-drain path). Verifies cross-MM_IC FP accumulation, clear-on-WO,
        and the MM_WO-during-compute deferral."""
        param = self.param
        BLEN = param["BLEN"]
        await self.reset()
        await self.init_signals()
        self.log.info(f"Reset finished, BLEN={BLEN} MLEN={param['MLEN']} "
                      f"KLEN={param['KLEN']} ROW_BLOCK_NUM={param['ROW_BLOCK_NUM']}")

        total_mismatch = 0
        seed = 0
        for combo_index, (n_ic, early) in enumerate(combos):
            golden = [[0.0] * BLEN for _ in range(BLEN)]
            for t in range(n_ic):
                # last tile of an `early` combo: don't settle → MM_WO races compute
                settle = 0 if (early and t == n_ic - 1) else 80
                tile = await self.mm_ic(seed, settle_cycles=settle)
                seed += 1
                for i in range(BLEN):
                    for j in range(BLEN):
                        golden[i][j] += tile[i][j]
            label = f"combo{combo_index}({n_ic}xMM_IC -> MM_WO{', early' if early else ''})"
            mismatches = await self.mm_wo(golden, label)
            self.log.info(f"{label}: {mismatches} mismatch(es)")
            total_mismatch += mismatches

        assert total_mismatch == 0, f"{total_mismatch} mismatches across {len(combos)} combos"
        self.log.info(f"All {len(combos)} MM_IC/MM_WO combos match")


@cocotb.test()
async def test(dut):
    tb = MXINTSystolicMCUTB(dut)
    tb.log.setLevel(logging.DEBUG)
    # Per-head (M_BTMM) path FIRST: ROW_BLOCK_NUM independent heads, K broadcast.
    await tb.run_per_head(seed=100)
    await tb.run_per_head(seed=101)
    # (n_ic, early): early issues MM_WO while the last MM_IC is still computing
    await tb.run_test(combos=((5, False), (3, False), (1, False), (2, True), (4, True)))


@pytest.mark.dev
def test_mxint_systolic_mcu():
    veri_runner(
        group="systolic_gemm_mxint",
        module="mxint_systolic_mcu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/conversion"),
        ],
        module_param_list=[{
            # System (SimTop) configuration: V_FP 6/5, 8-bit MXINT, KLEN=BLEN=4, MLEN=16
            "FP_EXP_WIDTH":      6,
            "FP_MANT_WIDTH":     5,
            "MX_T_INT_WIDTH":    8,
            "MX_L_INT_WIDTH":    8,
            "MXINT_SCALE_WIDTH": 8,
            "KLEN":              4,
            "BLEN":              4,
            "MLEN":              16,
            "MAX_SHIFT":         16,
        }],
        trace=True,
    )


if __name__ == "__main__":
    test_mxint_systolic_mcu()
