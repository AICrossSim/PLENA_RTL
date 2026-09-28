#!/usr/bin/env python3
"""
data_flow_control Testbench

REGRESSION TEST: m_load_in_process timing.

The critical bug: m_load_in_process dropped to 0 when the M SRAM counter
reached load_m_amount-2, but m_v_valid handshakes still had 3 cycles of
pipeline delay in flight from the VRAM read. This let MM_WO enter execute
before complete_loading fired inside mx_systolic_mcu.

The fix: m_load_in_process = (counter < load_m_amount-2)
                            || continuous_load_v_for_matrix_en
                            || m_v_valid

This testbench drives data_flow_control_tb_wrapper in isolation and verifies
that m_load_in_process stays HIGH for at least BLEN=8 m_v_valid handshakes
before dropping.

Config: MLEN=16, BLEN=8, VLEN=16, BLOCK_DIM=8
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))

import logging
import pytest
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────
MLEN      = 16
BLEN      = 8
VLEN      = 16
BLOCK_DIM = 8

CLOCK_PERIOD_NS = 2
TIMEOUT_CYCLES  = 200

# Enum values from operation.svh
STALL_M         = 0x0
MM_IC           = 0x5
MM_WO           = 0x7
STALL_V_ELEMENT = 0x0
STALL_V_REDUCT  = 0x0
STALL_S_FP      = 0x0
STALL_C         = 0x0
STALL_H         = 0x0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(sig):
    """Read a cocotb signal as int; return 0 on error."""
    try:
        return int(sig.value)
    except Exception:
        return 0


async def reset_dut(dut):
    """Assert rst for 4 cycles then release; set all inputs to safe defaults."""
    dut.rst.value = 1

    # OP_BUNDLE — all STALL
    dut.op_m_op.value           = STALL_M
    dut.op_v_ele_op.value       = STALL_V_ELEMENT
    dut.op_v_reduct_op.value    = STALL_V_REDUCT
    dut.op_s_fp_op.value        = STALL_S_FP
    dut.op_c_op.value           = STALL_C
    dut.op_h_op.value           = STALL_H
    dut.op_m_transposed_read.value  = 0
    dut.op_v_broadcast_en.value     = 0
    dut.op_fps1.value               = 0
    dut.op_fps2.value               = 0
    dut.op_fpd.value                = 0
    dut.op_gp_reg1.value            = 0
    dut.op_gp_reg2.value            = 0
    dut.op_gp_rstride.value         = 0
    dut.op_gp_rd.value              = 0
    dut.op_addr_1.value             = 0
    dut.op_addr_2.value             = 0
    dut.op_update_m_waddr.value     = 0
    dut.op_update_v_waddr.value     = 0

    # MEM_WEN_INFO — all disabled
    dut.mwc_w_m_sram_en.value          = 0
    dut.mwc_w_s_sram_port_a_en.value   = 0
    dut.mwc_w_s_sram_port_b_en.value   = 0
    dut.mwc_w_from_m.value             = 0

    # Matrix machine interface
    dut.m_m_ready.value         = 1
    dut.m_v_ready.value         = 1
    dut.m_out_valid.value       = 0
    dut.m_write_request.value   = 0
    dut.m_write_addr.value      = 0

    # Matrix SRAM interface
    dut.m_prefetch_data_not_ready.value = 1   # M not yet prefetched

    # Vector machine interface
    dut.v_v_a_ready.value       = 1
    dut.v_v_b_ready.value       = 1
    dut.v_s_in_ready.value      = 1
    dut.v_write_request.value   = 0
    dut.v_write_addr.value      = 0

    # Scalar machine interface
    dut.s_map_v_valid.value     = 0

    # Vector SRAM interface
    dut.v_prefetch_data_not_ready.value = 0

    # HBM interface
    dut.prefetch_m_valid.value      = 0
    dut.prefetch_v_valid.value      = 0
    dut.hbm_ready_to_write.value    = 0

    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


# ── Test ──────────────────────────────────────────────────────────────────────

@cocotb.test()
async def m_load_in_process_timing_test(dut):
    """
    REGRESSION TEST: m_load_in_process must stay HIGH until all BLEN=8
    m_v_valid handshakes complete.  Previously dropped early causing
    complete_loading to never fire in mx_systolic_mcu.

    Protocol:
      1. Drive MM_IC; keep m_prefetch_data_not_ready=1 initially.
      2. After 5 cycles release m_prefetch_data_not_ready=0 (M prefetch arrives).
      3. Monitor m_v_valid handshakes and m_load_in_process each cycle.
      4. When m_load_in_process drops, assert v_handshake_count >= BLEN.
      5. Drive MM_WO for 2 cycles then STALL_M.
    """
    cocotb.log.info("=" * 60)
    cocotb.log.info("data_flow_control m_load_in_process Timing Regression Test")
    cocotb.log.info(f"MLEN={MLEN} BLEN={BLEN} VLEN={VLEN} BLOCK_DIM={BLOCK_DIM}")
    cocotb.log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # ── Phase 1: Start MM_IC ──────────────────────────────────────────────────
    dut.op_m_op.value   = MM_IC
    dut.op_addr_1.value = 0
    dut.op_addr_2.value = 0
    # m_prefetch_data_not_ready still 1 — M not ready yet

    v_handshake_count = 0
    m_handshake_count = 0
    drop_cycle        = None
    drop_v_count      = None
    test_passed       = False

    for cyc in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)

        # After 5 cycles, simulate M prefetch arriving
        if cyc == 5:
            dut.m_prefetch_data_not_ready.value = 0
            cocotb.log.info(f"  cyc={cyc}: m_prefetch_data_not_ready released")

        m_load   = _safe_int(dut.m_load_in_process)
        m_v_v    = _safe_int(dut.m_v_valid)
        m_v_rdy  = _safe_int(dut.m_v_ready)
        m_m_v    = _safe_int(dut.m_m_valid)

        # Simulate pipeline stall: once m_load_in_process asserts, the real
        # pipeline_control inserts STALL_M into exe_stage_op (not MM_IC).
        # Without this, line 401 of data_flow_control.sv keeps resetting
        # v_sram_load_for_matrix_counter to 0 every cycle.
        if m_load == 1 and _safe_int(dut.op_m_op) == MM_IC:
            dut.op_m_op.value = STALL_M
            cocotb.log.info(f"  cyc={cyc}: m_load_in_process HIGH → switching to STALL_M (pipeline stall)")

        # Count handshakes
        if m_v_v and m_v_rdy:
            v_handshake_count += 1
        if m_m_v and _safe_int(dut.m_m_ready):
            m_handshake_count += 1

        cocotb.log.info(
            f"cyc={cyc:3d} m_load={m_load} m_v_valid={m_v_v} m_v_rdy={m_v_rdy}"
            f" v_cnt={v_handshake_count} m_m_valid={m_m_v}"
        )

        # Detect m_load_in_process dropping for the first time
        if m_load == 0 and drop_cycle is None and cyc > 6:
            drop_cycle    = cyc
            drop_v_count  = v_handshake_count
            cocotb.log.info(
                f"  [EVENT] m_load_in_process dropped at cyc={cyc},"
                f" v_handshakes so far={drop_v_count}"
            )
            # Immediately switch to MM_WO for 2 cycles then stall
            dut.op_m_op.value           = MM_WO
            dut.op_update_m_waddr.value = 1
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            dut.op_m_op.value           = STALL_M
            dut.op_update_m_waddr.value = 0
            break

    # ── Assertions ────────────────────────────────────────────────────────────
    cocotb.log.info("=" * 60)
    cocotb.log.info(
        f"[RESULT] m_load_in_process dropped after {drop_v_count} m_v_valid"
        f" handshakes (need >= {BLEN})"
    )
    cocotb.log.info(
        f"[RESULT] m_m_valid handshakes observed: {m_handshake_count}"
    )
    cocotb.log.info("=" * 60)

    assert drop_cycle is not None, (
        f"m_load_in_process never dropped within {TIMEOUT_CYCLES} cycles"
    )
    assert drop_v_count >= BLEN, (
        f"m_load_in_process dropped too early: only {drop_v_count} m_v_valid"
        f" handshakes before drop (need >= {BLEN}=BLEN). "
        f"Regression: premature pipeline unstall bug."
    )

    cocotb.log.info(
        f"PASSED: m_load_in_process held for {drop_v_count} >= {BLEN} handshakes"
    )

    # Drain a few more cycles
    for _ in range(10):
        await RisingEdge(dut.clk)


# ── Runner ────────────────────────────────────────────────────────────────────

@pytest.mark.dev
def test_data_flow_control_load_in_process():
    veri_runner(
        group="control",
        module="data_flow_control_tb_wrapper",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/systolic_gemm_mx"),
            str(SRC_PATH / "basic_components/systolic_gemm_fp"),
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/gemv"),
            str(SRC_PATH / "basic_components/linear_operation"),
            str(SRC_PATH / "basic_components/mx_int_operation"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=True,
        test_module="data_flow_control_tb",
    )


if __name__ == "__main__":
    test_data_flow_control_load_in_process()
