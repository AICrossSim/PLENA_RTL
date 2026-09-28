#!/usr/bin/env python3
"""
matrix_machine_v2 Testbench

Tests the GEMM tile lifecycle through matrix_machine_v2:
  1. MM_IC: stream M rows of weight + activation data
  2. Wait for complete_loading_q latch
  3. MM_WO: trigger drain and writeback
  4. Wait for mcu_active falling edge

Uses matrix_machine_tb_wrapper for OP_BUNDLE struct access via Verilator.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))

import logging
import pytest
import cocotb
import cocotb.utils
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

# --- Constants matching RTL parameters ---
MLEN      = 16
BLEN      = 8
BLOCK_DIM = 8
BLOCK_NUM = MLEN // BLOCK_DIM
WT_ELEM_W = 4
ACT_ELEM_W = 8
MX_SCALE_W = 8

STALL_M = 0x0
MM_IC   = 0x5
MM_WO   = 0x7

CLOCK_PERIOD_NS = 2
M_DIM = BLEN
TIMEOUT = 200

# --- MCU internal signal base path ---
MCU = "dut_i.matrix_compute_unit"


# =====================================================================
# Helpers
# =====================================================================

def pack_bus(n, w, val):
    """Pack n copies of val (w bits each) into a single integer."""
    mask = (1 << w) - 1
    return sum((int(val) & mask) << (i * w) for i in range(n))


def get_sig(dut, path, default="?"):
    """Read a hierarchical signal by dot-separated path. Returns default on failure."""
    try:
        obj = dut
        for p in path.split("."):
            obj = getattr(obj, p)
        return int(obj.value)
    except Exception:
        return default


async def reset_dut(dut):
    """Apply reset and zero all inputs."""
    dut.rst.value = 1
    dut.m_valid.value = 0
    dut.v_valid.value = 0
    dut.out_ready.value = 1
    for sig in ["m_element", "m_scale", "v_element", "v_scale",
                "op_m_op", "op_v_ele_op", "op_v_reduct_op", "op_s_fp_op",
                "op_c_op", "op_h_op", "op_m_transposed_read", "op_v_broadcast_en",
                "op_fps1", "op_fps2", "op_fpd", "op_gp_reg1", "op_gp_reg2",
                "op_gp_rstride", "op_gp_rd", "op_addr_1", "op_addr_2",
                "op_update_m_waddr", "op_update_v_waddr"]:
        getattr(dut, sig).value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


async def load_tile(dut, wt_val=0x4, act_val=0x38, scale_val=0x7F):
    """
    Drive MM_IC and stream M_DIM rows until complete_loading_q latches.
    Returns number of rows loaded.
    """
    dut.op_m_op.value = MM_IC
    dut.op_update_m_waddr.value = 0
    dut.m_element.value = pack_bus(MLEN, WT_ELEM_W, wt_val)
    dut.m_scale.value = pack_bus(MLEN, MX_SCALE_W, scale_val)
    dut.v_element.value = pack_bus(MLEN, ACT_ELEM_W, act_val)
    dut.v_scale.value = pack_bus(BLOCK_NUM, MX_SCALE_W, scale_val)
    dut.m_valid.value = 1
    dut.v_valid.value = 1

    rows = 0
    for _ in range(TIMEOUT):
        await RisingEdge(dut.clk)
        if rows < M_DIM and int(dut.m_ready.value) and int(dut.v_ready.value):
            rows += 1
            if rows >= M_DIM:
                dut.m_valid.value = 0
                dut.v_valid.value = 0
        if get_sig(dut, f"{MCU}.complete_loading_q") == 1:
            break

    dut.m_valid.value = 0
    dut.v_valid.value = 0
    dut.op_m_op.value = STALL_M
    return rows


async def drain_tile(dut):
    """
    Drive MM_WO for 2 cycles then wait for mcu_active 1→0.
    Returns the cycle count for the drain phase.
    """
    dut.op_m_op.value = MM_WO
    dut.op_addr_2.value = 0
    dut.op_update_m_waddr.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.op_m_op.value = STALL_M
    dut.op_update_m_waddr.value = 0

    prev_eip = 0
    eip_seen = False
    for cyc in range(TIMEOUT):
        await RisingEdge(dut.clk)
        eip = get_sig(dut, f"{MCU}.mcu_active")
        if eip == 1:
            eip_seen = True
        if eip_seen and prev_eip == 1 and eip == 0:
            return cyc
        prev_eip = eip

    assert False, "Drain timeout: mcu_active never fell"


async def collect_output(dut, max_cycles=200):
    """Wait for out_valid rows and return list of output bus values."""
    rows = []
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) and int(dut.out_ready.value):
            try:
                rows.append(int(dut.out_v_fp.value))
            except Exception:
                rows.append(None)
    return rows


# =====================================================================
# Tests
# =====================================================================

@cocotb.test()
async def single_tile_latency_test(dut):
    """Measure single-tile MM_IC→MM_WO latency through the MCU."""
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    start_ns = cocotb.utils.get_sim_time(units='ns')
    rows = await load_tile(dut)
    load_ns = cocotb.utils.get_sim_time(units='ns')

    drain_cyc = await drain_tile(dut)
    end_ns = cocotb.utils.get_sim_time(units='ns')

    load_cycles = (load_ns - start_ns) / CLOCK_PERIOD_NS
    drain_cycles = (end_ns - load_ns) / CLOCK_PERIOD_NS
    total_cycles = (end_ns - start_ns) / CLOCK_PERIOD_NS

    cocotb.log.info(f"Single-tile latency: load={load_cycles:.0f} drain={drain_cycles:.0f} total={total_cycles:.0f} cycles ({rows} rows)")
    assert rows >= M_DIM, f"Only loaded {rows}/{M_DIM} rows"


@cocotb.test()
async def two_tile_output_test(dut):
    """
    Load+drain two tiles (ACC_NUM=2). After tile 2, block_data_buffer
    should produce output rows.
    """
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # Tile 1
    rows1 = await load_tile(dut, wt_val=0x4, act_val=0x38)
    cocotb.log.info(f"Tile 1 loaded: {rows1} rows")
    drain1 = await drain_tile(dut)
    cocotb.log.info(f"Tile 1 drained in {drain1} cycles")

    # Tile 2
    rows2 = await load_tile(dut, wt_val=0x5, act_val=0x40)
    cocotb.log.info(f"Tile 2 loaded: {rows2} rows")
    drain2 = await drain_tile(dut)
    cocotb.log.info(f"Tile 2 drained in {drain2} cycles")

    # Collect output
    output = await collect_output(dut)
    cocotb.log.info(f"Output rows: {len(output)}")

    assert len(output) > 0, "No output after 2 tiles — block_data_buffer never triggered"
    cocotb.log.info(f"PASS: {len(output)} output rows produced")


@cocotb.test(expect_error=AssertionError)  # Known failure until PR #8 fixes test data encoding
async def golden_verification_test(dut):
    """
    Verify numerical correctness with uniform data.
    weight=0x6 (3.0 in E1M2), activation=0x40 (2.0 in E4M3), scale=1.0.
    Expected PE accumulator: 48.0 = 3.0 * 2.0 * 8 (BLOCK_DIM dot product).
    """
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    WT_VAL = 0x6    # 3.0 in E1M2
    ACT_VAL = 0x40  # 2.0 in E4M3
    SCALE_VAL = 0x7F  # bias=127 → scale=1.0

    # Resolve PE(0,0) handle for accumulator probing
    pe00 = None
    try:
        sa0 = dut.dut_i.matrix_compute_unit.genblk1[0].systolic_array_inst
        for path_fn in [
            lambda sa: sa.pe_row[0].pe_col[0].genblk1.first_row_mini_sys_init.row_inx[0].col_idx[0].genblk1.first_row_pe.default_pe_inst,
            lambda sa: sa.pe_row[0].pe_col[0].first_row_mini_sys_init.row_inx[0].col_idx[0].first_row_pe.default_pe_inst,
        ]:
            try:
                pe00 = path_fn(sa0)
                break
            except Exception:
                pass
    except Exception:
        pass

    # Track peak PE accumulator during load (drain clears it)
    peak_acc = 0

    # Tile 1
    rows1 = await load_tile(dut, wt_val=WT_VAL, act_val=ACT_VAL, scale_val=SCALE_VAL)
    # Sample accumulator after load settles
    for _ in range(20):
        await RisingEdge(dut.clk)
        if pe00 is not None:
            try:
                acc = int(pe00.out_fp.value)
                if acc > peak_acc:
                    peak_acc = acc
            except Exception:
                pass

    await drain_tile(dut)

    # Tile 2
    rows2 = await load_tile(dut, wt_val=WT_VAL, act_val=ACT_VAL, scale_val=SCALE_VAL)
    for _ in range(20):
        await RisingEdge(dut.clk)
        if pe00 is not None:
            try:
                acc = int(pe00.out_fp.value)
                if acc > peak_acc:
                    peak_acc = acc
            except Exception:
                pass

    await drain_tile(dut)

    # Verify PE accumulator
    # 48.0 in E6M5: exp=36, mant=16 → 0x490
    cocotb.log.info(f"Peak PE accumulator: 0x{peak_acc:X} (expected 0x490 = 48.0)")

    pe_elem = peak_acc & 0xFFF
    assert pe_elem == 0x490, f"PE output 0x{pe_elem:03X} != expected 0x490 (48.0)"

    # Also verify output bus
    output = await collect_output(dut)
    cocotb.log.info(f"Output rows: {len(output)}")
    assert len(output) > 0, "No output produced"


# =====================================================================
# Runner
# =====================================================================

@pytest.mark.dev
def test_matrix_machine_v2_latency():
    veri_runner(
        group="matrix_machine",
        module="matrix_machine_tb_wrapper",
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
        extra_build_args=["-DSIMULATION"],
        trace=True,
        test_module="matrix_machine_tb",
    )


if __name__ == "__main__":
    test_matrix_machine_v2_latency()
