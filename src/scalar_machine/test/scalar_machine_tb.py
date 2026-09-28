#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))

import logging
import pytest
import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

# FP16 (E5M10) encoding of 1.0:  sign=0, exp=bias=15, mant=0
# => bits = (15 << 10) | 0 = 0x3C00
FP16_ONE = 0x3C00

# S_FP_OP enum values (from operation.svh)
STALL_S_FP = 0x0
EXP_FP     = 0x5
RECI_FP    = 0x6

# Simulator-predicted cycle counts (behavioral mode, configuration.svh pipeline_pkg)
SCALAR_FP_EXP_CYCLES  = 2
SCALAR_FP_RECI_CYCLES = 2

TIMEOUT_CYCLES = 200


async def reset_dut(dut):
    """
    Assert active-high reset for 30 ns, release, wait 20 ns.
    Also force-clears sfu_in_use (not in reset block of RTL) so a fresh
    measurement can proceed.
    """
    dut.rst.value       = 1
    dut.operation.value = STALL_S_FP
    dut.data_in.value   = 0
    dut.reg_waddr.value = 0
    await Timer(30, units="ns")   # hold reset 3 cycles @ 10 ns
    dut.rst.value = 0
    # Force-clear sfu_in_use since the RTL never resets it.
    # The signal is exposed via --public-flat-rw as fp_sfu__DOT__sfu_in_use.
    # Use _id() directly (hasattr fails on names with __DOT__ due to Python
    # name-mangling of double-underscore identifiers).
    try:
        dut._id("sfu_in_use", extended=False).value = 0
        cocotb.log.info("[SCALAR] force-cleared sfu_in_use via internal signal")
    except Exception as e:
        cocotb.log.warning(f"[SCALAR] could not clear sfu_in_use: {e}")
    await Timer(20, units="ns")   # settle 2 cycles


async def measure_sfu(dut, op_val, op_name, sim_cycles):
    """
    Measure latency from operation assertion until data_out_valid=1.

    RTL state machine:
      - On posedge clk, if !sfu_in_use & operation != STALL_S_FP:
            recorded_operation <= operation
            data_in_valid      <= 1
            sfu_in_use         <= 1
      - data_out_valid pulses when result is ready (via register_slice)

    Counting: wait for first RisingEdge after asserting operation = cycle 1,
    keep incrementing until data_out_valid=1.
    """
    dut.data_in.value   = FP16_ONE
    dut.reg_waddr.value = 0
    dut.operation.value = op_val

    cycles = 0
    for cyc in range(1, TIMEOUT_CYCLES + 1):
        await RisingEdge(dut.clk)
        if int(dut.data_out_valid.value) == 1:
            cycles = cyc
            break

    # Deassert operation so the RTL completion branch can sample STALL
    dut.operation.value = STALL_S_FP

    if cycles == 0:
        cocotb.log.error(f"[SCALAR] {op_name} TIMEOUT: data_out_valid never asserted")
    else:
        match = "MATCH" if cycles == sim_cycles else f"DIFF (RTL={cycles} vs sim={sim_cycles})"
        cocotb.log.info(
            f"[SCALAR] {op_name}: {cycles} cycles  (sim={sim_cycles})  {match}"
        )

    return cycles


@cocotb.test()
async def fp_sfu_latency_test(dut):
    """
    Measure cycle latency for fp_sfu scalar FP special-function operations:
      1. EXP_FP  (op=0x5) - FP exponentiation  -> data_out_valid
      2. RECI_FP (op=0x6) - FP reciprocal       -> data_out_valid

    Input: FP16 (E5M10) encoding of 1.0 = 0x3C00.
    Each operation is measured independently with a reset between them.
    Note: sfu_in_use is not in the RTL reset block; we force it to 0 via
    --public-flat-rw after each measurement so the next op can proceed.
    """
    cocotb.log.info("========== FP SFU Latency Test ==========")
    cocotb.log.info(f"FP16(1.0) = 0x{FP16_ONE:04X}")

    # Start 10 ns clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # =========================================================================
    # EXP_FP
    # =========================================================================
    cocotb.log.info("--- EXP_FP (op=0x5) ---")
    await reset_dut(dut)
    exp_cycles = await measure_sfu(dut, EXP_FP, "S_EXP_FP", SCALAR_FP_EXP_CYCLES)
    assert exp_cycles > 0, "EXP_FP timed out"

    # Let the RTL self-clear recorded_operation, then force sfu_in_use=0
    for _ in range(3):
        await RisingEdge(dut.clk)

    # =========================================================================
    # RECI_FP
    # =========================================================================
    cocotb.log.info("--- RECI_FP (op=0x6) ---")
    await reset_dut(dut)
    reci_cycles = await measure_sfu(dut, RECI_FP, "S_RECI_FP", SCALAR_FP_RECI_CYCLES)
    assert reci_cycles > 0, "RECI_FP timed out"

    # =========================================================================
    # Summary
    # =========================================================================
    cocotb.log.info("=" * 50)
    exp_tag  = "MATCH" if exp_cycles  == SCALAR_FP_EXP_CYCLES  else f"DIFF (RTL={exp_cycles} vs sim={SCALAR_FP_EXP_CYCLES})"
    reci_tag = "MATCH" if reci_cycles == SCALAR_FP_RECI_CYCLES else f"DIFF (RTL={reci_cycles} vs sim={SCALAR_FP_RECI_CYCLES})"
    cocotb.log.info(f"[SCALAR] S_EXP_FP:  {exp_cycles} cycles  (sim={SCALAR_FP_EXP_CYCLES})  {exp_tag}")
    cocotb.log.info(f"[SCALAR] S_RECI_FP: {reci_cycles} cycles  (sim={SCALAR_FP_RECI_CYCLES})  {reci_tag}")
    cocotb.log.info("=" * 50)

    for _ in range(5):
        await RisingEdge(dut.clk)

    cocotb.log.info("FP SFU latency test complete")


@pytest.mark.dev
def fp_sfu_latency_pytest():
    veri_runner(
        group="scalar_machine",
        module="fp_sfu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/fixed_operation"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=True,
        test_module="scalar_machine_tb",
    )


if __name__ == "__main__":
    fp_sfu_latency_pytest()
