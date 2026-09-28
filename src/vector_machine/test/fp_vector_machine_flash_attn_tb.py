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
from cfl_cocotb.fp_generation import FpGenerator

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

# Python-side params (same convention as vector_machine_tb.py)
EXP, MANT, VLEN = 7, 8, 8
ELEM_W = 1 + EXP + MANT  # = 16 bits per element

# Simulator-predicted cycle counts
SIM_VECTOR_MAX_CYCLES = 4
SIM_VECTOR_ADD_CYCLES = 7
SIM_VECTOR_EXP_CYCLES_PIPELINE = 7   # 1 + VECTOR_EXP_CYCLES
SIM_VECTOR_SUM_CYCLES = 20
SIM_SOFTMAX_TOTAL = (SIM_VECTOR_MAX_CYCLES + SIM_VECTOR_ADD_CYCLES +
                     SIM_VECTOR_EXP_CYCLES_PIPELINE + SIM_VECTOR_SUM_CYCLES)  # = 38

TIMEOUT_CYCLES = 200


def pack_v(bus_elems, elem_width):
    """Pack list of element words [0..VLEN-1] into single int bus (lane 0 at LSB)."""
    v = 0
    mask = (1 << elem_width) - 1
    for i, w in enumerate(bus_elems):
        v |= (int(w) & mask) << (i * elem_width)
    return v


async def reset_dut(dut):
    """Active-high reset: assert rst=1, then release rst=0."""
    dut.rst.value = 1
    dut.v_a_valid.value = 0
    if hasattr(dut, "v_b_valid"):   dut.v_b_valid.value = 0
    if hasattr(dut, "s_in_valid"):  dut.s_in_valid.value = 0
    dut.element_v_control.value = 0
    dut.reduct_v_control.value = 0
    await Timer(30, units="ns")  # hold reset 3 cycles at 10ns
    dut.rst.value = 0
    await Timer(30, units="ns")  # settle


async def measure_reduction(dut, gen, op_val, v_bus, op_name, sim_cycles):
    """
    Measure latency for a reduction op (V_RED_MAX or V_RED_SUM).

    RTL timing: recorded_reduct_v_control updates ONE clock after reduct_v_control
    is set. s_in_buffer gates s_in_valid through only when recorded != STALL.
    So: set control → wait 1 clock → assert v_a_valid + s_in_valid for 2 cycles.
    """
    dut.reduct_v_control.value = op_val
    dut.element_v_control.value = 0
    if hasattr(dut, "v_out_ready"): dut.v_out_ready.value = 1
    if hasattr(dut, "s_out_ready"): dut.s_out_ready.value = 1

    # Wait ONE clock: recorded_reduct_v_control latches to op_val
    await RisingEdge(dut.clk)

    # Now assert both data signals
    dut.v_a_in.value = v_bus
    dut.v_a_valid.value = 1
    if hasattr(dut, "s_in_valid"):
        dut.s_in_valid.value = 1

    # Hold 2 cycles so both register_slices propagate
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.v_a_valid.value = 0
    if hasattr(dut, "s_in_valid"):
        dut.s_in_valid.value = 0
    dut.reduct_v_control.value = 0

    # Count cycles until s_out_valid
    cycles = 0
    for cyc in range(1, TIMEOUT_CYCLES + 1):
        await RisingEdge(dut.clk)
        if hasattr(dut, "s_out_valid") and int(dut.s_out_valid.value) == 1:
            cycles = cyc
            break

    if cycles == 0:
        cocotb.log.error(f"[FLASH_ATTN] {op_name} TIMEOUT: s_out_valid never asserted")
    else:
        match = "MATCH" if cycles == sim_cycles else f"diff (RTL={cycles} vs sim={sim_cycles})"
        cocotb.log.info(f"[FLASH_ATTN] {op_name}: {cycles} cycles  (sim={sim_cycles})  {match}")

    return cycles


async def measure_elementwise(dut, gen, op_val, v_a_bus, v_b_bus, op_name, sim_cycles):
    """
    Measure latency for an element-wise op (V_ADD_VV or V_EXP_V).

    element_v_out_valid (internal, accessible via --public-flat-rw) is used
    since v_wreq has a pipeline_compute_track alignment issue.
    Both v_a_valid AND v_b_valid are required (fp_elementwise_compute_unit has join2).
    """
    dut.element_v_control.value = op_val
    dut.reduct_v_control.value = 0
    if hasattr(dut, "v_out_ready"): dut.v_out_ready.value = 1
    if hasattr(dut, "s_out_ready"): dut.s_out_ready.value = 1

    dut.v_a_in.value = v_a_bus
    if hasattr(dut, "v_b_in"):    dut.v_b_in.value = v_b_bus
    dut.v_a_valid.value = 1
    if hasattr(dut, "v_b_valid"): dut.v_b_valid.value = 1

    await RisingEdge(dut.clk)
    dut.v_a_valid.value = 0
    if hasattr(dut, "v_b_valid"): dut.v_b_valid.value = 0

    # Count cycles until element_v_out_valid
    cycles = 0
    for cyc in range(1, TIMEOUT_CYCLES + 1):
        await RisingEdge(dut.clk)
        if hasattr(dut, "element_v_out_valid") and int(dut.element_v_out_valid.value) == 1:
            cycles = cyc
            break

    if cycles == 0:
        cocotb.log.error(f"[FLASH_ATTN] {op_name} TIMEOUT: element_v_out_valid never asserted")
    else:
        match = "MATCH" if cycles == sim_cycles else f"diff (RTL={cycles} vs sim={sim_cycles})"
        cocotb.log.info(f"[FLASH_ATTN] {op_name}: {cycles} cycles  (sim={sim_cycles})  {match}")

    return cycles


@cocotb.test()
async def flash_attn_softmax_latency_test(dut):
    """
    Measure cycle latency for each step of flash-attention online softmax:
      1. V_RED_MAX   - find max of score vector        -> s_out_valid
      2. V_ADD_VV    - subtract max (scores - max)     -> element_v_out_valid
      3. V_EXP_V     - compute exp(scores - max)       -> element_v_out_valid
      4. V_RED_SUM   - sum exp values                  -> s_out_valid

    Each step is run independently with a reset between them to avoid residual
    in_preparation_stage state. Total = sum of individual step latencies.
    """
    cocotb.log.info("========== Flash-Attention Softmax Latency Test ==========")
    gen = FpGenerator(EXP, MANT)
    cocotb.log.info(f"Config: EXP={EXP}, MANT={MANT}, VLEN={VLEN}, ELEM_W={ELEM_W}")

    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Initial defaults
    if hasattr(dut, "broadcast_fp2"):        dut.broadcast_fp2.value = 0
    if hasattr(dut, "s_in"):                 dut.s_in.value = 0
    if hasattr(dut, "s_wtarget"):            dut.s_wtarget.value = 0
    if hasattr(dut, "result_waddr"):         dut.result_waddr.value = 0
    if hasattr(dut, "result_waddr_update"):  dut.result_waddr_update.value = 0
    dut.v_a_in.value = 0
    if hasattr(dut, "v_b_in"): dut.v_b_in.value = 0

    # Build test vectors
    score_vals = [float(i * 0.5 + 0.1) for i in range(VLEN)]
    _, enc_scores = gen.generate_specified_value_fp_input(score_vals)
    v_scores_bus = pack_v(enc_scores[:VLEN], ELEM_W)
    cocotb.log.info(f"Score values: {score_vals}")

    # Dummy v_b for EXP (value doesn't matter for unary op, but join2 requires valid)
    dummy_bus = pack_v([0] * VLEN, ELEM_W)

    # =========================================================================
    # Step 1: V_RED_MAX
    # =========================================================================
    cocotb.log.info("--- Step 1: V_RED_MAX ---")
    await reset_dut(dut)
    step1_cycles = await measure_reduction(dut, gen, 2, v_scores_bus,
                                           "Step 1 V_RED_MAX", SIM_VECTOR_MAX_CYCLES)
    assert step1_cycles > 0, "Step 1 (V_RED_MAX) timed out"

    # =========================================================================
    # Step 2: V_ADD_VV (subtract max — for latency, use scores as both inputs)
    # =========================================================================
    cocotb.log.info("--- Step 2: V_ADD_VV ---")
    await reset_dut(dut)
    step2_cycles = await measure_elementwise(dut, gen, 1, v_scores_bus, v_scores_bus,
                                              "Step 2 V_ADD_VV", SIM_VECTOR_ADD_CYCLES)
    assert step2_cycles > 0, "Step 2 (V_ADD_VV) timed out"

    # =========================================================================
    # Step 3: V_EXP_V
    # =========================================================================
    cocotb.log.info("--- Step 3: V_EXP_V ---")
    await reset_dut(dut)
    step3_cycles = await measure_elementwise(dut, gen, 4, v_scores_bus, dummy_bus,
                                              "Step 3 V_EXP_V", SIM_VECTOR_EXP_CYCLES_PIPELINE)
    assert step3_cycles > 0, "Step 3 (V_EXP_V) timed out"

    # =========================================================================
    # Step 4: V_RED_SUM
    # =========================================================================
    cocotb.log.info("--- Step 4: V_RED_SUM ---")
    await reset_dut(dut)
    step4_cycles = await measure_reduction(dut, gen, 1, v_scores_bus,
                                           "Step 4 V_RED_SUM", SIM_VECTOR_SUM_CYCLES)
    assert step4_cycles > 0, "Step 4 (V_RED_SUM) timed out"

    # =========================================================================
    # Summary
    # =========================================================================
    total_cycles = step1_cycles + step2_cycles + step3_cycles + step4_cycles
    cocotb.log.info("=" * 60)
    cocotb.log.info(f"[FLASH_ATTN] Step 1 V_RED_MAX:  {step1_cycles} cycles  (sim={SIM_VECTOR_MAX_CYCLES})")
    cocotb.log.info(f"[FLASH_ATTN] Step 2 V_ADD_VV:   {step2_cycles} cycles  (sim={SIM_VECTOR_ADD_CYCLES})")
    cocotb.log.info(f"[FLASH_ATTN] Step 3 V_EXP_V:    {step3_cycles} cycles  (sim={SIM_VECTOR_EXP_CYCLES_PIPELINE})")
    cocotb.log.info(f"[FLASH_ATTN] Step 4 V_RED_SUM:  {step4_cycles} cycles  (sim={SIM_VECTOR_SUM_CYCLES})")
    cocotb.log.info(f"[FLASH_ATTN] Total softmax:     {total_cycles} cycles  (sim={SIM_SOFTMAX_TOTAL})")
    cocotb.log.info("=" * 60)

    for _ in range(5):
        await RisingEdge(dut.clk)

    cocotb.log.info("Flash-attention softmax latency test complete")


@pytest.mark.dev
def flash_attn_latency_test():
    veri_runner(
        group="vector_machine",
        module="vector_machine",
        additional_include_paths=[
            str(SRC_PATH / "basic_components" / "buffer"),
            str(SRC_PATH / "basic_components" / "common"),
            str(SRC_PATH / "basic_components" / "fp_operation"),
            str(SRC_PATH / "basic_components" / "hadamard_transform"),
            str(SRC_PATH / "basic_components" / "synopsis_ip_inst"),
            str(SRC_PATH / "basic_components" / "conversion"),
            str(SRC_PATH / "basic_components" / "fixed_operation"),
            str(SRC_PATH / "basic_components" / "int_operation"),
            str(SRC_PATH / "basic_components" / "synopsis"),
            str(SRC_PATH / "basic_components" / "cast"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=True,
        test_module="fp_vector_machine_flash_attn_tb",
    )

if __name__ == "__main__":
    flash_attn_latency_test()
