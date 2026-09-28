#!/usr/bin/env python3

import logging
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))

import pytest
import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH
from cfl_cocotb.fp_generation import FpGenerator

exp_width = 4
mant_width = 3
vect_dim = 4
level = math.ceil(math.log2(vect_dim))


generator = FpGenerator(exp_width, mant_width)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)

@cocotb.test()
async def prefix_scan_test(dut):
    """Test prefix scan operation with proper timing"""
    
    # Start clock first
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())  # Slower clock
    
    await Timer(5, units="ns")
    cocotb.log.info("Starting prefix scan test")
    
    # Proper reset sequence
    dut.rst.value = 0
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0
    dut.v_out_ready.value = 0
    dut.operation.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    # Test data: [1.0, 2.0, 3.0, 4.0]
    # Expected prefix scan: [1.0, 3.0, 6.0, 10.0]
    test_values = [1.0, 2.0, 3.0, 4.0]
    fp_values, results = generator.generate_specified_value_fp_input(test_values)
    input_data_a = sum((results[n] << (exp_width + mant_width + 1) * n) for n in range(vect_dim))
    
    cocotb.log.info(f"Input values: {test_values}")
    cocotb.log.info(f"Input data packed: {hex(input_data_a)}")
    
    # Set inputs
    dut.v_in_a.value = input_data_a
    dut.v_in_b.value = 0  # Not used for prefix scan
    dut.operation.value = 8  # PREFIX_SCAN_V_ELEMENT
    dut.v_out_ready.value = 1
    dut.v_in_a_valid.value = 1
    await RisingEdge(dut.clk)
    
    # Assert valid
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0
    
    cocotb.log.info("Inputs set, waiting for computation...")
    
    # Wait for prefix scan to complete - check for valid output
    timeout = 0
    max_timeout = 50
    
    while timeout < max_timeout:
        await RisingEdge(dut.clk)
        timeout += 1
        
        # Log intermediate signals for debugging
        if timeout % 10 == 0:
            cocotb.log.info(f"Cycle {timeout}: v_out={hex(dut.v_out.value)}, v_out_valid={dut.v_out_valid.value}")
        
        # Check if we have valid output
        if dut.v_out_valid.value == 1:
            cocotb.log.info(f"Valid output detected at cycle {timeout}")
            break
            
        # Also check if output is non-zero (for combinational logic)
        if dut.v_out.value != 0:
            cocotb.log.info(f"Non-zero output detected at cycle {timeout}")
            # Wait a few more cycles to be sure
            for _ in range(5):
                await RisingEdge(dut.clk)
            break
   # breakpoint()
    if timeout >= max_timeout:
        cocotb.log.error("Timeout waiting for prefix scan result")
    
    # Deassert valid
    
    
    # Get and verify results
    actual_result = generator.translate_packed_array_fp(vect_dim, exp_width, mant_width, dut.v_out.value)
    
    # Calculate expected prefix scan
    expected_result = []
    cumsum = 0.0
    for val in test_values:
        cumsum += val
        expected_result.append(cumsum)
    
    cocotb.log.info("<-------  Prefix Scan Results  --------->")
    cocotb.log.info(f"Input: {test_values}")
    cocotb.log.info(f"Expected: {expected_result}")
    cocotb.log.info(f"Hardware Binary: {hex(dut.v_out.value)}")
    cocotb.log.info(f"Hardware Result: {actual_result}")
    #breakpoint()
    # Debug internal signals
    try:
        cocotb.log.info(f"use_prefix_scan: {dut.use_prefix_scan.value}")
        cocotb.log.info(f"prefix_scan_valid: {dut.prefix_scan_valid.value}")
        cocotb.log.info(f"prefix_scan_ready: {dut.prefix_scan_ready.value}")
    except AttributeError:
        cocotb.log.warning("Some internal signals not accessible")
    
    # Verify results with reasonable tolerance
    tolerance = 0.5  # Adjust based on your FP precision
    all_passed = True
    
    for j in range(vect_dim):
        error = abs(actual_result[j] - expected_result[j]) if expected_result[j] != 0 else abs(actual_result[j])
        if error > tolerance:
            cocotb.log.error(f"✗ Mismatch at index {j}: expected {expected_result[j]}, got {actual_result[j]}, error {error}")
            all_passed = False
        else:
            cocotb.log.info(f"[PASS] Index {j}: {actual_result[j]} ≈ {expected_result[j]}")
    
    # Add assertion to actually fail the test if results are wrong
    assert all_passed, f"Prefix scan test failed - see errors above"
    
    cocotb.log.info("[PASS] Prefix scan test PASSED!")

@cocotb.test()
async def exp_latency_test(dut):
    """Measure latency of V_EXP_V (vector exp) operation.

    Expected (from configuration.svh, behavioral mode):
      VECTOR_EXP_CYCLES = 6
      pipelined formula : 1 + VECTOR_EXP_CYCLES = 7
      alone formula     : 6 + 3 + VECTOR_EXP_CYCLES + 1 = 16
    """

    LATENCY_VLEN        = 8
    EXP_OP              = 4   # EXP_V_ELEMENT = 4'h4
    SIM_EXP_CYCLES      = 6
    PIPELINED_FORMULA   = 1 + SIM_EXP_CYCLES           # = 7 (pipelined: 1+VECTOR_EXP_CYCLES)
    ALONE_FORMULA       = 6 + 3 + SIM_EXP_CYCLES + 1  # = 16 (with ISA dispatch overhead)
    TIMEOUT_CYCLES      = 200

    # --- clock ---
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # --- reset (active-high) ---
    dut.rst.value         = 0
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0
    dut.v_out_ready.value  = 0
    dut.operation.value    = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    dut.rst.value = 1           # assert reset
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0           # release reset
    await RisingEdge(dut.clk)

    # --- pack VLEN=8 FP values into v_in_a ---
    fp_gen_lat = FpGenerator(exp_width, mant_width)
    test_values = [1.0, 0.5, -0.25, 0.125, 2.0, -1.0, 0.75, -0.5]
    fp_values_lat, results_lat = fp_gen_lat.generate_specified_value_fp_input(test_values)

    bits_per_elem = exp_width + mant_width + 1  # 8 bits
    input_data_a = sum(
        (results_lat[n] << bits_per_elem * n)
        for n in range(LATENCY_VLEN)
    )

    dut.v_in_a.value = input_data_a
    dut.v_in_b.value = 0        # v_in_b unused by exp, but join2 needs both valid

    # --- drive operation and handshake signals ---
    dut.operation.value    = EXP_OP
    dut.v_out_ready.value  = 1
    dut.v_in_a_valid.value = 1
    dut.v_in_b_valid.value = 1  # join2 requires both valid simultaneously

    # --- record start cycle ---
    start_cycle = 0
    await RisingEdge(dut.clk)
    start_cycle = 1  # first cycle after valid was asserted

    # --- wait for v_out_valid ---
    elapsed = 0
    while elapsed < TIMEOUT_CYCLES:
        if dut.v_out_valid.value == 1:
            break
        await RisingEdge(dut.clk)
        elapsed += 1

    end_cycle = elapsed + start_cycle
    measured_latency = end_cycle

    # --- deassert inputs ---
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0

    # --- report ---
    match_str = "EXACT MATCH" if measured_latency == PIPELINED_FORMULA else f"DIFF (pipelined={PIPELINED_FORMULA}, alone={ALONE_FORMULA})"
    cocotb.log.info(
        f"[LATENCY] V_EXP_V: {measured_latency} cycles "
        f"(sim VECTOR_EXP_CYCLES={SIM_EXP_CYCLES}, pipelined_formula={PIPELINED_FORMULA})  {match_str}"
    )

    if elapsed >= TIMEOUT_CYCLES:
        cocotb.log.error(f"[LATENCY] V_EXP_V: TIMEOUT after {TIMEOUT_CYCLES} cycles — v_out_valid never asserted")

    assert elapsed < TIMEOUT_CYCLES, f"V_EXP_V latency test timed out after {TIMEOUT_CYCLES} cycles"


@pytest.mark.dev
def latency_test():
    veri_runner(
        group = "vector_machine",
        module = "fp_elementwise_compute_unit",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/cast")
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        module_param_list=[
            {"EXP_WIDTH": 4, "MANT_WIDTH": 3, "VLEN": 8},
        ],
        trace = True,
    )


if __name__ == "__main__":
    latency_test()
