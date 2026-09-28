#!/usr/bin/env python3

import logging
import math

import cocotb
import cocotb.utils
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import pytest
from cfl_cocotb import veri_runner, MXBlockFPConverter, SRC_PATH


# Parameters Definition
mxfp_exp_width = 4
mxfp_mant_width = 3
mxfp_scale_width = 8
block_dim = 2
fp_exp_width = 8
fp_mant_width = 23
M = 4
N = 4
K = 4   # K==M → ACC_NUM=1 (single block, cleaner latency measurement)

CLOCK_PERIOD_NS = 2  # 2ns = 500MHz

generator = MXBlockFPConverter(mxfp_exp_width, mxfp_mant_width, mxfp_scale_width, block_dim)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def perf_model_cycles(M, N, K, block_dim):
    """
    Analytical cycle estimate for one GEMM (M x K) @ (K x N) using perf_model formula.
    M_MM pipelined latency = 1 + BLEN per tile.
    Tiles = ceil(M/block_dim) * ceil(K/block_dim) * ceil(N/block_dim)
    """
    tiles = math.ceil(M / block_dim) * math.ceil(K / block_dim) * math.ceil(N / block_dim)
    cycles_per_tile = 1 + block_dim
    return tiles * cycles_per_tile


@cocotb.test()
async def random_mcu_test(dut):
    # Start clock generation
    TESTCASE_SIZE = 1
    TIMEOUT_CYCLES = 2000

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())

    await Timer(4, units="ns")
    cocotb.log.info("Starting MXFP MCU latency measurement test")
    # Apply Reset
    dut.rst.value = 1
    dut.acc_waddr.value = 0
    dut.fetch_next_acc_waddr_valid.value = 0
    dut.wait_for_output.value = 0
    await Timer(4, units="ns")  # Allow some settling time
    dut.rst.value = 0
    await Timer(4, units="ns")  # Hold reset for 5ns

    for i in range(TESTCASE_SIZE):
        # Generate random floating point values (K//block_dim blocks, each block_dim elements)
        num_blocks = K // block_dim
        input_vals = [1.24, 2.01, 1.0231, 0.9820, 1.34, 2.56, 0.75, 1.89][:num_blocks * block_dim]
        while len(input_vals) < num_blocks * block_dim:
            input_vals.append(1.0)
        v_mx_fp_scales, v_mx_fp_elems = generator.generate_certain_values(input_vals)
        v_ele_data = 0
        v_scale_data = 0
        for j in range(num_blocks):
            v_ele_data   += sum((v_mx_fp_elems[j][n] << (mxfp_exp_width + mxfp_mant_width + 1) * (n + j * block_dim)) for n in range(block_dim))
            v_scale_data += (v_mx_fp_scales[j] << (mxfp_scale_width) * j)

        # Drive inputs and record start time
        start_ns = cocotb.utils.get_sim_time(units='ns')

        dut.v1_element.value = v_ele_data
        dut.v1_scale.value = v_scale_data
        dut.v2_element.value = v_ele_data
        dut.v2_scale.value = v_scale_data
        dut.v_result_ready.value = 1

        # Phase 1: MM_IC (0x5) — load M rows into the systolic array
        dut.control.value = 5  # MM_IC
        dut.v1_in_valid.value = 1
        dut.v2_in_valid.value = 1

        rows_loaded = 0
        for _ in range(TIMEOUT_CYCLES):
            await RisingEdge(dut.clk)
            v1_rdy = int(dut.v1_in_ready.value)
            v2_rdy = int(dut.v2_in_ready.value)
            if v1_rdy and v2_rdy:
                rows_loaded += 1
                if rows_loaded >= M:
                    break

        # Phase 2: MM_WO (0x7) — trigger pipeline drain and output
        dut.v1_in_valid.value = 0
        dut.v2_in_valid.value = 0
        dut.control.value = 7  # MM_WO
        # ACC_NUM = K/M = 1 with K==M, so acc_waddr=0 is the only (and last) block
        dut.acc_waddr.value = 0
        dut.fetch_next_acc_waddr_valid.value = 1
        dut.wait_for_output.value = 1
        cocotb.log.info(f"MM_IC phase done ({rows_loaded} rows loaded). Switching to MM_WO.")

        # Wait for all M output rows (v_result_write_req asserts once per output row)
        first_output_ns = None
        rows_received = 0
        for cyc in range(TIMEOUT_CYCLES):
            await RisingEdge(dut.clk)
            if cyc < 30 or cyc % 50 == 0:
                try:
                    cocotb.log.info(
                        f"  [DBG cyc={cyc}] eip={int(dut.mcu_active.value)} "
                        f"rlo={int(dut.ready_to_load_output.value)} "
                        f"fc={int(dut.feed_counter.value)} "
                        f"gmv={int(dut.gemm_result_valid.value)} "
                        f"wrq={int(dut.v_result_write_req.value)}"
                    )
                except Exception:
                    pass
            if dut.v_result_write_req.value:
                if first_output_ns is None:
                    first_output_ns = cocotb.utils.get_sim_time(units='ns')
                rows_received += 1
                if rows_received >= M:
                    break

        end_ns = cocotb.utils.get_sim_time(units='ns')

        if first_output_ns is None:
            cocotb.log.error(f"[LATENCY] TIMEOUT: no output received within {TIMEOUT_CYCLES} cycles")
        else:
            pipeline_cycles = (first_output_ns - start_ns) / CLOCK_PERIOD_NS
            total_cycles    = (end_ns - start_ns) / CLOCK_PERIOD_NS
            expected_cycles = perf_model_cycles(M, N, K, block_dim)

            cocotb.log.info("=" * 60)
            cocotb.log.info(f"[LATENCY] GEMM ({M}x{K}) @ ({K}x{N}), BLOCK_DIM={block_dim}")
            cocotb.log.info(f"[LATENCY]   RTL pipeline latency (first output): {pipeline_cycles:.0f} cycles")
            cocotb.log.info(f"[LATENCY]   RTL total latency    (all {M} rows):  {total_cycles:.0f} cycles")
            cocotb.log.info(f"[LATENCY]   Simulator estimate   (perf_model):   {expected_cycles} cycles")
            cocotb.log.info(f"[LATENCY]   Ratio (RTL/sim):                     {total_cycles/expected_cycles:.3f}x")
            cocotb.log.info("=" * 60)


@pytest.mark.dev
def mcu_test():
    # Run tests with different params
    veri_runner(
        group = "systolic_gemm_mx",
        module = "mx_systolic_mcu",
        additional_include_paths = [
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/gemv")
        ],       
        test_module = "mxfp_systolic_mcu_latency_tb",
        definitions_path = [
            str(SRC_PATH / "definitions"),
            str(SRC_PATH / "memory/HBM/TileLink_Lib")],
        module_param_list=[
            {
                "MX_T_MANT_WIDTH" : mxfp_mant_width,
                "MX_T_EXP_WIDTH" : mxfp_exp_width,
                "MX_L_MANT_WIDTH" : mxfp_mant_width,
                "MX_L_EXP_WIDTH" : mxfp_exp_width,
                "MX_SCALE_WIDTH" : mxfp_scale_width,
                "BLOCK_DIM" : block_dim,
                "ACC_FP_MANT_WIDTH" : fp_mant_width,
                "ACC_FP_EXP_WIDTH" : fp_exp_width,
                "N" : N,
                "M" : M,
                "K" : K }
        ],
        trace = True,
    )

if __name__ == "__main__":
    mcu_test()
