#!/usr/bin/env python3
"""
MCU test with E4M3 x E4M3 -> E6M5 accumulator.
Exercises adder tree clk and signed_exp width fixes.
"""
import logging
import cocotb
import cocotb.utils
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import pytest
from cfl_cocotb.runner import veri_runner
from cfl_cocotb import MXBlockFPConverter, SRC_PATH

# Test E4M3 x E4M3 -> E6M5 (narrow accumulator, exercises bugs 2,3,6)
mx_t_exp_width = 4    # E4M3 (top)
mx_t_mant_width = 3
mx_l_exp_width = 4    # E4M3 (left)
mx_l_mant_width = 3
mx_scale_width = 8
block_dim = 2          # Same as passing test
acc_fp_exp_width = 6   # E6M5 accumulator (narrow! was float32 in original test)
acc_fp_mant_width = 5
M = 4
N = 4
K = 4

CLOCK_PERIOD_NS = 2

generator = MXBlockFPConverter(mx_l_exp_width, mx_l_mant_width, mx_scale_width, block_dim)
generator_t = MXBlockFPConverter(mx_t_exp_width, mx_t_mant_width, mx_scale_width, block_dim)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


@cocotb.test()
async def real_params_mcu_test(dut):
    TIMEOUT_CYCLES = 500

    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())

    await Timer(4, units="ns")
    cocotb.log.info("Starting MXFP MCU test with E1M2 x E4M3 -> E6M5")
    dut.rst.value = 1
    dut.acc_waddr.value = 0
    dut.fetch_next_acc_waddr_valid.value = 0
    dut.wait_for_output.value = 0
    await Timer(4, units="ns")
    dut.rst.value = 0
    await Timer(4, units="ns")

    # Generate test data (E4M3 values)
    num_blocks = K // block_dim
    t_vals = [1.25, 2.0, -1.5, 0.875][:num_blocks * block_dim]
    while len(t_vals) < num_blocks * block_dim:
        t_vals.append(1.0)

    l_vals = [1.5, -2.5, 0.625, 1.75][:num_blocks * block_dim]
    while len(l_vals) < num_blocks * block_dim:
        l_vals.append(1.0)

    t_scales, t_elems = generator_t.generate_certain_values(t_vals)
    l_scales, l_elems = generator.generate_certain_values(l_vals)

    # Pack top (E1M2) data
    v_t_ele_data = 0
    v_t_scale_data = 0
    for j in range(num_blocks):
        v_t_ele_data += sum((t_elems[j][n] << (mx_t_exp_width + mx_t_mant_width + 1) * (n + j * block_dim)) for n in range(block_dim))
        v_t_scale_data += (t_scales[j] << mx_scale_width * j)

    # Pack left (E4M3) data
    v_l_ele_data = 0
    v_l_scale_data = 0
    for j in range(num_blocks):
        v_l_ele_data += sum((l_elems[j][n] << (mx_l_exp_width + mx_l_mant_width + 1) * (n + j * block_dim)) for n in range(block_dim))
        v_l_scale_data += (l_scales[j] << mx_scale_width * j)

    cocotb.log.info(f"Top (E4M3) values: {t_vals}")
    cocotb.log.info(f"Left (E4M3) values: {l_vals}")

    start_ns = cocotb.utils.get_sim_time(units='ns')

    dut.v1_element.value = v_t_ele_data
    dut.v1_scale.value = v_t_scale_data
    dut.v2_element.value = v_l_ele_data
    dut.v2_scale.value = v_l_scale_data
    dut.v_result_ready.value = 1

    # Phase 1: MM_IC
    dut.control.value = 5
    dut.v1_in_valid.value = 1
    dut.v2_in_valid.value = 1

    rows_loaded = 0
    for _ in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.v1_in_ready.value) and int(dut.v2_in_ready.value):
            rows_loaded += 1
            if rows_loaded >= M:
                break

    cocotb.log.info(f"MM_IC done: {rows_loaded} rows loaded")

    # Phase 2: MM_WO
    dut.v1_in_valid.value = 0
    dut.v2_in_valid.value = 0
    dut.control.value = 7
    dut.acc_waddr.value = 0
    dut.fetch_next_acc_waddr_valid.value = 1
    dut.wait_for_output.value = 1

    rows_received = 0
    first_output_ns = None
    for cyc in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        # Debug probes every 50 cycles
        if cyc < 200 or cyc % 200 == 0:
            try:
                cocotb.log.info(
                    f"  [cyc={cyc}] "
                    f"eip={int(dut.mcu_active.value)} "
                    f"rlo={int(dut.ready_to_load_output.value)} "
                    f"fc={int(dut.feed_counter.value)} "
                    f"ctrl={int(dut.control_in_exe.value)} "
                    f"gmv={int(dut.gemm_result_valid.value)} "
                    f"gmr={int(dut.gemm_result_w_ready.value)} "
                    f"gbv={int(dut.gebm_result_valid.value)} "
                    f"gbr={int(dut.gebm_result_ready.value)} "
                    f"qv={int(dut.quantise_data_in_valid.value)} "
                    f"wrq={int(dut.v_result_write_req.value)}"
                )
            except Exception as e:
                cocotb.log.info(f"  [cyc={cyc}] probe error: {e}")
        if dut.v_result_write_req.value:
            if first_output_ns is None:
                first_output_ns = cocotb.utils.get_sim_time(units='ns')
            rows_received += 1
            cocotb.log.info(f"Output row {rows_received}: v_result = {hex(int(dut.v_result.value))}")
            if rows_received >= M:
                break

    end_ns = cocotb.utils.get_sim_time(units='ns')

    if first_output_ns is None:
        cocotb.log.error(f"TIMEOUT: no output within {TIMEOUT_CYCLES} cycles")
        assert False, "No output received"
    else:
        pipeline_cycles = (first_output_ns - start_ns) / CLOCK_PERIOD_NS
        total_cycles = (end_ns - start_ns) / CLOCK_PERIOD_NS
        cocotb.log.info("=" * 60)
        cocotb.log.info(f"E4M3 x E4M3 -> E6M5 MCU test PASSED")
        cocotb.log.info(f"Pipeline latency: {pipeline_cycles:.0f} cycles, Total: {total_cycles:.0f} cycles")
        cocotb.log.info(f"Received {rows_received} output rows")
        cocotb.log.info("=" * 60)


@pytest.mark.dev
def test_mcu_e1m2():
    veri_runner(
        group="systolic_gemm_mx",
        module="mx_systolic_mcu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/gemv"),
        ],
        test_module="mxfp_systolic_mcu_e1m2_latency_tb",
        definitions_path=[
            str(SRC_PATH / "definitions"),
            str(SRC_PATH / "memory/HBM/TileLink_Lib")],
        module_param_list=[
            {
                "MX_T_MANT_WIDTH": mx_t_mant_width,
                "MX_T_EXP_WIDTH": mx_t_exp_width,
                "MX_L_MANT_WIDTH": mx_l_mant_width,
                "MX_L_EXP_WIDTH": mx_l_exp_width,
                "MX_SCALE_WIDTH": mx_scale_width,
                "BLOCK_DIM": block_dim,
                "ACC_FP_MANT_WIDTH": acc_fp_mant_width,
                "ACC_FP_EXP_WIDTH": acc_fp_exp_width,
                "FP_EXP_WIDTH": acc_fp_exp_width,
                "FP_MANT_WIDTH": acc_fp_mant_width,
                "N": N,
                "M": M,
                "K": K,
            }
        ],
        trace=True,
    )


if __name__ == "__main__":
    test_mcu_e1m2()
