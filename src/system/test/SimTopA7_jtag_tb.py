#!/usr/bin/env python3
"""Full JTAG-to-activation testbench: the UNCHANGED PlenaDriver activates the PLENA core over the
custom BSCAN JTAG path (JtagBscanTransport -> jtag_axi_bscan -> SimTopA7 AXI-lite -> core), exactly
as the real board will over the FTDI JTAG cable with no Vivado. Proves the Vivado-free JTAG driver
works end to end in simulation. Result correctness is the standard V_SRAM_RESULT_FILE dump.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent.parent
_TOOLS_PATH = _PROJECT_PATH / "tools"
_TEST_PATH = Path(__file__).resolve().parent
for p in [str(_TOOLS_PATH), str(_TEST_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cfl_cocotb.runner import veri_runner, SRC_PATH
from cfl_tools.logger import get_logger
from test_platform import PLENATestPlatform

from plena_driver import PlenaDriver
from jtag_bscan_transport import JtagBscanTransport
# reuse the workload/offset plumbing from the AXI-lite tb
from SimTopA7_tb import _read_words, _read_offset_bytes, _next_pow2

logger = get_logger("simtopa7_jtag")
logger.setLevel(logging.DEBUG)

INSTRUCTION_LENGTH = 32


@cocotb.test()
async def test(dut):
    prog_file = os.environ["PROG_WORDS_FILE"]
    start_word = int(os.environ["START_WORD"])
    words = _read_words(prog_file)
    logger.info(f"JTAG activation: {len(words)} instr words, start IMEM word {start_word}")

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())   # core / AXI-lite clock
    cocotb.start_soon(Clock(dut.tck, 40, units="ns").start())   # JTAG clock (slower -> real CDC)

    # Reset core + JTAG tap.
    dut.sel.value = 0
    dut.capture.value = 0
    dut.shift.value = 0
    dut.update.value = 0
    dut.tdi.value = 0
    dut.rst.value = 1
    dut.tap_reset.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
    await RisingEdge(dut.tck)
    await RisingEdge(dut.tck)
    dut.rst.value = 0
    dut.tap_reset.value = 0
    await ClockCycles(dut.clk, 5)

    transport = JtagBscanTransport(dut, dut.tck, dut.clk, axi_wait=32)
    drv = PlenaDriver(transport, dbg_words=6, log=logger)

    logger.info("Waiting for DDR3 calibration over JTAG...")
    await drv.wait_calib()
    logger.info("Loading program into IMEM over JTAG...")
    await drv.load_program(words, start_word=start_word)
    logger.info("Launching core over JTAG (SOFT_RST)...")
    await drv.launch()
    logger.info("Polling system_break over JTAG...")
    done = await drv.wait_done(timeout_polls=200000)
    if not done:
        raise RuntimeError("BUG: core did not reach system_break over the JTAG path (hung)")

    rst_cnt = await drv.read_rst_cnt()
    logger.info(f"JTAG activation OK: system_break reached. RST_CNT={rst_cnt}")
    for _ in range(128):
        await RisingEdge(dut.clk)
    logger.info("=== JTAG ACTIVATION PATH PASSED (BSCAN scan -> load -> launch -> system_break) ===")


def get_module_params():
    return {
        "INSTRUCTION_LENGTH": INSTRUCTION_LENGTH,
        "IMEM_DEPTH": int(os.environ["IMEM_DEPTH"]),
        "FAKE_HBM_INIT_FILE": f"\"{os.environ['FAKE_HBM_INIT_FILE']}\"",
        "FP_MEM_INIT_FILE": f"\"{os.environ['FP_MEM_INIT_FILE']}\"",
        "INT_MEM_INIT_FILE": f"\"{os.environ['INT_MEM_INIT_FILE']}\"",
        "VECTOR_MEM_RESULT_FILE": f"\"{os.environ['VECTOR_MEM_RESULT_FILE']}\"",
        "FP_REG_RESULT_FILE": f"\"{os.environ.get('FP_REG_RESULT_FILE', '')}\"",
    }


def test_SimTopA7_jtag():
    skip_build = os.environ.get("SKIP_BUILD", "0") == "1"
    workload_dir = os.environ.get("WORKLOAD_DIR")
    sim_build_dir = Path(workload_dir) if workload_dir else None
    return veri_runner(
        group="system",
        module="SimTopA7_jtag",
        test_dir=Path(__file__).parent,
        extra_build_args=["-DSIMULATION"],
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/systolic_gemm_mx"),
            str(SRC_PATH / "basic_components/systolic_gemm_mxint"),
            str(SRC_PATH / "basic_components/gemv"),
            str(SRC_PATH / "basic_components/synopsis/rtl"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "frontend"),
            str(SRC_PATH / "control"),
            str(SRC_PATH / "matrix_machine"),
            str(SRC_PATH / "vector_machine"),
            str(SRC_PATH / "scalar_machine"),
            str(SRC_PATH / "memory/matrix_sram"),
            str(SRC_PATH / "memory/vector_sram"),
            str(SRC_PATH / "memory/scratch_sram"),
            str(SRC_PATH / "memory/scalar_sram"),
            str(SRC_PATH / "memory/HBM"),
            str(SRC_PATH / "core"),
            str(SRC_PATH / "fpga/common"),
        ],
        definitions_path=[
            str(SRC_PATH / "definitions"),
            str(SRC_PATH / "memory/HBM/TileLink_Lib"),
        ],
        module_param_list=[get_module_params()],
        trace=False,
        skip_build=skip_build,
        sim_build_dir=sim_build_dir,
    )


def run_with_workload(workload_dir: str):
    os.environ["WORKLOAD_DIR"] = workload_dir
    platform = PLENATestPlatform.from_workload(workload_dir)
    platform.prepare()

    prog_file = str(Path(workload_dir) / "generated_machine_code.mem")
    words = _read_words(prog_file)
    offset = _read_offset_bytes()
    start_word = offset >> 2
    imem_depth = _next_pow2(start_word + len(words) + 16)

    os.environ["PROG_WORDS_FILE"] = prog_file
    os.environ["START_WORD"] = str(start_word)
    os.environ["IMEM_DEPTH"] = str(imem_depth)
    logger.info(f"offset={offset}B start_word={start_word} nwords={len(words)} IMEM_DEPTH={imem_depth}")

    num_failed = test_SimTopA7_jtag()
    if num_failed:
        logger.error(f"RTL simulation FAILED: {num_failed} cocotb test(s) did not pass")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLENA SimTopA7 JTAG activation testbench")
    parser.add_argument("--workload-dir", type=str, required=True)
    args = parser.parse_args()
    run_with_workload(args.workload_dir)
