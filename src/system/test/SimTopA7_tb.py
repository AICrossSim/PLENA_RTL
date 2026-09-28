#!/usr/bin/env python3
"""SimTopA7 RTL testbench: validates the FPGA BRING-UP / activation path.

Unlike SimTop/SimTopDDR (which stream instructions from fake_hbm), SimTopA7 loads the
program into the on-chip IMEM over the AXI-lite control port using the PlenaDriver -- the
exact sequence the board's JTAG-to-AXI master performs: wait DDR3 calib -> load IMEM ->
SOFT_RST launch -> poll system_break. Result correctness is checked from the standard
V_SRAM_RESULT_FILE dump (same as SimTopDDR); this tb proves the ACTIVATION path.

The core fetches at (pc + INSTRUCTION_STORAGE_OFFSET), decoded to IMEM word addr>>2, so
program word k lands at IMEM word (offset>>2)+k. IMEM_DEPTH is sized to fit.
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

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
from cocotb_axil_bfm import CocotbAxiLiteBFM

logger = get_logger("simtopa7")
logger.setLevel(logging.DEBUG)

INSTRUCTION_LENGTH = 32


def _read_words(mem_file):
    words = []
    with open(mem_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            words.append(int(s, 16) if s.lower().startswith("0x") else int(s, 16))
    return words


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@cocotb.test()
async def test(dut):
    prog_file = os.environ["PROG_WORDS_FILE"]
    start_word = int(os.environ["START_WORD"])
    words = _read_words(prog_file)
    logger.info(f"Activation test: {len(words)} instr words, start IMEM word {start_word}")

    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())

    # Reset: hold high 20 cycles, release. BFM ctor idles the la_* inputs.
    bfm = CocotbAxiLiteBFM(dut, dut.clk)
    dut.rst.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    drv = PlenaDriver(bfm, dbg_words=6, log=logger)

    logger.info("Waiting for DDR3 calibration...")
    await drv.wait_calib()

    logger.info("Loading program over AXI-lite...")
    await drv.load_program(words, start_word=start_word)

    logger.info("Launching core (SOFT_RST)...")
    await drv.launch()

    logger.info("Waiting for system_break (C_BREAK)...")
    done = await drv.wait_done(timeout_polls=200000)
    if not done:
        raise RuntimeError("BUG: core did not reach system_break after activation (hung)")

    rst_cnt = await drv.read_rst_cnt()
    logger.info(f"Activation OK: system_break reached. RST_CNT={rst_cnt}")

    # Drain compute pipelines so the VSRAM writeback lands before the $finish dump.
    for _ in range(128):
        await RisingEdge(dut.clk)
    logger.info("=== ACTIVATION PATH PASSED (load -> launch -> system_break) ===")


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


def test_SimTopA7():
    skip_build = os.environ.get("SKIP_BUILD", "0") == "1"
    workload_dir = os.environ.get("WORKLOAD_DIR")
    sim_build_dir = Path(workload_dir) if workload_dir else None
    return veri_runner(
        group="system",
        module="SimTopA7",
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


def _parse_verilog_int(tok):
    """Parse a SystemVerilog integer literal: 32'h6C0, 12'd1728, 0x6C0, or 1728."""
    tok = tok.strip()
    m = re.match(r"(?:\d+)?'([hdbo])([0-9a-fA-F_]+)", tok)
    if m:
        base = {"h": 16, "d": 10, "b": 2, "o": 8}[m.group(1).lower()]
        return int(m.group(2).replace("_", ""), base)
    return int(tok, 0)


def _read_offset_bytes():
    cfg = SRC_PATH / "definitions" / "configuration.svh"
    txt = cfg.read_text()
    m = re.search(r"INSTRUCTION_STORAGE_OFFSET\s*=\s*([^;]+);", txt)
    if not m:
        raise RuntimeError("INSTRUCTION_STORAGE_OFFSET not found in configuration.svh")
    return _parse_verilog_int(m.group(1))


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

    num_failed = test_SimTopA7()
    if num_failed:
        logger.error(f"RTL simulation FAILED: {num_failed} cocotb test(s) did not pass")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLENA SimTopA7 activation testbench")
    parser.add_argument("--workload-dir", type=str, required=True)
    args = parser.parse_args()
    run_with_workload(args.workload_dir)
