#!/usr/bin/env python3
"""M3 full-loop test: the ENTIRE host-driven cycle over the Vivado-free BSCAN JTAG path, with a
genuinely empty DDR3 filled only over JTAG:

  DDR3 empty -> (JTAG) load DDR3 data region -> (JTAG) load IMEM -> (JTAG) SOFT_RST launch ->
  core reads DDR3 + runs -> (JTAG) read VSRAM rows back -> compare to the RTL's own V_SRAM dump.

Uses SimTopA7_jtag_dbg (SimTopA7_dbg has the real debug_vsram readback via the ported debug core in
src/core_dbg). The JTAG-read rows are written to <build>/vector_result_jtag.mem and compared to the
$finish dump <build>/vector_result.mem after the run: equal => the readback path returns the core's
actual VSRAM, and the DDR3 load + run + readback loop is proven end to end over JTAG.
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
for p in [
          str(_PROJECT_PATH / "tools"), str(Path(__file__).resolve().parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cfl_cocotb.runner import veri_runner, SRC_PATH
from cfl_tools.logger import get_logger
from test_platform import PLENATestPlatform

from plena_driver import PlenaDriver
from jtag_bscan_transport import JtagBscanTransport
from SimTopA7_tb import _read_words, _read_offset_bytes, _next_pow2

logger = get_logger("simtopa7_jtag_dbg")
logger.setLevel(logging.DEBUG)

INSTRUCTION_LENGTH = 32
N_ROWS_CHECK = 16   # VSRAM rows to read back over JTAG and self-consistency-check vs the dump


def _hbm_data_region_bytes(hbm_file, nbytes):
    """The DATA region = first `nbytes` bytes of hbm.mem (rows [0, offset); instructions live above
    and are served from IMEM, so they are not loaded into DDR3). hbm.mem = 32-byte little-endian rows."""
    out = bytearray()
    with open(hbm_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if s.lower().startswith("0x"):
                s = s[2:]
            if len(s) != 64:
                continue
            val = int(s, 16)
            out += val.to_bytes(32, "little")
    return bytes(out[:nbytes])


@cocotb.test()
async def test(dut):
    prog_file  = os.environ["PROG_WORDS_FILE"]
    hbm_file   = os.environ["HBM_MEM_FILE"]
    start_word = int(os.environ["START_WORD"])
    data_bytes = int(os.environ["DATA_REGION_BYTES"])
    jtag_out   = os.environ["JTAG_VSRAM_FILE"]
    words = _read_words(prog_file)
    data  = _hbm_data_region_bytes(hbm_file, data_bytes)
    logger.info(f"full-loop: {len(data)}B DDR3 data ({(len(data)+15)//16} beats), {len(words)} IMEM words")

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    cocotb.start_soon(Clock(dut.tck, 40, units="ns").start())

    dut.sel.value = 0; dut.capture.value = 0; dut.shift.value = 0; dut.update.value = 0; dut.tdi.value = 0
    dut.rst.value = 1; dut.tap_reset.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
    await RisingEdge(dut.tck); await RisingEdge(dut.tck)
    dut.rst.value = 0; dut.tap_reset.value = 0
    await ClockCycles(dut.clk, 5)

    drv = PlenaDriver(JtagBscanTransport(dut, dut.tck, dut.clk, axi_wait=32), dbg_words=6, log=logger)
    await drv.wait_calib()
    logger.info("loading DDR3 data over JTAG...")
    await drv.load_ddr3(0, data)
    logger.info("loading IMEM over JTAG...")
    await drv.load_program(words, start_word=start_word)
    logger.info("launching + running over JTAG...")
    await drv.launch()
    if not await drv.wait_done(timeout_polls=200000):
        raise RuntimeError("BUG: core did not reach system_break in the full JTAG loop")
    for _ in range(128):
        await RisingEdge(dut.clk)

    logger.info(f"reading {N_ROWS_CHECK} VSRAM rows back over JTAG...")
    rows = []
    for L in range(N_ROWS_CHECK):
        rows.append(await drv.read_vsram_row(L << 4))   # dump line L <-> element addr L*VLEN
    with open(jtag_out, "w") as f:
        for v in rows:
            f.write(f"{v:048x}\n")
    logger.info("=== FULL JTAG LOOP RAN (DDR3 load -> IMEM load -> launch -> break -> VSRAM readback) ===")


def get_module_params():
    return {
        "INSTRUCTION_LENGTH": INSTRUCTION_LENGTH,
        "IMEM_DEPTH": int(os.environ["IMEM_DEPTH"]),
        "FAKE_HBM_INIT_FILE": "\"\"",                       # EMPTY DDR3 -- filled only over JTAG
        "FP_MEM_INIT_FILE": f"\"{os.environ['FP_MEM_INIT_FILE']}\"",
        "INT_MEM_INIT_FILE": f"\"{os.environ['INT_MEM_INIT_FILE']}\"",
        "VECTOR_MEM_RESULT_FILE": f"\"{os.environ['VECTOR_MEM_RESULT_FILE']}\"",
        "FP_REG_RESULT_FILE": f"\"{os.environ.get('FP_REG_RESULT_FILE', '')}\"",
    }


def test_SimTopA7_jtag_dbg():
    return veri_runner(
        group="system", module="SimTopA7_jtag_dbg", test_module="SimTopA7_jtag_dbg_tb",
        test_dir=Path(__file__).parent, extra_build_args=["-DSIMULATION"],
        additional_include_paths=[str(SRC_PATH / "core_dbg")] + [   # core_dbg FIRST -> debug plena.sv wins
            str(SRC_PATH / d) for d in [
                "basic_components/common", "basic_components/mx_fp_operation", "basic_components/fp_operation",
                "basic_components/conversion", "basic_components/buffer", "basic_components/fixed_operation",
                "basic_components/int_operation", "basic_components/cast", "basic_components/systolic_gemm_mx",
                "basic_components/systolic_gemm_mxint", "basic_components/gemv", "basic_components/synopsis/rtl",
                "basic_components/synopsis", "basic_components/synopsis_ip_inst", "basic_components/hadamard_transform",
                "frontend", "control", "matrix_machine", "vector_machine", "scalar_machine",
                "memory/matrix_sram", "memory/vector_sram", "memory/scratch_sram", "memory/scalar_sram",
                "memory/HBM", "core", "fpga/common",
            ]
        ],
        definitions_path=[str(SRC_PATH / "definitions"), str(SRC_PATH / "memory/HBM/TileLink_Lib")],
        module_param_list=[get_module_params()],
        trace=False, skip_build=(os.environ.get("SKIP_BUILD", "0") == "1"),
        sim_build_dir=Path(os.environ["WORKLOAD_DIR"]),
    )


def run_with_workload(workload_dir: str):
    wd = Path(workload_dir)
    os.environ["WORKLOAD_DIR"] = workload_dir
    PLENATestPlatform.from_workload(workload_dir).prepare()

    prog_file = str(wd / "generated_machine_code.mem")
    words = _read_words(prog_file)
    offset = _read_offset_bytes()
    start_word = offset >> 2
    os.environ["PROG_WORDS_FILE"] = prog_file
    os.environ["HBM_MEM_FILE"] = str(wd / "hbm.mem")
    os.environ["START_WORD"] = str(start_word)
    os.environ["DATA_REGION_BYTES"] = str(offset)                 # data region = [0, offset)
    os.environ["IMEM_DEPTH"] = str(_next_pow2(start_word + len(words) + 16))
    os.environ["VECTOR_MEM_RESULT_FILE"] = str(wd / "vector_result.mem")
    os.environ["JTAG_VSRAM_FILE"] = str(wd / "vector_result_jtag.mem")
    logger.info(f"offset={offset}B start_word={start_word} IMEM_DEPTH={os.environ['IMEM_DEPTH']}")

    n = test_SimTopA7_jtag_dbg()
    if n:
        logger.error(f"RTL sim FAILED ({n})")
        sys.exit(1)

    # Self-consistency: JTAG-read VSRAM rows == the RTL's own $finish dump.
    dump = [l.strip().lower() for l in open(wd / "vector_result.mem") if l.strip()]
    jtag = [l.strip().lower() for l in open(wd / "vector_result_jtag.mem") if l.strip()]
    mism = [i for i in range(min(len(jtag), N_ROWS_CHECK)) if jtag[i] != dump[i]]
    if mism:
        logger.error(f"JTAG VSRAM readback != dump at rows {mism[:8]} (of {N_ROWS_CHECK}) -- readback path bug")
        sys.exit(1)
    logger.info(f"=== VSRAM READBACK OK: {N_ROWS_CHECK} JTAG-read rows == RTL dump (full loop verified) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLENA SimTopA7 JTAG full-loop testbench")
    parser.add_argument("--workload-dir", type=str, required=True)
    args = parser.parse_args()
    run_with_workload(args.workload_dir)
