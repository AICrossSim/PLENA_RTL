#!/usr/bin/env python3
"""M1: JTAG->DDR3 loader unit test. Reuses the SimTopA7_jtag RTL (which now carries the loader) with
an EMPTY fake_ddr3 (FAKE_HBM_INIT_FILE=""), and over the BSCAN JTAG path writes several 128-bit beats
into DDR3 then reads them back -- validating the loader FSM, the 2:1 bridge-input mux, PutFullData/Get
through tl_to_axi4, and the fake_ddr3 write path. The core is present but never launched.
"""
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
from plena_driver import PlenaDriver
from jtag_bscan_transport import JtagBscanTransport

logger = get_logger("ddr3_loader")
logger.setLevel(logging.DEBUG)


@cocotb.test()
async def test_ddr3_loader(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    cocotb.start_soon(Clock(dut.tck, 40, units="ns").start())

    dut.sel.value = 0; dut.capture.value = 0; dut.shift.value = 0; dut.update.value = 0; dut.tdi.value = 0
    dut.rst.value = 1; dut.tap_reset.value = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
    await RisingEdge(dut.tck); await RisingEdge(dut.tck)
    dut.rst.value = 0; dut.tap_reset.value = 0
    await ClockCycles(dut.clk, 5)

    drv = PlenaDriver(JtagBscanTransport(dut, dut.tck, dut.clk, axi_wait=32), log=logger)
    await drv.wait_calib()

    beats = [0x0123456789ABCDEFFEDCBA9876543210,
             0x11111111222222223333333344444444,
             0xDEADBEEFCAFEB0BA0102030405060708,
             0xFFFFFFFF00000000A5A5A5A55A5A5A5A]
    data = b"".join(b.to_bytes(16, "little") for b in beats)

    await drv.load_ddr3(0x000, data)
    logger.info(f"wrote {len(beats)} beats; DDR3_ADDR now {await drv.t.axi_read(0x1C):#x} (expect 0x40)")

    for i, exp in enumerate(beats):
        got = await drv.read_ddr3_beat(0x000 + i * 16)
        assert got == exp, f"beat {i} @ {i*16:#x}: {got:#034x} != {exp:#034x}"
    logger.info(f"=== DDR3 LOADER OK: {len(beats)} beats written + read back bit-exact over JTAG ===")


def _run():
    for k in ("FAKE_HBM_INIT_FILE", "FP_MEM_INIT_FILE", "INT_MEM_INIT_FILE",
              "VECTOR_MEM_RESULT_FILE", "FP_REG_RESULT_FILE"):
        os.environ[k] = ""
    os.environ["IMEM_DEPTH"] = "512"
    params = {
        "INSTRUCTION_LENGTH": 32, "IMEM_DEPTH": 512,
        "FAKE_HBM_INIT_FILE": "\"\"", "FP_MEM_INIT_FILE": "\"\"", "INT_MEM_INIT_FILE": "\"\"",
        "VECTOR_MEM_RESULT_FILE": "\"\"", "FP_REG_RESULT_FILE": "\"\"",
    }
    return veri_runner(
        group="system", module="SimTopA7_jtag", test_module="ddr3_loader_tb",
        test_dir=Path(__file__).parent, extra_build_args=["-DSIMULATION"],
        additional_include_paths=[
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
        module_param_list=[params], trace=False,
        skip_build=(os.environ.get("SKIP_BUILD", "0") == "1"),
    )


if __name__ == "__main__":
    n = _run()
    if n:
        logger.error(f"ddr3_loader test FAILED ({n})")
        sys.exit(1)
