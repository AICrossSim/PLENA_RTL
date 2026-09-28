#!/usr/bin/env python3
"""Unit test simulating JTAG communication through jtag_axi_bscan.

DUT = jtag_bscan_harness (jtag_axi_bscan BSCAN->AXI-lite master + a small AXI-lite register file).
The test drives the JTAG tap ports (the DR scan sequence) via the shared JtagBscanTransport code
and checks write/read round-trips -- validating DR bit-order, the tck<->aclk CDC handshake, the
pipelined response/flush, and the AXI-lite handshake, all without a real BSCANE2. tck (40ns) and
aclk (10ns) run at different rates so the clock-domain crossing is genuinely exercised.
"""
import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

THIS_DIR = Path(__file__).resolve().parent
RTL_MASTER = THIS_DIR.parent / "rtl" / "jtag_axi_bscan.sv"
RTL_HARNESS = THIS_DIR / "jtag_bscan_harness.sv"
# JtagBscanTransport + bscan_scan live with the other transports.
_SYS_TEST = THIS_DIR.parent.parent / "system" / "test"
if str(_SYS_TEST) not in sys.path:
    sys.path.insert(0, str(_SYS_TEST))

from jtag_bscan_transport import JtagBscanTransport, bscan_scan


async def _reset(dut):
    dut.sel.value = 0
    dut.capture.value = 0
    dut.shift.value = 0
    dut.update.value = 0
    dut.tdi.value = 0
    dut.tap_reset.value = 1
    dut.aresetn.value = 0
    await ClockCycles(dut.aclk, 4)
    await RisingEdge(dut.tck)
    await RisingEdge(dut.tck)
    dut.tap_reset.value = 0
    dut.aresetn.value = 1
    await ClockCycles(dut.aclk, 4)


@cocotb.test()
async def test_write_read_roundtrip(dut):
    """Explicit 3-scan pipeline check (write -> read-issue -> flush) with done verification."""
    cocotb.start_soon(Clock(dut.tck, 40, units="ns").start())
    cocotb.start_soon(Clock(dut.aclk, 10, units="ns").start())
    await _reset(dut)

    ADDR, DATA = 0x10, 0xDEADBEEF
    await bscan_scan(dut, (1 << 39) | ((ADDR & 0x7F) << 32) | DATA, dut.tck)  # scan A: write
    await ClockCycles(dut.aclk, 64)
    outB = await bscan_scan(dut, (0 << 39) | ((ADDR & 0x7F) << 32), dut.tck)  # scan B: read-issue
    await ClockCycles(dut.aclk, 64)
    outC = await bscan_scan(dut, 0, dut.tck)                                  # scan C: flush

    assert (outB >> 39) & 1 == 1, f"write did not complete (outB={outB:#012x})"
    assert (outC >> 39) & 1 == 1, f"read did not complete (outC={outC:#012x})"
    assert (outC & 0xFFFFFFFF) == DATA, f"readback {outC & 0xFFFFFFFF:#x} != {DATA:#x}"
    dut._log.info(f"pipeline round-trip OK: wrote {DATA:#x} @ {ADDR:#x}, read back {outC & 0xFFFFFFFF:#x}")


@cocotb.test()
async def test_transport_api(dut):
    """Same path via the JtagBscanTransport API (what PlenaDriver uses): several addr/data pairs."""
    cocotb.start_soon(Clock(dut.tck, 40, units="ns").start())
    cocotb.start_soon(Clock(dut.aclk, 10, units="ns").start())
    await _reset(dut)

    t = JtagBscanTransport(dut, dut.tck, dut.aclk)
    vectors = [(0x00, 0x11111111), (0x10, 0xCAFEB0BA), (0x40, 0x00000001), (0x54, 0xFFFFFFFF)]
    for addr, data in vectors:
        await t.axi_write(addr, data)
    for addr, data in vectors:
        got = await t.axi_read(addr)
        assert got == data, f"addr {addr:#x}: read {got:#x} != wrote {data:#x}"
    dut._log.info(f"transport API OK: {len(vectors)} write/read pairs round-tripped")


def test_runner():
    from cocotb.runner import get_runner, get_results

    sim = os.getenv("SIM", "verilator")
    work = THIS_DIR / "build" / "jtag_bscan"
    runner = get_runner(sim)
    runner.build(
        verilog_sources=[str(RTL_MASTER), str(RTL_HARNESS)],
        hdl_toplevel="jtag_bscan_harness",
        build_args=["-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL", "--timing"],
        parameters={"ADDR_W": 7},
        build_dir=str(work),
        always=True,
    )
    results_xml = runner.test(
        hdl_toplevel="jtag_bscan_harness",
        hdl_toplevel_lang="verilog",
        test_module="jtag_axi_bscan_tb",
        build_dir=str(work),
        test_dir=str(THIS_DIR),
        results_xml="results_jtag.xml",
    )
    num, fail = get_results(Path(results_xml))
    print(f"# jtag_axi_bscan: {num-fail}/{num} passed")
    assert fail == 0, f"{fail} jtag test(s) failed"
    print("# jtag_axi_bscan: ALL passed")


if __name__ == "__main__":
    test_runner()
