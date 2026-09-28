"""Standalone cocotb unit test for the tl_to_axi4 burst bridge.

The bridge has flat ports and no package dependencies, so it is built in isolation
(no TileLink/configuration includes needed). We attach a small behavioural AXI4 memory
slave in Python and drive TileLink Get / PutFullData transactions, checking a full
write -> read round trip. Two geometries are covered:

  * DATA_W=128, AXI_DATA_W=64  -> BURST_LEN=2 (multi-beat INCR burst)
  * DATA_W=128, AXI_DATA_W=128 -> BURST_LEN=1 (single beat; exercises the BEAT_CNT_W>=1 fix)

Run directly:  python3 src/fpga/common/test/tl_to_axi4_tb.py
"""

import os
from pathlib import Path
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, ClockCycles, with_timeout, Timer

# TileLink opcodes (match tl_to_axi4.v localparams)
TL_GET       = 4
TL_PUT_FULL  = 0
TL_ACK       = 0   # AccessAck (D opcode for writes)
TL_ACK_DATA  = 1   # AccessAckData (D opcode for reads)

THIS_DIR = Path(__file__).resolve().parent
RTL = THIS_DIR.parent / "rtl" / "tl_to_axi4.sv"


def _int(sig):
    return int(sig.value)


async def axi_mem_slave(dut, mem, axi_data_w):
    """Minimal single-outstanding AXI4 memory slave.

    Stores one AXI-beat-wide word per beat address in `mem` (dict: addr -> int).
    Handles AW/W/B and AR/R with INCR bursts. Beat ordering is preserved, so a
    write-then-read of the same base address round-trips regardless of bit layout.
    """
    beat_bytes = axi_data_w // 8
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_bvalid.value = 0
    dut.m_axi_bid.value = 0
    dut.m_axi_bresp.value = 0
    dut.m_axi_arready.value = 0
    dut.m_axi_rvalid.value = 0
    dut.m_axi_rid.value = 0
    dut.m_axi_rdata.value = 0
    dut.m_axi_rresp.value = 0
    dut.m_axi_rlast.value = 0

    while True:
        await RisingEdge(dut.clk)

        # ---- Write address ----
        if _int(dut.m_axi_awvalid):
            awaddr = _int(dut.m_axi_awaddr)
            awlen = _int(dut.m_axi_awlen)
            dut.m_axi_awready.value = 1
            await RisingEdge(dut.clk)
            dut.m_axi_awready.value = 0
            # ---- Write data beats ----
            dut.m_axi_wready.value = 1
            for i in range(awlen + 1):
                # wait for a valid beat
                while not _int(dut.m_axi_wvalid):
                    await RisingEdge(dut.clk)
                mem[awaddr + i * beat_bytes] = _int(dut.m_axi_wdata)
                last = _int(dut.m_axi_wlast)
                await RisingEdge(dut.clk)
                if last:
                    break
            dut.m_axi_wready.value = 0
            # ---- Write response ----
            dut.m_axi_bvalid.value = 1
            dut.m_axi_bresp.value = 0
            while not _int(dut.m_axi_bready):
                await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            dut.m_axi_bvalid.value = 0
            continue

        # ---- Read address ----
        if _int(dut.m_axi_arvalid):
            araddr = _int(dut.m_axi_araddr)
            arlen = _int(dut.m_axi_arlen)
            dut.m_axi_arready.value = 1
            await RisingEdge(dut.clk)
            dut.m_axi_arready.value = 0
            # ---- Read data beats ----
            # Present each beat for a full cycle and advance only after the bridge has
            # actually consumed it (rvalid && rready sampled in ReadOnly). Updating rdata
            # and stepping in the same delta would let the bridge latch the stale beat.
            i = 0
            while i <= arlen:
                beat = mem.get(araddr + i * beat_bytes, 0)
                dut.m_axi_rvalid.value = 1
                dut.m_axi_rdata.value = beat
                dut.m_axi_rlast.value = 1 if (i == arlen) else 0
                dut.m_axi_rresp.value = 0
                await ReadOnly()
                consumed = _int(dut.m_axi_rready) == 1
                await RisingEdge(dut.clk)
                if consumed:
                    i += 1
            dut.m_axi_rvalid.value = 0
            dut.m_axi_rlast.value = 0
            continue


async def tl_idle(dut):
    dut.tl_a_valid.value = 0
    dut.tl_a_opcode.value = 0
    dut.tl_a_size.value = 0
    dut.tl_a_source.value = 0
    dut.tl_a_address.value = 0
    dut.tl_a_mask.value = 0
    dut.tl_a_data.value = 0
    dut.tl_d_ready.value = 1


async def _wait_a_accept(dut):
    """With tl_a_valid already asserted, wait for the cycle tl_a_ready is high
    (sampled in ReadOnly, since tl_a_ready is registered and deasserts on accept),
    then take the accept edge."""
    while True:
        await ReadOnly()
        if _int(dut.tl_a_ready) == 1:
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # accept edge


async def _wait_d(dut):
    """Wait for a D-channel beat (tl_d_ready held high), capture it, consume it."""
    while True:
        await ReadOnly()
        if _int(dut.tl_d_valid) == 1:
            opcode = _int(dut.tl_d_opcode)
            data = _int(dut.tl_d_data)
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # accept the D beat
    return opcode, data


async def tl_put(dut, addr, data, size, mask):
    """Issue a PutFullData and wait for the AccessAck."""
    dut.tl_a_valid.value = 1
    dut.tl_a_opcode.value = TL_PUT_FULL
    dut.tl_a_size.value = size
    dut.tl_a_source.value = 1
    dut.tl_a_address.value = addr
    dut.tl_a_mask.value = mask
    dut.tl_a_data.value = data
    await _wait_a_accept(dut)
    dut.tl_a_valid.value = 0
    opcode, _ = await _wait_d(dut)
    assert opcode == TL_ACK, f"expected AccessAck({TL_ACK}), got {opcode}"


async def tl_get(dut, addr, size):
    """Issue a Get and return the data from AccessAckData."""
    dut.tl_a_valid.value = 1
    dut.tl_a_opcode.value = TL_GET
    dut.tl_a_size.value = size
    dut.tl_a_source.value = 2
    dut.tl_a_address.value = addr
    dut.tl_a_mask.value = 0
    dut.tl_a_data.value = 0
    await _wait_a_accept(dut)
    dut.tl_a_valid.value = 0
    opcode, data = await _wait_d(dut)
    assert opcode == TL_ACK_DATA, f"expected AccessAckData({TL_ACK_DATA}), got {opcode}"
    return data


@cocotb.test()
async def write_read_roundtrip(dut):
    data_w = len(dut.tl_a_data)
    axi_data_w = len(dut.m_axi_wdata)
    size = (data_w // 8).bit_length() - 1  # $clog2(DATA_W/8)
    full_mask = (1 << (data_w // 8)) - 1

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    mem = {}
    cocotb.start_soon(axi_mem_slave(dut, mem, axi_data_w))
    await tl_idle(dut)

    # reset
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    rng = random.Random(0xC0FFEE)
    mask_full = full_mask
    for n in range(6):
        addr = (n * (data_w // 8)) & 0xFFFF      # distinct, aligned addresses
        data = rng.getrandbits(data_w)
        await with_timeout(tl_put(dut, addr, data, size, mask_full), 5, "us")
        got = await with_timeout(tl_get(dut, addr, size), 5, "us")
        assert got == data, f"roundtrip @0x{addr:x}: wrote 0x{data:x} read 0x{got:x}"
        dut._log.info(f"ok  addr=0x{addr:04x}  data=0x{data:x}")

    await ClockCycles(dut.clk, 4)


def test_runner():
    from cocotb.runner import get_runner, get_results

    sim = os.getenv("SIM", "verilator")
    build_dir = THIS_DIR / "build"

    # Two geometries: multi-beat burst and single-beat (BEAT_CNT_W>=1 clamp).
    geometries = [
        {"DATA_W": 128, "AXI_DATA_W": 64},   # BURST_LEN = 2
        {"DATA_W": 128, "AXI_DATA_W": 128},  # BURST_LEN = 1
    ]
    common = {"ADDR_W": 32, "SOURCE_W": 4, "SINK_W": 1, "AXI_ID_W": 4, "AXI_ADDR_W": 32}

    total_fail = 0
    for i, geo in enumerate(geometries):
        params = {**common, **geo}
        work = build_dir / f"test_{i}"
        runner = get_runner(sim)
        runner.build(
            verilog_sources=[str(RTL)],
            hdl_toplevel="tl_to_axi4",
            build_args=["-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL"],
            parameters=params,
            build_dir=str(work),
            always=True,
        )
        results_xml = runner.test(
            hdl_toplevel="tl_to_axi4",
            hdl_toplevel_lang="verilog",
            test_module="tl_to_axi4_tb",
            build_dir=str(work),
            test_dir=str(THIS_DIR),
            results_xml="results.xml",
        )
        num, fail = get_results(Path(results_xml))
        print(f"# geometry {params}: {num-fail}/{num} passed")
        total_fail += fail

    assert total_fail == 0, f"{total_fail} bridge test(s) failed"
    print("# tl_to_axi4: ALL geometries passed")


if __name__ == "__main__":
    test_runner()
