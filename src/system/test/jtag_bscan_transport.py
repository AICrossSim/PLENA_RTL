"""cocotb JTAG-BSCAN transport backend for PlenaDriver.

Drives the jtag_axi_bscan tap ports (sel/capture/shift/update/tdi, samples tdo) to run the
40-bit DR scan protocol, implementing PlenaTransport.axi_write/axi_read. This lets the UNCHANGED
PlenaDriver run its activation sequence through the BSCAN JTAG path in simulation -- the same path
the real board uses. The host-side pyftdi driver (jtag_axi_transport.py) speaks the identical DR
format, so what passes here is what runs on hardware.

DR format (40 bits, LSB-first):
  in : [31:0]=wdata, [38:32]=byte addr (NOT >>2; the slave slices [7:2]), [39]=rw (1=wr,0=rd)
  out: [31:0]=rdata (prev txn), [38:32]=addr echo, [39]=done
A scan launches txn N and shifts out txn N-1's result; a read = issue-scan + flush-scan.
"""
from cocotb.triggers import RisingEdge, ReadOnly, ClockCycles

from plena_driver import PlenaTransport

DR_LEN = 40


async def bscan_scan(dut, cmd, tck):
    """One Capture-DR -> Shift-DR(40, LSB-first) -> Update-DR scan. Returns the 40-bit shift-out
    (the previous transaction's response). Mirrors jtag_axi_bscan's tck-domain FSM exactly."""
    dut.sel.value = 1
    dut.capture.value = 1
    dut.shift.value = 0
    dut.update.value = 0
    await RisingEdge(tck)                 # Capture-DR: sh_reg <= resp_holding
    dut.capture.value = 0
    dut.shift.value = 1
    out = 0
    for i in range(DR_LEN):
        dut.tdi.value = (cmd >> i) & 1
        await ReadOnly()                  # tdo = sh_reg[0] before this shift edge
        out |= int(dut.tdo.value) << i
        await RisingEdge(tck)             # Shift-DR: sh_reg <= {tdi, sh_reg[39:1]}
    dut.shift.value = 0
    dut.update.value = 1
    await RisingEdge(tck)                 # Update-DR: cmd_reg <= sh_reg, req toggle
    dut.update.value = 0
    dut.sel.value = 0
    return out


class JtagBscanTransport(PlenaTransport):
    def __init__(self, dut, tck, aclk, axi_wait=64):
        self.dut = dut
        self.tck = tck
        self.aclk = aclk
        self.axi_wait = axi_wait      # aclk cycles to let the AXI txn complete after a scan
        dut.sel.value = 0
        dut.capture.value = 0
        dut.shift.value = 0
        dut.update.value = 0
        dut.tdi.value = 0

    async def _scan(self, cmd):
        out = await bscan_scan(self.dut, cmd, self.tck)
        await ClockCycles(self.aclk, self.axi_wait)
        return out

    async def axi_write(self, byte_addr, data):
        await self._scan((1 << 39) | ((byte_addr & 0x7F) << 32) | (data & 0xFFFFFFFF))

    async def axi_read(self, byte_addr):
        await self._scan((0 << 39) | ((byte_addr & 0x7F) << 32))  # issue read
        out = await self._scan(0)                                 # flush -> prev read's {done,rdata}
        return out & 0xFFFFFFFF

    async def idle(self, n=1):
        await ClockCycles(self.aclk, n)
