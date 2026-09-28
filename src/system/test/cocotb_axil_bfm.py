"""cocotb AXI4-Lite master BFM implementing PlenaTransport against SimTopA7's exposed
`la_*` slave port. Matches the SimTopA7 register-block handshake exactly:
  write: awready = wready = (awvalid & wvalid & ~bvalid_r)   (combinational, single cycle)
         bvalid_r set the cycle after wr_fire; cleared when bready & bvalid
  read:  arready = ~rvalid_r ; rd_fire = arvalid & arready
         rvalid_r set (and rdata latched) the cycle after rd_fire; cleared when rready & rvalid
Combinational readies are sampled in the ReadOnly phase so the single-cycle assertion
is never missed.
"""
import cocotb
from cocotb.triggers import RisingEdge, ReadOnly

from plena_driver import PlenaTransport


class CocotbAxiLiteBFM(PlenaTransport):
    def __init__(self, dut, clk, prefix="la_"):
        self.dut = dut
        self.clk = clk
        self.p = prefix
        # Initialise all master-driven inputs to idle.
        self._s("awaddr").value = 0
        self._s("awvalid").value = 0
        self._s("wdata").value = 0
        self._s("wstrb").value = 0
        self._s("wvalid").value = 0
        self._s("bready").value = 0
        self._s("araddr").value = 0
        self._s("arvalid").value = 0
        self._s("rready").value = 0

    def _s(self, name):
        return getattr(self.dut, self.p + name)

    async def idle(self, n=1):
        for _ in range(n):
            await RisingEdge(self.clk)

    async def axi_write(self, byte_addr, data):
        await RisingEdge(self.clk)
        self._s("awaddr").value = byte_addr
        self._s("awvalid").value = 1
        self._s("wdata").value = data & 0xFFFFFFFF
        self._s("wstrb").value = 0xF
        self._s("wvalid").value = 1
        self._s("bready").value = 1
        # Wait for the combinational accept (awready & wready) in the ReadOnly phase.
        while True:
            await ReadOnly()
            if int(self._s("awready").value) == 1 and int(self._s("wready").value) == 1:
                break
            await RisingEdge(self.clk)
        # Accept happens on the coming edge; drop AW/W after it.
        await RisingEdge(self.clk)
        self._s("awvalid").value = 0
        self._s("wvalid").value = 0
        # Wait for the write response.
        while True:
            await ReadOnly()
            if int(self._s("bvalid").value) == 1:
                break
            await RisingEdge(self.clk)
        await RisingEdge(self.clk)
        self._s("bready").value = 0

    async def axi_read(self, byte_addr):
        await RisingEdge(self.clk)
        self._s("araddr").value = byte_addr
        self._s("arvalid").value = 1
        self._s("rready").value = 1
        # Wait for the address accept (arready = ~rvalid_r).
        while True:
            await ReadOnly()
            if int(self._s("arready").value) == 1:
                break
            await RisingEdge(self.clk)
        await RisingEdge(self.clk)
        self._s("arvalid").value = 0
        # Wait for read data.
        while True:
            await ReadOnly()
            if int(self._s("rvalid").value) == 1:
                data = int(self._s("rdata").value)
                break
            await RisingEdge(self.clk)
        await RisingEdge(self.clk)
        self._s("rready").value = 0
        return data
