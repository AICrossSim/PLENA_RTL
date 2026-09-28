"""Host-side JTAG transport for the real board (Vivado-free), for PlenaDriver.

Drives the custom jtag_axi_bscan master over the FT2232H JTAG cable using pyftdi -- no Vivado, no
hw_server, no Xilinx jtag_axi IP. Speaks the SAME 40-bit DR protocol validated in simulation by
JtagBscanTransport / jtag_axi_bscan_tb.py, so a sequence that passes in sim runs here unchanged:

    from pyftdi... import ...            # (installed pyftdi)
    from plena_driver import PlenaDriver
    from jtag_axi_transport import JtagAxiTransport
    drv = PlenaDriver(JtagAxiTransport())
    await? -- NOTE: this transport is SYNCHRONOUS (pyftdi is blocking). PlenaDriver is async; run it
    under a trivial event loop or use the provided SyncPlenaDriver-style calls. See __main__ demo.

HARDWARE-ONLY: this file cannot be exercised in this repo's Verilator/cocotb sim. Two things MUST be
confirmed on the bench at bring-up (flagged inline):
  * the USER instruction opcode / IR length against the XC7A200T BSDL (default USER4=0x23, IR_LEN=6),
  * that direct libusb/pyftdi access to the FT2232H does not collide with Digilent Adept (close Adept
    / Vivado hw_server first). OpenOCD's digilent_nexys_video.cfg (VID:PID 0403:6010, JTAG on
    interface B) is the reference that the direct path works.

DR format (must match jtag_axi_bscan.sv exactly, LSB-first on the wire):
  in : [31:0]=wdata, [38:32]=byte addr (NOT >>2), [39]=rw (1=wr,0=rd)
  out: [31:0]=rdata (previous txn), [38:32]=addr echo, [39]=done
"""
from plena_driver import PlenaTransport

# ---- protocol constants (shared with jtag_axi_bscan.sv) ----
CMD_BITS  = 40
DATA_BITS = 32
ADDR_BITS = 7
RW_SHIFT   = 39
ADDR_SHIFT = 32
DONE_BIT   = 39

# ---- JTAG / device constants (CONFIRM against BSDL at bring-up) ----
IR_LEN     = 6
USER1, USER2, USER3, USER4 = 0x02, 0x03, 0x22, 0x23     # 7-series UG470 Table 10-2
DEFAULT_USER = USER4                                     # jtag_axi_bscan wrapper uses BSCANE2 JTAG_CHAIN=4
XC7A200T_IDCODE = 0x13636093
IDCODE_VERSION_MASK = 0x0FFFFFFF                          # ignore the top (version) nibble
DEFAULT_URL = "ftdi://ftdi:2232h/2"                      # FT2232H interface B = JTAG on Nexys Video


class JtagAxiTransport(PlenaTransport):
    """pyftdi JtagEngine backend. Synchronous (pyftdi is blocking); the async methods below simply
    do not await anything, so they are awaitable no-op-coroutines usable by PlenaDriver."""

    def __init__(self, url=DEFAULT_URL, user_instr=DEFAULT_USER, frequency=1e6, log=None):
        from pyftdi.jtag import JtagEngine
        from pyftdi.bits import BitSequence
        self._BitSequence = BitSequence
        self.user_instr = user_instr
        self.log = log
        self.jtag = JtagEngine(trst=False, frequency=frequency)
        self.jtag.configure(url)
        self.jtag.reset()                                # -> Test-Logic-Reset
        self._check_idcode()
        self._select_user()

    def _log(self, m):
        if self.log:
            self.log.info(f"[JtagAxi] {m}")

    def _check_idcode(self):
        self.jtag.reset()
        self.jtag.change_state("shift_dr")
        idcode = int(self.jtag.read_dr(32))
        self.jtag.go_idle()
        if (idcode & IDCODE_VERSION_MASK) != (XC7A200T_IDCODE & IDCODE_VERSION_MASK):
            self._log(f"WARNING: IDCODE {idcode:#010x} != XC7A200T {XC7A200T_IDCODE:#010x}")
        else:
            self._log(f"IDCODE ok {idcode:#010x}")

    def _select_user(self):
        # Load USERx into IR once; BSCANE2 SEL stays asserted for all subsequent DR shifts.
        self.jtag.write_ir(self._BitSequence(value=self.user_instr, length=IR_LEN))
        self.jtag.go_idle()
        self._log(f"IR <- USER {self.user_instr:#04x}")

    def _scan(self, cmd):
        """One Capture/Shift(40)/Update DR pass. Returns the 40-bit shift-out (previous txn)."""
        self.jtag.change_state("shift_dr")
        captured = int(self.jtag.shift_register(self._BitSequence(value=cmd, length=CMD_BITS)))
        self.jtag.go_idle()
        return captured

    # ---- PlenaTransport API (awaitable but synchronous under the hood) ----
    async def axi_write(self, byte_addr, data):
        self._scan((1 << RW_SHIFT) | ((byte_addr & 0x7F) << ADDR_SHIFT) | (data & 0xFFFFFFFF))

    async def axi_read(self, byte_addr):
        self._scan((0 << RW_SHIFT) | ((byte_addr & 0x7F) << ADDR_SHIFT))   # issue read
        out = self._scan(0)                                                # flush -> prev read result
        # done = (out >> DONE_BIT) & 1  # JTAG is ms-slow so the txn is always complete by the flush
        return out & 0xFFFFFFFF

    async def idle(self, n=1):
        # No core clock control from JTAG; a run-test/idle dwell advances time on the board.
        for _ in range(max(1, n // 8)):
            self.jtag.go_idle()


if __name__ == "__main__":
    # Bench smoke test (needs the board + pyftdi). Loads a program and activates the core, mirroring
    # the sim flow. Run PlenaDriver's async methods under a trivial loop since this transport is sync.
    import asyncio, sys
    from plena_driver import PlenaDriver

    def load(path):
        return [int(l.strip(), 16) for l in open(path) if l.strip() and not l.startswith("//")]

    async def main(prog, start_word):
        drv = PlenaDriver(JtagAxiTransport(), dbg_words=6)
        await drv.wait_calib()
        await drv.load_program(load(prog), start_word=start_word)
        await drv.launch()
        ok = await drv.wait_done()
        print("system_break reached" if ok else "TIMEOUT")

    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
