"""Transport-agnostic host driver for activating the PLENA core over its AXI-lite
control port (the same port the FPGA exposes via the Xilinx JTAG-to-AXI master).

The driver speaks ONLY the AXI-lite register protocol through an abstract
`PlenaTransport` (axi_write/axi_read). Two backends implement that transport:
  - CocotbAxiLiteBFM (cocotb_axil_bfm.py) -- drives SimTopA7's exposed la_* port in sim.
  - a real JTAG-to-AXI backend (hw_server/run_hw_axi, or a custom BSCAN master) -- HW.
The activation logic here is identical for both, so a sim-validated sequence runs on
the board unchanged by swapping the transport.

Register map (word addr = byte_addr[7:2]):
  0x00 W IMEM_DATA  push one 32-bit instruction word, wptr++
  0x04 R STATUS     {..., init_calib_complete[1], system_break[0]}
  0x08 W DEBUG_CTRL bit0 -> debug_vsram_en
  0x0C W DEBUG_ADDR -> debug_row
  0x10 RW IMEM_WPTR set/read the IMEM write pointer
  0x14 W SOFT_RST   reset wptr=0 AND pulse a 16-cycle core reset -> run from PC0
  0x18 R RST_CNT    cycles the core was held in reset
  0x40.. R DEBUG_Dk VSRAM readback words (debug_vsram_data, DBG_WORDS of them)
"""

REG_IMEM_DATA  = 0x00
REG_STATUS     = 0x04
REG_DEBUG_CTRL = 0x08
REG_DEBUG_ADDR = 0x0C
REG_IMEM_WPTR  = 0x10
REG_SOFT_RST   = 0x14
REG_RST_CNT    = 0x18
REG_DEBUG_D0   = 0x40
# JTAG->DDR3 loader window (0x1C..0x3C)
REG_DDR3_ADDR   = 0x1C
REG_DDR3_W0     = 0x20   # W0/W1/W2/W3 at 0x20/0x24/0x28/0x2C; same words read back R0..R3
REG_DDR3_GO     = 0x30
REG_DDR3_STATUS = 0x34
REG_LOAD_MODE   = 0x38
REG_DDR3_RD     = 0x3C

STATUS_SYSTEM_BREAK = 1 << 0
STATUS_INIT_CALIB   = 1 << 1
DDR3_BUSY           = 1 << 0
DDR3_BEAT_BYTES     = 16   # BRIDGE_DATA_W(128)/8


class PlenaTransport:
    """Abstract AXI-lite transport. Each backend advances its own notion of time."""
    async def axi_write(self, byte_addr: int, data: int) -> None:
        raise NotImplementedError

    async def axi_read(self, byte_addr: int) -> int:
        raise NotImplementedError

    async def idle(self, n: int = 1) -> None:
        """Advance n cycles without a transaction (sim backends override)."""
        return None


class PlenaDriver:
    def __init__(self, transport: PlenaTransport, dbg_words: int = 6, log=None):
        self.t = transport
        self.dbg_words = dbg_words
        self.log = log

    def _log(self, msg):
        if self.log:
            self.log.info(f"[PlenaDriver] {msg}")

    async def wait_calib(self, timeout_polls: int = 4000) -> None:
        for _ in range(timeout_polls):
            s = await self.t.axi_read(REG_STATUS)
            if s & STATUS_INIT_CALIB:
                self._log("DDR3 init_calib_complete=1")
                return
        raise TimeoutError("init_calib_complete never asserted")

    async def load_program(self, words, start_word: int = 0) -> None:
        """Load 32-bit instruction words into IMEM starting at IMEM word `start_word`
        (= INSTRUCTION_STORAGE_OFFSET>>2, since fetch_addr = pc + offset and
        bram_addr = addr[2 +: W]). wptr auto-increments on each IMEM_DATA write."""
        await self.t.axi_write(REG_IMEM_WPTR, start_word)
        for w in words:
            await self.t.axi_write(REG_IMEM_DATA, w & 0xFFFFFFFF)
        wptr = await self.t.axi_read(REG_IMEM_WPTR)
        exp = start_word + len(words)
        assert wptr == exp, f"IMEM_WPTR after load = {wptr}, expected {exp}"
        self._log(f"loaded {len(words)} words at IMEM[{start_word}..{exp-1}], wptr={wptr}")

    async def launch(self) -> None:
        """SOFT_RST: resets wptr=0 and pulses the 16-cycle core reset -> run from PC0."""
        await self.t.axi_write(REG_SOFT_RST, 0)
        self._log("SOFT_RST issued (core launched)")

    async def wait_done(self, timeout_polls: int = 200000) -> bool:
        await self.t.idle(32)  # skip the 16-cycle reset window before polling
        for _ in range(timeout_polls):
            s = await self.t.axi_read(REG_STATUS)
            if s & STATUS_SYSTEM_BREAK:
                self._log("system_break=1 (C_BREAK reached)")
                return True
        self._log("TIMEOUT waiting for system_break")
        return False

    async def read_vsram_row(self, row: int) -> int:
        """Read one VSRAM row via the debug port. Returns the row as an integer of
        DBG_WORDS*32 bits (low word first)."""
        await self.t.axi_write(REG_DEBUG_ADDR, row)
        await self.t.axi_write(REG_DEBUG_CTRL, 1)
        await self.t.idle(4)  # let the VSRAM read settle into dbg_cap
        val = 0
        for k in range(self.dbg_words):
            w = await self.t.axi_read(REG_DEBUG_D0 + 4 * k)
            val |= (w & 0xFFFFFFFF) << (32 * k)
        await self.t.axi_write(REG_DEBUG_CTRL, 0)
        return val

    async def read_rst_cnt(self) -> int:
        return await self.t.axi_read(REG_RST_CNT)

    # ---- JTAG->DDR3 loader ---------------------------------------------------
    async def set_load_mode(self, on: bool) -> None:
        await self.t.axi_write(REG_LOAD_MODE, 1 if on else 0)

    async def _ddr3_beat(self, beat128: int) -> None:
        """Write one 128-bit beat (W0..W3) + GO at the current auto-incrementing DDR3_ADDR."""
        for k in range(4):
            await self.t.axi_write(REG_DDR3_W0 + 4 * k, (beat128 >> (32 * k)) & 0xFFFFFFFF)
        await self.t.axi_write(REG_DDR3_GO, 0)

    async def load_ddr3(self, base_addr: int, data: bytes, poll_each: bool = False) -> None:
        """Stream `data` into DDR3 starting at byte `base_addr`, one 16-byte little-endian beat at a
        time. Fire-and-forget by default (axi_write's own latency covers the write); set poll_each to
        wait on DDR3_STATUS.busy per beat. Pads the last partial beat with zeros."""
        await self.set_load_mode(True)
        await self.t.axi_write(REG_DDR3_ADDR, base_addr)
        n = (len(data) + DDR3_BEAT_BYTES - 1) // DDR3_BEAT_BYTES
        for i in range(n):
            chunk = data[i * DDR3_BEAT_BYTES:(i + 1) * DDR3_BEAT_BYTES].ljust(DDR3_BEAT_BYTES, b"\x00")
            await self._ddr3_beat(int.from_bytes(chunk, "little"))
            if poll_each:
                while (await self.t.axi_read(REG_DDR3_STATUS)) & DDR3_BUSY:
                    pass
        while (await self.t.axi_read(REG_DDR3_STATUS)) & DDR3_BUSY:  # drain the last beat
            pass
        await self.set_load_mode(False)
        self._log(f"loaded {len(data)} bytes into DDR3 @ {base_addr:#x} ({n} beats)")

    async def read_ddr3_beat(self, byte_addr: int) -> int:
        """Read one 128-bit beat back from DDR3 at `byte_addr` (for verification)."""
        await self.set_load_mode(True)
        await self.t.axi_write(REG_DDR3_ADDR, byte_addr)
        await self.t.axi_write(REG_DDR3_RD, 0)
        while (await self.t.axi_read(REG_DDR3_STATUS)) & DDR3_BUSY:
            pass
        val = 0
        for k in range(4):
            val |= (await self.t.axi_read(REG_DDR3_W0 + 4 * k)) << (32 * k)
        await self.set_load_mode(False)
        return val
