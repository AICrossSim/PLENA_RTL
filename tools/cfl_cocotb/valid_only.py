import numpy as np

from cocotb.binary import BinaryValue
from cocotb.triggers import *

from .driver import Driver
from .monitor import Monitor


class ValidOnlyDriver(Driver):
    """Driver for valid-only handshake (no ready signal)."""

    def __init__(self, clk, data, valid) -> None:
        super().__init__()
        self.clk = clk
        self.data = data
        self.valid = valid

    async def _driver_send(self, transaction) -> None:
        await RisingEdge(self.clk)
        if type(self.data) == tuple:
            for wire, val in zip(self.data, transaction):
                wire.value = val
        else:
            self.data.value = transaction
        self.valid.value = 1
        self.log.debug("Sent %s" % (transaction,))

        if self.send_queue.empty():
            await RisingEdge(self.clk)
            self.valid.value = 0


class ValidOnlyMonitor(Monitor):
    """Monitor for valid-only handshake (no ready signal)."""

    def __init__(self, clk, data, valid, check=True, name=None, unsigned=False):
        super().__init__(clk, check=check, name=name)
        self.data = data
        self.valid = valid
        self.unsigned = unsigned

    def _trigger(self):
        if "x" in self.valid.value.binstr:
            return False
        return self.valid.value == 1

    def _recv(self):
        def _get_sig_value(sig):
            if type(sig.value) == list:
                if self.unsigned:
                    return [x.integer for x in sig.value]
                else:
                    return [x.signed_integer for x in sig.value]
            elif type(sig.value) == BinaryValue:
                if self.unsigned:
                    return int(sig.value.integer)
                else:
                    return int(sig.value.signed_integer)

        if type(self.data) == tuple:
            data = tuple(_get_sig_value(s) for s in self.data)
        else:
            data = _get_sig_value(self.data)
        return data

    def _check(self, got, exp):
        def _check_sig(got, exp):
            if not np.equal(got, exp).all():
                self.log.error(
                    "%s: \nGot \n%s, \nExpected \n%s"
                    % (
                        self.name if self.name is not None else "Unnamed ValidOnlyMonitor",
                        got,
                        exp,
                    )
                )
                assert False, "Test Failed!"
            else:
                self.log.debug(
                    "Passed | %s: \nGot \n%s, \nExpected \n%s"
                    % (
                        self.name if self.name is not None else "Unnamed ValidOnlyMonitor",
                        got,
                        exp,
                    )
                )

        if self.check:
            if type(self.data) == tuple:
                assert type(got) == tuple
                assert type(exp) == tuple
                assert len(got) == len(exp), "Got & Exp Tuples are different length"
                for g, e in zip(got, exp):
                    _check_sig(g, e)
            else:
                _check_sig(got, exp)
