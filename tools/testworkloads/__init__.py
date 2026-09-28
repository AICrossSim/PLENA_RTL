"""PLENA test workload generators.

One module per workload; each defines a ``<Name>Workload(WorkloadGenerator)``
class and a CLI (``python -m tools.testworkloads.<name> --help``). Run them
through ``just rtl-sim <name> [rebuild] [--args]``.

Layers:      linear, rms_norm, silu, linear_silu, silu_down, softmax, rope,
             mha, gqa, attention, ffn, llama_layer (full decoder layer)
Infra tests: prefetch, loop, scratchpad
Suite:       test_suite (cases for ``just rtl-suite``)

Only the base class is imported eagerly; import a generator module explicitly,
e.g. ``from tools.testworkloads.linear import LinearWorkload``.
"""

from .base import WorkloadGenerator

__all__ = ["WorkloadGenerator"]
