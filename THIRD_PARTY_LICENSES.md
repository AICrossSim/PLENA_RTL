# Third-party code in this repository

The PLENA RTL and tooling are released under the Apache License 2.0 (see `LICENSE`). The
following vendored components keep their own licenses and copyright notices,
which must be preserved in redistributions.

## lowRISC / OpenTitan primitives — Apache License 2.0

Copyright lowRISC contributors (OpenTitan project).
Licensed under the Apache License, Version 2.0. Each file carries an
`SPDX-License-Identifier: Apache-2.0` header; the license text is at
<https://www.apache.org/licenses/LICENSE-2.0>.

- `src/memory/vector_sram/rtl/prim_assert.sv`
- `src/memory/vector_sram/rtl/prim_assert_dummy_macros.svh`
- `src/memory/vector_sram/rtl/prim_assert_sec_cm.svh`
- `src/memory/vector_sram/rtl/prim_flop_macros.sv`
- `src/memory/vector_sram/rtl/prim_generic_ram_2p.sv`
- `src/memory/vector_sram/rtl/prim_ram_2p_pkg.svh`
- `src/memory/vector_sram/rtl/prim_util_memload.svh`
- `src/memory/scalar_sram/rtl/prim_generic_ram_1p.sv`
- `src/memory/scalar_sram/rtl/prim_ram_1p_pkg.svh`
- `src/memory/HBM/TileLink_Lib/prim_count.sv`
- `src/memory/HBM/TileLink_Lib/prim_count_pkg.sv`
- `src/memory/HBM/TileLink_Lib/prim_fifo_sync.sv`
- `src/memory/HBM/TileLink_Lib/prim_fifo_sync_cnt.sv`
- `src/memory/HBM/TileLink_Lib/prim_util_pkg.svh`

## TileLink library — Gary Guo, BSD 2-Clause

Copyright (c) 2018, Gary Guo. All rights reserved. The full BSD 2-Clause text
is in the header of `src/memory/HBM/TileLink_Lib/openip_regslice.sv`; it applies
to that file and to the TileLink adapter/utility modules in the same directory
(`tl_pkg.svh`, `tl_util.svh`, `tl_adapter*.sv`, `tl_burst_tracker.sv`,
`tl_data_*sizer.sv`, `tl_fifo_*.sv`, `tl_regslice.sv`, `tl_sink_upsizer.sv`,
`tl_size_downsizer.sv`, `tl_socket_*.sv`, `tl_source_*.sv`, `tl_*_terminator.sv`,
`tl_broadcast.sv`, `tl_error_sink.sv`), which derive from the same library.
`tl_selector.sv` is PLENA-authored.

## Synopsys DesignWare (not shipped)

`src/basic_components/synopsis_ip_inst/rtl/DW_fp_*_inst.sv` only instantiate
Synopsys DesignWare `DW_fp_*` components under `` `ifdef DC_LIB_EN ``. No
DesignWare source is included; a Synopsys license is required to use that path.

## Python dependencies

PyTorch, cocotb, NumPy and the other packages listed in `setup.py` are
installed from PyPI under their own licenses and are not redistributed here.
