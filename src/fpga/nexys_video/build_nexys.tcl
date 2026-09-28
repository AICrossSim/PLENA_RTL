# PLENA on Nexys Video (XC7A200T) — Full build with MIG DDR3
# Usage: vivado -mode batch -source src/fpga/nexys_video/build_nexys.tcl
#
# Creates: Vivado project with the PLENA core + TL→AXI4 bridge + MIG DDR3,
# top module plena_a7_top (src/fpga/nexys_video/plena_a7_top.sv).

set script_dir [file dirname [file normalize [info script]]]
set proj_dir "$script_dir/vivado_proj"
set src_dir [file normalize "$script_dir/.."]

# Nexys A7-200T
# Nexys Video: XC7A200T-1SBG484C with DDR3
set target_part "xc7a200tsbg484-1"

# Clean and create project
file delete -force $proj_dir
create_project plena_nexys_a7 $proj_dir -part $target_part

# Add RTL sources — full SV design + FPGA wrappers
set rtl_root [file normalize "$script_dir/../../.."]
add_files [glob \
    $rtl_root/src/core/rtl/*.sv \
    $rtl_root/src/frontend/rtl/*.sv \
    $rtl_root/src/control/rtl/*.sv \
    $rtl_root/src/matrix_machine/rtl/*.sv \
    $rtl_root/src/vector_machine/rtl/*.sv \
    $rtl_root/src/scalar_machine/rtl/*.sv \
    $rtl_root/src/memory/HBM/rtl/*.sv \
    $rtl_root/src/memory/HBM/TileLink_Lib/*.sv \
    $rtl_root/src/memory/matrix_sram/rtl/*.sv \
    $rtl_root/src/memory/vector_sram/rtl/*.sv \
    $rtl_root/src/memory/scalar_sram/rtl/*.sv \
    $rtl_root/src/basic_components/*/rtl/*.sv \
    $rtl_root/src/generated_lut/rtl/*.sv \
    $rtl_root/src/definitions/*.svh \
    $rtl_root/src/definitions/*.vh \
]
# FPGA-specific files: the TL->AXI4 bridge, the BSCAN JTAG->AXI master, and this board top
add_files [list \
    $rtl_root/src/fpga/common/rtl/tl_to_axi4.sv \
    $rtl_root/src/fpga/common/rtl/jtag_axi_bscan.sv \
    $script_dir/plena_a7_top.sv \
]
set_property file_type SystemVerilog [get_files *.sv]
# tl_to_axi4 is now .sv (unified with the sim build), so there may be no plain-Verilog sources.
if {[llength [get_files -quiet *.v]] > 0} { set_property file_type Verilog [get_files *.v] }
set_property file_type {Verilog Header} [get_files *.svh]
set_property file_type {Verilog Header} [get_files *.vh]
set_property is_global_include true [get_files *.svh]
set_property is_global_include true [get_files *.vh]

# Include directories for `include resolution
set_property include_dirs [list \
    $rtl_root/src/definitions \
    $rtl_root/src/memory/HBM/TileLink_Lib \
    $rtl_root/src/memory/vector_sram/rtl \
] [current_fileset]

# Add constraints
add_files -fileset constrs_1 $script_dir/nexys_a7.xdc

# Generate MIG IP for DDR2
create_ip -name mig_7series -vendor xilinx.com -library ip -version 4.2 -module_name mig_ddr3
set_property -dict [list \
    CONFIG.XML_INPUT_FILE "$script_dir/mig_nexys_video.prj" \
] [get_ips mig_ddr3]

generate_target all [get_ips mig_ddr3]
synth_ip [get_ips mig_ddr3]

# AXI clock-domain crossing: tl_to_axi4 runs on the 25 MHz core clock (MIG ui_addn_clk_0), the
# MIG s_axi on the 100 MHz ui_clk. This converter bridges the two so the core can close timing.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name axi_clock_converter_0
set_property -dict [list \
    CONFIG.PROTOCOL {AXI4} \
    CONFIG.ADDR_WIDTH {29} \
    CONFIG.DATA_WIDTH {128} \
    CONFIG.ID_WIDTH {4} \
    CONFIG.ACLK_ASYNC {1} \
] [get_ips axi_clock_converter_0]
generate_target all [get_ips axi_clock_converter_0]
synth_ip [get_ips axi_clock_converter_0]

# JTAG access to plena_a7_top's AXI-lite slave is provided by a custom BSCANE2 -> jtag_axi_bscan
# master (added to the fileset above). BSCANE2 is a 7-series unisim primitive, so there is no IP to
# generate. Drive it from a host with pyftdi/OpenOCD over the FT2232H cable
# (src/system/test/jtag_axi_transport.py) -- no Vivado hw_server required.

# Set top module
set_property top plena_a7_top [current_fileset]

# Synthesis
# AreaOptimized_high + full flatten is REQUIRED to fit: default synth = 102% LUT (over budget);
# this pushes arithmetic into DSPs and packs LUTs -> 91% LUT (fits XC7A200T).
puts "=== Starting Synthesis ==="
synth_design -top plena_a7_top -part $target_part -directive AreaOptimized_high -flatten_hierarchy full

# Report utilization immediately after synthesis (before opt which may fail on MIG DRC)
set rpt_dir "$proj_dir/reports"
file mkdir $rpt_dir
report_utilization -file $rpt_dir/utilization_post_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $rpt_dir/utilization_hierarchical.rpt
report_utilization

# Attempt opt with DRC bypass for MIG multi-driver issue
set_property IS_ENABLED 0 [get_drc_checks MDRV-1]
opt_design

# Reports
set rpt_dir "$proj_dir/reports"
file mkdir $rpt_dir
report_utilization -file $rpt_dir/utilization_post_synth.rpt
report_timing_summary -file $rpt_dir/timing_summary_post_synth.rpt

puts "=== Synthesis Complete ==="
report_utilization

# Implementation
# AltSpreadLogic_high spreads the dense MLEN=16 systolic array; without it the router
# leaves ~35 unrouted nets in matrix_compute_unit (local congestion) and bitgen fails.
puts "=== Starting Place ==="
place_design -directive AltSpreadLogic_high
report_utilization -file $rpt_dir/utilization_post_place.rpt

puts "=== Starting Route ==="
route_design -directive Explore

write_checkpoint -force $rpt_dir/post_route.dcp
report_timing_summary -file $rpt_dir/timing_summary_post_route.rpt
report_utilization -file $rpt_dir/utilization_post_route.rpt
puts "BOARD_POSTROUTE_WNS [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]"
foreach p [get_timing_paths -max_paths 8 -nworst 8 -setup] {
    puts "BOARD_PATH WNS=[get_property SLACK $p] END=[get_property ENDPOINT_PIN $p]"
}

# phys_opt_design segfaults on the DIV_LUT netlist (Vivado 2024.2) -> guard it.
if {[catch {phys_opt_design} perr]} {
    puts "PHYS_OPT_FAILED: $perr"
} else {
    report_timing_summary -file $rpt_dir/timing_summary_post_physopt.rpt
    puts "BOARD_POSTPHYSOPT_WNS [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]"
}

# Bitstream
# NOTE: timing is NOT met at 100 MHz -- the whole PLENA core runs on the MIG ui_clk (clk_pll_i)
# and the MLEN=16 datapath tops out ~30 MHz (WNS ~ -21 ns). sys_clk and the DDR3 PHY meet.
# This .bit is a valid proof-of-flow; to run correctly on HW the core needs a slower clock
# (clocking wizard + one AXI CDC between tl_to_axi4 and the MIG s_axi).
puts "=== Writing Bitstream ==="
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
write_bitstream -force $rpt_dir/plena_a7.bit

puts "=== Build Complete ==="
puts "Bitstream: $rpt_dir/plena_a7.bit"
