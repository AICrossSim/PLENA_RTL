# Vivado TCL Script to compile plena.sv
# Usage: vivado -mode batch -source tools/vivado/build_plena.tcl [-tclargs <part> <top>]
#   Defaults: part=xc7a200tfbg676-2  top=plena_artix7_top

## -- Derive repo root from script location --
set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize "$script_dir/../.."]

## -- Configurable parameters (override via -tclargs) --
set target_part [expr {$argc >= 1 ? [lindex $argv 0] : "xc7a200tfbg676-2"}]
set top_module  [expr {$argc >= 2 ? [lindex $argv 1] : "plena_artix7_top"}]
set output_dir  "$repo_root/build_output"

file mkdir $output_dir
set_part $target_part

## -- Read package / header files --
read_verilog -sv $repo_root/src/memory/HBM/TileLink_Lib/prim_util_pkg.svh

puts "Reading source files..."
flush stdout

add_files -scan_for_includes $repo_root/src
set_property file_type {SystemVerilog} [get_files */prim_util_pkg.svh]

## -- Remove files not intended for synthesis (ref, testbench, legacy) --
set exclude_patterns [list \
    "*/ref/*.sv" \
    "*_acy.sv" \
    "*/matrix_machine_v1.sv" \
]
foreach pattern $exclude_patterns {
    set matched [get_files -quiet $pattern]
    if {[llength $matched] > 0} {
        remove_files $matched
        puts "Excluded: $matched"
    }
}

set_property include_dirs [list \
    [file normalize "$repo_root/src/definitions"] \
] [current_fileset]

## -- Suppress SIMULATION define for synthesis (avoids `string` type errors) --
set_property verilog_define {VIVADO_SYNTHESIS} [current_fileset]

## -- Constraints --
read_xdc $repo_root/tools/vivado/time_constraint.xdc

## -- Synthesis --
puts "Setting top module to '$top_module'..."
flush stdout
set_property top $top_module [current_fileset]

puts "Starting Synthesis..."
flush stdout

set_param synth.elaboration.rodinMoreOptions {rt::set_parameter dissolveMemorySizeLimit 196608}

if { [catch { synth_design -top $top_module -part $target_part -keep_equivalent_registers -no_timing_driven } err] } {
    puts "Error during synthesis: $err"
    exit 1
}

puts "Preserving BRAMs..."
flush stdout
set ramb_cells [get_cells -hierarchical -filter {REF_NAME =~ RAMB* || PRIMITIVE_TYPE =~ BMEM.*}]
if {[llength $ramb_cells] > 0} {
    set_property DONT_TOUCH true $ramb_cells
    puts "  Preserved [llength $ramb_cells] BRAM cells"
} else {
    puts "  Warning: No BRAM cells found to preserve"
}

opt_design -directive RuntimeOptimized

## -- Post-Synthesis Reporting and Checkpoint --
write_checkpoint -force $output_dir/post_synth.dcp

report_utilization -file $output_dir/utilization_post_synth.rpt
report_timing_summary -file $output_dir/timing_summary_post_synth.rpt

puts "Synthesis complete. Reports written to $output_dir"
puts ""

## -- Implementation --
set_param drc.disableLUTOverUtilError 1

puts "Starting Place..."
flush stdout
if { [catch { place_design } err] } {
    puts ""
    puts "==================================================================="
    puts "  Place failed: $err"
    puts "  Synthesis + utilization reports are in $output_dir"
    puts "==================================================================="
    puts ""
    exit 0
}
write_checkpoint -force $output_dir/post_place.dcp
report_utilization -file $output_dir/utilization_post_place.rpt

puts "Starting Route..."
flush stdout
if { [catch { route_design -directive AggressiveExplore } err] } {
    puts "Error during route_design: $err"
    exit 1
}

## -- Post-Route Physical Optimization (squeeze route-bound paths) --
puts "Starting phys_opt_design..."
flush stdout
phys_opt_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore
write_checkpoint -force $output_dir/post_route.dcp

## -- Post-Route Reports --
report_timing_summary -file $output_dir/timing_summary_post_route.rpt
report_utilization -file $output_dir/utilization_post_route.rpt
report_utilization -hierarchical -file $output_dir/utilization_hierarchical_post_route.rpt
report_route_status -file $output_dir/route_status.rpt

## -- Bitstream --
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
set_property SEVERITY {Warning} [get_drc_checks LUTLP-1]

puts "Writing bitstream..."
flush stdout
if { [catch { write_bitstream -force $output_dir/plena.bit } err] } {
    puts "Error during write_bitstream: $err"
    exit 1
}

puts "Implementation complete. Results are in $output_dir"
