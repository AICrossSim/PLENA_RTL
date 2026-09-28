#------------------------------
# Parameterized Synthesis Script
# Usage: dc_shell -f synth.tcl
# Environment variables:
#   SYNTH_MODULE       - Module name to synthesize (required)
#   SYNTH_CLK_PERIOD   - Clock period in ps (default: 1000)
#   SYNTH_COMPILE_MODE - "normal" or "ultra" (default: normal)
#   SYNTH_BUILD_DIR    - Output directory (required)
#------------------------------

set WORK_DIR "./"

#------------------------------
# Get parameters from environment
#------------------------------
if {[info exists env(SYNTH_MODULE)]} {
    set MODULE $env(SYNTH_MODULE)
} else {
    puts "ERROR: SYNTH_MODULE not defined"
    exit 1
}

if {[info exists env(SYNTH_BUILD_DIR)]} {
    set BUILD_DIR $env(SYNTH_BUILD_DIR)
} else {
    puts "ERROR: SYNTH_BUILD_DIR not defined"
    exit 1
}

if {[info exists env(SYNTH_CLK_PERIOD)]} {
    set CLK_PERIOD $env(SYNTH_CLK_PERIOD)
} else {
    set CLK_PERIOD 1000
}

if {[info exists env(SYNTH_COMPILE_MODE)]} {
    set COMPILE_MODE $env(SYNTH_COMPILE_MODE)
} else {
    set COMPILE_MODE "normal"
}

# Set top_design for compatibility with read_rtl.tcl
set top_design $MODULE

puts "=========================================="
puts "Synthesis Configuration"
puts "  Module:       $MODULE"
puts "  Clock Period: $CLK_PERIOD ps"
puts "  Compile Mode: $COMPILE_MODE"
puts "  Build Dir:    $BUILD_DIR"
puts "=========================================="

#------------------------------
# Frequency calculation
#------------------------------
proc cal_freq {clock_period} {
    return [expr 1000.0/$clock_period]
}

#------------------------------
# Set paths - use BUILD_DIR for outputs
#------------------------------
set src ${WORK_DIR}/../../src
set log_dir ${BUILD_DIR}/logs
set rpt_dir ${BUILD_DIR}/reports
set net_dir ${BUILD_DIR}/netlist
set out_dir ${BUILD_DIR}/out

# Directories should already exist from justfile, but ensure they do
file mkdir ${log_dir}
file mkdir ${rpt_dir}
file mkdir ${net_dir}
file mkdir ${out_dir}

#------------------------------
# Limit verbose messages
#------------------------------
set_message_info -id ELAB-405 -limit 10

#------------------------------
# Set source file path
#------------------------------
lappend search_path ${src}

#------------------------------
# Clock settings
#------------------------------
set top_clk_name "clk"
set reset "rst"
set clk_period $CLK_PERIOD

#------------------------------
# Read RTL files
#------------------------------
puts "\n>>> Reading RTL files..."
source ${WORK_DIR}/read_rtl.tcl

#------------------------------
# Check design (pre-synthesis)
#------------------------------
puts "\n>>> Pre-synthesis design check..."
check_design
check_design > ${log_dir}/${MODULE}_pre_check.log

#------------------------------
# Set current design and link
#------------------------------
puts "\n>>> Linking design..."
current_design ${MODULE}
link
uniquify

#------------------------------
# Apply constraints
#------------------------------
puts "\n>>> Applying constraints..."
create_clock -period $clk_period $top_clk_name
set_drive 0 $top_clk_name
set_clock_uncertainty -setup [expr 0.1*$clk_period] $top_clk_name
set auto_wire_load_selection true

set ALL_INS_EX_CLK [remove_from_collection [all_inputs] [get_ports $top_clk_name]]
set_input_delay [expr 0.08*$clk_period] -clock $top_clk_name [all_inputs]
set_output_delay [expr 0.05*$clk_period] -clock $top_clk_name [all_outputs]

set_max_delay $clk_period -to [all_outputs]
set_max_delay $clk_period -to [all_registers -data_pins]

#------------------------------
# Save unmapped design
#------------------------------
puts "\n>>> Saving unmapped design..."
write_file -f verilog -hierarchy -output ${net_dir}/${MODULE}_unmapped.v
write_file -f ddc     -hierarchy -output ${net_dir}/${MODULE}_unmapped.ddc

#------------------------------
# Compile
#------------------------------
puts "\n>>> Compiling design (mode: $COMPILE_MODE)..."
if {$COMPILE_MODE eq "ultra"} {
    puts "Using compile_ultra with retiming..."
    optimize_registers
    compile_ultra -retime
} else {
    puts "Using standard compile..."
    compile
}

#------------------------------
# Post-compile processing
#------------------------------
puts "\n>>> Post-compile processing..."
change_names -rules verilog -verbose -hier

#------------------------------
# Check timing and design
#------------------------------
puts "\n>>> Post-synthesis checks..."
check_timing > ${log_dir}/${MODULE}_timing_check.log
check_design > ${log_dir}/${MODULE}_post_check.log

#------------------------------
# Save mapped design
#------------------------------
puts "\n>>> Saving mapped design..."
write_file -f verilog -hierarchy -output ${out_dir}/${MODULE}_mapped.v
write_file -f ddc     -hierarchy -output ${out_dir}/${MODULE}_mapped.ddc
write_sdf ${out_dir}/${MODULE}.sdf
write_sdc ${out_dir}/${MODULE}.sdc
write_script -format dctcl -hierarchy -output ${net_dir}/${MODULE}.tcl

#------------------------------
# Generate reports
#------------------------------
puts "\n>>> Generating reports..."

# Units report
report_units > ${rpt_dir}/${MODULE}_units.rpt

# Clock report
report_clock -skew > ${rpt_dir}/${MODULE}_clock.rpt
set freq [cal_freq $clk_period]
set fp [open ${rpt_dir}/${MODULE}_clock.rpt a]
puts $fp ""
puts $fp "Target Frequency: $freq GHz"
puts $fp "Clock Period: $clk_period ps"
close $fp

# Area report
report_area -hierarchy > ${rpt_dir}/${MODULE}_area.rpt

# Timing reports
report_timing > ${rpt_dir}/${MODULE}_timing.rpt
report_timing -delay_type max -max_paths 50 -nworst 1 -significant_digits 3 -sort_by slack > ${rpt_dir}/${MODULE}_timing_slack.rpt

# Constraints report
report_constraints -all_violators > ${rpt_dir}/${MODULE}_constraints.rpt

# Reference report
report_reference -hierarchy > ${rpt_dir}/${MODULE}_reference.rpt

# Power report
report_power -hierarchy > ${rpt_dir}/${MODULE}_power.rpt

# Port report
report_port > ${rpt_dir}/${MODULE}_port.rpt

# QoR (Quality of Results) report
report_qor > ${rpt_dir}/${MODULE}_qor.rpt

# Library report
report_lib typical > ${rpt_dir}/${MODULE}_lib.rpt

#------------------------------
# Final summary
#------------------------------
puts ""
puts "=========================================="
puts "Synthesis Complete: $MODULE"
puts "=========================================="
puts "Output Directory: $BUILD_DIR"
puts ""
puts "Logs:"
puts "  ${log_dir}/${MODULE}_pre_check.log"
puts "  ${log_dir}/${MODULE}_post_check.log"
puts "  ${log_dir}/${MODULE}_timing_check.log"
puts ""
puts "Reports:"
puts "  ${rpt_dir}/${MODULE}_area.rpt"
puts "  ${rpt_dir}/${MODULE}_power.rpt"
puts "  ${rpt_dir}/${MODULE}_timing.rpt"
puts "  ${rpt_dir}/${MODULE}_qor.rpt"
puts ""
puts "Netlists:"
puts "  ${out_dir}/${MODULE}_mapped.v"
puts "  ${out_dir}/${MODULE}.sdf"
puts "  ${out_dir}/${MODULE}.sdc"
puts "=========================================="

# Print key metrics to stdout
puts ""
puts ">>> Quick Summary:"
report_area
puts ""
report_timing

exit
