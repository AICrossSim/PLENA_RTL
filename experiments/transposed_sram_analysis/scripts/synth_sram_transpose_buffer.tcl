#------------------------------
# Synthesis Script for sram_with_transpose_buffer
# Usage: dc_shell -f synth_sram_transpose_buffer.tcl
#------------------------------

set WORK_DIR [pwd]
set MODULE "sram_with_transpose_buffer"
set BUILD_DIR "${WORK_DIR}/../results/sram_transpose_buffer"
set CLK_PERIOD 1000  ;# 1ns clock period (1GHz)

puts "=========================================="
puts "Synthesizing: $MODULE"
puts "Clock Period: $CLK_PERIOD ps"
puts "Build Dir: $BUILD_DIR"
puts "=========================================="

#------------------------------
# Set paths
#------------------------------
set exp_rtl "${WORK_DIR}/../rtl"
set log_dir "${BUILD_DIR}/logs"
set rpt_dir "${BUILD_DIR}/reports"
set net_dir "${BUILD_DIR}/netlist"
set out_dir "${BUILD_DIR}/out"

# Create directories
file mkdir ${log_dir}
file mkdir ${rpt_dir}
file mkdir ${net_dir}
file mkdir ${out_dir}

#------------------------------
# Set search paths
#------------------------------
lappend search_path ${exp_rtl}

#------------------------------
# Limit verbose messages
#------------------------------
set_message_info -id ELAB-405 -limit 10

#------------------------------
# Analyze files
#------------------------------
puts "\n>>> Analyzing RTL files..."

# All required modules
analyze -f sverilog -lib work "${exp_rtl}/pure_sram_32x32.sv"
analyze -f sverilog -lib work "${exp_rtl}/transpose_buffer_32x32.sv"
analyze -f sverilog -lib work "${exp_rtl}/sram_with_transpose_buffer.sv"

#------------------------------
# Elaborate
#------------------------------
puts "\n>>> Elaborating design..."
elaborate ${MODULE}

#------------------------------
# Check design
#------------------------------
puts "\n>>> Pre-synthesis design check..."
check_design > ${log_dir}/${MODULE}_pre_check.log

#------------------------------
# Link design
#------------------------------
puts "\n>>> Linking design..."
current_design ${MODULE}
link
uniquify

#------------------------------
# Apply constraints
#------------------------------
puts "\n>>> Applying constraints..."
create_clock -period $CLK_PERIOD clk
set_drive 0 clk
set_clock_uncertainty -setup [expr 0.1*$CLK_PERIOD] clk
set auto_wire_load_selection true

set_input_delay [expr 0.08*$CLK_PERIOD] -clock clk [all_inputs]
set_output_delay [expr 0.05*$CLK_PERIOD] -clock clk [all_outputs]
set_max_delay $CLK_PERIOD -to [all_outputs]
set_max_delay $CLK_PERIOD -to [all_registers -data_pins]

#------------------------------
# Save unmapped design
#------------------------------
puts "\n>>> Saving unmapped design..."
write_file -f verilog -hierarchy -output ${net_dir}/${MODULE}_unmapped.v
write_file -f ddc     -hierarchy -output ${net_dir}/${MODULE}_unmapped.ddc

#------------------------------
# Compile
#------------------------------
puts "\n>>> Compiling design..."
compile

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

#------------------------------
# Generate reports
#------------------------------
puts "\n>>> Generating reports..."

report_units > ${rpt_dir}/${MODULE}_units.rpt
report_clock -skew > ${rpt_dir}/${MODULE}_clock.rpt
report_area -hierarchy > ${rpt_dir}/${MODULE}_area.rpt
report_timing > ${rpt_dir}/${MODULE}_timing.rpt
report_timing -delay_type max -max_paths 50 -nworst 1 -significant_digits 3 -sort_by slack > ${rpt_dir}/${MODULE}_timing_slack.rpt
report_constraints -all_violators > ${rpt_dir}/${MODULE}_constraints.rpt
report_reference -hierarchy > ${rpt_dir}/${MODULE}_reference.rpt
report_power -hierarchy > ${rpt_dir}/${MODULE}_power.rpt
report_qor > ${rpt_dir}/${MODULE}_qor.rpt

#------------------------------
# Print summary
#------------------------------
puts ""
puts "=========================================="
puts "Synthesis Complete: $MODULE"
puts "=========================================="
puts "Reports saved to: $rpt_dir"
puts ""
puts ">>> Area Summary:"
report_area
puts ""
puts ">>> Power Summary:"
report_power

exit
