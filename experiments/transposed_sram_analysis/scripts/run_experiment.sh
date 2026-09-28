#!/bin/bash
#------------------------------
# Run Transposed SRAM Analysis Experiment
# Compares biaccess_sram vs SRAM + transpose buffer
#------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
SYNOPSYS_DIR="${SCRIPT_DIR}/../../../tools/synopsys"

# Create results directories
mkdir -p "${RESULTS_DIR}/biaccess_sram"/{logs,reports,netlist,out}
mkdir -p "${RESULTS_DIR}/sram_transpose_buffer"/{logs,reports,netlist,out}

echo "=========================================="
echo "Transposed SRAM Analysis Experiment"
echo "=========================================="
echo ""
echo "Design 1: biaccess_sram_32x32"
echo "  - DATA_WIDTH = 8"
echo "  - SRAM_DEPTH = MLEN = 32"
echo "  - Parallel_Rd_Dim = 1"
echo ""
echo "Design 2: sram_with_transpose_buffer"
echo "  - Pure SRAM (32 x 256 bits)"
echo "  - + Transpose buffer (32x32 matrix)"
echo ""
echo "=========================================="

# Check if dc_shell is available
if ! command -v dc_shell &> /dev/null; then
    echo "ERROR: dc_shell not found. Please source Synopsys environment."
    echo "Example: source /path/to/synopsys/setup.sh"
    exit 1
fi

# Copy .synopsys_dc.setup if exists
if [ -f "${SYNOPSYS_DIR}/.synopsys_dc.setup" ]; then
    cp "${SYNOPSYS_DIR}/.synopsys_dc.setup" "${SCRIPT_DIR}/"
fi

cd "${SCRIPT_DIR}"

echo ""
echo ">>> Step 1: Synthesizing biaccess_sram_32x32..."
echo "=========================================="
dc_shell -f synth_biaccess_sram.tcl 2>&1 | tee "${RESULTS_DIR}/biaccess_sram/synth.log"

echo ""
echo ">>> Step 2: Synthesizing sram_with_transpose_buffer..."
echo "=========================================="
dc_shell -f synth_sram_transpose_buffer.tcl 2>&1 | tee "${RESULTS_DIR}/sram_transpose_buffer/synth.log"

echo ""
echo ">>> Step 3: Generating comparison report..."
echo "=========================================="

# Generate comparison report
python3 "${SCRIPT_DIR}/compare_results.py" > "${RESULTS_DIR}/comparison_report.txt"

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Key reports:"
echo "  - ${RESULTS_DIR}/biaccess_sram/reports/biaccess_sram_32x32_area.rpt"
echo "  - ${RESULTS_DIR}/biaccess_sram/reports/biaccess_sram_32x32_power.rpt"
echo "  - ${RESULTS_DIR}/sram_transpose_buffer/reports/sram_with_transpose_buffer_area.rpt"
echo "  - ${RESULTS_DIR}/sram_transpose_buffer/reports/sram_with_transpose_buffer_power.rpt"
echo "  - ${RESULTS_DIR}/comparison_report.txt"
echo ""
cat "${RESULTS_DIR}/comparison_report.txt"
