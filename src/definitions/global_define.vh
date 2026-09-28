`ifndef GLOBAL_DEFINE_VH
`ifndef VIVADO_SYNTHESIS
`ifndef SIMULATION
`define SIMULATION
`endif
`endif
// `define DC_LIB_EN
// `define HADAMARD_EN
`define MAMBA_EXTENSION_EN
// MAMBA_SCAN_EN gates ONLY the V_PS_V prefix-scan (fp_prefix_scan_syn, ~11.5k LUT). It is
// separate from MAMBA_EXTENSION_EN, which now gates only V_SHFT_V (fp_vec_shift, needed by GQA).
// Comment this out to reclaim the prefix-scan area on FPGA -- no workload emits V_PS_V, so it is
// dead unless running a Mamba selective-scan; llama attn/ffn and GQA are unaffected.
`define MAMBA_SCAN_EN
// `define ASIC
// `define SYNTHESIS_MEMORY_BLACK_BOXING
`endif



