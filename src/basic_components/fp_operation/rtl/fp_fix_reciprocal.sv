`timescale 1ns / 1ps
`include "global_define.vh"
/*
Module      : Floating Point Configurable Precision Adder (With Sign)
Timing      : Combinatorial Logic
Description : Adds two FP numbers with different exponents and signs.
              Aligns mantissas, preserves full precision (no bits discarded).
              Output format: {sign, exp_out, mant_out}.
              No rounding.
              It needs normalisation.
              The lossy part will be at the mantissa adder
Status      : Passed Simple Tests
*/

module fp_fix_reciprocal #(
    parameter int EXP_WIDTH = 5,
    parameter int MANT_WIDTH = 10,
    parameter int IEEE_COMPLIANCE = 1,
    // Register-slice depth around the reciprocal divider (timing-critical on FPGA).
    // Default 2 = pre-existing behaviour; raise for pipelining + retiming. See fp_reciprocal.
    parameter int REG_N = 2,
    // Mid-cone register slice splitting the round cone from the normalize cone. See fp_cp_reciprocal.
    parameter int NORM_PIPE = 0
)(
    input  logic clk,
    input  logic rst,
    input  logic data_in_valid,
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_in,  // {sign, exp, mant}
    output logic [EXP_WIDTH + MANT_WIDTH : 0] data_out,
    output logic data_out_valid
);

`ifdef DC_LIB_EN
    DW_fp_recip_inst #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) dc_lib_fp_fix_reciprocal (
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_in(data_in),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );
`else
    fp_cp_reciprocal #(
        .IN_EXP_WIDTH(EXP_WIDTH),
        .IN_MANT_WIDTH(MANT_WIDTH),
        .OUT_EXP_WIDTH(EXP_WIDTH),
        .OUT_MANT_WIDTH(MANT_WIDTH),
        .REG_N(REG_N),
        .NORM_PIPE(NORM_PIPE)
    ) fp_cp_reciprocal_inst (
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_in(data_in),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );
`endif // DC_LIB_EN

endmodule
