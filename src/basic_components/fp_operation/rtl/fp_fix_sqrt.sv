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

module fp_fix_sqrt #(
    parameter int EXP_WIDTH = 6,
    parameter int MANT_WIDTH = 5,
    // Constant-fold the integer sqrt into a ROM (see fp_cp_sqrt). Bit-identical.
    parameter int SQRT_LUT = 1
)(
    input  logic clk,
    input  logic rst,
    input  logic data_in_valid,
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_in,  // {sign, exp, mant}
    output logic [EXP_WIDTH + MANT_WIDTH : 0] data_out,
    output logic data_out_valid
);

`ifdef DC_LIB_EN
    DW_fp_sqrt_inst #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) dc_lib_fp_fix_sqrt (
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_in(data_in),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );
`else
    // Intermediate wire for combinational sqrt result
    logic [EXP_WIDTH + MANT_WIDTH : 0] sqrt_result;

    fp_cp_sqrt #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH),
        .SQRT_LUT(SQRT_LUT)
    ) fp_cp_sqrt_inst (
        .data_in(data_in),
        .data_out(sqrt_result)
    );

    // Register stage without handshake
    register_slice_wo_hs #(
        .DATA_WIDTH(EXP_WIDTH + MANT_WIDTH + 1)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(sqrt_result),
        .data_in_valid(data_in_valid),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );
`endif

endmodule
