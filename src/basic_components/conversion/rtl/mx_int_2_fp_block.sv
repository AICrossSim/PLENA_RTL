`timescale 1ns / 1ps

/*
Module      : Conversion a block unit of mx-int data to fp data
Timing      : Sequential Logic (uses register_slice internally)
Description : Wraps multiple mx_int_2_fp_unary instances for block-level conversion
Status      :
*/


module mx_int_2_fp_block #(
    parameter BLOCK_DIM = 8,
    parameter MXINT_WIDTH = 8,
    parameter MXINT_SCALE_WIDTH = 8,
    parameter FP_MANT_WIDTH = 7,
    parameter FP_EXP_WIDTH = 8
)(
    input   logic clk,
    input   logic rst,
    input   logic data_in_valid,
    input   logic [BLOCK_DIM-1:0][MXINT_WIDTH - 1 : 0] element_in,
    input   logic [MXINT_SCALE_WIDTH-1:0] scale_in,
    output  logic data_out_valid,
    output  logic [BLOCK_DIM-1:0][FP_MANT_WIDTH + FP_EXP_WIDTH : 0] fp_out
);

    logic [BLOCK_DIM-1:0] converted_data_out_valid;

    for (genvar i = 0; i < BLOCK_DIM; i++) begin : gen_mx_int_2_fp
        mx_int_2_fp_unary #(
            .MXINT_SCALE_WIDTH  (MXINT_SCALE_WIDTH),
            .MXINT_WIDTH        (MXINT_WIDTH),
            .FP_EXP_WIDTH       (FP_EXP_WIDTH),
            .FP_MANT_WIDTH      (FP_MANT_WIDTH)
        ) mx_int_2_fp_unary_inst (
            .clk            (clk),
            .rst            (rst),
            .data_in_valid  (data_in_valid),
            .element_data_in(element_in[i]),
            .scale_data_in  (scale_in),
            .data_out_valid (converted_data_out_valid[i]),
            .fp_out         (fp_out[i])
        );
    end

    assign data_out_valid = &converted_data_out_valid;

endmodule
