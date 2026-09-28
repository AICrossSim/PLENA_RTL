`timescale 1ns / 1ps

/*
Module      : Processing Element (PE) in Systolic GEMM for MXFP
Timing      : Sequential
Description : MXFP-specific PE with fixed-point accumulation for improved timing
Status      : Under Development
*/

module mxfp_default_pe #(
    // MX-FP Data Format
    parameter MX_T_EXP_WIDTH        = 4,
    parameter MX_T_MANT_WIDTH       = 3,
    parameter MX_L_EXP_WIDTH        = 4,
    parameter MX_L_MANT_WIDTH       = 3,
    parameter MX_SCALE_WIDTH        = 8,

    // Accumulator Data Format
    parameter ACC_FP_EXP_WIDTH      = 8,
    parameter ACC_FP_MANT_WIDTH     = 7,
    parameter ACC_INT_WIDTH         = 16,  // Fixed-point accumulator integer width
    parameter ACC_FRAC_WIDTH        = 16,  // Fixed-point accumulator fraction width
    parameter PROD_EXT_EXP_WIDTH    = 0,
    parameter PROD_EXT_MANT_WIDTH   = 0
)(

    input logic clk,
    input logic rst,
    input logic clear_accumulator,

    // Input from Top
    input  logic [MX_T_MANT_WIDTH + MX_T_EXP_WIDTH : 0] in_top_element,
    input  logic [MX_SCALE_WIDTH - 1 : 0]               in_top_scale,
    input  logic system_top_valid,

    // Input from Left
    input  logic [MX_L_MANT_WIDTH + MX_L_EXP_WIDTH : 0] in_left_element,
    input  logic [MX_SCALE_WIDTH - 1 : 0]               in_left_scale,
    input  logic system_left_valid,

    // Output to Bottom
    output logic [MX_T_MANT_WIDTH + MX_T_EXP_WIDTH : 0] out_bottom_element,
    output logic [MX_SCALE_WIDTH - 1 : 0]               out_bottom_scale,

    // Output to Right
    output logic [MX_L_MANT_WIDTH + MX_L_EXP_WIDTH : 0] out_right_element,
    output logic [MX_SCALE_WIDTH - 1 : 0]               out_right_scale,

    // Output Result
    output logic [ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] out_fp,
    output logic out_result_valid
);

    // ==============================================================================================
    // Declaration : registers, wires
    // ==============================================================================================
    localparam SCALE_BIAS = (1 << (MX_SCALE_WIDTH - 1)) - 1;

    logic [MX_T_MANT_WIDTH + MX_T_EXP_WIDTH : 0]        reg_top_element;
    logic [MX_SCALE_WIDTH - 1 : 0]                      reg_top_scale;
    logic [MX_L_MANT_WIDTH + MX_L_EXP_WIDTH : 0]        reg_left_element;
    logic [MX_SCALE_WIDTH - 1 : 0]                      reg_left_scale;

    // ==============================================================================================
    // STAGE 1: Pass Data from Top and Left to the Bottom and Right
    // ==============================================================================================

    always_ff @(posedge clk) begin
        if (rst) begin
            reg_top_element  <= {MX_T_MANT_WIDTH + MX_T_EXP_WIDTH + 1{1'b0}};
            reg_top_scale    <= {MX_SCALE_WIDTH{1'b0}};
            reg_left_element <= {MX_L_MANT_WIDTH + MX_L_EXP_WIDTH + 1{1'b0}};
            reg_left_scale   <= {MX_SCALE_WIDTH{1'b0}};
        end else begin
            if (system_top_valid) begin
                reg_top_element <= in_top_element;
                reg_top_scale   <= in_top_scale;
            end

            if (system_left_valid) begin
                reg_left_element <= in_left_element;
                reg_left_scale   <= in_left_scale;
            end
        end
    end

    assign out_bottom_element   =   reg_top_element;
    assign out_bottom_scale     =   reg_top_scale;

    assign out_right_element    =   reg_left_element;
    assign out_right_scale      =   reg_left_scale;

    // ==============================================================================================
    // STAGE 2: FP MAC
    // ==============================================================================================

    logic [MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0] block_mult_result;
    logic [MX_SCALE_WIDTH - 1 : 0] scale_sum_result, reg_scale_sum;
    logic block_mult_in_valid, block_mult_out_valid;
    logic scale_sum_in_valid;
    logic scale_sum_out_valid;

    assign block_mult_in_valid = system_top_valid && system_left_valid;
    assign scale_sum_in_valid = block_mult_in_valid;

    fp_cp_asym_mult #(
        .EXP_WIDTH_A    (MX_T_EXP_WIDTH),
        .MANT_WIDTH_A   (MX_T_MANT_WIDTH),
        .EXP_WIDTH_B    (MX_L_EXP_WIDTH),
        .MANT_WIDTH_B   (MX_L_MANT_WIDTH)
    ) element_mult (
        .clk(clk),
        .rst(rst),
        .data_in_valid  (block_mult_in_valid),
        .data_a         (reg_top_element),
        .data_b         (reg_left_element),
        .data_out       (block_mult_result),
        .data_out_valid (block_mult_out_valid)
    );

    assign scale_sum_result = reg_top_scale + reg_left_scale - SCALE_BIAS;

    fifo_wo_hs #(
        .DATA_WIDTH(MX_SCALE_WIDTH),
        .DEPTH(3)
    ) buffer_scale_sum (
        .clk(clk),
        .rst(rst),
        .data_in        (scale_sum_result),
        .data_in_valid  (scale_sum_in_valid),
        .data_out       (reg_scale_sum),
        .data_out_valid (scale_sum_out_valid)
    );

    // ==============================================================================================
    // STAGE 3: Shift the result according to the scale
    // ==============================================================================================
    logic [ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH : 0] shifted_result;
    logic converted_result_valid;
    logic mxfp_mult_valid;

    assign mxfp_mult_valid = block_mult_out_valid && scale_sum_out_valid;

    mx_fp_2_fp_unary #(
        .MXFP_EXP_WIDTH     (MX_L_EXP_WIDTH),
        .MXFP_MANT_WIDTH    (MX_L_MANT_WIDTH),
        .MXFP_SCALE_WIDTH   (MX_SCALE_WIDTH),
        .FP_EXP_WIDTH       (ACC_FP_EXP_WIDTH),
        .FP_MANT_WIDTH      (ACC_FP_MANT_WIDTH)
    ) mx_fp_to_fp (
        .clk                (clk),
        .rst                (rst),
        .data_in_valid      (mxfp_mult_valid),
        .element_data_in    (block_mult_result),
        .scale_data_in      (reg_scale_sum),
        .data_out_valid     (converted_result_valid),
        .fp_out             (shifted_result)
    );

    // ==============================================================================================
    // STAGE 4: Accumulation (Fixed-point for improved timing)
    // ==============================================================================================
    fp_fix_accumulator #(
        .MANT_WIDTH     (ACC_FP_MANT_WIDTH),
        .EXP_WIDTH      (ACC_FP_EXP_WIDTH),
        .ACC_INT_WIDTH  (ACC_INT_WIDTH),
        .ACC_FRAC_WIDTH (ACC_FRAC_WIDTH)
    ) acc_adder (
        .clk(clk),
        .rst(rst),
        .clear_accumulator  (clear_accumulator),
        .data_in_valid      (converted_result_valid),
        .data_in            (shifted_result),
        .data_out           (out_fp),
        .data_out_valid     (out_result_valid)
    );

endmodule
