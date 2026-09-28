`timescale 1ns / 1ps

/*
Module      : Summation across Systolic Array
Timing      : Sequential, adder tree.
*/

module mx_sum_across_sa #(
    parameter ACC_FP_MANT_WIDTH = 8,
    parameter ACC_FP_EXP_WIDTH = 7,
    parameter COMPUTE_DIM = 8,
    parameter SYS_ARRAY_AMOUNT = 4
)(
    input logic clk,
    input logic rst,

    // Input from Systolic Array
    input logic     [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM- 1: 0][COMPUTE_DIM- 1: 0][ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] m_in_data,
    input logic     in_valid,

    // Output to Top Level
    output  logic [COMPUTE_DIM- 1: 0][COMPUTE_DIM- 1: 0][ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH : 0] m_out_data,
    output  logic out_valid
);

    logic [COMPUTE_DIM * COMPUTE_DIM- 1: 0] per_pe_acc_valid;
    logic [COMPUTE_DIM- 1: 0][COMPUTE_DIM- 1: 0][ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH : 0] accumulated_data;

    // Pipeline register for input valid - breaks critical path from PE output to adder tree
    logic in_valid_q;
    always_ff @(posedge clk) begin
        if (rst)
            in_valid_q <= 1'b0;
        else
            in_valid_q <= in_valid;
    end

generate;

    for (genvar i = 0; i < COMPUTE_DIM; i++) begin : sum_across_row
        for (genvar j = 0; j < COMPUTE_DIM; j++) begin : sum_across_col
            // Pipeline register for adder input data - breaks critical path
            logic [SYS_ARRAY_AMOUNT - 1 : 0][ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH : 0] adder_in_data_q;
            always_ff @(posedge clk) begin
                for (int k = 0; k < SYS_ARRAY_AMOUNT; k++)
                    adder_in_data_q[k] <= m_in_data[k][i][j];
            end

            fp_adder_tree #(
                .VEC_DIM(SYS_ARRAY_AMOUNT),
                .IN_EXP_WIDTH(ACC_FP_EXP_WIDTH),
                .IN_MAN_WIDTH(ACC_FP_MANT_WIDTH),
                .EXT_MANT_WIDTH_PER_LAYER(0),
                .EXT_EXP_BITS_PER_LAYER(0)
            ) per_pe_adder_tree (
                .clk(clk),
                .rst(rst),
                .data_in        (adder_in_data_q),
                .data_in_valid  (in_valid_q),
                .data_out       (accumulated_data[i][j]),
                .data_out_valid (per_pe_acc_valid[i * COMPUTE_DIM + j])
            );
        end
    end

    logic acc_out_valid;
    assign acc_out_valid = &per_pe_acc_valid;

    register_slice_wo_hs #(
        .DATA_WIDTH(COMPUTE_DIM * COMPUTE_DIM * (ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH + 1))
    ) register_slice_inst (
        .clk(clk),
        .rst(rst),
        .data_in            (accumulated_data),
        .data_in_valid      (acc_out_valid),
        .data_out           (m_out_data),
        .data_out_valid     (out_valid)
    );

endgenerate

    

endmodule