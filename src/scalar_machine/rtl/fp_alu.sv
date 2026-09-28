`timescale 1ns / 1ps
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Scalar FP ALU
Timing      : Combinatorial Logic
Description : This module is used for all the FP operations
            : 1. FP Add, 2. FP Subtract, 3. FP Multiply, 
*/

module fp_alu import instruction_pkg::*; #(
    parameter   EXP_WIDTH = 5,
    parameter   MANT_WIDTH = 10
)(
    input  logic clk,
    input  logic rst,
    input  S_FP_OP operation,
    input  logic [FP_OPERAND_WIDTH - 1 : 0]     reg_waddr,
    output  logic [FP_OPERAND_WIDTH - 1 : 0]    stored_reg_waddr, 
    input  logic [EXP_WIDTH + MANT_WIDTH : 0]   data_a,  // {sign, exp, mant}
    input  logic [EXP_WIDTH + MANT_WIDTH : 0]   data_b,
    output logic [EXP_WIDTH + MANT_WIDTH : 0]   data_out,
    output  logic                               data_out_valid
);


logic data_in_valid;
logic mult_data_in_valid;
logic mult_data_out_valid;
logic add_data_in_valid;
logic add_data_out_valid;
logic [EXP_WIDTH + MANT_WIDTH : 0] p1_data_a, p1_data_b;

logic [EXP_WIDTH + MANT_WIDTH : 0] fp_add_out, fp_sub_out, fp_mul_out;
logic [EXP_WIDTH + MANT_WIDTH : 0] negated_data_b;
logic [EXP_WIDTH + MANT_WIDTH : 0] data_out_add, data_out_mul, data_out_exp;
logic negated_en;
logic data_out_ready;

// Scalar-FP max (MAX_FP): real-valued compare-select of the two operands.
logic       max_sign_a, max_sign_b;
logic       max_a_ge_b;
logic [EXP_WIDTH + MANT_WIDTH : 0] fp_max_out;

// Status Tracking
S_FP_OP recorded_operation;
logic alu_in_use;

always_ff @(posedge clk) begin
    if (rst) begin
        recorded_operation  <= STALL_S_FP;
        data_in_valid       <= 1'b0;
        stored_reg_waddr    <= 'b0;
        data_out_ready      <= 1'b0;
        alu_in_use          <= 1'b0;
        p1_data_a           <= 'b0;
        p1_data_b           <= 'b0;
    end else begin
        data_out_ready      <= 1'b1;
        p1_data_a           <= data_a;
        p1_data_b           <= data_b;
        if (!alu_in_use & (operation == ADD_FP || operation == SUB_FP || operation == MUL_FP || operation == MV_FP || operation == MAX_FP)) begin
            recorded_operation <= operation;
            stored_reg_waddr <= reg_waddr;
            data_in_valid <= 1'b1;
            alu_in_use <= 1'b1;
        end else if (data_out_valid & data_out_ready & alu_in_use) begin
            // At the end of the operation, reset the ALU (clearing alu_in_use
            // here is essential: stuck at 1 it kills the accept branch and
            // only the first FP ALU op of the program ever executes).
            recorded_operation  <= STALL_S_FP;
            data_in_valid       <= 1'b0;
            alu_in_use          <= 1'b0;
        end else if (alu_in_use) begin
            recorded_operation <= recorded_operation;
            data_in_valid <= 1'b0;
        end else begin
            recorded_operation <= STALL_S_FP;
            data_in_valid <= 1'b0;
        end
    end
end


always_comb begin
    // Combinational module to flip the sign bit for FP subtraction
    negated_data_b = {~p1_data_b[EXP_WIDTH + MANT_WIDTH], p1_data_b[EXP_WIDTH + MANT_WIDTH - 1 : 0]};

    // Use the REGISTERED operands (aligned with recorded_operation, like fp_add/fp_mul).
    max_sign_a = p1_data_a[EXP_WIDTH + MANT_WIDTH];
    max_sign_b = p1_data_b[EXP_WIDTH + MANT_WIDTH];
    if (max_sign_a != max_sign_b) begin
        // Differing signs: the positive operand (sign == 0) is always the larger.
        max_a_ge_b = ~max_sign_a;
    end else if (~max_sign_a) begin
        // Both positive: larger {exp,mant} magnitude wins.
        max_a_ge_b = (p1_data_a[EXP_WIDTH + MANT_WIDTH - 1 : 0] >= p1_data_b[EXP_WIDTH + MANT_WIDTH - 1 : 0]);
    end else begin
        // Both negative: smaller {exp,mant} magnitude wins.
        max_a_ge_b = (p1_data_a[EXP_WIDTH + MANT_WIDTH - 1 : 0] <= p1_data_b[EXP_WIDTH + MANT_WIDTH - 1 : 0]);
    end
    fp_max_out = max_a_ge_b ? p1_data_a : p1_data_b;

    // Defaults: without these, branches that don't assign the other unit's
    // in_valid latch its previous value and can spuriously re-fire it.
    add_data_in_valid  = 1'b0;
    mult_data_in_valid = 1'b0;
    case (recorded_operation)
        ADD_FP: begin
            negated_en = 1'b0;
            add_data_in_valid   = data_in_valid;
            data_out            = data_out_add;
            data_out_valid      = add_data_out_valid;
        end

        SUB_FP: begin
            negated_en          = 1'b1;
            add_data_in_valid   = data_in_valid;
            data_out            = data_out_add;
            data_out_valid      = add_data_out_valid;
        end

        MUL_FP: begin
            negated_en = 1'b0;
            mult_data_in_valid  = data_in_valid;
            data_out            = data_out_mul;
            data_out_valid      = mult_data_out_valid;
        end

        MV_FP: begin
            negated_en      = 1'b0;
            data_out        = data_a;
            data_out_valid  = data_in_valid;
        end

        MAX_FP: begin
            // Single-cycle select of the larger operand (like MV_FP; no add/mul pipeline).
            negated_en      = 1'b0;
            data_out        = fp_max_out;
            data_out_valid  = data_in_valid;
        end

        default: begin
            negated_en      = 1'b0;
            data_out        = 'b0;
            data_out_valid  = 1'b0;
        end

    endcase
end


fp_fix_adder #(
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(MANT_WIDTH)
) adder (
    .clk(clk),
    .rst(rst),
    .data_in_valid(add_data_in_valid),
    .data_a(p1_data_a),
    .data_b(negated_en ? negated_data_b : p1_data_b),
    .data_out(data_out_add),
    .data_out_valid(add_data_out_valid)
);

fp_fix_mult #(
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(MANT_WIDTH)
) multiplier (
    .clk(clk),
    .rst(rst),
    .data_in_valid(mult_data_in_valid),
    .data_a(p1_data_a),
    .data_b(p1_data_b),
    .data_out(data_out_mul),
    .data_out_valid(mult_data_out_valid)
);


endmodule