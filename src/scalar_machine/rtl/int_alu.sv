`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

/*
Module      : Scalar Fixed ALU
Timing      : Combinatorial Logic
Description : This module is mainly used for address manipulation
Status      : Passed Simple Test
*/

module int_alu #(
    parameter int BITWIDTH = 32,
    parameter int IMM_SHIFT_AMOUNT = 4
)(
    input  logic                  clk,
    input  logic                  rst,
    input  logic [BITWIDTH-1:0]   operand_a,
    input  logic [BITWIDTH-1:0]   operand_b,
    input  logic [BITWIDTH-1:0]   imm_value,
    input  S_INT_OP               operation,
    output logic                  result_valid,
    output logic [BITWIDTH-1:0]   computed_address,
    output logic [BITWIDTH-1:0]   result,

    // Loop control output
    output logic                  loop_counter_zero  // High when loop counter becomes 0 after decrement
);

    S_INT_OP             p1_operation;
    logic [BITWIDTH-1:0]   p1_operand_b;

    // 2-stage multiply pipeline: DSP48 multiply → register → result mux
    // Breaks the 7 ns DSP+CARRY4 critical path
    logic [BITWIDTH-1:0] mul_result_s1;
    logic                mul_valid_s1;

    // Stage 1: DSP multiply (registered output)
    always_ff @(posedge clk) begin
        if (rst) begin
            mul_result_s1 <= '0;
            mul_valid_s1  <= 1'b0;
        end else begin
            mul_result_s1 <= operand_a * operand_b;
            mul_valid_s1  <= (operation == MUL_INT);
        end
    end

    // Stage 2: ALU result mux + multiply writeback
    always_ff @(posedge clk) begin
        if (rst) begin
            result_valid <= 1'b0;
            result <= '0;
            p1_operation <= STALL_S_INT;
            p1_operand_b <= '0;
        end else begin
            p1_operation <= operation;
            p1_operand_b <= operand_b;
            if (mul_valid_s1) begin
                // Multiply result arrives 1 cycle later
                result <= mul_result_s1;
                result_valid <= 1'b1;
            end else begin
                case (operation)
                    ADD_INT: begin
                        result <= operand_a + operand_b;
                        result_valid <= 1'b1;
                    end

                    SUB_INT: begin
                        result <= operand_a - operand_b;
                        result_valid <= 1'b1;
                    end

                    MUL_INT: begin
                        // Multiply in flight — result comes next cycle via mul_valid_s1
                        result_valid <= 1'b0;
                    end

                    LUI_INT: begin
                        result <= { {(BITWIDTH - IMM_SHIFT_AMOUNT){1'b0}}, imm_value, {IMM_SHIFT_AMOUNT{1'b0}} };
                        result_valid <= 1'b1;
                    end

                    ADDI_INT: begin
                        result <= operand_a + imm_value;
                        result_valid <= 1'b1;
                    end

                    LOOP_INIT: begin
                        // C_LOOP_START: Write immediate value to destination register
                        result <= imm_value;
                        result_valid <= 1'b1;
                    end

                    LOOP_DEC: begin
                        // C_LOOP_END: Decrement loop counter by 1
                        result <= operand_a - 1;
                        result_valid <= 1'b1;
                    end

                    default: begin
                        result <= '0;
                        result_valid <= 1'b0;
                    end
                endcase
            end
        end
    end
    assign computed_address = ((operation == COMP_ADDR) || (operation == LD_INT) || (operation == ST_INT)) ? operand_a + imm_value : '0;

    // Loop counter zero detection (registered to match result timing)
    // High when LOOP_DEC results in counter = 0 (i.e., operand_a - 1 = 0, meaning operand_a = 1)
    always_ff @(posedge clk) begin
        if (rst) begin
            loop_counter_zero <= 1'b0;
        end else begin
            if (operation == LOOP_DEC) begin
                loop_counter_zero <= (operand_a == 1);  // After decrement, counter will be 0
            end else begin
                loop_counter_zero <= 1'b0;
            end
        end
    end

endmodule
