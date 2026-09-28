`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

module loop_controller import instruction_pkg::*; import configuration_pkg::*; #(
    parameter PC_WIDTH = 32
)(
    input   logic clk,
    input   logic rst,

    input   logic                               loop_start_valid,
    input   logic                               loop_end_valid,
    input   logic [INT_OPERAND_WIDTH-1:0]       loop_counter_reg,
    input   logic [PC_WIDTH-1:0]                current_pc,
    input   logic [PC_WIDTH-1:0]                loop_continue_pc,
    input   logic [PC_WIDTH-1:0]                loop_start_target,
    input   logic [IMM_WIDTH-1:0]               loop_count_imm,

    output  logic                               loop_jump_back,
    output  logic [PC_WIDTH-1:0]                loop_target_pc,
    output  logic                               loop_end_stall,
    output  logic                               loop_exit,
    output  logic [PC_WIDTH-1:0]                loop_exit_pc
);

    logic [PC_WIDTH-1:0] loop_start_pc_stack [MAX_LOOP_DEPTH-1:0];
    logic [INT_OPERAND_WIDTH-1:0] loop_counter_reg_stack [MAX_LOOP_DEPTH-1:0];
    logic [$clog2(MAX_LOOP_DEPTH):0] stack_ptr;

    logic reg_already_on_stack;
    always_comb begin
        reg_already_on_stack = 1'b0;
        for (int i = 0; i < MAX_LOOP_DEPTH; i++) begin
            if (i < stack_ptr && loop_counter_reg_stack[i] == loop_counter_reg) begin
                reg_already_on_stack = 1'b1;
            end
        end
    end

    logic [IMM_WIDTH-1:0] loop_counter_val_stack [MAX_LOOP_DEPTH-1:0];

    logic loop_counter_zero;
    assign loop_counter_zero = (stack_ptr > 0) && (loop_counter_val_stack[stack_ptr - 1] == 1);

    always_ff @(posedge clk) begin
        if (rst) begin
            stack_ptr <= '0;
            for (int i = 0; i < MAX_LOOP_DEPTH; i++) begin
                loop_start_pc_stack[i]    <= '0;
                loop_counter_reg_stack[i] <= '0;
                loop_counter_val_stack[i] <= '0;
            end
        end else begin
            if (loop_start_valid && stack_ptr < MAX_LOOP_DEPTH && !reg_already_on_stack) begin
                loop_start_pc_stack[stack_ptr]    <= loop_start_target;
                loop_counter_reg_stack[stack_ptr] <= loop_counter_reg;
                loop_counter_val_stack[stack_ptr] <= loop_count_imm;
                stack_ptr <= stack_ptr + 1;
            end
            else if (loop_end_valid && stack_ptr > 0) begin
                if (loop_counter_zero)
                    stack_ptr <= stack_ptr - 1;
                else
                    loop_counter_val_stack[stack_ptr - 1] <= loop_counter_val_stack[stack_ptr - 1] - 1;
            end
        end
    end

    assign loop_jump_back = loop_end_valid && !loop_counter_zero && (stack_ptr > 0);
    assign loop_target_pc = (stack_ptr > 0) ? loop_start_pc_stack[stack_ptr - 1] : '0;
    assign loop_exit      = loop_end_valid && loop_counter_zero && (stack_ptr > 0);
    assign loop_exit_pc   = loop_continue_pc;
    assign loop_end_stall = loop_end_valid;

endmodule
