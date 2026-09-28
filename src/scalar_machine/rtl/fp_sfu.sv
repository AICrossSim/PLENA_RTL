`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

/*
Module      : Scalar FP Special Function Unit
Timing      : Combinatorial Logic
Description : This module is used for all the FP operations
            : 1. FP Reciprocal 2. FP Sqrt 6. FP Exp
Note        : In this version of the FP_SFU, since we assume that if there are continous FP related 
              Instructions, they are very likely to be data dependent. Therefore, only when the single operation
              is completed, the next operation will be started. Does not support pipelining (Can be optimized in the future)
Status      : Under Development
*/


module fp_sfu import configuration_pkg::*; import instruction_pkg::*; #(
    parameter   EXP_WIDTH = 5,
    parameter   MANT_WIDTH = 10,
    parameter   FP_OPERAND_WIDTH = 3
)(
    input   logic clk,
    input   logic rst,
    input   logic [EXP_WIDTH + MANT_WIDTH : 0]  data_in,     
    input   logic [FP_OPERAND_WIDTH - 1 : 0]    reg_waddr, 
    input   S_FP_OP                             operation,         
    output  logic [EXP_WIDTH + MANT_WIDTH : 0]  data_out,
    output  logic [FP_OPERAND_WIDTH - 1 : 0]    stored_reg_waddr, 
    output  logic                               data_out_valid
);


// Status Tracking
S_FP_OP recorded_operation;
logic data_in_valid;
logic sfu_in_use;
// Registered copy of data_in, matching fp_alu's p1_data_a. data_in (= scalar
// fp_reg_1) is only valid the cycle the op is seen; data_in_valid is asserted the
// next cycle, by which point the live data_in has moved to the next op's read. The
// sub-units (combinational sqrt/reci + register_slice) latch on data_in_valid, so
// feeding the held p1_data_in keeps the operand stable -> avoids sqrt(0)=0.
logic [EXP_WIDTH + MANT_WIDTH : 0] p1_data_in;

always_ff @(posedge clk) begin
    if (rst) begin
        recorded_operation <= STALL_S_FP;
        data_in_valid <= 1'b0;
        stored_reg_waddr <= 'b0;
        sfu_in_use <= 1'b0;
        p1_data_in <= 'b0;
    end else begin
        p1_data_in <= data_in;
        if (!sfu_in_use & (operation == RECI_FP || operation == EXP_FP || operation == SQRT_FP)) begin
            recorded_operation <= operation;
            stored_reg_waddr <= reg_waddr;
            data_in_valid <= 1'b1;
            sfu_in_use <= 1'b1;
        end else if (data_out_valid & sfu_in_use) begin
            // At the end of the operation, reset the SFU
            recorded_operation <= STALL_S_FP;
            data_in_valid <= 1'b0;
            sfu_in_use <= 1'b0;
        end else if (sfu_in_use) begin
            recorded_operation <= recorded_operation;
            data_in_valid <= 1'b0;
        end else begin
            recorded_operation <= STALL_S_FP;
            data_in_valid <= 1'b0;
        end
    end
end

logic [EXP_WIDTH + MANT_WIDTH : 0] fp_reciprocal_out, fp_exp_out, fp_sqrt_out;
logic [EXP_WIDTH + MANT_WIDTH : 0] result_data;
logic result_valid;
logic reciprocal_in_valid, exp_in_valid;
logic reciprocal_out_valid, exp_out_valid;
logic sqrt_out_valid, sqrt_in_valid;


always_comb begin
    // Defaults: branches that don't assign another unit's in_valid would
    // latch its previous value and spuriously re-fire that unit.
    reciprocal_in_valid = 1'b0;
    exp_in_valid        = 1'b0;
    sqrt_in_valid       = 1'b0;
    case (recorded_operation)
        RECI_FP: begin
            result_data             = fp_reciprocal_out;
            reciprocal_in_valid     = data_in_valid;
            result_valid            = reciprocal_out_valid;
        end

        EXP_FP: begin
            result_data             = fp_exp_out;
            exp_in_valid            = data_in_valid;
            result_valid            = exp_out_valid;
        end
        SQRT_FP: begin
            result_data             = fp_sqrt_out;
            sqrt_in_valid           = data_in_valid;
            result_valid            = sqrt_out_valid;
        end

        default: begin
            result_data             = {(EXP_WIDTH + MANT_WIDTH){1'b0}}; // Default case to avoid latches
            result_valid            = 1'b0;
        end
    endcase
end

    fp_fix_reciprocal #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) scalar_fp_reciprocal_init (
        .clk(clk),
        .rst(rst),
        .data_in_valid  (reciprocal_in_valid),
        .data_in        (p1_data_in),
        .data_out_valid (reciprocal_out_valid),
        .data_out       (fp_reciprocal_out)
    );

    fp_fix_exp #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) scalar_fp_exp_init (
        .clk(clk),
        .rst(rst),
        .data_in_valid  (exp_in_valid),
        .data_in        (p1_data_in),
        .data_out_valid (exp_out_valid),
        .data_out       (fp_exp_out)
    );

    fp_fix_sqrt #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) scalar_fp_sqrt_init (
        .clk(clk),
        .rst(rst),
        .data_in_valid  (sqrt_in_valid),
        .data_in        (p1_data_in),
        .data_out_valid (sqrt_out_valid),
        .data_out       (fp_sqrt_out)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(EXP_WIDTH + MANT_WIDTH + 1)
    ) output_reg (
        .clk           (clk),
        .rst           (rst),
        .data_in       (result_data),
        .data_in_valid (result_valid),
        .data_out      (data_out),
        .data_out_valid(data_out_valid)
    );

endmodule