`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Testbench Wrapper for decoder
Description : Exposes OP_BUNDLE struct fields as flat output ports so
              cocotb/Verilator can probe them individually. Instantiates
              decoder and wires the structs through.
*/

module decoder_tb_wrapper import instruction_pkg::*; import configuration_pkg::*; (
    input  logic clk,
    input  logic rst,
    input  logic system_stall_flag,
    input  logic pipeline_stall,

    // Instruction input
    input  logic [INSTRUCTION_LENGTH - 1 : 0] instruction,
    input  logic instruction_valid,
    output logic instruction_ready,

    // Flat outputs for decode_stage_op fields
    output logic [3:0] dec_m_op,
    output logic [3:0] dec_v_ele_op,
    output logic [2:0] dec_v_reduct_op,
    output logic [3:0] dec_s_fp_op,
    output logic [2:0] dec_c_op,
    output logic [2:0] dec_h_op,
    output logic       dec_update_m_waddr,
    output logic       dec_update_v_waddr,

    // Integer outputs
    output logic [3:0] out_assigned_int_op,
    output logic [INT_OPERAND_WIDTH-1:0] out_rd,
    output logic [INT_OPERAND_WIDTH-1:0] out_rs1,
    output logic [INT_OPERAND_WIDTH-1:0] out_rs2,
    output logic [IMM_WIDTH-1:0] out_imm
);

    OP_BUNDLE decode_stage_op;
    S_INT_OP  assigned_int_op;
    logic [INT_OPERAND_WIDTH-1:0] rd, rs1, rs2;
    logic [IMM_WIDTH-1:0] imm;

    decoder dut_i (
        .clk(clk),
        .rst(rst),
        .system_stall_flag(system_stall_flag),
        .pipeline_stall(pipeline_stall),
        .instruction(instruction),
        .instruction_valid(instruction_valid),
        .instruction_ready(instruction_ready),
        .decode_stage_op(decode_stage_op),
        .assigned_int_op(assigned_int_op),
        .rd(rd), .rs1(rs1), .rs2(rs2), .imm(imm)
    );

    // Flatten struct for cocotb
    assign dec_m_op           = decode_stage_op.m_op;
    assign dec_v_ele_op       = decode_stage_op.v_ele_op;
    assign dec_v_reduct_op    = decode_stage_op.v_reduct_op;
    assign dec_s_fp_op        = decode_stage_op.s_fp_op;
    assign dec_c_op           = decode_stage_op.c_op;
    assign dec_h_op           = decode_stage_op.h_op;
    assign dec_update_m_waddr = decode_stage_op.update_m_waddr;
    assign dec_update_v_waddr = decode_stage_op.update_v_waddr;
    assign out_assigned_int_op = assigned_int_op;
    assign out_rd  = rd;
    assign out_rs1 = rs1;
    assign out_rs2 = rs2;
    assign out_imm = imm;

endmodule
