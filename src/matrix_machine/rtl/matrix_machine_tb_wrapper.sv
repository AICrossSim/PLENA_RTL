`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Testbench Wrapper for matrix_machine_v2
Description : Exposes OP_BUNDLE struct fields as flat ports so cocotb/Verilator
              can drive them individually. Instantiates matrix_machine_v2
              and wires the struct through.
*/

module matrix_machine_tb_wrapper import precision_pkg::*; import configuration_pkg::*; #(
    localparam  BLOCK_NUM       = MLEN / BLOCK_DIM,
    localparam  ADDR_WIDTH      = ON_CHIP_ADDR_WIDTH
) (
    input   logic   clk,
    input   logic   rst,

    // OP_BUNDLE fields exposed as flat ports
    input   logic [3:0]     op_m_op,
    input   logic [4:0]     op_v_ele_op,
    input   logic [2:0]     op_v_reduct_op,
    input   logic [3:0]     op_s_fp_op,
    input   logic [2:0]     op_c_op,
    input   logic [2:0]     op_h_op,
    input   logic           op_m_transposed_read,
    input   logic           op_v_broadcast_en,
    input   logic [2:0]     op_fps1,
    input   logic [2:0]     op_fps2,
    input   logic [2:0]     op_fpd,
    input   logic [3:0]     op_gp_reg1,
    input   logic [3:0]     op_gp_reg2,
    input   logic [3:0]     op_gp_rstride,
    input   logic [3:0]     op_gp_rd,
    input   logic [31:0]    op_addr_1,
    input   logic [31:0]    op_addr_2,
    input   logic           op_update_m_waddr,
    input   logic           op_update_v_waddr,

    // Pass-through ports
    output logic        mcu_active,

    input  logic [MLEN-1:0] [(WT_MX_MANT_WIDTH + WT_MX_EXP_WIDTH):0]            m_element,
    input  logic [MLEN-1:0] [MX_SCALE_WIDTH-1:0]                                  m_scale,
    input  logic                   m_valid,
    output logic                   m_ready,

    input  logic [MLEN-1:0] [(ACT_MXFP_MANT_WIDTH + ACT_MXFP_EXP_WIDTH):0]      v_element,
    input  logic [BLOCK_NUM-1:0] [MX_SCALE_WIDTH-1:0]                             v_scale,
    input  logic                   v_valid,
    output logic                   v_ready,

    output logic [MLEN-1:0] [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH:0]    out_v_fp,
    output logic                                                    out_valid,
    input  logic                                                    out_ready,
    output logic [ADDR_WIDTH-1:0]                                   m_waddr,
    output logic [1:0]                                              m_wreq
`ifdef SIMULATION
    ,
    output logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH:0]              dbg_gebm_result_0
`endif
);

    // Construct OP_BUNDLE from flat ports
    OP_BUNDLE exe_op;
    assign exe_op.m_op              = M_OP'(op_m_op);
    assign exe_op.v_ele_op          = V_ELEMENT_OP'(op_v_ele_op);
    assign exe_op.v_reduct_op       = V_REDUCT_OP'(op_v_reduct_op);
    assign exe_op.s_fp_op           = S_FP_OP'(op_s_fp_op);
    assign exe_op.c_op              = C_OP'(op_c_op);
    assign exe_op.h_op              = H_OP'(op_h_op);
    assign exe_op.m_transposed_read = op_m_transposed_read;
    assign exe_op.v_broadcast_en    = op_v_broadcast_en;
    assign exe_op.fps1              = op_fps1;
    assign exe_op.fps2              = op_fps2;
    assign exe_op.fpd               = op_fpd;
    assign exe_op.gp_reg1           = op_gp_reg1;
    assign exe_op.gp_reg2           = op_gp_reg2;
    assign exe_op.gp_rstride        = op_gp_rstride;
    assign exe_op.gp_rd             = op_gp_rd;
    assign exe_op.addr_1            = op_addr_1;
    assign exe_op.addr_2            = op_addr_2;
    assign exe_op.update_m_waddr    = op_update_m_waddr;
    assign exe_op.update_v_waddr    = op_update_v_waddr;

    matrix_machine dut_i (
        .clk                (clk),
        .rst                (rst),
        .exe_stage_op       (exe_op),
        .mcu_active  (mcu_active),
        .m_element          (m_element),
        .m_scale            (m_scale),
        .m_valid            (m_valid),
        .v_element          (v_element),
        .v_scale            (v_scale),
        .v_valid            (v_valid),
        .out_v_fp           (out_v_fp),
        .out_valid          (out_valid),
        .m_waddr            (m_waddr),
        .m_wreq             (m_wreq)
`ifdef SIMULATION
        ,
        .dbg_complete_v1_load(),
        .dbg_complete_v2_load(),
        .dbg_complete_loading_q(),
        .dbg_gebm_result_0(dbg_gebm_result_0)
`endif
    );

endmodule
