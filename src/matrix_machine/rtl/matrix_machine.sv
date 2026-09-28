`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Matrix Machine Module V2
Timing      : Sequential, Takes x cycles to compute the dot product
Description : This module is the newer version of the matrix machine, supporting both GEMM and GEMV operations.
Status      : Passed Simple Tests
*/


module matrix_machine import precision_pkg::*; import configuration_pkg::*; #(
    localparam  BLOCK_NUM       = MLEN / BLOCK_DIM,
    localparam  ADDR_WIDTH      = ON_CHIP_ADDR_WIDTH,
    // Element widths depend on whether MXINT or MXFP format is used
    localparam  WT_ELEMENT_WIDTH  = WT_MX_INT_ENABLE ? WT_MX_INT_WIDTH : (WT_MX_MANT_WIDTH + WT_MX_EXP_WIDTH + 1),
    localparam  ACT_ELEMENT_WIDTH = ACT_MX_INT_ENABLE ? ACT_MX_INT_WIDTH : (ACT_MXFP_MANT_WIDTH + ACT_MXFP_EXP_WIDTH + 1)
) (
    input   logic   clk,
    input   logic   rst,

    // Execution Control
    input  OP_BUNDLE    exe_stage_op,
    output logic        mcu_active,

    // Matix - row-major order
    input  logic [MLEN-1:0] [WT_ELEMENT_WIDTH-1:0]                                m_element,
    input  logic [MLEN-1:0] [MX_SCALE_WIDTH-1:0]                                  m_scale,
    input  logic                   m_valid,

    // Vector - row-major order
    input  logic [MLEN-1:0] [ACT_ELEMENT_WIDTH-1:0]                               v_element,
    input  logic [BLOCK_NUM-1:0] [MX_SCALE_WIDTH-1:0]                             v_scale,
    input  logic                   v_valid,

    // Output
    output logic [MLEN-1:0] [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH:0]    out_v_fp,
    output logic                                                    out_valid,
    output logic [ADDR_WIDTH-1:0]                                   m_waddr,
    output logic [1:0]                                              m_wreq
`ifdef SIMULATION
    // Debug Signals: Unconnected now.
    ,
    output logic dbg_complete_v1_load,
    output logic dbg_complete_v2_load,
    output logic dbg_complete_loading_q,
    output logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH:0] dbg_gebm_result_0
`endif
);

    // Declarations
    import pipeline_pkg::*;
    localparam ACC_ADDR_WIDTH = $clog2(MLEN / BLEN) + 1;
    M_OP    matrix_opcode; 

    logic   [ADDR_WIDTH-1:0] recorded_m_waddr;
    logic   [ADDR_WIDTH-1:0]  addr_in;
    logic   result_waddr_update;
    logic   wait_for_output;
    logic   [1:0] recorded_wr_mode; // 2'b01: M_MM_WO, 2'b10: M_MV_WO
    logic   [MLEN-1:0] [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0]   result_v;
    logic   result_out_valid;
    logic   [ACC_ADDR_WIDTH-1:0] acc_addr;
    logic   acc_addr_valid;
    logic   acc_addr_ready;
    logic   [VLEN-1:0] [ACT_ELEMENT_WIDTH-1:0]                         stored_v_element, stored_v_element_pre;
    logic   [BLOCK_NUM-1:0] [MX_SCALE_WIDTH-1:0]                      stored_v_scale, stored_v_scale_pre;
    logic   stored_v_ele_valid, stored_v_scale_valid;
    logic   stored_v_ele_valid_pre, stored_v_scale_valid_pre;

    // Control Signals

    assign matrix_opcode        = exe_stage_op.m_op;
    assign addr_in              = exe_stage_op.addr_2;
    assign result_waddr_update  = exe_stage_op.update_m_waddr;

    // Address Management
    // Storing the address written back to the vector SRAM

    always_ff @(posedge clk) begin
        if (rst) begin
            recorded_m_waddr    <= 'b0;
            wait_for_output     <= 1'b0;
        end else begin
            // Set result waddr 
            if (matrix_opcode == MM_WO)begin
                recorded_m_waddr <= addr_in;
                wait_for_output <= 1'b1;
                recorded_wr_mode <= 2'b01; // M_MM_WO
            end else if (matrix_opcode == MV_WO) begin
                recorded_m_waddr <= addr_in;
                wait_for_output <= 1'b1;
                recorded_wr_mode <= 2'b10; // M_MV_WO
            end else if (result_out_valid) begin
                wait_for_output <= 1'b0;
                recorded_wr_mode <= 2'b00; // No write
            end
        end
    end

    // Load Accumulation Address
    writeback_buffer_controller #(
        .BLEN (BLEN),
        .MLEN (MLEN),
        .ADDR_WIDTH (ACC_ADDR_WIDTH)
    ) write_buffer_controller (
        .clk(clk),
        .rst(rst),
        .buffer_addr_in             (exe_stage_op.addr_2[ACC_ADDR_WIDTH-1:0]),
        .writeback_buffer_enable    (exe_stage_op.update_m_waddr),
        .buffer_addr_out            (acc_addr),
        .buffer_addr_valid          (acc_addr_valid),
        .buffer_addr_ready          (acc_addr_ready)
    );


    // Data Preparation

    // Data from Matrix SRAM Buffering
    logic [MLEN-1:0] [WT_ELEMENT_WIDTH-1:0]                          stored_m_element;
    logic [MLEN-1:0] [MX_SCALE_WIDTH-1:0]                             stored_m_scale;
    logic stored_m_ele_valid, stored_m_scale_valid;
    logic stored_m_valid;

    // Delay the m (weight) data+valid one cycle so it aligns with the v (activation)
    // side, which arrives a stage later (its FP->MXINT re-quantize adds a stage).
    logic [MLEN-1:0] [WT_ELEMENT_WIDTH-1:0] m_element_d;
    logic [MLEN-1:0] [MX_SCALE_WIDTH-1:0]   m_scale_d;
    logic m_valid_d;
    always_ff @(posedge clk) begin
        if (rst) begin
            m_element_d <= '0;
            m_scale_d   <= '0;
            m_valid_d   <= 1'b0;
        end else begin
            m_element_d <= m_element;
            m_scale_d   <= m_scale;
            m_valid_d   <= m_valid;
        end
    end

    register_slice_wo_hs #(
        .DATA_WIDTH(MLEN * WT_ELEMENT_WIDTH)
    ) matrix_element_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (m_element_d),
        .data_in_valid  (m_valid_d),
        .data_out       (stored_m_element),
        .data_out_valid (stored_m_ele_valid)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(MLEN * MX_SCALE_WIDTH)
    ) matrix_scale_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (m_scale_d),
        .data_in_valid  (m_valid_d),
        .data_out       (stored_m_scale),
        .data_out_valid (stored_m_scale_valid)
    );

    assign stored_m_valid = stored_m_ele_valid & stored_m_scale_valid;

    // Vector SRAM buffering: latch with the undelayed v_valid; alignment with the
    // delayed m side is restored by the post-slice stage below.
    register_slice_wo_hs #(
        .DATA_WIDTH(VLEN * ACT_ELEMENT_WIDTH)
    ) vector_element_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (v_element),
        .data_in_valid  (v_valid),
        .data_out       (stored_v_element_pre),
        .data_out_valid (stored_v_ele_valid_pre)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(BLOCK_NUM * MX_SCALE_WIDTH)
    ) vector_scale_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (v_scale),
        .data_in_valid  (v_valid),
        .data_out       (stored_v_scale_pre),
        .data_out_valid (stored_v_scale_valid_pre)
    );

    // Post-slice +1 stage so stored_v_valid aligns with stored_m_valid.
    always_ff @(posedge clk) begin
        if (rst) begin
            stored_v_element     <= '0;
            stored_v_scale       <= '0;
            stored_v_ele_valid   <= 1'b0;
            stored_v_scale_valid <= 1'b0;
        end else begin
            stored_v_element     <= stored_v_element_pre;
            stored_v_scale       <= stored_v_scale_pre;
            stored_v_ele_valid   <= stored_v_ele_valid_pre;
            stored_v_scale_valid <= stored_v_scale_valid_pre;
        end
    end

    logic stored_v_valid;
    assign stored_v_valid = stored_v_ele_valid & stored_v_scale_valid;

    // Systolic Matrix Compute Unit

    // Expand BLOCK_DIM-wide activation scales to the MCU's BLEN-wide scale blocks.
    // +2 compensates the FP->MXINT mantissa grid (2^6) vs the MCU's int/2^7 dequant.
    logic [MLEN/BLEN-1:0][MX_SCALE_WIDTH-1:0] v2_scale_expanded;
    for (genvar sb = 0; sb < MLEN/BLEN; sb++) begin : gen_v2_scale_expand
        assign v2_scale_expanded[sb] = stored_v_scale[(sb*BLEN)/BLOCK_DIM] + 8'd2;
    end

    // HBM weights are sign-magnitude; the MCU computes in two's complement. Convert.
    logic [MLEN-1:0][WT_ELEMENT_WIDTH-1:0] stored_m_element_tc;
    for (genvar me = 0; me < MLEN; me++) begin : gen_m_elem_tc
        assign stored_m_element_tc[me] =
            (stored_m_element[me][WT_ELEMENT_WIDTH-1] && stored_m_element[me][WT_ELEMENT_WIDTH-2:0] != '0)
                ? {1'b1, (~stored_m_element[me][WT_ELEMENT_WIDTH-2:0] + 1'b1)}
                : (stored_m_element[me][WT_ELEMENT_WIDTH-1] ? '0 : stored_m_element[me]);
    end

    // The MCU uses one weight scale per KLEN block (assumes uniform within block),
    // but HBM weights carry differing per-K-row scales. Align each block to its max
    // scale (shift smaller-scaled mantissas right) and present the uniform scale.
    localparam MCU_KLEN = BLEN; // KLEN == BLEN at the MCU instantiation below
    logic [MLEN-1:0][WT_ELEMENT_WIDTH-1:0] m_elem_blkalign;
    logic [MLEN-1:0][MX_SCALE_WIDTH-1:0]   m_scale_blkalign;
    always_comb begin
        for (int blk = 0; blk < MLEN/MCU_KLEN; blk++) begin
            automatic logic [MX_SCALE_WIDTH-1:0] blk_max;
            blk_max = stored_m_scale[blk*MCU_KLEN];
            for (int e = 1; e < MCU_KLEN; e++) begin
                if (stored_m_scale[blk*MCU_KLEN+e] > blk_max)
                    blk_max = stored_m_scale[blk*MCU_KLEN+e];
            end
            for (int e = 0; e < MCU_KLEN; e++) begin
                automatic int k;
                automatic logic [MX_SCALE_WIDTH-1:0] sh;
                k = blk*MCU_KLEN + e;
                sh = blk_max - stored_m_scale[k];
                m_elem_blkalign[k]  = $signed(stored_m_element_tc[k]) >>> sh;
                m_scale_blkalign[k] = blk_max;
            end
        end
    end

    if (WT_MX_INT_ENABLE) begin : gen_mxint_systolic_mcu
        mxint_systolic_mcu #(
            .FP_EXP_WIDTH       (V_FP_EXP_WIDTH),
            .FP_MANT_WIDTH      (V_FP_MANT_WIDTH),
            .MX_T_INT_WIDTH     (WT_MX_INT_WIDTH),
            .MX_L_INT_WIDTH     (ACT_MX_INT_WIDTH),
            .MXINT_SCALE_WIDTH  (MX_SCALE_WIDTH),
            .KLEN               (BLEN), // Temporarily set KLEN to BLEN to make is consistent with the Compiler settings.
            .BLEN               (BLEN),
            .MLEN               (MLEN)
        ) matrix_compute_unit (
            .clk                (clk),
            .rst                (rst),
            .control            (matrix_opcode),
            .per_head           (exe_stage_op.m_per_head),
            .v1_element         (m_elem_blkalign),
            .v1_scale           (m_scale_blkalign),
            .v1_in_valid        (stored_m_valid),
            .v2_element         (stored_v_element),
            .v2_scale           (v2_scale_expanded),
            .v2_in_valid        (stored_v_valid),
            .v_result           (result_v),
            .v_result_write_req (result_out_valid),
            .mcu_active         (mcu_active)
        );
    end else begin : gen_mx_systolic_mcu
        mx_systolic_mcu #(
            .FP_EXP_WIDTH       (V_FP_EXP_WIDTH),
            .FP_MANT_WIDTH      (V_FP_MANT_WIDTH),
            .MX_T_EXP_WIDTH     (WT_MX_EXP_WIDTH),
            .MX_T_MANT_WIDTH    (WT_MX_MANT_WIDTH),
            .MX_L_EXP_WIDTH     (ACT_MXFP_EXP_WIDTH),
            .MX_L_MANT_WIDTH    (ACT_MXFP_MANT_WIDTH),
            .MX_SCALE_WIDTH     (MX_SCALE_WIDTH),
            .BLOCK_DIM          (BLOCK_DIM),
            .ACC_FP_EXP_WIDTH   (M_FP_EXP_WIDTH),
            .ACC_FP_MANT_WIDTH  (M_FP_MANT_WIDTH),
            .SYSTOLIC_PROCESSING_OVERHEAD (SYSTOLIC_PROCESSING_OVERHEAD),
            .M                  (BLEN),
            .K                  (MLEN),
            .N                  (BLEN),
            .ACC_ADDR_WIDTH     (ACC_ADDR_WIDTH),
            .L_MX_INT_EN        (WT_MX_INT_ENABLE)
        ) matrix_compute_unit (
            .clk                (clk),
            .rst                (rst),
            .control            (matrix_opcode),
            .acc_waddr          (acc_addr),
            .fetch_next_acc_waddr_valid  (acc_addr_valid),
            .fetch_next_acc_waddr_ready  (acc_addr_ready),
            .wait_for_output    (wait_for_output),
            .v1_element         (stored_m_element),
            .v1_scale           (stored_m_scale),
            .v1_in_valid        (stored_m_valid),
            .v2_element         (stored_v_element),
            .v2_scale           (stored_v_scale),
            .v2_in_valid        (stored_v_valid),
            .v_result           (result_v),
            .v_result_write_req (result_out_valid),
            .mcu_active  (mcu_active)
        );
    end

    logic delayed_result_out_valid;
    logic [MLEN-1:0] [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0] delayed_result_v;
    assign m_wreq   = (result_out_valid & ~delayed_result_out_valid) ? recorded_wr_mode : 2'b00;
    assign m_waddr  = recorded_m_waddr;

    always_ff @(posedge clk) begin
        if (rst) begin
            delayed_result_out_valid <= 1'b0;
            delayed_result_v <= '0;
        end else begin
            delayed_result_out_valid <= result_out_valid;
            delayed_result_v <= result_v;
        end
    end

    register_slice_wo_hs #(
        .DATA_WIDTH (MLEN * (V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1))
    ) result_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (delayed_result_v),
        .data_in_valid  (delayed_result_out_valid),
        .data_out       (out_v_fp),
        .data_out_valid (out_valid)
    );

endmodule
