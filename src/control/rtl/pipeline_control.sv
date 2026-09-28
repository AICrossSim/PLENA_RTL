`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

module pipeline_control import configuration_pkg::*; import instruction_pkg::*; #(
    parameter   INT_OPERAND_WIDTH       = 5,
    parameter   FP_OPERAND_WIDTH        = 5,
    parameter   INT_DATA_WIDTH          = 32,
    parameter   IMM_WIDTH               = 12
) (
    input       logic clk,
    input       logic rst,

    input       OP_BUNDLE       decode_stage_op,
    // 1 the cycle decode_stage_op is a freshly-dispatched op (reg_rd handshake: issue once).
    input       logic           decode_op_fresh,

    input       logic [INT_DATA_WIDTH - 1 : 0] gp_addr_1,
    input       logic [INT_DATA_WIDTH - 1 : 0] gp_addr_2,
    input       logic [INT_DATA_WIDTH - 1 : 0] gp_addr_3,

    input       logic v_sram_wen_a,
    input       logic [INT_DATA_WIDTH - 1 : 0]    v_sram_addr_a,
    input       logic v_sram_wen_b,
    input       logic [INT_DATA_WIDTH - 1 : 0]    v_sram_addr_b,
    input       logic hbm_m_prefetch_in_progress,
    input       logic hbm_v_prefetch_in_progress,
    input       logic continuous_write_to_v_sram,

    input       MEM_WREQ_INFO   mem_write_req,
    input       logic           hbm_in_used,
    input       logic           fp_stall_req,
    input       logic           fp_sram_stall_req,
    input       logic           m_load_in_process,
    input       logic           m_mcu_active,
    input       logic           s_received_v_reduct_result,

    output      logic           pipeline_stall_req,

    output      logic           v_drain_settle_o,
    output      OP_BUNDLE       exe_stage_op,
    output      MEM_WEN_INFO    mem_write_control,

    output      logic           v_elem_busy_o
);
    /* verilator no_inline_module */

    OP_BUNDLE   reg_rd_stage_op, check_stage_op, determine_stage_op, delayed_reg_rd_stage_op, invalid_op_bubble, recorded_check_stage_op;
    assign invalid_op_bubble = '{
        m_op                : STALL_M,
        v_ele_op            : STALL_V_ELEMENT,
        v_reduct_op         : STALL_V_REDUCT,
        s_fp_op             : STALL_S_FP,
        c_op                : STALL_C,
        h_op                : STALL_H,
        m_transposed_read   : 1'b0,
        m_per_head          : 1'b0,
        v_broadcast_en      : 1'b0,
        fps1                : '0,
        fps2                : '0,
        fpd                 : '0,
        gp_reg1             : '0,
        gp_reg2             : '0,
        gp_rstride          : '0,
        gp_rd               : '0,
        addr_1              : '0,
        addr_2              : '0,
        addr_3              : '0,
        update_m_waddr      : 1'b0,
        update_v_waddr      : 1'b0,
        pc_tag              : '0
    };

    import pipeline_pkg::*;
    logic   pipeline_stall;
    logic   stall_for_prefetch;
    logic   mem_vwrite_stall_req;
    logic   b1_pipeline_stall, b2_pipeline_stall, recover_from_stall, p1_recover_from_stall, p2_recover_from_stall, start_of_stall;
    // Skid bit: a fresh op presented while reg_rd had not yet advanced to take it.
    logic   srs_pending;
    // reg_rd's clock condition (mirrors the reg_rd-clocking branch below).
    logic   rr_advance;
    assign  rr_advance = !pipeline_stall && !recover_from_stall;
    logic   vector_reduct_in_process, tracking_vector_reduct_in_process;
    logic   [INT_DATA_WIDTH - 1 : 0] recorded_gp_addr_1, recorded_gp_addr_2, recorded_gp_addr_3;

    logic   fp_compute_in_exe, fp_compute_in_determine;
    assign  fp_compute_in_exe       = (exe_stage_op.s_fp_op == ADD_FP)  || (exe_stage_op.s_fp_op == SUB_FP)  || (exe_stage_op.s_fp_op == MUL_FP) ||
                                      (exe_stage_op.s_fp_op == SQRT_FP) || (exe_stage_op.s_fp_op == RECI_FP) || (exe_stage_op.s_fp_op == EXP_FP) ||
                                      (exe_stage_op.s_fp_op == MAX_FP);
    assign  fp_compute_in_determine = (determine_stage_op.s_fp_op == ADD_FP)  || (determine_stage_op.s_fp_op == SUB_FP)  || (determine_stage_op.s_fp_op == MUL_FP) ||
                                      (determine_stage_op.s_fp_op == SQRT_FP) || (determine_stage_op.s_fp_op == RECI_FP) || (determine_stage_op.s_fp_op == EXP_FP) ||
                                      (determine_stage_op.s_fp_op == MAX_FP);

    logic   v_elem_in_exe;
    logic   v_elem_busy;
    assign  v_elem_in_exe = (exe_stage_op.v_ele_op != STALL_V_ELEMENT);

    assign  v_elem_busy_o = v_elem_busy | v_elem_in_exe
                          | (reg_rd_stage_op.v_ele_op        != STALL_V_ELEMENT)
                          | (reg_rd_stage_op.v_reduct_op     != STALL_V_REDUCT)
                          | (delayed_reg_rd_stage_op.v_ele_op    != STALL_V_ELEMENT)
                          | (delayed_reg_rd_stage_op.v_reduct_op != STALL_V_REDUCT)
                          | (determine_stage_op.v_ele_op     != STALL_V_ELEMENT)
                          | (determine_stage_op.v_reduct_op  != STALL_V_REDUCT)
                          | (exe_stage_op.v_reduct_op        != STALL_V_REDUCT);

    always_ff @(posedge clk) begin
        if (rst)               v_elem_busy <= 1'b0;
        else if (v_elem_in_exe) v_elem_busy <= 1'b1;
        else if (v_sram_wen_a)  v_elem_busy <= 1'b0;
    end

    localparam int M_VSRAM_SETTLE = 64;
    logic [$clog2(M_VSRAM_SETTLE+1)-1:0] m_vsram_settle_cnt;

    always_ff @(posedge clk) begin
        if (rst)                          m_vsram_settle_cnt <= '0;
        else if (m_mcu_active)            m_vsram_settle_cnt <= M_VSRAM_SETTLE[$clog2(M_VSRAM_SETTLE+1)-1:0];
        else if (m_vsram_settle_cnt != 0) m_vsram_settle_cnt <= m_vsram_settle_cnt - 1'b1;
    end

    // Registered "previous exe op was a matrix writeout". m_result_write_pending is set
    // when a writeout is in exe, but m_load_in_process (stall term below) only reflects
    // it one cycle later. In a tight `M_MM_WO; M_MM` sequence the next tile's MM_IC
    // reaches `determine` in exactly that gap and slips into exe, arming a port-A load
    // that collides with the writeout's transient drain window (drain dropped ->
    // m_result_write_pending stuck -> pipeline hang). Holding the following matrix op
    // one extra cycle bridges the gap; m_load_in_process then covers the full drain.
    logic prev_exe_was_wo;
    always_ff @(posedge clk) begin
        if (rst) prev_exe_was_wo <= 1'b0;
        else     prev_exe_was_wo <= (exe_stage_op.m_op == MM_WO) || (exe_stage_op.m_op == MV_WO);
    end

    logic m_drain_settle_stall;
    assign m_drain_settle_stall = (m_vsram_settle_cnt != 0)
                                & ((determine_stage_op.v_ele_op != STALL_V_ELEMENT)
                                   || (determine_stage_op.v_reduct_op != STALL_V_REDUCT));
    assign v_drain_settle_o = m_drain_settle_stall;

    // Stall if the op in `determine` would collide with an in-flight resource.
    // Every branch was an identical stall, so this is just an OR of the hazards.
    always_comb begin
        pipeline_stall =
            // matrix prefetch busy vs a following PREFETCH_M
            (hbm_m_prefetch_in_progress & (determine_stage_op.h_op == PREFETCH_M_H || determine_stage_op.h_op == PREFETCH_M_L))
            // vector prefetch busy vs a following PREFETCH_V / C_BREAK
         || (hbm_v_prefetch_in_progress & (determine_stage_op.h_op == PREFETCH_V_H || determine_stage_op.c_op == C_BREAK))
            // MM_IC in exe vs any matrix op behind it
         || ((exe_stage_op.m_op == MM_IC) & (determine_stage_op.m_op != STALL_M))
            // writeout just left exe vs a matrix op behind it: bridge the 1-cycle
            // m_load_in_process lag so the next tile's load cannot start until this
            // tile's drain completes (see prev_exe_was_wo).
         || (prev_exe_was_wo & (determine_stage_op.m_op != STALL_M))
            // matrix operand load running vs matrix op behind it
         || (m_load_in_process & (determine_stage_op.m_op != STALL_M))
            // MCU active vs matrix op except its own write-out
         || (m_mcu_active & (determine_stage_op.m_op != STALL_M && determine_stage_op.m_op != MM_WO))
            // matrix drain-settle window
         || (m_drain_settle_stall)
            // vector prefetch / continuous vsram write vs vector or matrix op
         || ((hbm_v_prefetch_in_progress || continuous_write_to_v_sram)
             & (determine_stage_op.v_ele_op != STALL_V_ELEMENT || determine_stage_op.v_reduct_op != STALL_V_REDUCT
                || (determine_stage_op.m_op != STALL_M && determine_stage_op.m_op != MM_WO && determine_stage_op.m_op != MV_WO)))
            // scalar-SRAM port-A write vs vector/matrix op
         || (mem_write_req.wreq_s_sram_port_a
             & (determine_stage_op.v_ele_op != STALL_V_ELEMENT || determine_stage_op.v_reduct_op != STALL_V_REDUCT || determine_stage_op.m_op != STALL_M))
            // scalar-SRAM port-B write vs vector/matrix (non-write-out) op
         || (mem_write_req.wreq_s_sram_port_b
             & (determine_stage_op.v_ele_op != STALL_V_ELEMENT || (determine_stage_op.m_op != STALL_M & determine_stage_op.m_op != MM_WO & determine_stage_op.m_op != MV_WO)))
            // vector-element unit busy vs vector op
         || ((v_elem_in_exe | v_elem_busy) & (determine_stage_op.v_ele_op != STALL_V_ELEMENT || determine_stage_op.v_reduct_op != STALL_V_REDUCT))
            // fp compute in flight vs fp op
         || ((fp_stall_req | fp_compute_in_exe) & (fp_compute_in_determine || (determine_stage_op.s_fp_op == LD_OUT_FP)))
            // fp-sram stall vs fp reg-load / reg-store / map
         || (fp_sram_stall_req & (determine_stage_op.s_fp_op == LD_REG_FP || determine_stage_op.s_fp_op == ST_REG_FP || determine_stage_op.s_fp_op == MAP_V_FP))
            // vector reduction in process vs fp op
         || (vector_reduct_in_process & (determine_stage_op.s_fp_op != STALL_S_FP))
            // pending vector-write stall
         || (mem_vwrite_stall_req);
    end

    assign pipeline_stall_req = pipeline_stall || b1_pipeline_stall;

    addr_monitor #(
        .ADDR_WIDTH(INT_DATA_WIDTH),
        .PIPELINE_STAGES(MAX_PIPELINE_STAGE)
    ) addr_monitor_inst (
        .clk(clk),
        .rst(rst),
        .determine_stage_op     (check_stage_op),
        .v_sram_addr_a          (v_sram_addr_a),
        .v_sram_addr_b          (v_sram_addr_b),
        .v_sram_wen_a           (v_sram_wen_a),
        .v_sram_wen_b           (v_sram_wen_b),
        .stall_req              (mem_vwrite_stall_req),
        .sys_pipe_stall         (b1_pipeline_stall)
    );

    // ------------------------------------------------------------------------
    // Vector-address drift lock (matrix->vector boundary only).
    // The scalar machine produces a vector op's address on gp_out one cycle after
    // its indices are presented; but a SERIALIZED element chain (SiLU: sub->exp->
    // add->reci->mul) can delay a late op's `check` staple until AFTER gp_out has
    // moved on and collapsed to 0 -> that op reads VRAM address 0 (the ffn row-0
    // bug: the MUL reads the scratch operand from 0 instead of its real address).
    // Snapshot the last VALID vector address and replay it when a post-matrix
    // vector element op staples with a bubble (gp_addr==0). Registered value at
    // the existing staple point, so no timing change. Armed ONLY by the matrix
    // drain-settle, so pure-vector workloads (rms has no matrix op) never arm it
    // and are byte-for-byte unaffected.
    logic                            post_matrix_vec;
    logic [INT_DATA_WIDTH - 1 : 0]   vlock_addr_1;
    always_ff @(posedge clk) begin
        if (rst) begin
            post_matrix_vec <= 1'b0;
            vlock_addr_1    <= '0;
        end else begin
            if (m_drain_settle_stall)  post_matrix_vec <= 1'b1;   // at a matrix->vector boundary
            else if (!v_elem_busy_o)   post_matrix_vec <= 1'b0;   // vector chain fully drained
            if (post_matrix_vec && gp_addr_1 != '0)
                vlock_addr_1 <= gp_addr_1;                        // hold last valid operand addr
        end
    end
    logic vlock_sub_1;   // replace a drifted (==0) port-A addr of a post-matrix vector element op
    assign vlock_sub_1 = post_matrix_vec
                       & (delayed_reg_rd_stage_op.v_ele_op != STALL_V_ELEMENT)
                       & (gp_addr_1 == '0);

    always_comb begin
        check_stage_op.m_op            = delayed_reg_rd_stage_op.m_op;
        check_stage_op.v_ele_op        = delayed_reg_rd_stage_op.v_ele_op;
        check_stage_op.v_reduct_op     = delayed_reg_rd_stage_op.v_reduct_op;
        check_stage_op.s_fp_op         = delayed_reg_rd_stage_op.s_fp_op;
        check_stage_op.c_op            = delayed_reg_rd_stage_op.c_op;
        check_stage_op.h_op            = delayed_reg_rd_stage_op.h_op;
        check_stage_op.m_transposed_read = delayed_reg_rd_stage_op.m_transposed_read;
        check_stage_op.m_per_head      = delayed_reg_rd_stage_op.m_per_head;
        check_stage_op.v_broadcast_en  = delayed_reg_rd_stage_op.v_broadcast_en;
        check_stage_op.fps1            = delayed_reg_rd_stage_op.fps1;
        check_stage_op.fps2            = delayed_reg_rd_stage_op.fps2;
        check_stage_op.fpd             = delayed_reg_rd_stage_op.fpd;
        check_stage_op.gp_rd           = delayed_reg_rd_stage_op.gp_rd;
        check_stage_op.gp_reg1         = delayed_reg_rd_stage_op.gp_reg1;
        check_stage_op.gp_reg2         = delayed_reg_rd_stage_op.gp_reg2;
        check_stage_op.gp_rstride      = delayed_reg_rd_stage_op.gp_rstride;
        check_stage_op.addr_1          = vlock_sub_1 ? vlock_addr_1
                                         : (p2_recover_from_stall ? recorded_gp_addr_1 : gp_addr_1);
        check_stage_op.addr_2          = p2_recover_from_stall ? recorded_gp_addr_2 : gp_addr_2;
        check_stage_op.addr_3          = p2_recover_from_stall ? recorded_gp_addr_3 : gp_addr_3;
        check_stage_op.update_m_waddr  = delayed_reg_rd_stage_op.update_m_waddr;
        check_stage_op.update_v_waddr  = delayed_reg_rd_stage_op.update_v_waddr;
        check_stage_op.pc_tag          = delayed_reg_rd_stage_op.pc_tag;
    end

    assign recover_from_stall = (!pipeline_stall) && b1_pipeline_stall;
    assign start_of_stall = pipeline_stall && !b1_pipeline_stall;
    assign vector_reduct_in_process = tracking_vector_reduct_in_process || (exe_stage_op.v_reduct_op != STALL_V_REDUCT);

    always_ff @(posedge clk) begin
        if (rst) begin
            mem_write_control <= '{
                w_m_sram_en         : 1'b0,
                w_s_sram_port_a_en  : 1'b0,
                w_s_sram_port_b_en  : 1'b0,
                w_from_m            : 1'b0
            };
            b1_pipeline_stall                       <= 1'b0;
            b2_pipeline_stall                       <= 1'b0;
            tracking_vector_reduct_in_process       <= 1'b0;
            recorded_gp_addr_1                      <= 'b0;
            recorded_gp_addr_2                      <= 'b0;
            recorded_gp_addr_3                      <= 'b0;
            p1_recover_from_stall                   <= 1'b0;
            p2_recover_from_stall                   <= 1'b0;
            srs_pending                             <= 1'b0;

        end else begin

            // Hold a fresh op reg_rd hasn't taken yet; rr_advance takes/clears it.
            if (rr_advance)
                srs_pending <= 1'b0;
            else if (decode_op_fresh)
                srs_pending <= 1'b1;

            if (exe_stage_op.v_reduct_op != STALL_V_REDUCT) begin
                tracking_vector_reduct_in_process <= 1'b1;
            end else if (s_received_v_reduct_result) begin
                tracking_vector_reduct_in_process <= 1'b0;
            end

            mem_write_control <= '{
                w_m_sram_en           : mem_write_req.wreq_m_sram,
                w_s_sram_port_a_en    : mem_write_req.wreq_s_sram_port_a,
                w_s_sram_port_b_en    : mem_write_req.wreq_s_sram_port_b,
                w_from_m              : mem_write_req.wreq_from_m
            };

            b1_pipeline_stall <= pipeline_stall;
            b2_pipeline_stall <= b1_pipeline_stall;
            p1_recover_from_stall <= recover_from_stall;
            p2_recover_from_stall <= p1_recover_from_stall;

            if (!b2_pipeline_stall & b1_pipeline_stall) begin
                recorded_gp_addr_1 <= gp_addr_1;
                recorded_gp_addr_2 <= gp_addr_2;
                recorded_gp_addr_3 <= gp_addr_3;
            end

            if (recover_from_stall) begin
                determine_stage_op          <= recorded_check_stage_op;
                exe_stage_op                <= determine_stage_op;
            end else if (start_of_stall) begin
                recorded_check_stage_op     <= check_stage_op;
                delayed_reg_rd_stage_op     <= invalid_op_bubble;
            end else if (!pipeline_stall) begin
                // Issue once: clock a fresh/pending op into reg_rd, else a bubble
                // (don't re-issue a frozen op while the decoder is held).
                reg_rd_stage_op             <= (decode_op_fresh || srs_pending)
                                                ? decode_stage_op : invalid_op_bubble;
                delayed_reg_rd_stage_op     <= reg_rd_stage_op;
                determine_stage_op          <= check_stage_op;
                exe_stage_op                <= determine_stage_op;
            end else begin
                exe_stage_op                <= invalid_op_bubble;
            end

        end
    end

endmodule
