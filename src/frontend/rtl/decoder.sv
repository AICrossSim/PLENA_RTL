`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

module decoder import instruction_pkg::*; import configuration_pkg::*; #(
    parameter INSTRUCTION_LENGTH = 32,
    parameter OPERAND_WIDTH      = 4,
    parameter OPCODE_WIDTH       = 6,
    parameter IMM_WIDTH          = 22,
    parameter PC_ADDR_WIDTH      = 16
)(
    input   logic clk,
    input   logic rst,
    input   logic system_stall_flag,

    input   logic pipeline_stall,

    output  logic [PC_ADDR_WIDTH - 1 : 0] pc,
    input   logic [INSTRUCTION_LENGTH - 1 : 0] instruction_from_imem,
    input   logic [PC_ADDR_WIDTH - 1 : 0] instruction_addr_from_imem,
    input   logic instruction_ready,

    output      OP_BUNDLE       decode_stage_op,
    output      S_INT_OP        assigned_int_op,

    output      logic [INT_OPERAND_WIDTH - 1 : 0] rd,
    output      logic [INT_OPERAND_WIDTH - 1 : 0] rs1,
    output      logic [INT_OPERAND_WIDTH - 1 : 0] rs2,
    output      logic [IMM_WIDTH - 1 : 0] imm,

    input       logic v_elem_busy,

    input       logic v_drain_settle,

    `ifdef SIMULATION

    output      logic c_break_detected,
    `endif

    // 1 the cycle decode_stage_op is a freshly-dispatched op (reg_rd handshake).
    output      logic decode_op_fresh_o

);
/* verilator no_inline_module */

logic [31:0] pc_reg;
logic [31:0] pc_reg_d1;
logic [31:0] loop_continue_pc;
logic [31:0] loop_body_pc;

logic loop_start_valid;
logic loop_end_valid;
logic loop_jump_back;
logic [31:0] loop_target_pc;
logic loop_end_stall;
logic loop_exit;
logic [31:0] loop_exit_pc;

logic [31:0] pc_next;

always_comb begin
    if (loop_jump_back)
        pc_next = loop_target_pc;
    else if (loop_exit)
        pc_next = loop_exit_pc;
    else
        pc_next = pc_reg + 4;
end

logic [31:0] expected_decode_pc;
logic fetch_skipped_ahead;
assign fetch_skipped_ahead = load_instr_valid
    && (instruction_addr_from_imem > expected_decode_pc[PC_ADDR_WIDTH - 1 : 0]);

always_ff @(posedge clk) begin
    if (rst)
        pc_reg <= 32'h0;
    else if (loop_jump_back)
        pc_reg <= pc_next;
    else if (loop_exit)
        pc_reg <= pc_next;
    else if (loop_end_stall || (early_loop_end_stall && !fetch_skipped_ahead))
        pc_reg <= pc_reg;
    else if (read_instr & load_instr_valid & !pipeline_stall)
        pc_reg <= fetch_skipped_ahead ? expected_decode_pc : pc_next;
    // Replay back-up; skip during scalar_raw_stall and for a matrix WO in decode (kept until exe). Floor at 0.
    else if (!p1_pipeline_stall & pipeline_stall & !early_loop_end_stall_d1 & !loop_end_stall_d1 & !scalar_raw_stall
             & (decode_stage_op.m_op != MM_WO) & (decode_stage_op.m_op != MV_WO))
        pc_reg <= (pc_reg >= 32'd4) ? pc_reg - 32'd4 : 32'd0;
end

assign pc = pc_reg;

loop_controller #(
    .PC_WIDTH(32)
) u_loop_controller (
    .clk                (clk),
    .rst                (rst),
    .loop_start_valid   (loop_start_valid),
    .loop_end_valid     (loop_end_valid),
    .loop_counter_reg   (decode_instr_info.rd[INT_OPERAND_WIDTH-1:0]),
    .current_pc         (pc_reg),
    .loop_continue_pc   (loop_continue_pc),
    .loop_start_target  (loop_body_pc),
    .loop_count_imm     (decode_instr_info.imm),
    .loop_jump_back     (loop_jump_back),
    .loop_target_pc     (loop_target_pc),
    .loop_end_stall     (loop_end_stall),
    .loop_exit          (loop_exit),
    .loop_exit_pc       (loop_exit_pc)
);

assign loop_start_valid = decode_instr_valid && (decode_instr_info.opcode == C_LOOP_START);
assign loop_end_valid   = decode_instr_valid && (decode_instr_info.opcode == C_LOOP_END) && !pipeline_stall;

logic           system_stall;
logic           stall_for_read_rd;
logic           [INSTRUCTION_LENGTH - 1 : 0] loaded_instr;
logic           read_instr, load_instr_valid;
logic           decode_instr_valid;
logic           p1_pipeline_stall, recover_from_stall, start_from_stall;
OP_BUNDLE       recorded_op_bundle;
S_INT_OP        exe_int_op;
S_INT_OP        recorded_assigned_int_op;
CUSTOM_ISA_TYPE decode_instruction_type, active_decode_instruction_type;
INSTR_INFO decode_instr_info;

logic rd_operand_ready;
logic m_update_waddr, v_update_waddr;
logic recorded_m_update_waddr, recorded_v_update_waddr;
logic pass_m_update_waddr, pass_v_update_waddr;
logic [INT_OPERAND_WIDTH - 1 : 0] rd_to_load;
logic [INT_OPERAND_WIDTH - 1 : 0] recorded_rd_to_load;
logic [INT_OPERAND_WIDTH - 1 : 0] pass_rd_to_load;
logic stall_for_read_rd_flag;
logic decode_advanced;
logic recorded_stall_for_read_rd_flag;
logic fixed_op_stall_flag;

INSTR_INFO skid_instr_info;
logic      skid_valid;
logic      skid_injected;

logic scalar_raw_stall;
assign scalar_raw_stall = v_elem_busy & decode_instr_valid & (decode_instr_info.opcode == S_ADDI_INT);

assign  read_instr = !effective_pipeline_stall & !stall_for_read_rd & !fixed_op_stall_flag & (system_stall == 1'b0) & !loop_end_stall & !scalar_raw_stall;

logic fresh_fetch;
assign fresh_fetch = (instruction_addr_from_imem == expected_decode_pc[PC_ADDR_WIDTH - 1 : 0]);

always_ff @(posedge clk) begin
    if (rst) begin
        expected_decode_pc <= 32'h0;
    end else if (loop_jump_back) begin
        expected_decode_pc <= loop_target_pc;
    end else if (loop_exit) begin
        expected_decode_pc <= loop_exit_pc;
    end else if (!pipeline_stall && load_instr_valid && fresh_fetch
                 && (   (stall_for_read_rd && !skid_valid)
                     || (!stall_for_read_rd && !skid_valid && read_instr))) begin
        expected_decode_pc <= expected_decode_pc + 4;
    end
end

assign loaded_instr     = instruction_from_imem;
assign load_instr_valid = instruction_ready;

logic [OPCODE_WIDTH - 1 : 0]    loaded_opcode;
logic [OPERAND_WIDTH:0]         loaded_rs1;
logic [OPERAND_WIDTH:0]         loaded_rs2;
logic [OPERAND_WIDTH:0]         loaded_rstride;
logic [OPERAND_WIDTH:0]         loaded_rd;
logic [IMM_WIDTH - 1 : 0]       loaded_imm;
logic [FUNCT_WIDTH - 1 : 0]     loaded_funct1;

assign loaded_imm       = ((loaded_opcode == S_ADDI_INT) || (loaded_opcode == S_LD_FP)  || (loaded_opcode == S_ST_FP)
                                                         || (loaded_opcode == S_LD_INT) || (loaded_opcode == S_ST_INT) ) ?
                                                         {{(IMM_WIDTH - IMM_2_WIDTH){1'b0}} , loaded_instr[INSTRUCTION_LENGTH - 1 -: IMM_2_WIDTH]} :
                                                         loaded_instr[INSTRUCTION_LENGTH - 1 -: IMM_WIDTH];

assign loaded_rd        = loaded_instr[OPERAND_WIDTH + OPCODE_WIDTH - 1 -: OPERAND_WIDTH];
assign loaded_rs1       = loaded_instr[2 * OPERAND_WIDTH + OPCODE_WIDTH - 1 -: OPERAND_WIDTH];
assign loaded_rs2       = loaded_instr[3 * OPERAND_WIDTH + OPCODE_WIDTH - 1 -: OPERAND_WIDTH];

assign loaded_opcode    = loaded_instr[OPCODE_WIDTH - 1 : 0];

assign loaded_funct1    = loaded_instr[5 * OPERAND_WIDTH + OPCODE_WIDTH - 1 -: FUNCT_WIDTH];
assign loaded_rstride   = loaded_instr[4 * OPERAND_WIDTH + OPCODE_WIDTH - 1 -: OPERAND_WIDTH];

logic early_loop_end_stall;

logic c_loop_end_lookahead;
assign c_loop_end_lookahead = load_instr_valid && (loaded_opcode == C_LOOP_END) && !loop_end_stall;
assign early_loop_end_stall = c_loop_end_lookahead && !scalar_raw_stall;

logic early_loop_end_stall_d1;
always_ff @(posedge clk) begin
    if (rst)
        early_loop_end_stall_d1 <= 1'b0;
    else
        early_loop_end_stall_d1 <= early_loop_end_stall;
end

logic loop_jump_back_d1;
always_ff @(posedge clk) begin
    if (rst)
        loop_jump_back_d1 <= 1'b0;
    else
        loop_jump_back_d1 <= loop_jump_back;
end

logic loop_end_stall_d1;
always_ff @(posedge clk) begin
    if (rst)
        loop_end_stall_d1 <= 1'b0;
    else
        loop_end_stall_d1 <= loop_end_stall;
end

always_ff @(posedge clk) begin
    if (rst) begin
        pc_reg_d1        <= 32'h0;
        loop_continue_pc <= 32'h0;
        loop_body_pc     <= 32'h0;
    end else begin
        pc_reg_d1 <= pc_reg;

        if (c_loop_end_lookahead)
            loop_continue_pc <= pc_reg_d1 + 4;

        if (load_instr_valid && (loaded_opcode == C_LOOP_START))
            loop_body_pc <= pc_reg_d1 + 4;
    end
end

logic loop_in_progress;
assign loop_in_progress = early_loop_end_stall || loop_end_stall;

logic effective_pipeline_stall;

// A matrix WO in decode honors pipeline_stall even mid-loop, so a trailing C_LOOP_END can't drop its drain.
logic decode_matrix_wo;
assign decode_matrix_wo = (decode_stage_op.m_op == MM_WO) || (decode_stage_op.m_op == MV_WO);
assign effective_pipeline_stall = (pipeline_stall && (!loop_in_progress || decode_matrix_wo)) || v_drain_settle;
assign active_decode_instruction_type = decode_instr_valid ? decode_instr_info.instruction_type : INVALID_TYPE;

always_comb begin
    case (loaded_opcode)

        M_MM, M_TMM, M_BMM, M_BTMM, M_MM_WO, M_BMM_WO, M_MV, M_TMV, M_MV_WO: begin
            decode_instruction_type = M;
        end

        V_ADD_VV, V_ADD_VF, V_SUB_VV, V_SUB_VF, V_MUL_VV, V_MUL_VF, V_EXP_V, V_RECI_V, V_RED_SUM, V_RED_MAX, C_HADAMARD_TRANSFORM, V_PS_V, V_SHFT_V : begin
            decode_instruction_type = V;
        end

        S_ADD_INT, S_ADDI_INT, S_SUB_INT, S_MUL_INT, S_LUI_INT, S_LD_INT, S_ST_INT: begin
            decode_instruction_type = S_INT;
        end

        S_ADD_FP, S_SUB_FP, S_MAX_FP, S_MUL_FP, S_EXP_FP, S_RECI_FP, S_SQRT_FP, S_LD_FP, S_ST_FP, S_MAP_V_FP: begin
            decode_instruction_type = S_FP;
        end

        H_PREFETCH_M, H_PREFETCH_V, H_STORE_V: begin
            decode_instruction_type = H;
        end

        C_SET_ADDR_REG, C_SET_SCALE_REG, C_SET_STRIDE_REG, C_BREAK, C_LOOP_START, C_LOOP_END: begin
            decode_instruction_type = C;
        end

        default: begin
            decode_instruction_type = INVALID_TYPE;
        end

    endcase
end

always_ff @(posedge clk) begin
    if (rst) begin
        recorded_stall_for_read_rd_flag <= 1'b0;
        recorded_m_update_waddr         <= 1'b0;
        recorded_v_update_waddr         <= 1'b0;
        recorded_rd_to_load             <= {INT_OPERAND_WIDTH{1'b0}};
        p1_pipeline_stall               <= 1'b0;
        exe_int_op                      <= STALL_S_INT;
        decode_advanced_d               <= 1'b0;
        int_op_pending                  <= 1'b0;
        system_stall                    <= 1'b0;
        decode_instr_valid              <= 1'b0;
        decode_instr_info               <= '{opcode: '0, rs1: '0, rs2: '0, rstride: '0, rd: '0, imm: '0, funct1: '0, instruction_type: INVALID_TYPE};
        skid_instr_info                 <= '{opcode: '0, rs1: '0, rs2: '0, rstride: '0, rd: '0, imm: '0, funct1: '0, instruction_type: INVALID_TYPE};
        skid_valid                      <= 1'b0;
        skid_injected                   <= 1'b0;
    end else begin
        if (system_stall_flag) begin
            system_stall <= 1'b1;
        end
        if (!pipeline_stall) begin

            if (stall_for_read_rd) begin
                decode_instr_valid <= decode_instr_valid;
                skid_injected      <= 1'b0;

                if (load_instr_valid & fresh_fetch & !skid_valid) begin
                    skid_instr_info <= '{opcode: loaded_opcode, rs1: loaded_rs1, rs2: loaded_rs2, rstride: loaded_rstride, rd: loaded_rd, imm: loaded_imm, funct1: loaded_funct1, instruction_type: decode_instruction_type};
                    skid_valid      <= 1'b1;
                end
            end else if (scalar_raw_stall) begin

                decode_instr_valid <= decode_instr_valid;
                skid_injected      <= 1'b0;
            end else if (skid_valid) begin

                decode_instr_valid <= 1'b1;
                decode_instr_info  <= skid_instr_info;
                skid_valid         <= 1'b0;
                skid_injected      <= 1'b1;
            end else if (read_instr & load_instr_valid & fresh_fetch) begin
                decode_instr_valid <= 1'b1;
                decode_instr_info  <= '{opcode: loaded_opcode, rs1: loaded_rs1, rs2: loaded_rs2, rstride: loaded_rstride, rd: loaded_rd, imm: loaded_imm, funct1: loaded_funct1, instruction_type: decode_instruction_type};
                skid_injected      <= 1'b0;
            end else begin
                decode_instr_valid <= 1'b0;
                skid_injected      <= 1'b0;
            end
        end else if (load_instr_valid & fresh_fetch & !skid_valid & (loaded_opcode == C_LOOP_START)) begin

            skid_instr_info <= '{opcode: loaded_opcode, rs1: loaded_rs1, rs2: loaded_rs2, rstride: loaded_rstride, rd: loaded_rd, imm: loaded_imm, funct1: loaded_funct1, instruction_type: decode_instruction_type};
            skid_valid      <= 1'b1;
        end
        recorded_stall_for_read_rd_flag <= stall_for_read_rd_flag;
        recorded_m_update_waddr         <= m_update_waddr;
        recorded_v_update_waddr         <= v_update_waddr;
        recorded_rd_to_load             <= rd_to_load;
        p1_pipeline_stall               <= pipeline_stall;
        if (start_from_stall) begin
            recorded_assigned_int_op    <= assigned_int_op;
        end

        decode_advanced_d               <= decode_advanced;
        if (!pipeline_stall && (decode_advanced_d || int_op_pending)) begin
            exe_int_op                  <= assigned_int_op;
            int_op_pending              <= 1'b0;
        end else begin
            exe_int_op                  <= STALL_S_INT;
            if (decode_advanced_d && pipeline_stall) begin
                int_op_pending          <= 1'b1;
            end
        end
    end
end

assign recover_from_stall   = !pipeline_stall & p1_pipeline_stall;
assign start_from_stall     = pipeline_stall & !p1_pipeline_stall;

always_comb begin

    if (!start_from_stall & pipeline_stall & recorded_stall_for_read_rd_flag) begin
        stall_for_read_rd   = 1'b1;
    end else if (rd_operand_ready == 1'b0 & (decode_stage_op.v_ele_op != STALL_V_ELEMENT)) begin
        m_update_waddr          = 1'b0;
        v_update_waddr          = 1'b1;
        if (decode_stage_op.v_broadcast_en == 1'b0) begin
            stall_for_read_rd_flag  = 1'b1;
        end else begin
            stall_for_read_rd_flag  = 1'b0;
        end
        rd_to_load              = rd;
    end else begin
        m_update_waddr          = 1'b0;
        v_update_waddr          = 1'b0;
        stall_for_read_rd_flag  = 1'b0;
        rd_to_load              = {INT_OPERAND_WIDTH{1'b0}};
    end

    if (!pipeline_stall & !recorded_stall_for_read_rd_flag) begin
        stall_for_read_rd   = stall_for_read_rd_flag;
        pass_m_update_waddr = m_update_waddr;
        pass_v_update_waddr = v_update_waddr;
        pass_rd_to_load     = rd_to_load;
    end else if (recover_from_stall & recorded_stall_for_read_rd_flag) begin

        stall_for_read_rd   = 1'b1;
        pass_m_update_waddr = recorded_m_update_waddr;
        pass_v_update_waddr = recorded_v_update_waddr;
        pass_rd_to_load     = recorded_rd_to_load;
    end else begin
        stall_for_read_rd   = 1'b0;
        pass_m_update_waddr = 1'b0;
        pass_v_update_waddr = 1'b0;
        pass_rd_to_load     = {INT_OPERAND_WIDTH{1'b0}};
    end
end

logic decode_advanced_d;
logic int_op_pending;
assign decode_advanced = stall_for_read_rd
    || (!effective_pipeline_stall && !scalar_raw_stall && ((!early_loop_end_stall_d1 && !loop_end_stall) || loop_end_valid || loop_jump_back_d1 || skid_injected));

// Fresh-op dispatch condition, registered so decode_op_fresh aligns with decode_stage_op.
wire decode_dispatch_fire = !effective_pipeline_stall && !scalar_raw_stall
    && ((!early_loop_end_stall_d1 && !loop_end_stall) || loop_end_valid || loop_jump_back_d1 || skid_injected);
logic decode_op_fresh;
always_ff @(posedge clk) begin
    if (rst) decode_op_fresh <= 1'b0;
    else     decode_op_fresh <= decode_dispatch_fire;
end
assign decode_op_fresh_o = decode_op_fresh;

always_ff @(posedge clk) begin
    if (stall_for_read_rd) begin

        rd_operand_ready <= 1'b1;
        decode_stage_op.m_op            <= STALL_M;
        decode_stage_op.v_ele_op        <= STALL_V_ELEMENT;
        decode_stage_op.v_reduct_op     <= STALL_V_REDUCT;
        decode_stage_op.s_fp_op         <= STALL_S_FP;
        assigned_int_op                 <= PASS_ADDR_2;
        decode_stage_op.c_op            <= STALL_C;
        decode_stage_op.h_op            <= STALL_H;
        decode_stage_op.m_transposed_read   <= 1'b0;
        decode_stage_op.m_per_head          <= 1'b0;
        decode_stage_op.v_broadcast_en  <= 1'b0;
        decode_stage_op.fps1            <= 'b0;
        decode_stage_op.fps2            <= 'b0;
        decode_stage_op.fpd             <= 'b0;
        decode_stage_op.gp_reg1         <= 'b0;
        decode_stage_op.gp_reg2         <= 'b0;
        decode_stage_op.gp_rd           <= 'b0;
        decode_stage_op.pc_tag          <= '0;
        decode_stage_op.update_m_waddr  <= pass_m_update_waddr;
        decode_stage_op.update_v_waddr  <= pass_v_update_waddr;
        fixed_op_stall_flag             <= 1'b0;
        rs1                             <= 'b0;
        rs2                             <= 'b0;
        rd                              <= pass_rd_to_load;
        imm                             <= 'b0;
    end else if (!effective_pipeline_stall && !scalar_raw_stall && ((!early_loop_end_stall_d1 && !loop_end_stall) || loop_end_valid || loop_jump_back_d1 || skid_injected)) begin

        rd_operand_ready <= 1'b0;
        decode_stage_op.m_transposed_read       <= (decode_instr_info.opcode == M_TMM || decode_instr_info.opcode == M_BTMM) ? 1'b1 : 1'b0;
        // Per-head (batched) matmul family: M_BMM (Q@K, per-head), M_BTMM (Q@K^T,
        // per-head), and their writeout M_BMM_WO. The MCU keeps the mini-array
        // partials separate (one head each) instead of the M_MM cross-K sum.
        decode_stage_op.m_per_head              <= (decode_instr_info.opcode == M_BMM || decode_instr_info.opcode == M_BTMM
                                                 || decode_instr_info.opcode == M_BMM_WO) ? 1'b1 : 1'b0;

        decode_stage_op.v_broadcast_en          <= (decode_instr_info.opcode == V_ADD_VF || decode_instr_info.opcode == V_SUB_VF || decode_instr_info.opcode == V_MUL_VF
                                                 || decode_instr_info.opcode == V_EXP_V  || decode_instr_info.opcode == V_RECI_V) ? 1'b1 : 1'b0;
        decode_stage_op.update_m_waddr          <= 1'b0;
        decode_stage_op.update_v_waddr          <= (decode_instr_info.opcode == V_ADD_VF || decode_instr_info.opcode == V_SUB_VF || decode_instr_info.opcode == V_MUL_VF
                                                 || decode_instr_info.opcode == V_EXP_V  || decode_instr_info.opcode == V_RECI_V) ? 1'b1 : 1'b0;
        decode_stage_op.gp_reg1                 <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
        decode_stage_op.gp_reg2                 <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
        decode_stage_op.gp_rstride              <= decode_instr_info.rstride[INT_OPERAND_WIDTH - 1 : 0];
        decode_stage_op.gp_rd                   <= decode_instr_info.rd [INT_OPERAND_WIDTH - 1 : 0];

        decode_stage_op.pc_tag                  <= pc_reg_d1[9:2];

        case(active_decode_instruction_type)
            M: begin
                decode_stage_op.m_op            <=      (decode_instr_info.opcode == M_MM || decode_instr_info.opcode == M_TMM
                                                      || decode_instr_info.opcode == M_BMM || decode_instr_info.opcode == M_BTMM)       ? MM_IC  :
                                                        (decode_instr_info.opcode == M_MM_WO || decode_instr_info.opcode == M_BMM_WO)   ? MM_WO  :
                                                        (decode_instr_info.opcode == M_MV)                                              ? MV_IC  :
                                                        (decode_instr_info.opcode == M_MV_WO)                                           ? MV_WO  :  STALL_M;
                decode_stage_op.update_m_waddr  <= (decode_instr_info.opcode == M_MM_WO || decode_instr_info.opcode == M_BMM_WO) ? 1'b1 : 1'b0;
                decode_stage_op.v_ele_op      <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op   <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op       <= STALL_S_FP;
                assigned_int_op               <= (decode_instr_info.opcode == M_MM_WO || decode_instr_info.opcode == M_BMM_WO) ? PASS_ADDR_2 : PASS_ADDR;
                decode_stage_op.c_op          <= STALL_C;
                decode_stage_op.h_op          <= STALL_H;
                decode_stage_op.fps1          <= 'b0;
                decode_stage_op.fps2          <= 'b0;
                decode_stage_op.fpd           <= 'b0;
                rs1                           <= decode_instr_info.rs1  [INT_OPERAND_WIDTH - 1 : 0];
                rs2                           <= decode_instr_info.rs2  [INT_OPERAND_WIDTH - 1 : 0];
                rd                            <= decode_instr_info.rd   [INT_OPERAND_WIDTH - 1 : 0];
                imm     <= 'b0;
            end

            V: begin
                decode_stage_op.m_op     <= STALL_M;
                decode_stage_op.v_ele_op <=
                    (decode_instr_info.opcode == V_ADD_VV || decode_instr_info.opcode == V_ADD_VF) ? ADD_V_ELEMENT  :

                    ((decode_instr_info.opcode == V_SUB_VV || decode_instr_info.opcode == V_SUB_VF) && decode_instr_info.funct1 == 4'h1) ? RSUB_V_ELEMENT :
                    (decode_instr_info.opcode == V_SUB_VV || decode_instr_info.opcode == V_SUB_VF) ? SUB_V_ELEMENT  :
                    (decode_instr_info.opcode == V_MUL_VV || decode_instr_info.opcode == V_MUL_VF) ? MUL_V_ELEMENT  :
                    (decode_instr_info.opcode == V_EXP_V)                                          ? EXP_V_ELEMENT  :
                    (decode_instr_info.opcode == V_RECI_V)                                         ? RECI_V_ELEMENT :
                    (decode_instr_info.opcode == C_HADAMARD_TRANSFORM)                             ? INNER_HADAMARD_TRANSFORM :
                    (decode_instr_info.opcode == V_PS_V)                                           ? PREFIX_SCAN_V_ELEMENT :
                    (decode_instr_info.opcode == V_SHFT_V)                                         ? SHIFT_V_LANES_ELEMENT : STALL_V_ELEMENT;

                decode_stage_op.v_reduct_op <=      (decode_instr_info.opcode == V_RED_SUM)   ? SUM_V_REDUCT :
                                                    (decode_instr_info.opcode == V_RED_MAX)   ? MAX_V_REDUCT : STALL_V_REDUCT;

                assigned_int_op                         <= PASS_ADDR;
                decode_stage_op.c_op                    <= STALL_C;
                decode_stage_op.h_op                    <= STALL_H;
                if (decode_instr_info.opcode == V_ADD_VF || decode_instr_info.opcode == V_SUB_VF || decode_instr_info.opcode == V_MUL_VF) begin
                    decode_stage_op.s_fp_op <= LD_OUT_FP;
                    decode_stage_op.fps1    <= 'b0;
                    decode_stage_op.fps2    <= decode_instr_info.rs2[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fpd     <= 'b0;
                    rs1                     <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];

                    rs2                     <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    rd                      <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm                     <= {IMM_WIDTH{1'b0}};
                end else if (decode_instr_info.opcode == V_SHFT_V) begin

                    decode_stage_op.s_fp_op <= STALL_S_FP;
                    decode_stage_op.fps1    <= 'b0;
                    decode_stage_op.fps2    <= decode_instr_info.rs2[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fpd     <= 'b0;
                    rs1                     <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    // rs2 carries the SHIFT AMOUNT (gp<rs2> value): with assigned_int_op
                    // PASS_ADDR this reads gp[rs2] into addr_2, which reaches the vector
                    // machine as result_waddr and is used as shift_amount. (The V_SHFT_V
                    // write target is gp<rd> via addr_3, so addr_2 is free to carry it.)
                    rs2                     <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                    rd                      <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm                     <= {IMM_WIDTH{1'b0}};
                end else if (decode_instr_info.opcode == V_RED_SUM || decode_instr_info.opcode == V_RED_MAX) begin
                    decode_stage_op.s_fp_op             <= LD_OUT_FP;
                    decode_stage_op.fps1                <= 'b0;
                    decode_stage_op.fps2                <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fpd                 <= 'b0;
                    rs1                                 <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2                                 <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                    rd                                  <= {FP_OPERAND_WIDTH{1'b0}};
                    imm                                 <= {IMM_WIDTH{1'b0}};
                end else if (decode_instr_info.opcode == V_EXP_V || decode_instr_info.opcode == V_RECI_V) begin

                    decode_stage_op.s_fp_op             <= STALL_S_FP;
                    decode_stage_op.fps1                <= 'b0;
                    decode_stage_op.fps2                <= 'b0;
                    decode_stage_op.fpd                 <= 'b0;
                    rs1                                 <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2                                 <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    rd                                  <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm                                 <= {IMM_WIDTH{1'b0}};
                end else begin
                    decode_stage_op.s_fp_op             <= STALL_S_FP;
                    decode_stage_op.fps1                <= 'b0;
                    decode_stage_op.fps2                <= 'b0;
                    decode_stage_op.fpd                 <= 'b0;
                    rs1                                 <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2                                 <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                    rd                                  <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm                                 <= {IMM_WIDTH{1'b0}};
                end
            end

            S_INT: begin
                decode_stage_op.m_op                <= STALL_M;
                decode_stage_op.v_ele_op            <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op         <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op             <= STALL_S_FP;
                assigned_int_op                   <=    (decode_instr_info.opcode == S_ADD_INT)   ? ADD_INT   :
                                                        (decode_instr_info.opcode == S_ADDI_INT)  ? ADDI_INT  :
                                                        (decode_instr_info.opcode == S_SUB_INT)   ? SUB_INT   :
                                                        (decode_instr_info.opcode == S_MUL_INT)   ? MUL_INT   :
                                                        (decode_instr_info.opcode == S_LUI_INT)   ? LUI_INT   :
                                                        (decode_instr_info.opcode == S_LD_INT)    ? LD_INT    :
                                                        (decode_instr_info.opcode == S_ST_INT)    ? ST_INT    :  STALL_S_INT;
                decode_stage_op.c_op                <= STALL_C;
                decode_stage_op.h_op                <= STALL_H;
                if (decode_instr_info.opcode == S_ADDI_INT) begin

                    decode_stage_op.fps1            <= 'b0;
                    decode_stage_op.fps2            <= 'b0;
                    decode_stage_op.fpd             <= 'b0;
                    rs1             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd              <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm             <= {{IMM_WIDTH - IMM_2_WIDTH {1'b0}}, decode_instr_info.imm[IMM_2_WIDTH:0]};
                end else begin

                    decode_stage_op.fps1            <= 'b0;
                    decode_stage_op.fps2            <= 'b0;
                    decode_stage_op.fpd             <= 'b0;
                    rs1             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2             <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                    rd              <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm             <= decode_instr_info.imm;
                end
            end

            S_FP: begin
                decode_stage_op.m_op            <= STALL_M;
                decode_stage_op.v_ele_op        <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op     <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op         <=      (decode_instr_info.opcode == S_ADD_FP )   ? ADD_FP    :
                                                        (decode_instr_info.opcode == S_SUB_FP )   ? SUB_FP    :
                                                        (decode_instr_info.opcode == S_MAX_FP )   ? MAX_FP    :
                                                        (decode_instr_info.opcode == S_MUL_FP )   ? MUL_FP    :
                                                        (decode_instr_info.opcode == S_EXP_FP )   ? EXP_FP    :
                                                        (decode_instr_info.opcode == S_RECI_FP)   ? RECI_FP   :
                                                        (decode_instr_info.opcode == S_SQRT_FP)   ? SQRT_FP   :
                                                        (decode_instr_info.opcode == S_LD_FP)     ? LD_REG_FP :
                                                        (decode_instr_info.opcode == S_ST_FP)     ? ST_REG_FP :
                                                        (decode_instr_info.opcode == S_MAP_V_FP)  ? MAP_V_FP  : STALL_S_FP;

                decode_stage_op.c_op              <= STALL_C;
                decode_stage_op.h_op              <= STALL_H;
                if (decode_instr_info.opcode == S_ADD_FP || decode_instr_info.opcode == S_SUB_FP || decode_instr_info.opcode == S_MAX_FP || decode_instr_info.opcode == S_MUL_FP) begin

                    assigned_int_op               <= STALL_S_INT;
                    decode_stage_op.fps1            <= decode_instr_info.rs1[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fps2            <= decode_instr_info.rs2[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fpd             <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    rs1                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rs2                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd                              <= {INT_OPERAND_WIDTH{1'b0}};
                    imm                             <= {IMM_WIDTH{1'b0}};
                end else if (decode_instr_info.opcode == S_EXP_FP || decode_instr_info.opcode == S_RECI_FP || decode_instr_info.opcode == S_SQRT_FP) begin

                    assigned_int_op               <= STALL_S_INT;
                    decode_stage_op.fps1            <= decode_instr_info.rs1[FP_OPERAND_WIDTH - 1 : 0];
                    decode_stage_op.fps2            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fpd             <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    rs1                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rs2                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd                              <= {INT_OPERAND_WIDTH{1'b0}};
                    imm                             <= {IMM_WIDTH{1'b0}};
                end else if (decode_instr_info.opcode == S_LD_FP || decode_instr_info.opcode == S_ST_FP) begin

                    assigned_int_op               <= COMP_ADDR;
                    decode_stage_op.fps1            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fps2            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fpd             <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    rs1                             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd                              <= {INT_OPERAND_WIDTH{1'b0}};
                    imm                             <= decode_instr_info.imm;
                end else if (decode_instr_info.opcode == S_MAP_V_FP) begin
                    assigned_int_op               <= COMP_ADDR_2;
                    decode_stage_op.fps1            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fps2            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fpd             <= {FP_OPERAND_WIDTH{1'b0}};
                    rs1                             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd                              <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    imm                             <= decode_instr_info.imm;
                end else begin

                    assigned_int_op               <= STALL_S_INT;
                    decode_stage_op.fps1            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fps2            <= {FP_OPERAND_WIDTH{1'b0}};
                    decode_stage_op.fpd             <= decode_instr_info.rd[FP_OPERAND_WIDTH - 1 : 0];
                    rs1                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rs2                             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd                              <= {INT_OPERAND_WIDTH{1'b0}};
                    imm                             <= {IMM_WIDTH{1'b0}};
                end
            end

            C : begin
                decode_stage_op.m_op            <= STALL_M;
                decode_stage_op.v_ele_op        <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op     <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op         <= STALL_S_FP;
                decode_stage_op.h_op            <= STALL_H;

                if(decode_instr_info.opcode == C_SET_ADDR_REG) begin
                    assigned_int_op                   <= PASS_ADDR;
                    decode_stage_op.c_op              <= SET_ADDR_REG;
                end else if (decode_instr_info.opcode == C_SET_SCALE_REG) begin
                    assigned_int_op                   <= PASS_ADDR_2;
                    decode_stage_op.c_op              <= SET_SCALE_REG;
                end else if (decode_instr_info.opcode == C_SET_STRIDE_REG) begin
                    assigned_int_op                   <= PASS_ADDR_2;
                    decode_stage_op.c_op              <= SET_STRIDE_SIZE;
                end else if (decode_instr_info.opcode == C_BREAK) begin
                    assigned_int_op                     <= STALL_S_INT;
                    decode_stage_op.c_op                <= BREAK;
                end else if (decode_instr_info.opcode == C_LOOP_START) begin

                    assigned_int_op                     <= LOOP_INIT;
                    decode_stage_op.c_op                <= LOOP_START;
                end else if (decode_instr_info.opcode == C_LOOP_END) begin

                    assigned_int_op                     <= LOOP_DEC;
                    decode_stage_op.c_op                <= LOOP_END;
                end else begin
                    assigned_int_op                     <= STALL_S_INT;
                    decode_stage_op.c_op                <= STALL_C;
                end
                decode_stage_op.fps1              <= 'b0;
                decode_stage_op.fps2              <= 'b0;
                decode_stage_op.fpd               <= 'b0;

                if (decode_instr_info.opcode == C_LOOP_START) begin

                    rs1             <= {INT_OPERAND_WIDTH{1'b0}};
                    rs2             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd              <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm             <= decode_instr_info.imm;
                end else if (decode_instr_info.opcode == C_LOOP_END) begin

                    rs1             <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    rs2             <= {INT_OPERAND_WIDTH{1'b0}};
                    rd              <= decode_instr_info.rd[INT_OPERAND_WIDTH - 1 : 0];
                    imm             <= {IMM_WIDTH{1'b0}};
                end else begin
                    rs1             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                    rs2             <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                    rd              <= decode_instr_info.rd [INT_OPERAND_WIDTH - 1 : 0];
                    imm             <= {IMM_WIDTH{1'b0}};
                end
            end

            H : begin
                decode_stage_op.m_op            <= STALL_M;
                decode_stage_op.v_ele_op        <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op     <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op         <= STALL_S_FP;
                assigned_int_op                 <= PASS_ADDR_2;
                decode_stage_op.c_op            <= STALL_C;
                if (decode_instr_info.opcode == H_PREFETCH_M) begin
                    if (decode_instr_info.funct1 == 4'h0) begin
                        decode_stage_op.h_op <= PREFETCH_M_H;
                    end else if (decode_instr_info.funct1 == 4'h1) begin
                        decode_stage_op.h_op <= PREFETCH_M_L;
                    end else begin
                        decode_stage_op.h_op <= STALL_H;
                    end
                end else if (decode_instr_info.opcode == H_PREFETCH_V) begin
                    if (decode_instr_info.funct1 == 4'h0) begin
                        decode_stage_op.h_op <= PREFETCH_V_H;
                    end else if (decode_instr_info.funct1 == 4'h1) begin
                        decode_stage_op.h_op <= PREFETCH_V_L;
                    end else begin
                        decode_stage_op.h_op <= STALL_H;
                    end
                end else if (decode_instr_info.opcode == H_STORE_V) begin
                    if (decode_instr_info.funct1 == 4'h0) begin
                        decode_stage_op.h_op <= STORE_V_H;
                    end else if (decode_instr_info.funct1 == 4'h1) begin
                        decode_stage_op.h_op <= STORE_V_L;
                    end else begin
                        decode_stage_op.h_op <= STALL_H;
                    end
                end else begin
                    decode_stage_op.h_op          <= STALL_H;
                end
                decode_stage_op.fps1              <= 'b0;
                decode_stage_op.fps2              <= 'b0;
                decode_stage_op.fpd               <= 'b0;
                rs1             <= decode_instr_info.rs1[INT_OPERAND_WIDTH - 1 : 0];
                rs2             <= decode_instr_info.rs2[INT_OPERAND_WIDTH - 1 : 0];
                rd              <= decode_instr_info.rd [INT_OPERAND_WIDTH - 1 : 0];
                imm             <= {IMM_WIDTH{1'b0}};
            end

            default: begin
                decode_stage_op.m_op              <= STALL_M;
                decode_stage_op.v_ele_op          <= STALL_V_ELEMENT;
                decode_stage_op.v_reduct_op       <= STALL_V_REDUCT;
                decode_stage_op.s_fp_op           <= STALL_S_FP;
                assigned_int_op                   <= STALL_S_INT;
                decode_stage_op.c_op              <= STALL_C;
                decode_stage_op.h_op              <= STALL_H;
                decode_stage_op.fps1              <= 'b0;
                decode_stage_op.fps2              <= 'b0;
                decode_stage_op.fpd               <= 'b0;
                rs1             <= 'b0;
                rs2             <= 'b0;
                rd              <= 'b0;
                imm             <= {IMM_WIDTH{1'b0}};
            end
        endcase
    end else begin

        assigned_int_op     <= STALL_S_INT;
        decode_stage_op     <= decode_stage_op;
        rs1                 <= rs1;
        rs2                 <= rs2;
        rd                  <= rd;
        imm                 <= imm;
    end
end

`ifdef SIMULATION

assign c_break_detected = decode_instr_valid && (decode_instr_info.opcode == C_BREAK);
`endif

endmodule
