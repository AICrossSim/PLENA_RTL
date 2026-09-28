`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

module addr_monitor#(
    parameter   ADDR_WIDTH           = 5,
    parameter   PIPELINE_STAGES      = 4
) (
    input   logic clk,
    input   logic rst,

    input   OP_BUNDLE determine_stage_op,

    input   logic [ADDR_WIDTH - 1 : 0] v_sram_addr_a,
    input   logic [ADDR_WIDTH - 1 : 0] v_sram_addr_b,
    input   logic v_sram_wen_a,
    input   logic v_sram_wen_b,

    output  logic stall_req,
    input   logic sys_pipe_stall
);
    /* verilator no_inline_module */

    logic [ADDR_WIDTH - 1 : 0] v_sram_addr_a_r, v_sram_addr_b_r;
    logic v_sram_wen_a_r, v_sram_wen_b_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            v_sram_addr_a_r <= '0;
            v_sram_addr_b_r <= '0;
            v_sram_wen_a_r  <= 1'b0;
            v_sram_wen_b_r  <= 1'b0;
        end else begin
            v_sram_addr_a_r <= v_sram_addr_a;
            v_sram_addr_b_r <= v_sram_addr_b;
            v_sram_wen_a_r  <= v_sram_wen_a;
            v_sram_wen_b_r  <= v_sram_wen_b;
        end
    end

    typedef struct {
        logic [ADDR_WIDTH-1:0]          track_addr;
        logic                           activate;
        logic [7:0]                     owner_pc;
    } TRACK_ADDR;

    localparam TRACK_ADDR_WIDTH = $clog2(PIPELINE_STAGES) + 1;
    TRACK_ADDR v_write_addr_track [PIPELINE_STAGES - 1 : 0];
    logic [TRACK_ADDR_WIDTH - 1 : 0]     free_track_entry_idx;
    logic [ADDR_WIDTH - 1 : 0] locked_entry_1, locked_entry_2;
    logic lock_entry_1_valid, lock_entry_2_valid;
    logic found_invalid;
    logic pipe_full;

    always_comb begin
        found_invalid           = 1'b0;
        free_track_entry_idx    = '0;
        for (int i = 0; i < PIPELINE_STAGES; i++) begin
            if (!v_write_addr_track[i].activate && !found_invalid) begin
                free_track_entry_idx    = i;
                found_invalid           = 1'b1;
            end
        end
        pipe_full = !found_invalid;
    end

    logic [PIPELINE_STAGES - 1 : 0] addr_collide_flag;
    logic [ADDR_WIDTH - 1 : 0] locked_entry_1_next, locked_entry_2_next;
    logic lock_entry_1_valid_next, lock_entry_2_valid_next;
    logic stall_req_next;
    logic stall_req_q;

    assign stall_req = stall_req_q;

    always_comb begin
        addr_collide_flag      = '0;
        locked_entry_1_next    = locked_entry_1;
        locked_entry_2_next    = locked_entry_2;
        lock_entry_1_valid_next = lock_entry_1_valid;
        lock_entry_2_valid_next = lock_entry_2_valid;
        stall_req_next         = 1'b0;

        if (stall_req_q) begin

            for (int i = 0; i < PIPELINE_STAGES; i++) begin
                if (((v_write_addr_track[i].track_addr == locked_entry_1)) & (v_write_addr_track[i].activate == 1'b1) & lock_entry_1_valid) begin
                    addr_collide_flag[i] = 1'b1;
                end else if (((v_write_addr_track[i].track_addr == locked_entry_2)) & (v_write_addr_track[i].activate == 1'b1) & lock_entry_2_valid) begin
                    addr_collide_flag[i] = 1'b1;
                end else begin
                    addr_collide_flag[i] = 1'b0;
                end
            end
            stall_req_next = |addr_collide_flag;
            if (!stall_req_next) begin
                locked_entry_1_next     = '0;
                locked_entry_2_next     = '0;
                lock_entry_1_valid_next = 1'b0;
                lock_entry_2_valid_next = 1'b0;
            end

        end else if (!sys_pipe_stall) begin

            if ((determine_stage_op.m_op != STALL_M & determine_stage_op.m_op != MM_WO & determine_stage_op.m_op != MM_IC)) begin

                for (int i = 0; i < PIPELINE_STAGES; i++) begin
                    if ((v_write_addr_track[i].track_addr == determine_stage_op.addr_1) & (v_write_addr_track[i].activate == 1'b1) & (v_write_addr_track[i].owner_pc != determine_stage_op.pc_tag)) begin
                        addr_collide_flag[i] = 1'b1;
                        locked_entry_1_next = determine_stage_op.addr_1;
                        lock_entry_1_valid_next = 1'b1;
                    end else if ((v_write_addr_track[i].track_addr == determine_stage_op.addr_2) & (v_write_addr_track[i].activate == 1'b1) & (v_write_addr_track[i].owner_pc != determine_stage_op.pc_tag)) begin
                        addr_collide_flag[i] = 1'b1;
                        locked_entry_2_next = determine_stage_op.addr_2;
                        lock_entry_2_valid_next = 1'b1;
                    end else begin
                        addr_collide_flag[i] = 1'b0;
                    end
                end
                stall_req_next = |addr_collide_flag;
            end else if (determine_stage_op.v_reduct_op != STALL_V_REDUCT) begin

                for (int i = 0; i < PIPELINE_STAGES; i++) begin
                    if (((v_write_addr_track[i].track_addr == determine_stage_op.addr_1)) & (v_write_addr_track[i].activate == 1'b1) & (v_write_addr_track[i].owner_pc != determine_stage_op.pc_tag)) begin
                        addr_collide_flag[i] = 1'b1;
                        locked_entry_1_next = determine_stage_op.addr_1;
                        lock_entry_1_valid_next = 1'b1;
                    end else begin
                        addr_collide_flag[i] = 1'b0;
                    end
                end
                stall_req_next = |addr_collide_flag;
            end else if (determine_stage_op.h_op == STORE_V_H ||determine_stage_op.h_op == STORE_V_H ) begin

                for (int i = 0; i < PIPELINE_STAGES; i++) begin
                    if (((v_write_addr_track[i].track_addr == determine_stage_op.addr_2)) & (v_write_addr_track[i].activate == 1'b1) & (v_write_addr_track[i].owner_pc != determine_stage_op.pc_tag)) begin
                        addr_collide_flag[i] = 1'b1;
                        locked_entry_2_next = determine_stage_op.addr_2;
                        lock_entry_2_valid_next = 1'b1;
                    end else begin
                        addr_collide_flag[i] = 1'b0;
                    end
                end
                stall_req_next = |addr_collide_flag;
            end else begin
                locked_entry_1_next = '0;
                locked_entry_2_next = '0;
                lock_entry_1_valid_next = 1'b0;
                lock_entry_2_valid_next = 1'b0;
            end
        end else begin

            locked_entry_1_next = '0;
            locked_entry_2_next = '0;
            lock_entry_1_valid_next = 1'b0;
            lock_entry_2_valid_next = 1'b0;
        end

    end

    logic   [ADDR_WIDTH - 1 : 0]    insert_addr;
    logic                           insert_valid;
    logic   [TRACK_ADDR_WIDTH-1:0]  matched_track_entry_idx;
    logic                           matched_waddr;

    logic                           insert_pc_already_tracked;

    always_comb begin
        if (determine_stage_op.h_op == PREFETCH_V_H) begin
            insert_addr  = determine_stage_op.addr_2;
            insert_valid = 1'b1;
        end else if (determine_stage_op.update_m_waddr) begin
            insert_addr  = determine_stage_op.addr_2;
            insert_valid = 1'b1;

        end else if (determine_stage_op.s_fp_op == MAP_V_FP) begin
            insert_addr  = determine_stage_op.addr_2;
            insert_valid = 1'b1;
        end else begin
            insert_addr  = {ADDR_WIDTH{1'b0}};
            insert_valid = 1'b0;
        end

        matched_waddr                 = 1'b0;
        matched_track_entry_idx       = '0;
        for (int i = 0; i < PIPELINE_STAGES; i++) begin
            if (((v_sram_wen_a_r && v_write_addr_track[i].track_addr == v_sram_addr_a_r) ||
                    (v_sram_wen_b_r && v_write_addr_track[i].track_addr == v_sram_addr_b_r)) & !matched_waddr) begin
                matched_track_entry_idx     = i;
                matched_waddr                 = 1'b1;
            end
        end

        insert_pc_already_tracked = 1'b0;
        for (int i = 0; i < PIPELINE_STAGES; i++) begin
            if (v_write_addr_track[i].activate && (v_write_addr_track[i].owner_pc == determine_stage_op.pc_tag)) begin
                insert_pc_already_tracked = 1'b1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < PIPELINE_STAGES; i++) begin
                v_write_addr_track[i] <= '{
                    track_addr : {ADDR_WIDTH{1'b0}},
                    activate   : 1'b0,
                    owner_pc   : 8'b0
                };
            end
            locked_entry_1 <= '0;
            locked_entry_2 <= '0;
            lock_entry_1_valid <= 1'b0;
            lock_entry_2_valid <= 1'b0;
            stall_req_q <= 1'b0;
        end else begin
            locked_entry_1 <= locked_entry_1_next;
            locked_entry_2 <= locked_entry_2_next;
            lock_entry_1_valid <= lock_entry_1_valid_next;
            lock_entry_2_valid <= lock_entry_2_valid_next;
            stall_req_q <= stall_req_next;

            if (pipe_full == 1'b0 & insert_valid & !insert_pc_already_tracked) begin
                v_write_addr_track[free_track_entry_idx] <= '{
                    track_addr : insert_addr,
                    activate   : 1'b1,
                    owner_pc   : determine_stage_op.pc_tag
                };
            end

            if (matched_waddr) begin
                v_write_addr_track[matched_track_entry_idx] <= '{
                    track_addr : {ADDR_WIDTH{1'b0}},
                    activate   : 1'b0,
                    owner_pc   : 8'b0
                };
            end
        end
    end

endmodule
