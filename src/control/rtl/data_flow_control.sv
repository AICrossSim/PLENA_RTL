`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Data Flow Control
Description : Memory controller for matrix/scratch/scalar SRAM and HBM interface.
*/

module data_flow_control import precision_pkg::*; import configuration_pkg::*; #(
    localparam MATRIX_LOAD_ITERATION_GEMM   = BLEN,
    localparam MATRIX_LOAD_ITERATION_GEMV   = MLEN,
    localparam BLOCK_NUM                    = VLEN / BLOCK_DIM
) (

    input       logic clk,
    input       logic rst,

    // Current Execution
    input       OP_BUNDLE                       exe_stage_op,
    input       MEM_WEN_INFO                    mem_write_control,
    output      MEM_WREQ_INFO                   write_req,

    // Interface with Matrix Machine
    output      logic m_m_valid,
    output      logic m_v_valid,
    input       logic m_out_valid,
    output      logic m_out_ready,
    output      logic m_load_in_process,
    input       logic [1:0] m_write_request,
    input       logic [INT_DATA_WIDTH - 1 : 0] m_write_addr,

    // Interface with Matrix SRAM (pipelined outputs for timing)
    output      logic [INT_DATA_WIDTH - 1 : 0] m_sram_raddr,
    output      logic [INT_DATA_WIDTH - 1 : 0] m_sram_waddr,
    output      logic m_sram_wen,
    output      logic m_sram_req,
    output      logic m_sram_transposed_read,
    input       logic m_prefetch_data_not_ready,

    // Interface with Vector Machine
    output      logic v_v_a_valid,
    output      logic v_v_b_valid,
    output      logic v_s_in_valid,

    input       logic v_write_request,
    input       logic [INT_DATA_WIDTH - 1 : 0]    v_write_addr,

    // Interface with Scalar Machine
    input       logic s_map_v_valid,

    // Interface with Vector SRAM
    output      logic v_sram_req_a,
    output      logic v_sram_wen_a,
    output      logic [INT_DATA_WIDTH - 1 : 0]      v_sram_addr_a,
    output      logic [VLEN-1:0]                    v_sram_mask_a,
    output      logic select_write_data_a,          

    output      logic v_sram_req_b,
    output      logic [1:0] v_sram_mxfp_req_b,
    output      logic v_sram_wen_b,
    output      logic [INT_DATA_WIDTH - 1 : 0]      v_sram_addr_b,
    output      logic [VLEN-1:0]                    v_sram_mask_b,
    output      logic [1:0] select_write_data_b,
    input       logic v_prefetch_data_not_ready,
    output      logic continuous_write_to_v_sram_port_b,

    // Interface with HBM
    input       logic prefetch_m_valid,
    input       logic prefetch_v_valid,
    input       logic hbm_ready_to_write,
    // hbm_m_req_prefetch_data removed - matrix prefetch uses simple pipeline without handshake
    output      logic hbm_v_req_prefetch_data,

    // Status Tracking - covers entire prefetch process including SRAM writes
    output      logic hbm_m_prefetch_in_progress,
    output      logic hbm_v_prefetch_in_progress
);
    // Package Imports
    import pipeline_pkg::MAX_PIPELINE_STAGE;
    import configuration_pkg::*;

    localparam M_LD_COUNT_WIDTH = $clog2(MATRIX_LOAD_ITERATION_GEMV);
    localparam M_PF_COUNT_WIDTH = $clog2(HBM_M_Prefetch_Amount + 1);
    localparam V_PF_COUNT_WIDTH = $clog2(HBM_V_Prefetch_Amount + 1);
    localparam HBM_WRITE_AMOUNT = HBM_V_Writeback_Amount;
    localparam H_WR_COUNT_WIDTH = $clog2(HBM_WRITE_AMOUNT);


    // Memory Execution Control and Dependency Monitor
    OP_BUNDLE  mem_stage_op;
    MEM_WEN_INFO mem_stage_write_control;

    // Pipeline registers for matrix SRAM address (breaks critical path to data_not_ready)
    logic [INT_DATA_WIDTH - 1 : 0] m_sram_raddr_comb;
    logic [INT_DATA_WIDTH - 1 : 0] m_sram_waddr_comb;
    // Pre-pipeline control signals (will be registered to align with address)
    logic m_sram_req_pre;
    logic m_sram_transposed_read_pre;

    always_ff @(posedge clk ) begin
        mem_stage_op            <= exe_stage_op;
        mem_stage_write_control <= mem_write_control;
    end

    // Stall Request
    logic previous_dma_m_ready;
    logic previous_dma_v_ready;
    logic prefetch_m_valid_d;   // valid delayed to align with the SRAM write-data stage
    always_ff @(posedge clk) begin
        if (rst) begin
            previous_dma_m_ready <= 1'b0;
            previous_dma_v_ready <= 1'b0;
            prefetch_m_valid_d   <= 1'b0;
        end else begin
            previous_dma_m_ready <= prefetch_m_valid;
            previous_dma_v_ready <= prefetch_v_valid;
            prefetch_m_valid_d   <= prefetch_m_valid;
        end
    end

    //Request Asserted in single cycle
    always_comb begin
        write_req.wreq_m_sram        = ((prefetch_m_valid == 1'b1) & (previous_dma_m_ready == 1'b0))    ? 1'b1 : 1'b0;
        write_req.wreq_s_sram_port_a = ((m_write_request  == 1'b1) | (v_write_request == 1'b1))         ? 1'b1 : 1'b0;
        write_req.wreq_s_sram_port_b = ((prefetch_v_valid == 1'b1) & (previous_dma_v_ready == 1'b0))    ? 1'b1 : 1'b0;
        write_req.wreq_from_m        = m_write_request;
    end

    
    // -----------------------------
    // Matrix SRAM
    // -----------------------------
    logic [INT_DATA_WIDTH - 1 : 0] recorded_m_prefetch_addr, recorded_m_load_addr;

    // FIFO for queuing prefetch addresses (handles consecutive PREFETCH_M instructions)
    logic [INT_DATA_WIDTH - 1 : 0] m_prefetch_addr_fifo_in, m_prefetch_addr_fifo_out;
    logic m_prefetch_addr_fifo_in_valid, m_prefetch_addr_fifo_in_ready;
    logic m_prefetch_addr_fifo_out_valid, m_prefetch_addr_fifo_out_ready;
    logic m_prefetch_addr_fifo_empty, m_prefetch_addr_fifo_full;

    fifo #(
        .DATA_WIDTH(INT_DATA_WIDTH),
        .DEPTH(32)  // Allow up to 4 queued prefetch addresses
    ) m_prefetch_addr_fifo (
        .clk(clk),
        .rst(rst),
        .data_in(m_prefetch_addr_fifo_in),
        .data_in_valid(m_prefetch_addr_fifo_in_valid),
        .data_in_ready(m_prefetch_addr_fifo_in_ready),
        .data_out(m_prefetch_addr_fifo_out),
        .data_out_valid(m_prefetch_addr_fifo_out_valid),
        .data_out_ready(m_prefetch_addr_fifo_out_ready),
        .empty(m_prefetch_addr_fifo_empty),
        .full(m_prefetch_addr_fifo_full)
    );

    // Push address to FIFO when PREFETCH_M instruction arrives
    assign m_prefetch_addr_fifo_in = exe_stage_op.addr_2;
    assign m_prefetch_addr_fifo_in_valid = (exe_stage_op.h_op == PREFETCH_M_H || exe_stage_op.h_op == PREFETCH_M_L);

    // Pop address from FIFO when starting a new prefetch operation
    assign m_prefetch_addr_fifo_out_ready = prefetch_m_valid && !continuous_prefetch_m_en && m_sram_prefetch_counter == 0 && m_prefetch_addr_fifo_out_valid;
    logic [INT_DATA_WIDTH - 1 : 0] m_sram_raddr_offset;
    logic continuous_load_m_en, continuous_prefetch_m_en;
    logic [M_LD_COUNT_WIDTH : 0] m_sram_load_counter, load_m_amount;
    logic [M_PF_COUNT_WIDTH : 0] m_sram_prefetch_counter;
    logic m_m_load, m_v_load;
    logic p2_m_m_load, p1_m_m_load, m_v_load_cond;
    logic matrix_related_data_ready;
    assign matrix_related_data_ready = (!m_prefetch_data_not_ready) & (!v_prefetch_data_not_ready);
    logic next_clk_m_m_valid;

    // Result-writeback interlock: hold decoder stall until write burst completes.
    logic m_result_write_pending;
    logic continuous_v_write_from_matrix_en_q;


    // Update addr only when the exe operation is MV_IC or MV_WO
    // Compute addresses combinationally, then pipeline them
    always_comb begin
        if (continuous_load_m_en) begin
            m_sram_raddr_offset = m_sram_load_counter * MLEN;
        end else begin
            m_sram_raddr_offset = 'b0;
        end

        if ((mem_stage_op.m_op != STALL_M & mem_stage_op.m_op != MM_WO)|| continuous_load_m_en) begin
            m_sram_raddr_comb = recorded_m_load_addr + m_sram_raddr_offset;
        end else begin
            m_sram_raddr_comb = 'b0;
        end

        if (continuous_prefetch_m_en) begin
            m_sram_waddr_comb = recorded_m_prefetch_addr + m_sram_prefetch_counter * MLEN;
        end else begin
            m_sram_waddr_comb = 'b0;
        end
    end

    // Pipeline register for matrix SRAM addresses and control signals (breaks critical path)
    always_ff @(posedge clk) begin
        if (rst) begin
            m_sram_raddr <= 'b0;
            m_sram_waddr <= 'b0;
            m_sram_req   <= 1'b0;
            m_sram_wen   <= 1'b0;
            m_sram_transposed_read <= 1'b0;
        end else begin
            m_sram_raddr <= m_sram_raddr_comb;
            m_sram_waddr <= m_sram_waddr_comb;
            m_sram_req   <= m_sram_req_pre || (continuous_prefetch_m_en && prefetch_m_valid_d);
            // Gate prefetch write on 1-cycle-delayed HBM data valid.
            m_sram_wen   <= continuous_prefetch_m_en && prefetch_m_valid_d;
            m_sram_transposed_read <= m_sram_transposed_read_pre;
        end
    end

    // Read  Port -> Matrix Weight Load
    // Write Port -> Matrix Weight Prefetch

    logic end_of_load_m;
    logic permit_load_for_m;

    // Cold-start primer (one-time): the program's FIRST matmul hits a COLD weight-read
    // pipeline (matrix_sram rd_data_valid + element_out reg, then matrix_machine's
    // m_element_d + register_slice), so the counter-derived m_m_valid — calibrated for
    // the WARM steady state — leads the weight DATA and the systolic MCU latches ZERO
    // weights for the first BLENxBLEN tile. The cold weight-read datapath additionally
    // fills its MLEN-wide row SEQUENTIALLY (one KLEN group every few cycles) — the whole
    // 16-element row only settles ~20 cycles after the read is first issued, so the upper
    // K-segments would otherwise read as zero and their systolic partials come out garbage.
    // Fix: hold the first permit/valid off (referenced to the load START, NOT to
    // matrix_related_data_ready which is already high), so the read keeps spinning at
    // counter 0 and warms ALL sub-bank groups before it advances. A sticky
    // `matrix_pipe_warm` flag makes this a strict one-time event: once the first load is
    // primed the flag is set forever, so every subsequent matmul is byte-identical to
    // before this change (zero steady-state regression).
    localparam int WEIGHT_READ_PIPE_DEPTH = 4;
    localparam int COLD_START_PRIME_DEPTH = 2 * MLEN;
    // The activation (vector) SRAM read + FP->MXINT requantize path is LONGER than the
    // weight path, so on a warm matmul the vector read address naturally LEADS the weight
    // address. On cold start (both pipes empty) the vector address must therefore be
    // released a couple of cycles BEFORE the weight so the activation DATA arrives at the
    // MCU aligned with the (primed) weight beat. Release the vector read this many cycles
    // ahead of the weight permit/valid.
    localparam COLD_START_VEC_LEAD        = 5'd1;
    localparam COLD_START_VEC_PRIME_DEPTH = COLD_START_PRIME_DEPTH - COLD_START_VEC_LEAD;
    logic       matrix_pipe_warm;
    logic [$clog2(COLD_START_PRIME_DEPTH+1)-1:0] cold_start_primer_cnt;
    logic       cold_start_ok;
    logic       cold_start_vec_ok;
    // Idle-gap tracker: cycles the matrix weight-read pipe has been inactive (no m_m_load).
    // When a new matmul starts after > PIPE_COLD_THRESHOLD idle cycles the read pipe has
    // drained cold (e.g. softmax/vector ops ran between QK^T and P@V), so the one-time
    // matrix_pipe_warm latch is stale -> re-arm the primer. Re-priming is byte-identical to
    // the warm case (holds the read at beat 0), so this only ever adds latency, never error.
    logic [5:0] m_pipe_idle_cnt;
    localparam int PIPE_COLD_THRESHOLD = WEIGHT_READ_PIPE_DEPTH;
    assign cold_start_ok     = matrix_pipe_warm || (cold_start_primer_cnt >= COLD_START_PRIME_DEPTH);
    assign cold_start_vec_ok = matrix_pipe_warm || (cold_start_primer_cnt >= COLD_START_VEC_PRIME_DEPTH);
    // The weight-valid chain (p1/p2_m_m_load) is pre-charged during the cold spin, so it
    // would release the weight valid one cycle AHEAD of the activation valid (whose
    // p1->p2->m_v_valid chain starts empty). Gate the weight valid with a one-cycle-
    // delayed `cold_start_ok` so both release together (byte-identical to the warm case).
    logic       cold_start_ok_d;
    // High from an accumulate op (MM_IC/MV_IC) to its write-out drain; gates the idle-gap cold-start re-arm.
    logic       m_accum_active;

    always_ff @(posedge clk) begin
        if (rst) begin
            m_sram_req_pre <= 1'b0;
            m_sram_transposed_read_pre  <= 1'b0;
            continuous_load_m_en        <= 1'b0;
            continuous_prefetch_m_en    <= 1'b0;
            permit_load_for_m           <= 1'b0;
            matrix_pipe_warm            <= 1'b0;
            cold_start_primer_cnt       <= 5'd0;
            m_pipe_idle_cnt             <= 6'd0;
            cold_start_ok_d             <= 1'b0;
            m_accum_active              <= 1'b0;
            m_sram_load_counter         <= 'b1;
            load_m_amount               <= 'b0;
            m_sram_prefetch_counter     <= 'b0;
            m_m_load                    <= 1'b0;
            p1_m_m_load                 <= 1'b0;
            p2_m_m_load                 <= 1'b0;
            m_load_in_process           <= 1'b0;
            m_result_write_pending      <= 1'b0;
            continuous_v_write_from_matrix_en_q <= 1'b0;
            m_m_valid                   <= 1'b0;
            next_clk_m_m_valid          <= 1'b0;
            recorded_m_prefetch_addr    <= 'b0;
            recorded_m_load_addr        <= 'b0;
            end_of_load_m               <= 1'b0;
        end else begin
            // Address Management
            // Record prefetch address from FIFO when starting a new prefetch
            if (m_prefetch_addr_fifo_out_ready && m_prefetch_addr_fifo_out_valid) begin
                recorded_m_prefetch_addr        <= m_prefetch_addr_fifo_out;
            end

            if (exe_stage_op.m_op != STALL_M & exe_stage_op.m_op != MM_WO & exe_stage_op.m_op != MV_WO) begin
                recorded_m_load_addr            <= exe_stage_op.addr_1;  // rs1 = matrix/weight
            end 

            p1_m_m_load     <= m_m_load;
            p2_m_m_load     <= p1_m_m_load;

            // Track how long the weight-read pipe has been idle (no active load). Resets on
            // any load beat; saturates otherwise. Consumed by the cold-pipe re-arm below.
            if (m_m_load) begin
                m_pipe_idle_cnt <= 6'd0;
            end else if (m_pipe_idle_cnt != 6'd63) begin
                m_pipe_idle_cnt <= m_pipe_idle_cnt + 6'd1;
            end

            // Cold-start primer: while the first matmul is loading and the pipeline is
            // still cold, advance the primer counter (saturating). This keeps the read
            // spinning at counter 0 (permit gated below) long enough for the first real
            // weight beat to reach the MCU before m_m_valid asserts.
            if (!matrix_pipe_warm && m_m_load && cold_start_primer_cnt < COLD_START_PRIME_DEPTH) begin
                cold_start_primer_cnt <= cold_start_primer_cnt + 5'd1;
            end
            cold_start_ok_d <= cold_start_ok;

            if (exe_stage_op.m_op == MM_IC || exe_stage_op.m_op == MV_IC) begin
                m_accum_active <= 1'b1;
            end else if (exe_stage_op.m_op == MM_WO || exe_stage_op.m_op == MV_WO
                         || exe_stage_op.m_op == BMM_WO || exe_stage_op.m_op == BMV_WO) begin
                m_accum_active <= 1'b0;
            end

            continuous_v_write_from_matrix_en_q <= continuous_v_write_from_matrix_en;
            if (exe_stage_op.m_op == MM_WO || exe_stage_op.m_op == MV_WO) begin
                m_result_write_pending <= 1'b1;
            end else if (continuous_v_write_from_matrix_en_q && !continuous_v_write_from_matrix_en) begin
                m_result_write_pending <= 1'b0;
            end
            
            if (continuous_load_m_en) begin
                m_load_in_process       <= (m_sram_load_counter < load_m_amount - 1) || continuous_load_v_for_matrix_en || m_v_valid || m_m_valid || next_clk_m_m_valid || m_result_write_pending;
                // Fire one counter-step early (registered end branch reacts a cycle later).
                end_of_load_m           <= (m_sram_load_counter == load_m_amount - 2);
            end

            next_clk_m_m_valid <= (p2_m_m_load & p1_m_m_load) & (permit_load_for_m || !m_prefetch_data_not_ready) & cold_start_ok_d;
            m_m_valid <= next_clk_m_m_valid;

            // Matrix SRAM Read Port Control
            if (exe_stage_op.m_op != STALL_M & exe_stage_op.m_op != MM_WO & exe_stage_op.m_op != MV_WO) begin
                m_sram_req_pre      <= 1'b1;
                m_m_load        <= 1'b1;
                m_sram_transposed_read_pre  <= exe_stage_op.m_transposed_read;
                m_sram_load_counter     <= 'b0;
                load_m_amount           <= (exe_stage_op.m_op == MV_IC) ? MATRIX_LOAD_ITERATION_GEMV: MATRIX_LOAD_ITERATION_GEMM;
                continuous_load_m_en    <= 1'b1;
                m_load_in_process       <= 1'b1;
                // Re-arm the cold-start primer for the FIRST plain M_MM that follows a
                // per-head (M_BTMM/M_BMM) burst. The per-head QK^T leaves a stale K beat
                // in the weight-read pipe (matrix_sram rd -> element_out -> stored_m);
                // without a re-prime GQA head-0's first P@V tile reads that K instead of
                // V. recorded_m_per_head still holds the PREVIOUS matmul's flag here (it
                // is updated later this cycle), so this fires exactly once at the boundary.
                if ((exe_stage_op.m_op == MM_IC) && !exe_stage_op.m_per_head && recorded_m_per_head) begin
                    matrix_pipe_warm      <= 1'b0;
                    cold_start_primer_cnt <= 5'd0;
                end
                // Cold-pipe re-arm: this matmul starts after the read pipe drained (vector/
                // softmax ops ran in between), so the one-time warm latch is stale and the
                // first output tile would drop its corner. Re-prime. Fires only on a genuine
                // idle gap, so back-to-back matmuls (rope/ffn) are untouched.
                if (m_pipe_idle_cnt > PIPE_COLD_THRESHOLD && !m_accum_active) begin
                    matrix_pipe_warm      <= 1'b0;
                    cold_start_primer_cnt <= 5'd0;
                end
            end else if (continuous_load_m_en) begin
                if (end_of_load_m) begin
                    m_sram_req_pre              <= 1'b0;
                    m_m_load                    <= 1'b0;
                    m_sram_load_counter         <= 'b0;
                    continuous_load_m_en        <= 1'b0;
                    permit_load_for_m           <= 1'b0;
                end else begin
                    m_m_load                    <= 1'b1;
                    if (p1_m_m_load & matrix_related_data_ready & cold_start_ok) begin
                        permit_load_for_m   <= 1'b1;
                        m_sram_load_counter <= 'b1;
                        // First permit ever -> pipeline is now primed; latch warm so all
                        // subsequent matmuls skip the cold-start delay entirely.
                        matrix_pipe_warm    <= 1'b1;
                    end
                    if (permit_load_for_m) begin
                        m_sram_req_pre  <= 1'b1;
                        m_sram_load_counter <= m_sram_load_counter + 1'b1;
                    end else begin
                        m_sram_req_pre  <= 1'b1;
                        continuous_load_m_en <= 1'b1;
                    end
                end
            end else begin
                m_m_load   <= 1'b0;
                m_sram_req_pre <= 1'b0;
                continuous_load_m_en <= 1'b0;
                m_load_in_process <= continuous_load_v_for_matrix_en || m_v_valid || m_result_write_pending;
            end

            // Matrix SRAM Write Port Control (start prefetch only when FIFO addr ready)
            if (prefetch_m_valid && !continuous_prefetch_m_en && m_sram_prefetch_counter == 0 && m_prefetch_addr_fifo_out_valid) begin
                continuous_prefetch_m_en    <= 1'b1;
                m_sram_prefetch_counter     <= 'b0;
            end else if (continuous_prefetch_m_en && prefetch_m_valid && m_sram_prefetch_counter < HBM_M_Prefetch_Amount - 1) begin
                    // Advance the write row only on an actual HBM data beat.
                    m_sram_prefetch_counter <= m_sram_prefetch_counter + 'b1;
            end else if (m_sram_prefetch_counter == HBM_M_Prefetch_Amount - 1) begin
                // Prefetching finished, reset the counter
                continuous_prefetch_m_en <= 1'b0;
                m_sram_prefetch_counter <= 'b0;
            end
        end
    end

    // -----------------------------
    // Vector SRAM
    // -----------------------------

    // Assuming the read cycle is 1 cycle for both ports.

    // Port A ->  R: Matrix Multiplicand Vector & Vector Machine input Operand (RS1)   W: Vector Result from either Matrix or Vector Machine, 
    // Port B ->  R: Vector Machine input Operand (RS2)  or Load HBM Write Data        W: Vector Prefetch
    // For Port A, if loading it to the matrix machine, this takes extra cycle as we need to quantise the fp data (activation) into MX-FP format.

    logic [INT_DATA_WIDTH - 1 : 0] recorded_v_prefetch_addr;
    logic [INT_DATA_WIDTH - 1 : 0] recorded_v_load_for_matrix_addr;
    logic [INT_DATA_WIDTH - 1 : 0] recorded_v_load_addr_1, recorded_v_load_addr_2;
    logic [INT_DATA_WIDTH - 1 : 0] recorded_s_map_v_addr;
    logic [INT_DATA_WIDTH - 1 : 0] recorded_m_write_addr, recorded_v_write_addr;
    // In-place element-op write addr: use matching read addr when rd == a read reg.
    logic [INT_DATA_WIDTH - 1 : 0] recorded_v_inplace_waddr;
    logic                          recorded_v_inplace_valid;
    logic [1:0] recorded_m_write_mode;
    // Per-head (M_BTMM/M_BMM_WO) writeout: the matrix machine drains ROW_BLOCK_NUM
    // heads x BLEN rows (head-major) instead of the single BLEN-row M_MM tile. Latched
    // when the writeout op passes exe; held through the drain (one writeout in flight).
    logic recorded_m_per_head;
    // Head layout in the S buffer: MLEN/BLEN heads, each occupying an MLEN x MLEN region.
    localparam int PH_ROW_BLOCK_NUM = MLEN / BLEN;              // number of per-head banks
    localparam int PH_DRAIN_ROWS    = PH_ROW_BLOCK_NUM * BLEN;  // = MLEN total drained rows
    localparam int PH_BLEN_W        = $clog2(BLEN);
    localparam int PH_HEAD_W        = $clog2(PH_ROW_BLOCK_NUM);
    localparam int PH_HEAD_STRIDE   = MLEN * MLEN;              // per-head S region size (elements)
    logic [INT_DATA_WIDTH - 1 : 0] hbm_store_load_addr;
    logic port_b_prefetch_ready;
    logic continuous_v_prefetch_en, continuous_load_v_for_matrix_en, continuous_v_write_from_matrix_en, continuous_write_to_hbm;
    logic [M_LD_COUNT_WIDTH : 0] v_sram_load_for_matrix_counter, v_sram_write_from_matrix_counter;
    logic load_for_gemv_en;
    logic [V_PF_COUNT_WIDTH : 0] v_sram_prefetch_counter;
    logic [H_WR_COUNT_WIDTH : 0] hbm_write_counter;
    logic v_v_a_load, v_v_b_load;
    logic end_of_load_v_for_matrix;
    logic p1_vport_a_load_valid, p2_vport_a_load_valid;
    logic recorded_prefetch_precision; // 0 for high precision, 1 for low precision
    logic s_map_v_ready;

    assign continuous_write_to_v_sram_port_b = continuous_v_prefetch_en;
    assign v_sram_mask_b = {VLEN{1'b1}};
    always_comb begin
        // Port A Addr Mangement
         if (continuous_load_v_for_matrix_en) begin
            select_write_data_a = 1'b0;
            v_sram_addr_a = recorded_v_load_for_matrix_addr + v_sram_load_for_matrix_counter * VLEN;
        end else if (continuous_v_write_from_matrix_en) begin
            select_write_data_a = 1'b1;
            // MM_WO masks addressed columns; GEMV (2'b10) writes a full row.
            if (recorded_m_write_mode == 2'b01) begin
                v_sram_mask_a = {{(VLEN-MATRIX_LOAD_ITERATION_GEMM){1'b0}}, {MATRIX_LOAD_ITERATION_GEMM{1'b1}}}
                                 << recorded_m_write_addr[$clog2(VLEN)-1:0];
            end else begin
                v_sram_mask_a = {MLEN{1'b1}};
            end
            // Per-head drain: head-major (head = counter high bits, row = low bits).
            // Head h's tile lands at recorded base + h*(MLEN*MLEN) + row*VLEN. Plain
            // M_MM: consecutive rows (counter*VLEN) from the single summed tile.
            if (recorded_m_per_head) begin
                v_sram_addr_a = recorded_m_write_addr
                    + v_sram_write_from_matrix_counter[PH_BLEN_W +: PH_HEAD_W] * PH_HEAD_STRIDE
                    + v_sram_write_from_matrix_counter[PH_BLEN_W-1:0] * VLEN;
            end else begin
                v_sram_addr_a = recorded_m_write_addr + v_sram_write_from_matrix_counter * VLEN;
            end
        end else if (mem_stage_write_control.w_s_sram_port_a_en == 1'b1 && select_write_data_a == 1'b0) begin
            select_write_data_a = 1'b0;
            v_sram_mask_a = {VLEN{1'b1}};
            // In-place element ops use matching read addr; else nominal write addr.
            v_sram_addr_a = recorded_v_inplace_valid ? recorded_v_inplace_waddr : recorded_v_write_addr;
        end else begin
            select_write_data_a = 1'b0;
            v_sram_addr_a = recorded_v_load_addr_1;
        end

        // Port B Addr Mangement
        if (continuous_v_prefetch_en) begin
            v_sram_addr_b = recorded_v_prefetch_addr + v_sram_prefetch_counter * VLEN;
        end else if (continuous_write_to_hbm) begin
            v_sram_addr_b = hbm_store_load_addr + hbm_write_counter * VLEN;
        end else if (s_map_v_ready) begin
            v_sram_addr_b = recorded_s_map_v_addr;
        end else begin
            v_sram_addr_b = recorded_v_load_addr_2;
        end

        // Prefetch Record: capture address only when starting a new prefetch.
        if (rst) begin
            recorded_v_prefetch_addr = 'b0;
            hbm_store_load_addr = 'b0;
        end else if ((exe_stage_op.h_op == PREFETCH_V_H || exe_stage_op.h_op == PREFETCH_V_L)
                     && !continuous_v_prefetch_en && v_sram_prefetch_counter == 0) begin
            recorded_v_prefetch_addr = exe_stage_op.addr_2;
            recorded_prefetch_precision = (exe_stage_op.h_op == PREFETCH_V_H) ? 1'b0 : 1'b1;
        end else if (exe_stage_op.h_op == STORE_V_H || exe_stage_op.h_op == STORE_V_L) begin
            hbm_store_load_addr = exe_stage_op.addr_2;
            recorded_prefetch_precision = (exe_stage_op.h_op == STORE_V_H ) ? 1'b0 : 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            v_v_a_valid     <= 1'b0;
            v_v_b_valid     <= 1'b0;
            hbm_v_req_prefetch_data         <= 1'b0;
            recorded_v_load_addr_1          <= 'b0;
            recorded_v_load_addr_2          <= 'b0;
            recorded_v_load_for_matrix_addr <= 'b0;
            recorded_m_write_addr           <= 'b0;
            recorded_m_write_mode           <= 2'b0;
            recorded_m_per_head             <= 1'b0;
            recorded_v_write_addr           <= 'b0;
            recorded_v_inplace_waddr        <= 'b0;
            recorded_v_inplace_valid        <= 1'b0;
            load_for_gemv_en                <= 1'b0;
            p1_vport_a_load_valid           <= 1'b0;
            p2_vport_a_load_valid           <= 1'b0;
            m_v_valid                       <= 1'b0;
            m_v_load                        <= 1'b0;
            v_v_a_load                      <= 1'b0;
            v_v_b_load                      <= 1'b0;
            m_v_load_cond                   <= 1'b0;
            port_b_prefetch_ready           <= 1'b0;
            continuous_v_write_from_matrix_en   <= 1'b0;
            continuous_load_v_for_matrix_en     <= 1'b0;
            v_sram_mxfp_req_b               <= 2'b0;
            v_sram_req_b                    <= 1'b0;
            v_sram_req_a                    <= 1'b0;
            hbm_write_counter               <= 'b0;
            end_of_load_v_for_matrix        <= 1'b0;
            v_sram_load_for_matrix_counter  <= 'b1;
            v_sram_write_from_matrix_counter <= 'b0;
            s_map_v_ready                   <= 1'b0;
            select_write_data_b             <= 2'b0;

        end else begin
            v_v_a_valid                 <= v_v_a_load;
            v_v_b_valid                 <= v_v_b_load;
            m_v_load_cond               <= m_v_load & !end_of_load_v_for_matrix;
            // end_of_load already gates m_v_load_cond one stage above.
            p1_vport_a_load_valid       <= m_v_load_cond & (permit_load_for_m || matrix_related_data_ready) & cold_start_ok;
            p2_vport_a_load_valid       <= p1_vport_a_load_valid;
            m_v_valid                   <= p2_vport_a_load_valid;

            // One counter-step early (free-running counter, end mark at amount-1).
            end_of_load_v_for_matrix    <= (v_sram_load_for_matrix_counter == MATRIX_LOAD_ITERATION_GEMM - 1);
            //Port A
            if((exe_stage_op.m_op != STALL_M & exe_stage_op.m_op != MM_WO & exe_stage_op.m_op != MV_WO)) begin
                // Read Vector from SRAM
                m_v_load        <= 1'b1;
                v_v_a_load      <= 1'b0;
                v_sram_req_a    <= 1'b1;
                v_sram_wen_a    <= 1'b0;
                continuous_load_v_for_matrix_en <= 1'b1;
                v_sram_load_for_matrix_counter  <= 'b0;
                // Override stale end-flag from the previous load's old counter.
                end_of_load_v_for_matrix        <= 1'b0;
                load_for_gemv_en <= (exe_stage_op.m_op == MV_IC) ? 1'b1 : 1'b0;
            end else if (continuous_load_v_for_matrix_en) begin
                if (!load_for_gemv_en & end_of_load_v_for_matrix) begin
                    v_sram_req_a <= 1'b0;
                    m_v_load   <= 1'b0;
                    v_sram_load_for_matrix_counter  <= 'b0;
                    continuous_load_v_for_matrix_en <= 1'b0;
                end else begin
                    if (permit_load_for_m & m_v_load_cond) begin
                        if (load_for_gemv_en) begin
                            m_v_load                        <= 1'b0;
                            v_sram_req_a                    <= 1'b0;
                            v_sram_load_for_matrix_counter  <= 'b0;
                            continuous_load_v_for_matrix_en <= 1'b0;
                            load_for_gemv_en                <= 1'b0;
                        end else begin
                            m_v_load                        <= 1'b1;
                            v_sram_req_a                    <= 1'b1;
                            v_sram_load_for_matrix_counter  <= v_sram_load_for_matrix_counter + 'b1;
                        end
                    end else if (matrix_related_data_ready & cold_start_vec_ok) begin
                        // Pre-permit phase: keep the read stream advancing.
                        // Held at counter 0 during the one-time cold-start prime, then
                        // released COLD_START_VEC_LEAD cycles AHEAD of the weight permit so
                        // the longer activation read+requantize path delivers data to the
                        // MCU aligned with the primed weight beat (matches the warm phase).
                        v_sram_load_for_matrix_counter  <= v_sram_load_for_matrix_counter + 'b1;
                        m_v_load                        <= 1'b1;
                        v_sram_req_a                    <= 1'b1;
                    end else begin
                        m_v_load    <= 1'b1;
                        v_sram_req_a <= 1'b1;
                        continuous_load_v_for_matrix_en <= 1'b1;
                        v_sram_load_for_matrix_counter  <= 'b0;
                    end
                end
            end else if ((exe_stage_op.v_ele_op != STALL_V_ELEMENT) || (exe_stage_op.v_reduct_op != STALL_V_REDUCT)) begin
                // TODO: Need to introduce v_prefetch_data_not_ready for vector machine, for safe data fetching.
                m_v_load        <= 1'b0;
                v_v_a_load      <= 1'b1;
                v_sram_req_a    <= 1'b1;
                v_sram_wen_a    <= 1'b0;
            end else if (continuous_v_write_from_matrix_en & m_out_valid) begin
                // Write exactly BLEN rows of the drained tile. `<` (not `<=`): the
                // registered wen + combinational addr already emit BLEN writes, so `<=`
                // added a 5th (drift = BLEN*VLEN) overshoot row that spilled a stale
                // systolic row into the next VRAM region (corrupted MHA head-0's O).
                if (v_sram_write_from_matrix_counter < (recorded_m_per_head ? PH_DRAIN_ROWS : MATRIX_LOAD_ITERATION_GEMM) - 1) begin
                    m_v_load        <= 1'b0;
                    v_v_a_load      <= 1'b0;
                    v_sram_req_a    <= 1'b1;
                    v_sram_wen_a    <= 1'b1;
                    v_sram_write_from_matrix_counter <= v_sram_write_from_matrix_counter + 1'b1;
                end else begin
                    m_v_load        <= 1'b0;
                    v_v_a_load      <= 1'b0;
                    v_sram_req_a    <= 1'b0;
                    v_sram_wen_a    <= 1'b0;
                    continuous_v_write_from_matrix_en <= 1'b0;
                    v_sram_write_from_matrix_counter <= 'b0;
                end
            end else if (continuous_v_write_from_matrix_en && !m_out_valid && v_sram_write_from_matrix_counter != 0) begin
                // Drain stream ended (m_out_valid fell): close the burst.
                m_v_load        <= 1'b0;
                v_v_a_load      <= 1'b0;
                v_sram_req_a    <= 1'b0;
                v_sram_wen_a    <= 1'b0;
                continuous_v_write_from_matrix_en <= 1'b0;
                v_sram_write_from_matrix_counter <= 'b0;
            end else if (mem_write_control.w_s_sram_port_a_en == 1'b1) begin
                if (mem_write_control.w_from_m) begin
                    // Write the result from matrix machine to the s_sram
                    m_v_load        <= 1'b0;
                    v_v_a_load      <= 1'b0;
                    v_sram_req_a    <= 1'b1;
                    v_sram_wen_a    <= 1'b1;           
                    continuous_v_write_from_matrix_en <= 1'b1;         
                    v_sram_write_from_matrix_counter <= 'b0;
                end else begin
                    // Write the result from vector machine to the s_sram
                    m_v_load        <= 1'b0;
                    v_v_a_load      <= 1'b0;
                    v_sram_req_a    <= 1'b1;
                    v_sram_wen_a    <= 1'b1;                    
                end
            end else begin
                // No Scratchpad SRAM access request.
                m_v_load        <= 1'b0;
                v_v_a_load      <= 1'b0;
                v_sram_req_a    <= 1'b0;
                v_sram_wen_a    <= 1'b0;
            end

            //Port B
            if (((exe_stage_op.v_ele_op != STALL_V_ELEMENT) && (exe_stage_op.v_ele_op != INNER_HADAMARD_TRANSFORM) && (!exe_stage_op.v_broadcast_en))) begin
                // Read Port activated
                v_v_b_load                  <= 1'b1;
            end else if ((exe_stage_op.h_op == STORE_V_H) & hbm_ready_to_write) begin
                // Start HBM Writeback to the scratchpad sram
                continuous_write_to_hbm     <= 1'b1;
                hbm_write_counter           <= 'b0;
                v_v_b_load                  <= 1'b0;
                v_sram_mxfp_req_b           <= (exe_stage_op.h_op == STORE_V_H) ? 2'b01 : 2'b10; // 01 for C, 10 for S
            end else if (continuous_write_to_hbm && hbm_write_counter < HBM_WRITE_AMOUNT - 1 && hbm_ready_to_write) begin
                // Intermediate HBM Writeback to the scratchpad sram
                v_sram_mxfp_req_b           <= v_sram_mxfp_req_b;
                v_v_b_load                  <= 1'b1;
                hbm_write_counter           <= hbm_write_counter + 'b1;
            end else if (hbm_write_counter == HBM_WRITE_AMOUNT - 1 && hbm_ready_to_write) begin
                // Finish HBM Writeback, reset the counter
                v_sram_mxfp_req_b           <= 2'b0;
                v_v_b_load                  <= 1'b0;
                continuous_write_to_hbm     <= 1'b0;
                hbm_write_counter           <= 'b0;
            end else if ((exe_stage_op.h_op == PREFETCH_V_H || exe_stage_op.h_op == PREFETCH_V_L) && !continuous_v_prefetch_en && v_sram_prefetch_counter == 0) begin
                // Start HBM V prefetch
                continuous_v_prefetch_en        <= 1'b1;
                v_v_b_load                      <= 1'b0;
                hbm_v_req_prefetch_data         <= 1'b1;
            end else if (continuous_v_prefetch_en && v_sram_prefetch_counter < HBM_V_Prefetch_Amount) begin
                hbm_v_req_prefetch_data         <= 1'b1;
                v_v_b_load                      <= 1'b0;
                if (v_sram_wen_b) begin
                    v_sram_prefetch_counter     <= v_sram_prefetch_counter + 'b1;
                end
            end else if (v_sram_prefetch_counter == HBM_V_Prefetch_Amount) begin
                // Finish Prefetching, reset the counter
                hbm_v_req_prefetch_data         <= 1'b0;
                continuous_v_prefetch_en        <= 1'b0;
                v_sram_prefetch_counter         <= 'b0;
                v_v_b_load                      <= 1'b0;
            end  else begin
                v_v_b_load                      <= 1'b0;
                hbm_v_req_prefetch_data         <= 1'b0;
                continuous_v_prefetch_en        <= 1'b0;
            end

            if (hbm_v_req_prefetch_data && prefetch_v_valid) begin
                v_sram_req_b            <= 1'b1;
                v_sram_wen_b            <= 1'b1;
                s_map_v_ready           <= 1'b0;
                if (recorded_prefetch_precision == 1'b0) begin
                    select_write_data_b     <= 2'b01; // High Precision
                end else begin
                    select_write_data_b     <= 2'b10; // Low Precision
                end
            end else if ((exe_stage_op.v_ele_op != STALL_V_ELEMENT) && (exe_stage_op.v_ele_op != INNER_HADAMARD_TRANSFORM) && !exe_stage_op.v_broadcast_en) begin
                v_sram_req_b            <= 1'b1;
                v_sram_wen_b            <= 1'b0;
                s_map_v_ready           <= 1'b0;
                select_write_data_b     <= 2'b01;
            end else if (s_map_v_valid & !s_map_v_ready) begin
                v_sram_req_b            <= 1'b1;
                v_sram_wen_b            <= 1'b1;
                s_map_v_ready           <= 1'b1;
                select_write_data_b     <= 2'b11;
            end else begin
                v_sram_req_b            <= 1'b0;
                v_sram_wen_b            <= 1'b0;
                s_map_v_ready           <= 1'b0;
            end

            if (exe_stage_op.v_ele_op != STALL_V_ELEMENT || exe_stage_op.v_reduct_op != STALL_V_REDUCT) begin
                recorded_v_load_addr_1  <= exe_stage_op.addr_1;
                recorded_v_load_addr_2  <= exe_stage_op.addr_2;
            end

            // Resolve the vector-element write addr from the operand matching rd:
            // rs1 (addr_1), rs2 (addr_2), or - when rd is NEITHER source (out of
            // place) - the separately-resolved gp[rd] address (addr_3). The old
            // out-of-place fallback used a stale late v_write_addr and wrote the
            // result to the wrong VRAM row (rs2's address); addr_3 fixes that.
            // Only vector-VECTOR ops need addr_3: a broadcast op (V_*_VF/EXP/RECI)
            // carries gp[rd] on its own late v_write path (addr_2 rs2:=rd remap +
            // update_v_waddr), which is robust across a matrix->vector boundary,
            // whereas addr_3 has no drift-replay and would collapse to 0 there.
            if (exe_stage_op.v_ele_op != STALL_V_ELEMENT) begin
                if (exe_stage_op.gp_rd == exe_stage_op.gp_reg1) begin
                    recorded_v_inplace_waddr <= exe_stage_op.addr_1;
                    recorded_v_inplace_valid <= 1'b1;
                end else if (exe_stage_op.gp_rd == exe_stage_op.gp_reg2) begin
                    recorded_v_inplace_waddr <= exe_stage_op.addr_2;
                    recorded_v_inplace_valid <= 1'b1;
                end else if (!exe_stage_op.v_broadcast_en) begin
                    recorded_v_inplace_waddr <= exe_stage_op.addr_3;
                    recorded_v_inplace_valid <= 1'b1;
                end else begin
                    recorded_v_inplace_valid <= 1'b0;
                end
            end
            
            if (exe_stage_op.s_fp_op == MAP_V_FP) begin
                recorded_s_map_v_addr <= exe_stage_op.addr_1;
            end 
            
            if (exe_stage_op.m_op != STALL_M & exe_stage_op.m_op != MM_WO & exe_stage_op.m_op != MV_WO) begin
                recorded_v_load_for_matrix_addr <= exe_stage_op.addr_2;  // rs2 = vector/activation
            end

            // Latch per-head from the CURRENT matmul — at BOTH the load (MM_IC) and the
            // writeout (MM_WO) stage. Latching at MM_IC (which precedes this matmul's
            // MM_WO) guarantees recorded_m_per_head reflects THIS matmul by the time its
            // drain runs, so a prior per-head M_BMM_WO's flag can't go stale into a
            // following plain M_MM's drain (which would scatter its BLEN rows head-major).
            if (exe_stage_op.m_op == MM_IC || exe_stage_op.m_op == MM_WO) begin
                recorded_m_per_head <= exe_stage_op.m_per_head;
            end

            if (m_write_request) begin
                recorded_m_write_addr <= m_write_addr;
                recorded_m_write_mode <= m_write_request;
            end else begin
                recorded_m_write_addr <= recorded_m_write_addr;
            end

            if (v_write_request) begin
                recorded_v_write_addr <= v_write_addr;
            end else begin
                recorded_v_write_addr <= recorded_v_write_addr;
            end
        end

    end

    // Scalar Data Forwarding to Vector Machine
    always_ff @(posedge clk) begin
        if (rst) begin
            v_s_in_valid <= 1'b0;
        end else begin
            if ((mem_stage_op.v_broadcast_en == 1'b1) || (mem_stage_op.v_reduct_op != STALL_V_REDUCT)) begin
                v_s_in_valid <= 1'b1;
            end else begin
                v_s_in_valid <= 1'b0;
            end
        end
    end

    // -----------------------------
    // HBM Prefetch Status Tracking
    // -----------------------------
    // Cover the whole prefetch window: FIFO push + queued + HBM burst.
    assign hbm_m_prefetch_in_progress = continuous_prefetch_m_en
                                      || !m_prefetch_addr_fifo_empty
                                      || m_prefetch_addr_fifo_in_valid;
    assign hbm_v_prefetch_in_progress = continuous_v_prefetch_en;

endmodule
