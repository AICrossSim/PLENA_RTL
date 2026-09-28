`timescale 1ns / 1ps
`include "operation.svh"
/*
Module      : Systolic Array Based Matrix Compute Unit (MCU)
Timing      : Sequential
Description : It supports both for GEMM and GEMV operations.
            : (M, K) x (K, N) = (M, N)
            : (M, K) x (K, 1) = (M, 1)
            :  where M and N is the Batch_Size , K is the MLEN
            : Every CLK, it will receive (K) vector as multiplicand and (K) vector as multiplier.
            : During the write process, output single row per clk.
            : For GEMM, pass the v2_element and v1_element to the left and top streamers
            : For GEMV, hold the v2_element and stream v1_element.
Status      : Under Development
*/

module mx_systolic_mcu #(
    // MX-FP Data Format
    parameter FP_EXP_WIDTH          = 8,
    parameter FP_MANT_WIDTH         = 7,
    parameter MX_T_EXP_WIDTH      = 4,
    parameter MX_T_MANT_WIDTH     = 3,
    parameter MX_L_EXP_WIDTH      = 4,
    parameter MX_L_MANT_WIDTH     = 3,
    parameter MX_SCALE_WIDTH      = 8,
    parameter BLOCK_DIM             = 4,
    // Accumulator Data Format
    parameter ACC_FP_EXP_WIDTH      = 8,
    parameter ACC_FP_MANT_WIDTH     = 7,
    parameter SYSTOLIC_PROCESSING_OVERHEAD = 4,
    // Dimension
    parameter   M                     = 4,
    parameter   N                     = 4,
    parameter   K                     = 8,
    localparam  ROW_BLOCK_NUM         = K / BLOCK_DIM,
    parameter   ACC_ADDR_WIDTH        = 8,
    parameter   L_MX_INT_EN           = 0,
    parameter   T_MX_INT_EN           = 0 // Not implemented yet
)(
    input   logic clk,
    input   logic rst,
    input   M_OP  control,      // 0 for GEMV, 1 for GEMM
    input   logic [ACC_ADDR_WIDTH - 1 : 0] acc_waddr,
    input   logic fetch_next_acc_waddr_valid,
    output  logic fetch_next_acc_waddr_ready,
    input   logic wait_for_output,

    // Multiplicant Matrix 1 TOP
    input   logic [K - 1 : 0][MX_T_EXP_WIDTH + MX_T_MANT_WIDTH : 0]     v1_element,
    input   logic [K - 1 : 0][MX_SCALE_WIDTH - 1 : 0]                   v1_scale,
    input   logic v1_in_valid,

    // Multiplier   Matrix 2 LEFT
    input   logic [K - 1 : 0][MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0]     v2_element,
    input   logic [ROW_BLOCK_NUM - 1 : 0][MX_SCALE_WIDTH - 1 : 0]       v2_scale,
    input   logic v2_in_valid,

    // Vector Product Output
    output  logic [K - 1 : 0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0]         v_result,
    output  logic v_result_write_req,
    output  logic mcu_active
`ifdef SIMULATION
    ,
    output  logic dbg_complete_v1_load,
    output  logic dbg_complete_v2_load,
    output  logic dbg_complete_loading_q,
    output  logic [ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH:0] dbg_gebm_result_0
`endif
);

    initial begin
        if (M != N) begin
            $error("Systolic MCU only supports M == N, but got M = %0d, N = %0d", M, N);
            $finish;
        end
    end

    localparam SYS_ARRAY_AMOUNT = K / M;
    localparam COMPUTE_DIM = M;
    // If the BLOCK_DIM is larger than BLEN
    localparam BLOCK_NUM_PER_ARRAY = (M < BLOCK_DIM) ? 1 : M / BLOCK_DIM;
    localparam DUPLICATE_BLOCK_NUM = (M < BLOCK_DIM) ? BLOCK_DIM / M : 1;

    // -----------------------------
    // Data Wires Declaration
    // -----------------------------

    logic v1_for_mm_in_valid;
    logic v1_data_for_mm_in_valid;
    logic v1_load_for_gemv_in_valid;
    logic v1_for_mv_in_valid;
    logic [SYS_ARRAY_AMOUNT - 1 : 0] v1_data_for_mv_in_valid;

    logic v2_for_mm_in_valid;
    logic v2_data_for_mm_in_valid;
    logic v2_for_mv_in_valid;
    logic [SYS_ARRAY_AMOUNT - 1 : 0] v2_data_for_mv_in_valid;

    logic [SYS_ARRAY_AMOUNT - 1 : 0] array_top_in_valid;
    logic [SYS_ARRAY_AMOUNT - 1 : 0] array_left_in_valid;

    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_T_EXP_WIDTH + MX_T_MANT_WIDTH : 0]           array_top_in_element;
    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_SCALE_WIDTH - 1 : 0]                         array_top_in_scale;
    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_T_EXP_WIDTH + MX_T_MANT_WIDTH : 0]           array_top_v_in_element;
    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_SCALE_WIDTH - 1 : 0]                         array_top_v_in_scale;
    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0]           array_left_in_element;
    wire [SYS_ARRAY_AMOUNT - 1 : 0][BLOCK_NUM_PER_ARRAY - 1 : 0]   [MX_SCALE_WIDTH - 1 : 0]                 array_left_in_scale;
    logic [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM - 1 : 0]   [MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0]          array_left_v_in_element;
    logic [SYS_ARRAY_AMOUNT - 1 : 0][BLOCK_NUM_PER_ARRAY - 1 : 0]   [MX_SCALE_WIDTH - 1 : 0]                array_left_v_in_scale;


    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM- 1: 0][COMPUTE_DIM- 1: 0][ ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] gemm_result;
    logic [SYS_ARRAY_AMOUNT - 1 : 0] gemm_result_valid;

    wire [SYS_ARRAY_AMOUNT - 1 : 0][COMPUTE_DIM- 1: 0][ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] gemv_result;
    logic [SYS_ARRAY_AMOUNT - 1 : 0] gemv_result_valid;

    // -----------------------------
    // Control and Status Tracking
    // -----------------------------

    logic   load_array_data;
    logic   complete_loading;
    M_OP    control_in_exe;
    logic   sa_control;
    logic   output_reset;


    localparam COUNTER_BIT_WIDTH = $clog2(K);
    localparam DRAIN_CYCLES = 2 * M + SYSTOLIC_PROCESSING_OVERHEAD;
    localparam FEED_COUNTER_WIDTH = $clog2(DRAIN_CYCLES + 1);
    logic [COUNTER_BIT_WIDTH : 0] v1_load_counter;
    logic [COUNTER_BIT_WIDTH : 0] v1_load_amount;
    logic [COUNTER_BIT_WIDTH : 0] v2_load_counter;
    logic [FEED_COUNTER_WIDTH : 0] feed_counter;
    logic complete_v1_load, complete_v2_load;
    logic complete_loading_q;

    logic   ready_to_load_output;

    always_ff @(posedge clk) begin
        if (rst) begin
            v1_load_counter     <= '0;
            v1_load_amount      <= '0;
            complete_v1_load    <= 1'b0;
            v2_load_counter     <= '0;
            complete_v2_load    <= 1'b0;
            output_reset        <= 1'b0;
        end else begin
            v1_load_amount  <= (control_in_exe == MV_IC) ? K - 1 : M - 1;
            // Counter for v1
            if (v1_in_valid) begin
                if (v1_load_counter == v1_load_amount) begin
                    v1_load_counter <= '0;
                    complete_v1_load <= 1'b1;
                end else begin
                    v1_load_counter <= v1_load_counter + 'b1;
                end
            end
            // Counter for v2
            if (v2_in_valid) begin
                if (v2_load_counter == M - 1) begin
                    v2_load_counter <= '0;
                    complete_v2_load <= 1'b1;
                end else begin
                    v2_load_counter <= v2_load_counter + 'b1;
                end
            end
            // Output Reset
            output_reset <= (((control_in_exe == MV_WO) && ((&gemv_result_valid) == 1'b1)) || ((control_in_exe == MM_WO) && ((&gemm_result_valid) == 1'b1)));
        end
    end

    always_comb begin
        if ((control_in_exe == MM_IC) & complete_v1_load & complete_v2_load) begin
            complete_loading = 1'b1;
        end else if (control_in_exe == MV_IC & complete_v1_load) begin
            complete_loading = 1'b1;
        end else begin
            complete_loading = 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            control_in_exe   <= STALL_M;
            complete_loading_q <= 1'b0;
        end else begin
            if (control != STALL_M && !mcu_active) begin
                control_in_exe <= control;
            end
            if (complete_loading)
                complete_loading_q <= 1'b1;
            else if (mcu_active)
                complete_loading_q <= 1'b0;
        end
    end

    assign  sa_control = (control_in_exe == MV_IC) || (control_in_exe == MV_WO); // 0 for GEMM, 1 for GEMV

    // -----------------------------
    // Input Data Path
    // -----------------------------

    always_comb begin
        if (control_in_exe == MV_IC || control_in_exe == MV_WO) begin
            v1_for_mm_in_valid  = 1'b0;
            v1_load_for_gemv_in_valid  = v1_in_valid;
            v2_for_mm_in_valid  = 1'b0;

        end else begin
            v1_for_mm_in_valid  = v1_in_valid;
            v1_load_for_gemv_in_valid  = 1'b0;
            v2_for_mm_in_valid  = v2_in_valid;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            array_left_v_in_element <= 'b0;
            array_left_v_in_scale   <= 'b0;
        end else if (sa_control) begin
            if (complete_loading) begin
                array_left_v_in_element <= 'b0;
                array_left_v_in_scale   <= 'b0;
                v2_for_mv_in_valid      <= 1'b0;
            end else if (v2_in_valid) begin
                array_left_v_in_element <= v2_element;
                array_left_v_in_scale   <= v2_scale;
                v2_for_mv_in_valid      <= v2_in_valid;
            end
        end
    end

    assign v1_data_for_mm_in_valid = v1_for_mm_in_valid;
    assign v2_data_for_mm_in_valid = v2_for_mm_in_valid;

    logic v1_for_mv_ele_in_valid;
    logic v1_for_mv_scale_in_valid;
    logic v1_for_mv_ele_out_valid;
    logic v1_for_mv_scale_out_valid;


    register_slice_wo_hs #(
        .DATA_WIDTH(SYS_ARRAY_AMOUNT * COMPUTE_DIM * (MX_T_EXP_WIDTH + MX_T_MANT_WIDTH + 1))
    ) v1_gemv_ele_streamer (
            .clk           (clk),
            .rst           (rst),
            .data_in       (v1_element),
            .data_in_valid (v1_for_mv_ele_in_valid),
            .data_out      (array_top_v_in_element),
            .data_out_valid(v1_for_mv_ele_out_valid)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(SYS_ARRAY_AMOUNT * COMPUTE_DIM * MX_SCALE_WIDTH)
    ) v1_gemv_scale_streamer (
            .clk           (clk),
            .rst           (rst),
            .data_in       (v1_scale),
            .data_in_valid (v1_for_mv_scale_in_valid),
            .data_out      (array_top_v_in_scale),
            .data_out_valid(v1_for_mv_scale_out_valid)
    );

  assign v1_for_mv_in_valid = v1_for_mv_ele_out_valid & v1_for_mv_scale_out_valid;


    // -----------------------------
    // Systolic Array Computation Unit
    // -----------------------------

    generate;

        for (genvar i = 0; i < SYS_ARRAY_AMOUNT; i++) begin
            mx_systolic_top_streamer #(
                .MX_EXP_WIDTH       (MX_T_EXP_WIDTH),
                .MX_MANT_WIDTH      (MX_T_MANT_WIDTH),
                .MX_SCALE_WIDTH     (MX_SCALE_WIDTH),
                .BLOCK_DIM          (BLOCK_DIM / DUPLICATE_BLOCK_NUM),
                .ACC_FP_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                .ACC_FP_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                .COMPUTE_DIM        (COMPUTE_DIM)
            ) top_streamer (
                .clk(clk),
                .rst(rst),
                .data_elem_in   (v1_element[i * M +: M]),
                .data_scale_in  (v1_scale[i * M +: M]),
                .data_in_valid  (v1_data_for_mm_in_valid),
                .data_elem_out  (array_top_in_element[i]),
                .data_scale_out (array_top_in_scale[i]),
                .data_out_valid (array_top_in_valid[i])
            );

            mx_systolic_left_streamer #(
                .MX_EXP_WIDTH       (MX_L_EXP_WIDTH),
                .MX_MANT_WIDTH      (MX_L_MANT_WIDTH),
                .MX_SCALE_WIDTH     (MX_SCALE_WIDTH),
                .BLOCK_DIM          (BLOCK_DIM / DUPLICATE_BLOCK_NUM),
                .ACC_FP_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                .ACC_FP_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                .COMPUTE_DIM        (COMPUTE_DIM)
            ) left_streamer (
                .clk(clk),
                .rst(rst),
                .data_elem_in   (v2_element[i * M +: M]),
                .data_scale_in  (v2_scale[i / DUPLICATE_BLOCK_NUM]),
                .data_in_valid  (v2_data_for_mm_in_valid),
                .data_elem_out  (array_left_in_element[i]),
                .data_scale_out (array_left_in_scale[i]),
                .data_out_valid (array_left_in_valid[i])
            );

            mx_systolic_array #(
                .MX_T_EXP_WIDTH       (MX_T_EXP_WIDTH),
                .MX_T_MANT_WIDTH      (MX_T_MANT_WIDTH),
                .MX_L_EXP_WIDTH       (MX_L_EXP_WIDTH),
                .MX_L_MANT_WIDTH      (MX_L_MANT_WIDTH),
                .MX_SCALE_WIDTH       (MX_SCALE_WIDTH),
                .BLOCK_DIM            (BLOCK_DIM / DUPLICATE_BLOCK_NUM),
                .ACC_FP_EXP_WIDTH     (ACC_FP_EXP_WIDTH),
                .ACC_FP_MANT_WIDTH    (ACC_FP_MANT_WIDTH),
                .COMPUTE_DIM          (COMPUTE_DIM),
                .L_MX_INT_EN          (L_MX_INT_EN)
            ) systolic_array_inst (
                .clk(clk),
                .rst                (rst),
                .clear_accumulator  (output_reset),
                .control            (sa_control),

                .in_top_element     (array_top_in_element[i]),
                .in_top_scale       (array_top_in_scale[i]),
                .in_top_valid       (array_top_in_valid[i]),

                .in_top_v_element   (array_top_v_in_element[i]),
                .in_top_v_scale     (array_top_v_in_scale[i]),
                .in_top_v_valid     (v1_data_for_mv_in_valid[i]),

                .in_left_element    (array_left_in_element[i]),
                .in_left_scale      (array_left_in_scale[i]),
                .in_left_valid      (array_left_in_valid[i]),

                .in_left_v_element  (array_left_v_in_element[i]),
                .in_left_v_scale    (array_left_v_in_scale[i]),
                .in_left_v_valid    (v2_data_for_mv_in_valid[i]),

                .m_out_fp           (gemm_result[i]),
                .m_out_valid        (gemm_result_valid[i]),
                .v_out_fp           (gemv_result[i])
            );
        end
    endgenerate

    logic gebm_result_valid;
    logic [COMPUTE_DIM- 1: 0][COMPUTE_DIM- 1: 0][ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] gebm_result;

    logic quantise_data_in_valid;
    logic casted_data_in_valid;
    logic quantised_result_valid;
    logic block_data_in_valid;
    logic unrolled_data_out_valid;
    logic result_data_valid;
    logic [K - 1 : 0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0] result_data, unrolled_data_out;
    localparam GEBM_OUT_DIM = COMPUTE_DIM * COMPUTE_DIM;
    localparam MAX_K_GEBM_OUT_DIM = (K > GEBM_OUT_DIM) ? K : GEBM_OUT_DIM;

    mx_sum_across_sa #(
        .ACC_FP_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
        .ACC_FP_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
        .COMPUTE_DIM        (COMPUTE_DIM),
        .SYS_ARRAY_AMOUNT   (SYS_ARRAY_AMOUNT)
    ) sa_sum_across_inst (
        .clk(clk),
        .rst(rst),
        .m_in_data  (gemm_result),
        .in_valid   (&gemm_result_valid),
        .m_out_data (gebm_result),
        .out_valid  (gebm_result_valid)
    );


    // -----------------------------
    // Result Data Path
    // -----------------------------

    logic gemv_result_out_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            feed_counter            <= '0;
            gemv_result_out_valid   <= 1'b0;
            ready_to_load_output    <= 1'b0;
            mcu_active       <= 1'b0;
        end else begin
            if (complete_loading_q & (control_in_exe == MM_WO)) begin
                feed_counter        <= '0;
                mcu_active   <= 1'b1;
            end else if (mcu_active) begin
                if (feed_counter == 2 * COMPUTE_DIM + SYSTOLIC_PROCESSING_OVERHEAD) begin
                    feed_counter         <= '0;
                    mcu_active    <= 1'b0;
                    ready_to_load_output <= 1'b1;
                end else begin
                    feed_counter <= feed_counter + 'b1;
                    ready_to_load_output <= 1'b0;
                end
            end else begin
                mcu_active <= 1'b0;
                ready_to_load_output <= 1'b0;
            end
            gemv_result_out_valid <= (control_in_exe == MV_WO);
        end
    end



    // -----------------------------
    // Quantize into Required Precision for Storage (Vector SRAM)
    // -----------------------------
    // Note, the reason for K dim is because, it need to be used for both GEMM and GEMV

    logic [MAX_K_GEBM_OUT_DIM - 1 : 0][ACC_FP_EXP_WIDTH + ACC_FP_MANT_WIDTH : 0]    stored_result_v;
    logic [MAX_K_GEBM_OUT_DIM - 1 : 0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0]            quantised_result_v;
    logic [MAX_K_GEBM_OUT_DIM * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1) - 1 : 0]         stored_quantized_result;
    logic [MAX_K_GEBM_OUT_DIM - 1 : 0] split_quantised_result_valid;

    always_comb begin
        if (sa_control == 1'b0) begin
            // GEMM
            quantise_data_in_valid  = gebm_result_valid;
            block_data_in_valid     = quantised_result_valid;
            result_data             = unrolled_data_out;
            result_data_valid       = unrolled_data_out_valid;
        end else begin
            // GEMV
            quantise_data_in_valid  = gemv_result_out_valid;
            block_data_in_valid     = 1'b0;
            result_data             = stored_quantized_result[K * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1) - 1 : 0];
            result_data_valid       = quantised_result_valid;
        end
    end

    generate
        if (K > GEBM_OUT_DIM) begin
            always_comb begin
                if (sa_control == 1'b0) begin
                    // GEMM
                    stored_result_v = {{((K - GEBM_OUT_DIM) * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1)){1'b0}}, gebm_result};
                end else begin
                    // GEMV
                    stored_result_v = gemv_result;
                end
            end

            for (genvar i = 0; i < K; i++) begin : gen_quantize
                fp_ieee_casting #(
                    .IN_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                    .IN_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                    .OUT_EXP_WIDTH  (FP_EXP_WIDTH),
                    .OUT_MANT_WIDTH (FP_MANT_WIDTH)
                ) cast_inst (
                    .clk            (clk),
                    .rst            (rst),
                    .data_in_valid  (quantise_data_in_valid),
                    .data_in        (stored_result_v[i]),
                    .data_out_valid (split_quantised_result_valid[i]),
                    .data_out       (quantised_result_v[i])
                );
            end

        end else begin
            always_comb begin
                if (sa_control == 1'b0) begin
                    // GEMM
                    stored_result_v = gebm_result;
                end else begin
                    // GEMV
                    stored_result_v = {{(GEBM_OUT_DIM - K){1'b0}}, gemv_result};
                end
            end

            for (genvar i = 0; i < GEBM_OUT_DIM; i++) begin : gen_quantize
                fp_ieee_casting #(
                    .IN_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                    .IN_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                    .OUT_EXP_WIDTH  (FP_EXP_WIDTH),
                    .OUT_MANT_WIDTH (FP_MANT_WIDTH)
                ) cast_inst (
                    .clk            (clk),
                    .rst            (rst),
                    .data_in_valid  (quantise_data_in_valid),
                    .data_in        (stored_result_v[i]),
                    .data_out_valid (split_quantised_result_valid[i]),
                    .data_out       (quantised_result_v[i])
                );
            end
        end

        assign casted_data_in_valid = &split_quantised_result_valid;

        register_slice_wo_hs #(
            .DATA_WIDTH(MAX_K_GEBM_OUT_DIM * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1))
        ) quantized_result_buffer (
            .clk(clk),
            .rst(rst),
            .data_in        (quantised_result_v),
            .data_in_valid  (casted_data_in_valid),
            .data_out       (stored_quantized_result),
            .data_out_valid (quantised_result_valid)
        );
    endgenerate


    // -----------------------------
    // Storing the computed result and write to the Vector SRAM
    // -----------------------------

    block_data_buffer #(
        .M(M),
        .N(N),
        .K(K),
        .FP_EXP_WIDTH       (FP_EXP_WIDTH),
        .FP_MANT_WIDTH      (FP_MANT_WIDTH),
        .ACC_ADDR_WIDTH     (ACC_ADDR_WIDTH)
    ) hold_and_unroll_for_gemm (
        .clk(clk),
        .rst(rst),
        .acc_waddr              (acc_waddr),
        .acc_waddr_valid        (fetch_next_acc_waddr_valid),
        .acc_waddr_ready        (fetch_next_acc_waddr_ready),
        .wait_for_output        (wait_for_output),
        .block_data_in          (stored_quantized_result[0 +: GEBM_OUT_DIM * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1)]),
        .block_data_valid       (block_data_in_valid),
        .unrolled_data_out      (unrolled_data_out),
        .unrolled_data_out_valid(unrolled_data_out_valid)
    );

    logic v_result_valid;

    register_slice_wo_hs #(
        .DATA_WIDTH (K * (FP_EXP_WIDTH + FP_MANT_WIDTH + 1))
    ) result_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (result_data),
        .data_in_valid  (result_data_valid),
        .data_out       (v_result),
        .data_out_valid (v_result_valid)
    );

    assign v_result_write_req = v_result_valid;
endmodule
