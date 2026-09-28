`timescale 1ns / 1ps

/*
Module      : MXINT Mini Systolic Array (roller buffers + PE grid)
Timing      : Sequential — output-stationary systolic GEMM
Description : Computes one output tile C[BLOCK_DIM][BLOCK_DIM] where each element
              is a length-ACC_DEPTH dot product:
                  C[i][j] = Σ_{k=0..ACC_DEPTH-1} A[i][k] · B[k][j]

              Front-end: 2·BLOCK_DIM roller_buffer instances (PISO).
                * top_roller[j]  holds B's column j  (ACC_DEPTH K-values)
                * left_roller[i] holds A's row i      (ACC_DEPTH K-values)
              Back-end: BLOCK_DIM × BLOCK_DIM grid of mxint_default_pe.

LOAD phase (BLOCK_DIM cycles, load_valid high):
   - cycle t loads A's row t into left_roller[t] and B's column t into
     top_roller[t], plus that row/col's MXINT block scale.

COMPUTE phase (driven by the rollers, no extra control):
   - roller[t] loaded at cycle t emits its K-values starting at cycle t+1.
   - Adjacent rollers are loaded one cycle apart, so their serial outputs are
     skewed by one cycle — the exact systolic wavefront. PE(i,j) sees both
     valids high during cycles (i+j+1) .. (i+j+ACC_DEPTH) and accumulates
     exactly ACC_DEPTH MACs.

out_valid asserts when ALL PEs have completed. To run another tile, assert rst.
*/

module mxint_mini_systolic_array #(
    // MXINT data format
    parameter MX_T_INT_WIDTH      = 4,
    parameter MX_L_INT_WIDTH      = 4,
    parameter MXINT_SCALE_WIDTH   = 8,

    // Array geometry
    parameter BLOCK_DIM           = 4,

    // K-segment depth (= MXINT scale block size); each PE accumulates this many MACs
    parameter ACC_DEPTH           = 16,
    parameter ACC_EXPAND_WIDTH    = $clog2(ACC_DEPTH),
    localparam OUT_INT_WIDTH      = MX_T_INT_WIDTH + MX_L_INT_WIDTH + ACC_EXPAND_WIDTH,
    localparam OUT_SCALE_WIDTH    = MXINT_SCALE_WIDTH + 1
)(
    input  logic clk,
    input  logic rst,

    // ---- LOAD interface (BLOCK_DIM cycles, load_valid high) ----
    // cycle t: A's row t (one MXINT block) and B's column t (one MXINT block).
    input  logic [ACC_DEPTH-1:0][MX_L_INT_WIDTH-1:0] load_a_row,   // A[t][0..ACC_DEPTH-1]
    input  logic [ACC_DEPTH-1:0][MX_T_INT_WIDTH-1:0] load_b_col,   // B[0..ACC_DEPTH-1][t]
    input  logic [MXINT_SCALE_WIDTH-1:0]             load_a_scale, // scale of A row t
    input  logic [MXINT_SCALE_WIDTH-1:0]             load_b_scale, // scale of B col t
    input  logic                                     load_valid,

    // ---- Result: per-PE lossless INT accumulator + biased scale ----
    // out_scale[i][j] = a_scale[i] + b_scale[j] - bias  (per PE)
    output logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0][OUT_INT_WIDTH-1:0]   out_int,
    output logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0][OUT_SCALE_WIDTH-1:0] out_scale,
    output logic out_valid
);

    // ==============================================================================================
    // Load counter — selects which roller buffer (row/col index) loads this cycle.
    // ==============================================================================================
    localparam LOAD_CNT_W = $clog2(BLOCK_DIM + 1);
    logic [LOAD_CNT_W-1:0] load_cnt;

    always_ff @(posedge clk) begin
        if (rst)
            load_cnt <= '0;
        else if (load_valid && (load_cnt != (BLOCK_DIM-1)))
            load_cnt <= load_cnt + 1'b1;
    end

    // ==============================================================================================
    // Per-row / per-col MXINT block scale latches (one scale per A row / B col).
    // ==============================================================================================
    logic [BLOCK_DIM-1:0][MXINT_SCALE_WIDTH-1:0] a_scale_buf;
    logic [BLOCK_DIM-1:0][MXINT_SCALE_WIDTH-1:0] b_scale_buf;

    always_ff @(posedge clk) begin
        if (load_valid) begin
            a_scale_buf[load_cnt] <= load_a_scale;
            b_scale_buf[load_cnt] <= load_b_scale;
        end
    end

    // ==============================================================================================
    // Roller buffers — top (B columns) and left (A rows). roller[t] is loaded at
    // cycle t (load_cnt == t) and emits its K-values serially from cycle t+1.
    // ==============================================================================================
    logic [BLOCK_DIM-1:0][MX_T_INT_WIDTH-1:0]  top_roller_out;
    logic [BLOCK_DIM-1:0]                        top_roller_valid;
    logic [BLOCK_DIM-1:0][MX_L_INT_WIDTH-1:0]  left_roller_out;
    logic [BLOCK_DIM-1:0]                        left_roller_valid;

    generate
        for (genvar t = 0; t < BLOCK_DIM; t++) begin : g_rollers
            roller_buffer #(
                .DATA_WIDTH (MX_T_INT_WIDTH),
                .DEPTH      (ACC_DEPTH)
            ) top_roller (
                .clk            (clk),
                .rst            (rst),
                .data_in        (load_b_col),
                .load           (load_valid && (load_cnt == t[LOAD_CNT_W-1:0])),
                .data_out       (top_roller_out[t]),
                .data_out_valid (top_roller_valid[t])
            );

            roller_buffer #(
                .DATA_WIDTH (MX_L_INT_WIDTH),
                .DEPTH      (ACC_DEPTH)
            ) left_roller (
                .clk            (clk),
                .rst            (rst),
                .data_in        (load_a_row),
                .load           (load_valid && (load_cnt == t[LOAD_CNT_W-1:0])),
                .data_out       (left_roller_out[t]),
                .data_out_valid (left_roller_valid[t])
            );
        end
    endgenerate

    // ==============================================================================================
    // PE grid wave wires (top→bottom, left→right; propagated by PE registers)
    // ==============================================================================================
    logic [BLOCK_DIM:0]   [BLOCK_DIM-1:0][MX_T_INT_WIDTH-1:0]    vert_elem;
    logic [BLOCK_DIM:0]   [BLOCK_DIM-1:0][MXINT_SCALE_WIDTH-1:0] vert_scale;
    logic [BLOCK_DIM-1:0] [BLOCK_DIM:0]  [MX_L_INT_WIDTH-1:0]    hori_elem;
    logic [BLOCK_DIM-1:0] [BLOCK_DIM:0]  [MXINT_SCALE_WIDTH-1:0] hori_scale;

    logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0] top_valid_reg;
    logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0] left_valid_reg;
    logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0] pe_top_valid;
    logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0] pe_left_valid;

    // Edge fill from the roller outputs
    generate
        for (genvar j = 0; j < BLOCK_DIM; j++) begin : fill_top_edge
            assign vert_elem[0][j]  = top_roller_out[j];
            assign vert_scale[0][j] = b_scale_buf[j];
        end
        for (genvar i = 0; i < BLOCK_DIM; i++) begin : fill_left_edge
            assign hori_elem[i][0]  = left_roller_out[i];
            assign hori_scale[i][0] = a_scale_buf[i];
        end
    endgenerate

    // Valid wiring (edge = roller valid, interior = registered chain)
    generate
        for (genvar i = 0; i < BLOCK_DIM; i++) begin : valid_wires_row
            for (genvar j = 0; j < BLOCK_DIM; j++) begin : valid_wires_col
                if (i == 0) assign pe_top_valid[i][j] = top_roller_valid[j];
                else        assign pe_top_valid[i][j] = top_valid_reg[i-1][j];

                if (j == 0) assign pe_left_valid[i][j] = left_roller_valid[i];
                else        assign pe_left_valid[i][j] = left_valid_reg[i][j-1];
            end
        end
    endgenerate

    generate
        for (genvar i = 0; i < BLOCK_DIM; i++) begin : valid_prop_row
            for (genvar j = 0; j < BLOCK_DIM; j++) begin : valid_prop_col
                always_ff @(posedge clk) begin
                    if (rst) begin
                        top_valid_reg[i][j]  <= 1'b0;
                        left_valid_reg[i][j] <= 1'b0;
                    end else begin
                        top_valid_reg[i][j]  <= pe_top_valid[i][j];
                        left_valid_reg[i][j] <= pe_left_valid[i][j];
                    end
                end
            end
        end
    endgenerate

    // ==============================================================================================
    // PE grid
    // ==============================================================================================
    logic [BLOCK_DIM-1:0][BLOCK_DIM-1:0] pe_out_valid;

    generate
        for (genvar i = 0; i < BLOCK_DIM; i++) begin : pe_row
            for (genvar j = 0; j < BLOCK_DIM; j++) begin : pe_col
                mxint_default_pe #(
                    .MX_T_INT_WIDTH    (MX_T_INT_WIDTH),
                    .MX_L_INT_WIDTH    (MX_L_INT_WIDTH),
                    .MXINT_SCALE_WIDTH (MXINT_SCALE_WIDTH),
                    .ACC_DEPTH         (ACC_DEPTH)
                ) pe (
                    .clk               (clk),
                    .rst               (rst),
                    .in_top_element    (vert_elem[i][j]),
                    .in_top_scale      (vert_scale[i][j]),
                    .in_top_valid      (pe_top_valid[i][j]),
                    .in_left_element   (hori_elem[i][j]),
                    .in_left_scale     (hori_scale[i][j]),
                    .in_left_valid     (pe_left_valid[i][j]),
                    .out_bottom_element(vert_elem[i+1][j]),
                    .out_bottom_scale  (vert_scale[i+1][j]),
                    .out_right_element (hori_elem[i][j+1]),
                    .out_right_scale   (hori_scale[i][j+1]),
                    .out_int           (out_int[i][j]),
                    .out_scale         (out_scale[i][j]),
                    .out_valid         (pe_out_valid[i][j])
                );
            end
        end
    endgenerate

    // ==============================================================================================
    // out_valid: every PE has completed.
    // ==============================================================================================
    assign out_valid = &pe_out_valid;

endmodule
