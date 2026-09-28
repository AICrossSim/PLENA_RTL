`timescale 1ns / 1ps

/*
Module      : MXINT Sum-Across (cross-K-segment reduction, integer domain)
Timing      : Sequential (1 cycle latency, like mx_sum_across_sa)
Description : Reduces SYS_ARRAY_AMOUNT partial MXINT tiles (one per K-segment)
              into a single MXINT tile, staying in the INTEGER domain so the
              expensive INT→FP conversion happens only ONCE per output position
              downstream (instead of once per PE).

              Each partial p at position (i,j) is (int_p, scale_p) where the real
              value is  int_p · 2^(scale_p - bias) · 2^-frac.  The segments have
              DIFFERENT scales, so the integers cannot be added directly — they
              are aligned to a common exponent first.

Alignment (per output position, combinational):
   max_scale = max_p scale_p
   min_scale = min_p scale_p
   eff_min   = max(min_scale, max_scale - MAX_SHIFT)   ← left-align to the low
                                                          scale, but never spread
                                                          the accumulator wider
                                                          than MAX_SHIFT bits
   for each p:
     diff = scale_p - eff_min
     diff >= 0 : aligned = int_p <<<  diff     (lossless left shift)
     diff <  0 : aligned = int_p >>> (-diff)    (segment far below eff_min:
                                                 right-shift, low bits lost)
   sum_int = Σ_p aligned          (width = INT_WIDTH + MAX_SHIFT + clog2(SYS))
   out_int = sum_int,  out_scale = eff_min

When all segment scales are within MAX_SHIFT of each other (the common case for
one GEMM tile), the reduction is fully lossless. MAX_SHIFT only bounds the worst
case so the accumulator width stays fixed.
*/

module mxint_sum_across #(
    parameter INT_WIDTH         = 12,  // per-segment accumulator width (signed)
    parameter SCALE_WIDTH       = 9,   // per-segment biased scale width
    parameter COMPUTE_DIM       = 4,   // output tile edge (BLEN)
    parameter SYS_ARRAY_AMOUNT  = 4,   // number of K-segments (BLOCK_NUM)
    parameter MAX_SHIFT         = 16,  // max alignment spread (accumulator headroom)
    localparam SUM_EXTRA        = $clog2(SYS_ARRAY_AMOUNT + 1),
    localparam OUT_INT_WIDTH    = INT_WIDTH + MAX_SHIFT + SUM_EXTRA
)(
    input  logic clk,
    input  logic rst,

    input  logic [SYS_ARRAY_AMOUNT-1:0][COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][INT_WIDTH-1:0]   m_in_int,
    input  logic [SYS_ARRAY_AMOUNT-1:0][COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][SCALE_WIDTH-1:0]  m_in_scale,
    input  logic in_valid,

    output logic [COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][OUT_INT_WIDTH-1:0]  m_out_int,
    output logic [COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][SCALE_WIDTH-1:0]    m_out_scale,
    output logic out_valid
);

    // Register the inputs (mirrors mx_sum_across_sa's pipeline register)
    logic [SYS_ARRAY_AMOUNT-1:0][COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][INT_WIDTH-1:0]  in_int_q;
    logic [SYS_ARRAY_AMOUNT-1:0][COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][SCALE_WIDTH-1:0] in_scale_q;
    logic in_valid_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            in_valid_q <= 1'b0;
        end else begin
            in_int_q   <= m_in_int;
            in_scale_q <= m_in_scale;
            in_valid_q <= in_valid;
        end
    end

    // Combinational reduce per output position (one always_comb per (i,j) so
    // every signal is assigned on every path — no latch inference).
    logic [COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][OUT_INT_WIDTH-1:0] sum_int_c;
    logic [COMPUTE_DIM-1:0][COMPUTE_DIM-1:0][SCALE_WIDTH-1:0]   eff_min_c;

    generate
        for (genvar gi = 0; gi < COMPUTE_DIM; gi++) begin : g_reduce_row
            for (genvar gj = 0; gj < COMPUTE_DIM; gj++) begin : g_reduce_col
                logic [SCALE_WIDTH-1:0]   max_scale, min_scale, floor_scale, eff_min;
                logic [OUT_INT_WIDTH-1:0] acc;
                logic signed [OUT_INT_WIDTH-1:0] ext;
                logic signed [SCALE_WIDTH:0]     diff;  // signed, one extra bit

                always_comb begin
                    max_scale = in_scale_q[0][gi][gj];
                    min_scale = in_scale_q[0][gi][gj];
                    for (int p = 1; p < SYS_ARRAY_AMOUNT; p++) begin
                        if (in_scale_q[p][gi][gj] > max_scale) max_scale = in_scale_q[p][gi][gj];
                        if (in_scale_q[p][gi][gj] < min_scale) min_scale = in_scale_q[p][gi][gj];
                    end

                    // eff_min = max(min_scale, max_scale - MAX_SHIFT), saturating at 0
                    floor_scale = (max_scale > MAX_SHIFT[SCALE_WIDTH-1:0])
                                  ? (max_scale - MAX_SHIFT[SCALE_WIDTH-1:0]) : '0;
                    eff_min = (min_scale > floor_scale) ? min_scale : floor_scale;

                    // align each segment to eff_min and accumulate
                    acc = '0;
                    ext = '0;
                    diff = '0;
                    for (int p = 0; p < SYS_ARRAY_AMOUNT; p++) begin
                        ext  = OUT_INT_WIDTH'($signed(in_int_q[p][gi][gj]));
                        diff = $signed({1'b0, in_scale_q[p][gi][gj]}) - $signed({1'b0, eff_min});
                        if (diff >= 0)
                            acc = acc + (ext <<< diff);
                        else
                            acc = acc + (ext >>> (-diff));
                    end

                    sum_int_c[gi][gj] = acc;
                    eff_min_c[gi][gj] = eff_min;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
        end else begin
            m_out_int   <= sum_int_c;
            m_out_scale <= eff_min_c;
            out_valid   <= in_valid_q;
        end
    end

endmodule
