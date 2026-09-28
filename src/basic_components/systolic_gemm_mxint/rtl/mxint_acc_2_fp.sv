`timescale 1ns / 1ps

/*
Module      : MXINT Accumulator → IEEE FP
Timing      : Combinational
Description : Converts a lossless MXINT dot-product accumulator (2's complement
              signed integer) plus its biased block scale into an IEEE FP value.

              The mxint PE accumulates  acc_int = Σ (top_mag · left_mag)  as a
              2's-complement integer, and tracks  scale_sum = t_scale + l_scale
              - SCALE_BIAS  (biased). Since each operand mantissa is mag/2^(W-1):

                  real = acc_int · 2^-(ACC_FRAC_WIDTH) · 2^(scale_sum - SCALE_BIAS)

              where ACC_FRAC_WIDTH = (MX_T-1) + (MX_L-1) = MX_T + MX_L - 2.

              So we feed fp_ieee_normalize a fixed-point mantissa with
              ACC_FRAC_WIDTH fractional bits and an unbiased exponent
              (scale_sum - SCALE_BIAS), then cast to the target FP format.
              Same normalize → exponent-cast → mantissa-cast pipeline as
              mx_int_2_fp_unary, but the input is a real 2's-complement integer
              (not a sign-magnitude denormal fraction).
*/

module mxint_acc_2_fp #(
    parameter ACC_WIDTH         = 12,  // 2's complement accumulator width
    parameter ACC_FRAC_WIDTH    = 6,   // fractional bits = MX_T + MX_L - 2
    parameter IN_SCALE_WIDTH    = 9,   // biased scale_sum width (= MXINT_SCALE_WIDTH+1)
    parameter MXINT_SCALE_WIDTH = 8,   // original MXINT scale width (for the bias)
    parameter FP_EXP_WIDTH      = 8,
    parameter FP_MANT_WIDTH     = 7,
    // PIPE=1 registers the fp_ieee_normalize output, splitting the deep int->FP
    // convert (CLZ + barrel-shift normalize | exp/mant casts) into two stages.
    // Adds 1 cycle of latency; the systolic MCU realigns the accumulate pulse.
    // Default 0 = the prior purely-combinational behaviour.
    parameter PIPE              = 0
)(
    input  logic                               clk,        // used only when PIPE=1
    input  logic signed [ACC_WIDTH-1:0]        acc_in,
    input  logic [IN_SCALE_WIDTH-1:0]          scale_in,   // biased: t_scale+l_scale-bias
    output logic [FP_EXP_WIDTH + FP_MANT_WIDTH : 0] fp_out
);
    localparam SCALE_BIAS = (1 << (MXINT_SCALE_WIDTH - 1)) - 1;

    // fp_ieee_normalize requires OUT_MANT_WIDTH >= IN_FIXED_WIDTH - 2
    localparam INTER_MANT_WIDTH = (FP_MANT_WIDTH >= ACC_WIDTH - 2) ? FP_MANT_WIDTH : (ACC_WIDTH - 2);

    // signed exponent fed to normalize = scale_sum - SCALE_BIAS  (extra bit for sign)
    localparam UNB_EXP_WIDTH        = IN_SCALE_WIDTH + 1;
    localparam NORM_OUT_EXP_WIDTH   = UNB_EXP_WIDTH + 1;

    logic signed [UNB_EXP_WIDTH-1:0] unbiased_exp;
    assign unbiased_exp = $signed({1'b0, scale_in}) - SCALE_BIAS;

    logic [NORM_OUT_EXP_WIDTH + INTER_MANT_WIDTH : 0] normalized;

    fp_ieee_normalize #(
        .IN_FIXED_WIDTH      (ACC_WIDTH),
        .IN_FIXED_FRAC_WIDTH (ACC_FRAC_WIDTH),
        .IN_EXP_WIDTH        (UNB_EXP_WIDTH),
        .OUT_MANT_WIDTH      (INTER_MANT_WIDTH)
    ) normalize (
        .signed_mant (acc_in),
        .signed_exp  (unbiased_exp),
        .fp_out      (normalized)
    );

    // Optional pipeline register on the normalize output (see PIPE).
    logic [NORM_OUT_EXP_WIDTH + INTER_MANT_WIDTH : 0] normalized_r;
    generate
        if (PIPE) begin : g_norm_reg
            always_ff @(posedge clk) normalized_r <= normalized;
        end else begin : g_norm_comb
            assign normalized_r = normalized;
        end
    endgenerate

    // Cast exponent (and mantissa if widened) down to the target FP format.
    generate
        if (INTER_MANT_WIDTH == FP_MANT_WIDTH) begin : g_exp_only
            fp_ieee_exponent_casting #(
                .IN_EXP_WIDTH  (NORM_OUT_EXP_WIDTH),
                .OUT_EXP_WIDTH (FP_EXP_WIDTH),
                .MANT_WIDTH    (FP_MANT_WIDTH)
            ) exp_cast (
                .data_in  (normalized_r),
                .data_out (fp_out)
            );
        end else begin : g_exp_and_mant
            logic [FP_EXP_WIDTH + INTER_MANT_WIDTH : 0] exp_casted;

            fp_ieee_exponent_casting #(
                .IN_EXP_WIDTH  (NORM_OUT_EXP_WIDTH),
                .OUT_EXP_WIDTH (FP_EXP_WIDTH),
                .MANT_WIDTH    (INTER_MANT_WIDTH)
            ) exp_cast (
                .data_in  (normalized_r),
                .data_out (exp_casted)
            );

            fp_ieee_mantissa_casting #(
                .EXP_WIDTH      (FP_EXP_WIDTH),
                .IN_MANT_WIDTH  (INTER_MANT_WIDTH),
                .OUT_MANT_WIDTH (FP_MANT_WIDTH)
            ) mant_cast (
                .data_in  (exp_casted),
                .data_out (fp_out)
            );
        end
    endgenerate

endmodule
