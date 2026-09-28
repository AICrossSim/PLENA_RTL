`timescale 1ns / 1ps

/*
Module      : Unary Conversion from MX-INT to FP
Timing      : Sequential (1 cycle latency)
Description : Converts MXINT (sign-magnitude with biased scale) to IEEE FP
              MXINT format: [sign(1)][magnitude(N-1)] with biased shared exponent
              Value = (-1)^sign × (magnitude / 2^(N-1)) × 2^(scale - bias)
Status      :
*/

module mx_int_2_fp_unary #(
    parameter MXINT_SCALE_WIDTH = 8,
    parameter MXINT_WIDTH = 8,
    parameter FP_EXP_WIDTH = 4,
    parameter FP_MANT_WIDTH = 3
)(
    input   logic clk,
    input   logic rst,
    input   logic data_in_valid,
    input   logic [MXINT_WIDTH - 1 : 0] element_data_in,
    input   logic [MXINT_SCALE_WIDTH - 1 : 0] scale_data_in,
    output  logic data_out_valid,
    output  logic [FP_EXP_WIDTH + FP_MANT_WIDTH : 0] fp_out
);
    // fp_ieee_normalize requires: OUT_MANT_WIDTH >= IN_FIXED_WIDTH - 2
    // So intermediate mantissa width must be at least MXINT_WIDTH - 2
    // Use the larger of FP_MANT_WIDTH and (MXINT_WIDTH - 2) for intermediate
    localparam INTER_MANT_WIDTH = (FP_MANT_WIDTH >= MXINT_WIDTH - 2) ? FP_MANT_WIDTH : (MXINT_WIDTH - 2);

    // MXINT scale is biased (like IEEE FP exponent), need to unbias before normalization
    localparam SCALE_BIAS = 2**(MXINT_SCALE_WIDTH - 1) - 1;
    localparam UNBIASED_SCALE_WIDTH = MXINT_SCALE_WIDTH + 1;  // Extra bit for signed result
    localparam NORMALIZE_OUT_EXP_WIDTH = UNBIASED_SCALE_WIDTH + 1;

    // MXINT element format: [sign(1)][magnitude(MXINT_WIDTH-1)] - sign-magnitude format
    // Need to convert to two's complement for fp_ieee_normalize
    logic [MXINT_WIDTH - 2 : 0] element_magnitude;
    logic element_sign;
    logic signed [MXINT_WIDTH - 1 : 0] element_twos_complement;

    assign element_sign = element_data_in[MXINT_WIDTH - 1];
    assign element_magnitude = element_data_in[MXINT_WIDTH - 2 : 0];
    // Convert sign-magnitude to two's complement
    assign element_twos_complement = element_sign ? -$signed({1'b0, element_magnitude}) : $signed({1'b0, element_magnitude});

    logic signed [UNBIASED_SCALE_WIDTH - 1 : 0] unbiased_scale;
    logic [NORMALIZE_OUT_EXP_WIDTH + INTER_MANT_WIDTH:0] normalized_data;
    logic [FP_EXP_WIDTH + FP_MANT_WIDTH : 0] reg_fp_out;

    // Unbias the scale: actual_exponent = scale - SCALE_BIAS
    assign unbiased_scale = $signed({1'b0, scale_data_in}) - SCALE_BIAS;

    // First normalize to intermediate width (satisfies fp_ieee_normalize constraint)
    // MXINT is "all denorm" - magnitude represents fraction in [0,1), so all bits are fractional
    fp_ieee_normalize #(
        .IN_FIXED_WIDTH         (MXINT_WIDTH),
        .IN_FIXED_FRAC_WIDTH    (MXINT_WIDTH - 1),  // All magnitude bits are fractional (denorm)
        .IN_EXP_WIDTH           (UNBIASED_SCALE_WIDTH),
        .OUT_MANT_WIDTH         (INTER_MANT_WIDTH)
    ) fp_normalize (
        .signed_mant    (element_twos_complement),
        .signed_exp     (unbiased_scale),
        .fp_out         (normalized_data)
    );

    // Then cast exponent and mantissa to target widths
    generate
        if (INTER_MANT_WIDTH == FP_MANT_WIDTH) begin : gen_direct_cast
            // No mantissa casting needed, just exponent casting
            fp_ieee_exponent_casting #(
                .IN_EXP_WIDTH       (NORMALIZE_OUT_EXP_WIDTH),
                .OUT_EXP_WIDTH      (FP_EXP_WIDTH),
                .MANT_WIDTH         (FP_MANT_WIDTH)
            ) fp_casting (
                .data_in    (normalized_data),
                .data_out   (reg_fp_out)
            );
        end else begin : gen_full_cast
            // Need both exponent and mantissa casting
            logic [FP_EXP_WIDTH + INTER_MANT_WIDTH:0] exp_casted_data;

            fp_ieee_exponent_casting #(
                .IN_EXP_WIDTH       (NORMALIZE_OUT_EXP_WIDTH),
                .OUT_EXP_WIDTH      (FP_EXP_WIDTH),
                .MANT_WIDTH         (INTER_MANT_WIDTH)
            ) fp_exp_casting (
                .data_in    (normalized_data),
                .data_out   (exp_casted_data)
            );

            fp_ieee_mantissa_casting #(
                .EXP_WIDTH          (FP_EXP_WIDTH),
                .IN_MANT_WIDTH      (INTER_MANT_WIDTH),
                .OUT_MANT_WIDTH     (FP_MANT_WIDTH)
            ) fp_mant_casting (
                .data_in    (exp_casted_data),
                .data_out   (reg_fp_out)
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            fp_out <= '0;
            data_out_valid <= 1'b0;
        end else begin
            fp_out <= reg_fp_out;
            data_out_valid <= data_in_valid;
        end
    end

endmodule