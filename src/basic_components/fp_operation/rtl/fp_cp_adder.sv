`timescale 1ns / 1ps
/*
Module      : Floating Point Configurable Precision Adder (With Sign)
Timing      : 4-stage pipeline (adder -> normalize -> exponent cast -> mantissa cast)
Description : Adds two FP numbers with different exponents and signs.
              Aligns mantissas, preserves full precision (no bits discarded).
              Output format: {sign, exp_out, mant_out}.
              No rounding.
              It needs normalisation.
              The lossy part will be at the mantissa adder
Status      : Passed Simple Tests
*/

module fp_cp_adder #(
    parameter int EXP_WIDTH = 6,
    parameter int MANT_WIDTH = 5,
    // Amount of bits needed to shift mantissas for alignment
    parameter int EXT_MANT_WIDTH = 0,
    // Need to increase exp width by 1 to handle overflow
    parameter int EXT_EXP_WIDTH = 0
)(
    input  logic clk,
    input  logic rst,
    input  logic data_in_valid,
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_a,  // {sign, exp, mant}
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_b,
    output logic [EXP_WIDTH + EXT_EXP_WIDTH + MANT_WIDTH + EXT_MANT_WIDTH : 0] data_out,
    output logic data_out_valid
);

    localparam int IN_EXP_WIDTH = EXP_WIDTH;
    localparam int IN_FIXED_WIDTH = MANT_WIDTH + 2;
    localparam int IN_FIXED_FRAC_WIDTH = MANT_WIDTH;

    localparam int ADDER_OUT_EXP_WIDTH = IN_EXP_WIDTH + 1;
    localparam int ADDER_OUT_FIXED_WIDTH = IN_FIXED_WIDTH + IN_FIXED_FRAC_WIDTH + 1;
    localparam int ADDER_OUT_FIXED_FRAC_WIDTH = IN_FIXED_FRAC_WIDTH + IN_FIXED_FRAC_WIDTH;

    localparam int NORMALIZE_OUT_EXP_WIDTH = ADDER_OUT_EXP_WIDTH + 1;
    localparam int NORMALIZE_OUT_MANT_WIDTH = ADDER_OUT_FIXED_WIDTH - 1;

    // Internal signal declarations
    logic signed [IN_EXP_WIDTH:0]   signed_exp_a, signed_exp_b;
    logic signed [IN_FIXED_WIDTH - 1:0] signed_mant_a, signed_mant_b;

    logic signed [ADDER_OUT_EXP_WIDTH - 1:0]    signed_exp_out;
    logic signed [ADDER_OUT_FIXED_WIDTH - 1:0]  signed_mant_out;
    logic signed [ADDER_OUT_EXP_WIDTH - 1:0]    p1_signed_exp_out;
    logic signed [ADDER_OUT_FIXED_WIDTH - 1:0]  p1_signed_mant_out;

    logic p0_align_out_valid;  // Valid after fp_adder internal pipeline stage
    logic p1_adder_out_valid;
    logic p2_normalize_out_valid;

    logic [NORMALIZE_OUT_EXP_WIDTH + NORMALIZE_OUT_MANT_WIDTH: 0] normalized_data;
    logic [NORMALIZE_OUT_EXP_WIDTH + NORMALIZE_OUT_MANT_WIDTH: 0] p2_normalized_data;

    // Instantiate fp_ieee_partition for data_a
    logic signed [IN_EXP_WIDTH:0]       comb_exp_a;
    logic signed [IN_FIXED_WIDTH - 1:0] comb_mant_a;
    fp_ieee_partition #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) partition_a (
        .data_in        (data_a),
        .signed_exp     (comb_exp_a),
        .signed_mant    (comb_mant_a)
    );

    // Instantiate fp_ieee_partition for data_b
    logic signed [IN_EXP_WIDTH:0]       comb_exp_b;
    logic signed [IN_FIXED_WIDTH - 1:0] comb_mant_b;
    fp_ieee_partition #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) partition_b (
        .data_in        (data_b),
        .signed_exp     (comb_exp_b),
        .signed_mant    (comb_mant_b)
    );

    // Pipeline register after partition — breaks input_regstore → fp_adder route
    logic p_part_valid;
    always_ff @(posedge clk) begin
        if (rst) begin
            signed_exp_a  <= '0;
            signed_mant_a <= '0;
            signed_exp_b  <= '0;
            signed_mant_b <= '0;
            p_part_valid  <= 1'b0;
        end else begin
            signed_exp_a  <= comb_exp_a;
            signed_mant_a <= comb_mant_a;
            signed_exp_b  <= comb_exp_b;
            signed_mant_b <= comb_mant_b;
            p_part_valid  <= data_in_valid;
        end
    end

    // Instantiate fp_adder
    fp_adder #(
        .IN_EXP_WIDTH       (IN_EXP_WIDTH + 1),
        .IN_FIX_WIDTH       (IN_FIXED_WIDTH),
        .IN_FIX_FRAC_WIDTH  (IN_FIXED_FRAC_WIDTH),
        .OUT_EXP_WIDTH      (ADDER_OUT_EXP_WIDTH),
        .OUT_FIX_WIDTH      (ADDER_OUT_FIXED_WIDTH),
        .OUT_FIX_FRAC_WIDTH (ADDER_OUT_FIXED_FRAC_WIDTH),
        .PIPELINE_STAGE     (1)
    ) fp_adder_inst (
        .clk(clk),
        .rst(rst),
        .exp_a      (signed_exp_a),
        .mant_a     (signed_mant_a),
        .exp_b      (signed_exp_b),
        .mant_b     (signed_mant_b),
        .exp_out    (signed_exp_out),
        .mant_out   (signed_mant_out)
    );

    // Pipeline valid signal to match fp_adder internal PIPELINE_STAGE=1
    always_ff @(posedge clk) begin
        if (rst)
            p0_align_out_valid <= 1'b0;
        else
            p0_align_out_valid <= p_part_valid;
    end

    register_slice_wo_hs #(
        .DATA_WIDTH(ADDER_OUT_EXP_WIDTH + ADDER_OUT_FIXED_WIDTH)
    ) register_add_inst (
        .clk(clk),
        .rst(rst),
        .data_in        ({signed_exp_out, signed_mant_out}),
        .data_in_valid  (p0_align_out_valid),
        .data_out       ({p1_signed_exp_out, p1_signed_mant_out}),
        .data_out_valid (p1_adder_out_valid)
    );

    // Instantiate fp_ieee_normalize for output
    fp_ieee_normalize #(
        .IN_FIXED_WIDTH         (ADDER_OUT_FIXED_WIDTH),
        .IN_FIXED_FRAC_WIDTH    (ADDER_OUT_FIXED_FRAC_WIDTH),
        .IN_EXP_WIDTH           (ADDER_OUT_EXP_WIDTH),
        .OUT_MANT_WIDTH         (NORMALIZE_OUT_MANT_WIDTH)
    ) fp_normalize (
        .signed_mant    (p1_signed_mant_out),
        .signed_exp     (p1_signed_exp_out),
        .fp_out         (normalized_data)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(NORMALIZE_OUT_EXP_WIDTH + NORMALIZE_OUT_MANT_WIDTH + 1)
    ) register_normalize_inst (
        .clk(clk),
        .rst(rst),
        .data_in        (normalized_data),
        .data_in_valid  (p1_adder_out_valid),
        .data_out       (p2_normalized_data),
        .data_out_valid (p2_normalize_out_valid)
    );

    fp_ieee_casting #(
        .IN_EXP_WIDTH       (NORMALIZE_OUT_EXP_WIDTH),
        .IN_MANT_WIDTH      (NORMALIZE_OUT_MANT_WIDTH),
        .OUT_EXP_WIDTH      (EXP_WIDTH + EXT_EXP_WIDTH),
        .OUT_MANT_WIDTH     (MANT_WIDTH + EXT_MANT_WIDTH)
    ) fp_casting (
        .clk(clk),
        .rst(rst),
        .data_in_valid(p2_normalize_out_valid),
        .data_in    (p2_normalized_data),
        .data_out_valid(data_out_valid),
        .data_out   (data_out)
    );



endmodule
