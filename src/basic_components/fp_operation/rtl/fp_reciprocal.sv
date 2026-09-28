`timescale 1ns / 1ps
// `include "operation.svh"

/*
Module      : FP Reciprocal
Timing      : Combinatorial Logic
Description : This module computes reciprocal of floating point numbers
            : represented with separate mantissa and exponent
Status      : Under Development
*/

module fp_reciprocal #(
    parameter   IN_EXP_WIDTH = 5,
    parameter   IN_FIX_WIDTH = 8,
    parameter   IN_FIX_FRAC_WIDTH = 5,
    parameter   OUT_EXP_WIDTH = -1,
    parameter   OUT_FIX_WIDTH = -1,
    parameter   OUT_FIX_FRAC_WIDTH = -1,
    // Pipeline register-slice depth around the combinational divider (the timing-critical
    // path on FPGA). Default 2 = pre-existing behaviour. Increasing it inserts matched
    // register slices on the data and valid paths; with synthesis retiming the tool
    // redistributes them through the divider to shorten the critical path. Latency is
    // carried by data_out_valid, so consumers that track valid adapt automatically.
    parameter   REG_N = 2,
    // The reciprocal divisor (unsigned_mant_in) is only IN_WIDTH bits, so (1<<W)/mant
    // has 2**IN_WIDTH possible results. DIV_LUT=1 constant-folds it into a ROM indexed
    // by the mantissa (a few LUT levels) instead of a ~2**IN_WIDTH-deep CARRY4 restoring
    // divider (25 CARRY4 / ~21ns at e6m5). Bit-identical, zero latency change.
    parameter   DIV_LUT = 1
)(
    input logic clk,
    input logic rst,
    input logic data_in_valid,
    output logic data_out_valid,
    input logic signed [IN_FIX_WIDTH - 1:0] signed_mant_in,
    input logic signed [IN_EXP_WIDTH:0] signed_exp_in,
    output logic signed [OUT_EXP_WIDTH - 1:0] signed_exp_out,
    output logic signed [OUT_FIX_WIDTH - 1:0] signed_mant_out
);
    logic stall;
    assign stall = 1'b0;

    localparam IN_WIDTH = IN_FIX_WIDTH - 1;
    localparam EXTEND_EXP_WIDTH = OUT_EXP_WIDTH + 1;
    localparam RECIPROCAL_MANTISSA_WIDTH = OUT_FIX_WIDTH + IN_FIX_WIDTH; //RANDOM set a large width for the reciprocal mantissa


    logic sign;
    logic unsigned [IN_WIDTH - 1:0] unsigned_mant_in;
    logic signed [RECIPROCAL_MANTISSA_WIDTH - 1:0] unsigned_reciprocal_mantissa;

    data_reg #(
        .DATA_WIDTH(1),
        .REG_N(REG_N)
    ) data_in_reg (
        .data_in(data_in_valid),
        .data_out(data_out_valid),
        .*
    );

    logic signed [OUT_EXP_WIDTH - 1:0] leading_zeros;
    logic signed [EXTEND_EXP_WIDTH - 1:0] extend_exp;
    logic signed [EXTEND_EXP_WIDTH - 1:0] exp_difference;
    logic signed [EXTEND_EXP_WIDTH - 1:0] shift_value;

    logic unsigned [RECIPROCAL_MANTISSA_WIDTH - 1:0] unsigned_output_mantissa_lossless;
    logic unsigned [OUT_FIX_WIDTH - 1 - 1:0] unsigned_mant_out;

    assign sign = signed_mant_in[IN_FIX_WIDTH - 1];
    assign unsigned_mant_in = (sign) ? ~signed_mant_in + 1 : signed_mant_in;

    // Calculate reciprocal mantissa
    generate
        if (DIV_LUT) begin : g_div_lut
            logic signed [RECIPROCAL_MANTISSA_WIDTH - 1:0] recip_rom [(1 << IN_WIDTH) - 1:0];
            for (genvar gi = 0; gi < (1 << IN_WIDTH); gi++) begin : g_rom
                assign recip_rom[gi] = (gi == 0)
                    ? RECIPROCAL_MANTISSA_WIDTH'(1 << (RECIPROCAL_MANTISSA_WIDTH - 1) - 1)
                    : RECIPROCAL_MANTISSA_WIDTH'((1 << (RECIPROCAL_MANTISSA_WIDTH - 1)) / gi);
            end
            assign unsigned_reciprocal_mantissa = recip_rom[unsigned_mant_in];
        end else begin : g_div
            always_comb begin
                if (unsigned_mant_in == 0) begin
                    unsigned_reciprocal_mantissa = (1<<(RECIPROCAL_MANTISSA_WIDTH - 1) - 1);
                end else begin
                    unsigned_reciprocal_mantissa = (1<<(RECIPROCAL_MANTISSA_WIDTH - 1)) / unsigned_mant_in;
                end
            end
        end
    endgenerate
    logic signed [IN_EXP_WIDTH:0] signed_exp_in_reg;
    logic unsigned [RECIPROCAL_MANTISSA_WIDTH - 1:0] unsigned_reciprocal_mantissa_reg;
    logic sign_reg;
    // reg here
    // now the unsigned_reciprocal_mantissa becomes *.IN_WIDTH
    data_reg #(
        .DATA_WIDTH(RECIPROCAL_MANTISSA_WIDTH + IN_EXP_WIDTH + 1 + 1),
        .REG_N(REG_N)
    ) data_reg_inst (
        .data_in({sign, signed_exp_in, unsigned_reciprocal_mantissa}),
        .data_out({sign_reg, signed_exp_in_reg, unsigned_reciprocal_mantissa_reg}),
        .*
    );

    clz_int #(
        .width_i(RECIPROCAL_MANTISSA_WIDTH)
    ) clz_inst (
        .i_num(unsigned_reciprocal_mantissa_reg),
        .o_lz(leading_zeros[$clog2(RECIPROCAL_MANTISSA_WIDTH + 1) - 1:0])
    );
    
    // Calculate leading zeros and extended exponent
    assign extend_exp = -(signed_exp_in_reg - IN_FIX_FRAC_WIDTH) - leading_zeros; // for the int part, [sign, int, *frac]
    
    // Clamp exponent to valid range using signed_clamp module
    signed_clamp #(
        .IN_WIDTH (EXTEND_EXP_WIDTH),
        .OUT_WIDTH(OUT_EXP_WIDTH)
    ) exp_clamp (
        .in_data (extend_exp),
        .out_data(signed_exp_out)
    );
    
    // Calculate exponent difference
    assign exp_difference = extend_exp - signed_exp_out;
    
    // Scale mantissa by leading zeros and exponent difference
    assign shift_value = (leading_zeros + exp_difference);

    bit_width_aware_left_shift #(
        .IN_WIDTH (RECIPROCAL_MANTISSA_WIDTH),
        .OUT_WIDTH(RECIPROCAL_MANTISSA_WIDTH),
        .SHIFT_WIDTH(EXTEND_EXP_WIDTH)
    ) shift_inst (
        .in_data (unsigned_reciprocal_mantissa_reg),
        .shift_amt(shift_value),
        .out_data(unsigned_output_mantissa_lossless)
    );
    // Clamp mantissa to valid range using signed_clamp module
    round_to_nearest_even #(
        .IN_WIDTH (RECIPROCAL_MANTISSA_WIDTH),
        .OUT_WIDTH(OUT_FIX_FRAC_WIDTH + 1)
    ) mant_round (
        .data_in (unsigned_output_mantissa_lossless),
        .data_out(unsigned_mant_out)
    );
    assign signed_mant_out = (sign_reg) ? -unsigned_mant_out: unsigned_mant_out;

endmodule