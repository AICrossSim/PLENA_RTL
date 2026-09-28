`timescale 1ns / 1ps
/*
Module      : Fixed-Point Accumulator with FP I/O
Timing      : 1-cycle latency (register at output)
Description : Converts FP input to fixed-point, accumulates using simple integer
              addition (fast), and converts back to FP at output.
              Critical path is now just integer addition, not FP alignment+normalize.
Status      : Refactored for timing optimization
*/

module fp_fix_accumulator #(
    parameter int EXP_WIDTH = 5,
    parameter int MANT_WIDTH = 10,
    // Accumulator sizing: user-specified for flexibility
    // Default: 16 bits integer + 16 bits fraction = 32-bit accumulator
    parameter int ACC_INT_WIDTH = 16,
    parameter int ACC_FRAC_WIDTH = 16
)(
    input  logic clk,
    input  logic rst,
    input  logic clear_accumulator,
    input  logic data_in_valid,
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_in,  // {sign, exp, mant}
    output logic [EXP_WIDTH + MANT_WIDTH : 0] data_out,
    output logic data_out_valid
);

    // Accumulator total width
    localparam int ACC_WIDTH = ACC_INT_WIDTH + ACC_FRAC_WIDTH;
    localparam int BIAS = (EXP_WIDTH == 1) ? 1 : ((1 << (EXP_WIDTH - 1)) - 1);
    localparam int MANT_WITH_IMPLICIT = MANT_WIDTH + 1;

    // Fixed-point accumulator register
    logic signed [ACC_WIDTH-1:0] accumulator;
    logic acc_valid;

    // =========================================================================
    // Stage 1: FP to Fixed-Point Conversion (Combinational)
    // =========================================================================
    logic sign_in;
    logic [EXP_WIDTH-1:0] exp_in;
    logic [MANT_WIDTH-1:0] mant_in;
    logic [MANT_WITH_IMPLICIT-1:0] mant_with_implicit;
    logic signed [EXP_WIDTH:0] actual_exp;
    logic signed [ACC_WIDTH-1:0] fixed_in;

    assign sign_in = data_in[EXP_WIDTH + MANT_WIDTH];
    assign exp_in = data_in[EXP_WIDTH + MANT_WIDTH - 1 : MANT_WIDTH];
    assign mant_in = data_in[MANT_WIDTH-1:0];

    // Handle denormal (exp=0) vs normal numbers
    assign mant_with_implicit = (exp_in == 0) ? {1'b0, mant_in} : {1'b1, mant_in};
    assign actual_exp = (exp_in == 0) ? (1 - BIAS) : ($signed({1'b0, exp_in}) - BIAS);

    // Convert to fixed-point by shifting mantissa
    // Fixed-point format: ACC_INT_WIDTH.ACC_FRAC_WIDTH
    // Shift amount determines where mantissa bits land
    always_comb begin
        logic [ACC_WIDTH-1:0] unsigned_fixed;
        logic signed [EXP_WIDTH+1:0] shift_amount;

        // Position mantissa MSB at (ACC_FRAC_WIDTH + actual_exp) from LSB
        // Mantissa MSB (implicit 1) represents 2^actual_exp
        shift_amount = ACC_FRAC_WIDTH + actual_exp - MANT_WIDTH;

        if (shift_amount >= 0 && shift_amount < ACC_WIDTH) begin
            unsigned_fixed = {{(ACC_WIDTH-MANT_WITH_IMPLICIT){1'b0}}, mant_with_implicit} << shift_amount;
        end else if (shift_amount < 0 && shift_amount > -MANT_WITH_IMPLICIT) begin
            unsigned_fixed = {{(ACC_WIDTH-MANT_WITH_IMPLICIT){1'b0}}, mant_with_implicit} >> (-shift_amount);
        end else begin
            unsigned_fixed = '0;  // Out of range
        end

        fixed_in = sign_in ? -$signed(unsigned_fixed) : $signed(unsigned_fixed);
    end

    // =========================================================================
    // Stage 2: Fixed-Point Accumulation (Single Integer Add - Fast!)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst || clear_accumulator) begin
            accumulator <= '0;
            acc_valid <= 1'b0;
        end else if (data_in_valid) begin
            accumulator <= accumulator + fixed_in;
            acc_valid <= 1'b1;
        end
    end

    // =========================================================================
    // Stage 3: Fixed-Point to FP Conversion (Combinational + Output Register)
    // =========================================================================

    // CLZ to find leading one position
    logic [ACC_WIDTH-1:0] abs_acc;
    logic [$clog2(ACC_WIDTH+1)-1:0] leading_zeros;
    logic sign_out;

    assign sign_out = accumulator[ACC_WIDTH-1];
    assign abs_acc = sign_out ? -accumulator : accumulator;

    clz_int #(
        .width_i(ACC_WIDTH)
    ) clz_inst (
        .i_num(abs_acc),
        .o_lz(leading_zeros)
    );

    // Normalize and extract FP components
    logic [ACC_WIDTH-1:0] normalized_acc;
    logic signed [EXP_WIDTH+2:0] computed_exp;
    logic [EXP_WIDTH-1:0] exp_out;
    logic [MANT_WIDTH-1:0] mant_out;
    logic [EXP_WIDTH + MANT_WIDTH:0] fp_result;

    always_comb begin
        // Shift to normalize (put leading 1 at MSB position)
        if (leading_zeros < ACC_WIDTH - 1) begin
            normalized_acc = abs_acc << (leading_zeros + 1);  // +1 to remove leading 1
        end else begin
            normalized_acc = '0;
        end

        // Compute exponent
        // MSB of abs_acc (after removing sign) represents 2^(ACC_INT_WIDTH-1)
        // After leading_zeros, the leading 1 is at position (ACC_WIDTH - 1 - leading_zeros)
        // This represents 2^(ACC_INT_WIDTH - 1 - leading_zeros)
        computed_exp = (ACC_INT_WIDTH - 1 - $signed({{($bits(computed_exp)-$clog2(ACC_WIDTH+1)){1'b0}}, leading_zeros})) + BIAS;

        // Clamp exponent and extract mantissa
        if (accumulator == 0) begin
            exp_out = '0;
            mant_out = '0;
        end else if (computed_exp <= 0) begin
            // Denormal or underflow - simplified handling
            exp_out = '0;
            mant_out = normalized_acc[ACC_WIDTH-1 -: MANT_WIDTH];
        end else if (computed_exp >= (1 << EXP_WIDTH) - 1) begin
            // Overflow - saturate to max
            exp_out = (1 << EXP_WIDTH) - 2;  // Max normal exponent
            mant_out = {MANT_WIDTH{1'b1}};
        end else begin
            exp_out = computed_exp[EXP_WIDTH-1:0];
            mant_out = normalized_acc[ACC_WIDTH-1 -: MANT_WIDTH];
        end

        fp_result = {sign_out, exp_out, mant_out};
    end

    // Output register
    register_slice_wo_hs #(
        .DATA_WIDTH(EXP_WIDTH + MANT_WIDTH + 1)
    ) register_slice_inst (
        .clk(clk),
        .rst(rst || clear_accumulator),
        .data_in(fp_result),
        .data_in_valid(acc_valid),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );

endmodule
