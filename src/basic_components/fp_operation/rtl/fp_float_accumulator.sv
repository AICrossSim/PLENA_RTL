`timescale 1ns / 1ps
/*
Module      : Floating-Point Accumulator
Timing      : 1-cycle latency at output (register_slice_wo_hs)
Description : Running floating-point accumulation. Each cycle the incoming FP
              value is aligned to the running accumulator: the larger exponent
              wins and the smaller-exponent mantissa is shifted right by the
              exponent difference. The two mantissas are added in two's
              complement and the result is renormalized via CLZ (carry -> exp+1,
              cancellation -> exp-down). The internal state uses the SAME format as
              the I/O FP (mantissa = MANT_WIDTH+1 incl. implicit bit, no guard
              bits), so the alignment right-shift truncates each step.

Contrast    : fp_fix_accumulator keeps an EXACT wide fixed-point register (one
              integer add on the loop, no per-step rounding, bounded range).
              This module keeps a narrow {sign,exp,mant} state instead: smaller
              register and unbounded dynamic range, but it rounds every step
              (error grows with the reduction length) and the align+add+normalize
              all sit on the accumulate critical path.

Interface   : identical to fp_fix_accumulator (no handshake / ready signals).
*/

module fp_float_accumulator #(
    parameter int EXP_WIDTH      = 5,
    parameter int MANT_WIDTH     = 10
)(
    input  logic clk,
    input  logic rst,
    input  logic clear_accumulator,
    input  logic data_in_valid,
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_in,   // {sign, exp, mant}
    output logic [EXP_WIDTH + MANT_WIDTH : 0] data_out,
    output logic data_out_valid
);

    localparam int M     = MANT_WIDTH + 1;            // internal mantissa width (implicit bit, no guard)
    localparam int SUM_W = M + 1;                     // mantissa-sum magnitude width
    localparam int E_W   = EXP_WIDTH + 3;             // signed internal exponent width
    localparam int LZ_W  = $clog2(SUM_W + 1);

    // =========================================================================
    // Accumulator state. Convention (exponent kept BIASED throughout — bias is a
    // constant offset that cancels in both the compare and the align difference,
    // so there is no need to un-bias internally; we only keep it signed+wide so
    // the running exponent can grow past the field on the high end and detect
    // underflow (<=0) on the low end):
    //   value = (-1)^acc_sign * acc_mant * 2^((acc_exp - BIAS) - (M-1))
    //   acc_exp = biased weight of mantissa MSB (bit M-1), set when non-zero.
    // =========================================================================
    logic                  acc_sign;
    logic signed [E_W-1:0] acc_exp;
    logic [M-1:0]          acc_mant;
    logic                  acc_zero;
    logic                  acc_valid;

    // =========================================================================
    // Stage 1: decode incoming FP value
    // =========================================================================
    logic                  sign_in;
    logic [EXP_WIDTH-1:0]  exp_field;
    logic [MANT_WIDTH-1:0] mant_field_in;
    logic                  implicit_in;
    logic signed [E_W-1:0] e_in;
    logic [M-1:0]          mant_in_ext;

    assign sign_in       = data_in[EXP_WIDTH + MANT_WIDTH];
    assign exp_field     = data_in[EXP_WIDTH + MANT_WIDTH - 1 : MANT_WIDTH];
    assign mant_field_in = data_in[MANT_WIDTH-1:0];
    assign implicit_in   = (exp_field != 0);
    // biased exponent directly: normals use the field as-is; a denormal's
    // effective biased exponent is 1 (it shares the 2^(1-BIAS) scale).
    assign e_in          = (exp_field == 0) ? 1 : $signed({1'b0, exp_field});
    // mantissa with implicit bit, same width as the internal field (M = MANT_WIDTH+1)
    assign mant_in_ext   = {implicit_in, mant_field_in};

    // =========================================================================
    // Stage 2a: align to the larger exponent, add in two's complement
    // =========================================================================
    logic signed [E_W-1:0] e_max;
    logic                  res_sign;
    logic [SUM_W-1:0]      sum_mag;

    always_comb begin
        logic signed [E_W-1:0] d_acc, d_in;
        logic [M-1:0]          aln_acc, aln_in;
        logic signed [SUM_W:0] s_acc, s_in, s_sum;   // width M+2 (sign + carry)

        if (acc_zero) begin
            // First term (or freshly cleared): result is just the input.
            e_max    = e_in;
            res_sign = sign_in;
            sum_mag  = {1'b0, mant_in_ext};
        end else begin
            e_max = (acc_exp >= e_in) ? acc_exp : e_in;
            d_acc = e_max - acc_exp;                  // >= 0
            d_in  = e_max - e_in;                     // >= 0

            aln_acc = (d_acc >= M) ? '0 : (acc_mant    >> d_acc);
            aln_in  = (d_in  >= M) ? '0 : (mant_in_ext >> d_in);

            s_acc = acc_sign ? -$signed({1'b0, aln_acc}) : $signed({1'b0, aln_acc});
            s_in  = sign_in  ? -$signed({1'b0, aln_in})  : $signed({1'b0, aln_in});
            s_sum = s_acc + s_in;

            res_sign = s_sum[SUM_W];                  // sign bit of the M+2-bit sum
            sum_mag  = res_sign ? (-s_sum) : s_sum;   // magnitude fits SUM_W bits
        end
    end

    // =========================================================================
    // Stage 2b: renormalize the mantissa sum (leading 1 -> bit M-1)
    //   lz = leading zeros of the (M+1)-bit magnitude
    //   shift = 1 - lz  (lz=0 carry -> >>1, exp+1 ; lz=1 already normal ;
    //                    lz>=2 cancellation -> <<(lz-1), exp down)
    // =========================================================================
    logic [LZ_W-1:0]       lz;
    logic [SUM_W-1:0]      mant_norm_full;
    logic signed [E_W-1:0] e_new;

    clz_int #(
        .width_i(SUM_W)
    ) clz_inst (
        .i_num(sum_mag),
        .o_lz (lz)
    );

    always_comb begin
        if (lz == 0)      mant_norm_full = sum_mag >> 1;
        else if (lz == 1) mant_norm_full = sum_mag;
        else              mant_norm_full = sum_mag << (lz - 1);
        e_new = e_max + 1 - $signed({1'b0, lz});
    end

    // =========================================================================
    // Stage 2c: accumulator register (the feedback loop)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst || clear_accumulator) begin
            acc_sign  <= 1'b0;
            acc_exp   <= '0;
            acc_mant  <= '0;
            acc_zero  <= 1'b1;
            acc_valid <= 1'b0;
        end else if (data_in_valid) begin
            acc_valid <= 1'b1;
            if (sum_mag == 0) begin
                acc_sign <= 1'b0;
                acc_exp  <= '0;
                acc_mant <= '0;
                acc_zero <= 1'b1;
            end else begin
                acc_sign <= res_sign;
                acc_exp  <= e_new;
                acc_mant <= mant_norm_full[M-1:0];
                acc_zero <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Stage 3: convert accumulator -> output FP (truncate mantissa), register out
    // =========================================================================
    logic signed [E_W-1:0]        exp_biased;
    logic [EXP_WIDTH-1:0]         exp_out;
    logic [MANT_WIDTH-1:0]        mant_out;
    logic [EXP_WIDTH+MANT_WIDTH:0] fp_result;

    logic out_sign;
    always_comb begin
        exp_biased = acc_exp;          // already biased, no +BIAS needed
        out_sign   = acc_sign;
        if (acc_zero || exp_biased <= 0) begin
            // zero, or exponent underflow (magnitude too small) -> cap to 0
            out_sign = 1'b0;
            exp_out  = '0;
            mant_out = '0;
        end else if (exp_biased >= (1 << EXP_WIDTH) - 1) begin
            // exponent overflow (magnitude too large) -> saturate to max
            // representable, keeping the sign (+max / -max)
            exp_out  = (1 << EXP_WIDTH) - 2;
            mant_out = {MANT_WIDTH{1'b1}};
        end else begin
            exp_out  = exp_biased[EXP_WIDTH-1:0];
            mant_out = acc_mant[M-2 -: MANT_WIDTH];   // bits just below the implicit 1
        end
        fp_result = {out_sign, exp_out, mant_out};
    end

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
