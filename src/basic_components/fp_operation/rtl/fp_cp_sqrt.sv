`timescale 1ns / 1ps
// `include "operation.svh"

/*
Module      : FP Square Root
Timing      : Combinatorial Logic
Description : This module computes the square root of a floating point number
Status      : Under Development
*/

module fp_cp_sqrt #(
    parameter   EXP_WIDTH = 5,
    parameter   MANT_WIDTH = 10,
    // The integer sqrt only depends on m_eff (MANT_WIDTH+2 bits), so SQRT_LUT=1
    // constant-folds isqrt() into a ROM indexed by m_eff (a few LUT levels) instead
    // of the ISQRT_WIDTH-iteration restoring-sqrt CARRY4 chain. Bit-identical.
    parameter   SQRT_LUT = 1
)(
    input  logic [EXP_WIDTH + MANT_WIDTH : 0] data_in,  // {sign, exp, mant}
    output logic [EXP_WIDTH + MANT_WIDTH : 0] data_out
);

    // initial begin
    //     assert (data_in[EXP_WIDTH + MANT_WIDTH : 0] != 0) else $error("data_in is not positive");
    // end
    logic sign_bit;
    assign sign_bit = data_in[EXP_WIDTH + MANT_WIDTH];

    // Runtime check for positive input
    // TODO Hide here due to multiple drive for data_out.

    logic [EXP_WIDTH + MANT_WIDTH : 0] computed_out;

    // always_comb begin
    //     if (sign_bit) begin
    //         // Handle negative input - return 0 or NaN
    //         data_out = '0; // Return 0 for negative inputs
    //     end
    // end
    assign data_out = sign_bit ? '0 : computed_out;

    localparam SIGN_MANT_WIDTH = MANT_WIDTH + 2;
    localparam UNSIGNED_MANT_WIDTH = MANT_WIDTH + 1;

    logic signed [SIGN_MANT_WIDTH-1:0] sign_mant;
    logic signed [EXP_WIDTH:0] signed_exp;

    fp_ieee_partition #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(MANT_WIDTH)
    ) fp_ieee_partition_inst (
        .data_in(data_in),
        .signed_mant(sign_mant),
        .signed_exp(signed_exp)
    );

    localparam BIAS = (EXP_WIDTH == 1) ? 1 : ((1 << (EXP_WIDTH - 1)) - 1);

    logic signed [EXP_WIDTH:0]  new_exp;          // floor(e/2), unbiased
    logic signed [EXP_WIDTH+1:0] new_exp_biased;

    // sqrt(m * 2^e) = sqrt(m_eff) * 2^floor(e/2), where the input mantissa
    // sign_mant is fixed-point Q(MANT_WIDTH) with the implicit 1 at bit MANT_WIDTH
    // (i.e. 1.0 -> 2^MANT_WIDTH). With out_exp = floor(e/2) and the odd bit folded
    // into the mantissa (m_eff = m << (e & 1)), m_eff/2^MANT_WIDTH lies in [1,4) so
    // its sqrt lies in [1,2) -- already a normalized mantissa, so we can assemble the
    // IEEE result directly (no re-normalization needed).
    //
    //   out_mant/2^MANT_WIDTH = sqrt(m_eff/2^MANT_WIDTH)
    //   => out_mant = sqrt(2^MANT_WIDTH * m_eff)   (in [2^MANT_WIDTH, 2^(MANT_WIDTH+1)))
    //
    // Computed with an integer sqrt. To round to nearest we run the isqrt one extra
    // binary place (<<2 inside the radicand == *2 on the result) and round-shift back.
    localparam int RADICAND_WIDTH = (2 * (MANT_WIDTH + 1)) + 2; // fits 2^(MANT_WIDTH+2) * m_eff
    localparam int ISQRT_WIDTH    = RADICAND_WIDTH / 2;

    // Integer floor(sqrt(num)) via the classic digit-by-digit (restoring) algorithm.
    function automatic logic [ISQRT_WIDTH-1:0] isqrt(input logic [RADICAND_WIDTH-1:0] num);
        logic [RADICAND_WIDTH-1:0] rem;
        logic [RADICAND_WIDTH-1:0] root;
        logic [RADICAND_WIDTH-1:0] bitpos;
        int i;
        rem  = num;
        root = '0;
        bitpos = 1 << (RADICAND_WIDTH-2);   // highest power of four <= 2^(RADICAND_WIDTH-1)
        for (i = 0; i < ISQRT_WIDTH; i++) begin
            if (rem >= root + bitpos) begin
                rem  = rem - (root + bitpos);
                root = (root >> 1) + bitpos;
            end else begin
                root = root >> 1;
            end
            bitpos = bitpos >> 2;
        end
        return root[ISQRT_WIDTH-1:0];
    endfunction

    localparam int MEFF_WIDTH = UNSIGNED_MANT_WIDTH + 1;         // m_eff bit width

    logic [UNSIGNED_MANT_WIDTH : 0]      m_eff;       // mantissa, odd-exponent doubled
    logic [ISQRT_WIDTH-1:0]              root2x;      // 2 * sqrt(2^MANT_WIDTH * m_eff)
    logic [UNSIGNED_MANT_WIDTH-1:0]      out_mant;    // sqrt mantissa, in [2^MANT_WIDTH, 2^(MANT_WIDTH+1))

    assign m_eff = signed_exp[0] ? (sign_mant << 1) : sign_mant[UNSIGNED_MANT_WIDTH-1:0];

    generate
        if (SQRT_LUT) begin : g_sqrt_lut
            logic [ISQRT_WIDTH-1:0] sqrt_rom [(1 << MEFF_WIDTH) - 1:0];
            for (genvar k = 0; k < (1 << MEFF_WIDTH); k++) begin : g_rom
                assign sqrt_rom[k] = isqrt((RADICAND_WIDTH'(k)) << (MANT_WIDTH + 2));
            end
            assign root2x = sqrt_rom[m_eff];
        end else begin : g_sqrt_iter
            assign root2x = isqrt((RADICAND_WIDTH'(m_eff)) << (MANT_WIDTH + 2));
        end
    endgenerate

    always_comb begin
        new_exp        = signed_exp >>> 1;                       // floor(e/2)
        out_mant       = (root2x + 1) >> 1;                      // round-to-nearest Q(MANT_WIDTH)
        new_exp_biased = $signed({1'b0, new_exp}) + BIAS;

        // sign_mant == 0 means the input was zero -> sqrt(0) = 0. Otherwise the input
        // is normalized so out_mant has its implicit 1 at bit MANT_WIDTH; drop it and
        // pair the fraction with the biased exponent.
        if (sign_mant == 0)
            computed_out = '0;
        else
            computed_out = {1'b0, new_exp_biased[EXP_WIDTH-1:0], out_mant[MANT_WIDTH-1:0]};
    end

endmodule
