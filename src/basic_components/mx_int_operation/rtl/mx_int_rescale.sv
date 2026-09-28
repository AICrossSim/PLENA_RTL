`timescale 1ns / 1ps

/*
Module      : MX-INT Rescale Unit
Timing      : Sequential, Takes 1 cycle to rescale
Description : Rescale a block of MXINT elements with individual scales to share a common scale.
            : For row-wise read (no transpose), scales are already the same per block.
            : For column-wise read (transposed), scales vary per element and need rescaling.
            : Format: {e1, e2, e3, e4} with individual scales -> {e1', e2', e3', e4'} with shared max_scale
Status      :
*/

module mx_int_rescale #(
    // MX-INT Data Format
    parameter MXINT_WIDTH       = 8,    // Total width including sign bit
    parameter MXINT_SCALE_WIDTH = 8,

    // Dimension
    parameter BLOCK_DIM         = 4
) (
    input  logic clk,
    input  logic rst,

    // Input: block of MXINT elements with individual scales
    input  logic [BLOCK_DIM - 1 : 0][MXINT_WIDTH - 1 : 0]       element_in,
    input  logic [BLOCK_DIM - 1 : 0][MXINT_SCALE_WIDTH - 1 : 0] scale_in,

    // Output: block of MXINT elements with shared scale
    output logic [BLOCK_DIM - 1 : 0][MXINT_WIDTH - 1 : 0]       element_data_out,
    output logic [MXINT_SCALE_WIDTH - 1 : 0]                    scale_data_out
);

    // Pipeline stage 0: Find max scale (unsigned_max has 1 cycle latency)
    logic [MXINT_SCALE_WIDTH - 1 : 0] scale_max;
    logic [BLOCK_DIM - 1 : 0][MXINT_WIDTH - 1 : 0] p1_element;
    logic [BLOCK_DIM - 1 : 0][MXINT_SCALE_WIDTH - 1 : 0] p1_scale;

    // Max finder for scales - output is registered internally (1 cycle delay)
    unsigned_max #(
        .width(MXINT_SCALE_WIDTH),
        .length(BLOCK_DIM),
        .flop_output(0)
    ) scale_max_finder (
        .clk(clk),
        .input_data(scale_in),
        .max_val(scale_max)
    );

    // Pipeline register for elements and scales (1 cycle delay, aligned with scale_max)
    always_ff @(posedge clk) begin
        if (rst) begin
            p1_element <= '0;
            p1_scale <= '0;
        end else begin
            p1_element <= element_in;
            p1_scale <= scale_in;
        end
    end

    // Pipeline stage 1: Compute shift amounts and shift mantissas
    // Both scale_max (from unsigned_max) and p1_element/p1_scale are available after 1 cycle
    generate
        for (genvar i = 0; i < BLOCK_DIM; i++) begin : gen_rescale
            logic [MXINT_SCALE_WIDTH - 1 : 0] shift_amount;
            logic signed [MXINT_WIDTH - 1 : 0] shifted_mant;

            always_comb begin
                // shift_amount = max_scale - element_scale (always >= 0)
                shift_amount = scale_max - p1_scale[i];

                // Arithmetic right shift to preserve sign
                // When shift_amount is 0, no shift needed
                // When shift_amount > 0, right shift the mantissa
                if (shift_amount >= MXINT_WIDTH) begin
                    // Overflow: shift out all bits, preserve sign
                    shifted_mant = p1_element[i][MXINT_WIDTH-1] ? {MXINT_WIDTH{1'b1}} : {MXINT_WIDTH{1'b0}};
                end else begin
                    shifted_mant = $signed(p1_element[i]) >>> shift_amount;
                end

                element_data_out[i] = shifted_mant;
            end
        end
    endgenerate

    // Output the max scale as shared scale
    assign scale_data_out = scale_max;

endmodule
