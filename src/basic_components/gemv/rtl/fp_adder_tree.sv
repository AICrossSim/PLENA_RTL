`timescale 1ns / 1ps

/*
Module      : Floating Point Adder Tree
Description : Hierarchical adder tree for floating point numbers.
Timing      : PIPELINE=0: $clog2(LEVELS) cycles (1 register per level, original).
              PIPELINE=1: 3*$clog2(LEVELS) cycles (3 register stages per level:
                fp_adder internal + adder→normalize + normalize→cast).
              register_slice_wo_hs sits between levels in both modes.
Input   d1  |   d12  |  d1234  Output
        d2  |   d34  |  x
        d3  |   x    |  x
        d4  |   x    |  x
Status      : Passed Simple Tests
*/

module fp_adder_tree #(
    parameter VEC_DIM       = 4,
    parameter IN_EXP_WIDTH  = 3,
    parameter IN_MAN_WIDTH  = 4,

    // Precision Control
    parameter EXT_MANT_WIDTH_PER_LAYER = 1,
    parameter EXT_EXP_BITS_PER_LAYER = 1,

    // Pipeline control: 0 = 1-stage per level (original), 1 = 3-stage per level
    parameter PIPELINE = 1,

    localparam LEVELS = $clog2(VEC_DIM),

    localparam OVERALL_MANT_EXT_BITS = LEVELS * EXT_MANT_WIDTH_PER_LAYER,
    localparam OUT_MAN_WIDTH = OVERALL_MANT_EXT_BITS + IN_MAN_WIDTH,

    localparam OVERALL_EXP_EXT_BITS = LEVELS * EXT_EXP_BITS_PER_LAYER,
    localparam OUT_EXP_WIDTH  = OVERALL_EXP_EXT_BITS + IN_EXP_WIDTH,

    localparam IN_WIDTH       = IN_MAN_WIDTH + IN_EXP_WIDTH + 1,
    localparam OUT_WIDTH      = OUT_MAN_WIDTH + OUT_EXP_WIDTH + 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic clk,
    input  logic rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [VEC_DIM-1:0] [IN_WIDTH - 1 : 0] data_in,
    input  logic data_in_valid,

    output logic [OUT_WIDTH - 1 : 0] data_out,
    output logic data_out_valid
);

  initial begin
    assert (VEC_DIM > 0);
  end

  generate
    if (LEVELS == 0) begin : gen_skip_adder_tree

      assign data_out = {{OVERALL_MANT_EXT_BITS{1'b0}}, data_in[0]};
      assign data_out_valid = data_in_valid;

    end else begin : gen_adder_tree

      // data_storage & sum wires are oversized on purpose for vivado.
      logic [OUT_WIDTH*VEC_DIM-1:0] data_storage [LEVELS:0];  // TODO: Need to be optimized, memory inefficient
      logic [OUT_WIDTH*VEC_DIM-1:0] sum  [LEVELS-1:0];
      logic valid[VEC_DIM-1:0];
      // layer_valid[i] = valid signal out of fp_adder_tree_layer i (before register_slice_wo_hs)
      logic layer_valid[LEVELS-1:0];

      // Generate adder for each layer
      for (genvar i = 0; i < LEVELS; i++) begin : level

        localparam LEVEL_IN_DIM = (VEC_DIM + ((1 << i) - 1)) >> i;     // Ceiling(VEC_DIM / 2^i)
        localparam LEVEL_IN_MAN_WIDTH   = IN_MAN_WIDTH + i * EXT_MANT_WIDTH_PER_LAYER;
        localparam LEVEL_IN_EXP_WIDTH   = IN_EXP_WIDTH + i * EXT_EXP_BITS_PER_LAYER;

        localparam LEVEL_OUT_DIM = (LEVEL_IN_DIM + 1) / 2;
        localparam LEVEL_OUT_MAN_WIDTH  = IN_MAN_WIDTH + (i + 1) * EXT_MANT_WIDTH_PER_LAYER;
        localparam LEVEL_OUT_EXP_WIDTH  = IN_EXP_WIDTH + (i + 1) * EXT_EXP_BITS_PER_LAYER;
        localparam LEVEL_OUT_WIDTH = LEVEL_OUT_MAN_WIDTH + LEVEL_OUT_EXP_WIDTH + 1;

        // In wo_hs mode there is no backpressure: pipe_enable = valid[i] always.
        fp_adder_tree_layer #(
            .OVERALL_INPUT_WIDTH  (OUT_WIDTH*VEC_DIM),
            .LAYER_DIM            (LEVEL_IN_DIM),
            .IN_MAN_WIDTH         (LEVEL_IN_MAN_WIDTH),
            .IN_EXP_WIDTH         (LEVEL_IN_EXP_WIDTH),
            .EXT_MANT_WIDTH       (EXT_MANT_WIDTH_PER_LAYER),
            .EXT_EXP_WIDTH        (EXT_EXP_BITS_PER_LAYER),
            .PIPELINE             (PIPELINE)
        ) full_precision_add_layer (
            .clk                  (clk),
            .rst                  (rst),
            .data_in              (data_storage[i]),
            .data_out             (sum[i]),
            .pipe_enable          (valid[i]),        // no backpressure: always enable
            .pipe_valid_out       (layer_valid[i])   // propagated valid out of layer
        );

        register_slice_wo_hs #(
            .DATA_WIDTH(LEVEL_OUT_DIM * LEVEL_OUT_WIDTH)
        ) register_slice (
            .clk           (clk),
            .rst           (rst),
            .data_in       (sum[i][LEVEL_OUT_DIM * LEVEL_OUT_WIDTH - 1 : 0]),
            .data_in_valid (layer_valid[i]),
            .data_out      (data_storage[i+1][LEVEL_OUT_DIM * LEVEL_OUT_WIDTH - 1 : 0]),
            .data_out_valid(valid[i+1])
        );
      end

      assign data_storage[0] = data_in;
      assign valid[0] = data_in_valid;
      assign data_out = data_storage[LEVELS][OUT_WIDTH-1:0];
      assign data_out_valid = valid[LEVELS];

    end
  endgenerate


endmodule
