`timescale 1ns / 1ps
/*
Module      : Floating Point Configurable Precision Adder (With Sign)
Timing      : Combinatorial Logic
Description : Adds two FP numbers with different exponents and signs.
              Aligns mantissas, preserves full precision (no bits discarded).
              Output format: {sign, exp_out, mant_out}.
              No rounding.
              It needs normalisation.
              The lossy part will be at the mantissa adder
Status      : Passed Simple Tests
*/

module fix_accumulator #(
    parameter int WIDTH = 16,
    parameter int EXPAND_WIDTH = 0
)(
    input  logic clk,
    input  logic rst,
    input  logic clear_accumulator,
    input  logic data_in_valid,
    input  logic [WIDTH - 1 : 0] data_in,
    output logic [WIDTH+EXPAND_WIDTH - 1 : 0] data_out,
    output logic data_out_valid
);


  logic [WIDTH+EXPAND_WIDTH - 1 : 0] partial_sum;
  /* verilator lint_off WIDTH */
  assign data_out_valid  = 1'b1;
  /* verilator lint_on WIDTH */

  // data_out
  // Priority (top wins):
  //   rst                              → clear to 0
  //   clear + data_in_valid (new batch) → LOAD data_in (start fresh from this MAC)
  //   clear alone                       → clear to 0
  //   data_in_valid alone               → accumulate (partial_sum)
  //   none                              → hold
  always_ff @(posedge clk)
    if (rst)                                     data_out <= '0;
    else if (clear_accumulator && data_in_valid) data_out <= $signed(data_in);
    else if (clear_accumulator)                  data_out <= '0;
    else if (data_in_valid)                      data_out <= partial_sum;

    (* use_dsp = "yes" *) assign partial_sum = $signed(data_in) + $signed(data_out);


endmodule
