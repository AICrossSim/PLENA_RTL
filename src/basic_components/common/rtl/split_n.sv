/*
Module      : splitn
Description : This module implements a 1-to-N streaming interface handshake.
*/

`timescale 1ns / 1ps

module split_n #(
    parameter N = 10
) (
    input  logic [  0:0] data_in_valid,
    output logic [  0:0] data_in_ready,
    output logic [N-1:0] data_out_valid,
    input  logic [N-1:0] data_out_ready
);

  logic [N-1:0] ready_intermediate;

  if (N == 1) begin

    assign data_out_valid = data_in_valid;
    assign data_in_ready  = data_out_ready;

  end else begin

    for (genvar i = 0; i < N; i++) begin : handshake
      localparam logic [N-1:0] SELF_MASK = (N'(1) << i);
      assign ready_intermediate[i] = &(data_out_ready | SELF_MASK);
      assign data_out_valid[i] = data_in_valid && ready_intermediate[i];
    end

    assign data_in_ready = &data_out_ready;

  end

endmodule
