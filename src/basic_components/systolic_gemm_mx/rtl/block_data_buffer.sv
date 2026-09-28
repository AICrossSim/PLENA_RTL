`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"


/*
Module      : Partial Result Buffer for Matrix Machine V2
Timing      : Sequential, Takes x cycles to compute the dot product
Description : This module will buffer [Batch, Batch] result every time and output [MLEN, 1] every time.
            : |a, b| |e, f|
              |c, d| |g, h|
            : will output [a, b, e, f] every time.
            : Note, the address here is the column index, controlling the inserted position of the block data.
*/

module block_data_buffer #(
    parameter M = 4,
    parameter K = 8,
    parameter N = 4,
    parameter FP_EXP_WIDTH = 8,
    parameter FP_MANT_WIDTH = 7,
    parameter ACC_ADDR_WIDTH = 4,
    localparam ACC_NUM = K / M
    
)(
    input   logic clk,
    input   logic rst,
    // Input, accepting [M, N] partial results
    input   logic [M-1:0][N-1:0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0] block_data_in,
    input   logic [ACC_ADDR_WIDTH-1:0] acc_waddr,
    input   logic acc_waddr_valid,
    output  logic acc_waddr_ready,
    input   logic wait_for_output,
    input   logic block_data_valid,
    // Output, outputting [K] results for M cycles
    output  logic [K-1:0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0] unrolled_data_out,
    output  logic unrolled_data_out_valid
);

initial begin
    assert (M == N) else $fatal("M must be equal to N for this block data buffer.");
end

  logic write_to_buffer_valid;

  // One-shot trigger: pulse write_to_buffer_valid once to start loading,
  // then loading continues based on block_data_valid alone
  assign write_to_buffer_valid = acc_waddr_valid;
  assign acc_waddr_ready = write_to_buffer_valid;

  logic [M : 0] load_out_counter;
  logic [M-1:0][K-1:0][FP_EXP_WIDTH + FP_MANT_WIDTH : 0] buffered_data;
  logic start_to_load_out;
  logic [ACC_ADDR_WIDTH-1:0] latched_acc_waddr;
  logic complete_extracting_data_from_sa;

  always_ff @(posedge clk or posedge rst) begin
      if (rst) begin
          latched_acc_waddr <= 'b0;
      end else if (acc_waddr_valid) begin
          latched_acc_waddr <= acc_waddr;
      end
  end

  always_ff @(posedge clk or posedge rst) begin
      if (rst) begin
          load_out_counter <= 'b0;
          unrolled_data_out_valid <= 1'b0;
          buffered_data <= 'b0;
          start_to_load_out <= 1'b0;
          complete_extracting_data_from_sa <= 1'b0;
      end else begin
          if (write_to_buffer_valid) begin
              // Load the block data into the buffer
              for (int i = 0; i < M; i++) begin
                  buffered_data[i][latched_acc_waddr * N +: N] <= block_data_in[i];
              end

              if (latched_acc_waddr == ACC_NUM - 1 && wait_for_output) begin
                  load_out_counter <= 'b0;
                  start_to_load_out <= 1'b1;
              end
          end

          complete_extracting_data_from_sa <= write_to_buffer_valid;

          if (load_out_counter == M) begin
              load_out_counter        <= 'b0;
              start_to_load_out       <= 1'b0;
              unrolled_data_out       <= 'b0;
          end else if (start_to_load_out) begin
              unrolled_data_out <= buffered_data[load_out_counter];
              load_out_counter  <= load_out_counter + 1;
          end
          unrolled_data_out_valid <= complete_extracting_data_from_sa;
      end
  end


endmodule
