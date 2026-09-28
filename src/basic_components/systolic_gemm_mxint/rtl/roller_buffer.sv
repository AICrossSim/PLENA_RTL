`timescale 1ns / 1ps

/*
Module      : Roller Buffer (parallel-in, serial-out)
Timing      : Sequential
Description : A classic PISO shift register. On a `load` pulse it latches DEPTH
              parallel words; starting the NEXT cycle it rolls one word out per
              cycle (index 0 first), asserting data_out_valid for DEPTH cycles,
              then goes idle.

              Used to feed a systolic array edge: load row/column t at cycle t,
              and the buffer emits its DEPTH K-values starting at cycle t+1.
              Because adjacent edges are loaded one cycle apart, their serial
              outputs are automatically skewed by one cycle — exactly the
              systolic wavefront the PE grid needs.

Timing:
   cycle t   : load = 1, data_in = {w[DEPTH-1], ..., w[1], w[0]}
   cycle t+1 : data_out = w[0], valid = 1
   cycle t+2 : data_out = w[1], valid = 1
   ...
   cycle t+DEPTH : data_out = w[DEPTH-1], valid = 1
   cycle t+DEPTH+1 : valid = 0
*/

module roller_buffer #(
    parameter DATA_WIDTH = 4,
    parameter DEPTH      = 16
)(
    input  logic clk,
    input  logic rst,

    input  logic [DEPTH-1:0][DATA_WIDTH-1:0] data_in,   // parallel load
    input  logic                              load,      // latch + start rolling

    output logic [DATA_WIDTH-1:0]             data_out,  // serial out (index 0 first)
    output logic                              data_out_valid
);

    localparam CNT_W = $clog2(DEPTH + 1);

    logic [DEPTH-1:0][DATA_WIDTH-1:0] shift_reg;
    logic [CNT_W-1:0]                 count;     // words remaining to roll out
    logic                             active;

    always_ff @(posedge clk) begin
        if (rst) begin
            shift_reg <= '0;
            count     <= '0;
            active    <= 1'b0;
        end else if (load) begin
            shift_reg <= data_in;
            count     <= DEPTH[CNT_W-1:0];
            active    <= 1'b1;
        end else if (active) begin
            // roll: drop word 0, shift the rest down
            shift_reg <= shift_reg >> DATA_WIDTH;
            if (count == 1) active <= 1'b0;
            count <= count - 1'b1;
        end
    end

    assign data_out       = shift_reg[0];
    assign data_out_valid = active;

endmodule
