`timescale 1ns / 1ps
module register_slice_wo_hs #(
    parameter DATA_WIDTH = 8
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic data_in_valid,

    output logic [DATA_WIDTH-1:0] data_out,
    output logic data_out_valid
);

  // Simple pipeline register - delays data and valid by one cycle

  always_ff @(posedge clk) begin
    if (rst) begin
      data_out <= '0;
      data_out_valid <= 1'b0;
    end else begin
      data_out <= data_in;
      data_out_valid <= data_in_valid;
    end
  end

endmodule
