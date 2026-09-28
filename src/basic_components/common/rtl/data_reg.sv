`timescale 1ns / 1ps

module data_reg #(
  parameter DATA_WIDTH = 8,
  parameter REG_N = 1
)(
  input logic clk,
  input logic rst,
  input logic stall,
  input logic [DATA_WIDTH - 1:0] data_in,
  output logic [DATA_WIDTH - 1:0] data_out
);

  generate
    if (REG_N <= 1) begin : gen_single
      always_ff @(posedge clk) begin
        if (rst) begin
          data_out <= 0;
        end else if (!stall) begin
          data_out <= data_in;
        end
      end
    end else begin : gen_chain
      logic [DATA_WIDTH-1:0] pipe [REG_N-1:0];

      always_ff @(posedge clk) begin
        if (rst) begin
          pipe[0] <= 0;
        end else if (!stall) begin
          pipe[0] <= data_in;
        end
      end

      for (genvar i = 1; i < REG_N; i++) begin : gen_stage
        always_ff @(posedge clk) begin
          if (rst) begin
            pipe[i] <= 0;
          end else if (!stall) begin
            pipe[i] <= pipe[i-1];
          end
        end
      end

      assign data_out = pipe[REG_N-1];
    end
  endgenerate

endmodule
