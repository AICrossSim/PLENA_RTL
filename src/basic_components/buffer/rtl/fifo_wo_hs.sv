/*
Module      : fifo_wo_hs
Description : FIFO without handshake signals.
              Assumes upstream will not overflow (no backpressure).
              Assumes downstream is always ready to consume.
              Has one cycle read latency.
*/

`timescale 1ns / 1ps

module fifo_wo_hs #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 16
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic                  data_in_valid,

    output logic [DATA_WIDTH-1:0] data_out,
    output logic                  data_out_valid,

    output logic empty,
    output logic full
);

  generate
    if (DEPTH == 1) begin : gen_single_reg

      register_slice_wo_hs #(
          .DATA_WIDTH(DATA_WIDTH)
      ) reg_inst (
          .clk           (clk),
          .rst           (rst),
          .data_in       (data_in),
          .data_in_valid (data_in_valid),
          .data_out      (data_out),
          .data_out_valid(data_out_valid)
      );

      assign empty = !data_out_valid;
      assign full  = data_out_valid;

    end else begin : gen_fifo

      localparam ADDR_WIDTH = $clog2(DEPTH);
      localparam PTR_WIDTH = ADDR_WIDTH + 1;

      logic [PTR_WIDTH-1:0] write_ptr;
      logic [PTR_WIDTH-1:0] read_ptr;
      logic [PTR_WIDTH-1:0] size;
      logic ram_dout_valid;
      logic [DATA_WIDTH-1:0] ram_rd_dout;
      logic ram_wr_en;

      // Write logic
      always_ff @(posedge clk) begin
        if (rst) begin
          write_ptr <= '0;
        end else if (data_in_valid) begin
          if (write_ptr == DEPTH - 1) begin
            write_ptr <= '0;
          end else begin
            write_ptr <= write_ptr + 1;
          end
        end
      end

      assign ram_wr_en = data_in_valid;

      // Read logic - read when FIFO is not empty
      always_ff @(posedge clk) begin
        if (rst) begin
          read_ptr <= '0;
          ram_dout_valid <= 1'b0;
        end else begin
          if (size != 0) begin
            if (read_ptr == DEPTH - 1) begin
              read_ptr <= '0;
            end else begin
              read_ptr <= read_ptr + 1;
            end
            ram_dout_valid <= 1'b1;
          end else begin
            ram_dout_valid <= 1'b0;
          end
        end
      end

      // Size tracking
      always_ff @(posedge clk) begin
        if (rst) begin
          size <= '0;
        end else begin
          case ({data_in_valid, (size != 0)})
            2'b10:   size <= size + 1;  // Write only
            2'b01:   size <= size - 1;  // Read only
            default: size <= size;      // Both or neither
          endcase
        end
      end

      // Output register
      always_ff @(posedge clk) begin
        if (rst) begin
          data_out <= '0;
          data_out_valid <= 1'b0;
        end else begin
          data_out <= ram_rd_dout;
          data_out_valid <= ram_dout_valid;
        end
      end

      simple_dual_port_ram #(
          .DATA_WIDTH(DATA_WIDTH),
          .ADDR_WIDTH(ADDR_WIDTH),
          .SIZE      (DEPTH)
      ) ram_inst (
          .clk    (clk),
          .wr_addr(write_ptr[ADDR_WIDTH-1:0]),
          .wr_din (data_in),
          .wr_en  (ram_wr_en),
          .rd_addr(read_ptr[ADDR_WIDTH-1:0]),
          .rd_dout(ram_rd_dout)
      );

      assign empty = (size == 0);
      assign full  = (size == DEPTH);

    end
  endgenerate

endmodule
