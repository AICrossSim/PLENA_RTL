// Pure SRAM design (32 rows, each 256 bits = 32 elements * 8 bits)
// This is the baseline SRAM without transpose capability

`timescale 1ns/1ps

module pure_sram_32x32 (
    input  logic clk,

    // Read interface
    input  logic req,
    input  logic [4:0] raddr,
    output logic [31:0] [7:0] rdata,
    output logic read_valid,

    // Write interface
    input  logic wen,
    input  logic [4:0] waddr,
    input  logic [255:0] wdata,
    output logic write_response
);

    // Memory array: 32 rows, each 256 bits
    logic [255:0] mem [32];

    // Write logic
    always_ff @(posedge clk) begin
        if (wen) begin
            mem[waddr] <= wdata;
            write_response <= 1'b1;
        end else begin
            write_response <= 1'b0;
        end
    end

    // Read logic
    always_ff @(posedge clk) begin
        if (req) begin
            rdata <= mem[raddr];
            read_valid <= 1'b1;
        end else begin
            read_valid <= 1'b0;
        end
    end

endmodule
