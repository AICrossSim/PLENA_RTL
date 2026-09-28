// Wrapper for biaccess_sram with specific parameters for the experiment
// DATA_WIDTH = 8, SRAM_DEPTH = MLEN = 32, Parallel_Rd_Dim = 1

`timescale 1ns/1ps

module biaccess_sram_32x32 (
    input  logic clk,

    // Read interface
    input  logic req,
    input  logic transposed_read,
    input  logic [4:0] sram_raddr,  // $clog2(32) = 5
    output logic [31:0] [7:0] out_data,  // Parallel_Rd_Dim * MLEN = 1 * 32 = 32 elements, each 8 bits

    // Write interface
    input  logic wen_req,
    output logic write_response,
    input  logic [4:0] sram_waddr,
    input  logic [255:0] write_data  // Parallel_Rd_Dim * MLEN * DataWidth = 1 * 32 * 8 = 256 bits
);

    biaccess_sram #(
        .DataWidth(8),
        .SRAM_DEPTH(32),
        .MLEN(32),
        .Parallel_Rd_Dim(1)
    ) u_biaccess_sram (
        .clk(clk),
        .req(req),
        .transposed_read(transposed_read),
        .sram_raddr(sram_raddr),
        .out_data(out_data),
        .wen_req(wen_req),
        .write_response(write_response),
        .sram_waddr(sram_waddr),
        .write_data(write_data)
    );

endmodule
