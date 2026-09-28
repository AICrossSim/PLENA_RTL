`timescale 1ns / 1ps
`include "tl_util.svh"

/*
Module      : Fake Instruction Memory with TileLink Interface
Description : Single-port instruction memory for simulation
              - TileLink device interface
              - 32-bit instruction width
              - Memory initialization from .mem file
Status      : For Simulation Only
*/

module fake_instr_mem #(
    parameter int ADDR_WIDTH            = 32,
    parameter int DATA_WIDTH            = 32,
    parameter int BRAM_ADDR_WIDTH       = 16,
    parameter int SourceWidth           = 1,
    parameter int SinkWidth             = 1,
    parameter string MemInitFile        = ""
)(
    input logic clk,
    input logic rst,

    // TileLink Device Port for Instruction Memory
    `TL_DECLARE_DEVICE_PORT(DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, instr)
);

// Memory depth
localparam int MEM_DEPTH = 1 << BRAM_ADDR_WIDTH;
localparam int MASK_WIDTH = DATA_WIDTH / 8;

// Internal link declaration
`TL_DECLARE(DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, instr_link);

// Bind port to internal link
`TL_BIND_DEVICE_PORT(instr, instr_link);

// ============================================================================
// Memory Array
// ============================================================================
logic [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];
logic [DATA_WIDTH-1:0] rdata;

// ============================================================================
// TileLink to BRAM Adapter
// ============================================================================
logic [ADDR_WIDTH-1:0]   bram_addr;
logic [DATA_WIDTH-1:0]   bram_wdata;
logic [MASK_WIDTH-1:0]   bram_wmask;
logic                    bram_en;
logic                    bram_we;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_instr (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, instr_link),
    .bram_en_o(bram_en),
    .bram_we_o(bram_we),
    .bram_addr_o(bram_addr),
    .bram_wmask_o(bram_wmask),
    .bram_wdata_o(bram_wdata),
    .bram_rdata_i(rdata)
);

// ============================================================================
// BRAM Read/Write Logic
// ============================================================================
always_ff @(posedge clk) begin
    if (bram_en) begin
        if (bram_we) begin
            // Write operation (masked)
            for (int i = 0; i < DATA_WIDTH/8; i++) begin
                if (bram_wmask[i])
                    mem[bram_addr][8*i +: 8] <= bram_wdata[8*i +: 8];
            end
        end else begin
            // Read operation
            rdata <= mem[bram_addr];
        end
    end
end

// ============================================================================
// Memory Initialization from .mem file
// Format: Plain hex data, one instruction per line
//   0xDEADBEEF
//   0x12345678
//   ...
// or with address markers:
//   @0 0xDEADBEEF
//   @4 0x12345678
// ============================================================================
initial begin
    // Initialize all memory to NOPs or zeros
    for (int i = 0; i < MEM_DEPTH; i++) begin
        mem[i] = 32'h0;
    end

    if (MemInitFile != "") begin
        integer fd;
        string line;
        int addr;
        logic [DATA_WIDTH-1:0] data_word;
        int line_num;

        fd = $fopen(MemInitFile, "r");

        if (fd) begin
            addr = 0;
            line_num = 0;

            while (!$feof(fd)) begin
                line = "";
                void'($fgets(line, fd));
                line_num++;

                // Skip empty lines and comments
                if (line.len() == 0 || line.substr(0, 1) == "//" || line.substr(0, 0) == "#")
                    continue;

                // Check for address marker (@ADDR)
                if (line.substr(0, 0) == "@") begin
                    // Format: @ADDR DATA
                    int scan_result;
                    scan_result = $sscanf(line, "@%h %h", addr, data_word);
                    if (scan_result == 2) begin
                        // Convert byte address to word address
                        addr = addr >> 2;
                        if (addr < MEM_DEPTH) begin
                            mem[addr] = data_word;
                            $display("[INSTR_MEM] @%0h: 0x%h", addr << 2, data_word);
                        end
                        addr++;
                    end
                end else if (line.substr(0, 1) == "0x" || line.substr(0, 1) == "0X") begin
                    // Format: 0xDATA (sequential addressing)
                    void'($sscanf(line, "%h", data_word));
                    if (addr < MEM_DEPTH) begin
                        mem[addr] = data_word;
                        $display("[INSTR_MEM] @%0h: 0x%h", addr << 2, data_word);
                    end
                    addr++;
                end
            end

            $fclose(fd);
            $display("[INSTR_MEM] Loaded %0d instructions", addr);
        end else begin
            $display("[INSTR_MEM] ERROR: Could not open file: %s", MemInitFile);
        end
    end
end

endmodule
