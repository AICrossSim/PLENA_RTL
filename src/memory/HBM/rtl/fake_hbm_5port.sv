`timescale 1ns / 1ps
`include "tl_util.svh"

/*
Module      : Fake TileLink interface HBM with 5 ports for simulation.
Description : Single unified memory accessed by 5 ports (m_element, m_scale,
              v_element, v_scale, instr). Addresses from the system already
              include offsets - no separate addressing needed.

              IMPORTANT: tl_adapter_bram outputs word-aligned addresses, not byte
              addresses. The lower $clog2(DataWidth/8) bits are stripped. We must
              convert back to byte addresses: byte_addr = word_addr * (DataWidth/8)

File Format:
  Init file (hbm.mem): Sequential hex lines (256-bit / 32-byte rows)
    0xDATA_ROW_0     <- Byte address 0x00
    0xDATA_ROW_1     <- Byte address 0x20
    ...

Status      : Under Development
*/

module fake_hbm_5port #(
    parameter int ADDR_WIDTH            = 32,
    parameter int ELE_DATA_WIDTH        = 64,
    parameter int SCALE_DATA_WIDTH      = 16,   // HBM_SCALE_WIDTH from configuration
    parameter int INSTR_DATA_WIDTH      = 32,
    parameter int BRAM_ADDR_WIDTH       = 20,
    parameter int SourceWidth           = 1,
    parameter int SinkWidth             = 1,
    parameter string MemInitFile        = "",
    parameter string ResultFile         = ""
)(
    input logic clk,
    input logic rst,

    // TileLink Device Ports for Matrix Machine
    `TL_DECLARE_DEVICE_PORT(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_element),
    `TL_DECLARE_DEVICE_PORT(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_scale),

    // TileLink Device Ports for Vector Machine
    `TL_DECLARE_DEVICE_PORT(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_element),
    `TL_DECLARE_DEVICE_PORT(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_scale),

    // TileLink Device Port for Instruction Memory
    `TL_DECLARE_DEVICE_PORT(INSTR_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, instr)
);

// ============================================================================
// Memory Configuration
// ============================================================================
localparam int MEM_DEPTH = 1 << BRAM_ADDR_WIDTH;
localparam int MEM_SIZE_BYTES = MEM_DEPTH * 32;

localparam int ELE_BYTES = ELE_DATA_WIDTH / 8;
localparam int SCALE_BYTES = SCALE_DATA_WIDTH / 8;
localparam int INSTR_BYTES = INSTR_DATA_WIDTH / 8;

localparam int ELE_MASK_WIDTH = ELE_BYTES;
localparam int SCALE_MASK_WIDTH = SCALE_BYTES;
localparam int INSTR_MASK_WIDTH = INSTR_BYTES;

// ============================================================================
// Single Unified Memory - byte array
// ============================================================================
logic [7:0] mem [0:MEM_SIZE_BYTES-1];

// Track highest written address for result dump
int max_written_addr;

// Internal link declarations
`TL_DECLARE(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_element_link);
`TL_DECLARE(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_scale_link);
`TL_DECLARE(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_element_link);
`TL_DECLARE(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_scale_link);
`TL_DECLARE(INSTR_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, instr_link);

// Bind ports to internal links
`TL_BIND_DEVICE_PORT(m_element, m_element_link);
`TL_BIND_DEVICE_PORT(m_scale, m_scale_link);
`TL_BIND_DEVICE_PORT(v_element, v_element_link);
`TL_BIND_DEVICE_PORT(v_scale, v_scale_link);
`TL_BIND_DEVICE_PORT(instr, instr_link);

// Read data registers
logic [ELE_DATA_WIDTH-1:0]   m_ele_rdata;
logic [SCALE_DATA_WIDTH-1:0] m_scale_rdata;
logic [ELE_DATA_WIDTH-1:0]   v_ele_rdata;
logic [SCALE_DATA_WIDTH-1:0] v_scale_rdata;
logic [INSTR_DATA_WIDTH-1:0] instr_rdata;

// ============================================================================
// M Element Port (ELE_DATA_WIDTH-bit access)
// ============================================================================
logic [BRAM_ADDR_WIDTH-1:0]  m_ele_word_addr;
logic [ELE_DATA_WIDTH-1:0]   m_ele_bram_wdata;
logic [ELE_MASK_WIDTH-1:0]   m_ele_bram_wmask;
logic                        m_ele_bram_en;
logic                        m_ele_bram_we;

// Convert word address to byte address
wire [ADDR_WIDTH-1:0] m_ele_byte_addr = {{(ADDR_WIDTH-BRAM_ADDR_WIDTH){1'b0}}, m_ele_word_addr} * ELE_BYTES;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(ELE_DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_m_ele (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, m_element_link),
    .bram_en_o(m_ele_bram_en),
    .bram_we_o(m_ele_bram_we),
    .bram_addr_o(m_ele_word_addr),
    .bram_wmask_o(m_ele_bram_wmask),
    .bram_wdata_o(m_ele_bram_wdata),
    .bram_rdata_i(m_ele_rdata)
);

genvar m_ele_i;
generate
    for (m_ele_i = 0; m_ele_i < ELE_MASK_WIDTH; m_ele_i++) begin : gen_m_ele_write
        always_ff @(posedge clk) begin
            if (m_ele_bram_en && m_ele_bram_we && m_ele_bram_wmask[m_ele_i]) begin
                mem[m_ele_byte_addr + m_ele_i] <= m_ele_bram_wdata[8*m_ele_i +: 8];
            end
        end
    end
endgenerate

genvar m_ele_r;
generate
    for (m_ele_r = 0; m_ele_r < ELE_MASK_WIDTH; m_ele_r++) begin : gen_m_ele_read
        always_ff @(posedge clk) begin
            if (m_ele_bram_en && !m_ele_bram_we) begin
                m_ele_rdata[8*m_ele_r +: 8] <= mem[m_ele_byte_addr + m_ele_r];
            end
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (m_ele_bram_en && m_ele_bram_we) begin
        if (m_ele_byte_addr + ELE_BYTES - 1 > max_written_addr)
            max_written_addr <= m_ele_byte_addr + ELE_BYTES - 1;
    end
end

// ============================================================================
// M Scale Port (SCALE_DATA_WIDTH-bit access)
// ============================================================================
logic [BRAM_ADDR_WIDTH-1:0]  m_scale_word_addr;
logic [SCALE_DATA_WIDTH-1:0] m_scale_bram_wdata;
logic [SCALE_MASK_WIDTH-1:0] m_scale_bram_wmask;
logic                        m_scale_bram_en;
logic                        m_scale_bram_we;

wire [ADDR_WIDTH-1:0] m_scale_byte_addr = {{(ADDR_WIDTH-BRAM_ADDR_WIDTH){1'b0}}, m_scale_word_addr} * SCALE_BYTES;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(SCALE_DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_m_scale (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, m_scale_link),
    .bram_en_o(m_scale_bram_en),
    .bram_we_o(m_scale_bram_we),
    .bram_addr_o(m_scale_word_addr),
    .bram_wmask_o(m_scale_bram_wmask),
    .bram_wdata_o(m_scale_bram_wdata),
    .bram_rdata_i(m_scale_rdata)
);

genvar m_scale_i;
generate
    for (m_scale_i = 0; m_scale_i < SCALE_MASK_WIDTH; m_scale_i++) begin : gen_m_scale_write
        always_ff @(posedge clk) begin
            if (m_scale_bram_en && m_scale_bram_we && m_scale_bram_wmask[m_scale_i]) begin
                mem[m_scale_byte_addr + m_scale_i] <= m_scale_bram_wdata[8*m_scale_i +: 8];
            end
        end
    end
endgenerate

genvar m_scale_r;
generate
    for (m_scale_r = 0; m_scale_r < SCALE_MASK_WIDTH; m_scale_r++) begin : gen_m_scale_read
        always_ff @(posedge clk) begin
            if (m_scale_bram_en && !m_scale_bram_we) begin
                m_scale_rdata[8*m_scale_r +: 8] <= mem[m_scale_byte_addr + m_scale_r];
            end
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (m_scale_bram_en && m_scale_bram_we) begin
        if (m_scale_byte_addr + SCALE_BYTES - 1 > max_written_addr)
            max_written_addr <= m_scale_byte_addr + SCALE_BYTES - 1;
    end
end

// ============================================================================
// V Element Port (ELE_DATA_WIDTH-bit access)
// ============================================================================
logic [BRAM_ADDR_WIDTH-1:0]  v_ele_word_addr;
logic [ELE_DATA_WIDTH-1:0]   v_ele_bram_wdata;
logic [ELE_MASK_WIDTH-1:0]   v_ele_bram_wmask;
logic                        v_ele_bram_en;
logic                        v_ele_bram_we;

wire [ADDR_WIDTH-1:0] v_ele_byte_addr = {{(ADDR_WIDTH-BRAM_ADDR_WIDTH){1'b0}}, v_ele_word_addr} * ELE_BYTES;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(ELE_DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_v_ele (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, v_element_link),
    .bram_en_o(v_ele_bram_en),
    .bram_we_o(v_ele_bram_we),
    .bram_addr_o(v_ele_word_addr),
    .bram_wmask_o(v_ele_bram_wmask),
    .bram_wdata_o(v_ele_bram_wdata),
    .bram_rdata_i(v_ele_rdata)
);

genvar v_ele_i;
generate
    for (v_ele_i = 0; v_ele_i < ELE_MASK_WIDTH; v_ele_i++) begin : gen_v_ele_write
        always_ff @(posedge clk) begin
            if (v_ele_bram_en && v_ele_bram_we && v_ele_bram_wmask[v_ele_i]) begin
                mem[v_ele_byte_addr + v_ele_i] <= v_ele_bram_wdata[8*v_ele_i +: 8];
            end
        end
    end
endgenerate

genvar v_ele_r;
generate
    for (v_ele_r = 0; v_ele_r < ELE_MASK_WIDTH; v_ele_r++) begin : gen_v_ele_read
        always_ff @(posedge clk) begin
            if (v_ele_bram_en && !v_ele_bram_we) begin
                v_ele_rdata[8*v_ele_r +: 8] <= mem[v_ele_byte_addr + v_ele_r];
            end
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (v_ele_bram_en && v_ele_bram_we) begin
        if (v_ele_byte_addr + ELE_BYTES - 1 > max_written_addr)
            max_written_addr <= v_ele_byte_addr + ELE_BYTES - 1;
    end
end

// ============================================================================
// V Scale Port (SCALE_DATA_WIDTH-bit access)
// ============================================================================
logic [BRAM_ADDR_WIDTH-1:0]  v_scale_word_addr;
logic [SCALE_DATA_WIDTH-1:0] v_scale_bram_wdata;
logic [SCALE_MASK_WIDTH-1:0] v_scale_bram_wmask;
logic                        v_scale_bram_en;
logic                        v_scale_bram_we;

wire [ADDR_WIDTH-1:0] v_scale_byte_addr = {{(ADDR_WIDTH-BRAM_ADDR_WIDTH){1'b0}}, v_scale_word_addr} * SCALE_BYTES;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(SCALE_DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_v_scale (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, v_scale_link),
    .bram_en_o(v_scale_bram_en),
    .bram_we_o(v_scale_bram_we),
    .bram_addr_o(v_scale_word_addr),
    .bram_wmask_o(v_scale_bram_wmask),
    .bram_wdata_o(v_scale_bram_wdata),
    .bram_rdata_i(v_scale_rdata)
);

genvar v_scale_i;
generate
    for (v_scale_i = 0; v_scale_i < SCALE_MASK_WIDTH; v_scale_i++) begin : gen_v_scale_write
        always_ff @(posedge clk) begin
            if (v_scale_bram_en && v_scale_bram_we && v_scale_bram_wmask[v_scale_i]) begin
                mem[v_scale_byte_addr + v_scale_i] <= v_scale_bram_wdata[8*v_scale_i +: 8];
            end
        end
    end
endgenerate

genvar v_scale_r;
generate
    for (v_scale_r = 0; v_scale_r < SCALE_MASK_WIDTH; v_scale_r++) begin : gen_v_scale_read
        always_ff @(posedge clk) begin
            if (v_scale_bram_en && !v_scale_bram_we) begin
                v_scale_rdata[8*v_scale_r +: 8] <= mem[v_scale_byte_addr + v_scale_r];
            end
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (v_scale_bram_en && v_scale_bram_we) begin
        if (v_scale_byte_addr + SCALE_BYTES - 1 > max_written_addr)
            max_written_addr <= v_scale_byte_addr + SCALE_BYTES - 1;
    end
end

// ============================================================================
// Instruction Port (INSTR_DATA_WIDTH-bit access)
// ============================================================================
logic [BRAM_ADDR_WIDTH-1:0]  instr_word_addr;
logic [INSTR_DATA_WIDTH-1:0] instr_bram_wdata;
logic [INSTR_MASK_WIDTH-1:0] instr_bram_wmask;
logic                        instr_bram_en;
logic                        instr_bram_we;

wire [ADDR_WIDTH-1:0] instr_byte_addr = {{(ADDR_WIDTH-BRAM_ADDR_WIDTH){1'b0}}, instr_word_addr} * INSTR_BYTES;

tl_adapter_bram #(
    .AddrWidth(ADDR_WIDTH),
    .DataWidth(INSTR_DATA_WIDTH),
    .SourceWidth(SourceWidth),
    .BramAddrWidth(BRAM_ADDR_WIDTH)
) tl_adapter_instr (
    .clk_i(clk),
    .rst_ni(!rst),
    `TL_CONNECT_DEVICE_PORT(host, instr_link),
    .bram_en_o(instr_bram_en),
    .bram_we_o(instr_bram_we),
    .bram_addr_o(instr_word_addr),
    .bram_wmask_o(instr_bram_wmask),
    .bram_wdata_o(instr_bram_wdata),
    .bram_rdata_i(instr_rdata)
);

genvar instr_i;
generate
    for (instr_i = 0; instr_i < INSTR_MASK_WIDTH; instr_i++) begin : gen_instr_write
        always_ff @(posedge clk) begin
            if (instr_bram_en && instr_bram_we && instr_bram_wmask[instr_i]) begin
                mem[instr_byte_addr + instr_i] <= instr_bram_wdata[8*instr_i +: 8];
            end
        end
    end
endgenerate

genvar instr_r;
generate
    for (instr_r = 0; instr_r < INSTR_MASK_WIDTH; instr_r++) begin : gen_instr_read
        always_ff @(posedge clk) begin
            if (instr_bram_en && !instr_bram_we) begin
                instr_rdata[8*instr_r +: 8] <= mem[instr_byte_addr + instr_r];
            end
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (instr_bram_en && instr_bram_we) begin
        if (instr_byte_addr + INSTR_BYTES - 1 > max_written_addr)
            max_written_addr <= instr_byte_addr + INSTR_BYTES - 1;
    end
end

// ============================================================================
// Memory Initialization
// Load from hbm.mem file: Sequential 256-bit (32-byte) hex rows
// Each line: 0xDATA_256BIT  ->  stored at sequential byte addresses
// ============================================================================
initial begin
    integer init_i;
    integer len;
    string suffix;
    integer fd;
    string line;
    integer line_num;
    integer base_byte_addr;
    integer i;
    logic [255:0] row_data;

    // Initialize memory to zero
    max_written_addr = 0;
    for (init_i = 0; init_i < MEM_SIZE_BYTES; init_i++) begin
        mem[init_i] = '0;
    end

    if (MemInitFile != "") begin
        len = MemInitFile.len();

        if (len >= 4) begin
            suffix = MemInitFile.substr(len-4, len-1);
        end else begin
            suffix = "";
        end

        if (suffix == ".mem") begin
            $display("Loading unified HBM memory from: %s", MemInitFile);
            fd = $fopen(MemInitFile, "r");

            if (fd) begin
                line_num = 0;

                while (!$feof(fd)) begin
                    line = "";
                    void'($fgets(line, fd));

                    // Skip empty lines and comments
                    if (line.len() < 3) continue;
                    if (line.substr(0, 1) == "//") continue;

                    // Parse hex data line
                    if (line.substr(0, 1) == "0x" || line.substr(0, 1) == "0X") begin
                        void'($sscanf(line, "%h", row_data));

                        // Each 256-bit row = 32 bytes
                        // Row N starts at byte address N*32
                        base_byte_addr = line_num * 32;

                        for (i = 0; i < 32; i++) begin
                            if (base_byte_addr + i < MEM_SIZE_BYTES) begin
                                mem[base_byte_addr + i] = row_data[i*8 +: 8];
                            end
                        end

                        if (base_byte_addr + 31 > max_written_addr)
                            max_written_addr = base_byte_addr + 31;

                        line_num++;
                    end
                end

                $fclose(fd);
                $display("Loaded %0d rows (%0d bytes) into unified HBM memory", line_num, line_num * 32);
            end else begin
                $display("ERROR: Could not open file: %s", MemInitFile);
            end
        end else begin
            $display("ERROR: Unknown file format: %s (use .mem)", MemInitFile);
        end
    end
end

// ============================================================================
// Result File Output - Write memory at end of simulation
// Output format: 256-bit (32-byte) hex rows, matching input format
// ============================================================================
final begin
    integer fd;
    integer num_rows;
    integer row;
    integer base_byte;
    integer i;
    logic [255:0] row_data;

    if (ResultFile != "") begin
        fd = $fopen(ResultFile, "w");
        if (fd) begin
            num_rows = (max_written_addr + 32) / 32;  // Round up to full rows
            $display("Writing HBM result to: %s (%0d rows)", ResultFile, num_rows);

            for (row = 0; row < num_rows; row++) begin
                base_byte = row * 32;

                // Assemble 256-bit row from 32 bytes
                for (i = 0; i < 32; i++) begin
                    row_data[i*8 +: 8] = mem[base_byte + i];
                end

                $fwrite(fd, "0x%064H\n", row_data);
            end

            $fclose(fd);
        end
    end
end

endmodule
