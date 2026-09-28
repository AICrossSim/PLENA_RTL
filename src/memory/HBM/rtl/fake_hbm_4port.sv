`timescale 1ns / 1ps
`include "tl_util.svh"

/*
Module      : Fake TileLink interface HBM with 4 ports for simulation.
Description : Combines 4 fake_hbm instances (m_element, m_scale, v_element, v_scale)
              into a single module with unified init and result files.

File Format:
  Init file contains 4 sections separated by @<section_offset>:
    @0          - M Element data
    @<offset1>  - M Scale data
    @<offset2>  - V Element data
    @<offset3>  - V Scale data

  Result file is written with labeled sections for each port.

Status      : Under Development
*/

module fake_hbm_4port #(
    parameter int ADDR_WIDTH            = 32,
    parameter int ELE_DATA_WIDTH        = 64,
    parameter int SCALE_DATA_WIDTH      = 32,
    parameter int BRAM_ADDR_WIDTH       = 20,
    parameter int SourceWidth           = 1,
    parameter int SinkWidth             = 1,
    // Combined init file (contains all 4 sections: m_ele, m_scale, v_ele, v_scale)
    parameter string MemInitFile        = "",
    // Combined result file
    parameter string ResultFile         = "",
    // Scale section byte offset in binary file (0 = auto-detect from header)
    parameter int ScaleByteOffset       = 0
)(
    input logic clk,
    input logic rst,

    // TileLink Device Ports for Matrix Machine
    `TL_DECLARE_DEVICE_PORT(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_element),
    `TL_DECLARE_DEVICE_PORT(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_scale),

    // TileLink Device Ports for Vector Machine
    `TL_DECLARE_DEVICE_PORT(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_element),
    `TL_DECLARE_DEVICE_PORT(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_scale)
);

// Memory depth
localparam int MEM_DEPTH = 1 << BRAM_ADDR_WIDTH;
localparam int ELE_MASK_WIDTH = ELE_DATA_WIDTH / 8;
localparam int SCALE_MASK_WIDTH = SCALE_DATA_WIDTH / 8;

// Internal link declarations
`TL_DECLARE(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_element_link);
`TL_DECLARE(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, m_scale_link);
`TL_DECLARE(ELE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_element_link);
`TL_DECLARE(SCALE_DATA_WIDTH, ADDR_WIDTH, SourceWidth, SinkWidth, v_scale_link);

// Bind ports to internal links
`TL_BIND_DEVICE_PORT(m_element, m_element_link);
`TL_BIND_DEVICE_PORT(m_scale, m_scale_link);
`TL_BIND_DEVICE_PORT(v_element, v_element_link);
`TL_BIND_DEVICE_PORT(v_scale, v_scale_link);

// ============================================================================
// Memory Arrays - 4 separate memories due to different data widths
// ============================================================================
logic [ELE_DATA_WIDTH-1:0]   mem_m_ele   [0:MEM_DEPTH-1];
logic [SCALE_DATA_WIDTH-1:0] mem_m_scale [0:MEM_DEPTH-1];
logic [ELE_DATA_WIDTH-1:0]   mem_v_ele   [0:MEM_DEPTH-1];
logic [SCALE_DATA_WIDTH-1:0] mem_v_scale [0:MEM_DEPTH-1];

// Read data registers
logic [ELE_DATA_WIDTH-1:0]   m_ele_rdata;
logic [SCALE_DATA_WIDTH-1:0] m_scale_rdata;
logic [ELE_DATA_WIDTH-1:0]   v_ele_rdata;
logic [SCALE_DATA_WIDTH-1:0] v_scale_rdata;

// ============================================================================
// M Element Port
// ============================================================================
logic [ADDR_WIDTH-1:0]       m_ele_bram_addr;
logic [ELE_DATA_WIDTH-1:0]   m_ele_bram_wdata;
logic [ELE_MASK_WIDTH-1:0]   m_ele_bram_wmask;
logic                        m_ele_bram_en;
logic                        m_ele_bram_we;

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
    .bram_addr_o(m_ele_bram_addr),
    .bram_wmask_o(m_ele_bram_wmask),
    .bram_wdata_o(m_ele_bram_wdata),
    .bram_rdata_i(m_ele_rdata)
);

always_ff @(posedge clk) begin
    if (m_ele_bram_en) begin
        if (m_ele_bram_we) begin
            for (int i = 0; i < ELE_DATA_WIDTH/8; i++) begin
                if (m_ele_bram_wmask[i])
                    mem_m_ele[m_ele_bram_addr][8*i +: 8] <= m_ele_bram_wdata[8*i +: 8];
            end
        end else begin
            m_ele_rdata <= mem_m_ele[m_ele_bram_addr];
        end
    end
end

// ============================================================================
// M Scale Port
// ============================================================================
logic [ADDR_WIDTH-1:0]       m_scale_bram_addr;
logic [SCALE_DATA_WIDTH-1:0] m_scale_bram_wdata;
logic [SCALE_MASK_WIDTH-1:0] m_scale_bram_wmask;
logic                        m_scale_bram_en;
logic                        m_scale_bram_we;

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
    .bram_addr_o(m_scale_bram_addr),
    .bram_wmask_o(m_scale_bram_wmask),
    .bram_wdata_o(m_scale_bram_wdata),
    .bram_rdata_i(m_scale_rdata)
);

always_ff @(posedge clk) begin
    if (m_scale_bram_en) begin
        if (m_scale_bram_we) begin
            for (int i = 0; i < SCALE_DATA_WIDTH/8; i++) begin
                if (m_scale_bram_wmask[i])
                    mem_m_scale[m_scale_bram_addr][8*i +: 8] <= m_scale_bram_wdata[8*i +: 8];
            end
        end else begin
            m_scale_rdata <= mem_m_scale[m_scale_bram_addr];
        end
    end
end

// ============================================================================
// V Element Port
// ============================================================================
logic [ADDR_WIDTH-1:0]       v_ele_bram_addr;
logic [ELE_DATA_WIDTH-1:0]   v_ele_bram_wdata;
logic [ELE_MASK_WIDTH-1:0]   v_ele_bram_wmask;
logic                        v_ele_bram_en;
logic                        v_ele_bram_we;

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
    .bram_addr_o(v_ele_bram_addr),
    .bram_wmask_o(v_ele_bram_wmask),
    .bram_wdata_o(v_ele_bram_wdata),
    .bram_rdata_i(v_ele_rdata)
);

always_ff @(posedge clk) begin
    if (v_ele_bram_en) begin
        if (v_ele_bram_we) begin
            for (int i = 0; i < ELE_DATA_WIDTH/8; i++) begin
                if (v_ele_bram_wmask[i])
                    mem_v_ele[v_ele_bram_addr][8*i +: 8] <= v_ele_bram_wdata[8*i +: 8];
            end
        end else begin
            v_ele_rdata <= mem_v_ele[v_ele_bram_addr];
        end
    end
end

// ============================================================================
// V Scale Port
// ============================================================================
logic [ADDR_WIDTH-1:0]       v_scale_bram_addr;
logic [SCALE_DATA_WIDTH-1:0] v_scale_bram_wdata;
logic [SCALE_MASK_WIDTH-1:0] v_scale_bram_wmask;
logic                        v_scale_bram_en;
logic                        v_scale_bram_we;

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
    .bram_addr_o(v_scale_bram_addr),
    .bram_wmask_o(v_scale_bram_wmask),
    .bram_wdata_o(v_scale_bram_wdata),
    .bram_rdata_i(v_scale_rdata)
);

always_ff @(posedge clk) begin
    if (v_scale_bram_en) begin
        if (v_scale_bram_we) begin
            for (int i = 0; i < SCALE_DATA_WIDTH/8; i++) begin
                if (v_scale_bram_wmask[i])
                    mem_v_scale[v_scale_bram_addr][8*i +: 8] <= v_scale_bram_wdata[8*i +: 8];
            end
        end else begin
            v_scale_rdata <= mem_v_scale[v_scale_bram_addr];
        end
    end
end

// ============================================================================
// Memory Initialization
// MemInitFile formats supported:
//   1. Single .mem file with section markers (preferred for RTL):
//      // HBM_ELEMENTS
//      0xDATA...
//      // HBM_SCALES
//      0xSCALE...
//   2. Single .bin file: binary with 8-byte header + elements + scales
// ============================================================================
initial begin
    if (MemInitFile != "") begin
        int len;
        string suffix;
        len = MemInitFile.len();

        if (len >= 4) begin
            suffix = MemInitFile.substr(len-4, len-1);
        end else begin
            suffix = "";
        end

        if (suffix == ".mem") begin
            // Single .mem file with section markers
            integer fd;
            string line;
            int addr_ele, addr_scale;
            int in_elements, in_scales;
            logic [ELE_DATA_WIDTH-1:0] ele_word;
            logic [SCALE_DATA_WIDTH-1:0] scale_word;

            $display("Loading HBM memory from: %s", MemInitFile);
            fd = $fopen(MemInitFile, "r");

            if (fd) begin
                addr_ele = 0;
                addr_scale = 0;
                in_elements = 0;
                in_scales = 0;

                while (!$feof(fd)) begin
                    line = "";
                    void'($fgets(line, fd));

                    // Check for section markers
                    if (line.substr(0, 14) == "// HBM_ELEMENTS") begin
                        in_elements = 1;
                        in_scales = 0;
                    end else if (line.substr(0, 12) == "// HBM_SCALES") begin
                        in_elements = 0;
                        in_scales = 1;
                    end else if (line.substr(0, 1) == "0x" || line.substr(0, 1) == "0X") begin
                        // Parse hex data line
                        if (in_elements && addr_ele < MEM_DEPTH) begin
                            void'($sscanf(line, "%h", ele_word));
                            mem_m_ele[addr_ele] = ele_word;
                            mem_v_ele[addr_ele] = ele_word;
                            addr_ele++;
                        end else if (in_scales && addr_scale < MEM_DEPTH) begin
                            void'($sscanf(line, "%h", scale_word));
                            mem_m_scale[addr_scale] = scale_word;
                            mem_v_scale[addr_scale] = scale_word;
                            addr_scale++;
                        end
                    end
                end

                $fclose(fd);
                $display("Loaded %0d element words and %0d scale words", addr_ele, addr_scale);
            end else begin
                $display("ERROR: Could not open file: %s", MemInitFile);
            end
        end else if (suffix == ".bin") begin
            // Combined binary file format
            integer fd;
            int bytes_read;
            logic [7:0] byte_data;
            int addr_ele, addr_scale;
            int byte_idx;
            int total_bytes_read;
            int scale_offset;
            logic [63:0] header_value;
            logic [ELE_DATA_WIDTH-1:0] ele_word;
            logic [SCALE_DATA_WIDTH-1:0] scale_word;

            $display("Loading combined HBM binary from: %s", MemInitFile);
            fd = $fopen(MemInitFile, "rb");

            if (fd) begin
                // Read 8-byte header containing scale section byte offset
                header_value = '0;
                for (byte_idx = 0; byte_idx < 8; byte_idx++) begin
                    bytes_read = $fread(byte_data, fd);
                    if (bytes_read > 0)
                        header_value[byte_idx*8 +: 8] = byte_data;
                end
                scale_offset = header_value[31:0];
                $display("Binary header: scale_offset = %0d bytes", scale_offset);

                total_bytes_read = 8;

                // Read element data until we reach scale_offset
                addr_ele = 0;
                while (total_bytes_read < scale_offset && addr_ele < MEM_DEPTH) begin
                    ele_word = '0;
                    for (byte_idx = 0; byte_idx < ELE_DATA_WIDTH/8; byte_idx++) begin
                        bytes_read = $fread(byte_data, fd);
                        if (bytes_read == 0) break;
                        ele_word[byte_idx*8 +: 8] = byte_data;
                        total_bytes_read++;
                    end
                    if (bytes_read > 0) begin
                        mem_m_ele[addr_ele] = ele_word;
                        mem_v_ele[addr_ele] = ele_word;
                        addr_ele++;
                    end
                end

                // Read scale data
                addr_scale = 0;
                while (!$feof(fd) && addr_scale < MEM_DEPTH) begin
                    scale_word = '0;
                    for (byte_idx = 0; byte_idx < SCALE_DATA_WIDTH/8; byte_idx++) begin
                        bytes_read = $fread(byte_data, fd);
                        if (bytes_read == 0) break;
                        scale_word[byte_idx*8 +: 8] = byte_data;
                    end
                    if (bytes_read > 0) begin
                        mem_m_scale[addr_scale] = scale_word;
                        mem_v_scale[addr_scale] = scale_word;
                        addr_scale++;
                    end
                end

                $fclose(fd);
                $display("Loaded %0d element words and %0d scale words from binary", addr_ele, addr_scale);
            end else begin
                $display("ERROR: Could not open binary file: %s", MemInitFile);
            end
        end else begin
            $display("ERROR: Unknown file format: %s (use .mem or .bin)", MemInitFile);
        end
    end
end

// ============================================================================
// Result File Output - Write combined result file at end of simulation
// ============================================================================
final begin
    if (ResultFile != "") begin
        integer fd;
        fd = $fopen(ResultFile, "w");
        if (fd) begin
            $display("Writing combined HBM result to: %s", ResultFile);

            // Write M Element section
            $fwrite(fd, "// ========== M_ELEMENT ==========\n");
            for (int i = 0; i < MEM_DEPTH; i++) begin
                if (mem_m_ele[i] !== 'x && mem_m_ele[i] !== 0)
                    $fwrite(fd, "@%0h %h\n", i, mem_m_ele[i]);
            end

            // Write M Scale section
            $fwrite(fd, "\n// ========== M_SCALE ==========\n");
            for (int i = 0; i < MEM_DEPTH; i++) begin
                if (mem_m_scale[i] !== 'x && mem_m_scale[i] !== 0)
                    $fwrite(fd, "@%0h %h\n", i, mem_m_scale[i]);
            end

            // Write V Element section
            $fwrite(fd, "\n// ========== V_ELEMENT ==========\n");
            for (int i = 0; i < MEM_DEPTH; i++) begin
                if (mem_v_ele[i] !== 'x && mem_v_ele[i] !== 0)
                    $fwrite(fd, "@%0h %h\n", i, mem_v_ele[i]);
            end

            // Write V Scale section
            $fwrite(fd, "\n// ========== V_SCALE ==========\n");
            for (int i = 0; i < MEM_DEPTH; i++) begin
                if (mem_v_scale[i] !== 'x && mem_v_scale[i] !== 0)
                    $fwrite(fd, "@%0h %h\n", i, mem_v_scale[i]);
            end

            $fclose(fd);
        end
    end
end

endmodule
