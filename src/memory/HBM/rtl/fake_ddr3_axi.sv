`timescale 1ns / 1ps
// Behavioural DDR3-like AXI4 memory slave for simulating the FPGA HBM datapath
// (plena TileLink -> 4-way mux -> tl_to_axi4 -> HERE), standing in for the Xilinx MIG
// DDR3 controller. Byte-addressed memory loaded from the SAME hbm.mem the fake_hbm uses,
// so the VRAM result matches the direct-fake_hbm sim bit-for-bit. Single-outstanding with
// configurable read/write latency to mimic a single DDR3 controller (no request pipelining),
// which is exactly what exposes the Nexys serialized/priority-muxed prefetch behaviour.
module fake_ddr3_axi #(
    parameter int AXI_ADDR_W    = 29,
    parameter int AXI_DATA_W    = 128,
    parameter int AXI_ID_W      = 4,
    parameter int MEM_BYTES     = 1 << 21,   // 2 MB window (>= any workload's HBM footprint)
    parameter int RD_LATENCY    = 20,        // cycles from AR accept to first R beat (DDR3-ish)
    parameter int WR_LATENCY    = 20,        // cycles from last W beat to B response
    parameter int CALIB_CYCLES  = 32,        // ui/init_calib_complete delay after reset
    parameter string MemInitFile = ""
)(
    input  logic clk,
    input  logic rst,

    input  logic [AXI_ID_W-1:0]     s_axi_awid,
    input  logic [AXI_ADDR_W-1:0]   s_axi_awaddr,
    input  logic [7:0]              s_axi_awlen,
    input  logic [2:0]              s_axi_awsize,
    input  logic [1:0]              s_axi_awburst,
    input  logic                    s_axi_awvalid,
    output logic                    s_axi_awready,

    input  logic [AXI_DATA_W-1:0]   s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0] s_axi_wstrb,
    input  logic                    s_axi_wlast,
    input  logic                    s_axi_wvalid,
    output logic                    s_axi_wready,

    output logic [AXI_ID_W-1:0]     s_axi_bid,
    output logic [1:0]              s_axi_bresp,
    output logic                    s_axi_bvalid,
    input  logic                    s_axi_bready,

    input  logic [AXI_ID_W-1:0]     s_axi_arid,
    input  logic [AXI_ADDR_W-1:0]   s_axi_araddr,
    input  logic [7:0]              s_axi_arlen,
    input  logic [2:0]              s_axi_arsize,
    input  logic [1:0]              s_axi_arburst,
    input  logic                    s_axi_arvalid,
    output logic                    s_axi_arready,

    output logic [AXI_ID_W-1:0]     s_axi_rid,
    output logic [AXI_DATA_W-1:0]   s_axi_rdata,
    output logic [1:0]              s_axi_rresp,
    output logic                    s_axi_rlast,
    output logic                    s_axi_rvalid,
    input  logic                    s_axi_rready,

    output logic                    init_calib_complete
);
    localparam int BEAT_BYTES  = AXI_DATA_W / 8;
    localparam int MEM_AW      = $clog2(MEM_BYTES);

    logic [7:0] mem [0:MEM_BYTES-1];

    // ---- load: mirrors fake_hbm_5port (hbm.mem = sequential 256-bit / 32-byte hex rows) ----
    initial begin
        integer len, fd, line_num, base_byte_addr, i;
        string  suffix, line;
        logic [255:0] row_data;
        for (i = 0; i < MEM_BYTES; i++) mem[i] = '0;
        if (MemInitFile != "") begin
            len    = MemInitFile.len();
            suffix = (len >= 4) ? MemInitFile.substr(len-4, len-1) : "";
            if (suffix == ".mem") begin
                fd = $fopen(MemInitFile, "r");
                if (fd) begin
                    line_num = 0;
                    while (!$feof(fd)) begin
                        line = "";
                        void'($fgets(line, fd));
                        if (line.len() < 3) continue;
                        if (line.substr(0, 1) == "//") continue;
                        if (line.substr(0, 1) == "0x" || line.substr(0, 1) == "0X") begin
                            void'($sscanf(line, "%h", row_data));
                            base_byte_addr = line_num * 32;
                            for (i = 0; i < 32; i++)
                                if (base_byte_addr + i < MEM_BYTES)
                                    mem[base_byte_addr + i] = row_data[i*8 +: 8];
                            line_num++;
                        end
                    end
                    $fclose(fd);
                    $display("fake_ddr3_axi: loaded %0d rows (%0d bytes) from %s", line_num, line_num*32, MemInitFile);
                end
            end
        end
    end

    // ---- init calibration (MIG asserts init_calib_complete after DDR3 training) ----
    logic [$clog2(CALIB_CYCLES+1)-1:0] calib_cnt;
    always_ff @(posedge clk) begin
        if (rst)                        calib_cnt <= '0;
        else if (calib_cnt != CALIB_CYCLES) calib_cnt <= calib_cnt + 1'b1;
    end
    assign init_calib_complete = (calib_cnt == CALIB_CYCLES);

    // ---- single-outstanding AXI4 slave FSM (read or write at a time) ----
    typedef enum logic [2:0] {IDLE, R_WAIT, R_DATA, W_DATA, B_WAIT, B_RESP} state_t;
    state_t state;

    logic [AXI_ADDR_W-1:0] addr_q;
    logic [7:0]            len_q;
    logic [AXI_ID_W-1:0]   id_q;
    logic [7:0]            beat_q;
    logic [$clog2((RD_LATENCY>WR_LATENCY?RD_LATENCY:WR_LATENCY)+1)-1:0] lat_cnt;

    function automatic logic [AXI_DATA_W-1:0] read_beat(input logic [AXI_ADDR_W-1:0] base, input logic [7:0] b);
        logic [AXI_ADDR_W-1:0] ba;
        logic [AXI_DATA_W-1:0] d;
        ba = base + b * BEAT_BYTES;
        for (int k = 0; k < BEAT_BYTES; k++) d[8*k +: 8] = mem[(ba + k) & (MEM_BYTES-1)];
        return d;
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; beat_q <= '0; lat_cnt <= '0;
            s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rlast <= 1'b0;
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
            s_axi_rdata <= '0; s_axi_rid <= '0; s_axi_bid <= '0;
            s_axi_rresp <= 2'b00; s_axi_bresp <= 2'b00;
        end else begin
            s_axi_arready <= 1'b0;
            s_axi_awready <= 1'b0;
            case (state)
                IDLE: begin
                    // write has priority; bridge is single-outstanding so only one is valid.
                    if (s_axi_awvalid) begin
                        s_axi_awready <= 1'b1;
                        addr_q <= s_axi_awaddr; len_q <= s_axi_awlen; id_q <= s_axi_awid;
                        beat_q <= '0; s_axi_wready <= 1'b1; state <= W_DATA;
                    end else if (s_axi_arvalid) begin
                        s_axi_arready <= 1'b1;
                        addr_q <= s_axi_araddr; len_q <= s_axi_arlen; id_q <= s_axi_arid;
                        beat_q <= '0; lat_cnt <= RD_LATENCY[$bits(lat_cnt)-1:0]; state <= R_WAIT;
                    end
                end
                R_WAIT: begin
                    if (lat_cnt != 0) lat_cnt <= lat_cnt - 1'b1;
                    else begin
                        s_axi_rvalid <= 1'b1;
                        s_axi_rid    <= id_q;
                        s_axi_rresp  <= 2'b00;
                        s_axi_rdata  <= read_beat(addr_q, beat_q);
                        s_axi_rlast  <= (len_q == 0);
                        state <= R_DATA;
                    end
                end
                R_DATA: begin
                    if (s_axi_rvalid && s_axi_rready) begin
                        if (beat_q == len_q) begin
                            s_axi_rvalid <= 1'b0; s_axi_rlast <= 1'b0; state <= IDLE;
                        end else begin
                            beat_q <= beat_q + 1'b1;
                            s_axi_rdata <= read_beat(addr_q, beat_q + 1'b1);
                            s_axi_rlast <= ((beat_q + 1'b1) == len_q);
                        end
                    end
                end
                W_DATA: begin
                    s_axi_wready <= 1'b1;
                    if (s_axi_wvalid && s_axi_wready) begin
                        for (int k = 0; k < BEAT_BYTES; k++)
                            if (s_axi_wstrb[k])
                                mem[(addr_q + beat_q*BEAT_BYTES + k) & (MEM_BYTES-1)] <= s_axi_wdata[8*k +: 8];
                        if (s_axi_wlast) begin
                            s_axi_wready <= 1'b0;
                            lat_cnt <= WR_LATENCY[$bits(lat_cnt)-1:0]; state <= B_WAIT;
                        end else beat_q <= beat_q + 1'b1;
                    end
                end
                B_WAIT: begin
                    if (lat_cnt != 0) lat_cnt <= lat_cnt - 1'b1;
                    else begin
                        s_axi_bvalid <= 1'b1; s_axi_bid <= id_q; s_axi_bresp <= 2'b00; state <= B_RESP;
                    end
                end
                B_RESP: begin
                    if (s_axi_bvalid && s_axi_bready) begin s_axi_bvalid <= 1'b0; state <= IDLE; end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
