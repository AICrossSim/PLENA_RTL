`timescale 1ns / 1ps
`include "tl_util.svh"
`include "tl_pkg.svh"
`include "configuration.svh"
`include "prim_util_pkg.svh"

// Simulation top that validates the FPGA BRING-UP / activation path of plena_a7_top without
// any Xilinx IP. It composes:
//   - plena_a7_top's control half: AXI-lite register block (program load, STATUS, SOFT_RST,
//     debug readback stubs) + the on-chip IMEM BRAM served to the core via tl_adapter_bram.
//   - SimTopDDR's validated datapath half: per-channel skid buffers -> round-robin arbiter ->
//     per-channel response FIFOs -> tl_to_axi4 bridge -> behavioural fake_ddr3_axi.
// The Xilinx jtag_axi master is REPLACED by the exposed AXI-lite port (la_*), driven from a
// cocotb AXI-lite master; the MIG (mig_ddr3) is replaced by fake_ddr3_axi. This lets a cocotb
// testbench exercise the real activation sequence: wait DDR3 calib -> load program over AXI-lite
// -> SOFT_RST launch -> run to system_break -> (result via the standard V_SRAM_RESULT_FILE dump).
// The core is held in reset until the first SOFT_RST (core_launched), so it cannot execute an
// empty/garbage IMEM before the program is loaded -- a robustness gate the board top should adopt.
module SimTopA7_dbg import instruction_pkg::*; import configuration_pkg::*; #(
    parameter           INSTRUCTION_LENGTH            = 32,
    parameter int       IMEM_DEPTH                    = 512,
    parameter string    FAKE_HBM_INIT_FILE            = "",
    parameter string    FP_MEM_INIT_FILE              = "",
    parameter string    INT_MEM_INIT_FILE             = "",
    parameter string    VECTOR_MEM_RESULT_FILE        = "",
    parameter string    FP_REG_RESULT_FILE            = ""
) (
    input  logic        clk,
    input  logic        rst,               // active-high hard reset (== board ui_clk_sync_rst)

    // AXI4-Lite slave (32-bit) -- driven in sim by a cocotb master in place of the jtag_axi IP.
    input  logic [31:0] la_awaddr,
    input  logic        la_awvalid,
    output logic        la_awready,
    input  logic [31:0] la_wdata,
    input  logic [3:0]  la_wstrb,
    input  logic        la_wvalid,
    output logic        la_wready,
    output logic [1:0]  la_bresp,
    output logic        la_bvalid,
    input  logic        la_bready,
    input  logic [31:0] la_araddr,
    input  logic        la_arvalid,
    output logic        la_arready,
    output logic [31:0] la_rdata,
    output logic [1:0]  la_rresp,
    output logic        la_rvalid,
    input  logic        la_rready
);
    import simulation_pkg::*;

    // ---- Bridge / AXI geometry (matches plena_a7_top / mig_nexys_video.prj) ----
    localparam int BRIDGE_DATA_W = HBM_WIDTH;
    localparam int BRIDGE_MASK_W = BRIDGE_DATA_W / 8;
    localparam int AXI_DATA_W    = 128;
    localparam int AXI_ADDR_W    = 29;
    localparam int AXI_ID_W      = 4;
    localparam int AXI_STRB_W    = AXI_DATA_W / 8;
    localparam logic [`TL_SIZE_WIDTH-1:0] BRIDGE_TL_SIZE = `TL_SIZE_WIDTH'($clog2(BRIDGE_DATA_W/8));
    localparam int IMEM_ADDR_W   = $clog2(IMEM_DEPTH);

    `TL_DECLARE(INSTRUCTION_LENGTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, instr_link);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_scale);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_scale);

    wire        init_calib_complete;   // from fake_ddr3_axi
    logic       system_break;          // from plena core

    // ---- Soft reset + launch gate: the core stays in reset until the host issues the first
    // SOFT_RST (0x14), then a 16-cycle reset pulse restarts it at PC0 reading the loaded IMEM. ----
    logic        soft_rst_pulse;
    logic [4:0]  soft_rst_cnt;
    logic        core_launched;
    always_ff @(posedge clk) begin
        if (rst) begin
            soft_rst_cnt  <= '0;
            core_launched <= 1'b0;
        end else begin
            if (soft_rst_pulse)         begin soft_rst_cnt <= 5'd16; core_launched <= 1'b1; end
            else if (soft_rst_cnt != 0) soft_rst_cnt <= soft_rst_cnt - 1'b1;
        end
    end
    wire soft_rst  = (soft_rst_cnt != 0);
    wire plena_rst = rst | soft_rst | ~core_launched;

    // system_break is a COMBINATIONAL 1-cycle pulse (exe_stage_op.c_op==BREAK); the core keeps
    // fetching NOPs past it. Polling that pulse over slow AXI/JTAG misses it, so latch it sticky
    // (cleared on each launch). STATUS reports the latch. plena_a7_top needs this same latch.
    logic system_break_latched;
    always_ff @(posedge clk) begin
        if (rst | plena_rst) system_break_latched <= 1'b0;
        else if (system_break) system_break_latched <= 1'b1;
    end

    plena #(
        .FP_MEM_INIT_FILE(FP_MEM_INIT_FILE),
        .INT_MEM_INIT_FILE(INT_MEM_INIT_FILE),
        .V_SRAM_RESULT_FILE(VECTOR_MEM_RESULT_FILE),
        .FP_REG_RESULT_FILE(FP_REG_RESULT_FILE)
    ) dut (
        .clk(clk),
        .rst(plena_rst),
        .system_break(system_break),
        .debug_vsram_data(dbg_vsram_data), .debug_vsram_en(dbg_vsram_en), .debug_row(dbg_row),
        `TL_CONNECT_HOST_PORT(instr_mem_tl,  instr_link),
        `TL_CONNECT_HOST_PORT(m_out_element, m_element),
        `TL_CONNECT_HOST_PORT(m_out_scale,   m_scale),
        `TL_CONNECT_HOST_PORT(v_out_element, v_element),
        `TL_CONNECT_HOST_PORT(v_out_scale,   v_scale)
    );

    // =========================================================
    // Instruction memory: dual-port BRAM, host-loaded over AXI-lite (port A), read by the core's
    // instr_mem_tl TileLink port via tl_adapter_bram (port B). Mirrors plena_a7_top.
    // =========================================================
    logic [31:0]            imem [0:IMEM_DEPTH-1];

    logic                   imem_bram_en, imem_bram_we;
    logic [IMEM_ADDR_W-1:0] imem_bram_addr;
    logic [3:0]             imem_bram_wmask;
    logic [31:0]            imem_bram_wdata, imem_bram_rdata;

    logic                   imem_we_a;
    logic [IMEM_ADDR_W-1:0] imem_addr_a;
    logic [31:0]            imem_wdata_a;

    always_ff @(posedge clk) begin
        if (imem_we_a)
            imem[imem_addr_a] <= imem_wdata_a;                   // port A: AXI-lite write
        if (imem_bram_en) begin                                  // port B: TileLink read
            if (imem_bram_we)
                for (int i = 0; i < 4; i++)
                    if (imem_bram_wmask[i])
                        imem[imem_bram_addr][8*i +: 8] <= imem_bram_wdata[8*i +: 8];
            imem_bram_rdata <= imem[imem_bram_addr];
        end
    end

    tl_adapter_bram #(
        .AddrWidth    (HBM_ADDR_WIDTH),
        .DataWidth    (INSTRUCTION_LENGTH),
        .SourceWidth  (SourceWidth),
        .BramAddrWidth(IMEM_ADDR_W)
    ) u_imem_adapter (
        .clk_i        (clk),
        .rst_ni       (~rst),               // survives core soft-reset; IMEM persists
        `TL_CONNECT_DEVICE_PORT(host, instr_link),
        .bram_en_o    (imem_bram_en),
        .bram_we_o    (imem_bram_we),
        .bram_addr_o  (imem_bram_addr),
        .bram_wmask_o (imem_bram_wmask),
        .bram_wdata_o (imem_bram_wdata),
        .bram_rdata_i (imem_bram_rdata)
    );

    // =========================================================
    // AXI-lite slave (32-bit). Word addr = addr[7:2]:
    //   0x00 W IMEM_DATA (push instr word, wptr++)   0x10 RW IMEM_WPTR
    //   0x04 R STATUS {init_calib, system_break}      0x14 W SOFT_RST / R FREE_RUN
    //   0x08 W DEBUG_CTRL bit0->debug_vsram_en        0x18 R RST_CNT
    //   0x0C W DEBUG_ADDR -> debug_row                0x40.. R DEBUG_D0..D{N-1}
    // The debug-VSRAM readback path needs core debug ports (branch-only); here it is stubbed
    // (dbg_vsram_data=0) -- result verification uses the standard V_SRAM_RESULT_FILE dump.
    // =========================================================
    localparam int DBG_WORDS = (VECTOR_SRAM_WIDTH + 31) / 32;
    logic                          dbg_vsram_en;
    logic [ON_CHIP_ADDR_WIDTH-1:0] dbg_row;
    logic [VECTOR_SRAM_WIDTH-1:0]  dbg_vsram_data;

    logic [DBG_WORDS*32-1:0] dbg_cap;
    always_ff @(posedge clk)
        dbg_cap <= {{(DBG_WORDS*32-VECTOR_SRAM_WIDTH){1'b0}}, dbg_vsram_data};

    logic [IMEM_ADDR_W-1:0] imem_wptr;
    logic [31:0]            free_run, rst_cnt;
    always_ff @(posedge clk) begin
        free_run <= free_run + 1'b1;
        if (rst)            rst_cnt <= '0;
        else if (plena_rst) rst_cnt <= rst_cnt + 1'b1;
    end

    // ---- JTAG->DDR3 loader: host streams 128-bit beats into DDR3 over the same AXI-lite path
    // (DDR3_ADDR/W0..W3/GO), gated by LOAD_MODE which hands the tl_to_axi4 bridge to the loader
    // while the core is held in reset. DDR3_RD reads a beat back for verification. ----
    localparam int LDR_BEATB = BRIDGE_DATA_W / 8;   // bytes per beat (16)
    logic [AXI_ADDR_W-1:0]     ddr3_addr_q;
    logic [BRIDGE_DATA_W-1:0]  ddr3_wbeat_q;
    logic [BRIDGE_DATA_W-1:0]  ddr3_rbeat_q;
    logic                      ddr3_go_pulse, ddr3_rd_pulse, load_mode;
    logic                      loader_busy, loader_done;

    // ---- Write channel (accept AW+W together; single outstanding) ----
    logic        bvalid_r;
    wire         wr_fire = la_awvalid & la_wvalid & ~bvalid_r;
    wire  [5:0]  wr_w    = la_awaddr[7:2];
    assign la_awready = wr_fire;
    assign la_wready  = wr_fire;
    assign la_bvalid  = bvalid_r;
    assign la_bresp   = 2'b00;
    assign imem_we_a    = wr_fire & (wr_w == 6'h00);
    assign imem_addr_a  = imem_wptr;
    assign imem_wdata_a = la_wdata;

    always_ff @(posedge clk) begin
        if (rst) begin
            bvalid_r <= 1'b0; imem_wptr <= '0;
            dbg_vsram_en <= 1'b0; dbg_row <= '0; soft_rst_pulse <= 1'b0;
            ddr3_addr_q <= '0; ddr3_wbeat_q <= '0; load_mode <= 1'b0;
            ddr3_go_pulse <= 1'b0; ddr3_rd_pulse <= 1'b0;
        end else begin
            soft_rst_pulse <= 1'b0; ddr3_go_pulse <= 1'b0; ddr3_rd_pulse <= 1'b0;
            if (ddr3_go_pulse) ddr3_addr_q <= ddr3_addr_q + LDR_BEATB;  // auto-inc after each write beat
            if (wr_fire)        bvalid_r <= 1'b1;
            else if (la_bready) bvalid_r <= 1'b0;
            if (wr_fire) begin
                unique case (wr_w)
                    6'h00: imem_wptr <= imem_wptr + 1'b1;                  // IMEM_DATA
                    6'h02: dbg_vsram_en <= la_wdata[0];                    // 0x08 DEBUG_CTRL
                    6'h03: dbg_row <= la_wdata[ON_CHIP_ADDR_WIDTH-1:0];    // 0x0C DEBUG_ADDR
                    6'h04: imem_wptr <= la_wdata[IMEM_ADDR_W-1:0];         // 0x10 IMEM_WPTR
                    6'h05: begin soft_rst_pulse <= 1'b1; imem_wptr <= '0; end // 0x14 SOFT_RST
                    6'h07: ddr3_addr_q          <= la_wdata[AXI_ADDR_W-1:0];  // 0x1C DDR3_ADDR
                    6'h08: ddr3_wbeat_q[31:0]   <= la_wdata;                  // 0x20 DDR3_W0
                    6'h09: ddr3_wbeat_q[63:32]  <= la_wdata;                  // 0x24 DDR3_W1
                    6'h0A: ddr3_wbeat_q[95:64]  <= la_wdata;                  // 0x28 DDR3_W2
                    6'h0B: ddr3_wbeat_q[127:96] <= la_wdata;                  // 0x2C DDR3_W3
                    6'h0C: ddr3_go_pulse <= 1'b1;                             // 0x30 DDR3_GO (write beat)
                    6'h0E: load_mode     <= la_wdata[0];                      // 0x38 LOAD_MODE
                    6'h0F: ddr3_rd_pulse <= 1'b1;                             // 0x3C DDR3_RD (read beat)
                    default: ;
                endcase
            end
        end
    end

    // ---- Read channel ----
    logic        rvalid_r;
    logic [31:0] rdata_r;
    wire         rd_fire = la_arvalid & la_arready;
    wire  [5:0]  rd_w    = la_araddr[7:2];
    assign la_arready = ~rvalid_r;
    assign la_rvalid  = rvalid_r;
    assign la_rresp   = 2'b00;
    assign la_rdata   = rdata_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            rvalid_r <= 1'b0; rdata_r <= 32'h0;
        end else if (rd_fire) begin
            rvalid_r <= 1'b1;
            case (rd_w)
                6'h01:   rdata_r <= {30'b0, init_calib_complete, system_break_latched}; // 0x04 STATUS
                6'h04:   rdata_r <= {{(32-IMEM_ADDR_W){1'b0}}, imem_wptr};      // 0x10 IMEM_WPTR
                6'h05:   rdata_r <= free_run;                                   // 0x14 FREE_RUN
                6'h06:   rdata_r <= rst_cnt;                                    // 0x18 RST_CNT
                6'h07:   rdata_r <= {{(32-AXI_ADDR_W){1'b0}}, ddr3_addr_q};     // 0x1C DDR3_ADDR
                6'h08:   rdata_r <= ddr3_rbeat_q[31:0];                         // 0x20 DDR3_R0
                6'h09:   rdata_r <= ddr3_rbeat_q[63:32];                        // 0x24 DDR3_R1
                6'h0A:   rdata_r <= ddr3_rbeat_q[95:64];                        // 0x28 DDR3_R2
                6'h0B:   rdata_r <= ddr3_rbeat_q[127:96];                       // 0x2C DDR3_R3
                6'h0D:   rdata_r <= {30'b0, loader_done, loader_busy};          // 0x34 DDR3_STATUS
                6'h0E:   rdata_r <= {31'b0, load_mode};                         // 0x38 LOAD_MODE
                default: rdata_r <= (rd_w >= 6'h10 && rd_w < 6'h10 + DBG_WORDS)
                                    ? dbg_cap[(rd_w - 6'h10)*32 +: 32] : 32'h0; // 0x40.. DEBUG_Dk
            endcase
        end else if (la_rready) begin
            rvalid_r <= 1'b0;
        end
    end

    // ---- Loader FSM: on GO (write) / RD (read), issue one 128-bit PutFullData / Get on the loader
    // link into the shared tl_to_axi4 bridge (single outstanding, muxed in below). ----
    wire                       ldr_a_ready;   // from the 2:1 bridge-input mux (below)
    wire                       ldr_d_valid;
    logic                      ldr_a_valid;
    logic [2:0]                ldr_a_opcode;
    logic [HBM_ADDR_WIDTH-1:0] ldr_a_address;
    logic [BRIDGE_DATA_W-1:0]  ldr_a_data;
    logic                      ldr_d_ready;
    typedef enum logic [1:0] {LD_IDLE, LD_REQ, LD_RESP} lstate_t;
    lstate_t lstate;
    always_ff @(posedge clk) begin
        if (rst) begin
            lstate <= LD_IDLE; ldr_a_valid <= 1'b0; ldr_a_opcode <= '0;
            ldr_a_address <= '0; ldr_a_data <= '0; ldr_d_ready <= 1'b0;
            loader_busy <= 1'b0; loader_done <= 1'b0; ddr3_rbeat_q <= '0;
        end else begin
            case (lstate)
                LD_IDLE: if (ddr3_go_pulse || ddr3_rd_pulse) begin
                    loader_busy   <= 1'b1; loader_done <= 1'b0;
                    ldr_a_valid   <= 1'b1;
                    ldr_a_opcode  <= ddr3_go_pulse ? 3'd0 : 3'd4;   // PutFullData : Get
                    ldr_a_address <= {{(HBM_ADDR_WIDTH-AXI_ADDR_W){1'b0}}, ddr3_addr_q};
                    ldr_a_data    <= ddr3_wbeat_q;
                    lstate        <= LD_REQ;
                end
                LD_REQ: if (ldr_a_ready) begin ldr_a_valid <= 1'b0; ldr_d_ready <= 1'b1; lstate <= LD_RESP; end
                LD_RESP: if (ldr_d_valid) begin
                    ddr3_rbeat_q <= bridge_d_data;   // meaningful for a Get; ignored for a write
                    ldr_d_ready  <= 1'b0; loader_busy <= 1'b0; loader_done <= 1'b1;
                    lstate       <= LD_IDLE;
                end
                default: lstate <= LD_IDLE;
            endcase
        end
    end

    // =========================================================
    // Datapath: per-channel skid buffers -> round-robin arbiter -> per-channel response FIFOs
    // -> tl_to_axi4 bridge -> fake_ddr3_axi. Verbatim from the validated SimTopDDR, with a 2:1
    // mux at the bridge input so the JTAG loader owns the bridge while LOAD_MODE=1.
    // =========================================================
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_element_r);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_scale_r);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_element_r);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_scale_r);

    tl_regslice #(.SourceWidth(SourceWidth), .SinkWidth(SinkWidth), .AddrWidth(HBM_ADDR_WIDTH), .DataWidth(HBM_ELE_WIDTH),   .RequestMode(7)) rs_m_ele (.clk_i(clk), .rst_ni(!rst), `TL_CONNECT_DEVICE_PORT(host, m_element), `TL_CONNECT_HOST_PORT(device, m_element_r));
    tl_regslice #(.SourceWidth(SourceWidth), .SinkWidth(SinkWidth), .AddrWidth(HBM_ADDR_WIDTH), .DataWidth(HBM_SCALE_WIDTH), .RequestMode(7)) rs_m_sc  (.clk_i(clk), .rst_ni(!rst), `TL_CONNECT_DEVICE_PORT(host, m_scale),   `TL_CONNECT_HOST_PORT(device, m_scale_r));
    tl_regslice #(.SourceWidth(SourceWidth), .SinkWidth(SinkWidth), .AddrWidth(HBM_ADDR_WIDTH), .DataWidth(HBM_ELE_WIDTH),   .RequestMode(7)) rs_v_ele (.clk_i(clk), .rst_ni(!rst), `TL_CONNECT_DEVICE_PORT(host, v_element), `TL_CONNECT_HOST_PORT(device, v_element_r));
    tl_regslice #(.SourceWidth(SourceWidth), .SinkWidth(SinkWidth), .AddrWidth(HBM_ADDR_WIDTH), .DataWidth(HBM_SCALE_WIDTH), .RequestMode(7)) rs_v_sc  (.clk_i(clk), .rst_ni(!rst), `TL_CONNECT_DEVICE_PORT(host, v_scale),   `TL_CONNECT_HOST_PORT(device, v_scale_r));

    localparam logic [1:0] CH_M_ELE = 2'd0, CH_M_SC = 2'd1, CH_V_ELE = 2'd2, CH_V_SC = 2'd3;

    wire bridge_a_ready;
    wire any_a_valid = m_element_r_a_valid | m_scale_r_a_valid | v_element_r_a_valid | v_scale_r_a_valid;
    wire plena_bridge_a_ready = bridge_a_ready & ~load_mode;   // plena mux is gated off during JTAG DDR3 load

    wire [3:0] rr_req = {v_scale_r_a_valid, v_element_r_a_valid, m_scale_r_a_valid, m_element_r_a_valid};
    reg  [1:0] rr_ptr;
    logic [1:0] active_ch;
    always_comb begin
        if      (rr_req[rr_ptr])       active_ch = rr_ptr;
        else if (rr_req[rr_ptr+2'd1])  active_ch = rr_ptr + 2'd1;
        else if (rr_req[rr_ptr+2'd2])  active_ch = rr_ptr + 2'd2;
        else                           active_ch = rr_ptr + 2'd3;
    end
    always_ff @(posedge clk) begin
        if (rst)                                rr_ptr <= 2'd0;
        else if (any_a_valid && plena_bridge_a_ready) rr_ptr <= active_ch + 2'd1;
    end

    logic [2:0]                a_opcode_mux;
    logic [SourceWidth-1:0]    a_source_mux;
    logic [HBM_ADDR_WIDTH-1:0] a_address_mux;
    logic [BRIDGE_DATA_W-1:0]  a_data_mux;
    always_comb begin
        a_opcode_mux = '0; a_source_mux = '0; a_address_mux = '0; a_data_mux = '0;
        unique case (active_ch)
            CH_M_ELE: begin a_opcode_mux=m_element_r_a.opcode; a_source_mux=m_element_r_a.source; a_address_mux=m_element_r_a.address; a_data_mux[HBM_ELE_WIDTH-1:0]=m_element_r_a.data; end
            CH_M_SC:  begin a_opcode_mux=m_scale_r_a.opcode;   a_source_mux=m_scale_r_a.source;   a_address_mux=m_scale_r_a.address;   a_data_mux[HBM_SCALE_WIDTH-1:0]=m_scale_r_a.data; end
            CH_V_ELE: begin a_opcode_mux=v_element_r_a.opcode; a_source_mux=v_element_r_a.source; a_address_mux=v_element_r_a.address; a_data_mux[HBM_ELE_WIDTH-1:0]=v_element_r_a.data; end
            CH_V_SC:  begin a_opcode_mux=v_scale_r_a.opcode;   a_source_mux=v_scale_r_a.source;   a_address_mux=v_scale_r_a.address;   a_data_mux[HBM_SCALE_WIDTH-1:0]=v_scale_r_a.data; end
        endcase
    end

    assign m_element_r_a_ready = (active_ch == CH_M_ELE) && plena_bridge_a_ready;
    assign m_scale_r_a_ready   = (active_ch == CH_M_SC)  && plena_bridge_a_ready;
    assign v_element_r_a_ready = (active_ch == CH_V_ELE) && plena_bridge_a_ready;
    assign v_scale_r_a_ready   = (active_ch == CH_V_SC)  && plena_bridge_a_ready;

    reg [1:0] resp_ch;
    always_ff @(posedge clk) begin
        if (rst) resp_ch <= CH_M_ELE;
        else if (any_a_valid && plena_bridge_a_ready) resp_ch <= active_ch;
    end

    wire                     bridge_d_valid;
    wire [BRIDGE_DATA_W-1:0] bridge_d_data;
    wire                     bridge_d_error;
    wire [SourceWidth-1:0]   bridge_d_source;
    wire plena_bridge_d_valid = bridge_d_valid & ~load_mode;   // FIFOs ignore the loader's response

    localparam int ELE_FW = 1 + SourceWidth + HBM_ELE_WIDTH;
    localparam int SC_FW  = 1 + SourceWidth + HBM_SCALE_WIDTH;

    wire fifo_mele_wr, fifo_msc_wr, fifo_vele_wr, fifo_vsc_wr;
    wire plena_bridge_d_ready = (resp_ch == CH_M_ELE) ? fifo_mele_wr :
                          (resp_ch == CH_M_SC)  ? fifo_msc_wr  :
                          (resp_ch == CH_V_ELE) ? fifo_vele_wr : fifo_vsc_wr;

    wire [ELE_FW-1:0] mele_rd, vele_rd;
    wire [SC_FW-1:0]  msc_rd,  vsc_rd;

    prim_fifo_sync #(.Width(ELE_FW), .Pass(1'b0), .Depth(32)) rf_mele (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(plena_bridge_d_valid && (resp_ch==CH_M_ELE)), .wready_o(fifo_mele_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_ELE_WIDTH-1:0]}),
        .rvalid_o(m_element_r_d_valid), .rready_i(m_element_r_d_ready), .rdata_o(mele_rd),
        .full_o(), .depth_o(), .err_o());
    assign m_element_r_d.opcode=tl_pkg::AccessAckData; assign m_element_r_d.param='0; assign m_element_r_d.sink='0;
    assign m_element_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_ELE_WIDTH/8)); assign m_element_r_d.source=mele_rd[HBM_ELE_WIDTH +: SourceWidth];
    assign m_element_r_d.denied=mele_rd[ELE_FW-1]; assign m_element_r_d.corrupt=mele_rd[ELE_FW-1]; assign m_element_r_d.data=mele_rd[HBM_ELE_WIDTH-1:0];

    prim_fifo_sync #(.Width(SC_FW), .Pass(1'b0), .Depth(32)) rf_msc (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(plena_bridge_d_valid && (resp_ch==CH_M_SC)), .wready_o(fifo_msc_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_SCALE_WIDTH-1:0]}),
        .rvalid_o(m_scale_r_d_valid), .rready_i(m_scale_r_d_ready), .rdata_o(msc_rd),
        .full_o(), .depth_o(), .err_o());
    assign m_scale_r_d.opcode=tl_pkg::AccessAckData; assign m_scale_r_d.param='0; assign m_scale_r_d.sink='0;
    assign m_scale_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_SCALE_WIDTH/8)); assign m_scale_r_d.source=msc_rd[HBM_SCALE_WIDTH +: SourceWidth];
    assign m_scale_r_d.denied=msc_rd[SC_FW-1]; assign m_scale_r_d.corrupt=msc_rd[SC_FW-1]; assign m_scale_r_d.data=msc_rd[HBM_SCALE_WIDTH-1:0];

    prim_fifo_sync #(.Width(ELE_FW), .Pass(1'b0), .Depth(32)) rf_vele (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(plena_bridge_d_valid && (resp_ch==CH_V_ELE)), .wready_o(fifo_vele_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_ELE_WIDTH-1:0]}),
        .rvalid_o(v_element_r_d_valid), .rready_i(v_element_r_d_ready), .rdata_o(vele_rd),
        .full_o(), .depth_o(), .err_o());
    assign v_element_r_d.opcode=tl_pkg::AccessAckData; assign v_element_r_d.param='0; assign v_element_r_d.sink='0;
    assign v_element_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_ELE_WIDTH/8)); assign v_element_r_d.source=vele_rd[HBM_ELE_WIDTH +: SourceWidth];
    assign v_element_r_d.denied=vele_rd[ELE_FW-1]; assign v_element_r_d.corrupt=vele_rd[ELE_FW-1]; assign v_element_r_d.data=vele_rd[HBM_ELE_WIDTH-1:0];

    prim_fifo_sync #(.Width(SC_FW), .Pass(1'b0), .Depth(32)) rf_vsc (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(plena_bridge_d_valid && (resp_ch==CH_V_SC)), .wready_o(fifo_vsc_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_SCALE_WIDTH-1:0]}),
        .rvalid_o(v_scale_r_d_valid), .rready_i(v_scale_r_d_ready), .rdata_o(vsc_rd),
        .full_o(), .depth_o(), .err_o());
    assign v_scale_r_d.opcode=tl_pkg::AccessAckData; assign v_scale_r_d.param='0; assign v_scale_r_d.sink='0;
    assign v_scale_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_SCALE_WIDTH/8)); assign v_scale_r_d.source=vsc_rd[HBM_SCALE_WIDTH +: SourceWidth];
    assign v_scale_r_d.denied=vsc_rd[SC_FW-1]; assign v_scale_r_d.corrupt=vsc_rd[SC_FW-1]; assign v_scale_r_d.data=vsc_rd[HBM_SCALE_WIDTH-1:0];

    // Unused TL-UH channels on the registered (mux-side) links.
    assign m_element_r_b_valid=1'b0; assign m_element_r_b='0; assign m_element_r_c_ready=1'b1; assign m_element_r_e_ready=1'b1;
    assign m_scale_r_b_valid=1'b0;   assign m_scale_r_b='0;   assign m_scale_r_c_ready=1'b1;   assign m_scale_r_e_ready=1'b1;
    assign v_element_r_b_valid=1'b0; assign v_element_r_b='0; assign v_element_r_c_ready=1'b1; assign v_element_r_e_ready=1'b1;
    assign v_scale_r_b_valid=1'b0;   assign v_scale_r_b='0;   assign v_scale_r_c_ready=1'b1;   assign v_scale_r_e_ready=1'b1;

    // ---- 2:1 mux at the bridge input: the JTAG loader owns the shared bridge while load_mode=1
    // (core held in reset, plena mux gated off); otherwise the plena 4-way mux drives it. ----
    wire                      sel_a_valid   = load_mode ? ldr_a_valid   : any_a_valid;
    wire [2:0]                sel_a_opcode  = load_mode ? ldr_a_opcode  : a_opcode_mux;
    wire [SourceWidth-1:0]    sel_a_source  = load_mode ? '0            : a_source_mux;
    wire [HBM_ADDR_WIDTH-1:0] sel_a_address = load_mode ? ldr_a_address : a_address_mux;
    wire [BRIDGE_DATA_W-1:0]  sel_a_data    = load_mode ? ldr_a_data    : a_data_mux;
    assign ldr_a_ready = bridge_a_ready & load_mode;
    assign ldr_d_valid = bridge_d_valid & load_mode;
    wire sel_d_ready   = load_mode ? ldr_d_ready : plena_bridge_d_ready;

    // ---- AXI wires (bridge master <-> fake_ddr3 slave) ----
    wire [AXI_ID_W-1:0]   axi_awid, axi_arid, axi_bid, axi_rid;
    wire [AXI_ADDR_W-1:0] axi_awaddr, axi_araddr;
    wire [7:0]            axi_awlen, axi_arlen;
    wire [2:0]            axi_awsize, axi_arsize;
    wire [1:0]            axi_awburst, axi_arburst, axi_bresp, axi_rresp;
    wire                  axi_awvalid, axi_awready, axi_arvalid, axi_arready;
    wire [AXI_DATA_W-1:0] axi_wdata, axi_rdata;
    wire [AXI_STRB_W-1:0] axi_wstrb;
    wire                  axi_wlast, axi_wvalid, axi_wready, axi_bvalid, axi_bready, axi_rlast, axi_rvalid, axi_rready;

    tl_to_axi4 #(
        .ADDR_W(HBM_ADDR_WIDTH), .DATA_W(BRIDGE_DATA_W),
        .SOURCE_W(SourceWidth), .SINK_W(SinkWidth),
        .AXI_ID_W(AXI_ID_W), .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_bridge (
        .clk(clk), .rst(rst),
        .tl_a_valid(sel_a_valid), .tl_a_ready(bridge_a_ready),
        .tl_a_opcode(sel_a_opcode), .tl_a_size(BRIDGE_TL_SIZE),
        .tl_a_source(sel_a_source), .tl_a_address(sel_a_address),
        .tl_a_mask({BRIDGE_MASK_W{1'b1}}), .tl_a_data(sel_a_data),
        .tl_d_valid(bridge_d_valid), .tl_d_ready(sel_d_ready),
        .tl_d_opcode(), .tl_d_size(), .tl_d_source(bridge_d_source), .tl_d_sink(),
        .tl_d_data(bridge_d_data), .tl_d_error(bridge_d_error),
        .m_axi_awid(axi_awid), .m_axi_awaddr(axi_awaddr), .m_axi_awlen(axi_awlen),
        .m_axi_awsize(axi_awsize), .m_axi_awburst(axi_awburst), .m_axi_awvalid(axi_awvalid), .m_axi_awready(axi_awready),
        .m_axi_wdata(axi_wdata), .m_axi_wstrb(axi_wstrb), .m_axi_wlast(axi_wlast), .m_axi_wvalid(axi_wvalid), .m_axi_wready(axi_wready),
        .m_axi_bid(axi_bid), .m_axi_bresp(axi_bresp), .m_axi_bvalid(axi_bvalid), .m_axi_bready(axi_bready),
        .m_axi_arid(axi_arid), .m_axi_araddr(axi_araddr), .m_axi_arlen(axi_arlen),
        .m_axi_arsize(axi_arsize), .m_axi_arburst(axi_arburst), .m_axi_arvalid(axi_arvalid), .m_axi_arready(axi_arready),
        .m_axi_rid(axi_rid), .m_axi_rdata(axi_rdata), .m_axi_rresp(axi_rresp), .m_axi_rlast(axi_rlast), .m_axi_rvalid(axi_rvalid), .m_axi_rready(axi_rready)
    );

    fake_ddr3_axi #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .AXI_ID_W(AXI_ID_W),
        .MemInitFile(FAKE_HBM_INIT_FILE)
    ) u_ddr3 (
        .clk(clk), .rst(rst),
        .s_axi_awid(axi_awid), .s_axi_awaddr(axi_awaddr), .s_axi_awlen(axi_awlen),
        .s_axi_awsize(axi_awsize), .s_axi_awburst(axi_awburst), .s_axi_awvalid(axi_awvalid), .s_axi_awready(axi_awready),
        .s_axi_wdata(axi_wdata), .s_axi_wstrb(axi_wstrb), .s_axi_wlast(axi_wlast), .s_axi_wvalid(axi_wvalid), .s_axi_wready(axi_wready),
        .s_axi_bid(axi_bid), .s_axi_bresp(axi_bresp), .s_axi_bvalid(axi_bvalid), .s_axi_bready(axi_bready),
        .s_axi_arid(axi_arid), .s_axi_araddr(axi_araddr), .s_axi_arlen(axi_arlen),
        .s_axi_arsize(axi_arsize), .s_axi_arburst(axi_arburst), .s_axi_arvalid(axi_arvalid), .s_axi_arready(axi_arready),
        .s_axi_rid(axi_rid), .s_axi_rdata(axi_rdata), .s_axi_rresp(axi_rresp), .s_axi_rlast(axi_rlast), .s_axi_rvalid(axi_rvalid), .s_axi_rready(axi_rready),
        .init_calib_complete(init_calib_complete)
    );

endmodule
