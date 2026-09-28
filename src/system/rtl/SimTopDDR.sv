`timescale 1ns / 1ps
`include "tl_util.svh"
`include "tl_pkg.svh"
`include "configuration.svh"
`include "prim_util_pkg.svh"

// Simulation top that drives the PLENA core's HBM traffic through the ACTUAL FPGA
// datapath (4-way priority mux -> tl_to_axi4 bridge -> fake_ddr3_axi) instead of the
// direct fake_hbm. Mirrors the Nexys plena_a7_top HBM path so the serialized,
// single-controller, matrix>vector-priority prefetch behaviour is exercised in sim.
// Instructions are still served from a single-port fake_hbm (the FPGA loads IMEM
// separately); only the 4 data channels go through DDR3. VRAM dump plumbing is
// identical to SimTop so vector_result.mem can be diffed against the same golden.
module SimTopDDR import instruction_pkg::*; import configuration_pkg::*; #(
    parameter           INSTRUCTION_LENGTH            = 32,
    parameter string    FAKE_HBM_INIT_FILE            = "",
    parameter string    FP_MEM_INIT_FILE              = "",
    parameter string    INT_MEM_INIT_FILE             = "",
    parameter string    VECTOR_MEM_RESULT_FILE        = "",
    parameter string    FP_REG_RESULT_FILE            = "",
    parameter string    HBM_RESULT_FILE               = ""
) (
    input logic clk,
    input logic rst
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

    `TL_DECLARE(INSTRUCTION_LENGTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, instr_link);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_scale);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_scale);

    plena #(
        .FP_MEM_INIT_FILE(FP_MEM_INIT_FILE),
        .INT_MEM_INIT_FILE(INT_MEM_INIT_FILE),
        .V_SRAM_RESULT_FILE(VECTOR_MEM_RESULT_FILE),
        .FP_REG_RESULT_FILE(FP_REG_RESULT_FILE)
    ) dut (
        .clk(clk),
        .rst(rst),
        `TL_CONNECT_HOST_PORT(instr_mem_tl,  instr_link),
        `TL_CONNECT_HOST_PORT(m_out_element, m_element),
        `TL_CONNECT_HOST_PORT(m_out_scale,   m_scale),
        `TL_CONNECT_HOST_PORT(v_out_element, v_element),
        `TL_CONNECT_HOST_PORT(v_out_scale,   v_scale)
    );

    // ---- Instructions: served by the proven fake_hbm_5port (correctly parses the 256-bit-row
    // hbm.mem, fast BRAM-like fetch matching the FPGA IMEM). Its 4 HBM DATA device ports are
    // idle-tied because plena's data ports go through the DDR3 path instead. ----
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, dmy_m_ele);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, dmy_m_sc);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, dmy_v_ele);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, dmy_v_sc);
    // idle-tie the host side of each dummy data link (no requests issued)
    assign dmy_m_ele_a_valid=1'b0; assign dmy_m_ele_a='0; assign dmy_m_ele_b_ready=1'b1; assign dmy_m_ele_c_valid=1'b0; assign dmy_m_ele_c='0; assign dmy_m_ele_d_ready=1'b1; assign dmy_m_ele_e_valid=1'b0; assign dmy_m_ele_e='0;
    assign dmy_m_sc_a_valid=1'b0;  assign dmy_m_sc_a='0;  assign dmy_m_sc_b_ready=1'b1;  assign dmy_m_sc_c_valid=1'b0;  assign dmy_m_sc_c='0;  assign dmy_m_sc_d_ready=1'b1;  assign dmy_m_sc_e_valid=1'b0;  assign dmy_m_sc_e='0;
    assign dmy_v_ele_a_valid=1'b0; assign dmy_v_ele_a='0; assign dmy_v_ele_b_ready=1'b1; assign dmy_v_ele_c_valid=1'b0; assign dmy_v_ele_c='0; assign dmy_v_ele_d_ready=1'b1; assign dmy_v_ele_e_valid=1'b0; assign dmy_v_ele_e='0;
    assign dmy_v_sc_a_valid=1'b0;  assign dmy_v_sc_a='0;  assign dmy_v_sc_b_ready=1'b1;  assign dmy_v_sc_c_valid=1'b0;  assign dmy_v_sc_c='0;  assign dmy_v_sc_d_ready=1'b1;  assign dmy_v_sc_e_valid=1'b0;  assign dmy_v_sc_e='0;

    fake_hbm_5port #(
        .ADDR_WIDTH       (HBM_ADDR_WIDTH),
        .ELE_DATA_WIDTH   (HBM_ELE_WIDTH),
        .SCALE_DATA_WIDTH (HBM_SCALE_WIDTH),
        .INSTR_DATA_WIDTH (INSTRUCTION_LENGTH),
        .BRAM_ADDR_WIDTH  (FAKE_HBM_ADDR_WIDTH),
        .SourceWidth      (SourceWidth),
        .SinkWidth        (SinkWidth),
        .MemInitFile      (FAKE_HBM_INIT_FILE)
    ) instr_hbm (
        .clk(clk), .rst(rst),
        `TL_CONNECT_DEVICE_PORT(m_element, dmy_m_ele),
        `TL_CONNECT_DEVICE_PORT(m_scale,   dmy_m_sc),
        `TL_CONNECT_DEVICE_PORT(v_element, dmy_v_ele),
        `TL_CONNECT_DEVICE_PORT(v_scale,   dmy_v_sc),
        `TL_CONNECT_DEVICE_PORT(instr,     instr_link)
    );

    // =========================================================
    // Per-channel skid buffers (tl_regslice) then a 4-way priority mux -> tl_to_axi4 -> fake_ddr3.
    // The skid buffer grants a_ready when its 2-deep buffer has room, INDEPENDENT of the shared-
    // bridge arbitration. plena gates a_valid on seeing a_ready (fake_hbm's tl_adapter_bram asserts
    // a_ready unconditionally when idle); the hand-rolled mux only granted a_ready to an already-
    // valid channel -> deadlock. This is a real bug in the untested plena_a7_top mux, caught here.
    // Priority m_element > m_scale > v_element > v_scale.
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


    // Round-robin arbiter across the 4 HBM channels. Strict priority (m_ele>m_sc>v_ele>v_sc,
    // as in the untested plena_a7_top mux) DEADLOCKS: each controller's element and scale reads
    // feed a join2 that needs them paired, but strict priority drains a whole element burst
    // (matrix LOAD_AMOUNT=MLEN=16) before any scale, overflowing the response FIFO while scale
    // is starved. Round-robin interleaves element/scale so the join always makes progress with
    // bounded buffering. bit0=m_ele bit1=m_sc bit2=v_ele bit3=v_sc.
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
        else if (any_a_valid && bridge_a_ready) rr_ptr <= active_ch + 2'd1;
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

    assign m_element_r_a_ready = (active_ch == CH_M_ELE) && bridge_a_ready;
    assign m_scale_r_a_ready   = (active_ch == CH_M_SC)  && bridge_a_ready;
    assign v_element_r_a_ready = (active_ch == CH_V_ELE) && bridge_a_ready;
    assign v_scale_r_a_ready   = (active_ch == CH_V_SC)  && bridge_a_ready;

    reg [1:0] resp_ch;
    always_ff @(posedge clk) begin
        if (rst) resp_ch <= CH_M_ELE;
        else if (any_a_valid && bridge_a_ready) resp_ch <= active_ch;
    end

    wire                     bridge_d_valid;
    wire [BRIDGE_DATA_W-1:0] bridge_d_data;
    wire                     bridge_d_error;
    wire [SourceWidth-1:0]   bridge_d_source;   // echoed request source (single outstanding)

    // Per-channel response FIFOs (Pass=0 => wready = !full, decoupled from the read side).
    // Essential: the single-outstanding bridge otherwise stalls in TL_RESP holding one channel's
    // response until plena consumes it, but plena's tl_master holds that response pending its join
    // partner (element<->scale) whose read is queued behind the stalled bridge -> deadlock. With a
    // FIFO the bridge always drains (never back-pressured), so it can service the partner read.
    // Depth 8 > LOAD_AMOUNT(4) outstanding per channel. This is exactly the buffering the untested
    // plena_a7_top mux is missing; the real single-MIG FPGA datapath needs it too.
    localparam int ELE_FW = 1 + SourceWidth + HBM_ELE_WIDTH;    // {denied, source, data}
    localparam int SC_FW  = 1 + SourceWidth + HBM_SCALE_WIDTH;

    wire fifo_mele_wr, fifo_msc_wr, fifo_vele_wr, fifo_vsc_wr;
    wire bridge_d_ready = (resp_ch == CH_M_ELE) ? fifo_mele_wr :
                          (resp_ch == CH_M_SC)  ? fifo_msc_wr  :
                          (resp_ch == CH_V_ELE) ? fifo_vele_wr : fifo_vsc_wr;

    wire [ELE_FW-1:0] mele_rd, vele_rd;
    wire [SC_FW-1:0]  msc_rd,  vsc_rd;

    prim_fifo_sync #(.Width(ELE_FW), .Pass(1'b0), .Depth(32)) rf_mele (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(bridge_d_valid && (resp_ch==CH_M_ELE)), .wready_o(fifo_mele_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_ELE_WIDTH-1:0]}),
        .rvalid_o(m_element_r_d_valid), .rready_i(m_element_r_d_ready), .rdata_o(mele_rd),
        .full_o(), .depth_o(), .err_o());
    assign m_element_r_d.opcode=tl_pkg::AccessAckData; assign m_element_r_d.param='0; assign m_element_r_d.sink='0;
    assign m_element_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_ELE_WIDTH/8)); assign m_element_r_d.source=mele_rd[HBM_ELE_WIDTH +: SourceWidth];
    assign m_element_r_d.denied=mele_rd[ELE_FW-1]; assign m_element_r_d.corrupt=mele_rd[ELE_FW-1]; assign m_element_r_d.data=mele_rd[HBM_ELE_WIDTH-1:0];

    prim_fifo_sync #(.Width(SC_FW), .Pass(1'b0), .Depth(32)) rf_msc (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(bridge_d_valid && (resp_ch==CH_M_SC)), .wready_o(fifo_msc_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_SCALE_WIDTH-1:0]}),
        .rvalid_o(m_scale_r_d_valid), .rready_i(m_scale_r_d_ready), .rdata_o(msc_rd),
        .full_o(), .depth_o(), .err_o());
    assign m_scale_r_d.opcode=tl_pkg::AccessAckData; assign m_scale_r_d.param='0; assign m_scale_r_d.sink='0;
    assign m_scale_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_SCALE_WIDTH/8)); assign m_scale_r_d.source=msc_rd[HBM_SCALE_WIDTH +: SourceWidth];
    assign m_scale_r_d.denied=msc_rd[SC_FW-1]; assign m_scale_r_d.corrupt=msc_rd[SC_FW-1]; assign m_scale_r_d.data=msc_rd[HBM_SCALE_WIDTH-1:0];

    prim_fifo_sync #(.Width(ELE_FW), .Pass(1'b0), .Depth(32)) rf_vele (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(bridge_d_valid && (resp_ch==CH_V_ELE)), .wready_o(fifo_vele_wr),
        .wdata_i({bridge_d_error, bridge_d_source, bridge_d_data[HBM_ELE_WIDTH-1:0]}),
        .rvalid_o(v_element_r_d_valid), .rready_i(v_element_r_d_ready), .rdata_o(vele_rd),
        .full_o(), .depth_o(), .err_o());
    assign v_element_r_d.opcode=tl_pkg::AccessAckData; assign v_element_r_d.param='0; assign v_element_r_d.sink='0;
    assign v_element_r_d.size=`TL_SIZE_WIDTH'($clog2(HBM_ELE_WIDTH/8)); assign v_element_r_d.source=vele_rd[HBM_ELE_WIDTH +: SourceWidth];
    assign v_element_r_d.denied=vele_rd[ELE_FW-1]; assign v_element_r_d.corrupt=vele_rd[ELE_FW-1]; assign v_element_r_d.data=vele_rd[HBM_ELE_WIDTH-1:0];

    prim_fifo_sync #(.Width(SC_FW), .Pass(1'b0), .Depth(32)) rf_vsc (
        .clk_i(clk), .rst_ni(!rst), .clr_i(1'b0),
        .wvalid_i(bridge_d_valid && (resp_ch==CH_V_SC)), .wready_o(fifo_vsc_wr),
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
        .tl_a_valid(any_a_valid), .tl_a_ready(bridge_a_ready),
        .tl_a_opcode(a_opcode_mux), .tl_a_size(BRIDGE_TL_SIZE),
        .tl_a_source(a_source_mux), .tl_a_address(a_address_mux),
        .tl_a_mask({BRIDGE_MASK_W{1'b1}}), .tl_a_data(a_data_mux),
        .tl_d_valid(bridge_d_valid), .tl_d_ready(bridge_d_ready),
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
        .init_calib_complete()
    );

endmodule
