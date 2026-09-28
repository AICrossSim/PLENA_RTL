`timescale 1ns / 1ps

`include "global_define.vh"
`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"
`include "tl_pkg.svh"
`include "tl_util.svh"

/*
Module      : Plena v80 FPGA Top (gated TileLink / AXI4 HBM interface)
Description : Wraps the plena core and selects the HBM-side interface at elaboration time
              via configuration_pkg::HBM_AXI_BRIDGE_EN (sourced from
              configuration.svh HBM_AXI_BRIDGE_EN):

                HBM_AXI_BRIDGE_EN == 0 (default):
                    The four HBM TileLink host ports (matrix element/scale, vector
                    element/scale) are terminated internally exactly as plena_artix7_top
                    does. The AXI4 master ports below are tied off. Behavior is identical
                    to the raw-TileLink build.

                HBM_AXI_BRIDGE_EN == 1:
                    Matrix (element+scale) and vector (element+scale) TileLink streams are
                    each muxed into one BRIDGE_DATA_W-wide TileLink-UL channel and bridged
                    to an AXI4 master port through a tl_to_axi4 burst bridge
                    (m0_axi = matrix, m1_axi = vector). Each wide TileLink beat is split
                    into BRIDGE_DATA_W/AXI_HBM_DATA_W AXI4 INCR beats.

              Note: a SystemVerilog module's port list cannot be made conditional, so the
              AXI4 master ports are always present and are simply tied off when the bridge
              is disabled.

              The plena core fetches instructions over its TileLink instr_mem_tl host port.
              This wrapper terminates that port internally (always-ready / never-valid),
              mirroring how the HBM ports are terminated when the bridge is disabled — i.e.
              this is a synthesis/elaboration wrapper for the HBM-bridge path. Driving real
              instructions (a RAM-backed instr_mem_tl responder) is a
              follow-up to make the top runnable on hardware.
*/

module plena_v80_top import configuration_pkg::*; import instruction_pkg::*; #(
    parameter integer AXI_HBM_DATA_W = 128,   // AXI4 data width (must divide BRIDGE_DATA_W)
    parameter integer AXI_HBM_ADDR_W = 34,    // AXI4 address width presented to the memory
    parameter integer AXI_HBM_ID_W   = 4
    `ifdef SIMULATION
        , parameter string FP_MEM_INIT_FILE   = ""
        , parameter string INT_MEM_INIT_FILE  = ""
        , parameter string V_SRAM_RESULT_FILE = ""
        , parameter string FP_REG_RESULT_FILE = ""
    `endif
)(
    input  logic clk,
    input  logic rst,
    output logic system_break,

    // ---------------------------------------------------------------------
    // AXI4 master port 0 — Matrix channels (driven only when HBM_AXI_BRIDGE_EN==1)
    // ---------------------------------------------------------------------
    output logic [AXI_HBM_ID_W-1:0]      m0_axi_awid,
    output logic [AXI_HBM_ADDR_W-1:0]    m0_axi_awaddr,
    output logic [7:0]                   m0_axi_awlen,
    output logic [2:0]                   m0_axi_awsize,
    output logic [1:0]                   m0_axi_awburst,
    output logic                         m0_axi_awvalid,
    input  logic                         m0_axi_awready,
    output logic [AXI_HBM_DATA_W-1:0]    m0_axi_wdata,
    output logic [AXI_HBM_DATA_W/8-1:0]  m0_axi_wstrb,
    output logic                         m0_axi_wlast,
    output logic                         m0_axi_wvalid,
    input  logic                         m0_axi_wready,
    input  logic [AXI_HBM_ID_W-1:0]      m0_axi_bid,
    input  logic [1:0]                   m0_axi_bresp,
    input  logic                         m0_axi_bvalid,
    output logic                         m0_axi_bready,
    output logic [AXI_HBM_ID_W-1:0]      m0_axi_arid,
    output logic [AXI_HBM_ADDR_W-1:0]    m0_axi_araddr,
    output logic [7:0]                   m0_axi_arlen,
    output logic [2:0]                   m0_axi_arsize,
    output logic [1:0]                   m0_axi_arburst,
    output logic                         m0_axi_arvalid,
    input  logic                         m0_axi_arready,
    input  logic [AXI_HBM_ID_W-1:0]      m0_axi_rid,
    input  logic [AXI_HBM_DATA_W-1:0]    m0_axi_rdata,
    input  logic [1:0]                   m0_axi_rresp,
    input  logic                         m0_axi_rlast,
    input  logic                         m0_axi_rvalid,
    output logic                         m0_axi_rready,

    // ---------------------------------------------------------------------
    // AXI4 master port 1 — Vector channels (driven only when HBM_AXI_BRIDGE_EN==1)
    // ---------------------------------------------------------------------
    output logic [AXI_HBM_ID_W-1:0]      m1_axi_awid,
    output logic [AXI_HBM_ADDR_W-1:0]    m1_axi_awaddr,
    output logic [7:0]                   m1_axi_awlen,
    output logic [2:0]                   m1_axi_awsize,
    output logic [1:0]                   m1_axi_awburst,
    output logic                         m1_axi_awvalid,
    input  logic                         m1_axi_awready,
    output logic [AXI_HBM_DATA_W-1:0]    m1_axi_wdata,
    output logic [AXI_HBM_DATA_W/8-1:0]  m1_axi_wstrb,
    output logic                         m1_axi_wlast,
    output logic                         m1_axi_wvalid,
    input  logic                         m1_axi_wready,
    input  logic [AXI_HBM_ID_W-1:0]      m1_axi_bid,
    input  logic [1:0]                   m1_axi_bresp,
    input  logic                         m1_axi_bvalid,
    output logic                         m1_axi_bready,
    output logic [AXI_HBM_ID_W-1:0]      m1_axi_arid,
    output logic [AXI_HBM_ADDR_W-1:0]    m1_axi_araddr,
    output logic [7:0]                   m1_axi_arlen,
    output logic [2:0]                   m1_axi_arsize,
    output logic [1:0]                   m1_axi_arburst,
    output logic                         m1_axi_arvalid,
    input  logic                         m1_axi_arready,
    input  logic [AXI_HBM_ID_W-1:0]      m1_axi_rid,
    input  logic [AXI_HBM_DATA_W-1:0]    m1_axi_rdata,
    input  logic [1:0]                   m1_axi_rresp,
    input  logic                         m1_axi_rlast,
    input  logic                         m1_axi_rvalid,
    output logic                         m1_axi_rready
);

    // Bridge stream width = full HBM element width; the (narrower) scale channel is
    // zero-extended into the low bits of the same stream.
    localparam int BRIDGE_DATA_W  = HBM_WIDTH;            // = HBM_ELE_WIDTH (power of two)
    localparam int BRIDGE_MASK_W  = BRIDGE_DATA_W / 8;

    // Internal TileLink host connections (declared with the core's port widths)
    `TL_DECLARE(INSTRUCTION_LENGTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, instr_mem);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_scale);
    `TL_DECLARE(HBM_ELE_WIDTH,   HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_element);
    `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_scale);

    // Instantiate the core plena module
    (* DONT_TOUCH = "TRUE" *)
    plena #(
        `ifdef SIMULATION
            .FP_MEM_INIT_FILE(FP_MEM_INIT_FILE),
            .INT_MEM_INIT_FILE(INT_MEM_INIT_FILE),
            .V_SRAM_RESULT_FILE(V_SRAM_RESULT_FILE),
            .FP_REG_RESULT_FILE(FP_REG_RESULT_FILE)
        `endif
    ) plena_core (
        .clk(clk),
        .rst(rst),
        .system_break(system_break),
        `TL_CONNECT_HOST_PORT(instr_mem_tl,  instr_mem),
        `TL_CONNECT_HOST_PORT(m_out_element, m_element),
        `TL_CONNECT_HOST_PORT(m_out_scale,   m_scale),
        `TL_CONNECT_HOST_PORT(v_out_element, v_element),
        `TL_CONNECT_HOST_PORT(v_out_scale,   v_scale),
        // VSRAM debug readback unused in this wrapper
        .debug_vsram_en (1'b0),
        .debug_row      ('0),
        .debug_vsram_data ()
    );

    // Instruction TileLink port is terminated internally (mode-independent): the core may
    // issue fetches but receives no response in this synthesis/elaboration wrapper.
    assign instr_mem_a_ready = 1'b1;
    assign instr_mem_b_valid = 1'b0; assign instr_mem_b = '0;
    assign instr_mem_c_ready = 1'b1;
    assign instr_mem_d_valid = 1'b0; assign instr_mem_d = '0;
    assign instr_mem_e_ready = 1'b1;

    generate
    if (HBM_AXI_BRIDGE_EN) begin : g_bridge
        // =================================================================
        // Matrix TL mux (m_element + m_scale → bridge 0)
        // =================================================================
        logic                      m_tl_a_valid_mux;
        logic                      m_tl_a_is_scale;
        logic [2:0]                m_tl_a_opcode_mux;
        logic [`TL_SIZE_WIDTH-1:0] m_tl_a_size_mux;
        logic [SourceWidth-1:0]    m_tl_a_source_mux;
        logic [HBM_ADDR_WIDTH-1:0] m_tl_a_address_mux;
        logic [BRIDGE_MASK_W-1:0]  m_tl_a_mask_mux;
        logic [BRIDGE_DATA_W-1:0]  m_tl_a_data_mux;

        always_comb begin
            m_tl_a_valid_mux   = 1'b0;
            m_tl_a_is_scale    = 1'b0;
            m_tl_a_opcode_mux  = '0;
            m_tl_a_size_mux    = '0;
            m_tl_a_source_mux  = '0;
            m_tl_a_address_mux = '0;
            m_tl_a_mask_mux    = '0;
            m_tl_a_data_mux    = '0;
            if (m_element_a_valid) begin
                m_tl_a_valid_mux   = 1'b1;
                m_tl_a_is_scale    = 1'b0;
                m_tl_a_opcode_mux  = m_element_a.opcode;
                m_tl_a_size_mux    = m_element_a.size;
                m_tl_a_source_mux  = m_element_a.source;
                m_tl_a_address_mux = m_element_a.address;
                m_tl_a_mask_mux[HBM_ELE_WIDTH/8-1:0] = m_element_a.mask;
                m_tl_a_data_mux[HBM_ELE_WIDTH-1:0]   = m_element_a.data;
            end else if (m_scale_a_valid) begin
                m_tl_a_valid_mux   = 1'b1;
                m_tl_a_is_scale    = 1'b1;
                m_tl_a_opcode_mux  = m_scale_a.opcode;
                m_tl_a_size_mux    = m_scale_a.size;
                m_tl_a_source_mux  = m_scale_a.source;
                m_tl_a_address_mux = m_scale_a.address;
                m_tl_a_mask_mux[HBM_SCALE_WIDTH/8-1:0] = m_scale_a.mask;
                m_tl_a_data_mux[HBM_SCALE_WIDTH-1:0]   = m_scale_a.data;
            end
        end

        wire m_tl_a_ready_bridge;
        assign m_element_a_ready = m_tl_a_ready_bridge;
        assign m_scale_a_ready   = m_tl_a_ready_bridge && !m_element_a_valid;

        // Response routing: remember whether the in-flight request was element or scale.
        logic m_resp_is_scale;
        always_ff @(posedge clk) begin
            if (rst)
                m_resp_is_scale <= 1'b0;
            else if (m_tl_a_valid_mux && m_tl_a_ready_bridge)
                m_resp_is_scale <= m_tl_a_is_scale;
        end

        wire                      m_tl_d_valid_bridge;
        wire [2:0]                m_tl_d_opcode_bridge;
        wire [`TL_SIZE_WIDTH-1:0] m_tl_d_size_bridge;
        wire [SourceWidth-1:0]    m_tl_d_source_bridge;
        wire [SinkWidth-1:0]      m_tl_d_sink_bridge;
        wire [BRIDGE_DATA_W-1:0]  m_tl_d_data_bridge;
        wire                      m_tl_d_error_bridge;
        wire                      m_tl_d_ready_bridge;

        always_comb begin
            m_element_d_valid = 1'b0;
            m_scale_d_valid   = 1'b0;
            m_element_d       = '0;
            m_scale_d         = '0;
            m_element_d.opcode = (m_tl_d_opcode_bridge == 3'd1) ? tl_pkg::AccessAckData : tl_pkg::AccessAck;
            m_scale_d.opcode   = (m_tl_d_opcode_bridge == 3'd1) ? tl_pkg::AccessAckData : tl_pkg::AccessAck;
            m_element_d.size    = m_tl_d_size_bridge;
            m_scale_d.size      = m_tl_d_size_bridge;
            m_element_d.source  = m_tl_d_source_bridge;
            m_scale_d.source    = m_tl_d_source_bridge;
            m_element_d.sink    = m_tl_d_sink_bridge;
            m_scale_d.sink      = m_tl_d_sink_bridge;
            m_element_d.denied  = m_tl_d_error_bridge;
            m_scale_d.denied    = m_tl_d_error_bridge;
            m_element_d.corrupt = m_tl_d_error_bridge;
            m_scale_d.corrupt   = m_tl_d_error_bridge;
            m_element_d.data    = m_tl_d_data_bridge[HBM_ELE_WIDTH-1:0];
            m_scale_d.data      = m_tl_d_data_bridge[HBM_SCALE_WIDTH-1:0];
            if (m_resp_is_scale)
                m_scale_d_valid = m_tl_d_valid_bridge;
            else
                m_element_d_valid = m_tl_d_valid_bridge;
        end

        assign m_tl_d_ready_bridge = m_resp_is_scale ? m_scale_d_ready : m_element_d_ready;

        assign m_element_b_valid = 1'b0; assign m_element_b = '0;
        assign m_element_c_ready = 1'b1; assign m_element_e_ready = 1'b1;
        assign m_scale_b_valid   = 1'b0; assign m_scale_b = '0;
        assign m_scale_c_ready   = 1'b1; assign m_scale_e_ready = 1'b1;

        // =================================================================
        // Vector TL mux (v_element + v_scale → bridge 1)
        // =================================================================
        logic                      v_tl_a_valid_mux;
        logic                      v_tl_a_is_scale;
        logic [2:0]                v_tl_a_opcode_mux;
        logic [`TL_SIZE_WIDTH-1:0] v_tl_a_size_mux;
        logic [SourceWidth-1:0]    v_tl_a_source_mux;
        logic [HBM_ADDR_WIDTH-1:0] v_tl_a_address_mux;
        logic [BRIDGE_MASK_W-1:0]  v_tl_a_mask_mux;
        logic [BRIDGE_DATA_W-1:0]  v_tl_a_data_mux;

        always_comb begin
            v_tl_a_valid_mux   = 1'b0;
            v_tl_a_is_scale    = 1'b0;
            v_tl_a_opcode_mux  = '0;
            v_tl_a_size_mux    = '0;
            v_tl_a_source_mux  = '0;
            v_tl_a_address_mux = '0;
            v_tl_a_mask_mux    = '0;
            v_tl_a_data_mux    = '0;
            if (v_element_a_valid) begin
                v_tl_a_valid_mux   = 1'b1;
                v_tl_a_is_scale    = 1'b0;
                v_tl_a_opcode_mux  = v_element_a.opcode;
                v_tl_a_size_mux    = v_element_a.size;
                v_tl_a_source_mux  = v_element_a.source;
                v_tl_a_address_mux = v_element_a.address;
                v_tl_a_mask_mux[HBM_ELE_WIDTH/8-1:0] = v_element_a.mask;
                v_tl_a_data_mux[HBM_ELE_WIDTH-1:0]   = v_element_a.data;
            end else if (v_scale_a_valid) begin
                v_tl_a_valid_mux   = 1'b1;
                v_tl_a_is_scale    = 1'b1;
                v_tl_a_opcode_mux  = v_scale_a.opcode;
                v_tl_a_size_mux    = v_scale_a.size;
                v_tl_a_source_mux  = v_scale_a.source;
                v_tl_a_address_mux = v_scale_a.address;
                v_tl_a_mask_mux[HBM_SCALE_WIDTH/8-1:0] = v_scale_a.mask;
                v_tl_a_data_mux[HBM_SCALE_WIDTH-1:0]   = v_scale_a.data;
            end
        end

        wire v_tl_a_ready_bridge;
        assign v_element_a_ready = v_tl_a_ready_bridge;
        assign v_scale_a_ready   = v_tl_a_ready_bridge && !v_element_a_valid;

        logic v_resp_is_scale;
        always_ff @(posedge clk) begin
            if (rst)
                v_resp_is_scale <= 1'b0;
            else if (v_tl_a_valid_mux && v_tl_a_ready_bridge)
                v_resp_is_scale <= v_tl_a_is_scale;
        end

        wire                      v_tl_d_valid_bridge;
        wire [2:0]                v_tl_d_opcode_bridge;
        wire [`TL_SIZE_WIDTH-1:0] v_tl_d_size_bridge;
        wire [SourceWidth-1:0]    v_tl_d_source_bridge;
        wire [SinkWidth-1:0]      v_tl_d_sink_bridge;
        wire [BRIDGE_DATA_W-1:0]  v_tl_d_data_bridge;
        wire                      v_tl_d_error_bridge;
        wire                      v_tl_d_ready_bridge;

        always_comb begin
            v_element_d_valid = 1'b0;
            v_scale_d_valid   = 1'b0;
            v_element_d       = '0;
            v_scale_d         = '0;
            v_element_d.opcode = (v_tl_d_opcode_bridge == 3'd1) ? tl_pkg::AccessAckData : tl_pkg::AccessAck;
            v_scale_d.opcode   = (v_tl_d_opcode_bridge == 3'd1) ? tl_pkg::AccessAckData : tl_pkg::AccessAck;
            v_element_d.size    = v_tl_d_size_bridge;
            v_scale_d.size      = v_tl_d_size_bridge;
            v_element_d.source  = v_tl_d_source_bridge;
            v_scale_d.source    = v_tl_d_source_bridge;
            v_element_d.sink    = v_tl_d_sink_bridge;
            v_scale_d.sink      = v_tl_d_sink_bridge;
            v_element_d.denied  = v_tl_d_error_bridge;
            v_scale_d.denied    = v_tl_d_error_bridge;
            v_element_d.corrupt = v_tl_d_error_bridge;
            v_scale_d.corrupt   = v_tl_d_error_bridge;
            v_element_d.data    = v_tl_d_data_bridge[HBM_ELE_WIDTH-1:0];
            v_scale_d.data      = v_tl_d_data_bridge[HBM_SCALE_WIDTH-1:0];
            if (v_resp_is_scale)
                v_scale_d_valid = v_tl_d_valid_bridge;
            else
                v_element_d_valid = v_tl_d_valid_bridge;
        end

        assign v_tl_d_ready_bridge = v_resp_is_scale ? v_scale_d_ready : v_element_d_ready;

        assign v_element_b_valid = 1'b0; assign v_element_b = '0;
        assign v_element_c_ready = 1'b1; assign v_element_e_ready = 1'b1;
        assign v_scale_b_valid   = 1'b0; assign v_scale_b = '0;
        assign v_scale_c_ready   = 1'b1; assign v_scale_e_ready = 1'b1;

        // =================================================================
        // TL→AXI4 Bridge 0: Matrix channels → AXI master port 0
        // =================================================================
        tl_to_axi4 #(
            .ADDR_W(HBM_ADDR_WIDTH), .DATA_W(BRIDGE_DATA_W),
            .SOURCE_W(SourceWidth), .SINK_W(SinkWidth),
            .AXI_ID_W(AXI_HBM_ID_W), .AXI_ADDR_W(AXI_HBM_ADDR_W),
            .AXI_DATA_W(AXI_HBM_DATA_W)
        ) u_bridge_matrix (
            .clk(clk), .rst(rst),
            .tl_a_valid(m_tl_a_valid_mux), .tl_a_ready(m_tl_a_ready_bridge),
            .tl_a_opcode(m_tl_a_opcode_mux), .tl_a_size(m_tl_a_size_mux),
            .tl_a_source(m_tl_a_source_mux), .tl_a_address(m_tl_a_address_mux),
            .tl_a_mask(m_tl_a_mask_mux), .tl_a_data(m_tl_a_data_mux),
            .tl_d_valid(m_tl_d_valid_bridge), .tl_d_ready(m_tl_d_ready_bridge),
            .tl_d_opcode(m_tl_d_opcode_bridge), .tl_d_size(m_tl_d_size_bridge),
            .tl_d_source(m_tl_d_source_bridge), .tl_d_sink(m_tl_d_sink_bridge),
            .tl_d_data(m_tl_d_data_bridge), .tl_d_error(m_tl_d_error_bridge),
            .m_axi_awid(m0_axi_awid), .m_axi_awaddr(m0_axi_awaddr),
            .m_axi_awlen(m0_axi_awlen), .m_axi_awsize(m0_axi_awsize),
            .m_axi_awburst(m0_axi_awburst), .m_axi_awvalid(m0_axi_awvalid),
            .m_axi_awready(m0_axi_awready),
            .m_axi_wdata(m0_axi_wdata), .m_axi_wstrb(m0_axi_wstrb),
            .m_axi_wlast(m0_axi_wlast), .m_axi_wvalid(m0_axi_wvalid),
            .m_axi_wready(m0_axi_wready),
            .m_axi_bid(m0_axi_bid), .m_axi_bresp(m0_axi_bresp),
            .m_axi_bvalid(m0_axi_bvalid), .m_axi_bready(m0_axi_bready),
            .m_axi_arid(m0_axi_arid), .m_axi_araddr(m0_axi_araddr),
            .m_axi_arlen(m0_axi_arlen), .m_axi_arsize(m0_axi_arsize),
            .m_axi_arburst(m0_axi_arburst), .m_axi_arvalid(m0_axi_arvalid),
            .m_axi_arready(m0_axi_arready),
            .m_axi_rid(m0_axi_rid), .m_axi_rdata(m0_axi_rdata),
            .m_axi_rresp(m0_axi_rresp), .m_axi_rlast(m0_axi_rlast),
            .m_axi_rvalid(m0_axi_rvalid), .m_axi_rready(m0_axi_rready)
        );

        // =================================================================
        // TL→AXI4 Bridge 1: Vector channels → AXI master port 1
        // =================================================================
        tl_to_axi4 #(
            .ADDR_W(HBM_ADDR_WIDTH), .DATA_W(BRIDGE_DATA_W),
            .SOURCE_W(SourceWidth), .SINK_W(SinkWidth),
            .AXI_ID_W(AXI_HBM_ID_W), .AXI_ADDR_W(AXI_HBM_ADDR_W),
            .AXI_DATA_W(AXI_HBM_DATA_W)
        ) u_bridge_vector (
            .clk(clk), .rst(rst),
            .tl_a_valid(v_tl_a_valid_mux), .tl_a_ready(v_tl_a_ready_bridge),
            .tl_a_opcode(v_tl_a_opcode_mux), .tl_a_size(v_tl_a_size_mux),
            .tl_a_source(v_tl_a_source_mux), .tl_a_address(v_tl_a_address_mux),
            .tl_a_mask(v_tl_a_mask_mux), .tl_a_data(v_tl_a_data_mux),
            .tl_d_valid(v_tl_d_valid_bridge), .tl_d_ready(v_tl_d_ready_bridge),
            .tl_d_opcode(v_tl_d_opcode_bridge), .tl_d_size(v_tl_d_size_bridge),
            .tl_d_source(v_tl_d_source_bridge), .tl_d_sink(v_tl_d_sink_bridge),
            .tl_d_data(v_tl_d_data_bridge), .tl_d_error(v_tl_d_error_bridge),
            .m_axi_awid(m1_axi_awid), .m_axi_awaddr(m1_axi_awaddr),
            .m_axi_awlen(m1_axi_awlen), .m_axi_awsize(m1_axi_awsize),
            .m_axi_awburst(m1_axi_awburst), .m_axi_awvalid(m1_axi_awvalid),
            .m_axi_awready(m1_axi_awready),
            .m_axi_wdata(m1_axi_wdata), .m_axi_wstrb(m1_axi_wstrb),
            .m_axi_wlast(m1_axi_wlast), .m_axi_wvalid(m1_axi_wvalid),
            .m_axi_wready(m1_axi_wready),
            .m_axi_bid(m1_axi_bid), .m_axi_bresp(m1_axi_bresp),
            .m_axi_bvalid(m1_axi_bvalid), .m_axi_bready(m1_axi_bready),
            .m_axi_arid(m1_axi_arid), .m_axi_araddr(m1_axi_araddr),
            .m_axi_arlen(m1_axi_arlen), .m_axi_arsize(m1_axi_arsize),
            .m_axi_arburst(m1_axi_arburst), .m_axi_arvalid(m1_axi_arvalid),
            .m_axi_arready(m1_axi_arready),
            .m_axi_rid(m1_axi_rid), .m_axi_rdata(m1_axi_rdata),
            .m_axi_rresp(m1_axi_rresp), .m_axi_rlast(m1_axi_rlast),
            .m_axi_rvalid(m1_axi_rvalid), .m_axi_rready(m1_axi_rready)
        );

    end else begin : g_term
        // =================================================================
        // Raw-TileLink build: terminate all four HBM TileLink host ports
        // ("always ready / never valid"), identical to plena_artix7_top.
        // =================================================================
        assign m_element_a_ready = 1'b1;
        assign m_element_b_valid = 1'b0; assign m_element_b = '0;
        assign m_element_c_ready = 1'b1;
        assign m_element_d_valid = 1'b0; assign m_element_d = '0;
        assign m_element_e_ready = 1'b1;

        assign m_scale_a_ready = 1'b1;
        assign m_scale_b_valid = 1'b0; assign m_scale_b = '0;
        assign m_scale_c_ready = 1'b1;
        assign m_scale_d_valid = 1'b0; assign m_scale_d = '0;
        assign m_scale_e_ready = 1'b1;

        assign v_element_a_ready = 1'b1;
        assign v_element_b_valid = 1'b0; assign v_element_b = '0;
        assign v_element_c_ready = 1'b1;
        assign v_element_d_valid = 1'b0; assign v_element_d = '0;
        assign v_element_e_ready = 1'b1;

        assign v_scale_a_ready = 1'b1;
        assign v_scale_b_valid = 1'b0; assign v_scale_b = '0;
        assign v_scale_c_ready = 1'b1;
        assign v_scale_d_valid = 1'b0; assign v_scale_d = '0;
        assign v_scale_e_ready = 1'b1;

        // AXI master ports unused in this mode — tie off all outputs.
        assign m0_axi_awid = '0; assign m0_axi_awaddr = '0; assign m0_axi_awlen = '0;
        assign m0_axi_awsize = '0; assign m0_axi_awburst = '0; assign m0_axi_awvalid = 1'b0;
        assign m0_axi_wdata = '0; assign m0_axi_wstrb = '0; assign m0_axi_wlast = 1'b0;
        assign m0_axi_wvalid = 1'b0; assign m0_axi_bready = 1'b0;
        assign m0_axi_arid = '0; assign m0_axi_araddr = '0; assign m0_axi_arlen = '0;
        assign m0_axi_arsize = '0; assign m0_axi_arburst = '0; assign m0_axi_arvalid = 1'b0;
        assign m0_axi_rready = 1'b0;

        assign m1_axi_awid = '0; assign m1_axi_awaddr = '0; assign m1_axi_awlen = '0;
        assign m1_axi_awsize = '0; assign m1_axi_awburst = '0; assign m1_axi_awvalid = 1'b0;
        assign m1_axi_wdata = '0; assign m1_axi_wstrb = '0; assign m1_axi_wlast = 1'b0;
        assign m1_axi_wvalid = 1'b0; assign m1_axi_bready = 1'b0;
        assign m1_axi_arid = '0; assign m1_axi_araddr = '0; assign m1_axi_arlen = '0;
        assign m1_axi_arsize = '0; assign m1_axi_arburst = '0; assign m1_axi_arvalid = 1'b0;
        assign m1_axi_rready = 1'b0;
    end
    endgenerate

endmodule
