`timescale 1ns / 1ps
// Test harness for jtag_axi_bscan: the BSCAN->AXI-lite master driving a tiny AXI4-Lite register
// file whose handshake mirrors the SimTopA7 control block (awready=wready=wr_fire combinational;
// arready=~rvalid_r). A cocotb testbench drives the JTAG tap ports and checks write/read round-trips
// -- this validates the DR bit-order, the tck<->aclk CDC handshake, the pipeline/flush, and the
// AXI-lite handshake, all without a real BSCANE2.
module jtag_bscan_harness #(
    parameter int ADDR_W = 7
)(
    input  logic tck,
    input  logic tap_reset,
    input  logic sel,
    input  logic capture,
    input  logic shift,
    input  logic update,
    input  logic tdi,
    output logic tdo,
    input  logic aclk,
    input  logic aresetn
);
    wire [31:0] m_awaddr, m_wdata, m_araddr, m_rdata;
    wire [3:0]  m_wstrb;
    wire [1:0]  m_bresp, m_rresp;
    wire        m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
    wire        m_arvalid, m_arready, m_rvalid, m_rready;

    jtag_axi_bscan #(.ADDR_W(ADDR_W), .DATA_W(32)) u_master (
        .tck(tck), .tap_reset(tap_reset), .sel(sel), .capture(capture),
        .shift(shift), .update(update), .tdi(tdi), .tdo(tdo),
        .aclk(aclk), .aresetn(aresetn),
        .m_awaddr(m_awaddr), .m_awvalid(m_awvalid), .m_awready(m_awready),
        .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wvalid(m_wvalid), .m_wready(m_wready),
        .m_bresp(m_bresp), .m_bvalid(m_bvalid), .m_bready(m_bready),
        .m_araddr(m_araddr), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rvalid(m_rvalid), .m_rready(m_rready)
    );

    // ---- Tiny AXI-lite register file (32 words), SimTopA7-style single-outstanding handshake ----
    logic [31:0] regmem [0:31];
    logic        bvalid_r, rvalid_r;
    logic [31:0] rdata_r;

    wire wr_fire = m_awvalid & m_wvalid & ~bvalid_r;
    wire rd_fire = m_arvalid & (~rvalid_r);
    assign m_awready = wr_fire;
    assign m_wready  = wr_fire;
    assign m_bvalid  = bvalid_r;
    assign m_bresp   = 2'b00;
    assign m_arready = ~rvalid_r;
    assign m_rvalid  = rvalid_r;
    assign m_rdata   = rdata_r;
    assign m_rresp   = 2'b00;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            bvalid_r <= 1'b0; rvalid_r <= 1'b0; rdata_r <= 32'h0;
        end else begin
            if (wr_fire)        begin regmem[m_awaddr[6:2]] <= m_wdata; bvalid_r <= 1'b1; end
            else if (m_bready)  bvalid_r <= 1'b0;

            if (rd_fire)        begin rdata_r <= regmem[m_araddr[6:2]]; rvalid_r <= 1'b1; end
            else if (m_rready)  rvalid_r <= 1'b0;
        end
    end
endmodule
