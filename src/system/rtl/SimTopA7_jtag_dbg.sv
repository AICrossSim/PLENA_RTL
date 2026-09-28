`timescale 1ns / 1ps
// Full JTAG-to-activation simulation: SimTopA7 (the PLENA core + AXI-lite control block + fixed
// HBM datapath + fake_ddr3) driven through the custom jtag_axi_bscan BSCAN->AXI-lite master
// instead of the raw AXI-lite port. The JTAG tap ports are exposed so a cocotb testbench (via
// JtagBscanTransport + the unchanged PlenaDriver) can activate the core exactly the way the real
// board will over the FTDI JTAG cable -- no Vivado, no Xilinx jtag_axi IP. The AXI-lite master
// runs on the core clock (aclk=clk); tck is the separate JTAG clock, so the tck<->clk CDC is real.
module SimTopA7_jtag_dbg #(
    parameter           INSTRUCTION_LENGTH            = 32,
    parameter int       IMEM_DEPTH                    = 512,
    parameter string    FAKE_HBM_INIT_FILE            = "",
    parameter string    FP_MEM_INIT_FILE              = "",
    parameter string    INT_MEM_INIT_FILE             = "",
    parameter string    VECTOR_MEM_RESULT_FILE        = "",
    parameter string    FP_REG_RESULT_FILE            = ""
) (
    input  logic clk,
    input  logic rst,
    // JTAG tap (driven by the cocotb JtagBscanTransport in sim; a real BSCANE2 on the board)
    input  logic tck,
    input  logic tap_reset,
    input  logic sel,
    input  logic capture,
    input  logic shift,
    input  logic update,
    input  logic tdi,
    output logic tdo
);
    // AXI-lite master (jtag_axi_bscan) <-> AXI-lite slave (SimTopA7.la_*)
    wire [31:0] la_awaddr, la_wdata, la_araddr, la_rdata;
    wire [3:0]  la_wstrb;
    wire [1:0]  la_bresp, la_rresp;
    wire        la_awvalid, la_awready, la_wvalid, la_wready, la_bvalid, la_bready;
    wire        la_arvalid, la_arready, la_rvalid, la_rready;

    jtag_axi_bscan #(.ADDR_W(7), .DATA_W(32)) u_jtag (
        .tck(tck), .tap_reset(tap_reset), .sel(sel), .capture(capture),
        .shift(shift), .update(update), .tdi(tdi), .tdo(tdo),
        .aclk(clk), .aresetn(~rst),
        .m_awaddr(la_awaddr), .m_awvalid(la_awvalid), .m_awready(la_awready),
        .m_wdata(la_wdata), .m_wstrb(la_wstrb), .m_wvalid(la_wvalid), .m_wready(la_wready),
        .m_bresp(la_bresp), .m_bvalid(la_bvalid), .m_bready(la_bready),
        .m_araddr(la_araddr), .m_arvalid(la_arvalid), .m_arready(la_arready),
        .m_rdata(la_rdata), .m_rresp(la_rresp), .m_rvalid(la_rvalid), .m_rready(la_rready)
    );

    SimTopA7_dbg #(
        .INSTRUCTION_LENGTH(INSTRUCTION_LENGTH), .IMEM_DEPTH(IMEM_DEPTH),
        .FAKE_HBM_INIT_FILE(FAKE_HBM_INIT_FILE), .FP_MEM_INIT_FILE(FP_MEM_INIT_FILE),
        .INT_MEM_INIT_FILE(INT_MEM_INIT_FILE), .VECTOR_MEM_RESULT_FILE(VECTOR_MEM_RESULT_FILE),
        .FP_REG_RESULT_FILE(FP_REG_RESULT_FILE)
    ) u_a7 (
        .clk(clk), .rst(rst),
        .la_awaddr(la_awaddr), .la_awvalid(la_awvalid), .la_awready(la_awready),
        .la_wdata(la_wdata), .la_wstrb(la_wstrb), .la_wvalid(la_wvalid), .la_wready(la_wready),
        .la_bresp(la_bresp), .la_bvalid(la_bvalid), .la_bready(la_bready),
        .la_araddr(la_araddr), .la_arvalid(la_arvalid), .la_arready(la_arready),
        .la_rdata(la_rdata), .la_rresp(la_rresp), .la_rvalid(la_rvalid), .la_rready(la_rready)
    );
endmodule
