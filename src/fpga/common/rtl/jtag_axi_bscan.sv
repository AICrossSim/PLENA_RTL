`timescale 1ns / 1ps
// Vivado-free JTAG -> AXI4-Lite master. Turns a Xilinx BSCANE2 USER-register DR scan into an
// AXI-lite transaction so a host (pyftdi/OpenOCD over the FT2232H) can drive the PLENA AXI-lite
// control block WITHOUT the Xilinx jtag_axi IP / hw_server. Replaces jtag_axi in plena_a7_top.
//
// The JTAG tap interface is exposed as PLAIN MODULE PORTS (tck/tdi/tdo/sel/capture/shift/update/
// tap_reset) so a cocotb testbench can drive the scan sequence directly; the real BSCANE2 is
// instantiated only in the FPGA top wrapper (Verilator cannot elaborate the unisim BSCANE2).
//
// DR format (40 bits, LSB-first on the wire):
//   shift-IN  : [31:0]=wdata (ignored on reads), [ADDR_W+31:32]=byte addr, [39]=rw (1=wr,0=rd)
//   shift-OUT : [31:0]=rdata (prev), [ADDR_W+31:32]=addr echo (prev), [39]=done (prev)
// Pipelined/zero-poll: a scan launches txn N (on Update-DR) and shifts out the result of txn N-1.
// A read issues a read scan then a second (dummy) scan to clock out {done, rdata}.
//
// The tap shifter is clocked on TCK (DRCK is dead during Update-DR); the AXI master runs on aclk.
// The two domains are joined by a toggle request/ack handshake (2-flop synchronizers) with a
// quasi-static payload (stable for ms at USB-JTAG speeds), so no per-bit synchronization is needed.
module jtag_axi_bscan #(
    parameter int ADDR_W = 7,          // byte-address bits carried in the DR (0x00..0x7F)
    parameter int DATA_W = 32
)(
    // ---- JTAG tap side (from BSCANE2, or a cocotb driver in sim) ----
    input  logic              tck,
    input  logic              tap_reset,     // BSCANE2 RESET (sync clear of the tap FSM)
    input  logic              sel,           // BSCANE2 SEL (this USER instruction is active)
    input  logic              capture,       // Capture-DR
    input  logic              shift,         // Shift-DR
    input  logic              update,        // Update-DR
    input  logic              tdi,
    output logic              tdo,

    // ---- AXI4-Lite master (drives the target register block; aclk/aresetn = core domain) ----
    input  logic              aclk,
    input  logic              aresetn,
    output logic [31:0]       m_awaddr,
    output logic              m_awvalid,
    input  logic              m_awready,
    output logic [31:0]       m_wdata,
    output logic [3:0]        m_wstrb,
    output logic              m_wvalid,
    input  logic              m_wready,
    input  logic [1:0]        m_bresp,
    input  logic              m_bvalid,
    output logic              m_bready,
    output logic [31:0]       m_araddr,
    output logic              m_arvalid,
    input  logic              m_arready,
    input  logic [31:0]       m_rdata,
    input  logic [1:0]        m_rresp,
    input  logic              m_rvalid,
    output logic              m_rready
);
    localparam int DR_LEN = 1 + ADDR_W + DATA_W;

    // aclk-domain signals read across into the tck domain (quasi-static / toggle-synced).
    logic              axi_ack_tog;
    logic [DR_LEN-1:0] resp_next;

    // ================= TCK domain: DR shift register + command latch + response readout =========
    logic [DR_LEN-1:0] sh_reg;
    logic [DR_LEN-1:0] cmd_reg;
    logic [DR_LEN-1:0] resp_holding;
    logic              jtag_req_tog;
    logic [1:0]        ack_sync_tck;

    assign tdo = sh_reg[0];

    always_ff @(posedge tck) begin
        if (tap_reset) begin
            sh_reg <= '0; cmd_reg <= '0; jtag_req_tog <= 1'b0;
            resp_holding <= '0; ack_sync_tck <= 2'b00;
        end else begin
            // Response readout CDC: pull resp_next across on the AXI ack toggle.
            ack_sync_tck <= {ack_sync_tck[0], axi_ack_tog};
            if (ack_sync_tck[1] ^ ack_sync_tck[0]) resp_holding <= resp_next;
            // DR scan (all tap signals must be qualified by SEL: they assert for every USER chain).
            if (sel) begin
                if (capture)      sh_reg <= resp_holding;
                else if (shift)   sh_reg <= {tdi, sh_reg[DR_LEN-1:1]};
                if (update) begin cmd_reg <= sh_reg; jtag_req_tog <= ~jtag_req_tog; end
            end
        end
    end

    // ================= ACLK domain: request sync + AXI-lite master FSM ===========================
    logic [1:0]        req_sync;
    logic [ADDR_W-1:0] cmd_addr_q;
    typedef enum logic [2:0] {A_IDLE, A_W, A_B, A_AR, A_R, A_DONE} astate_t;
    astate_t astate;

    wire req_edge = req_sync[1] ^ req_sync[0];

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            req_sync <= 2'b00; axi_ack_tog <= 1'b0; resp_next <= '0; cmd_addr_q <= '0;
            m_awaddr <= '0; m_awvalid <= 1'b0; m_wdata <= '0; m_wstrb <= 4'h0; m_wvalid <= 1'b0; m_bready <= 1'b0;
            m_araddr <= '0; m_arvalid <= 1'b0; m_rready <= 1'b0;
            astate <= A_IDLE;
        end else begin
            req_sync <= {req_sync[0], jtag_req_tog};
            case (astate)
                A_IDLE: if (req_edge) begin
                    cmd_addr_q <= cmd_reg[DATA_W +: ADDR_W];
                    if (cmd_reg[DR_LEN-1]) begin  // write
                        m_awaddr  <= {{(32-ADDR_W){1'b0}}, cmd_reg[DATA_W +: ADDR_W]};
                        m_awvalid <= 1'b1;
                        m_wdata   <= cmd_reg[DATA_W-1:0];
                        m_wstrb   <= 4'hF; m_wvalid <= 1'b1; m_bready <= 1'b1;
                        astate    <= A_W;
                    end else begin                // read
                        m_araddr  <= {{(32-ADDR_W){1'b0}}, cmd_reg[DATA_W +: ADDR_W]};
                        m_arvalid <= 1'b1; m_rready <= 1'b1;
                        astate    <= A_AR;
                    end
                end
                A_W:  if (m_awready && m_wready) begin m_awvalid <= 1'b0; m_wvalid <= 1'b0; astate <= A_B; end
                A_B:  if (m_bvalid) begin
                    m_bready    <= 1'b0;
                    resp_next   <= {1'b1, cmd_addr_q, {DATA_W{1'b0}}};
                    axi_ack_tog <= ~axi_ack_tog; astate <= A_DONE;
                end
                A_AR: if (m_arready) begin m_arvalid <= 1'b0; astate <= A_R; end
                A_R:  if (m_rvalid) begin
                    m_rready    <= 1'b0;
                    resp_next   <= {1'b1, cmd_addr_q, m_rdata};
                    axi_ack_tog <= ~axi_ack_tog; astate <= A_DONE;
                end
                A_DONE: astate <= A_IDLE;
                default: astate <= A_IDLE;
            endcase
        end
    end
endmodule
