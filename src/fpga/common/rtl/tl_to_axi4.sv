// TileLink-UL to AXI4 Bridge (Burst Mode)
// Converts TileLink Uncached Lightweight (Get/Put) to AXI4 burst read/write.
// 512-bit TileLink data is split into 4 x 128-bit AXI4 beats (INCR burst).
// Designed to connect PLENA's TileLink host ports to Xilinx MIG DDR2 on Nexys A7.

module tl_to_axi4 #(
    parameter ADDR_W   = 128,
    parameter DATA_W   = 512,
    parameter SOURCE_W = 6,
    parameter SINK_W   = 1,
    parameter AXI_ID_W = 4,
    parameter AXI_ADDR_W = 27,    // 128MB DDR2 address space
    parameter AXI_DATA_W = 128,   // MIG DDR2 native width
    parameter AXI_STRB_W = AXI_DATA_W / 8
) (
    input  wire clk,
    input  wire rst,

    // === TileLink Host Port (from PLENA) ===
    // Channel A (request from host)
    input  wire                tl_a_valid,
    output reg                 tl_a_ready,
    input  wire [2:0]          tl_a_opcode,   // 4=Get, 0=PutFullData
    input  wire [3:0]          tl_a_size,
    input  wire [SOURCE_W-1:0] tl_a_source,
    input  wire [ADDR_W-1:0]   tl_a_address,
    input  wire [DATA_W/8-1:0] tl_a_mask,
    input  wire [DATA_W-1:0]   tl_a_data,

    // Channel D (response to host)
    output reg                 tl_d_valid,
    input  wire                tl_d_ready,
    output reg  [2:0]          tl_d_opcode,   // 1=AccessAckData, 0=AccessAck
    output reg  [3:0]          tl_d_size,
    output reg  [SOURCE_W-1:0] tl_d_source,
    output reg  [SINK_W-1:0]   tl_d_sink,
    output reg  [DATA_W-1:0]   tl_d_data,
    output reg                 tl_d_error,

    // === AXI4 Master Port (to MIG DDR) ===
    // Write Address Channel
    output reg  [AXI_ID_W-1:0]   m_axi_awid,
    output reg  [AXI_ADDR_W-1:0] m_axi_awaddr,
    output reg  [7:0]            m_axi_awlen,
    output reg  [2:0]            m_axi_awsize,
    output reg  [1:0]            m_axi_awburst,
    output reg                   m_axi_awvalid,
    input  wire                  m_axi_awready,

    // Write Data Channel
    output reg  [AXI_DATA_W-1:0] m_axi_wdata,
    output reg  [AXI_STRB_W-1:0] m_axi_wstrb,
    output reg                   m_axi_wlast,
    output reg                   m_axi_wvalid,
    input  wire                  m_axi_wready,

    // Write Response Channel
    input  wire [AXI_ID_W-1:0]   m_axi_bid,
    input  wire [1:0]            m_axi_bresp,
    input  wire                  m_axi_bvalid,
    output reg                   m_axi_bready,

    // Read Address Channel
    output reg  [AXI_ID_W-1:0]   m_axi_arid,
    output reg  [AXI_ADDR_W-1:0] m_axi_araddr,
    output reg  [7:0]            m_axi_arlen,
    output reg  [2:0]            m_axi_arsize,
    output reg  [1:0]            m_axi_arburst,
    output reg                   m_axi_arvalid,
    input  wire                  m_axi_arready,

    // Read Data Channel
    input  wire [AXI_ID_W-1:0]   m_axi_rid,
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output reg                   m_axi_rready
);

    // TileLink opcodes
    localparam TL_GET          = 3'd4;
    localparam TL_PUT_FULL     = 3'd0;
    localparam TL_ACK          = 3'd0;
    localparam TL_ACK_DATA     = 3'd1;

    // Burst parameters. Two regimes:
    //  - WIDE  (DATA_W >= AXI_DATA_W): a wide TL beat splits into BURST_LEN AXI beats (INCR).
    //  - NARROW(DATA_W <  AXI_DATA_W): one TL beat is smaller than an AXI word (e.g. MLEN=8 ->
    //    64b TL into the 128b MIG). Handled as ONE full-width AXI beat with WSTRB masking the
    //    inactive lanes; the address is aligned to the AXI word and the active DATA_W chunk is
    //    steered to the sub-lane selected by the low TL address bits.
    localparam NARROW      = (DATA_W < AXI_DATA_W);
    localparam BURST_LEN   = NARROW ? 1 : (DATA_W / AXI_DATA_W);
    // Clamp to >=1 so the single-beat case (BURST_LEN==1) is legal: $clog2(1)==0 would make
    // beat_cnt zero-width and {BEAT_CNT_W{..}} an illegal 0-replication.
    localparam BEAT_CNT_W  = (BURST_LEN > 1) ? $clog2(BURST_LEN) : 1;
    localparam AXI_SIZE    = $clog2(AXI_DATA_W/8); // full AXI beat size, log2 bytes (16B -> 4)
    // Internal beat buffers are padded to at least one AXI word so all AXI-width slices are in
    // range in both regimes. NBEATS = AXI beats the buffer holds (WIDE: BURST_LEN; NARROW: 1).
    localparam BUF_W       = NARROW ? AXI_DATA_W : DATA_W;
    localparam NBEATS      = BUF_W / AXI_DATA_W;
    localparam SUBLANES    = NARROW ? (AXI_DATA_W / DATA_W) : 1;    // DATA_W chunks per AXI word
    localparam LANE_W      = (SUBLANES > 1) ? $clog2(SUBLANES) : 1;
    localparam LANE_LSB    = $clog2(DATA_W/8);     // TL addr bit where the sub-lane index starts
    localparam AXI_ALIGN   = $clog2(AXI_DATA_W/8); // AXI-word address alignment (low bits cleared)

    // Sub-lane this request targets within the AXI word (always 0 in WIDE mode), and the
    // AXI-word-aligned address.
    wire [LANE_W-1:0]     req_lane = NARROW ? tl_a_address[LANE_LSB +: LANE_W] : {LANE_W{1'b0}};
    wire [AXI_ADDR_W-1:0] axi_addr_aligned = NARROW
        ? {tl_a_address[AXI_ADDR_W-1:AXI_ALIGN], {AXI_ALIGN{1'b0}}}
        : tl_a_address[AXI_ADDR_W-1:0];

    // FSM states
    localparam IDLE       = 3'd0;
    localparam RD_ADDR    = 3'd1;
    localparam RD_DATA    = 3'd2;
    localparam WR_ADDR    = 3'd3;
    localparam WR_DATA    = 3'd4;
    localparam WR_RESP    = 3'd5;
    localparam TL_RESP    = 3'd6;

    reg [2:0] state;
    reg [SOURCE_W-1:0] saved_source;
    reg [3:0] saved_size;
    reg [BUF_W-1:0] saved_rdata;
    reg saved_is_read;
    reg saved_error;
    reg [LANE_W-1:0] saved_lane;   // NARROW: which DATA_W chunk of the AXI word (else 0)

    // Beat counter for burst transactions
    reg [BEAT_CNT_W-1:0] beat_cnt;

    // Saved write data and mask (padded to BUF_W so AXI-width slices are always in range)
    reg [BUF_W-1:0]   saved_wdata;
    reg [BUF_W/8-1:0] saved_wmask;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tl_a_ready <= 1'b1;
            tl_d_valid <= 1'b0;
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_bready <= 1'b0;
            m_axi_rready <= 1'b0;
            m_axi_wlast <= 1'b0;
            beat_cnt <= {BEAT_CNT_W{1'b0}};
            saved_error <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    tl_d_valid <= 1'b0;
                    if (tl_a_valid && tl_a_ready) begin
                        saved_source <= tl_a_source;
                        saved_size <= tl_a_size;
                        tl_a_ready <= 1'b0;
                        beat_cnt <= {BEAT_CNT_W{1'b0}};
                        saved_error <= 1'b0;
                        saved_lane <= req_lane;

                        if (tl_a_opcode == TL_GET) begin
                            // Read request: aligned AXI word address, full-width beats.
                            m_axi_arid <= tl_a_source[AXI_ID_W-1:0];
                            m_axi_araddr <= axi_addr_aligned;
                            m_axi_arlen <= BURST_LEN - 1;  // WIDE: 3=4 beats; NARROW: 0=1 beat
                            m_axi_arsize <= AXI_SIZE[2:0]; // 4 = 16 bytes (full AXI word)
                            m_axi_arburst <= 2'b01;        // INCR
                            m_axi_arvalid <= 1'b1;
                            saved_is_read <= 1'b1;
                            saved_rdata <= {BUF_W{1'b0}};
                            state <= RD_ADDR;
                        end else begin
                            // Write request: aligned AXI word address, full-width beats.
                            m_axi_awid <= tl_a_source[AXI_ID_W-1:0];
                            m_axi_awaddr <= axi_addr_aligned;
                            m_axi_awlen <= BURST_LEN - 1;
                            m_axi_awsize <= AXI_SIZE[2:0];
                            m_axi_awburst <= 2'b01;        // INCR
                            m_axi_awvalid <= 1'b1;
                            saved_is_read <= 1'b0;
                            // Steer the DATA_W chunk (+ its mask) to the addressed sub-lane; in
                            // WIDE mode req_lane=0 so this is the plain full-width capture.
                            saved_wdata <= (BUF_W'(tl_a_data)  << (req_lane * DATA_W));
                            saved_wmask <= ((BUF_W/8)'(tl_a_mask) << (req_lane * (DATA_W/8)));
                            state <= WR_ADDR;
                        end
                    end
                end

                // ============ READ PATH ============

                RD_ADDR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        state <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    // Collect burst beats into saved_rdata
                    if (m_axi_rvalid && m_axi_rready) begin
                        // Place beat data at correct position in DATA_W-bit register
                        saved_rdata[beat_cnt*AXI_DATA_W +: AXI_DATA_W] <= m_axi_rdata;

                        // Track errors across all beats
                        if (m_axi_rresp != 2'b00)
                            saved_error <= 1'b1;

                        if (m_axi_rlast || beat_cnt == BURST_LEN - 1) begin
                            // All beats received
                            m_axi_rready <= 1'b0;
                            beat_cnt <= {BEAT_CNT_W{1'b0}};
                            state <= TL_RESP;
                        end else begin
                            beat_cnt <= beat_cnt + 1'b1;
                        end
                    end
                end

                // ============ WRITE PATH ============

                WR_ADDR: begin
                    // Wait for AW acceptance, then present first W beat
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        // Present beat 0 on W channel
                        m_axi_wdata <= saved_wdata[1*AXI_DATA_W-1 -: AXI_DATA_W];
                        m_axi_wstrb <= saved_wmask[1*AXI_STRB_W-1 -: AXI_STRB_W];
                        m_axi_wlast <= (BURST_LEN == 1) ? 1'b1 : 1'b0;
                        m_axi_wvalid <= 1'b1;
                        state <= WR_DATA;
                    end
                end

                WR_DATA: begin
                    // Send write beats; beat_cnt tracks which beat is currently presented
                    if (m_axi_wready && m_axi_wvalid) begin
                        if (beat_cnt == BURST_LEN - 1) begin
                            // Last beat accepted
                            m_axi_wvalid <= 1'b0;
                            m_axi_wlast <= 1'b0;
                            m_axi_bready <= 1'b1;
                            beat_cnt <= {BEAT_CNT_W{1'b0}};
                            state <= WR_RESP;
                        end else begin
                            beat_cnt <= beat_cnt + 1'b1;
                            // Load next beat data. Mask the look-ahead index to the buffer's beat
                            // count (NBEATS, power of two) so the part-select is always in range;
                            // this branch only runs in WIDE mode (BURST_LEN>1) but must elaborate
                            // safely for NARROW (NBEATS==1 -> index forced to 0).
                            m_axi_wdata <= saved_wdata[((beat_cnt+1) & (NBEATS-1))*AXI_DATA_W +: AXI_DATA_W];
                            m_axi_wstrb <= saved_wmask[((beat_cnt+1) & (NBEATS-1))*AXI_STRB_W +: AXI_STRB_W];
                            m_axi_wlast <= (beat_cnt + 1'b1 == BURST_LEN - 1) ? 1'b1 : 1'b0;
                        end
                    end
                end

                WR_RESP: begin
                    if (m_axi_bvalid) begin
                        saved_error <= (m_axi_bresp != 2'b00);
                        m_axi_bready <= 1'b0;
                        state <= TL_RESP;
                    end
                end

                // ============ TILELINK RESPONSE ============

                TL_RESP: begin
                    // Send TileLink D response
                    tl_d_valid <= 1'b1;
                    tl_d_opcode <= saved_is_read ? TL_ACK_DATA : TL_ACK;
                    tl_d_size <= saved_size;
                    tl_d_source <= saved_source;
                    tl_d_sink <= {SINK_W{1'b0}};
                    // NARROW: return the addressed DATA_W chunk of the AXI word; WIDE: the whole
                    // buffer (saved_lane==0, DATA_W==BUF_W -> low-DATA_W = entire saved_rdata).
                    tl_d_data <= saved_is_read ? saved_rdata[saved_lane*DATA_W +: DATA_W] : {DATA_W{1'b0}};
                    tl_d_error <= saved_error;

                    // Complete only once tl_d_valid is actually asserted. Gating on
                    // tl_d_ready alone loses the beat when the consumer holds tl_d_ready
                    // high (ready-before-valid): tl_d_valid would be set and cleared in
                    // the same cycle and never observed.
                    if (tl_d_valid && tl_d_ready) begin
                        tl_d_valid <= 1'b0;
                        tl_a_ready <= 1'b1;
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
