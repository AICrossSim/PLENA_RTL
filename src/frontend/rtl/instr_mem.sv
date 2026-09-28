`timescale 1ns / 1ps
`include "tl_util.svh"

/*
Module      : Instruction Memory with TileLink Interface
Description :
  - Simple instruction buffer holding INST_BUFF_DEPTH instructions
  - PC comes from decoder (byte-addressed)
  - When PC reaches end of buffer, automatically loads next INST_BUFF_DEPTH instructions
  - No cache, no ready signal - just direct PC-indexed access
*/

module instr_mem import tl_pkg::*; #(
    parameter int INSTRUCTION_WIDTH = 32,
    parameter int PC_WIDTH = 32,
    parameter int BUFFER_DEPTH = 16,
    parameter int HBM_ADDR_WIDTH = 32,
    parameter int SourceWidth = 4,
    parameter int SinkWidth = 1,
    parameter int InstrStorageOffset = 0  // Byte offset in HBM for instruction storage
)(
    input  logic clk,
    input  logic rst,

    // PC from decoder (byte-addressed)
    input  logic [PC_WIDTH-1:0] pc,

    // Instruction output
    output logic [INSTRUCTION_WIDTH-1:0] instruction_out,
    output logic [PC_WIDTH-1:0]          instruction_addr,   // Fetch address instruction_out was served for
    output logic                         instruction_ready,  // High when instruction_out is valid for current PC

    // TileLink Interface
    `TL_DECLARE_HOST_PORT(INSTRUCTION_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, instr_tl)
);

    // =========================================================================
    // Instruction Buffer
    // =========================================================================
    logic [INSTRUCTION_WIDTH-1:0] instr_buffer [BUFFER_DEPTH-1:0];
    logic [HBM_ADDR_WIDTH-1:0]    buffer_base_addr;
    logic                         buffer_valid;
    logic [$clog2(BUFFER_DEPTH)-1:0] buffer_index;
    logic                         pc_in_range;

    // Calculate buffer index from PC
    assign buffer_index = (pc - buffer_base_addr) >> 2;  // Divide by 4 (byte to word)
    assign pc_in_range  = (pc >= buffer_base_addr) &&
                          (pc < buffer_base_addr + (BUFFER_DEPTH << 2));

    // Registered outputs. instruction_addr is registered from the same-cycle pc
    // that selected buffer_index, so (addr, data, ready) always describe the
    // same fetch - the consumer can reject stale words around refills/rewinds.
    always_ff @(posedge clk) begin
        if (rst) begin
            instruction_out   <= '0;
            instruction_addr  <= '0;
            instruction_ready <= 1'b0;
        end else begin
            instruction_out   <= instr_buffer[buffer_index];
            instruction_addr  <= pc;
            instruction_ready <= buffer_valid && pc_in_range;
        end
    end

    // =========================================================================
    // TileLink Master for Loading Instructions
    // =========================================================================
    `TL_DECLARE(INSTRUCTION_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, tl_instr);

    logic                         fetch_req;
    logic [HBM_ADDR_WIDTH-1:0]    fetch_addr;
    logic [INSTRUCTION_WIDTH-1:0] fetch_data;
    logic                         fetch_data_ready;
    logic                         fetch_data_valid;
    logic                         fetch_complete;

    tl_master #(
        .DataWidth      (INSTRUCTION_WIDTH),
        .AddrWidth      (HBM_ADDR_WIDTH),
        .SourceWidth    (SourceWidth),
        .SinkWidth      (SinkWidth),
        .LOAD_AMOUNT    (BUFFER_DEPTH),
        .WRITE_AMOUNT   (1),
        .ONCHIP_ADDR    (PC_WIDTH)
    ) tl_master_inst (
        .clk                (clk),
        .rst                (rst),
        .stride_mode        (1'b0),
        .stride_offset      ('0),
        .req_en             (fetch_req),
        .addr               (fetch_addr),
        .fetch_data         (fetch_data),
        .fetch_data_ready   (fetch_data_ready),
        .fetch_data_valid   (fetch_data_valid),
        .complete_fetch     (fetch_complete),
        .write_en           (1'b0),
        .write_mask         ('0),
        .write_data         ('0),
        .ready_to_write     (),
        `TL_CONNECT_HOST_PORT(host, tl_instr)
    );

    `TL_BIND_HOST_PORT(instr_tl, tl_instr);

    // =========================================================================
    // Buffer Fill Controller
    // =========================================================================
    typedef enum logic [1:0] {
        IDLE,
        FETCH_REQUEST,
        FETCH_WAIT,
        FILL_BUFFER
    } state_t;

    state_t state, next_state;
    logic [$clog2(BUFFER_DEPTH)-1:0] fill_counter;

    // State register
    always_ff @(posedge clk) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // A burst's LAST data beat is delivered after tl_master's fetch_complete
    // pulse, so it lingers and (without the FETCH_WAIT gate below) used to be
    // written as buffer[0] of the NEXT refill - shifting every buffer's
    // content one word against its address. Seamless refills hid this as a
    // self-consistent global relabel, but non-seamless refill sequences
    // (odd base after a stall, decoder pull-back, back-to-back refills)
    // changed the shift locally and silently skipped/duplicated instructions.
    // Count exactly BUFFER_DEPTH beats inside FETCH_WAIT and drain any
    // out-of-window beat without writing it.
    logic last_fill_beat;
    assign last_fill_beat = (state == FETCH_WAIT) && fetch_data_valid && fetch_data_ready
                            && (fill_counter == BUFFER_DEPTH - 1);

    // Fill counter for writing fetched instructions to buffer
    always_ff @(posedge clk) begin
        if (rst) begin
            fill_counter <= '0;
        end else if (state == FETCH_REQUEST) begin
            fill_counter <= '0;
        end else if (state == FETCH_WAIT && fetch_data_valid && fetch_data_ready) begin
            fill_counter <= fill_counter + 1;
        end
    end

    // Buffer management
    always_ff @(posedge clk) begin
        if (rst) begin
            buffer_base_addr <= '0;
            buffer_valid <= 1'b0;
            for (int i = 0; i < BUFFER_DEPTH; i++) begin
                instr_buffer[i] <= '0;
            end
        end else if (state == FETCH_REQUEST) begin
            buffer_valid <= 1'b0;
            buffer_base_addr <= pc;  // Store PC, not HBM address (fetch_addr includes offset)
        end else begin
            if (state == FETCH_WAIT && fetch_data_valid && fetch_data_ready) begin
                instr_buffer[fill_counter] <= fetch_data;
            end
            // tl_master registers fetch_data one cycle behind its combinational
            // complete_fetch, so the final beat lands a cycle after fetch_complete.
            // Mark the page valid only once that LAST beat has actually been
            // written (last_fill_beat), otherwise instr_buffer[BUFFER_DEPTH-1]
            // is left stale and the last instruction of every page is dropped.
            if (last_fill_beat) begin
                buffer_valid <= 1'b1;
            end
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;

        case (state)
            IDLE: begin
                // Trigger fetch if buffer not valid or PC is outside current buffer range
                if (!buffer_valid || !pc_in_range) begin
                    next_state = FETCH_REQUEST;
                end
            end

            FETCH_REQUEST: begin
                if (fetch_req) begin
                    next_state = FETCH_WAIT;
                end
            end

            FETCH_WAIT: begin
                // Stay until the final beat is consumed into the buffer
                // (last_fill_beat). Leaving on fetch_complete (combinational on
                // the last D-beat handshake) would deassert fetch_data_ready
                // before tl_master presents that registered last beat,
                // dropping instr_buffer[BUFFER_DEPTH-1].
                if (last_fill_beat) begin
                    next_state = IDLE;
                end
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // Output logic
    always_comb begin
        fetch_req = 1'b0;
        fetch_addr = '0;
        // Stay ready outside FETCH_WAIT too: a late beat from the previous
        // burst is accepted and DISCARDED (fills are gated to FETCH_WAIT),
        // instead of lingering until the next refill and corrupting buffer[0].
        fetch_data_ready = 1'b1;

        case (state)
            FETCH_REQUEST: begin
                fetch_req = 1'b1;
                // Load instructions starting from current PC + InstrStorageOffset in HBM
                fetch_addr = pc + InstrStorageOffset;
            end

            default: begin
                // No action
            end
        endcase
    end

endmodule
