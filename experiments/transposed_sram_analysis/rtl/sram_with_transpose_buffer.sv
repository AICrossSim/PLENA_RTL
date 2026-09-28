// Combined SRAM + Transpose Buffer design
// This design provides similar functionality to biaccess_sram:
// - Store a 32x32 matrix in SRAM
// - Support both normal and transposed reads
//
// Architecture:
// - Pure SRAM stores the 32x32 matrix (32 rows * 256 bits)
// - Transpose buffer enables column-wise reads
// - For transposed reads, must first load matrix into transpose buffer

`timescale 1ns/1ps

module sram_with_transpose_buffer (
    input  logic clk,
    input  logic rst_n,

    // Read interface
    input  logic req,
    input  logic transposed_read,
    input  logic [4:0] sram_raddr,
    output logic [31:0] [7:0] out_data,
    output logic read_valid,

    // Write interface
    input  logic wen_req,
    output logic write_response,
    input  logic [4:0] sram_waddr,
    input  logic [255:0] write_data,

    // Transpose buffer control
    input  logic load_transpose_buffer,  // Signal to load entire matrix into transpose buffer
    output logic transpose_buffer_ready  // Indicates transpose buffer is loaded
);

    // Internal signals
    logic [255:0] sram_rdata;
    logic sram_read_valid;
    logic [255:0] tb_read_data;
    logic tb_read_valid;

    // State machine for loading transpose buffer
    typedef enum logic [1:0] {
        IDLE,
        LOADING,
        READY
    } state_t;

    state_t state, next_state;
    logic [4:0] load_counter;
    logic load_en;
    logic [4:0] load_row_idx;

    // Pure SRAM instance
    pure_sram_32x32 u_sram (
        .clk(clk),
        .req(req || (state == LOADING)),  // Read for user request or buffer loading
        .raddr((state == LOADING) ? load_counter : sram_raddr),
        .rdata(sram_rdata),
        .read_valid(sram_read_valid),
        .wen(wen_req),
        .waddr(sram_waddr),
        .wdata(write_data),
        .write_response(write_response)
    );

    // Transpose buffer instance
    transpose_buffer_32x32 u_transpose_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .load_en(load_en),
        .load_row_idx(load_row_idx),
        .load_data(sram_rdata),
        .read_en(req && transposed_read && (state == READY)),
        .read_transposed(1'b1),  // Always read as column when using transpose buffer
        .read_idx(sram_raddr),
        .read_data(tb_read_data),
        .read_valid(tb_read_valid)
    );

    // State machine for loading transpose buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            load_counter <= '0;
            load_row_idx <= '0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    if (load_transpose_buffer) begin
                        load_counter <= '0;
                    end
                end
                LOADING: begin
                    if (sram_read_valid) begin
                        load_row_idx <= load_counter;
                        if (load_counter == 5'd31) begin
                            load_counter <= '0;
                        end else begin
                            load_counter <= load_counter + 1'b1;
                        end
                    end
                end
                READY: begin
                    if (wen_req) begin
                        // If writing to SRAM, transpose buffer becomes stale
                        load_counter <= '0;
                    end
                end
                default: begin
                    load_counter <= '0;
                end
            endcase
        end
    end

    always_comb begin
        next_state = state;
        load_en = 1'b0;
        transpose_buffer_ready = 1'b0;

        case (state)
            IDLE: begin
                if (load_transpose_buffer) begin
                    next_state = LOADING;
                end
            end
            LOADING: begin
                load_en = sram_read_valid;
                if (sram_read_valid && (load_counter == 5'd31)) begin
                    next_state = READY;
                end
            end
            READY: begin
                transpose_buffer_ready = 1'b1;
                if (wen_req) begin
                    // Invalidate transpose buffer on write
                    next_state = IDLE;
                end
            end
            default: next_state = IDLE;
        endcase
    end

    // Output mux
    always_comb begin
        if (transposed_read && (state == READY)) begin
            out_data = tb_read_data;
            read_valid = tb_read_valid;
        end else begin
            out_data = sram_rdata;
            read_valid = sram_read_valid && !transposed_read;
        end
    end

endmodule
