// Transpose Buffer for 32x32 matrix (8-bit elements)
// This buffer can load a full 32x32 matrix and output rows or columns

`timescale 1ns/1ps

module transpose_buffer_32x32 (
    input  logic clk,
    input  logic rst_n,

    // Load interface (load one row per cycle)
    input  logic load_en,
    input  logic [4:0] load_row_idx,
    input  logic [255:0] load_data,  // 32 elements * 8 bits

    // Read interface
    input  logic read_en,
    input  logic read_transposed,    // 0: read as row, 1: read as column
    input  logic [4:0] read_idx,
    output logic [255:0] read_data,
    output logic read_valid
);

    // Storage for 32x32 matrix (each element is 8 bits)
    // Organized as [row][col]
    logic [7:0] matrix [32][32];

    // Load logic - store one row per cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 32; i++) begin
                for (int j = 0; j < 32; j++) begin
                    matrix[i][j] <= '0;
                end
            end
        end else if (load_en) begin
            for (int j = 0; j < 32; j++) begin
                matrix[load_row_idx][j] <= load_data[j*8 +: 8];
            end
        end
    end

    // Read logic - output row or column
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_data <= '0;
            read_valid <= 1'b0;
        end else if (read_en) begin
            if (read_transposed) begin
                // Read column (transposed)
                for (int i = 0; i < 32; i++) begin
                    read_data[i*8 +: 8] <= matrix[i][read_idx];
                end
            end else begin
                // Read row (normal)
                for (int j = 0; j < 32; j++) begin
                    read_data[j*8 +: 8] <= matrix[read_idx][j];
                end
            end
            read_valid <= 1'b1;
        end else begin
            read_valid <= 1'b0;
        end
    end

endmodule
