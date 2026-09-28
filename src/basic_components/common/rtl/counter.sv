`timescale 1ns / 1ps

// Saturating up/down counter. incr=+1, decr=-1, both=hold. Saturates at 0 and
// at all-ones (no wrap). nonzero is high while count != 0.
module counter #(
    parameter WIDTH = 4
) (
    input  logic             clk,
    input  logic             rst,
    input  logic             incr,
    input  logic             decr,
    output logic [WIDTH-1:0] count,
    output logic             nonzero
);
    localparam logic [WIDTH-1:0] MAX_VAL = {WIDTH{1'b1}};

    always_ff @(posedge clk) begin
        if (rst) begin
            count <= '0;
        end else begin
            case ({incr, decr})
                2'b10:   if (count != MAX_VAL) count <= count + 1'b1;
                2'b01:   if (count != '0)      count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    assign nonzero = (count != '0);
endmodule
