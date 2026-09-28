`timescale 1ns / 1ps

/*
Module      : lock
Description : Holds a transient request until a busy consumer can take it.
            : The rising edge of `set` arms `locked`; `locked` stays high until
            : `clear` (the consumer accepted the request). This lets a request
            : that is only asserted for a cycle or two survive a consumer that is
            : busy at the moment the request arrives.
*/

module lock (
    input  logic clk,
    input  logic rst,
    input  logic set,      // request indication (level); its rising edge arms the lock
    input  logic clear,    // request accepted by the consumer
    output logic locked    // held request -> drive the consumer's req input
);
    logic set_q;
    always_ff @(posedge clk) begin
        if (rst) set_q <= 1'b0;
        else     set_q <= set;
    end
    wire set_edge = set & ~set_q;

    always_ff @(posedge clk) begin
        if (rst)           locked <= 1'b0;
        else if (set_edge) locked <= 1'b1;   // new request; arm (dominates clear)
        else if (clear)    locked <= 1'b0;   // consumer took it
    end
endmodule
