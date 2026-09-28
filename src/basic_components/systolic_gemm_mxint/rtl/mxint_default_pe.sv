`timescale 1ns / 1ps

/*
Module      : Processing Element (PE) in Systolic GEMM
Timing      : Sequential
Description : Self-contained MAC unit. Streams (top, left) operand pairs and
              auto-completes a length-ACC_DEPTH dot product. When the final MAC
              has been absorbed:
                * out_valid pulses for one cycle
                * the accumulator is auto-cleared the same cycle
                * acc_counter wraps to 0, ready for the next dot product

USAGE CONSTRAINT (driver/MCU responsibility):
   The driver MUST insert at least ONE gap cycle between consecutive dot
   products — i.e., (system_top_valid && system_left_valid) must be deasserted
   for one cycle after out_valid fires before the next batch of MACs starts.

   Why: out_valid is also the clear_accumulator strobe to fix_accumulator. If a
   new MAC fires on the same cycle as out_valid, clear wins and the new MAC is
   discarded. This is the explicit design choice (option 1 in our review on
   2026-05-24) — the MCU inserts the gap rather than fix_accumulator supporting
   "load-on-clear" semantics. See doc/design-decisions/pe-gap-cycle.md.

Status      : Under Development
*/

module mxint_default_pe #(
    // MX-INT Data Format
    parameter MX_L_INT_WIDTH        = 4,
    parameter MX_T_INT_WIDTH        = 4,
    parameter MXINT_SCALE_WIDTH     = 8,

    parameter ACC_DEPTH             = 16,  // Depth of the accumulator
    parameter ACC_EXPAND_WIDTH      = $clog2(ACC_DEPTH),
    localparam OUT_INT_WIDTH        = MX_T_INT_WIDTH + MX_L_INT_WIDTH + ACC_EXPAND_WIDTH,
    localparam OUT_SCALE_WIDTH      = MXINT_SCALE_WIDTH + 1
)(

    input logic clk,
    input logic rst,

    // Input from Top
    input  logic [MX_T_INT_WIDTH - 1 : 0]       in_top_element,
    input  logic [MXINT_SCALE_WIDTH - 1 : 0]    in_top_scale,
    input  logic in_top_valid,
    // Input from Left
    input  logic [MX_L_INT_WIDTH - 1 : 0]       in_left_element,
    input  logic [MXINT_SCALE_WIDTH - 1 : 0]    in_left_scale,
    input  logic in_left_valid,


    // Output to Bottom
    output logic [MX_T_INT_WIDTH - 1 : 0]       out_bottom_element,
    output logic [MXINT_SCALE_WIDTH - 1 : 0]    out_bottom_scale,

    // Output to Right
    output logic [MX_L_INT_WIDTH - 1 : 0]       out_right_element,
    output logic [MXINT_SCALE_WIDTH - 1 : 0]    out_right_scale,

    // Output Result — lossless accumulator (full width, signed 2's complement) + biased scale.
    // INT→FP conversion is deferred to the MCU output stage (after the mxint adder
    // tree cross-K reduction), so the PE itself stays integer-only.
    output logic [OUT_INT_WIDTH - 1 : 0]      out_int,
    output logic [OUT_SCALE_WIDTH - 1 : 0]    out_scale,
    output logic out_valid
);
    // ==============================================================================================
    // Declaration : registers, wires
    // ==============================================================================================
    localparam SCALE_BIAS = (1 << (MXINT_SCALE_WIDTH - 1)) - 1;

    logic [MX_T_INT_WIDTH - 1 : 0]                  reg_top_element;
    logic [MXINT_SCALE_WIDTH - 1 : 0]               reg_top_scale;
    logic [MX_L_INT_WIDTH - 1 : 0]                  reg_left_element;
    logic [MXINT_SCALE_WIDTH - 1 : 0]               reg_left_scale;

    // ==============================================================================================
    // ASSERTION: in_top_valid and in_left_valid must be synchronised.
    // The PE assumes both operand streams arrive on the same cycle. A mismatch
    // would split a MAC across two cycles (only one operand updated) which is
    // not what this PE supports — the MCU/driver is expected to gate both
    // valids together.
    // ==============================================================================================
    // Temporarily hide for dataflow verification
    // always_ff @(posedge clk) begin
    //     if (!rst) begin
    //         assert (in_top_valid == in_left_valid)
    //             else $error("[%0t] %m: in_top_valid (%0b) != in_left_valid (%0b)",
    //                         $time, in_top_valid, in_left_valid);
    //     end
    // end

    // ==============================================================================================
    // STAGE 1: Pass Data from Top and Left to the Bottom and Right
    // ==============================================================================================

    always_ff @(posedge clk) begin
        if (rst) begin
            reg_top_element  <= {MX_T_INT_WIDTH{1'b0}};
            reg_top_scale    <= {MXINT_SCALE_WIDTH{1'b0}};
            reg_left_element <= {MX_L_INT_WIDTH{1'b0}};
            reg_left_scale   <= {MXINT_SCALE_WIDTH{1'b0}};
        end else begin
            if (in_top_valid) begin
                reg_top_element <= in_top_element;
                reg_top_scale   <= in_top_scale;
            end

            if (in_left_valid) begin
                reg_left_element <= in_left_element;
                reg_left_scale   <= in_left_scale;
            end
        end
    end
    // ==============================================================================================
    // MAC depth counter: counts the number of MACs that have settled into the accumulator.
    //   - Increments one cycle AFTER a MAC fires (i.e. when its result is in the accumulator).
    //   - Saturates at ACC_DEPTH — that's the "dot product complete" state.
    //   - out_valid = (acc_counter == ACC_DEPTH) — output is ready to be read.
    // ==============================================================================================
    logic [$clog2(ACC_DEPTH + 1) - 1 : 0] acc_counter;
    logic mac_fire, mac_fire_d;
    assign mac_fire = in_top_valid && in_left_valid;

    // Pipeline mac_fire by one cycle to align with reg_top/reg_left being valid.
    // mult_result = $signed(reg_top) * $signed(reg_left) becomes valid one cycle
    // after the inputs are presented, so the accumulator must absorb it then.
    always_ff @(posedge clk) begin
        if (rst) mac_fire_d <= 1'b0;
        else     mac_fire_d <= mac_fire;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            acc_counter <= '0;
        end else if (mac_fire_d) begin
            if (acc_counter == ACC_DEPTH) begin
                acc_counter <= '0;                       // auto-wrap after completion
            end else begin
                acc_counter <= acc_counter + 1;
            end
        end
    end

    assign out_bottom_element   =   reg_top_element;
    assign out_bottom_scale     =   reg_top_scale;

    assign out_right_element    =   reg_left_element;
    assign out_right_scale      =   reg_left_scale; 

    // ==============================================================================================
    // STAGE 2: Multiplication of the elements from Top and Left, Scale Summation
    // ==============================================================================================
    // Note: Here we assum Left is higher precision.
    // scale_sum_result widened by 1 bit so reg_top_scale + reg_left_scale - SCALE_BIAS
    // doesn't lose the carry-out when both scales are near the unsigned max.
    logic [OUT_SCALE_WIDTH - 1 : 0] scale_sum_result;
    logic [MX_T_INT_WIDTH + MX_L_INT_WIDTH - 1 : 0] mult_result;

    (* use_dsp = "yes" *) assign mult_result = $signed(reg_top_element) * $signed(reg_left_element);
    assign scale_sum_result = {1'b0, reg_top_scale} + {1'b0, reg_left_scale} - SCALE_BIAS;

    // ==============================================================================================
    // STAGE 3: Accumulation
    // ==============================================================================================
    logic [OUT_INT_WIDTH - 1 : 0] acc_result;
    logic clear_accumulator;
    assign clear_accumulator = (acc_counter == ACC_DEPTH) && mac_fire_d;

    fix_accumulator #(
        .WIDTH (MX_T_INT_WIDTH + MX_L_INT_WIDTH),
        .EXPAND_WIDTH (ACC_EXPAND_WIDTH)
    ) acc_adder (
        .clk(clk),
        .rst(rst),
        .clear_accumulator (clear_accumulator),  // clear on dot-product complete
        .data_in_valid  (mac_fire_d),    // delayed: aligns with mult_result
        .data_in        (mult_result),
        .data_out       (acc_result),
        .data_out_valid ()
    );

    // ==============================================================================================
    // STAGE 4: Outputs — lossless INT accumulator + biased scale (no FP conversion here)
    // ==============================================================================================
    assign out_int   = acc_result;
    assign out_scale = scale_sum_result;
    assign out_valid = (acc_counter == ACC_DEPTH);

endmodule
