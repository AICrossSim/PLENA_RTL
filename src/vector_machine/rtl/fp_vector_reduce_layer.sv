`timescale 1ns / 1ps

`include "configuration.svh"
`include "operation.svh"

/*
Module      : Floating Point Reduction Tree
Timing      : Combinatorial Logic
Description : Binary Tree Reduction of Floating Point Numbers, supporting:
            1. SUM
            2. MAX
            We assume the Layer Dim is in power of 2.
Status      : Under Development
*/

module fp_vector_reduce_layer #(
    // Declared Input Width
    parameter OVERALL_INPUT_WIDTH = 16,

    parameter LAYER_DIM  = 2,
    parameter IN_MAN_WIDTH = 4,
    parameter IN_EXP_WIDTH  = 3, 
    
    // In the first version vector machine, we assume all the extension related bits are zero.
    parameter EXT_MANT_WIDTH = 0,
    parameter EXT_EXP_WIDTH = 0,   

    localparam OUT_DIM  = (LAYER_DIM + 1) / 2,
    localparam INPUT_DATA_WIDTH = IN_MAN_WIDTH + IN_EXP_WIDTH + 1,
    localparam OUTPUT_DATA_WIDTH = IN_MAN_WIDTH + EXT_MANT_WIDTH + IN_EXP_WIDTH + EXT_EXP_WIDTH + 1,

    // Per-pair latencies (valid in -> out): fp_cp_adder = 6 cycles, fp_max = 1.
    localparam SUM_LATENCY = 6,
    localparam MAX_LATENCY = 1
) (
    input   logic clk,
    input   logic rst,
    input   V_REDUCT_OP operation, // 0: SUM, 1: MAX  (op of the item AT THE INPUT)
    input   logic [OVERALL_INPUT_WIDTH -1 : 0] data_in,
    input   logic data_in_valid,
    output  logic data_in_ready,
    output  logic [OVERALL_INPUT_WIDTH -1 : 0] data_out,
    output  logic data_out_valid,
    // Op of the item at data_out this cycle (the launched op, latency-matched).
    output  V_REDUCT_OP operation_out,
    input   logic data_out_ready
);

    // Ready signal always high (no backpressure)
    assign data_in_ready = 1'b1;

    logic [OUT_DIM * OUTPUT_DATA_WIDTH -1 : 0] layer_add_out, layer_max_out;

    logic [LAYER_DIM / 2 - 1: 0] adder_data_in_valid_array;
    logic [LAYER_DIM / 2 - 1: 0] adder_data_out_valid_array;
    logic [LAYER_DIM / 2 - 1: 0] max_data_in_valid_array;
    logic [LAYER_DIM / 2 - 1: 0] max_data_out_valid_array;

    logic sum_data_in_valid;
    logic sum_data_out_valid;
    logic max_data_in_valid;
    logic max_data_out_valid;

    // INPUT-side op gate: steer the entering item into the SUM or MAX path.
    always_comb begin
        sum_data_in_valid = 1'b0;
        max_data_in_valid = 1'b0;

        case (operation)
            SUM_V_REDUCT: sum_data_in_valid = data_in_valid;
            MAX_V_REDUCT: max_data_in_valid = data_in_valid;
            default: ; // STALL / unknown: drive neither path
        endcase
    end

    // Select output by which path produced a valid (not the live op). Pad MAX to
    // SUM_LATENCY so both paths share one latency and outputs never collide.
    localparam MAX_EXTRA_DELAY = SUM_LATENCY - MAX_LATENCY;

    logic                              max_valid_dly;
    logic [OUT_DIM*OUTPUT_DATA_WIDTH-1:0] max_data_dly;
    generate
        if (MAX_EXTRA_DELAY == 0) begin : g_no_max_pad
            assign max_valid_dly = max_data_out_valid;
            assign max_data_dly  = layer_max_out;
        end else begin : g_max_pad
            logic                              max_v_pipe [MAX_EXTRA_DELAY-1:0];
            logic [OUT_DIM*OUTPUT_DATA_WIDTH-1:0] max_d_pipe [MAX_EXTRA_DELAY-1:0];
            always_ff @(posedge clk) begin
                if (rst) begin
                    for (int s = 0; s < MAX_EXTRA_DELAY; s++) begin
                        max_v_pipe[s] <= 1'b0;
                        max_d_pipe[s] <= '0;
                    end
                end else begin
                    max_v_pipe[0] <= max_data_out_valid;
                    max_d_pipe[0] <= layer_max_out;
                    for (int s = 1; s < MAX_EXTRA_DELAY; s++) begin
                        max_v_pipe[s] <= max_v_pipe[s-1];
                        max_d_pipe[s] <= max_d_pipe[s-1];
                    end
                end
            end
            assign max_valid_dly = max_v_pipe[MAX_EXTRA_DELAY-1];
            assign max_data_dly  = max_d_pipe[MAX_EXTRA_DELAY-1];
        end
    endgenerate

    always_comb begin
        data_out_valid = sum_data_out_valid | max_valid_dly;
        if (max_valid_dly)
            data_out = {{OVERALL_INPUT_WIDTH - OUT_DIM * OUTPUT_DATA_WIDTH{1'b0}}, max_data_dly};
        else
            data_out = {{OVERALL_INPUT_WIDTH - OUT_DIM * OUTPUT_DATA_WIDTH{1'b0}}, layer_add_out};
    end

    // Broadcast valid to all adders/max units
    assign adder_data_in_valid_array = {(LAYER_DIM/2){sum_data_in_valid}};
    assign max_data_in_valid_array = {(LAYER_DIM/2){max_data_in_valid}};

    // Per-item op tag at the output, from whichever path produced the valid.
    assign operation_out = max_valid_dly    ? MAX_V_REDUCT
                         : sum_data_out_valid ? SUM_V_REDUCT
                         : STALL_V_REDUCT;



    generate;
        for (genvar i = 0; i < LAYER_DIM / 2; i++) begin : adder_pair
            fp_cp_adder #(
                .EXP_WIDTH(IN_EXP_WIDTH),
                .MANT_WIDTH(IN_MAN_WIDTH)
            )   layer_fp_add (
                .clk(clk),
                .rst(rst),
                .data_in_valid      (adder_data_in_valid_array[i]),
                .data_a             (data_in[2*i*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .data_b             (data_in[(2*i + 1)*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .data_out           (layer_add_out[i * OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH]),
                .data_out_valid     (adder_data_out_valid_array[i])
            );

            fp_max #(
                .EXP_WIDTH(IN_EXP_WIDTH),
                .MANT_WIDTH(IN_MAN_WIDTH)
            )   layer_fp_max (
                .clk(clk),
                .rst(rst),
                .data_in_valid      (max_data_in_valid_array[i]),
                .data_a             (data_in[2*i*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .data_b             (data_in[(2*i + 1)*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .data_out           (layer_max_out[i * OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH]),
                .data_out_valid     (max_data_out_valid_array[i])
            );

        end
    endgenerate

    // Odd-dimension carry: carry the unpaired top input into the top output slot,
    // latency-matched and zero-extended to OUTPUT_DATA_WIDTH.
    generate
        if (LAYER_DIM % 2 == 1) begin : g_odd_carry
            // Unpaired input element, extended to the output element width.
            logic [OUTPUT_DATA_WIDTH-1:0] carry_in;
            assign carry_in = {
                // sign
                data_in[(LAYER_DIM-1)*INPUT_DATA_WIDTH + IN_MAN_WIDTH + IN_EXP_WIDTH],
                // exponent (MSB-padded with zeros for extension)
                {EXT_EXP_WIDTH{1'b0}},
                data_in[(LAYER_DIM-1)*INPUT_DATA_WIDTH + IN_MAN_WIDTH +: IN_EXP_WIDTH],
                // mantissa (LSB-padded with zeros for extension)
                data_in[(LAYER_DIM-1)*INPUT_DATA_WIDTH +: IN_MAN_WIDTH],
                {EXT_MANT_WIDTH{1'b0}}
            };

            // SUM-path delay line (SUM_LATENCY stages).
            logic [OUTPUT_DATA_WIDTH-1:0] sum_carry_pipe [SUM_LATENCY-1:0];
            always_ff @(posedge clk) begin
                if (rst) begin
                    for (int s = 0; s < SUM_LATENCY; s++)
                        sum_carry_pipe[s] <= '0;
                end else begin
                    sum_carry_pipe[0] <= carry_in;
                    for (int s = 1; s < SUM_LATENCY; s++)
                        sum_carry_pipe[s] <= sum_carry_pipe[s-1];
                end
            end

            // MAX-path carry: match MAX_LATENCY; the output padding delays the rest.
            logic [OUTPUT_DATA_WIDTH-1:0] max_carry_pipe [MAX_LATENCY-1:0];
            always_ff @(posedge clk) begin
                if (rst) begin
                    for (int s = 0; s < MAX_LATENCY; s++)
                        max_carry_pipe[s] <= '0;
                end else begin
                    max_carry_pipe[0] <= carry_in;
                    for (int s = 1; s < MAX_LATENCY; s++)
                        max_carry_pipe[s] <= max_carry_pipe[s-1];
                end
            end

            // Drive the top (otherwise-undriven) output slot with the carried element.
            assign layer_add_out[(OUT_DIM-1)*OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH] =
                       sum_carry_pipe[SUM_LATENCY-1];
            assign layer_max_out[(OUT_DIM-1)*OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH] =
                       max_carry_pipe[MAX_LATENCY-1];
        end
    endgenerate

    // Combine all valid signals
    assign sum_data_out_valid = &adder_data_out_valid_array;
    assign max_data_out_valid = &max_data_out_valid_array;

endmodule
