`timescale 1ns / 1ps
/*
Module      : Floating Point Adder Tree Layer
Timing      : Combinational (PIPELINE=0) or 3-stage pipelined (PIPELINE=1):
              stage 1 — fp_adder internal register (PIPELINE_STAGE=1)
              stage 2 — register between adder output and normalize  (pipe_enable-gated)
              stage 3 — register between normalize and cast           (pipe_valid_mid-gated)
              When PIPELINE=0, one unconditional register sits between adder and normalize
              (original behavior) and valid propagates with 1-cycle latency.
Status      : Passed Simple Tests
*/

module fp_adder_tree_layer #(
    // Declared Input Width
    parameter OVERALL_INPUT_WIDTH = 16,

    parameter LAYER_DIM  = 2,
    parameter IN_MAN_WIDTH = 4,
    parameter IN_EXP_WIDTH  = 3,

    // Max possible shift bits needed
    parameter EXT_MANT_WIDTH = 0,
    parameter EXT_EXP_WIDTH = 0,

    // Pipeline control: 0 = 1-stage register (original), 1 = 2-stage registered midpoints
    parameter PIPELINE = 0,

    localparam OUT_DIM  = (LAYER_DIM + 1) / 2,
    localparam INPUT_DATA_WIDTH = IN_MAN_WIDTH + IN_EXP_WIDTH + 1,
    localparam OUTPUT_DATA_WIDTH = IN_MAN_WIDTH + EXT_MANT_WIDTH + IN_EXP_WIDTH + EXT_EXP_WIDTH + 1
) (
    input  logic [OVERALL_INPUT_WIDTH -1 : 0] data_in,
    input  logic clk,
    input  logic rst,
    output logic [OVERALL_INPUT_WIDTH -1 : 0] data_out,
    // Pipeline flow control:
    //   PIPELINE=0: pipe_enable unused; pipe_valid_out = 1-cycle delayed pipe_enable
    //   PIPELINE=1: pipe_enable gates both internal registers;
    //               pipe_valid_out asserted 2 cycles after pipe_enable
    input  logic pipe_enable,
    output logic pipe_valid_out
);

    logic last_element_sign;
    localparam UNUSED_BITS = OVERALL_INPUT_WIDTH - OUTPUT_DATA_WIDTH * LAYER_DIM;

    localparam int BIAS = (1 << (IN_EXP_WIDTH - 1)) - 1;
    localparam int NEW_BIAS = (1 << ((IN_EXP_WIDTH + EXT_EXP_WIDTH) - 1)) - 1;
    localparam int UPDATED_BIAS = NEW_BIAS - BIAS;

    logic [IN_EXP_WIDTH + EXT_EXP_WIDTH - 1:0] updated_exp;
    logic [IN_MAN_WIDTH + EXT_MANT_WIDTH - 1:0] updated_man;

    // Combinational FP adder parameters (mirroring fp_cp_adder localparams)
    localparam int IN_FIXED_WIDTH = IN_MAN_WIDTH + 2;
    localparam int IN_FIXED_FRAC_WIDTH = IN_MAN_WIDTH;
    localparam int ADDER_OUT_EXP_WIDTH = IN_EXP_WIDTH;
    localparam int ADDER_OUT_FIXED_WIDTH = IN_FIXED_WIDTH + IN_FIXED_FRAC_WIDTH + 1;
    localparam int ADDER_OUT_FIXED_FRAC_WIDTH = IN_FIXED_FRAC_WIDTH + IN_FIXED_FRAC_WIDTH;
    localparam int NORMALIZE_OUT_EXP_WIDTH = ADDER_OUT_EXP_WIDTH + 1;
    localparam int NORMALIZE_OUT_MANT_WIDTH = ADDER_OUT_FIXED_WIDTH - 1;

    // Module-level 1-cycle-delayed pipe_enable used to gate the normalize→cast register.
    // Declared here so the pair genvar loop can reference it.
    logic pipe_valid_mid;

    generate;
        for (genvar i = 0; i < LAYER_DIM / 2; i++) begin : pair
            // Internal wires for combinational FP addition pipeline
            logic signed [IN_EXP_WIDTH:0]       signed_exp_a, signed_exp_b;
            logic signed [IN_FIXED_WIDTH-1:0]   signed_mant_a, signed_mant_b;
            logic signed [ADDER_OUT_EXP_WIDTH-1:0]   signed_exp_out;
            logic signed [ADDER_OUT_FIXED_WIDTH-1:0]  signed_mant_out;
            logic [NORMALIZE_OUT_EXP_WIDTH + NORMALIZE_OUT_MANT_WIDTH:0] normalized_data;
            logic [(IN_EXP_WIDTH + EXT_EXP_WIDTH) + NORMALIZE_OUT_MANT_WIDTH:0] exp_casted_data;

            // Step 1: Partition inputs into signed exp/mant (combinational)
            fp_ieee_partition #(
                .EXP_WIDTH(IN_EXP_WIDTH),
                .MANT_WIDTH(IN_MAN_WIDTH)
            ) partition_a (
                .data_in(data_in[2*i*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .signed_exp(signed_exp_a),
                .signed_mant(signed_mant_a)
            );

            fp_ieee_partition #(
                .EXP_WIDTH(IN_EXP_WIDTH),
                .MANT_WIDTH(IN_MAN_WIDTH)
            ) partition_b (
                .data_in(data_in[(2*i + 1)*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH]),
                .signed_exp(signed_exp_b),
                .signed_mant(signed_mant_b)
            );

            // Step 2: FP addition with mantissa alignment
            // PIPELINE_STAGE=1 registers between align (shifts) and add (CARRY4)
            fp_adder #(
                .IN_EXP_WIDTH(IN_EXP_WIDTH),
                .IN_FIX_WIDTH(IN_FIXED_WIDTH),
                .IN_FIX_FRAC_WIDTH(IN_FIXED_FRAC_WIDTH),
                .OUT_EXP_WIDTH(ADDER_OUT_EXP_WIDTH),
                .OUT_FIX_WIDTH(ADDER_OUT_FIXED_WIDTH),
                .OUT_FIX_FRAC_WIDTH(ADDER_OUT_FIXED_FRAC_WIDTH),
                .PIPELINE_STAGE(1)
            ) fp_adder_inst (
                .clk(clk),
                .rst(rst),
                .exp_a(signed_exp_a),
                .mant_a(signed_mant_a),
                .exp_b(signed_exp_b),
                .mant_b(signed_mant_b),
                .exp_out(signed_exp_out),
                .mant_out(signed_mant_out)
            );

            // ----- Stage 2: Pipeline register between adder output and normalize -----
            logic signed [ADDER_OUT_EXP_WIDTH-1:0]   norm_exp_in;
            logic signed [ADDER_OUT_FIXED_WIDTH-1:0]  norm_mant_in;

            if (PIPELINE == 1) begin : gen_pipe_mid
                // Enable-gated: only captures when pipe_enable is asserted
                always_ff @(posedge clk) begin
                    if (rst) begin
                        norm_exp_in  <= '0;
                        norm_mant_in <= '0;
                    end else if (pipe_enable) begin
                        norm_exp_in  <= signed_exp_out;
                        norm_mant_in <= signed_mant_out;
                    end
                end
            end else begin : gen_comb_mid
                // Unconditional register (original behavior — always latches every cycle)
                always_ff @(posedge clk) begin
                    if (rst) begin
                        norm_exp_in  <= '0;
                        norm_mant_in <= '0;
                    end else begin
                        norm_exp_in  <= signed_exp_out;
                        norm_mant_in <= signed_mant_out;
                    end
                end
            end

            // Step 3: Normalize back to IEEE format (combinational)
            fp_ieee_normalize #(
                .IN_FIXED_WIDTH(ADDER_OUT_FIXED_WIDTH),
                .IN_FIXED_FRAC_WIDTH(ADDER_OUT_FIXED_FRAC_WIDTH),
                .IN_EXP_WIDTH(ADDER_OUT_EXP_WIDTH),
                .OUT_MANT_WIDTH(NORMALIZE_OUT_MANT_WIDTH)
            ) fp_normalize (
                .signed_mant(norm_mant_in),
                .signed_exp(norm_exp_in),
                .fp_out(normalized_data)
            );

            // ----- Stage 3: Pipeline register between normalize and cast (PIPELINE=1 only) -----
            // Gated by pipe_valid_mid (= pipe_enable delayed 1 cycle, declared at module scope)
            logic [NORMALIZE_OUT_EXP_WIDTH + NORMALIZE_OUT_MANT_WIDTH:0] cast_in;

            if (PIPELINE == 1) begin : gen_pipe_norm
                always_ff @(posedge clk) begin
                    if (rst)                 cast_in <= '0;
                    else if (pipe_valid_mid) cast_in <= normalized_data;
                end
            end else begin : gen_comb_norm
                assign cast_in = normalized_data;
            end

            // Step 4: Cast to output format (combinational)
            fp_ieee_exponent_casting #(
                .IN_EXP_WIDTH(NORMALIZE_OUT_EXP_WIDTH),
                .OUT_EXP_WIDTH(IN_EXP_WIDTH + EXT_EXP_WIDTH),
                .MANT_WIDTH(NORMALIZE_OUT_MANT_WIDTH)
            ) exp_cast (
                .data_in(cast_in),
                .data_out(exp_casted_data)
            );

            fp_ieee_mantissa_casting #(
                .EXP_WIDTH(IN_EXP_WIDTH + EXT_EXP_WIDTH),
                .IN_MANT_WIDTH(NORMALIZE_OUT_MANT_WIDTH),
                .OUT_MANT_WIDTH(IN_MAN_WIDTH + EXT_MANT_WIDTH)
            ) mant_cast (
                .data_in(exp_casted_data),
                .data_out(data_out[i * OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH])
            );
        end
    endgenerate


    always_comb begin
        if (LAYER_DIM % 2 != 0) begin : left
            last_element_sign = data_in[(LAYER_DIM) * INPUT_DATA_WIDTH - 1];
            updated_exp = {{EXT_EXP_WIDTH{1'b0}}, data_in[(LAYER_DIM)*INPUT_DATA_WIDTH - 2 -: IN_EXP_WIDTH]} + UPDATED_BIAS[IN_EXP_WIDTH + EXT_EXP_WIDTH - 1 : 0];
            updated_man = {data_in[(LAYER_DIM) * INPUT_DATA_WIDTH - 2 - IN_MAN_WIDTH : (LAYER_DIM - 1) * INPUT_DATA_WIDTH], {EXT_MANT_WIDTH{1'b0}}};
            data_out[(OUT_DIM-1)*OUTPUT_DATA_WIDTH +: OUTPUT_DATA_WIDTH] = {last_element_sign, updated_exp, updated_man};
        end
        else begin
            last_element_sign = 1'b0;
            updated_exp = 0;
            updated_man = 0;
        end
    end

    // Pipeline valid tracking — drives pipe_valid_mid (module scope) and pipe_valid_out (port)
    generate
        if (PIPELINE == 1) begin : gen_pipe_valid
            // Stage 2 gate: 1 cycle after pipe_enable
            always_ff @(posedge clk)
                if (rst) pipe_valid_mid <= 1'b0;
                else     pipe_valid_mid <= pipe_enable;
            // Stage 3 gate / module output: 2 cycles after pipe_enable
            always_ff @(posedge clk)
                if (rst) pipe_valid_out <= 1'b0;
                else     pipe_valid_out <= pipe_valid_mid;
        end else begin : gen_comb_valid
            // PIPELINE=0: unconditional mid register, valid is 1-cycle delayed pipe_enable
            assign pipe_valid_mid = 1'b0;  // unused in this mode (cast_in is combinational)
            always_ff @(posedge clk)
                if (rst) pipe_valid_out <= 1'b0;
                else     pipe_valid_out <= pipe_enable;
        end
    endgenerate


endmodule
