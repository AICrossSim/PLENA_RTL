`timescale 1ns / 1ps

module fp_2_mx_int_block #(
    parameter BLOCK_DIM = 8,
    parameter FP_MANT_WIDTH = 3,
    parameter FP_EXP_WIDTH = 4,
    parameter MXINT_WIDTH = 8,
    parameter MXINT_SCALE_WIDTH = FP_EXP_WIDTH
)(
    input  logic clk,
    input  logic rst,
    input  logic [BLOCK_DIM-1:0][FP_MANT_WIDTH + FP_EXP_WIDTH : 0] data_in,
    input  logic data_in_valid,
    output logic [BLOCK_DIM-1:0][MXINT_WIDTH - 1 : 0] element_data_out,
    output logic [MXINT_SCALE_WIDTH-1:0] scale_data_out,
    output logic mx_int_data_out_valid
);
    localparam SIGNED_MANT_WIDTH = FP_MANT_WIDTH + 2;

    logic signed [FP_EXP_WIDTH:0] signed_exp [BLOCK_DIM-1:0];
    logic signed [SIGNED_MANT_WIDTH-1:0] signed_mant [BLOCK_DIM-1:0];

    for (genvar i = 0; i < BLOCK_DIM; i++) begin : gen_partition
        fp_ieee_partition #(
            .EXP_WIDTH(FP_EXP_WIDTH),
            .MANT_WIDTH(FP_MANT_WIDTH)
        ) partition_inst (
            .data_in        (data_in[i]),
            .signed_exp     (signed_exp[i]),
            .signed_mant    (signed_mant[i])
        );
    end

    logic signed [FP_EXP_WIDTH:0] signed_exp_max;

    always_comb begin
        signed_exp_max = signed_exp[0];
        for (int i = 1; i < BLOCK_DIM; i++) begin
            if ($signed(signed_exp[i]) > $signed(signed_exp_max)) begin
                signed_exp_max = signed_exp[i];
            end
        end
    end

    logic all_mant_zero;
    always_comb begin
        all_mant_zero = 1'b1;
        for (int i = 0; i < BLOCK_DIM; i++) begin
            if (signed_mant[i] != '0) begin
                all_mant_zero = 1'b0;
            end
        end
    end

    localparam signed [FP_EXP_WIDTH:0] MIN_SHARED_EXP = -8;
    logic block_negligible;
    assign block_negligible = ($signed(signed_exp_max) < MIN_SHARED_EXP);

    localparam INTERNAL_EXP_WIDTH = (FP_EXP_WIDTH + 1 > MXINT_SCALE_WIDTH) ? (FP_EXP_WIDTH + 1) : MXINT_SCALE_WIDTH;
    logic signed [INTERNAL_EXP_WIDTH-1:0] p1_signed_exp_max;
    logic signed [FP_EXP_WIDTH:0] p1_signed_exp [BLOCK_DIM-1:0];
    logic signed [SIGNED_MANT_WIDTH-1:0] p1_signed_mant [BLOCK_DIM-1:0];
    logic p1_data_valid;
    logic p1_all_mant_zero;
    logic p1_block_negligible;

    always_ff @(posedge clk) begin
        if (rst) begin
            p1_signed_exp_max <= '0;
            p1_data_valid <= 1'b0;
            p1_all_mant_zero <= 1'b0;
            p1_block_negligible <= 1'b0;
            for (int i = 0; i < BLOCK_DIM; i++) begin
                p1_signed_exp[i] <= '0;
                p1_signed_mant[i] <= '0;
            end
        end else begin

            p1_signed_exp_max <= INTERNAL_EXP_WIDTH'(signed'(signed_exp_max));
            p1_data_valid <= data_in_valid;
            p1_all_mant_zero <= all_mant_zero;
            p1_block_negligible <= block_negligible;
            for (int i = 0; i < BLOCK_DIM; i++) begin
                p1_signed_exp[i] <= signed_exp[i];
                p1_signed_mant[i] <= signed_mant[i];
            end
        end
    end

    logic signed [FP_EXP_WIDTH:0] shift_amount [BLOCK_DIM-1:0];
    logic [MXINT_WIDTH-1:0] shifted_result [BLOCK_DIM-1:0];

    for (genvar i = 0; i < BLOCK_DIM; i++) begin : gen_shift
        assign shift_amount[i] = p1_signed_exp[i] - p1_signed_exp_max;

        bit_width_aware_signed_left_shift #(
            .IN_WIDTH(SIGNED_MANT_WIDTH),
            .OUT_WIDTH(MXINT_WIDTH),
            .SHIFT_WIDTH(FP_EXP_WIDTH + 1)
        ) shift_inst (
            .clk(clk),
            .rst(rst),
            .in_data(p1_signed_mant[i]),
            .shift_amt(shift_amount[i]),
            .out_data(shifted_result[i])
        );
    end

    localparam signed [INTERNAL_EXP_WIDTH-1:0] SCALE_BIAS = 2**(MXINT_SCALE_WIDTH - 1) - 1;

    logic [MXINT_SCALE_WIDTH-1:0] p2_scale;
    logic p2_data_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            p2_scale <= '0;
            p2_data_valid <= 1'b0;
            for (int i = 0; i < BLOCK_DIM; i++) begin
                element_data_out[i] <= '0;
            end
        end else begin

            if (p1_all_mant_zero || p1_block_negligible) begin
                p2_scale <= MXINT_SCALE_WIDTH'(SCALE_BIAS);
                for (int i = 0; i < BLOCK_DIM; i++) begin
                    element_data_out[i] <= '0;
                end
            end else begin

                p2_scale <= MXINT_SCALE_WIDTH'(p1_signed_exp_max + SCALE_BIAS);
                for (int i = 0; i < BLOCK_DIM; i++) begin
                    element_data_out[i] <= shifted_result[i];
                end
            end
            p2_data_valid <= p1_data_valid;
        end
    end

    assign scale_data_out = p2_scale;
    assign mx_int_data_out_valid = p2_data_valid;

endmodule
