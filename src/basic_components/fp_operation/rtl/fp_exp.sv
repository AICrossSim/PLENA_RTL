`timescale 1ns / 1ps
/*
 fp_exp: Computes exp(x) via x*log2(e) decomposition + Taylor series.

 Pipeline stages (5 stages for FPGA timing closure):
   Stage 1: log2(e) multiply + round + negate (PIPELINE_OUTPUT=1 on DSP)
   Stage 2: barrel shift (PIPELINE_STAGE=1 on signed_left_shift)
   Stage 3: int/frac split + Taylor term1 (ln2*x) + term2 (x²/2)
   Stage 4: Taylor term3 first mult (term2*term1)
   Stage 5: Taylor term3 first mult rounding + sideband alignment
   Stage 6: Taylor term3 second mult (*TERM_1) + registered 4-way sum
*/
module fp_exp #(
    parameter   IN_EXP_WIDTH = 5,
    parameter   IN_FIX_WIDTH = 8,
    parameter   IN_FIX_FRAC_WIDTH = 5,
    parameter   EXTEND_WIDTH = 5,
    parameter   OUT_EXP_WIDTH = -1,
    parameter   OUT_FIX_WIDTH = -1,
    parameter   OUT_FIX_FRAC_WIDTH = -1
)(
    input logic clk,
    input logic rst,
    input  logic data_in_valid,
    input logic signed [IN_FIX_WIDTH - 1:0] signed_mant_in,
    input logic signed [IN_EXP_WIDTH:0] signed_exp_in,
    output logic data_out_valid,
    output logic signed [OUT_EXP_WIDTH - 1:0] signed_exp_out,
    output logic signed [OUT_FIX_WIDTH - 1:0] signed_mant_out
);
  /* verilator no_inline_module */  // keep scope so cocotb can probe internals

  // REG_N must equal the data path latency from data_in to signed_mant_out so
  // data_out_valid aligns with the SETTLED result. Measured at FP12: the Taylor
  // result settles one cycle LATER than the old REG_N=12 asserted valid, so the
  // output was sampled mid-flight (e.g. exp(1.0): taylor 1024->1070->1382 over
  // c11-c13, but valid fired at c12 capturing the partial 1070). 13 aligns it.
  localparam REG_N = 13;
  localparam signed [7-1:0] MLOG2_E = 7'd92;
  localparam signed ELOG2_E = 4'd1;

  localparam LOG2_E_WIDTH = IN_FIX_WIDTH + EXTEND_WIDTH;

  localparam MAX_INT_WIDTH = 10;
  localparam FIXED_POINT_DATA_WIDTH = LOG2_E_WIDTH + MAX_INT_WIDTH;
  localparam FIXED_POINT_DATA_FRAC_WIDTH = IN_FIX_FRAC_WIDTH + EXTEND_WIDTH;
  localparam TAYLOR_OUTPUT_WIDTH = FIXED_POINT_DATA_FRAC_WIDTH + 3;

  logic unsigned [IN_FIX_WIDTH - 1:0] unsigned_mant_in;
  logic unsigned [LOG2_E_WIDTH - 1:0] unsigned_mant_in_extended;
  assign unsigned_mant_in_extended = {unsigned_mant_in, {EXTEND_WIDTH{1'b0}}};

  logic [IN_EXP_WIDTH:0] signed_exp_in_log2_e;
  logic [LOG2_E_WIDTH - 1:0] signed_mant_in_log2_e;
  logic [LOG2_E_WIDTH - 1:0] unsigned_mant_in_log2_e;

  logic mant_sign;
  logic stall;
  assign stall = 1'b0;

  assign mant_sign = signed_mant_in[IN_FIX_WIDTH-1];
  assign unsigned_mant_in = mant_sign ? -signed_mant_in : signed_mant_in;

  // Valid tracking pipeline (matches REG_N data stages)
  data_reg #(
    .DATA_WIDTH(1),
    .REG_N(REG_N)
  ) data_reg_inst (
    .data_in(data_in_valid),
    .data_out(data_out_valid),
    .*
  );

  // =========================================================================
  // Stage 1: log2(e) multiply + round + negate
  // =========================================================================

  // PIPELINE_OUTPUT=1: DSP48 MREG absorbs the flop (+1 cycle).
  integer_mult #(
    .WIDTH(LOG2_E_WIDTH),
    .IN_1_WIDTH(7),
    .PIPELINE_OUTPUT(1)
  ) integer_mult_inst_0 (
    .clk(clk),
    .rst(rst),
    .data_in_0(unsigned_mant_in_extended),
    .data_in_1(MLOG2_E),
    .data_out(unsigned_mant_in_log2_e)
  );

  // mant_sign must be delayed 1 cycle to line up with the MREG-registered product.
  logic mant_sign_d;
  data_reg #(.DATA_WIDTH(1), .REG_N(1))
    pipe_mant_sign (.data_in(mant_sign), .data_out(mant_sign_d), .*);
  assign signed_mant_in_log2_e = mant_sign_d ? -unsigned_mant_in_log2_e : unsigned_mant_in_log2_e;
  assign signed_exp_in_log2_e = $signed(signed_exp_in) + ELOG2_E;

  // --- Pipeline register: after log2e mult + negate ---
  // pipe_s1_mant REG_N=1 since integer_mult_inst_0 MREG took over retiming.
  // pipe_s1_exp REG_N=2 to match latency: mant = MREG(1) + pipe(1) = 2, exp = 0 + 2.
  logic [LOG2_E_WIDTH - 1:0] s1_signed_mant;
  logic [IN_EXP_WIDTH:0]     s1_signed_exp;

  data_reg #(.DATA_WIDTH(LOG2_E_WIDTH), .REG_N(1))
    pipe_s1_mant (.data_in(signed_mant_in_log2_e), .data_out(s1_signed_mant), .*);
  data_reg #(.DATA_WIDTH(IN_EXP_WIDTH + 1), .REG_N(2))
    pipe_s1_exp (.data_in(signed_exp_in_log2_e), .data_out(s1_signed_exp), .*);

  // =========================================================================
  // Stage 2: barrel shift → data_reg_inst_0
  // =========================================================================

  logic [FIXED_POINT_DATA_WIDTH - 1:0] fixed_point_data_in_reg;
  logic [FIXED_POINT_DATA_WIDTH - 1:0] fixed_point_data_in;
  logic [TAYLOR_OUTPUT_WIDTH - 1:0] taylor_output;

  // PIPELINE_STAGE=1: register between sign-decode and barrel mux tree.
  bit_width_aware_signed_left_shift #(
    .IN_WIDTH(LOG2_E_WIDTH),
    .OUT_WIDTH(FIXED_POINT_DATA_WIDTH),
    .SHIFT_WIDTH(IN_EXP_WIDTH + 1),
    .PIPELINE_STAGE(1)
  ) bit_width_aware_signed_left_shift_inst (
    .clk(clk),
    .rst(rst),
    .in_data(s1_signed_mant),
    .shift_amt(s1_signed_exp),
    .out_data(fixed_point_data_in_reg)
  );

  // REG_N=3: provides retiming slots for the barrel-mux output cloud.
  data_reg #(
    .DATA_WIDTH(FIXED_POINT_DATA_WIDTH),
    .REG_N(3)
  ) data_reg_inst_0 (
    .data_in(fixed_point_data_in_reg),
    .data_out(fixed_point_data_in),
    .*
  );

  // =========================================================================
  // Stage 3 + 4: int/frac split → Taylor series
  // =========================================================================

  logic [FIXED_POINT_DATA_WIDTH - 1:FIXED_POINT_DATA_FRAC_WIDTH] fixed_point_int_part, fixed_point_int_part_reg;
  logic [FIXED_POINT_DATA_FRAC_WIDTH - 1:0] fixed_point_frac_part;
  assign fixed_point_int_part_reg = fixed_point_data_in[FIXED_POINT_DATA_WIDTH - 1:FIXED_POINT_DATA_FRAC_WIDTH];
  assign fixed_point_frac_part = $signed(fixed_point_data_in) - {fixed_point_int_part_reg, {FIXED_POINT_DATA_FRAC_WIDTH{1'b0}}};

  taylor_series_expansion #(
    .IN_WIDTH(FIXED_POINT_DATA_FRAC_WIDTH),
    .OUT_WIDTH(TAYLOR_OUTPUT_WIDTH)
  ) taylor_series_expansion_inst (
    .clk(clk),
    .rst(rst),
    .stall(stall),
    .data_in(fixed_point_frac_part),
    .data_out(taylor_output)
  );

  // Delay int_part to match Taylor pipeline depth
  // (inst_0 MREG + inst_1 MREG + inst_2 MREG + stage-5 pipe + output reg)
  data_reg #(
    .DATA_WIDTH($bits(fixed_point_int_part_reg)),
    .REG_N(7)
  ) data_reg_inst_1 (
    .data_in(fixed_point_int_part_reg),
    .data_out(fixed_point_int_part),
    .*
  );

  assign signed_exp_out = fixed_point_int_part;
  assign signed_mant_out = taylor_output >> EXTEND_WIDTH;
endmodule



module taylor_series_expansion #(
    // The input is assumed to be in the range of [0, 1]
    parameter   IN_WIDTH = 8,
    parameter   OUT_WIDTH = IN_WIDTH + 2
)(
    input logic clk,
    input logic rst,
    input logic stall,
    input logic unsigned [IN_WIDTH - 1:0] data_in,
    output logic unsigned [OUT_WIDTH - 1:0] data_out
);

  localparam COEFFICIENT_WIDTH = IN_WIDTH + 2; // the maximum is 2.7..

  localparam [IN_WIDTH : 0] TERM_0 = 1 << (IN_WIDTH); // 1.0 in fixed point
  localparam [IN_WIDTH - 1 : 0] TERM_1 = TERM_0 / 3; // 1/3 in fixed point
  localparam [5-1:0] LN_2 = 22 ; // ln(2) ≈ 0.693

  logic [IN_WIDTH - 1:0] element_list_1, element_list_2;

  // =========================================================================
  // Taylor Stage 1 (combinational): term1 + term2
  // =========================================================================

  // Term 1: ln(2) * x
  // PIPELINE_OUTPUT=1: register DSP output before rounding feeds inst_1.
  // Breaks 5.7 ns DSP→round→DSP cascade. +1 cycle.
  integer_mult #(
    .WIDTH(IN_WIDTH),
    .IN_1_WIDTH(5),
    .PIPELINE_OUTPUT(1)
  ) integer_mult_inst_0 (
    .clk(clk),
    .rst(rst),
    .data_in_0(data_in),
    .data_in_1(LN_2),
    .data_out(element_list_1)
  );

  // Term 2: ln²(2) * x² / 2 = term1 * term1 / 2
  // PIPELINE_OUTPUT=1: DSP48 MREG breaks cascade into inst_2. +1 cycle.
  logic [IN_WIDTH - 1:0] intermediate_data_out, intermediate_data_out_2;
  integer_mult #(
    .WIDTH(IN_WIDTH),
    .PIPELINE_OUTPUT(1)
  ) integer_mult_inst_1 (
    .clk(clk),
    .rst(rst),
    .data_in_0(element_list_1),
    .data_in_1(element_list_1),
    .data_out(intermediate_data_out)
  );
  logic [IN_WIDTH - 1:0] element_list_1_reg, element_list_2_reg;
  assign element_list_2 = (intermediate_data_out) >> 1;

  // data_reg_inst_1 REG_N=3: +1 for inst_0 PIPELINE_OUTPUT, +1 for inst_1 PIPELINE_OUTPUT
  data_reg #(
    .DATA_WIDTH(IN_WIDTH),
    .REG_N(3)
  ) data_reg_inst_1 (
    .data_in(element_list_1),
    .data_out(element_list_1_reg),
    .*
  );
  data_reg #(
    .DATA_WIDTH(IN_WIDTH),
    .REG_N(2)
  ) data_reg_inst_2 (
    .data_in(element_list_2),
    .data_out(element_list_2_reg),
    .*
  );

  // =========================================================================
  // Taylor Stage 2: term3 first multiply
  // =========================================================================

  // Term 3 part 1: element_list_2_reg * element_list_1_reg
  // PIPELINE_OUTPUT=1 breaks the remaining post-route critical DSP->round->DSP path.
  integer_mult #(
    .WIDTH(IN_WIDTH),
    .PIPELINE_OUTPUT(1)
  ) integer_mult_inst_2 (
    .clk(clk),
    .rst(rst),
    .data_in_0(element_list_2_reg),
    .data_in_1(element_list_1_reg),
    .data_out(intermediate_data_out_2)
  );

  // --- Pipeline register: after term3 first mult ---
  logic [IN_WIDTH - 1:0] s4_intermediate, s4_elem1, s4_elem2;

  data_reg #(.DATA_WIDTH(IN_WIDTH), .REG_N(1))
    pipe_inter (.data_in(intermediate_data_out_2), .data_out(s4_intermediate), .*);
  data_reg #(.DATA_WIDTH(IN_WIDTH), .REG_N(2))
    pipe_elem1 (.data_in(element_list_1_reg), .data_out(s4_elem1), .*);
  data_reg #(.DATA_WIDTH(IN_WIDTH), .REG_N(2))
    pipe_elem2 (.data_in(element_list_2_reg), .data_out(s4_elem2), .*);

  // =========================================================================
  // Output: term3 second multiply + registered 4-way sum
  // =========================================================================

  // Term 3 part 2: intermediate * TERM_1 (= 1/3)
  logic [IN_WIDTH - 1:0] element_list_3_reg;
  integer_mult #(
    .WIDTH(IN_WIDTH),
    .IN_1_WIDTH(IN_WIDTH)
  ) integer_mult_inst_3 (
    .clk(clk),
    .rst(rst),
    .data_in_0(s4_intermediate),
    .data_in_1(TERM_1),
    .data_out(element_list_3_reg)
  );

  // Sum all terms (uses stage-matched registered values)
  logic [OUT_WIDTH - 1:0] fixed_tree_in [4-1:0];

  assign fixed_tree_in[0] = TERM_0;
  assign fixed_tree_in[1] = s4_elem1;
  assign fixed_tree_in[2] = s4_elem2;
  assign fixed_tree_in[3] = element_list_3_reg;

  logic [OUT_WIDTH - 1:0] sum_result;
  assign sum_result = fixed_tree_in[0] + fixed_tree_in[1] + fixed_tree_in[2] + fixed_tree_in[3];

  always_ff @(posedge clk) begin
    if (rst) data_out <= '0;
    else if (!stall) data_out <= sum_result;
  end

endmodule

module integer_mult #(
    parameter   WIDTH = 8,
    parameter   IN_1_WIDTH = WIDTH,
    // PIPELINE_OUTPUT=1: register output into DSP48E1 MREG (zero fabric cost).
    // Adds 1 cycle of latency.
    parameter   PIPELINE_OUTPUT = 0
)(
    input  logic clk,
    input  logic rst,
    input  logic [WIDTH - 1:0] data_in_0,
    input  logic [IN_1_WIDTH - 1:0] data_in_1,
    output logic [WIDTH - 1:0] data_out
);

  localparam INTERMEDIATE_WIDTH = WIDTH + IN_1_WIDTH;
  logic signed [INTERMEDIATE_WIDTH - 1:0] intermediate_data_out;
  logic signed [INTERMEDIATE_WIDTH - 1:0] post_mult_data;
  assign intermediate_data_out = data_in_0 * data_in_1;

  generate
    if (PIPELINE_OUTPUT) begin : gen_pipelined_mult
      always_ff @(posedge clk) begin
        if (rst) post_mult_data <= '0;
        else     post_mult_data <= intermediate_data_out;
      end
    end else begin : gen_comb_mult
      assign post_mult_data = intermediate_data_out;
    end
  endgenerate

  round_to_nearest_even #(
    .IN_WIDTH(INTERMEDIATE_WIDTH),
    .OUT_WIDTH(WIDTH)
  ) round_to_nearest_even_inst (
    .data_in(post_mult_data),
    .data_out(data_out)
  );

endmodule
