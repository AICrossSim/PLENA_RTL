`timescale 1ns / 1ps

/*
Module      : MXFP Systolic Array
Timing      : Sequential
Description : It can be used for both GEMM and GEMV operations.
Status      : Under Development
*/

module mx_systolic_array #(
    // MX-FP Data Format
    parameter MX_T_EXP_WIDTH        = 4,
    parameter MX_T_MANT_WIDTH       = 3,
    parameter MX_L_EXP_WIDTH        = 4,
    parameter MX_L_MANT_WIDTH       = 3,
    parameter MX_SCALE_WIDTH        = 8,
    parameter BLOCK_DIM             = 4,
    // Accumulator Data Format
    parameter ACC_FP_EXP_WIDTH      = 8,
    parameter ACC_FP_MANT_WIDTH     = 7,
    // Dimension
    parameter COMPUTE_DIM           = 8,
    localparam BLOCK_NUM            = COMPUTE_DIM / BLOCK_DIM,
    parameter L_MX_INT_EN           = 0,
    parameter T_MX_INT_EN           = 0
)(

    input   logic clk,
    input   logic rst,
    input   logic clear_accumulator,
    input   logic control,

    // Input from Top Array
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * (MX_T_EXP_WIDTH + MX_T_MANT_WIDTH + 1) - 1 : 0] in_top_element,
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * MX_SCALE_WIDTH - 1 : 0] in_top_scale,
    input   logic in_top_valid,

    // Input from Top Vector Array
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * (MX_T_EXP_WIDTH + MX_T_MANT_WIDTH + 1) - 1 : 0] in_top_v_element,
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * MX_SCALE_WIDTH - 1 : 0] in_top_v_scale,
    input   logic in_top_v_valid,

    // Input from Left Array
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * (MX_L_MANT_WIDTH + MX_L_EXP_WIDTH + 1) - 1 : 0] in_left_element,
    input   logic [BLOCK_NUM - 1: 0]    [MX_SCALE_WIDTH - 1 : 0] in_left_scale,
    input   logic in_left_valid,

    // Input from Left Vector Array
    input   logic [BLOCK_NUM - 1: 0]    [BLOCK_DIM * (MX_L_MANT_WIDTH + MX_L_EXP_WIDTH + 1) - 1 : 0] in_left_v_element,
    input   logic [BLOCK_NUM - 1: 0]    [MX_SCALE_WIDTH - 1 : 0] in_left_v_scale,
    input   logic in_left_v_valid,

    // Output GEMM
    output  logic [COMPUTE_DIM - 1: 0]  [COMPUTE_DIM - 1: 0] [ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] m_out_fp,
    output  logic m_out_valid,

    // Output GEMV
    output  logic [COMPUTE_DIM - 1: 0]  [ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH : 0] v_out_fp
    
);

    initial begin
        if (COMPUTE_DIM % BLOCK_DIM != 0) begin
            $error("COMPUTE_DIM must be a multiple of BLOCK_DIM");
            $finish;
        end
    end

    logic [BLOCK_DIM * ( MX_L_MANT_WIDTH + MX_L_EXP_WIDTH + 1 ) - 1 : 0]        ho_transfer_elem      [BLOCK_NUM - 1:0][BLOCK_NUM :0];
    logic [MX_SCALE_WIDTH - 1 : 0]                                              ho_transfer_scale     [BLOCK_NUM - 1:0][BLOCK_NUM :0];
    logic [BLOCK_DIM * ( MX_T_MANT_WIDTH + MX_T_EXP_WIDTH + 1 ) - 1 : 0]        ve_transfer_elem      [BLOCK_NUM : 0][BLOCK_NUM - 1:0];
    logic [BLOCK_DIM * MX_SCALE_WIDTH - 1 : 0]                                  ve_transfer_scale     [BLOCK_NUM : 0][BLOCK_NUM - 1:0];

    logic [BLOCK_NUM- 1: 0] [BLOCK_NUM- 1: 0][BLOCK_DIM - 1: 0][BLOCK_DIM * (ACC_FP_MANT_WIDTH + ACC_FP_EXP_WIDTH + 1 ) - 1 : 0] result_values;


    logic system_right_shift_valid, system_down_shift_valid;
    logic [BLOCK_NUM- 1: 0] [BLOCK_NUM - 1: 0] pe_result_valid;
    logic determined_left_valid;
    logic determined_top_valid;

    assign determined_left_valid    = (control == 1'b0) ? in_left_valid : in_left_v_valid;
    assign determined_top_valid     = (control == 1'b0) ? in_top_valid  : in_top_v_valid;

    generate;
        for (genvar i = 0; i < BLOCK_NUM; i = i + 1) begin : fill_with_input_data
            // Fill the Top Row
            assign ve_transfer_elem [0][i]     = in_top_element [i];
            assign ve_transfer_scale[0][i]     = in_top_scale   [i];

            // Fill the Left Column
            assign ho_transfer_elem [i][0]     = in_left_element[i];
            assign ho_transfer_scale[i][0]     = in_left_scale  [i];
        end
        assign system_down_shift_valid  = determined_top_valid;
        assign system_right_shift_valid = determined_left_valid;
    endgenerate

    // Computation
    generate;
        for (genvar i = 0; i < BLOCK_NUM; i = i + 1) begin : pe_row
            for (genvar j = 0; j < BLOCK_NUM; j = j + 1) begin : pe_col
                if (i == 0) begin
                    mx_first_row_mini_systolic_array #(
                        .MX_T_EXP_WIDTH   (MX_T_EXP_WIDTH),
                        .MX_T_MANT_WIDTH  (MX_T_MANT_WIDTH),
                        .MX_L_EXP_WIDTH   (MX_L_EXP_WIDTH),
                        .MX_L_MANT_WIDTH  (MX_L_MANT_WIDTH),
                        .MX_SCALE_WIDTH   (MX_SCALE_WIDTH),
                        .BLOCK_DIM          (BLOCK_DIM),
                        .ACC_FP_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                        .ACC_FP_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                        .L_MX_INT_EN        (L_MX_INT_EN),
                        .T_MX_INT_EN        (T_MX_INT_EN)
                    ) first_row_mini_sys_init (
                        .clk(clk),
                        .rst(rst),
                        .clear_accumulator  (clear_accumulator),
                        .control            (control),
                        .in_top_element     (ve_transfer_elem[i][j]),
                        .in_top_scale       (ve_transfer_scale[i][j]),
                        .system_top_valid   (system_down_shift_valid),
                        .in_top_v_element   (in_top_v_element[j]),
                        .in_top_v_scale     (in_top_v_scale[j]),
                        .in_left_element    (ho_transfer_elem[i][j]),
                        .in_left_scale      (ho_transfer_scale[i][j]),
                        .system_left_valid  (system_right_shift_valid),
                        .in_left_v_element  (in_left_v_element[j]),
                        .in_left_v_scale    (in_left_v_scale[j]),
                        .out_bottom_element (ve_transfer_elem[i+1][j]),
                        .out_bottom_scale   (ve_transfer_scale[i+1][j]),
                        .out_right_element  (ho_transfer_elem[i][j+1]),
                        .out_right_scale    (ho_transfer_scale[i][j+1]),
                        .out_fp             (result_values[i][j]),
                        .out_result_valid   (pe_result_valid[i][j])
                    );
                end else begin
                    mx_mini_systolic_array #(
                        .MX_T_EXP_WIDTH     (MX_T_EXP_WIDTH),
                        .MX_T_MANT_WIDTH    (MX_T_MANT_WIDTH),
                        .MX_L_EXP_WIDTH     (MX_L_EXP_WIDTH),
                        .MX_L_MANT_WIDTH    (MX_L_MANT_WIDTH),
                        .MX_SCALE_WIDTH     (MX_SCALE_WIDTH),
                        .BLOCK_DIM          (BLOCK_DIM),
                        .ACC_FP_EXP_WIDTH   (ACC_FP_EXP_WIDTH),
                        .ACC_FP_MANT_WIDTH  (ACC_FP_MANT_WIDTH),
                        .L_MX_INT_EN        (L_MX_INT_EN),
                        .T_MX_INT_EN        (T_MX_INT_EN)
                    ) default_mini_sys_init (
                        .clk(clk),
                        .rst(rst),
                        .clear_accumulator  (clear_accumulator),
                        .in_top_element     (ve_transfer_elem[i][j]),
                        .in_top_scale       (ve_transfer_scale[i][j]),
                        .system_top_valid   (system_down_shift_valid),
                        .in_left_element    (ho_transfer_elem[i][j]),
                        .in_left_scale      (ho_transfer_scale[i][j]),
                        .system_left_valid  (system_right_shift_valid),
                        .out_bottom_element (ve_transfer_elem[i + 1][j]),
                        .out_bottom_scale   (ve_transfer_scale[i + 1][j]),
                        .out_right_element  (ho_transfer_elem[i][j + 1]),
                        .out_right_scale    (ho_transfer_scale[i][j + 1]),
                        .out_fp             (result_values[i][j]),
                        .out_result_valid   (pe_result_valid[i][j])
                    );
                end
            end
        end
    endgenerate

    assign m_out_valid      = &pe_result_valid;
    assign v_out_fp         = m_out_fp[0];

    always_ff @(posedge clk) begin
        if (rst) begin
            m_out_fp <= '0;
        end else begin
            for (int i = 0; i < COMPUTE_DIM; i++) begin
                for (int j = 0; j < BLOCK_NUM; j++) begin
                 m_out_fp[i][j * BLOCK_DIM +: BLOCK_DIM] <= result_values[i / BLOCK_DIM][j][i % BLOCK_DIM];
                end
            end
        end
    end

endmodule