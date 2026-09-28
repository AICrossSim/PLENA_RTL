`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

module vector_machine
    import precision_pkg::*;
    import configuration_pkg::*;
    import instruction_pkg::*;
#(
    localparam   ADDR_WIDTH     = ON_CHIP_ADDR_WIDTH
) (
    input   logic                                                               clk,
    input   logic                                                               rst,

    input   logic                                                               broadcast_fp2,
    input   V_ELEMENT_OP                                                        element_v_control,
    input   V_REDUCT_OP                                                         reduct_v_control,

    input   logic [VLEN-1:0] [(V_FP_MANT_WIDTH + V_FP_EXP_WIDTH):0]             v_a_in,
    input   logic                                                               v_a_valid,

    input   logic [VLEN-1:0] [(V_FP_MANT_WIDTH + V_FP_EXP_WIDTH):0]             v_b_in,
    input   logic                                                               v_b_valid,

    input   logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0]                        s_in,
    input   logic                                                               s_in_valid,
    input   logic [FP_OPERAND_WIDTH - 1 : 0]                                    s_wtarget,

    input   logic [ADDR_WIDTH - 1 : 0]                                          result_waddr,
    input   logic                                                               result_waddr_update,

    output  logic [VLEN-1:0] [(V_FP_MANT_WIDTH + V_FP_EXP_WIDTH):0]             v_out,

    output  logic [ADDR_WIDTH - 1: 0]                                           v_waddr,
    output  logic                                                               v_wreq,

    output  logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0]                        s_out,
    output  logic                                                               s_out_valid,
    output  logic  [FP_OPERAND_WIDTH - 1 : 0]                                   s_out_rd
);

    import pipeline_pkg::*;

    typedef struct packed {
        logic [ADDR_WIDTH-1:0]             waddr;
        V_ELEMENT_OP                       ele_op;
        V_REDUCT_OP                        red_op;
    } RECORDED_INFO_TYPE;

    localparam int VM_TRACK_DEPTH = VECTOR_LONGEST_OPERATE_CYCLES;
    localparam int VM_TRACK_W     = $bits(RECORDED_INFO_TYPE);

    RECORDED_INFO_TYPE          elem_din, elem_head;
    logic                       elem_push, elem_pop, elem_head_valid, elem_src_valid;

    RECORDED_INFO_TYPE          red_din, red_head;
    logic                       red_push, red_pop, red_head_valid, red_full;

    logic recorded_broadcast_en;
    V_ELEMENT_OP recorded_element_v_control;
    V_REDUCT_OP  recorded_reduct_v_control;
    logic [FP_OPERAND_WIDTH - 1:0] recorded_s_wtarget;
    logic [ADDR_WIDTH - 1:0] recorded_result_waddr;

    logic v_port_a_valid;
    logic v_port_b_valid;

    logic complete_element_prepare, complete_reduct_prepare;

    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] prepared_v_a;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] prepared_v_b;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] unpacked_v_s;

    logic s_acc_in_valid;
    logic red_v_in_valid;
    logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0] s_acc_in;

    V_REDUCT_OP                       red_din_op;
    logic [FP_OPERAND_WIDTH - 1 : 0]  red_din_wtarget;

    logic element_v_in_a_valid;
    logic element_v_in_b_valid;
    logic element_v_out_valid;

    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] element_v_out;
    logic [VLEN-1:0] [(V_FP_MANT_WIDTH + V_FP_EXP_WIDTH):0]   result_v_out;
    logic [VLEN-1:0] [(V_FP_MANT_WIDTH + V_FP_EXP_WIDTH):0]   p1_result_v_out, p2_result_v_out;
    logic [ADDR_WIDTH-1:0] stored_result_waddr;
    logic compute_result_valid;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] element_in_v_a;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] element_in_v_b;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] reduct_in_v;
    logic [V_FP_EXP_WIDTH + V_FP_MANT_WIDTH : 0] reduct_in_s;

`ifdef HADAMARD_EN
    logic hadamard_transform_in_valid, hadamard_transform_out_valid;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] hadamard_transform_v_in;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] hadamard_transform_v_out;
`endif

`ifdef MAMBA_SCAN_EN
    logic prefix_scan_in_valid, prefix_scan_out_valid;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] prefix_scan_v_in;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] prefix_scan_v_out;
`endif
`ifdef MAMBA_EXTENSION_EN
    logic shift_in_valid, shift_out_valid;
    logic [$clog2(VLEN)-1:0] shift_amount;
    logic [$clog2(VLEN)-1:0] shift_amount_hold;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] shift_v_in;
    logic [VLEN-1:0] [(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH) : 0] shift_v_out;
`endif

    always_ff @(posedge clk) begin
        if (rst) begin
            recorded_element_v_control <= STALL_V_ELEMENT;
            recorded_reduct_v_control <= STALL_V_REDUCT;
            recorded_broadcast_en <= 1'b0;
            recorded_s_wtarget <= 'b0;
            recorded_result_waddr <= 'b0;
        end else begin
            if (result_waddr_update) begin
                recorded_result_waddr <= result_waddr;
            end

            if (element_v_control != STALL_V_ELEMENT) begin
                recorded_element_v_control  <= element_v_control;
                recorded_broadcast_en       <= broadcast_fp2;
            end

            if (reduct_v_control != STALL_V_REDUCT) begin
                recorded_reduct_v_control   <= reduct_v_control;
                recorded_s_wtarget          <= s_wtarget;
                recorded_element_v_control  <= STALL_V_ELEMENT;
                recorded_broadcast_en       <= 1'b0;
            end
        end
    end

    assign elem_push = (recorded_element_v_control != STALL_V_ELEMENT) & complete_element_prepare;

    assign red_push  = red_v_in_valid;

    assign elem_din = '{
        waddr  : recorded_result_waddr,
        ele_op : recorded_element_v_control,
        red_op : recorded_reduct_v_control
    };
    assign red_din = '{
        waddr  : {{(ADDR_WIDTH - FP_OPERAND_WIDTH){1'b0}}, red_din_wtarget},
        ele_op : recorded_element_v_control,
        red_op : red_din_op
    };

    assign elem_pop = elem_head_valid & elem_src_valid;
    assign red_pop  = red_head_valid  & s_out_valid;

    fifo #(
        .DATA_WIDTH (VM_TRACK_W),
        .DEPTH      (VM_TRACK_DEPTH)
    ) elem_track_fifo (
        .clk            (clk),
        .rst            (rst),
        .data_in        (elem_din),
        .data_in_valid  (elem_push),
        .data_in_ready  (),
        .data_out       (elem_head),
        .data_out_valid (elem_head_valid),
        .data_out_ready (elem_pop),
        .empty          (),
        .full           ()
    );

    fifo #(
        .DATA_WIDTH (VM_TRACK_W),
        .DEPTH      (VM_TRACK_DEPTH)
    ) red_track_fifo (
        .clk            (clk),
        .rst            (rst),
        .data_in        (red_din),
        .data_in_valid  (red_push),
        .data_in_ready  (),
        .data_out       (red_head),
        .data_out_valid (red_head_valid),
        .data_out_ready (red_pop),
        .empty          (),
        .full           (red_full)
    );

    always_comb begin
        if (rst) begin
            complete_element_prepare = 1'b0;
            complete_reduct_prepare = 1'b0;
        end else begin

            if ((recorded_element_v_control == PREFIX_SCAN_V_ELEMENT) && v_port_a_valid) begin
                complete_element_prepare    = 1'b1;
                complete_reduct_prepare     = 1'b0;
            end else if ((recorded_element_v_control == SHIFT_V_LANES_ELEMENT) && v_port_a_valid) begin
                complete_element_prepare    = 1'b1;
                complete_reduct_prepare     = 1'b0;
            end else

            if (((recorded_element_v_control != STALL_V_ELEMENT) & !recorded_broadcast_en & v_port_a_valid & v_port_b_valid) ||
                         ((recorded_element_v_control != STALL_V_ELEMENT) &  recorded_broadcast_en & v_port_a_valid)) begin
                complete_element_prepare    = 1'b1;
                complete_reduct_prepare     = 1'b0;
            end else if ((recorded_reduct_v_control != STALL_V_REDUCT) & v_port_a_valid & s_acc_in_valid & !red_full) begin
                complete_element_prepare    = 1'b0;
                complete_reduct_prepare     = 1'b1;
            end
        `ifdef HADAMARD_EN
            else if ((recorded_element_v_control == INNER_HADAMARD_TRANSFORM) & v_port_a_valid) begin
                complete_element_prepare    = 1'b1;
                complete_reduct_prepare     = 1'b0;
            end
        `endif

        `ifdef MAMBA_EXTENSION_EN
            else if (((recorded_element_v_control == PREFIX_SCAN_V_ELEMENT) || (recorded_element_v_control == SHIFT_V_LANES_ELEMENT)) & v_port_a_valid) begin
                complete_element_prepare    = 1'b1;
                complete_reduct_prepare     = 1'b0;
            end
        `endif
            else begin
                complete_element_prepare    = 1'b0;
                complete_reduct_prepare     = 1'b0;
            end
        end
    end

    broadcast #(
        .DATA_WIDTH(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1),
        .BROADCAST_DIM(VLEN)
    ) broadcaset_scalar (
        .in_data(s_in),
        .out_data(unpacked_v_s)
    );
    logic v_a_valid_d1;
    always_ff @(posedge clk) begin
        if (rst) v_a_valid_d1 <= 1'b0;
        else     v_a_valid_d1 <= v_a_valid;
    end

    wire ps_mode_now = (recorded_element_v_control == PREFIX_SCAN_V_ELEMENT);
    wire shift_mode_now = (recorded_element_v_control == SHIFT_V_LANES_ELEMENT);
    wire v_a_valid_eff = (ps_mode_now|shift_mode_now) ? v_a_valid_d1 : v_a_valid;

`ifdef MAMBA_EXTENSION_EN
    // Capture the V_SHFT_V shift amount (gp<rs2> value, arriving on result_waddr=addr_2)
    // the cycle the shift op is presented, and hold it across the vector-read latency so
    // it is stable when shift_in_valid samples shift_amount. The pack issues a run of
    // identical-shift V_SHFT_Vs per head, so holding the last value is always correct.
    always_ff @(posedge clk) begin
        if (rst)                                            shift_amount_hold <= '0;
        else if (element_v_control == SHIFT_V_LANES_ELEMENT) shift_amount_hold <= result_waddr[$clog2(VLEN)-1:0];
    end
`endif

    register_slice_wo_hs #(
        .DATA_WIDTH(VLEN * (V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1))
    ) v_a_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (v_a_in),
        .data_in_valid  (v_a_valid),
        .data_out       (prepared_v_a),
        .data_out_valid (v_port_a_valid)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(VLEN * (V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1))
    ) v_b_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (recorded_broadcast_en ? unpacked_v_s : v_b_in ),
        .data_in_valid  (recorded_broadcast_en ? s_in_valid : v_b_valid),
        .data_out       (prepared_v_b),
        .data_out_valid (v_port_b_valid)
    );

    register_slice_wo_hs #(
        .DATA_WIDTH(V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1)
    ) s_in_buffer (
        .clk(clk),
        .rst(rst),
        .data_in        (s_in),
        .data_in_valid  ((recorded_reduct_v_control != STALL_V_REDUCT) ? s_in_valid : 1'b0),
        .data_out       (s_acc_in),
        .data_out_valid (s_acc_in_valid)
    );

    always_ff @(posedge clk) begin
        `ifdef HADAMARD_EN
            hadamard_transform_in_valid <= v_port_a_valid & (recorded_element_v_control == INNER_HADAMARD_TRANSFORM);
            hadamard_transform_v_in     <= prepared_v_a;
        `endif

        `ifdef MAMBA_SCAN_EN
            prefix_scan_in_valid <= v_port_a_valid & (recorded_element_v_control == PREFIX_SCAN_V_ELEMENT);
            prefix_scan_v_in     <= prepared_v_a;
        `endif
        `ifdef MAMBA_EXTENSION_EN
            shift_in_valid       <= v_port_a_valid & (recorded_element_v_control == SHIFT_V_LANES_ELEMENT);
            shift_v_in           <= prepared_v_a;
            // V_SHFT_V shift amount = gp<rs2> value, delivered as result_waddr (=addr_2).
            // Held (shift_amount_hold) from when the op was in exe so it is stable through
            // the vector-read latency; the s_acc_in path only carries the reduct FP seed
            // (never loaded for an element op) so it read 0 -> the shift was a silent no-op.
            shift_amount         <= shift_amount_hold;
        `endif

        if (((recorded_element_v_control != STALL_V_ELEMENT) && (recorded_element_v_control != INNER_HADAMARD_TRANSFORM) && (recorded_element_v_control != PREFIX_SCAN_V_ELEMENT) && (recorded_element_v_control != SHIFT_V_LANES_ELEMENT))) begin
            element_v_in_a_valid            <= v_port_a_valid;
            element_v_in_b_valid            <= v_port_b_valid;
            element_in_v_a                  <= prepared_v_a;
            element_in_v_b                  <= prepared_v_b;
        end else begin
            element_v_in_a_valid <= 1'b0;
            element_v_in_b_valid <= 1'b0;
            element_in_v_a       <= 'b0;
            element_in_v_b       <= 'b0;
            reduct_in_v          <= 'b0;
        end

        if (rst) begin
            reduct_in_v          <= 'b0;
            red_v_in_valid       <= 1'b0;
            red_din_op           <= STALL_V_REDUCT;
            red_din_wtarget      <= 'b0;
        end else if (complete_reduct_prepare) begin
            reduct_in_v          <= prepared_v_a;
            red_v_in_valid       <= 1'b1;
            red_din_op           <= recorded_reduct_v_control;
            red_din_wtarget      <= recorded_s_wtarget;
        end else begin
            reduct_in_v          <= 'b0;
            red_v_in_valid       <= 1'b0;
        end

        reduct_in_s              <= s_acc_in;

    end

    fp_elementwise_compute_unit #(
        .EXP_WIDTH(V_FP_EXP_WIDTH),
        .MANT_WIDTH(V_FP_MANT_WIDTH),
        .VLEN(VLEN)
    ) element_unit (
        .clk(clk),
        .rst(rst),
        .v_in_a         (element_in_v_a),
        .v_in_a_valid   (element_v_in_a_valid),
        .v_in_b         (element_in_v_b),
        .v_in_b_valid   (element_v_in_b_valid),
        .operation      (recorded_element_v_control),
        .v_out          (element_v_out),
        .v_out_valid    (element_v_out_valid)
    );

    always_comb begin
        result_v_out         = 'b0;
        elem_src_valid       = 1'b0;
        stored_result_waddr  = 'b0;

        if (elem_head_valid) begin
            case (elem_head.ele_op)
                ADD_V_ELEMENT, SUB_V_ELEMENT, RSUB_V_ELEMENT, MUL_V_ELEMENT,
                EXP_V_ELEMENT, RECI_V_ELEMENT: begin
                    result_v_out        = element_v_out;
                    elem_src_valid      = element_v_out_valid;
                    stored_result_waddr = elem_head.waddr;
                end
            `ifdef MAMBA_SCAN_EN
                PREFIX_SCAN_V_ELEMENT: begin
                    result_v_out        = prefix_scan_v_out;
                    elem_src_valid      = prefix_scan_out_valid;
                    stored_result_waddr = elem_head.waddr;
                end
            `endif
            `ifdef MAMBA_EXTENSION_EN
                SHIFT_V_LANES_ELEMENT: begin
                    result_v_out        = shift_v_out;
                    elem_src_valid      = shift_out_valid;
                    stored_result_waddr = elem_head.waddr;
                end
            `endif
            `ifdef HADAMARD_EN
                INNER_HADAMARD_TRANSFORM: begin
                    result_v_out        = hadamard_transform_v_out;
                    elem_src_valid      = hadamard_transform_out_valid;
                    stored_result_waddr = elem_head.waddr;
                end
            `endif
                default: begin
                    result_v_out        = 'b0;
                    elem_src_valid      = 1'b0;
                    stored_result_waddr = 'b0;
                end
            endcase
        end

        compute_result_valid = elem_src_valid;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            p1_result_v_out <= 'b0;
            p2_result_v_out <= 'b0;
            v_out           <= 'b0;
        end else begin
            if (compute_result_valid) begin
                v_wreq  <= 1'b1;
                v_waddr <= stored_result_waddr;
            end else begin
                v_wreq  <= 1'b0;
                v_waddr <= 'b0;
            end

            if (compute_result_valid) begin
                p1_result_v_out <= result_v_out;
            end else begin
                p1_result_v_out <= 'b0;
            end
            p2_result_v_out <= p1_result_v_out;
            v_out           <= p2_result_v_out;
        end
    end

    fp_reduction_compute_unit #(
        .EXP_WIDTH  (V_FP_EXP_WIDTH),
        .MANT_WIDTH (V_FP_MANT_WIDTH),
        .VLEN       (VLEN)
    ) reduction_unit (
        .clk(clk),
        .rst(rst),
        .v_in           ({reduct_in_v, reduct_in_s}),
        .v_in_valid     (red_v_in_valid),
        .operation      (recorded_reduct_v_control),
        .s_out          (s_out),
        .s_out_valid    (s_out_valid),
        .s_out_ready    (1'b1)
    );

    assign s_out_rd = (red_head_valid &&
                       (red_head.red_op == SUM_V_REDUCT || red_head.red_op == MAX_V_REDUCT))
                    ? red_head.waddr[FP_OPERAND_WIDTH-1:0] : 'b0;

    `ifdef HADAMARD_EN
        per_tile_hadamard_transform #(
            .TILESIZE   (VLEN),
            .EXP_WIDTH  (V_FP_EXP_WIDTH),
            .MANT_WIDTH (V_FP_MANT_WIDTH)
        ) hadamard_transform_unit (
            .clk(clk),
            .rst(rst),
            .data_in_valid      (hadamard_transform_in_valid),
            .data_in            (prepared_v_a),
            .data_out_valid     (hadamard_transform_out_valid),
            .data_out           (hadamard_transform_v_out)
        );
    `endif

    `ifdef MAMBA_EXTENSION_EN
        fp_vec_shift #(
            .VLEN   (VLEN),
            .BITWIDTH (V_FP_EXP_WIDTH + V_FP_MANT_WIDTH + 1),
            // RIGHT_SHIFT=0 => data moves toward HIGHER lanes: [a0,a1,a2,a3] shift 2 ->
            // [0,0,a0,a1]. The GQA output pack (program_attention _pack_o_head_to_output)
            // is the sole V_SHFT_V user and needs this "up" placement (head h's PV in
            // lanes 0..head_dim -> output lanes h*head_dim). The default (=1) shifts the
            // other way ([a2,a3,0,0]), which drops every head but head 0 to zero.
            .RIGHT_SHIFT (1'b0)
        ) vec_shift_unit (
            .clk            (clk),
            .rst            (rst),
            .v_in_valid     (shift_in_valid),
            .v_in           (shift_v_in),
            .shift_amount   (shift_amount),
            .v_out_valid    (shift_out_valid),
            .v_out          (shift_v_out)
        );
    `endif

    `ifdef MAMBA_SCAN_EN
        fp_prefix_scan_syn #(
            .VLEN(VLEN),
            .EXP_WIDTH(V_FP_EXP_WIDTH),
            .MANT_WIDTH(V_FP_MANT_WIDTH)
        ) prefix_scan_unit (
            .clk        (clk),
            .rst        (rst),
            .vin        (prefix_scan_v_in),
            .vout       (prefix_scan_v_out),
            .in_valid   (prefix_scan_in_valid),
            .out_valid  (prefix_scan_out_valid)
        );

    `endif

endmodule
