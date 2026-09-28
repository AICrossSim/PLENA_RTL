`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

/*
Module      : Scalar Machine Module
Timing      : Sequential, all the operations completed in 1 cycle
Description : This module contains two modules:
            : FP ALU for all the fp computation related operations
            : Fixed ALU, only have addition and subtraction operations for address manipulation
Status      : Under Testing
*/

module scalar_machine import precision_pkg::*; import configuration_pkg::*; import instruction_pkg::*; #(
    `ifdef SIMULATION
        // Simulation Purpose
        parameter string FP_MEM_INIT_FILE = "",
        parameter string INT_MEM_INIT_FILE = "",
        parameter string FP_REG_RESULT_FILE = ""
    `endif
) (
    input   logic clk,
    input   logic rst,

    // Control
    input   OP_BUNDLE   exe_stage_op,
    input   S_INT_OP    assigned_int_op,

    // GP Register Control
    input   logic [INT_OPERAND_WIDTH - 1 : 0] rs1,
    input   logic [INT_OPERAND_WIDTH - 1 : 0] rs2,
    input   logic [INT_OPERAND_WIDTH - 1 : 0] rd,

    // Loaded GP Register Value
    input   logic [IMM_WIDTH - 1 : 0]           imm_in,
    output  logic [INT_DATA_WIDTH - 1 : 0]      gp_out_1,
    output  logic [INT_DATA_WIDTH - 1 : 0]      gp_out_2,
    // Resolved gp[rd] (write destination) for out-of-place vector-element writes.
    output  logic [INT_DATA_WIDTH - 1 : 0]      gp_out_3,

    // FP Value input
    input   logic [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH : 0]                external_fp_in,
    input   logic                                                       external_fp_in_valid,
    input   logic [FP_OPERAND_WIDTH - 1 : 0]                            external_fp_wtarget,
    output  logic [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH : 0]                fp_out,
    output  logic [VLEN - 1 : 0] [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH : 0] fp_vector_out,
    output  logic                                                       fp_vector_out_valid,

    // Stall Detection
    output  logic received_v_reduct_result,
    output  logic fp_stall_req,
    output  logic fp_sram_stall_req,

    // Loop Control
    output  logic loop_counter_zero  // From int_alu: loop counter reached 0
);

    import pipeline_pkg::*;
    import configuration_pkg::*;
    localparam FP_SRAM_ADDR_WIDTH       = $clog2(FP_SRAM_DEPTH);
    localparam INT_SRAM_ADDR_WIDTH      = $clog2(INT_SRAM_DEPTH);
    localparam VLEN_COUNTER_WIDTH       = $clog2(VLEN);

    //----------------------------//
    // FP Unit
    //----------------------------//

    S_FP_OP fp_control, exe_fp_control;
    logic [FP_OPERAND_WIDTH - 1 : 0] fp_rs1;
    logic [FP_OPERAND_WIDTH - 1 : 0] fp_rs2;
    logic [FP_OPERAND_WIDTH - 1 : 0] fp_rd, p1_fp_rd;
    logic [FP_OPERAND_WIDTH - 1 : 0] fp_reg_addr_1, fp_reg_addr_2;
    logic fp_reg_we;
    logic general_fp_operation;
    logic [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH : 0] fp_reg_1, fp_reg_2, fp_alu_out, fp_sfu_out, fp_reg_wdata, fp_ld_from_sram;
    logic [FP_OPERAND_WIDTH - 1 : 0] recorded_fp_waddr_sfu, recorded_fp_waddr_alu, recorded_fp_waddr_sram;
    logic fp_sfu_valid;
    logic load_fp_sram_valid, fp_sram_req, fp_sram_wen;
    logic [VLEN_COUNTER_WIDTH : 0] acc_vec_counter;
    logic continuous_load_fp_sram;
    logic write_data_from_external_fp;
    logic fp_alu_valid;
    logic [FP_SRAM_ADDR_WIDTH - 1 : 0] fp_sram_addr, recorded_fp_sram_addr;
    logic [MLEN - 1 : 0] [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH : 0] fp_vector_buffer;

    assign  fp_control = exe_stage_op.s_fp_op;

    // ------------------- Tracing Register for Stall Detection -------------------
    localparam int TRACE_SIZE = 2 << FP_OPERAND_WIDTH; // Number of FP registers
    logic [FP_OPERAND_WIDTH - 1 : 0] fp_wtarget;

    assign fp_vector_out = fp_vector_buffer;

    always_ff @(posedge clk) begin
        if (rst) begin
            exe_fp_control          <= STALL_S_FP;
            load_fp_sram_valid      <= 1'b0;
            recorded_fp_waddr_sram  <= 'b0;
            p1_fp_rd                <= 'b0;
            fp_stall_req            <= 1'b0;
            fp_sram_stall_req       <= 1'b0;
            fp_vector_buffer        <= 'b0;
            continuous_load_fp_sram <= 1'b0;
            acc_vec_counter         <= 'b0;
            fp_out                  <= 'b0;
            fp_sram_addr            <= 'b0;
            fp_sram_req             <= 1'b0;
            fp_sram_wen             <= 1'b0;

        end else begin

            p1_fp_rd                <= fp_rd;
            exe_fp_control          <= fp_control;
            load_fp_sram_valid      <= (exe_fp_control == LD_REG_FP) ? 1'b1 : 1'b0;
            recorded_fp_waddr_sram  <= p1_fp_rd;

            // Stall the pipeline while ANY scalar FP compute op is in flight, so a
            // dependent FP op cannot issue and read a stale register before the result
            // is written back. Both fp_alu (ADD/SUB/MUL) and fp_sfu (SQRT/RECI/EXP) are
            // non-pipelined (single op via *_in_use); without this, a back-to-back
            // dependent op (e.g. S_MUL_FP -> S_ADD_FP, or S_SQRT_FP -> S_RECI_FP) is
            // dropped by the busy unit and reads the pre-op register value.
            if (fp_control == ADD_FP || fp_control == SUB_FP || fp_control == MUL_FP ||
                fp_control == SQRT_FP || fp_control == RECI_FP || fp_control == EXP_FP ||
                fp_control == MAX_FP) begin
                fp_stall_req <= 1'b1; // ALU/SFU is busy with a compute op
            end else if (fp_alu_valid || fp_sfu_valid) begin
                fp_stall_req <= 1'b0; // compute op completed and written back
            end

            if (fp_control == MAP_V_FP) begin
                continuous_load_fp_sram <= 1'b1;
                recorded_fp_sram_addr   <= exe_stage_op.addr_1[FP_SRAM_ADDR_WIDTH - 1 : 0];
                fp_sram_addr            <= exe_stage_op.addr_1[FP_SRAM_ADDR_WIDTH - 1 : 0];
                acc_vec_counter         <= 'b1;
                fp_sram_req             <= 1'b1;
                fp_sram_stall_req       <= 1'b1;
                fp_vector_out_valid     <= 1'b0;  
            end else if (acc_vec_counter == VLEN + 1) begin
                acc_vec_counter         <=  'b0;
                continuous_load_fp_sram <= 1'b0;
                fp_sram_req             <= 1'b0;
                fp_vector_buffer[acc_vec_counter - 2]   <= fp_ld_from_sram;
                fp_vector_out_valid     <= 1'b1;
            end else if (continuous_load_fp_sram) begin
                acc_vec_counter <= acc_vec_counter + 1'b1;
                fp_sram_addr    <= recorded_fp_sram_addr + acc_vec_counter;
                if (acc_vec_counter > 'b1) begin
                    fp_vector_buffer[acc_vec_counter - 2]   <= fp_ld_from_sram;
                end
                fp_sram_req                                 <= 1'b1;
            end else if (fp_vector_out_valid) begin
                fp_sram_stall_req           <= 1'b0;
                fp_vector_buffer            <= 'b0;
                fp_vector_out_valid         <= 1'b0;
            end else begin
                fp_sram_addr <= exe_stage_op.addr_1[FP_SRAM_ADDR_WIDTH - 1 : 0];
                fp_sram_req  <= (fp_control == LD_REG_FP) || (fp_control == ST_REG_FP);
            end

            fp_sram_wen <= (fp_control == ST_REG_FP);

            // Loading fp reg data out.
            if (exe_fp_control == LD_OUT_FP) begin
                if ((fp_reg_addr_2 == fp_wtarget) & (fp_reg_addr_2 != 'b0)) begin
                    // Forwarding
                    fp_out <= fp_reg_wdata;
                end else begin
                    fp_out <= fp_reg_2;
                end
            end else begin
                fp_out <= 'b0;
            end
        end
    end

    /*
    Note: There is a case that fp_reg might be written from fp_alu and fp_sram at the same time, need to implement stall logic to prevent this.
    */

    always_comb begin
        if (fp_alu_valid) begin
            // From ALU
            fp_reg_we       = 1'b1;
            fp_reg_wdata    = fp_alu_out;
            fp_wtarget      = recorded_fp_waddr_alu;
            write_data_from_external_fp = 1'b0;
        // Must be else-if: as a second independent if-chain, the final else
        // overrode fp_reg_we back to 0 whenever only fp_alu_valid was high -
        // every FP ALU result was silently dropped before reaching the regfile.
        end else if (fp_sfu_valid) begin
            fp_reg_we       = 1'b1;
            fp_reg_wdata    = fp_sfu_out;
            fp_wtarget      = recorded_fp_waddr_sfu;
            write_data_from_external_fp = 1'b0;                
        end else if (load_fp_sram_valid) begin
            fp_reg_we       = 1'b1;
            fp_reg_wdata    = fp_ld_from_sram;
            fp_wtarget      = recorded_fp_waddr_sram;
            write_data_from_external_fp = 1'b0;
        end else if (external_fp_in_valid) begin
            fp_reg_we       = 1'b1;
            fp_reg_wdata    = external_fp_in;
            fp_wtarget      = external_fp_wtarget;
            write_data_from_external_fp = 1'b1;
        end else begin
            fp_reg_we       = 1'b0;
            fp_reg_wdata    = 'b0;
            fp_wtarget      = 'b0;
            write_data_from_external_fp = 1'b0;
        end
    end

    assign fp_rs1 = exe_stage_op.fps1;
    assign fp_rs2 = exe_stage_op.fps2;
    assign fp_rd  = exe_stage_op.fpd;
    assign fp_reg_addr_1 = (fp_control == ST_REG_FP) ? fp_rd : fp_rs1;
    assign fp_reg_addr_2 = fp_rs2;
    assign received_v_reduct_result = external_fp_in_valid & write_data_from_external_fp;

    fp_alu #(
        .EXP_WIDTH(S_FP_EXP_WIDTH),
        .MANT_WIDTH(S_FP_MANT_WIDTH)
    ) fp_alu_init (
        .clk                (clk),
        .rst                (rst),
        .operation          (exe_fp_control),
        .reg_waddr          (p1_fp_rd),
        .stored_reg_waddr   (recorded_fp_waddr_alu),
        .data_a             (fp_reg_1),
        .data_b             (fp_reg_2),
        .data_out           (fp_alu_out),
        .data_out_valid     (fp_alu_valid)
    );

    fp_sfu #(
        .EXP_WIDTH  (S_FP_EXP_WIDTH),
        .MANT_WIDTH (S_FP_MANT_WIDTH)      
    ) fp_sfu_init (
        .clk                (clk),
        .rst                (rst),
        .data_in            (fp_reg_1),
        .reg_waddr          (p1_fp_rd),
        // Use the registered op (exe_fp_control), matching fp_alu. The regfile read
        // is registered (1-cycle latency), so fp_reg_1 / p1_fp_rd are 1 cycle behind
        // fp_control; driving the SFU with combinational fp_control latched the op one
        // cycle before its operand/waddr were valid -> sqrt(0)=0 written to reg 0.
        .operation          (exe_fp_control),
        .data_out           (fp_sfu_out),
        .stored_reg_waddr   (recorded_fp_waddr_sfu),
        .data_out_valid     (fp_sfu_valid)
    );

    regfile_2p1w #(
        .BITWIDTH(S_FP_EXP_WIDTH + S_FP_MANT_WIDTH + 1),
        .DEPTH(1 << FP_OPERAND_WIDTH)
        `ifdef SIMULATION
        , .ResultFile(FP_REG_RESULT_FILE)
        `endif
    ) fp_reg_file (
        .clk        (clk),
        .we         (fp_reg_we),
        .waddr      (fp_wtarget),
        .wdata      (fp_reg_wdata),
        .raddr1     (fp_reg_addr_1),
        .raddr2     (fp_reg_addr_2),
        .raddr3     ('0),            // 3rd read port unused by the FP reg file
        .rdata1     (fp_reg_1),
        .rdata2     (fp_reg_2),
        .rdata3     ()
    );
    

    // SRAM for FP
    scalar_sram #(
        .DATA_WIDTH(FP_SRAM_WIDTH),
        .DEPTH(FP_SRAM_DEPTH)
        `ifdef SIMULATION
        , .MemInitFile(FP_MEM_INIT_FILE)
        `endif
    ) fp_scalar_sram (
        .clk            (clk),
        .rst            (rst),
        .req            (fp_sram_req),
        .write_en       (fp_sram_wen),
        .sram_addr      (fp_sram_addr),
        .sram_data_in   (fp_reg_1),
        .sram_data_out  (fp_ld_from_sram)
    );

    //----------------------------//
    // INT Unit
    //----------------------------//

    logic [INT_DATA_WIDTH - 1 : 0] gp_reg_1, gp_reg_2, gp_alu_out, gp_reg_wdata, gp_ld_from_sram, recorded_alu_out, computed_address;
    logic [INT_DATA_WIDTH - 1 : 0] gp_loaded_reg_1, gp_loaded_reg_2, gp_loaded_reg_3, gp_reg_3;
    logic gp_reg_wen, gp_write_from_sram_req, p1_gp_write_from_sram_req, gp_alu_valid;
    logic [INT_OPERAND_WIDTH - 1 : 0] gp_reg_waddr, recorded_gp_reg_exe_waddr, p1_recorded_gp_reg_exe_waddr;
    S_INT_OP exe_gp_op;
    logic [INT_OPERAND_WIDTH - 1 : 0] p1_rd, p1_rs1, p1_rs2, p2_rd;
    logic [IMM_WIDTH - 1 : 0] recorded_imm_in;
    logic [INT_OPERAND_WIDTH - 1 : 0] gp_reg_addr_1, gp_reg_addr_2, gp_reg_addr_3;
    logic [INT_OPERAND_WIDTH - 1 : 0] gp_reg_2_fwd_addr;  // operand the port-2 forward compares

    always_comb begin
        if (p1_gp_write_from_sram_req) begin
            gp_reg_waddr = p1_recorded_gp_reg_exe_waddr;
            gp_reg_wdata = gp_ld_from_sram;
            gp_reg_wen   = 1'b1;
        end  else begin
            gp_reg_waddr = p2_rd;
            gp_reg_wdata = gp_alu_out;
            gp_reg_wen   = gp_alu_valid;
        end
        gp_reg_1 = ((p1_rs1 == p2_rd) & gp_reg_wen) ? gp_reg_wdata : gp_loaded_reg_1;
        // Port-2 forward must compare the operand actually read (rd for these ops, per gp_reg_addr_2), not rs2.
        if ((exe_gp_op == PASS_ADDR_2) || (exe_gp_op == ST_INT) || (exe_gp_op == MAP_V_FP)) begin
            gp_reg_2_fwd_addr = p1_rd;
        end else begin
            gp_reg_2_fwd_addr = p1_rs2;
        end

        if ((gp_reg_2_fwd_addr == p2_rd) & gp_reg_wen) begin
            gp_reg_2 = gp_reg_wdata;      // forward the in-flight write-back
        end else begin
            gp_reg_2 = gp_loaded_reg_2;   // use the register-file read
        end

        // Port 3 always reads gp[rd] (the write destination); same level-1 forward.
        if ((p1_rd == p2_rd) & gp_reg_wen) begin
            gp_reg_3 = gp_reg_wdata;
        end else begin
            gp_reg_3 = gp_loaded_reg_3;
        end

    end

    always_ff @(posedge clk) begin
        if (rst) begin
            recorded_gp_reg_exe_waddr       <= 'b0;
            gp_write_from_sram_req          <= 1'b0;
            p1_gp_write_from_sram_req       <= 1'b0;
            p1_recorded_gp_reg_exe_waddr    <= 'b0;
            exe_gp_op                       <= STALL_S_INT;
            p1_rd                           <= 'b0;
            p2_rd                           <= 'b0;
            p1_rs1                          <= 'b0;
            p1_rs2                          <= 'b0;
            recorded_alu_out                <= 'b0;
            gp_out_1                        <= 'b0;
            gp_out_2                        <= 'b0;
            gp_out_3                        <= 'b0;
            recorded_imm_in                 <= 'b0;

        end else begin
            exe_gp_op                    <= assigned_int_op;
            recorded_imm_in              <= imm_in;
            // Always capture rd - the write enable (gp_alu_valid) will gate actual writes
            // Conditional capture caused pipeline misalignment with LOOP_DEC operations
            p1_rd                        <= rd;
            p2_rd                        <= p1_rd;
            p1_rs1                       <= rs1;
            p1_rs2                       <= rs2;
            p1_gp_write_from_sram_req    <= gp_write_from_sram_req;
            p1_recorded_gp_reg_exe_waddr <= recorded_gp_reg_exe_waddr;

            if (assigned_int_op == LD_INT) begin
                recorded_gp_reg_exe_waddr    <= rd;
                gp_write_from_sram_req       <= 1'b1;
            end else begin
                gp_write_from_sram_req       <= 1'b0;
            end

            if (exe_gp_op == PASS_ADDR) begin
                if ((gp_reg_waddr == p1_rs1 || gp_reg_waddr == p1_rs2) & gp_reg_wen) begin
                    gp_out_1 <= (gp_reg_waddr == p1_rs1) ? gp_reg_wdata : gp_reg_1;
                    gp_out_2 <= (gp_reg_waddr == p1_rs2) ? gp_reg_wdata : gp_reg_2;
                end else begin
                    gp_out_1 <= gp_reg_1;
                    gp_out_2 <= gp_reg_2;
                end
                // Write destination gp[rd] (for an out-of-place vector-element op).
                gp_out_3 <= ((gp_reg_waddr == p1_rd) & gp_reg_wen) ? gp_reg_wdata : gp_reg_3;
            end else if (exe_gp_op == PASS_ADDR_2) begin
                if ((gp_reg_waddr == p1_rd || gp_reg_waddr == p1_rs1) & gp_reg_wen) begin
                    gp_out_1 <= (gp_reg_waddr == p1_rs1)  ? gp_reg_wdata : gp_reg_1;
                    gp_out_2 <= (gp_reg_waddr == p1_rd)   ? gp_reg_wdata : gp_reg_2;
                end else begin
                    gp_out_1 <= gp_reg_1;
                    gp_out_2 <= gp_reg_2;
                end
                gp_out_3 <= ((gp_reg_waddr == p1_rd) & gp_reg_wen) ? gp_reg_wdata : gp_reg_3;
            end else if (exe_gp_op == COMP_ADDR) begin
                gp_out_1                 <= computed_address;
                gp_out_2                 <= 'b0;
                gp_out_3                 <= gp_out_3;
            end else if (exe_gp_op == COMP_ADDR_2) begin
                gp_out_1                 <= computed_address;
                gp_out_2                 <= gp_reg_2;
                gp_out_3                 <= gp_out_3;
            end else begin
                // Hold the last PASS/COMP result instead of zeroing: the
                // downstream addr merge may sample one cycle late around
                // stall boundaries and must still see the captured value.
                gp_out_1                 <= gp_out_1;
                gp_out_2                 <= gp_out_2;
                gp_out_3                 <= gp_out_3;
            end
        end
    end

    assign gp_reg_addr_1 = rs1;
    assign gp_reg_addr_2 = ((assigned_int_op == PASS_ADDR_2) || (assigned_int_op == ST_INT) || (assigned_int_op == MAP_V_FP)) ? rd : rs2;
    // Port 3 unconditionally reads gp[rd] so a vector-element op that writes
    // out-of-place (rd is neither read operand) still resolves its write address.
    assign gp_reg_addr_3 = rd;

    int_alu #(
        .BITWIDTH(INT_DATA_WIDTH)
    ) int_alu_init (
        .clk                (clk),
        .rst                (rst),
        .operand_a          (gp_reg_1),
        .operand_b          (gp_reg_2),
        .imm_value          ({{(INT_DATA_WIDTH - IMM_WIDTH){1'b0}}, recorded_imm_in}),
        .operation          (exe_gp_op),
        .result_valid       (gp_alu_valid),
        .computed_address   (computed_address),
        .result             (gp_alu_out),
        .loop_counter_zero  (loop_counter_zero)
    );

    regfile_2p1w #(
        .BITWIDTH(INT_DATA_WIDTH),
        .DEPTH(1 << INT_OPERAND_WIDTH)
    ) gp_reg_file (
        .clk        (clk),
        .we         (gp_reg_wen),
        .waddr      (gp_reg_waddr),
        .wdata      (gp_reg_wdata),
        .raddr1     (gp_reg_addr_1),
        .raddr2     (gp_reg_addr_2),
        .raddr3     (gp_reg_addr_3),
        .rdata1     (gp_loaded_reg_1),
        .rdata2     (gp_loaded_reg_2),
        .rdata3     (gp_loaded_reg_3)
    );

    scalar_sram #(
        .DATA_WIDTH     (INT_SRAM_WIDTH),
        .DEPTH          (INT_SRAM_DEPTH)
        `ifdef SIMULATION
            ,
            .MemInitFile(INT_MEM_INIT_FILE)
        `endif
    ) int_scalar_sram (
        .clk(clk),
        .rst(rst),
        .req            ((exe_gp_op == LD_INT) || (exe_gp_op == ST_INT)),
        .write_en       ((exe_gp_op == ST_INT)),
        .sram_addr      (computed_address[INT_SRAM_ADDR_WIDTH - 1 : 0]),
        .sram_data_in   (gp_loaded_reg_2),
        .sram_data_out  (gp_ld_from_sram)
    );

endmodule