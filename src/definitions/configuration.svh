`ifndef CONFIGURATION_SVH
`define CONFIGURATION_SVH
`include "global_define.vh"
`include "precision.svh"

import precision_pkg::*;

package configuration_pkg;
    // Compute Unit Related
    localparam   BLEN = 4; // 4
    localparam   HLEN = 8;
    localparam   MLEN = 8; // 64
    localparam   VLEN = 8; // 64
    localparam   INST_BUFF_DEPTH = 32;
    localparam   ON_CHIP_ADDR_WIDTH = precision_pkg::INT_DATA_WIDTH;
    // SourceWidth must satisfy: SourceWidth + SubbeatBits <= DeviceSourceWidth
    // where SubbeatBits = $clog2(HBM_WIDTH/32) for 32-bit host bus
    // Formula: base (2) + $clog2(HBM_WIDTH/32) + margin (2)
    localparam   SourceWidth = 4 + $clog2((precision_pkg::WT_MX_MANT_WIDTH + precision_pkg::WT_MX_EXP_WIDTH + 1) * MLEN / 16);  // [original: 5, tested: 6]
    localparam   SinkWidth = 1;
    // Memory Related
    localparam   MATRIX_SRAM_WIDTH = (precision_pkg::WT_MX_MANT_WIDTH + precision_pkg::WT_MX_EXP_WIDTH + 1 + precision_pkg::MX_SCALE_WIDTH) * MLEN;
    localparam   MATRIX_SRAM_DEPTH = 1024; // must be > 2x MLEN
    localparam   VECTOR_SRAM_WIDTH = (precision_pkg::V_FP_MANT_WIDTH + precision_pkg::V_FP_EXP_WIDTH + 1) * VLEN;
    localparam   VECTOR_SRAM_DEPTH = 1024; // must be > HEAD_DIM + HIDDEN_DIM/VLEN
    localparam   VECTOR_RESET_AMOUNT = 8;            // Need to be the same as Head_Dim for assembly code.
    localparam   INT_SRAM_WIDTH      = precision_pkg::INT_DATA_WIDTH;
    localparam   INT_SRAM_DEPTH      = 32;
    localparam   FP_SRAM_WIDTH       = (precision_pkg::S_FP_MANT_WIDTH + precision_pkg::S_FP_EXP_WIDTH + 1);
    localparam   FP_SRAM_DEPTH       = 512;
    localparam   HBM_ADDR_WIDTH      = 128;
    localparam   PC_ADDR_WIDTH       = 20;

    // Loop Control Related
    localparam   MAX_LOOP_DEPTH      = 4;  // Maximum nesting depth for hardware loops

    // HBM Related (calculated from precision parameters)
    localparam   HBM_M_Prefetch_Amount   = MLEN;
    localparam   HBM_V_Prefetch_Amount   = 4;
    localparam   HBM_V_Writeback_Amount  = 4;
    // Element width: (mant + exp + sign) * MLEN
    localparam   HBM_ELE_WIDTH_RAW       = (precision_pkg::WT_MX_MANT_WIDTH + precision_pkg::WT_MX_EXP_WIDTH + 1) * MLEN;
    localparam   HBM_ELE_WIDTH           = (1 << $clog2(HBM_ELE_WIDTH_RAW * 2));
    localparam   HBM_SCALE_WIDTH         = precision_pkg::MX_SCALE_WIDTH * (MLEN / precision_pkg::BLOCK_DIM);
    localparam   HBM_WIDTH               = HBM_ELE_WIDTH;
    // TileLink->AXI4 HBM bridge enable. 0 = HBM interface stays raw TileLink (default,
    // current behavior); 1 = route the four HBM TileLink host ports through the
    // tl_to_axi4 burst bridge and expose AXI4 master ports at the FPGA top.
    // `parameter` (not localparam) so an FPGA flow can override it (-G / defparam).
    parameter    HBM_AXI_BRIDGE_EN       = 0;
    localparam   INSTRUCTION_STORAGE_OFFSET = 32'h3840;  // Byte offset computed by workload generator = 14400
endpackage

package instruction_pkg;
    localparam INT_OPERAND_WIDTH     = 4;
    localparam FP_OPERAND_WIDTH      = 3;
    localparam HBM_ADR_OPERAND_WIDTH = 3;
    localparam STRIDE_OPERAND_WIDTH  = 3;
    localparam OPERAND_WIDTH         = 4;
    localparam FUNCT_WIDTH           = 4;
    localparam OPCODE_WIDTH          = 6;
    localparam IMM_WIDTH             = 22;
    localparam IMM_2_WIDTH           = 18;
    localparam INSTRUCTION_LENGTH    = 32;
endpackage

package simulation_pkg;
    // Word-address width of each port of the simulation-only fake HBM model
    // (fake_hbm_5port). Every port truncates its byte address to this many words,
    // so the 2-byte scale port can only reach 2^N * 2 bytes of the image: at 16 bits
    // any workload whose hbm.mem exceeds 128 KiB read garbage scales (e.g.
    // silu_down 8x128x256, linear --out-features 512). 20 bits = 2 MiB reach.
    localparam   FAKE_HBM_ADDR_WIDTH             = 20;
endpackage

`ifdef DC_LIB_EN // Define for DC Library Enabled, the pipeline stage lib changed accordingly.

    package pipeline_pkg;
        localparam   MAX_PIPELINE_STAGE             = 10;
        localparam   SYSTOLIC_PROCESSING_OVERHEAD   = 8;
        localparam   VECTOR_LONGEST_OPERATE_CYCLES  = 10;
    endpackage

`else

    package pipeline_pkg;
        localparam   MAX_PIPELINE_STAGE             = 10;
        localparam   SYSTOLIC_PROCESSING_OVERHEAD   = 8;
        localparam   VECTOR_LONGEST_OPERATE_CYCLES  = 30;
    endpackage

`endif

`endif