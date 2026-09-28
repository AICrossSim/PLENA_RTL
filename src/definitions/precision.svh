`ifndef PRECISION_SVH
`define PRECISION_SVH

/*
Module      : Definitions for Precision
Description :
            : Here we assume KV precision is less or equal to WT precision.
*/


package precision_pkg;
    // HBM Storage Precision
    localparam   ACT_MXFP_MANT_WIDTH  = 3;
    localparam   ACT_MXFP_EXP_WIDTH   = 4;
    localparam   ACT_MX_INT_WIDTH     = 8;
    localparam   ACT_MX_SCALE_WIDTH = 8;
    localparam   ACT_MX_INT_ENABLE    = 1; 

    localparam   KV_MX_MANT_WIDTH     = 2;
    localparam   KV_MX_EXP_WIDTH      = 1;
    localparam   KV_MX_INT_WIDTH      = 8;
    localparam   KV_MX_SCALE_WIDTH    = 8;
    localparam   KV_MX_INT_ENABLE     = 1; 

    localparam   WT_MX_MANT_WIDTH     = 2;
    localparam   WT_MX_EXP_WIDTH      = 1;
    localparam   WT_MX_INT_ENABLE     = 1;
    localparam   WT_MX_INT_WIDTH      = 8;
    localparam   WT_MX_SCALE_WIDTH    = 8;

    // Shared MX scale width (all types use the same scale width)
    localparam   MX_SCALE_WIDTH       = 8;

    localparam   BLOCK_DIM = 8;
    // Per Unit Precision
    localparam   V_FP_EXP_WIDTH  = 6;  // [original: 6, tested: 8]
    localparam   V_FP_MANT_WIDTH = 5;  // [original: 5, tested: 8]
    localparam   M_FP_EXP_WIDTH  = 6;  // [original: 6, tested: 8]
    localparam   M_FP_MANT_WIDTH = 5;  // [original: 5, tested: 9]
    localparam   S_FP_EXP_WIDTH  = 6;  // [original: 6, tested: 8]
    localparam   S_FP_MANT_WIDTH = 5;  // [original: 5, tested: 9]
    localparam   INT_DATA_WIDTH  = 32;
    // Compute Related Precision
    localparam   PRODUCT_EXT_EXP_WIDTH       = 0;
    localparam   PRODUCT_EXT_MANT_WIDTH      = 0;  // [original: 0, tested: 1]
    localparam   BLOCK_ADD_EXT_EXP_WIDTH     = 1;
    localparam   BLOCK_ADD_EXT_MANT_WIDTH    = 0;  // [original: 0, tested: 1]
    localparam   FP_ADD_EXT_EXP_WIDTH        = 1;
    localparam   FP_ADD_EXT_MANT_WIDTH       = 0;  // [original: 0, tested: 1]
    localparam   ROUND_FP_EXP_WIDTH          = 4;  // [original: 4, tested: 5]
    localparam   ROUND_FP_MANT_WIDTH         = 3;  // [original: 3, tested: 8]
endpackage

`endif