/*
 * Copyright © 2018 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Daniel Schürmann (daniel.schuermann@campus.tu-berlin.de)
 *
 */

#ifndef ACO_IR_H
#define ACO_IR_H

#include <stdbool.h>
#include <stdint.h>
#include "aco_opcodes.h"

typedef enum {
   MODE = (1 << 0),
   STATUS = (1 << 1),
   M0 = (1 << 2),
   TRAPSTS = (1 << 3),
   EXEC = (1 << 4),
   EXECZ = (1 << 5),
   VCC = (1 << 6),
   VCCZ = (1 << 7),
   SCC = (1 << 8),
   SCCZ = (1 << 9),
   PC = (1 << 10),
   IB_STS = (1 << 11),
   GPR_ALLOC = (1 << 12),
   LDS_ALLOC = (1 << 13),
} SPR;

typedef enum {
   b = 0,
   s1 = 1,
   s2 = 2,
   s3 = 3,
   s4 = 4,
   s8 = 8,
   s16 = 16,
   v1 = s1 << 5,
   v2 = s2 << 5,
   v3 = s3 << 5,
   v4 = s4 << 5,
} aco_data_type;

typedef enum {
   ssa_def,
   constant,
   sgpr,
   vgpr,
   spr,
} val_type;

typedef struct aco_src {

   //aco_instr *parent_instr;
   //struct list_head use_link;
   
   val_type type;
   
   /* the meaning of src_val depends on the type:
    * ssa_def: the ssa index
    * constant: the constant value
    * sgpr/vgpr: the start register number
    * spr: one of the spr's defined above */
   unsigned src_val;
   
   /* the number of sgpr/vgpr's needed for the value */
   unsigned size;

} aco_src;


typedef struct aco_dst {

   //aco_instr *parent_instr;
   //struct list_head use_link;

   val_type type;
   unsigned dst_val;
   unsigned size;
} aco_dst;

typedef enum aco_format {
   SOP2,
   SOPK,
   SOP1,
   SOPC,
   SOPP,
   SMEM,
   VOP2,
   VOP2_SDWA,
   VOP2_DPP,
   VOP1,
   VOP1_SDWA,
   VOP1_DPP,
   VOPC,
   VOPC_SDWA,
   VOPC_DPP,
   VOP3a,
   VOP3b,
   VINTRP,
   DS,
   MUBUF,
   MTBUF,
   MIMG,
   EXP,
   FLAT,
   num_formats,
} aco_format;

typedef struct aco_instr {
   //struct exec_node node;
   aco_opcode opcode;
   aco_format format;

   /** generic instruction index. */
   unsigned index;

   /** we have a maximum of 4 inputs and 2 outputs on every instruction 
    *  Check opcode_infos[opcode] for the actual numbers */
   aco_src src[4];
   aco_dst dst[2];
} aco_instr;

typedef struct aco_SOP2 {
   aco_instr instr;

   /* ssrc0 = src[0] */
   /* ssrc1 = src[1] */
   /* sdst = dst[0] */
   /* scc = dst[1] / src[2] */
} aco_SOP2;

typedef struct aco_SOPK {
   aco_instr instr;

   /* simm = src[0] */
   /* sdst = dst[0] */
   /* scc = dst[1] / src[1] */
   // addk & mulk use D = src[2]
} aco_SOPK;

typedef struct aco_SOP1 {
   aco_instr instr;

   /* ssrc0 = src[0] */
   /* sdst = dst[0] */
   /* scc = dst[1] */
} aco_SOP1;

typedef struct aco_SOPC {
   aco_instr instr;

   /* ssrc0 = src[0] */
   /* ssrc1 = src[1] */
   /* scc = dst[1] */
} aco_SOPC;

typedef struct aco_SOPP {
   aco_instr instr;

   /* simm src[0] */
   /* scc/vcc = src[1] */
} aco_SOPP;

typedef struct aco_SMEM {
   aco_instr instr;
   
   /* sbase = src[0] */
   /* offset = src[1] */
   /* sdata = src[2] / dst[0] */
   /* soffset = src[3] VEGA only */
   bool glc : 1;
   bool imm : 1;
} aco_SMEM;

typedef struct aco_VOP2 {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vdst = dst[0] */
   /* madmk: imm = src[2] */
   /* vcc = src[3] / dst[1] */
} aco_VOP2;

typedef struct aco_VOP2_SDWA {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vdst = dst[0] */
   /* vcc = dst[1] */
   unsigned dst_sel : 3;
   unsigned dst_unused : 2;
   bool clamp : 1;
   unsigned src0_sel : 3;
   bool src0_sext : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   unsigned src1_sel : 3;
   bool src1_sext : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
} aco_VOP2_SDWA;

typedef struct aco_VOP2_DPP {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vdst = dst[0] */
   /* vcc = dst[1] */
   unsigned dpp_ctrl : 9;
   bool bound_ctrl : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   unsigned bank_mask : 4;
   unsigned row_mask : 4;
} aco_VOP2_DPP;

typedef struct aco_VOP1 {
   aco_instr instr;

   /* src0 = src[0] */
   /* vdst = dst[0] */
   /* swap uses src[1] and dst[1] as well */
} aco_VOP1;

typedef struct aco_VOP1_SDWA {
   aco_instr instr;

   /* src0 = src[0] */
   /* vdst = dst[0] */
   unsigned dst_sel : 3;
   unsigned dst_unused : 2;
   bool clamp : 1;
   unsigned src0_sel : 3;
   bool src0_sext : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   unsigned src1_sel : 3;
   bool src1_sext : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
} aco_VOP1_SDWA;

typedef struct aco_VOP1_DPP {
   aco_instr instr;

   /* src0 = src[0] */
   /* vdst = dst[0] */
   unsigned dpp_ctrl : 9;
   bool bound_ctrl : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   unsigned bank_mask : 4;
   unsigned row_mask : 4;
} aco_VOP1_DPP;

typedef struct aco_VOPC {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vcc = dst[0] */
} aco_VOPC;

typedef struct aco_VOPC_SDWA {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vcc = dst[0] */
   unsigned dst_sel : 3;
   unsigned dst_unused : 2;
   bool clamp : 1;
   unsigned src0_sel : 3;
   bool src0_sext : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   unsigned src1_sel : 3;
   bool src1_sext : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
} aco_VOPC_SDWA;

typedef struct aco_VOPC_DPP {
   aco_instr instr;

   /* src0 = src[0] */
   /* vsrc1 = src[1] */
   /* vcc = dst[0] */
   unsigned dpp_ctrl : 9;
   bool bound_ctrl : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   unsigned bank_mask : 4;
   unsigned row_mask : 4;
} aco_VOPC_DPP;

typedef struct aco_VINTRP {
   aco_instr instr;

   /* vsrc = src[0] */
   /* vdst = dst[0] */
   /* prim_mask = src[1] ??? */
   unsigned attrchan : 2;
   unsigned attr : 6;
} aco_VINTRP;

typedef struct aco_VOP3a {
   aco_instr instr;

   /* src0 = src[0] */
   /* src1 = src[1] */
   /* src2 = src[2] */
   /* vdst = dst[0] */
   /* vcc = src[3]  */
   bool clamp : 1;
   bool abs[3];
   bool neg[3];
   unsigned omod : 2;
} aco_VOP3a;

typedef struct aco_VOP3b {
   aco_instr instr;

   /* src0 = src[0] */
   /* src1 = src[1] */
   /* src2 = src[2] */
   /* vdst = dst[0] */
   /* sdst = dst[1]  */
   bool clamp : 1;
   bool neg[3];
   unsigned omod : 2;
   
} aco_VOP3b;

typedef struct aco_DS {
   aco_instr instr;

   /* addr = src[0] */
   /* data0 = src[1] */
   /* data1 = src[2] */
   /* vdst = dst[0] */
   unsigned offset0 : 8;
   unsigned offset1 : 8;
   bool gds : 1;
} aco_DS;

typedef struct aco_MUBUF {
   aco_instr instr;

   /* vaddr = src[0] */
   /* soffset = src[1] */
   /* vdata = src[2] / dst[0] */
   /* srsrc = src[3] VEGA only */
   unsigned offset : 12;
   bool offen : 1;
   bool idxen : 1;
   bool glc : 1;
   bool lds : 1;
   bool slc : 1;
   bool tfe : 1;
} aco_MUBUF;

typedef struct aco_MTBUF {
   aco_instr instr;

   /* vaddr = src[0] */
   /* soffset = src[1] */
   /* vdata = src[2] / dst[0] */
   /* srsrc = src[3] VEGA only */
   unsigned offset : 12;
   bool offen : 1;
   bool idxen : 1;
   bool glc : 1;
   unsigned dfmt : 4;
   unsigned nfmt : 3;
   bool slc : 1;
   bool tfe : 1;
} aco_MTBUF;

typedef struct aco_MIMG {
   aco_instr instr;
   
   /* vaddr = src[0] */
   /* srsrc = src[1] */
   /* ssamp = src[2] */
   /* vdata = src[3] / dst[0] */
   unsigned dmask : 4;
   bool unorm : 1;
   bool glc : 1;
   bool da : 1;
   bool r128 : 1;
   bool tfe : 1;
   bool lwe : 1;
   bool slc : 1;
   bool d16 : 1;
} aco_MIMG;

typedef struct aco_EXP {
   aco_instr instr;

   /* vsrc0 = src[0] */
   /* vsrc1 = src[1] */
   /* vsrc2 = src[2] */
   /* vsrc3 = src[3] */
   unsigned en : 4;
   unsigned tgt : 6;
   bool compr : 1;
   bool done : 1;
   bool vm : 1;
} aco_EXP;

typedef struct aco_FLAT {
   aco_instr instr;

   /* addr = src[0] */
   /* data = src[1] */
   /* saddr = src[2] VEGA only */
   /* vdst = dst[0] */
   unsigned offset : 13;
   bool lds : 1; // VEGA
   unsigned seg : 2; // VEGA
   bool glc : 1;
   bool slc : 1;
   bool tfe : 1;
} aco_FLAT;

typedef struct {
   const char *name;
   unsigned num_inputs;
   unsigned num_outputs;
   aco_data_type output_type[2];
   SPR read_reg;
   SPR write_reg;
   bool kills_input[4];
   /* TODO: everything that depends on the instruction rather than the format */
   // like sideeffects on spr's

} opcode_info;

extern const opcode_info opcode_infos[num_opcodes];

typedef struct {

   /* Flags */
   // TODO: everything that depends only on the microcode format

   unsigned encoding;
} format_info;

extern const format_info format_infos[num_formats];

#endif /* ACO_IR_H */
