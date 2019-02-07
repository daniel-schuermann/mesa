/*
 * Copyright © 2015 Red Hat
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
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "nir.h"
#include "nir_builder.h"

/* Has two paths
 * One lowers idiv/udiv/umod and is based on NV50LegalizeSSA::handleDIV()
 *
 * Note that this path probably does not have not enough precision for
 * compute shaders. Perhaps we want a second higher precision (looping)
 * version of this? Or perhaps we assume if you can do compute shaders you
 * can also branch out to a pre-optimized shader library routine..
 *
 * The other path (enabled with use_urcp) requires nir_op_urcp and is
 * based off of code used by LLVM's AMDGPU target. It should handle 32-bit
 * idiv/irem/imod/udiv/umod exactly.
 */

static bool
convert_instr(nir_builder *bld, nir_alu_instr *alu)
{
   nir_ssa_def *numer, *denom, *af, *bf, *a, *b, *q, *r;
   nir_op op = alu->op;
   bool is_signed;

   if ((op != nir_op_idiv) &&
       (op != nir_op_udiv) &&
       (op != nir_op_imod) &&
       (op != nir_op_umod) &&
       (op != nir_op_irem))
      return false;

   is_signed = (op == nir_op_idiv ||
                op == nir_op_imod ||
                op == nir_op_irem);

   bld->cursor = nir_before_instr(&alu->instr);

   numer = nir_ssa_for_alu_src(bld, alu, 0);
   denom = nir_ssa_for_alu_src(bld, alu, 1);

   if (is_signed) {
      af = nir_i2f32(bld, numer);
      bf = nir_i2f32(bld, denom);
      af = nir_fabs(bld, af);
      bf = nir_fabs(bld, bf);
      a  = nir_iabs(bld, numer);
      b  = nir_iabs(bld, denom);
   } else {
      af = nir_u2f32(bld, numer);
      bf = nir_u2f32(bld, denom);
      a  = numer;
      b  = denom;
   }

   /* get first result: */
   bf = nir_frcp(bld, bf);
   bf = nir_isub(bld, bf, nir_imm_int(bld, 2));  /* yes, really */
   q  = nir_fmul(bld, af, bf);

   if (is_signed) {
      q = nir_f2i32(bld, q);
   } else {
      q = nir_f2u32(bld, q);
   }

   /* get error of first result: */
   r = nir_imul(bld, q, b);
   r = nir_isub(bld, a, r);
   r = nir_u2f32(bld, r);
   r = nir_fmul(bld, r, bf);
   r = nir_f2u32(bld, r);

   /* add quotients: */
   q = nir_iadd(bld, q, r);

   /* correction: if modulus >= divisor, add 1 */
   r = nir_imul(bld, q, b);
   r = nir_isub(bld, a, r);

   r = nir_uge(bld, r, b);
   r = nir_b2i32(bld, r);

   q = nir_iadd(bld, q, r);
   if (is_signed)  {
      /* fix the sign: */
      r = nir_ixor(bld, numer, denom);
      r = nir_ilt(bld, r, nir_imm_int(bld, 0));
      b = nir_ineg(bld, q);
      q = nir_bcsel(bld, r, b, q);

      if (op == nir_op_imod || op == nir_op_irem) {
         q = nir_imul(bld, q, denom);
         q = nir_isub(bld, numer, q);
         if (op == nir_op_imod) {
            q = nir_bcsel(bld, nir_ieq(bld, q, nir_imm_int(bld, 0)),
                          nir_imm_int(bld, 0),
                          nir_bcsel(bld, r, nir_iadd(bld, q, denom), q));
         }
      }
   }

   if (op == nir_op_umod) {
      /* division result in q */
      r = nir_imul(bld, q, b);
      q = nir_isub(bld, a, r);
   }

   assert(alu->dest.dest.is_ssa);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, nir_src_for_ssa(q));

   return true;
}

/* ported from LLVM's AMDGPUTargetLowering::LowerUDIVREM */
static nir_ssa_def *
emit_udiv(nir_builder *bld, nir_ssa_def *numer, nir_ssa_def *denom, bool modulo)
{
   nir_ssa_def *RCP = nir_urcp(bld, denom);
   nir_ssa_def *RCP_LO = nir_imul(bld, RCP, denom);
   nir_ssa_def *RCP_HI = nir_umul_high(bld, RCP, denom);
   nir_ssa_def *RCP_HI_NE_Zero = nir_ine(bld, RCP_HI, nir_imm_int(bld, 0));
   nir_ssa_def *NEG_RCP_LO = nir_ineg(bld, RCP_LO);
   nir_ssa_def *ABS_RCP_LO = nir_bcsel(bld, RCP_HI_NE_Zero, RCP_LO, NEG_RCP_LO);
   nir_ssa_def *E = nir_umul_high(bld, ABS_RCP_LO, RCP);
   nir_ssa_def *RCP_A_E = nir_iadd(bld, RCP, E);
   nir_ssa_def *RCP_S_E = nir_isub(bld, RCP, E);
   nir_ssa_def *Tmp0 = nir_bcsel(bld, RCP_HI_NE_Zero, RCP_S_E, RCP_A_E);
   nir_ssa_def *Quotient = nir_umul_high(bld, Tmp0, numer);
   nir_ssa_def *Num_S_Remainder = nir_imul(bld, Quotient, denom);
   nir_ssa_def *Remainder = nir_isub(bld, numer, Num_S_Remainder);
   nir_ssa_def *Remainder_GE_Den = nir_uge(bld, Remainder, denom);
   nir_ssa_def *Remainder_GE_Zero = nir_uge(bld, numer, Num_S_Remainder);
   nir_ssa_def *Tmp1 = nir_iand(bld, Remainder_GE_Den, Remainder_GE_Zero);

   if (modulo) {
      nir_ssa_def *Remainder_S_Den = nir_isub(bld, Remainder, denom);
      nir_ssa_def *Remainder_A_Den = nir_iadd(bld, Remainder, denom);
      nir_ssa_def *Rem = nir_bcsel(bld, Tmp1, Remainder_S_Den, Remainder);
      return nir_bcsel(bld, Remainder_GE_Zero, Rem, Remainder_A_Den);
   } else {
      nir_ssa_def *Quotient_A_One = nir_iadd(bld, Quotient, nir_imm_int(bld, 1));
      nir_ssa_def *Quotient_S_One = nir_isub(bld, Quotient, nir_imm_int(bld, 1));
      nir_ssa_def *Div = nir_bcsel(bld, Tmp1, Quotient_A_One, Quotient);
      return nir_bcsel(bld, Remainder_GE_Zero, Div, Quotient_S_One);
   }
}

/* ported from LLVM's AMDGPUTargetLowering::LowerSDIVREM */
static nir_ssa_def *
emit_idiv(nir_builder *bld, nir_ssa_def *numer, nir_ssa_def *denom, nir_op op)
{
   nir_ssa_def *LHSign = nir_ilt(bld, numer, nir_imm_int(bld, 0));
   nir_ssa_def *RHSign = nir_ilt(bld, denom, nir_imm_int(bld, 0));
   LHSign = nir_bcsel(bld, LHSign, nir_imm_int(bld, -1), nir_imm_int(bld, 0));
   RHSign = nir_bcsel(bld, RHSign, nir_imm_int(bld, -1), nir_imm_int(bld, 0));

   nir_ssa_def *LHS = nir_iadd(bld, numer, LHSign);
   nir_ssa_def *RHS = nir_iadd(bld, denom, RHSign);
   LHS = nir_ixor(bld, LHS, LHSign);
   RHS = nir_ixor(bld, RHS, RHSign);

   if (op == nir_op_idiv) {
      nir_ssa_def *DSign = nir_ixor(bld, LHSign, RHSign);
      nir_ssa_def *res = emit_udiv(bld, LHS, RHS, false);
      res = nir_ixor(bld, res, DSign);
      return nir_isub(bld, res, DSign);
   } else {
      nir_ssa_def *res = emit_udiv(bld, LHS, RHS, true);
      res = nir_ixor(bld, res, LHSign);
      res = nir_isub(bld, res, LHSign);
      if (op == nir_op_imod) {
         nir_ssa_def *cond = nir_ieq(bld, res, nir_imm_int(bld, 0));
         cond = nir_ior(bld, nir_ieq(bld, LHSign, RHSign), cond);
         res = nir_bcsel(bld, cond, res, nir_iadd(bld, res, denom));
      }
      return res;
   }
}

static bool
convert_instr_urcp(nir_builder *bld, nir_alu_instr *alu)
{
   nir_op op = alu->op;

   if ((op != nir_op_idiv) &&
       (op != nir_op_imod) &&
       (op != nir_op_irem) &&
       (op != nir_op_udiv) &&
       (op != nir_op_umod))
      return false;

   if (alu->dest.dest.ssa.bit_size != 32)
      return false;

   bld->cursor = nir_before_instr(&alu->instr);

   nir_ssa_def *numer = nir_ssa_for_alu_src(bld, alu, 0);
   nir_ssa_def *denom = nir_ssa_for_alu_src(bld, alu, 1);

   nir_ssa_def *res = NULL;

   if (op == nir_op_udiv || op == nir_op_umod)
      res = emit_udiv(bld, numer, denom, op == nir_op_umod);
   else
      res = emit_idiv(bld, numer, denom, op);

   assert(alu->dest.dest.is_ssa);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, nir_src_for_ssa(res));

   return true;
}

static bool
convert_impl(nir_function_impl *impl, bool use_urcp)
{
   nir_builder b;
   nir_builder_init(&b, impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_alu && use_urcp)
            progress |= convert_instr_urcp(&b, nir_instr_as_alu(instr));
         else if (instr->type == nir_instr_type_alu)
            progress |= convert_instr(&b, nir_instr_as_alu(instr));
      }
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);

   return progress;
}

bool
nir_lower_idiv(nir_shader *shader, bool use_urcp)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= convert_impl(function->impl, use_urcp);
   }

   return progress;
}
