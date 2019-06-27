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

#include <bitset>
#include "aco_ir.h"

namespace aco {

/**
 * The optimizer works in 4 phases:
 * (1) The first pass collects information for each ssa-def,
 *     propagates reg->reg operands of the same type, inline constants
 *     and neg/abs input modifiers.
 * (2) The second pass combines instructions like mad, omod, clamp and
 *     propagates sgpr's on VALU instructions.
 *     This pass depends on information collected in the first pass.
 * (3) The third pass goes backwards, and selects instructions,
 *     i.e. decides if a mad instruction is profitable and eliminates dead code.
 * (4) The fourth pass cleans up the sequence: literals get applied and dead
 *     instructions are removed from the sequence.
 */


struct mad_info {
   aco_ptr<Instruction> add_instr;
   uint32_t mul_temp_id;
   uint32_t literal_idx;
   bool needs_vop3;
   bool check_literal;

   mad_info(aco_ptr<Instruction> instr, uint32_t id, bool vop3)
   : add_instr(std::move(instr)), mul_temp_id(id), needs_vop3(vop3), check_literal(false) {}
};

struct ssa_info {
   uint32_t val;
   union {
      Temp temp;
      Instruction* instr;
   };
   std::bitset<32> label;

   void set_vec(Instruction* vec)
   {
      label.set(0,1);
      instr = vec;
   }

   bool is_vec()
   {
      return label.test(0);
   }

   void set_constant(uint32_t constant)
   {
      label.set(1,1);
      val = constant;
   }

   bool is_constant()
   {
      return label.test(1);
   }

   void set_abs(Temp abs_temp)
   {
      label.set(2,1);
      temp = abs_temp;
   }

   bool is_abs()
   {
      return label.test(2);
   }

   void set_neg(Temp neg_temp)
   {
      label.set(3,1);
      temp = neg_temp;
   }

   bool is_neg()
   {
      return label.test(3);
   }

   void set_mul(Instruction* mul)
   {
      label.set(4,1);
      instr = mul;
   }

   bool is_mul()
   {
      return label.test(4);
   }

   void set_temp(Temp tmp)
   {
      label.set(5,1);
      temp = tmp;
   }

   bool is_temp()
   {
      return label.test(5);
   }

   void set_literal(uint32_t lit)
   {
      label.set(6,1);
      val = lit;
   }

   bool is_literal()
   {
      return label.test(6);
   }

   void set_mad(Instruction* mad, uint32_t mad_info_idx)
   {
      label.set(7,1);
      val = mad_info_idx;
      instr = mad;
   }

   bool is_mad()
   {
      return label.test(7);
   }

   void set_check_literal(uint32_t idx)
   {
      label.set(8, 1);
      val = idx;
   }

   int is_check_literal()
   {
      return label.test(8);
   }

   void set_omod2()
   {
      label.set(9,1);
   }

   bool is_omod2()
   {
      return label.test(9);
   }

   void set_omod4()
   {
      label.set(10);
   }

   bool is_omod4()
   {
      return label.test(10);
   }

   void set_omod5()
   {
      label.set(11,1);
   }

   bool is_omod5()
   {
      return label.test(11);
   }

   void set_omod_success(Instruction* omod_instr)
   {
      instr = omod_instr;
      label.set(12,1);
   }

   bool is_omod_success()
   {
      return label.test(12);
   }

   void set_clamp()
   {
      label.set(13,1);
   }

   bool is_clamp()
   {
      return label.test(13);
   }

   void set_clamp_success(Instruction* clamp_instr)
   {
      instr = clamp_instr;
      label.set(14,1);
   }

   bool is_clamp_success()
   {
      return label.test(14);
   }

   void set_undefined()
   {
      label.set(15,1);
   }

   bool is_undefined()
   {
      return label.test(15);
   }

   void set_vcc(Temp vcc)
   {
      label.set(16,1);
      temp = vcc;
   }

   bool is_vcc()
   {
      return label.test(16);
   }

   bool is_constant_or_literal()
   {
      return is_constant() || is_literal();
   }

   void set_base_offset(Temp base, uint32_t offset)
   {
      label.set(17,1);
      temp = base;
      val = offset;
   }

   bool is_base_offset()
   {
      return label.test(17);
   }

   void set_b2f(Temp val)
   {
      label.set(18,1);
      temp = val;
   }

   bool is_b2f()
   {
      return label.test(18);
   }

};

struct opt_ctx {
   Program* program;
   std::vector<aco_ptr<Instruction>> instructions;
   ssa_info* info;
   std::pair<uint32_t,Temp> last_literal;
   std::vector<mad_info> mad_infos;
   std::vector<uint16_t> uses;
};

bool can_swap_operands(aco_ptr<Instruction>& instr)
{
   if (instr->getOperand(0).isConstant() ||
       (instr->getOperand(0).isTemp() && instr->getOperand(0).getTemp().type() == sgpr))
      return false;

   switch (instr->opcode) {
   case aco_opcode::v_add_f32:
   case aco_opcode::v_mul_f32:
   case aco_opcode::v_or_b32:
   case aco_opcode::v_and_b32:
   case aco_opcode::v_xor_b32:
   case aco_opcode::v_max_f32:
   case aco_opcode::v_min_f32:
   case aco_opcode::v_cmp_eq_f32:
   case aco_opcode::v_cmp_lg_f32:
      return true;
   case aco_opcode::v_sub_f32:
      instr->opcode = aco_opcode::v_subrev_f32;
      return true;
   case aco_opcode::v_cmp_lt_f32:
      instr->opcode = aco_opcode::v_cmp_gt_f32;
      return true;
   case aco_opcode::v_cmp_ge_f32:
      instr->opcode = aco_opcode::v_cmp_le_f32;
      return true;
   case aco_opcode::v_cmp_lt_i32:
      instr->opcode = aco_opcode::v_cmp_gt_i32;
      return true;
   default:
      return false;
   }
}

bool can_use_VOP3(aco_ptr<Instruction>& instr)
{
   if (instr->num_operands && instr->getOperand(0).isLiteral())
      return false;

   return instr->opcode != aco_opcode::v_madmk_f32 &&
          instr->opcode != aco_opcode::v_madak_f32 &&
          instr->opcode != aco_opcode::v_madmk_f16 &&
          instr->opcode != aco_opcode::v_madak_f16;
}

void to_VOP3(opt_ctx& ctx, aco_ptr<Instruction>& instr)
{
   if (instr->isVOP3())
      return;

   assert(!instr->getOperand(0).isLiteral());
   aco_ptr<Instruction> tmp = std::move(instr);
   Format format = asVOP3(tmp->format);
   instr.reset(create_instruction<VOP3A_instruction>(tmp->opcode, format, tmp->num_operands, tmp->num_definitions));
   for (unsigned i = 0; i < instr->num_operands; i++)
      instr->getOperand(i) = tmp->getOperand(i);
   for (unsigned i = 0; i < instr->num_definitions; i++)
      instr->getDefinition(i) = tmp->getDefinition(i);
}

/* only covers special cases */
bool can_accept_constant(aco_ptr<Instruction>& instr, unsigned operand)
{
   switch (instr->opcode) {
   case aco_opcode::v_interp_p2_f32:
   case aco_opcode::v_mac_f32:
   case aco_opcode::v_writelane_b32:
      return operand != 2;
   case aco_opcode::s_addk_i32:
   case aco_opcode::s_mulk_i32:
   case aco_opcode::p_wqm:
      return operand != 0;
   default:
      if ((instr->format == Format::MUBUF ||
           instr->format == Format::MIMG) &&
          instr->num_definitions == 1 &&
          instr->num_operands == 4) {
         return operand != 3;
      }
      return true;
   }
}

void label_instruction(opt_ctx &ctx, aco_ptr<Instruction>& instr)
{

   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (!instr->getOperand(i).isTemp())
         continue;

      ssa_info info = ctx.info[instr->getOperand(i).tempId()];
      /* propagate undef */
      if (info.is_undefined())
         instr->getOperand(i) = Operand(instr->getOperand(i).regClass());
      /* propagate reg->reg of same type */
      if (info.is_temp() && info.temp.regClass() == instr->getOperand(i).getTemp().regClass()) {
         instr->getOperand(i).setTemp(ctx.info[instr->getOperand(i).tempId()].temp);
         info = ctx.info[info.temp.id()];
      }

      /* SALU / PSEUDO: propagate inline constants */
      if (instr->isSALU() || instr->format == Format::PSEUDO) {
         if (info.is_temp() && info.temp.type() == sgpr) {
            instr->getOperand(i).setTemp(info.temp);
            info = ctx.info[info.temp.id()];
         } else if (info.is_temp() && info.temp.type() == vgpr) {
            /* propagate vgpr if it can take it */
            switch (instr->opcode) {
            case aco_opcode::p_create_vector:
            case aco_opcode::p_split_vector:
            case aco_opcode::p_extract_vector:
            case aco_opcode::p_linear_phi:
            case aco_opcode::p_phi: {
               bool all_vgpr = true;
               for (unsigned i = 0; all_vgpr && i < instr->num_definitions; i++) {
                  if (instr->getDefinition(i).getTemp().type() != vgpr)
                     all_vgpr = false;
               }
               if (all_vgpr) {
                  instr->getOperand(i) = Operand(info.temp);
                  info = ctx.info[info.temp.id()];
               }
               break;
            }
            default:
               break;
            }
         }
         if ((info.is_constant() || (info.is_literal() && instr->format == Format::PSEUDO)) && !instr->getOperand(i).isFixed() && can_accept_constant(instr, i)) {
            instr->getOperand(i) = Operand(info.val);
            continue;
         }
      }

      /* VALU: propagate neg, abs & inline constants */
      else if (instr->isVALU()) {

         if (info.is_temp() && info.temp.type() == vgpr) {
            instr->getOperand(i).setTemp(info.temp);
            info = ctx.info[info.temp.id()];
         }
         if (info.is_abs() && can_use_VOP3(instr) && instr_info.can_use_input_modifiers[(int)instr->opcode]) {
            to_VOP3(ctx, instr);
            instr->getOperand(i) = Operand(info.temp);
            static_cast<VOP3A_instruction*>(instr.get())->abs[i] = true;
         }
         if (info.is_neg() && instr->opcode == aco_opcode::v_add_f32) {
            instr->opcode = i ? aco_opcode::v_sub_f32 : aco_opcode::v_subrev_f32;
            instr->getOperand(i).setTemp(info.temp);
            continue;
         } else if (info.is_neg() && can_use_VOP3(instr) && instr_info.can_use_input_modifiers[(int)instr->opcode]) {
            to_VOP3(ctx, instr);
            instr->getOperand(i) = Operand(info.temp);
            static_cast<VOP3A_instruction*>(instr.get())->neg[i] = true;
            continue;
         }
         if (info.is_constant() && can_accept_constant(instr, i)) {
            assert(i != 2 || instr->opcode != aco_opcode::v_cndmask_b32);
            if (i == 0) {
               instr->getOperand(i) = Operand(info.val);
               continue;
            } else if (!instr->isVOP3() && can_swap_operands(instr)) {
               instr->getOperand(i) = instr->getOperand(0);
               instr->getOperand(0) = Operand(info.val);
               continue;
            } else if (can_use_VOP3(instr)) {
               to_VOP3(ctx, instr);
               instr->getOperand(i) = Operand(info.val);
               continue;
            }
         }
      }

      /* MUBUF: propagate constants and combine additions */
      else if (instr->format == Format::MUBUF) {

         MUBUF_instruction *mubuf = static_cast<MUBUF_instruction *>(instr.get());
         if (mubuf->offen && i == 0 && info.is_constant_or_literal() && mubuf->offset + info.val < 4096) {
            assert(!mubuf->idxen);
            instr->getOperand(i) = Operand(v1);
            mubuf->offset += info.val;
            mubuf->offen = false;
            continue;
         } else if (i == 2 && info.is_constant_or_literal() && mubuf->offset + info.val < 4096) {
            instr->getOperand(2) = Operand((uint32_t) 0);
            mubuf->offset += info.val;
            continue;
         } else if (mubuf->offen && i == 0 && info.is_base_offset() && info.temp.regClass() == v1 && mubuf->offset + info.val < 4096) {
            assert(!mubuf->idxen);
            instr->getOperand(i).setTemp(info.temp);
            mubuf->offset += info.val;
            continue;
         } else if (i == 2 && info.is_base_offset() && info.temp.regClass() == s1 && mubuf->offset + info.val < 4096) {
            instr->getOperand(i).setTemp(info.temp);
            mubuf->offset += info.val;
            continue;
         }
      }

      /* DS: combine additions */
      else if (instr->format == Format::DS) {

         DS_instruction *ds = static_cast<DS_instruction *>(instr.get());
         if (i == 0 && info.is_base_offset() && info.temp.regClass() == instr->getOperand(i).regClass()) {
            Temp base = info.temp;
            uint32_t offset = info.val;
            if (instr->opcode == aco_opcode::ds_write2_b32 || instr->opcode == aco_opcode::ds_read2_b32) {
               if (offset % 4 == 0 &&
                   ds->offset0 + (offset >> 2) <= 255 &&
                   ds->offset1 + (offset >> 2) <= 255) {
                  instr->getOperand(i).setTemp(base);
                  ds->offset0 += offset >> 2;
                  ds->offset1 += offset >> 2;
               }
            } else {
               if (ds->offset0 + offset <= 65535) {
                  instr->getOperand(i).setTemp(base);
                  ds->offset0 += offset;
               }
            }
         }
      }

      /* SMEM: propagate constants and combine additions */
      else if (instr->format == Format::SMEM) {

         SMEM_instruction *smem = static_cast<SMEM_instruction *>(instr.get());
         if (i == 1 && info.is_constant_or_literal() && info.val <= 0xFFFFF) {
            instr->getOperand(i) = Operand(info.val);
            continue;
         } else if (i == 1 && info.is_base_offset() && info.temp.regClass() == s1 && info.val <= 0xFFFFF && ctx.program->chip_class >= GFX9) {
            bool soe = smem->num_operands >= (smem->num_definitions ? 3 : 4);
            if (soe &&
                (!ctx.info[smem->getOperand(smem->num_operands - 1).tempId()].is_constant_or_literal() ||
                 ctx.info[smem->getOperand(smem->num_operands - 1).tempId()].val != 0)) {
               continue;
            }
            if (soe) {
               smem->getOperand(1) = Operand(info.val);
               smem->getOperand(smem->num_operands - 1) = Operand(info.temp);
            } else {
               SMEM_instruction *new_instr = create_instruction<SMEM_instruction>(smem->opcode, Format::SMEM, smem->num_operands + 1, smem->num_definitions);
               new_instr->getOperand(0) = smem->getOperand(0);
               new_instr->getOperand(1) = Operand(info.val);
               if (!smem->num_definitions)
                  new_instr->getOperand(2) = smem->getOperand(2);
               new_instr->getOperand(new_instr->num_operands - 1) = Operand(info.temp);
               if (smem->num_definitions)
                  new_instr->getDefinition(0) = smem->getDefinition(0);
               instr.reset(new_instr);
               smem = static_cast<SMEM_instruction *>(instr.get());
            }
            continue;
         }
      }
   }

   /* if this instruction doesn't define anything, return */
   if (!instr->num_definitions)
      return;

   switch (instr->opcode) {
   case aco_opcode::p_create_vector: {
      unsigned num_ops = instr->num_operands;
      bool is_undefined = true;
      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (!instr->getOperand(i).isUndefined())
            is_undefined = false;
         if (instr->getOperand(i).isTemp() && ctx.info[instr->getOperand(i).tempId()].is_vec())
            num_ops += ctx.info[instr->getOperand(i).tempId()].instr->num_operands - 1;
      }
      if (is_undefined) {
         ctx.info[instr->getDefinition(0).tempId()].set_undefined();
      } else if (num_ops != instr->num_operands) {
         aco_ptr<Instruction> old_vec = std::move(instr);
         instr.reset(create_instruction<Pseudo_instruction>(aco_opcode::p_create_vector, Format::PSEUDO, num_ops, 1));
         instr->getDefinition(0) = old_vec->getDefinition(0);
         unsigned k = 0;
         for (unsigned i = 0; i < old_vec->num_operands; i++) {
            if (old_vec->getOperand(i).isTemp() && ctx.info[old_vec->getOperand(i).tempId()].is_vec()) {
               for (unsigned j = 0; j < ctx.info[old_vec->getOperand(i).tempId()].instr->num_operands; j++)
                  instr->getOperand(k++) = ctx.info[old_vec->getOperand(i).tempId()].instr->getOperand(j);
            } else {
               instr->getOperand(k++) = old_vec->getOperand(i);
            }
         }
         assert(k == num_ops);
      }
      if (instr->getDefinition(0).getTemp().size() == instr->num_operands)
         ctx.info[instr->getDefinition(0).tempId()].set_vec(instr.get());
      break;
   }
   case aco_opcode::p_split_vector: {
      if (instr->getOperand(0).isUndefined()) {
         for (unsigned i = 0; i < instr->num_definitions; i++)
            ctx.info[instr->getDefinition(i).tempId()].set_undefined();
         break;
      }
      if (!ctx.info[instr->getOperand(0).tempId()].is_vec())
         break;
      Instruction* vec = ctx.info[instr->getOperand(0).tempId()].instr;
      assert(instr->num_definitions == vec->num_operands);
      for (unsigned i = 0; i < instr->num_definitions; i++) {
         Operand vec_op = vec->getOperand(i);
         if (vec_op.isConstant()) {
            if (vec_op.isLiteral())
               ctx.info[instr->getDefinition(i).tempId()].set_literal(vec_op.constantValue());
            else
               ctx.info[instr->getDefinition(i).tempId()].set_constant(vec_op.constantValue());
         } else if (vec_op.isUndefined()) {
            ctx.info[instr->getDefinition(i).tempId()].set_undefined();
         } else {
            assert(vec_op.isTemp());
            ctx.info[instr->getDefinition(i).tempId()].set_temp(vec_op.getTemp());
         }
      }
      break;
   }
   case aco_opcode::p_extract_vector: { /* mov */
      if (instr->getOperand(0).isUndefined()) {
         ctx.info[instr->getDefinition(0).tempId()].set_undefined();
         break;
      }
      if (!ctx.info[instr->getOperand(0).tempId()].is_vec())
         break;
      Instruction* vec = ctx.info[instr->getOperand(0).tempId()].instr;
      if (vec->getDefinition(0).getTemp().size() == vec->num_operands) { /* TODO: what about 64bit or other combinations? */

         /* convert this extract into a mov instruction */
         Operand vec_op = vec->getOperand(instr->getOperand(1).constantValue());
         bool is_vgpr = instr->getDefinition(0).getTemp().type() == vgpr;
         aco_opcode opcode = is_vgpr ? aco_opcode::v_mov_b32 : aco_opcode::s_mov_b32;
         Format format = is_vgpr ? Format::VOP1 : Format::SOP1;
         instr->opcode = opcode;
         instr->format = format;
         instr->num_operands = 1;
         instr->getOperand(0) = vec_op;

         if (vec_op.isConstant()) {
            if (vec_op.isLiteral())
               ctx.info[instr->getDefinition(0).tempId()].set_literal(vec_op.constantValue());
            else
               ctx.info[instr->getDefinition(0).tempId()].set_constant(vec_op.constantValue());
         } else if (vec_op.isUndefined()) {
            ctx.info[instr->getDefinition(0).tempId()].set_undefined();
         } else {
            assert(vec_op.isTemp());
            ctx.info[instr->getDefinition(0).tempId()].set_temp(vec_op.getTemp());
         }
      }
      break;
   }
   case aco_opcode::s_mov_b32: /* propagate */
   case aco_opcode::s_mov_b64:
   case aco_opcode::v_mov_b32:
   case aco_opcode::p_as_uniform:
      if (instr->getDefinition(0).isFixed()) {
         /* don't copy-propagate copies into fixed registers */
      } else if (instr->getOperand(0).isConstant()) {
         if (instr->getOperand(0).isLiteral())
            ctx.info[instr->getDefinition(0).tempId()].set_literal(instr->getOperand(0).constantValue());
         else
            ctx.info[instr->getDefinition(0).tempId()].set_constant(instr->getOperand(0).constantValue());
      } else if (instr->isDPP()) {
         // TODO
      } else if (instr->getOperand(0).isUndefined()) {
         ctx.info[instr->getDefinition(0).tempId()].set_undefined();
      } else if (instr->getOperand(0).isTemp()) {
         ctx.info[instr->getDefinition(0).tempId()].set_temp(instr->getOperand(0).getTemp());
      } else {
         assert(instr->getOperand(0).isFixed());
      }
      break;
   case aco_opcode::p_is_helper:
      if (!ctx.program->needs_wqm)
         ctx.info[instr->getDefinition(0).tempId()].set_constant(0u);
      break;
   case aco_opcode::s_movk_i32: {
      uint32_t v = static_cast<SOPK_instruction*>(instr.get())->imm;
      v = v & 0x8000 ? (v | 0xffff0000) : v;
      if (v <= 64 && v >= 0xfffffff0)
         ctx.info[instr->getDefinition(0).tempId()].set_constant(v);
      else
         ctx.info[instr->getDefinition(0).tempId()].set_literal(v);
      break;
   }
   case aco_opcode::v_mul_f32: /* omod */
      /* TODO: try to move the negate/abs modifier to the consumer instead */
      if (instr->isVOP3())
         break;

      if (instr->getOperand(0).isConstant() && instr->getOperand(1).isTemp()) {
         if (instr->getOperand(0).constantValue() == 0x40000000) { /* 2.0 */
            ctx.info[instr->getOperand(1).tempId()].set_omod2();
         } else if (instr->getOperand(0).constantValue() == 0x40800000) { /* 4.0 */
            ctx.info[instr->getOperand(1).tempId()].set_omod4();
         } else if (instr->getOperand(0).constantValue() == 0x3f000000) { /* 0.5 */
            ctx.info[instr->getOperand(1).tempId()].set_omod5();
         }
      }
      break;
   case aco_opcode::v_and_b32: /* abs */
      if (instr->getOperand(0).constantEquals(0x7FFFFFFF) && instr->getOperand(1).isTemp())
         ctx.info[instr->getDefinition(0).tempId()].set_abs(instr->getOperand(1).getTemp());
      break;
   case aco_opcode::v_xor_b32: { /* neg */
      if (instr->getOperand(0).constantEquals(0x80000000u) && instr->getOperand(1).isTemp()) {
         if (ctx.info[instr->getOperand(1).tempId()].is_neg()) {
            ctx.info[instr->getDefinition(0).tempId()].set_temp(ctx.info[instr->getOperand(1).tempId()].temp);
         } else {
            ctx.info[instr->getDefinition(0).tempId()].set_neg(instr->getOperand(1).getTemp());
            if (ctx.info[instr->getOperand(1).tempId()].is_abs()) /* neg(abs(x)) */
               ctx.info[instr->getDefinition(0).tempId()].set_abs(ctx.info[instr->getOperand(1).tempId()].temp);
         }
      }
      break;
   }
   case aco_opcode::v_med3_f32: { /* clamp */
      VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr.get());
      if (vop3->abs[0] || vop3->neg[0] || vop3->opsel[0] ||
          vop3->abs[1] || vop3->neg[1] || vop3->opsel[1] ||
          vop3->abs[2] || vop3->neg[2] || vop3->opsel[2] ||
          vop3->omod != 0)
         break;

      unsigned idx = 0;
      bool found_zero = false, found_one = false;
      for (unsigned i = 0; i < 3; i++)
      {
         if (instr->getOperand(i).constantEquals(0))
            found_zero = true;
         else if (instr->getOperand(i).constantEquals(0x3f800000)) /* 1.0 */
            found_one = true;
         else
            idx = i;
      }
      if (found_zero && found_one && instr->getOperand(idx).isTemp()) {
         ctx.info[instr->getOperand(idx).tempId()].set_clamp();
      }
      break;
   }
   case aco_opcode::v_cndmask_b32:
      if (instr->getOperand(0).constantEquals(0) &&
          instr->getOperand(1).constantEquals(0xFFFFFFFF) &&
          instr->getOperand(2).isTemp())
         ctx.info[instr->getDefinition(0).tempId()].set_vcc(instr->getOperand(2).getTemp());
      else if (instr->getOperand(0).constantEquals(0) &&
               instr->getOperand(1).constantEquals(0x3f800000u) &&
               instr->getOperand(2).isTemp())
         ctx.info[instr->getDefinition(0).tempId()].set_b2f(instr->getOperand(2).getTemp());
      break;
   case aco_opcode::v_cmp_lg_u32:
      if (instr->getOperand(0).constantEquals(0) &&
          instr->getOperand(1).isTemp() && ctx.info[instr->getOperand(1).tempId()].is_vcc())
         ctx.info[instr->getDefinition(0).tempId()].set_temp(ctx.info[instr->getOperand(1).tempId()].temp);
      break;
   case aco_opcode::p_phi:
   case aco_opcode::p_linear_phi: {
      /* lower_bool_phis() can create phis like this */
      bool all_same_temp = instr->getOperand(0).isTemp();
      /* this check is needed when moving uniform loop counters out of a divergent loop */
      if (all_same_temp)
         all_same_temp = instr->getDefinition(0).regClass() == instr->getOperand(0).regClass();
      for (unsigned i = 1; all_same_temp && (i < instr->num_operands); i++) {
         if (!instr->getOperand(i).isTemp() || instr->getOperand(i).tempId() != instr->getOperand(0).tempId())
            all_same_temp = false;
      }
      if (all_same_temp) {
         ctx.info[instr->getDefinition(0).tempId()].set_temp(instr->getOperand(0).getTemp());
      } else {
         bool all_undef = instr->getOperand(0).isUndefined();
         for (unsigned i = 1; all_undef && (i < instr->num_operands); i++) {
            if (!instr->getOperand(i).isUndefined())
               all_undef = false;
         }
         if (all_undef)
            ctx.info[instr->getDefinition(0).tempId()].set_undefined();
      }
      break;
   }
   case aco_opcode::v_add_u32:
   case aco_opcode::v_add_co_u32:
   case aco_opcode::s_add_i32:
   case aco_opcode::s_add_u32:
      for (unsigned i = 0; i < 2; i++) {
         Operand base = instr->getOperand(i);
         Operand offset_op = instr->getOperand(!i);
         if (!base.isTemp() || (!offset_op.isConstant() && !offset_op.isTemp()))
            continue;
         if (offset_op.isTemp() && !ctx.info[offset_op.tempId()].is_constant_or_literal())
            continue;

         uint32_t offset = offset_op.isTemp() ? ctx.info[offset_op.tempId()].val : offset_op.constantValue();
         if (ctx.info[base.tempId()].is_base_offset()) {
            ssa_info *prev = &ctx.info[base.tempId()];
            ctx.info[instr->getDefinition(0).tempId()].set_base_offset(prev->temp, prev->val + offset);

            instr->getOperand(0) = Operand(prev->val + offset);
            instr->getOperand(1) = Operand(prev->temp);
         } else {
            ctx.info[instr->getDefinition(0).tempId()].set_base_offset(base.getTemp(), offset);
         }
         break;
      }
      break;
   default:
      break;
   }
}

// TODO: we could possibly move the whole label_instruction pass to combine_instruction:
// this would mean that we'd have to fix the instruction uses while value propagation

void combine_instruction(opt_ctx &ctx, aco_ptr<Instruction>& instr)
{
   if (!instr->isVALU() || !ctx.uses[instr->getDefinition(0).tempId()])
      return;

   /* apply sgprs */
   uint32_t sgpr_idx = 0;
   uint32_t sgpr_info_id = 0;
   bool has_sgpr = false;
   uint32_t sgpr_ssa_id = 0;
   /* find 'best' possible sgpr */
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (instr->getOperand(i).isLiteral()) {
         has_sgpr = true;
         break;
      }
      if (!instr->getOperand(i).isTemp())
         continue;
      if (instr->getOperand(i).getTemp().type() == sgpr) {
         has_sgpr = true;
         sgpr_ssa_id = instr->getOperand(i).tempId();
         continue;
      }
      ssa_info& info = ctx.info[instr->getOperand(i).tempId()];
      if (info.is_temp() && info.temp.type() == sgpr) {
         uint16_t uses = ctx.uses[instr->getOperand(i).tempId()];
         if (sgpr_info_id == 0 || uses < ctx.uses[sgpr_info_id]) {
            sgpr_idx = i;
            sgpr_info_id = instr->getOperand(i).tempId();
         }
      }
   }
   if (!has_sgpr && sgpr_info_id != 0) {
      ssa_info& info = ctx.info[sgpr_info_id];
      if (sgpr_idx == 0 || instr->isVOP3()) {
         instr->getOperand(sgpr_idx) = Operand(info.temp);
         ctx.uses[sgpr_info_id]--;
         ctx.uses[info.temp.id()]++;
      } else if (can_swap_operands(instr)) {
         instr->getOperand(sgpr_idx) = instr->getOperand(0);
         instr->getOperand(0) = Operand(info.temp);
         ctx.uses[sgpr_info_id]--;
         ctx.uses[info.temp.id()]++;
      } else if (can_use_VOP3(instr)) {
         to_VOP3(ctx, instr);
         instr->getOperand(sgpr_idx) = Operand(info.temp);
         ctx.uses[sgpr_info_id]--;
         ctx.uses[info.temp.id()]++;
      }

   /* we can have two sgprs on one instruction if it is the same sgpr! */
   } else if (sgpr_info_id != 0 &&
              sgpr_ssa_id == sgpr_info_id &&
              ctx.uses[sgpr_info_id] == 1 &&
              can_use_VOP3(instr)) {
      to_VOP3(ctx, instr);
      instr->getOperand(sgpr_idx) = Operand(ctx.info[sgpr_info_id].temp);
      ctx.uses[sgpr_info_id]--;
      ctx.uses[ctx.info[sgpr_info_id].temp.id()]++;
   }

   /* check if we could apply omod on predecessor */
   if (instr->opcode == aco_opcode::v_mul_f32) {
      if (instr->getOperand(1).isTemp() && ctx.info[instr->getOperand(1).tempId()].is_omod_success()) {

         /* omod was successfully applied */
         /* if the omod instruction is v_mad, we also have to change the original add */
         if (ctx.info[instr->getOperand(1).tempId()].is_mad()) {
            Instruction* add_instr = ctx.mad_infos[ctx.info[instr->getOperand(1).tempId()].val].add_instr.get();
            if (ctx.info[instr->getDefinition(0).tempId()].is_clamp())
               static_cast<VOP3A_instruction*>(add_instr)->clamp = true;
            add_instr->getDefinition(0) = instr->getDefinition(0);
         }

         Instruction* omod_instr = ctx.info[instr->getOperand(1).tempId()].instr;
         /* check if we have an additional clamp modifier */
         if (ctx.info[instr->getDefinition(0).tempId()].is_clamp() && ctx.uses[instr->getDefinition(0).tempId()] == 1) {
            static_cast<VOP3A_instruction*>(omod_instr)->clamp = true;
            ctx.info[instr->getDefinition(0).tempId()].set_clamp_success(omod_instr);
         }
         /* change definition ssa-id of modified instruction */
         omod_instr->getDefinition(0) = instr->getDefinition(0);

         /* change the definition of instr to something unused, e.g. the original omod def */
         instr->getDefinition(0) = Definition(instr->getOperand(1).getTemp());
         ctx.uses[instr->getDefinition(0).tempId()] = 0;
         return;
      }
      /* in all other cases, label this instruction as option for multiply-add */
      ctx.info[instr->getDefinition(0).tempId()].set_mul(instr.get());
   }

   /* check if we could apply clamp on predecessor */
   if (instr->opcode == aco_opcode::v_med3_f32) {
      unsigned idx = 0;
      bool found_zero = false, found_one = false;
      for (unsigned i = 0; i < 3; i++)
      {
         if (instr->getOperand(i).constantEquals(0))
            found_zero = true;
         else if (instr->getOperand(i).constantEquals(0x3f800000)) /* 1.0 */
            found_one = true;
         else
            idx = i;
      }
      if (found_zero && found_one && instr->getOperand(idx).isTemp() &&
          ctx.info[instr->getOperand(idx).tempId()].is_clamp_success()) {
         /* clamp was successfully applied */
         /* if the clamp instruction is v_mad, we also have to change the original add */
         if (ctx.info[instr->getOperand(idx).tempId()].is_mad()) {
            Instruction* add_instr = ctx.mad_infos[ctx.info[instr->getOperand(idx).tempId()].val].add_instr.get();
            add_instr->getDefinition(0) = instr->getDefinition(0);
         }
         Instruction* clamp_instr = ctx.info[instr->getOperand(idx).tempId()].instr;
         /* change definition ssa-id of modified instruction */
         clamp_instr->getDefinition(0) = instr->getDefinition(0);

         /* change the definition of instr to something unused, e.g. the original omod def */
         instr->getDefinition(0) = Definition(instr->getOperand(idx).getTemp());
         ctx.uses[instr->getDefinition(0).tempId()] = 0;
         return;
      }
   }

   /* apply omod / clamp modifiers if the def is used only once and the instruction can have modifiers */
   if (instr->num_definitions && ctx.uses[instr->getDefinition(0).tempId()] == 1 &&
       can_use_VOP3(instr) && instr_info.can_use_output_modifiers[(int)instr->opcode]) {
      if(ctx.info[instr->getDefinition(0).tempId()].is_omod2()) {
         to_VOP3(ctx, instr);
         static_cast<VOP3A_instruction*>(instr.get())->omod = 1;
         ctx.info[instr->getDefinition(0).tempId()].set_omod_success(instr.get());
      } else if (ctx.info[instr->getDefinition(0).tempId()].is_omod4()) {
         to_VOP3(ctx, instr);
         static_cast<VOP3A_instruction*>(instr.get())->omod = 2;
         ctx.info[instr->getDefinition(0).tempId()].set_omod_success(instr.get());
      } else if (ctx.info[instr->getDefinition(0).tempId()].is_omod5()) {
         to_VOP3(ctx, instr);
         static_cast<VOP3A_instruction*>(instr.get())->omod = 3;
         ctx.info[instr->getDefinition(0).tempId()].set_omod_success(instr.get());
      } else if (ctx.info[instr->getDefinition(0).tempId()].is_clamp()) {
         to_VOP3(ctx, instr);
         static_cast<VOP3A_instruction*>(instr.get())->clamp = true;
         ctx.info[instr->getDefinition(0).tempId()].set_clamp_success(instr.get());
      }
   }

   /* neg(mul(a, b)) -> mul(neg(a), b) */
   if (ctx.info[instr->getDefinition(0).tempId()].is_neg() && ctx.uses[instr->getOperand(1).tempId()] == 1) {
      Temp val = ctx.info[instr->getDefinition(0).tempId()].temp;

      if (!ctx.info[val.id()].is_mul())
         return;

      Instruction* mul_instr = ctx.info[val.id()].instr;

      if (mul_instr->getOperand(0).isLiteral())
         return;
      if (mul_instr->isVOP3() && static_cast<VOP3A_instruction*>(mul_instr)->clamp)
         return;

      /* convert to mul(neg(a), b) */
      ctx.uses[mul_instr->getDefinition(0).tempId()]--;
      Definition def = instr->getDefinition(0);
      instr.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_mul_f32, asVOP3(Format::VOP2), 2, 1));
      instr->getOperand(0) = mul_instr->getOperand(0);
      instr->getOperand(1) = mul_instr->getOperand(1);
      instr->getDefinition(0) = def;
      VOP3A_instruction* new_mul = static_cast<VOP3A_instruction*>(instr.get());
      if (mul_instr->isVOP3()) {
         VOP3A_instruction* mul = static_cast<VOP3A_instruction*>(mul_instr);
         new_mul->neg[0] = mul->neg[0];
         new_mul->neg[1] = mul->neg[1];
         new_mul->abs[0] = mul->abs[0];
         new_mul->abs[1] = mul->abs[1];
         new_mul->omod = mul->omod;
      }
      new_mul->neg[0] ^= true;
      new_mul->clamp = false;

      ctx.info[instr->getDefinition(0).tempId()].set_mul(instr.get());
      return;
   }
   /* combine mul+add -> mad */
   else if (instr->opcode == aco_opcode::v_add_f32 ||
       instr->opcode == aco_opcode::v_sub_f32 ||
       instr->opcode == aco_opcode::v_subrev_f32) {

      uint32_t uses_src0 = UINT32_MAX;
      uint32_t uses_src1 = UINT32_MAX;
      Instruction* mul_instr = nullptr;
      unsigned add_op_idx;
      /* check if any of the operands is a multiplication */
      if (instr->getOperand(0).isTemp() && ctx.info[instr->getOperand(0).tempId()].is_mul())
         uses_src0 = ctx.uses[instr->getOperand(0).tempId()];
      if (instr->getOperand(1).isTemp() && ctx.info[instr->getOperand(1).tempId()].is_mul())
         uses_src1 = ctx.uses[instr->getOperand(1).tempId()];

      /* find the 'best' mul instruction to combine with the add */
      if (uses_src0 < uses_src1) {
         mul_instr = ctx.info[instr->getOperand(0).tempId()].instr;
         add_op_idx = 1;
      } else if (uses_src1 < uses_src0) {
         mul_instr = ctx.info[instr->getOperand(1).tempId()].instr;
         add_op_idx = 0;
      } else if (uses_src0 != UINT32_MAX) {
         /* tiebreaker: quite random what to pick */
         if (ctx.info[instr->getOperand(0).tempId()].instr->getOperand(0).isLiteral()) {
            mul_instr = ctx.info[instr->getOperand(1).tempId()].instr;
            add_op_idx = 0;
         } else {
            mul_instr = ctx.info[instr->getOperand(0).tempId()].instr;
            add_op_idx = 1;
         }
      }
      if (mul_instr) {
         Operand op[3] = {Operand(v1), Operand(v1), Operand(v1)};
         bool neg[3] = {false, false, false};
         bool abs[3] = {false, false, false};
         unsigned omod = 0;
         bool clamp = false;
         bool need_vop3 = false;
         int num_sgpr = 0;
         op[0] = mul_instr->getOperand(0);
         op[1] = mul_instr->getOperand(1);
         op[2] = instr->getOperand(add_op_idx);
         for (unsigned i = 0; i < 3; i++)
         {
            if (op[i].isLiteral())
               return;
            if (op[i].isTemp() && op[i].getTemp().type() == sgpr)
               num_sgpr++;
            if (!(i == 0 || (op[i].isTemp() && op[i].getTemp().type() == vgpr)))
               need_vop3 = true;
         }
         // TODO: would be better to check this before selecting a mul instr?
         if (num_sgpr > 1)
            return;

         if (mul_instr->isVOP3()) {
            VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*> (mul_instr);
            neg[0] = vop3->neg[0];
            neg[1] = vop3->neg[1];
            abs[0] = vop3->abs[0];
            abs[1] = vop3->abs[1];
            need_vop3 = true;
            /* we cannot use these modifiers between mul and add */
            if (vop3->clamp || vop3->omod)
               return;
         }

         /* convert to mad */
         ctx.uses[mul_instr->getDefinition(0).tempId()]--;
         if (ctx.uses[mul_instr->getDefinition(0).tempId()]) {
            if (op[0].isTemp())
               ctx.uses[op[0].tempId()]++;
            if (op[1].isTemp())
               ctx.uses[op[1].tempId()]++;
         }

         if (instr->isVOP3()) {
            VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*> (instr.get());
            neg[2] = vop3->neg[add_op_idx];
            abs[2] = vop3->abs[add_op_idx];
            omod = vop3->omod;
            clamp = vop3->clamp;
            /* abs of the multiplication result */
            if (vop3->abs[1 - add_op_idx]) {
               neg[0] = false;
               neg[1] = false;
               abs[0] = true;
               abs[1] = true;
            }
            /* neg of the multiplication result */
            neg[1] = neg[1] ^ vop3->neg[1 - add_op_idx];
            need_vop3 = true;
         }
         if (instr->opcode == aco_opcode::v_sub_f32) {
            neg[1 + add_op_idx] = neg[1 + add_op_idx] ^ true;
            need_vop3 = true;
         } else if (instr->opcode == aco_opcode::v_subrev_f32) {
            neg[2 - add_op_idx] = neg[2 - add_op_idx] ^ true;
            need_vop3 = true;
         }

         aco_ptr<VOP3A_instruction> mad{create_instruction<VOP3A_instruction>(aco_opcode::v_mad_f32, Format::VOP3A, 3, 1)};
         for (unsigned i = 0; i < 3; i++)
         {
            mad->getOperand(i) = op[i];
            mad->neg[i] = neg[i];
            mad->abs[i] = abs[i];
         }
         mad->omod = omod;
         mad->clamp = clamp;
         mad->getDefinition(0) = instr->getDefinition(0);

         /* mark this ssa_def to be re-checked for profitability and literals */
         ctx.mad_infos.emplace_back(std::move(instr), mul_instr->getDefinition(0).tempId(), need_vop3);
         ctx.info[mad->getDefinition(0).tempId()].set_mad(mad.get(), ctx.mad_infos.size() - 1);
         instr.reset(mad.release());
         return;
      }
   }
   /* v_mul_f32(v_cndmask_b32(0, 1.0, cond), a) -> v_cndmask_b32(0, a, cond) */
   else if (instr->opcode == aco_opcode::v_mul_f32 && !instr->isVOP3()) {
      for (unsigned i = 0; i < 2; i++) {
         if (instr->getOperand(i).isTemp() && ctx.info[instr->getOperand(i).tempId()].is_b2f() &&
             ctx.uses[instr->getOperand(i).tempId()] == 1 &&
             instr->getOperand(!i).isTemp() && instr->getOperand(!i).getTemp().type() == vgpr) {
            ctx.uses[instr->getOperand(i).tempId()]--;
            ctx.uses[ctx.info[instr->getOperand(i).tempId()].temp.id()]++;

            aco_ptr<VOP2_instruction> new_instr{create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1)};
            new_instr->getOperand(0) = Operand(0u);
            new_instr->getOperand(1) = instr->getOperand(!i);
            new_instr->getOperand(2) = Operand(ctx.info[instr->getOperand(i).tempId()].temp);
            new_instr->getDefinition(0) = instr->getDefinition(0);
            instr.reset(new_instr.release());
            ctx.info[instr->getDefinition(0).tempId()].label.reset();
            return;
         }
      }
   }
}


void select_instruction(opt_ctx &ctx, aco_ptr<Instruction>& instr)
{
   const uint32_t threshold = 4;

   /* Dead Code Elimination:
    * We remove instructions if they define temporaries which all are unused */
   bool is_used = instr->num_definitions == 0;
   for (unsigned i = 0; !is_used && i < instr->num_definitions; i++) {
      if (ctx.uses[instr->getDefinition(i).tempId()])
         is_used = true;
   }
   if (!is_used) {
      instr.reset();
      return;
   }

   /* convert split_vector into extract_vector if only one definition is ever used */
   if (instr->opcode == aco_opcode::p_split_vector) {
      unsigned num_used = 0;
      unsigned idx = 0;
      for (unsigned i = 0; i < instr->num_definitions; i++) {
         if (ctx.uses[instr->getDefinition(i).tempId()]) {
            num_used++;
            idx = i;
         }
      }
      if (num_used == 1) {
         aco_ptr<Pseudo_instruction> extract{create_instruction<Pseudo_instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1)};
         extract->getOperand(0) = instr->getOperand(0);
         extract->getOperand(1) = Operand((uint32_t) idx);
         extract->getDefinition(0) = instr->getDefinition(idx);
         instr.reset(extract.release());
      }
   }

   /* re-check mad instructions */
   if (instr->opcode == aco_opcode::v_mad_f32 && ctx.info[instr->getDefinition(0).tempId()].is_mad()) {
      mad_info* info = &ctx.mad_infos[ctx.info[instr->getDefinition(0).tempId()].val];
      /* first, check profitability */
      if (ctx.uses[info->mul_temp_id]) {
         ctx.uses[info->mul_temp_id]++;
         instr.swap(info->add_instr);

      /* second, check possible literals */
      } else if (!info->needs_vop3) {
         uint32_t literal_idx = 0;
         uint32_t literal_uses = UINT32_MAX;
         for (unsigned i = 0; i < instr->num_operands; i++)
         {
            if (!instr->getOperand(i).isTemp())
               continue;
            /* if one of the operands is sgpr, we cannot add a literal somewhere else */
            if (instr->getOperand(i).getTemp().type() == sgpr) {
               if (ctx.info[instr->getOperand(i).tempId()].is_literal()) {
                  literal_uses = ctx.uses[instr->getOperand(i).tempId()];
                  literal_idx = i;
               } else {
                  literal_uses = UINT32_MAX;
               }
               break;
            }
            else if (ctx.info[instr->getOperand(i).tempId()].is_literal() &&
                ctx.uses[instr->getOperand(i).tempId()] < literal_uses) {
               literal_uses = ctx.uses[instr->getOperand(i).tempId()];
               literal_idx = i;
            }
         }
         if (literal_uses < threshold) {
            ctx.uses[instr->getOperand(literal_idx).tempId()]--;
            info->check_literal = true;
            info->literal_idx = literal_idx;
         }
      }
      return;
   }

   /* check for literals */
   /* we do not apply the literals yet as we don't know if it is profitable */
   if (instr->isSALU()) {
      uint32_t literal_idx = 0;
      uint32_t literal_uses = UINT32_MAX;
      bool has_literal = false;
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (!instr->getOperand(i).isTemp())
            continue;
         if (instr->getOperand(i).isLiteral()) {
            has_literal = true;
            break;
         }
         if (ctx.info[instr->getOperand(i).tempId()].is_literal() &&
             ctx.uses[instr->getOperand(i).tempId()] < literal_uses) {
            literal_uses = ctx.uses[instr->getOperand(i).tempId()];
            literal_idx = i;
         }
      }
      if (!has_literal && literal_uses < threshold) {
         ctx.uses[instr->getOperand(literal_idx).tempId()]--;
         if (ctx.uses[instr->getOperand(literal_idx).tempId()] == 0)
            instr->getOperand(literal_idx) = Operand(ctx.info[instr->getOperand(literal_idx).tempId()].val);
      }
   } else if (instr->isVALU() && !instr->isVOP3() &&
       instr->getOperand(0).isTemp() &&
       ctx.info[instr->getOperand(0).tempId()].is_literal() &&
       ctx.uses[instr->getOperand(0).tempId()] < threshold) {
      ctx.uses[instr->getOperand(0).tempId()]--;
      if (ctx.uses[instr->getOperand(0).tempId()] == 0)
         instr->getOperand(0) = Operand(ctx.info[instr->getOperand(0).tempId()].val);
   }

}


void apply_literals(opt_ctx &ctx, aco_ptr<Instruction>& instr)
{
   /* Cleanup Dead Instructions */
   if (!instr)
      return;

   /* apply literals on SALU */
   if (instr->isSALU()) {
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (!instr->getOperand(i).isTemp())
            continue;
         if (instr->getOperand(i).isLiteral())
            break;
         if (ctx.info[instr->getOperand(i).tempId()].is_literal() &&
             ctx.uses[instr->getOperand(i).tempId()] == 0)
            instr->getOperand(i) = Operand(ctx.info[instr->getOperand(i).tempId()].val);
      }
   }

   /* apply literals on VALU */
   else if (instr->isVALU() && !instr->isVOP3() &&
       instr->getOperand(0).isTemp() &&
       ctx.info[instr->getOperand(0).tempId()].is_literal() &&
       ctx.uses[instr->getOperand(0).tempId()] == 0) {
      instr->getOperand(0) = Operand(ctx.info[instr->getOperand(0).tempId()].val);
   }

   /* apply literals on MAD */
   else if (instr->opcode == aco_opcode::v_mad_f32 && ctx.info[instr->getDefinition(0).tempId()].is_mad()) {
      mad_info* info = &ctx.mad_infos[ctx.info[instr->getDefinition(0).tempId()].val];
      aco_ptr<Instruction> new_mad;
      if (info->check_literal && ctx.uses[instr->getOperand(info->literal_idx).tempId()] == 0) {
         if (info->literal_idx == 2) { /* add literal -> madak */
            new_mad.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madak_f32, Format::VOP2, 3, 1));
            new_mad->getOperand(0) = instr->getOperand(0);
            new_mad->getOperand(1) = instr->getOperand(1);
         } else { /* mul literal -> madmk */
            new_mad.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madmk_f32, Format::VOP2, 3, 1));
            new_mad->getOperand(0) = instr->getOperand(1 - info->literal_idx);
            new_mad->getOperand(1) = instr->getOperand(2);
         }
         new_mad->getOperand(2) = Operand(ctx.info[instr->getOperand(info->literal_idx).tempId()].val);
         new_mad->getDefinition(0) = instr->getDefinition(0);
         instr.swap(new_mad);
      }
   }

   ctx.instructions.emplace_back(std::move(instr));
}


void optimize(Program* program)
{
   opt_ctx ctx;
   ctx.program = program;
   std::vector<ssa_info> info(program->peekAllocationId());
   ctx.info = info.data();

   /* 1. Bottom-Up DAG pass (forward) to label all ssa-defs */
   for (Block& block : program->blocks) {
      for (aco_ptr<Instruction>& instr : block.instructions)
         label_instruction(ctx, instr);
   }

   ctx.uses = std::move(dead_code_analysis(program));

   /* 2. Combine v_mad, omod, clamp and propagate sgpr on VALU instructions */
   for (Block& block : program->blocks) {
      for (aco_ptr<Instruction>& instr : block.instructions)
         combine_instruction(ctx, instr);
   }

   /* 3. Top-Down DAG pass (backward) to select instructions (includes DCE) */
   for (std::vector<Block>::reverse_iterator it = program->blocks.rbegin(); it != program->blocks.rend(); ++it) {
      Block* block = &(*it);
      for (std::vector<aco_ptr<Instruction>>::reverse_iterator it = block->instructions.rbegin(); it != block->instructions.rend(); ++it)
         select_instruction(ctx, *it);
   }

   /* 4. Add literals to instructions */
   for (Block& block : program->blocks) {
      ctx.instructions.clear();
      for (aco_ptr<Instruction>& instr : block.instructions)
         apply_literals(ctx, instr);
      block.instructions.swap(ctx.instructions);
   }

}

}
