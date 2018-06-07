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

#include "aco_ir.h"
#include <map>

namespace aco {

struct combinator_ctx_fw {
   Program* program;
   std::vector<std::unique_ptr<Instruction>> instructions;
   std::map<uint32_t, Operand> values;
   std::map<uint32_t, std::array<Operand,4>> vectors;
   std::map<uint64_t, Temp> vector_extracts;
   std::map<uint32_t, Operand> neg;
   std::map<uint32_t, Operand> abs;
   std::map<uint32_t, Instruction*> mul;
   std::pair<uint32_t,Temp> last_literal;
};

bool canUseVOP3(std::unique_ptr<Instruction>& instr)
{
   // TODO
   return instr->opcode != aco_opcode::v_madmk_f32 &&
          instr->opcode != aco_opcode::v_madak_f32 &&
          instr->opcode != aco_opcode::v_madmk_f16 &&
          instr->opcode != aco_opcode::v_madak_f16;
}

Temp rematerializeLiteral(combinator_ctx_fw& ctx, uint32_t literal)
{
   if (literal == ctx.last_literal.first) {
      return ctx.last_literal.second;
   } else { /* rematerialize */
      std::unique_ptr<SOP1_instruction> mov{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1)};
      mov->getOperand(0) = Operand(literal);
      Temp t = Temp(ctx.program->allocateId(), s1);
      mov->getDefinition(0) = Definition(t);
      ctx.instructions.emplace_back(std::move(mov));
      ctx.last_literal = {literal, t};
      return t;
   }
}

void toVOP3(combinator_ctx_fw& ctx, std::unique_ptr<Instruction>& instr)
{
   if ((int) instr->format & (int) Format::VOP3A)
      return;
   if (instr->getOperand(0).isConstant() && instr->getOperand(0).physReg().reg == 255)
      instr->getOperand(0) = Operand(rematerializeLiteral(ctx, instr->getOperand(0).constantValue()));

   std::unique_ptr<Instruction> tmp = std::move(instr);
   Format format = (Format) ((int) tmp->format | (int) Format::VOP3A);
   instr.reset(create_instruction<VOP3A_instruction>(tmp->opcode, format, tmp->num_operands, tmp->num_definitions));
   for (unsigned i = 0; i < instr->num_operands; i++)
      instr->getOperand(i) = tmp->getOperand(i);
   for (unsigned i = 0; i < instr->num_definitions; i++)
      instr->getDefinition(i) = tmp->getDefinition(i);
}

void handle_instruction(combinator_ctx_fw& ctx, std::unique_ptr<Instruction>& instr)
{
   if (instr->format == Format::PSEUDO || instr->isSALU()) {
      /* due to constant folding in nir, it shouldn't happen that we have 2 literals on SALU */
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (!instr->getOperand(i).isTemp())
            continue;
         std::map<uint32_t, Operand>::iterator it = ctx.values.find(instr->getOperand(i).tempId());
         if (it != ctx.values.end())
            instr->getOperand(i) = it->second;
      }
   }

   else if (instr->isVALU()) {
      bool uses_sgpr = false;
      /* check if the instruction already uses an sgpr */
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == sgpr) {
            uses_sgpr = true;
            break;
         }
      }

      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (!instr->getOperand(i).isTemp())
            continue;
         std::map<uint32_t, Operand>::iterator it = ctx.values.find(instr->getOperand(i).tempId());
         if (it != ctx.values.end()) {
            /* apply literals */
            if (it->second.isConstant() && it->second.physReg().reg == 255) {
               if (i == 0 && !((int) instr->format & (int) Format::VOP3A)) {
                  uses_sgpr = uses_sgpr && instr->getOperand(i).getTemp().type() == sgpr ? false : uses_sgpr;
                  instr->getOperand(i) = it->second;
               } else if ((int) instr->format & (int) Format::VOP3A) {
                  uses_sgpr = true;
                  instr->getOperand(i) = Operand(rematerializeLiteral(ctx, it->second.constantValue()));
               }
            }

            /* easy case: propagate vgprs */
            else if (it->second.isTemp() && it->second.getTemp().type() == vgpr) {
               instr->getOperand(i) = it->second;

            } else if (it->second.isTemp() && it->second.getTemp().type() == sgpr && !uses_sgpr) {
               if (i == 0) {
                  uses_sgpr = true;
                  instr->getOperand(i) = it->second;
               } else if (canUseVOP3(instr)) {
                  uses_sgpr = true;
                  toVOP3(ctx, instr);
                  instr->getOperand(i) = it->second;
               }
            } else { /* is constant non-literal */
               if (i == 0) {
                  instr->getOperand(i) = it->second;
               } else if (canUseVOP3(instr)) {
                  toVOP3(ctx, instr);
                  instr->getOperand(i) = it->second;
               }
            }
         }

         /* propagate negation flag */
         it = ctx.neg.find(instr->getOperand(i).tempId());
         if (it != ctx.neg.end() && canUseVOP3(instr)) {
            toVOP3(ctx, instr);
            VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr.get());
            vop3->getOperand(i) = it->second;
            vop3->neg[i] = !vop3->neg[i];
         }

         /* propagate abs flag */
         it = ctx.abs.find(instr->getOperand(i).tempId());
         if (it != ctx.abs.end() && canUseVOP3(instr)) {
            toVOP3(ctx, instr);
            VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr.get());
            vop3->getOperand(i) = it->second;
            vop3->abs[i] = true;
         }
      }

      /* optimize multiply-add */
      if (instr->opcode == aco_opcode::v_add_f32) {
         for (unsigned i = 0; i < instr->num_operands; i++)
         {
            if (!instr->getOperand(i).isTemp())
               continue;
            std::map<uint32_t, Instruction*>::iterator mul_it = ctx.mul.find(instr->getOperand(i).tempId());
            if (mul_it == ctx.mul.end())
               continue;
            Instruction* mul = mul_it->second;
            bool mul_lit = mul->getOperand(0).isConstant() && mul->getOperand(0).physReg().reg == 255;
            bool add_lit = instr->getOperand(0).isConstant() && instr->getOperand(0).physReg().reg == 255;
            std::unique_ptr<Instruction> mad;

            /* check if we can use madmk */
            if (!uses_sgpr && instr->format == Format::VOP2 && mul_lit && !add_lit) {
               mad.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madmk_f32, Format::VOP2, 3, 1));
               mad->getOperand(0) = mul->getOperand(1);
               mad->getOperand(1) = instr->getOperand(i == 0 ? 1 : 0);
               mad->getOperand(2) = mul->getOperand(0); /* the literal */
               mad->getDefinition(0) = instr->getDefinition(0);
               instr.reset(mad.release());

            /* check if we can use madak */
            } else if (!uses_sgpr && mul->format == Format::VOP2 && add_lit) {
               mad.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madak_f32, Format::VOP2, 3, 1));
               if (mul_lit)
                  mad->getOperand(0) = Operand(rematerializeLiteral(ctx, mul->getOperand(0).constantValue()));
               else
                  mad->getOperand(0) = mul->getOperand(0);
               mad->getOperand(1) = mul->getOperand(1);
               mad->getOperand(2) = instr->getOperand(0); /* the literal */
               mad->getDefinition(0) = instr->getDefinition(0);
               instr.reset(mad.release());

            /* we have to use mad (maybe later optimized to mac) */
            } else {
               std::unique_ptr<VOP3A_instruction> mad{create_instruction<VOP3A_instruction>(aco_opcode::v_mad_f32, Format::VOP3A, 3, 1)};
               if (mul->format == Format::VOP2) {
                  if (mul_lit)
                     mad->getOperand(0) = Operand(rematerializeLiteral(ctx, mul->getOperand(0).constantValue()));
                  else
                     mad->getOperand(0) = mul->getOperand(0);
               } else { /* mul is VOP3 */
                  VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(mul);
                  // TODO caution: clamp & omod
                  mad->abs[0] = vop3->abs[0];
                  mad->abs[1] = vop3->abs[1];
                  mad->neg[0] = vop3->neg[0];
                  mad->neg[1] = vop3->neg[1];
                  mad->getOperand(0) = vop3->getOperand(0);
               }
               mad->getOperand(1) = mul->getOperand(1);
               unsigned idx = i == 0 ? 1 : 0;
               if (instr->format == Format::VOP2) {
                  if (add_lit) {
                     mad->getOperand(2) = Operand(rematerializeLiteral(ctx, instr->getOperand(0).constantValue()));
                  } else {
                     mad->getOperand(2) = instr->getOperand(idx);
                  }
               } else { /* add is VOP3 */
                  VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr.get());
                  mad->abs[2] = vop3->abs[idx];
                  if (vop3->abs[i]) {
                     /* to get the absolute value of the multiplication, we remove the neg-flags and add abs flags */
                     mad->abs[0] = true;
                     mad->abs[1] = true;
                     mad->neg[0] = false;
                     mad->neg[1] = false;
                  }
                  mad->neg[2] = vop3->neg[idx];
                  mad->neg[1] = mad->neg[1] ^ vop3->neg[i];
                  mad->getOperand(2) = instr->getOperand(idx);
               }
               mad->getDefinition(0) = instr->getDefinition(0);
               /* only emplace this new instruction if it uses at most 1 sgpr */
               int num_sgpr = 0;
               for (unsigned i = 0; i < 3; i++)
                  num_sgpr += mad->getOperand(i).isTemp() && mad->getOperand(i).getTemp().type() == sgpr ? 1 : 0;
               if (num_sgpr <= 1)
                  instr.reset(mad.release());
            }
            break;
         }
      }

   } else if (instr->format == Format::EXP) {
      /* export instructions only accept vgpr inputs */
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (instr->getOperand(i).isTemp()) {
            std::map<uint32_t, Operand>::iterator it = ctx.values.find(instr->getOperand(i).tempId());
            if (it != ctx.values.end() && it->second.isTemp() && it->second.getTemp().type() == vgpr)
               instr->getOperand(i) = it->second;
         }
      }
   } else if (instr->format == Format::VINTRP) {
      std::map<uint32_t, Operand>::iterator it = ctx.values.find(instr->getOperand(0).tempId());
      if (it != ctx.values.end() && it->second.isTemp() && it->second.regClass() == instr->getOperand(0).regClass())
         instr->getOperand(0) = it->second;

   }

   if (instr->opcode == aco_opcode::p_create_vector) {
      std::array<Operand,4> ops;
      for (unsigned i = 0; i < instr->num_operands; i++)
         ops[i] = instr->getOperand(i);
      ctx.vectors.insert({instr->getDefinition(0).tempId(), ops});
   } else if (instr->opcode == aco_opcode::p_extract_vector) {
      std::map<uint64_t, Temp>::iterator it = ctx.vector_extracts.find((uint64_t) instr->getOperand(0).tempId() << 32 | instr->getOperand(1).constantValue());
      if (it != ctx.vector_extracts.end()) {
         ctx.values.insert({instr->getDefinition(0).tempId(), Operand(it->second)});
      } else {
         std::map<uint32_t, std::array<Operand,4>>::iterator it = ctx.vectors.find(instr->getOperand(0).tempId());
         if (it != ctx.vectors.end()) {
            Operand op = it->second[instr->getOperand(1).constantValue()];
            bool is_vgpr = instr->getDefinition(0).getTemp().type();
            aco_opcode opcode = is_vgpr ? aco_opcode::v_mov_b32 : aco_opcode::s_mov_b32;
            Format format = is_vgpr ? Format::VOP1 : Format::SOP1;
            instr->opcode = opcode;
            instr->format = format;
            instr->num_operands = 1;
            instr->getOperand(0) = op;
            ctx.values.insert({instr->getDefinition(0).tempId(), op});
         }
         ctx.vector_extracts.insert({(uint64_t) instr->getOperand(0).tempId() << 32 | instr->getOperand(1).constantValue(),
                                     instr->getDefinition(0).getTemp()});
      }
   } else if (instr->opcode == aco_opcode::s_mov_b32) {
      ctx.values.insert({instr->getDefinition(0).tempId(), instr->getOperand(0)});
   } else if (instr->opcode == aco_opcode::v_sub_f32 &&
              instr->getOperand(0).isConstant() &&
              instr->getOperand(0).constantValue() == 0 &&
              instr->format == Format::VOP2) {
      /* negation */
      // TODO: we should be able to handle double negations and -abs(value)
      ctx.neg.insert({instr->getDefinition(0).tempId(), instr->getOperand(1)});
   } else if (instr->opcode == aco_opcode::v_mul_f32) {
      ctx.mul.insert({instr->getDefinition(0).tempId(), instr.get()});
   } else if (instr->opcode == aco_opcode::v_and_b32 &&
              instr->getOperand(0).isConstant() &&
              instr->getOperand(0).constantValue() == 0x7FFFFFFF &&
              instr->format == Format::VOP2) {
      /* abs */
      ctx.abs.insert({instr->getDefinition(0).tempId(), instr->getOperand(1)});
   }
   ctx.instructions.emplace_back(std::move(instr));
}

void combine_fw(Program* program)
{
   combinator_ctx_fw ctx;
   ctx.program = program;
   for (auto&& block : program->blocks)
   {
      ctx.instructions.clear();
      for (std::unique_ptr<Instruction>& instr : block->instructions)
      {
         handle_instruction(ctx, instr);
      }
      block->instructions.swap(ctx.instructions);
   }
}
}
