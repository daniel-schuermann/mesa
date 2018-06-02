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
   std::vector<std::unique_ptr<Instruction>> instructions;
   std::map<int, Operand> values;
   std::map<int, std::array<Operand,4>> vectors;
};

void handle_instruction(combinator_ctx_fw& ctx, std::unique_ptr<Instruction>& instr)
{
   if (instr->isVALU() || instr->isSALU() || instr->format == Format::PSEUDO) {
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (!instr->getOperand(i).isTemp())
            continue;
         std::map<int, Operand>::iterator it = ctx.values.find(instr->getOperand(i).tempId());
         if (it != ctx.values.end()) {

            if (it->second.isConstant()) {
               /* Literals should only appear as src0 and not on VOP3 instructions */
               if (it->second.physReg().reg == 255 && (i != 0 || ((int) instr->format & (int) Format::VOP3A)))
                  continue;
               /* For VALU, inline constants can only appear as src0 (Except for VOP3) */
               if (i != 0 && instr->isVALU() && !((int) instr->format & (int) Format::VOP3A))
                  continue;
            } else {
               /* For VALU, sgpr can only appear as src0 */ // TODO: exception being VOP3 if no other sgpr is read
               if (it->second.getTemp().type() == sgpr && instr->isVALU() && i != 0)
                  continue;
               /* v_addc, v_subb & v_cndmask implicitly use VCC */
               if (it->second.getTemp().type() == sgpr && (instr->opcode == aco_opcode::v_cndmask_b32 ||
                                                           instr->opcode == aco_opcode::v_add_co_u32 ||
                                                           instr->opcode == aco_opcode::v_sub_co_u32 ||
                                                           instr->opcode == aco_opcode::v_subrev_co_u32))
                  continue;
            }
            instr->getOperand(i) = it->second;
         }
      }
   }

   if (instr->opcode == aco_opcode::p_create_vector) {
      std::array<Operand,4> ops;
      for (unsigned i = 0; i < instr->num_operands; i++)
         ops[i] = instr->getOperand(i);
      ctx.vectors.insert({instr->getDefinition(0).tempId(), ops});
   } else if (instr->opcode == aco_opcode::p_extract_vector) {
      std::map<int, std::array<Operand,4>>::iterator it = ctx.vectors.find(instr->getOperand(0).tempId());
      if (it != ctx.vectors.end())
         ctx.values.insert({instr->getDefinition(0).tempId(), it->second[instr->getOperand(1).constantValue()]});
   } else if (instr->opcode == aco_opcode::s_mov_b32) {
      ctx.values.insert({instr->getDefinition(0).tempId(), instr->getOperand(0)});
   }
   ctx.instructions.emplace_back(std::move(instr));
}

void combine_fw(Program* program)
{
   combinator_ctx_fw ctx;
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
