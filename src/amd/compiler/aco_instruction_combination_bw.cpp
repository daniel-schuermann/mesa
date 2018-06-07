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

namespace aco {

struct combinator_ctx {
   uint32_t* uses;
};

void handle_instruction(combinator_ctx& ctx, std::unique_ptr<Instruction>& instr)
{
   for (unsigned i = 0; i < instr->num_definitions; i++)
   {
      if (ctx.uses[instr->getDefinition(i).tempId()] == 0) {
         instr->format = Format::PSEUDO;
         instr->num_operands = 0;
         instr->num_definitions = 0;
         return;
      }
   }

   if (instr->opcode == aco_opcode::v_mad_f32 &&
      (!instr->getOperand(2).isTemp() ||
       ctx.uses[instr->getOperand(2).tempId()] == 0)) {
      /* check if it could be made a v_mac */
      VOP3A_instruction* mad = static_cast<VOP3A_instruction*>(instr.get());
      if (!(mad->getOperand(1).getTemp().type() != vgpr ||
          mad->getOperand(2).getTemp().type() != vgpr ||
          mad->abs[0] || mad->abs[1] || mad->abs[2] ||
          mad->neg[0] || mad->neg[1] || mad->neg[2])) {

         instr->opcode = aco_opcode::v_mac_f32;
         std::unique_ptr<Instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_mac_f32, Format::VOP2, 3, 1)};
         for (unsigned i = 0; i < 3; i++)
            vop2->getOperand(i) = instr->getOperand(i);
         vop2->getDefinition(0) = instr->getDefinition(0);
         instr.reset(vop2.release());
      }
   }
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (!instr->getOperand(i).isTemp())
         continue;
      ctx.uses[instr->getOperand(i).tempId()]++;
   }
}

void combine_bw(Program* program)
{
   uint32_t uses[program->peekAllocationId()];
   for (unsigned i = 0; i < program->peekAllocationId(); i++)
      uses[i] = 0;
   combinator_ctx ctx;
   ctx.uses = uses;
   for (auto&& block : program->blocks)
   {
      for (std::vector<std::unique_ptr<Instruction>>::reverse_iterator rit = block->instructions.rbegin(); rit != block->instructions.rend(); ++rit)
      {
         handle_instruction(ctx, *rit);
      }
   }
}
}
