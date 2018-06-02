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
#include <set>

namespace aco {

struct combinator_ctx {
   std::set<int> uses;
};

void handle_instruction(combinator_ctx& ctx, std::unique_ptr<Instruction>& instr)
{
   for (unsigned i = 0; i < instr->num_definitions; i++)
   {
      if (ctx.uses.find(instr->getDefinition(i).tempId()) != ctx.uses.end())
         break;

      instr->format = Format::PSEUDO;
      instr->num_operands = 0;
      instr->num_definitions = 0;
      return;
   }
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (!instr->getOperand(i).isTemp())
         continue;

      ctx.uses.insert(instr->getOperand(i).tempId());
   }
}

void combine_bw(Program* program)
{
   combinator_ctx ctx;
   for (auto&& block : program->blocks)
   {
      for (std::vector<std::unique_ptr<Instruction>>::reverse_iterator rit = block->instructions.rbegin(); rit != block->instructions.rend(); ++rit)
      {
         handle_instruction(ctx, *rit);
      }
   }
}
}
