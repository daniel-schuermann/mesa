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

/* we use a map: block-id -> pair (dest, src) to store phi information */
typedef std::map<uint32_t, std::vector<std::pair<Definition, Operand>>> phi_info;

void collect_phi_info(phi_info& ctx, std::unique_ptr<Instruction>& phi, std::unique_ptr<Block>& block)
{
   std::vector<Block*>& preds = phi->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
   assert(!(phi->opcode == aco_opcode::p_phi && phi->getDefinition(0).getTemp().type() == sgpr) && "smart merging for bools not yet implemented.");
   for (unsigned i = 0; i < preds.size(); i++)
   {
      const auto result = ctx.emplace(preds[i]->index, std::vector<std::pair<Definition, Operand>>());
      result.first->second.emplace_back(phi->getDefinition(0), phi->getOperand(i));
   }

}

void eliminate_phis(Program* program)
{
   phi_info ctx;

   /* 1. Collect information about every phi-instruction */
   for (auto&& block : program->blocks) {
      for (std::unique_ptr<Instruction>& instr : block->instructions)
      {
         if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi)
            collect_phi_info(ctx, instr, block);
      }
   }

   /* 2. we replace the p_logical_end instructions with a parallelcopy (we don't need the former anymore) */
   for (auto&& entry : ctx) {
      std::unique_ptr<Block>& block = program->blocks[entry.first];
      for (std::vector<std::unique_ptr<Instruction>>::reverse_iterator it = block->instructions.rbegin(); ; ++it)
      {
         assert(it != block->instructions.rend() && "Couldn't find a p_logical_end instruction");
         if ((*it)->opcode == aco_opcode::p_logical_end) {
            std::unique_ptr<Instruction> pc{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, entry.second.size(), entry.second.size())};
            unsigned idx = 0;
            for (std::pair<Definition, Operand>& pair : entry.second)
            {
               pc->getDefinition(idx) = pair.first;
               pc->getOperand(idx) = pair.second;
               idx++;
            }
            (*it).swap(pc);
            break;
         }
      }
   }
}
}
