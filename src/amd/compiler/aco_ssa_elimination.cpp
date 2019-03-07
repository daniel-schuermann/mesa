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
 */


#include "aco_ir.h"

#include <map>

namespace aco {
namespace {

/* map: block-id -> pair (dest, src) to store phi information */
typedef std::map<uint32_t, std::vector<std::pair<Definition, Operand>>> phi_info;

struct ssa_elimination_ctx {
   phi_info logical_phi_info;
   phi_info linear_phi_info;
   Program* program;

   ssa_elimination_ctx(Program* program) : program(program) {}
};

void collect_phi_info(ssa_elimination_ctx& ctx)
{
   for (std::unique_ptr<Block>& block : ctx.program->blocks) {
      for (aco_ptr<Instruction>& phi : block->instructions) {
         if (phi->opcode == aco_opcode::p_phi || phi->opcode == aco_opcode::p_linear_phi) {
            for (unsigned i = 0; i < phi->num_operands; i++) {
               if (phi->getOperand(i).isUndefined())
                  continue;
               if (phi->getOperand(i).isTemp() && phi->getOperand(i).physReg() == phi->getDefinition(0).physReg())
                  continue;

               std::vector<Block*>& preds = phi->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
               phi_info& info = phi->opcode == aco_opcode::p_phi ? ctx.logical_phi_info : ctx.linear_phi_info;
               const auto result = info.emplace(preds[i]->index, std::vector<std::pair<Definition, Operand>>());
               result.first->second.emplace_back(phi->getDefinition(0), phi->getOperand(i));
            }
         } else {
            break;
         }
      }
   }
}

void insert_parallelcopies(ssa_elimination_ctx& ctx)
{
   /* insert the parallelcopies from logical phis before p_logical_end */
   for (auto&& entry : ctx.logical_phi_info) {
      std::unique_ptr<Block>& block = ctx.program->blocks[entry.first];
      unsigned idx = block->instructions.size() - 1;
      while (block->instructions[idx]->opcode != aco_opcode::p_logical_end) {
         assert(idx > 0);
         idx--;
      }

      std::vector<aco_ptr<Instruction>>::iterator it = std::next(block->instructions.begin(), idx);
      aco_ptr<Instruction> pc{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, entry.second.size(), entry.second.size())};
      unsigned i = 0;
      for (std::pair<Definition, Operand>& pair : entry.second)
      {
         pc->getDefinition(i) = pair.first;
         pc->getOperand(i) = pair.second;
         i++;
      }
      block->instructions.insert(it, std::move(pc));
   }

   /* insert parallelcopies for the linear phis at the end of blocks just before the branch */
   for (auto&& entry : ctx.linear_phi_info) {
      std::unique_ptr<Block>& block = ctx.program->blocks[entry.first];
      std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.end();
      --it;
      assert((*it)->format == Format::PSEUDO_BRANCH);
      aco_ptr<Instruction> pc{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, entry.second.size(), entry.second.size())};
      unsigned i = 0;
      for (std::pair<Definition, Operand>& pair : entry.second)
      {
         pc->getDefinition(i) = pair.first;
         pc->getOperand(i) = pair.second;
         i++;
      }
      block->instructions.insert(it, std::move(pc));
   }
}

} /* end namespace */


void eliminate_phis(Program* program)
{
   ssa_elimination_ctx ctx(program);

   /* Collect information about every phi-instruction */
   collect_phi_info(ctx);

   insert_parallelcopies(ctx);

}
}
