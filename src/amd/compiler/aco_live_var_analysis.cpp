/*
 * Copyright © 2018 Valve Corporation
 * Copyright © 2018 Google
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
 *    Bas Nieuwenhuizen (bas@basnieuwenhuizen.nl)
 *
 */

 #include "aco_ir.h"
 #include <vector>
 #include <set>

 namespace aco {

 void process_live_temps_per_block(std::vector<std::set<Temp>>& live_temps, Block* block, std::set<unsigned>& worklist)
{
   std::set<Temp> live_sgprs;
   std::set<Temp> live_vgprs;
   unsigned vgpr_demand = 0;
   unsigned sgpr_demand = 0;
   block->vgpr_demand = 0;
   block->sgpr_demand = 0;
   /* first, insert the live-outs from this block into our temporary sets */
   for (std::set<Temp>::iterator it = live_temps[block->index].begin(); it != live_temps[block->index].end(); ++it)
   {
      if ((*it).type() == vgpr) {
         live_vgprs.insert(*it);
         vgpr_demand += (*it).size();
      } else {
         live_sgprs.insert(*it);
         sgpr_demand += (*it).size();
      }
   }

   /* traverse the instructions backwards */
   for (auto it = block->instructions.rbegin(); it != block->instructions.rend(); ++it)
   {
      Instruction *insn = it->get();
      /* KILL */
      for (unsigned i = 0; i < insn->definitionCount(); ++i)
      {
         auto& definition = insn->getDefinition(i);
         if (definition.isTemp()) {
            if (definition.getTemp().type() == vgpr) {
               vgpr_demand -= definition.size() * live_vgprs.erase(definition.getTemp());
             } else {
               sgpr_demand -= definition.size() * live_sgprs.erase(definition.getTemp());
            }
         }
      }

      /* GEN */
      if (insn->opcode == aco_opcode::p_phi ||
          insn->opcode == aco_opcode::p_linear_phi) {
         /* directly insert into the predecessors live-out set */
         std::vector<Block*>& preds = insn->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
         for (unsigned i = 0; i < preds.size(); ++i)
         {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               auto it = live_temps[preds[i]->index].insert(operand.getTemp());
               /* check if we changed an already processed block */
               if (it.second)
                  worklist.insert(preds[i]->index);
            }
         }
         continue;
      }

      for (unsigned i = 0; i < insn->operandCount(); ++i)
      {
         auto& operand = insn->getOperand(i);
         if (operand.isTemp()) {
            if (operand.getTemp().type() == vgpr) {
               auto d = live_vgprs.insert(operand.getTemp());
               if (d.second)
                  vgpr_demand += operand.size();
            } else {
               auto d = live_sgprs.insert(operand.getTemp());
               if (d.second)
                  sgpr_demand += operand.size();
            }
         }
      }
      block->vgpr_demand = std::max(block->vgpr_demand, vgpr_demand);
      block->sgpr_demand = std::max(block->sgpr_demand, sgpr_demand);
   }

   /* now, we have the live-in sets and need to merge them into the live-out sets */
   for (Block* predecessor : block->logical_predecessors) {
      for (Temp vgpr : live_vgprs) {
         auto it = live_temps[predecessor->index].insert(vgpr);
         if (it.second)
            worklist.insert(predecessor->index);
      }
   }

   for (Block* predecessor : block->linear_predecessors) {
      for (Temp sgpr : live_sgprs) {
         auto it = live_temps[predecessor->index].insert(sgpr);
         if (it.second)
            worklist.insert(predecessor->index);
      }
   }

   assert(block->linear_predecessors.size() != 0 || (live_vgprs.empty() && live_sgprs.empty()));
   assert(block->linear_predecessors.size() != 0 || (vgpr_demand == 0 && sgpr_demand == 0));
}

std::vector<std::set<Temp>> live_temps_at_end_of_block(Program* program)
{
   std::vector<std::set<Temp>> result(program->blocks.size());
   std::set<unsigned> worklist;
   program->vgpr_demand = 0;
   program->sgpr_demand = 0;
   /* this implementation assumes that the block idx corresponds to the block's position in program->blocks vector */
   for (auto& block : program->blocks)
      worklist.insert(block->index);
   while (!worklist.empty()) {
      std::set<unsigned>::reverse_iterator b_it = worklist.rbegin();
      unsigned block_idx = *b_it;
      worklist.erase(block_idx);
      process_live_temps_per_block(result, program->blocks[block_idx].get(), worklist);
      program->vgpr_demand = std::max(program->vgpr_demand, program->blocks[block_idx]->vgpr_demand);
      program->sgpr_demand = std::max(program->sgpr_demand, program->blocks[block_idx]->sgpr_demand);
   }

   return result;
}

}

