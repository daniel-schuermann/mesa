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

template<bool reg_demand_cond>
void process_live_temps_per_block(live& lives, Block* block, std::set<unsigned>& worklist)
{
   std::vector<std::pair<uint16_t,uint16_t>>& register_demand = lives.register_demand[block->index];
   uint16_t vgpr_demand = 0;
   uint16_t sgpr_demand = 0;
   if (reg_demand_cond) {
      register_demand.resize(block->instructions.size());
      block->vgpr_demand = 0;
      block->sgpr_demand = 0;
   }
   std::set<Temp> live_sgprs;
   std::set<Temp> live_vgprs;

   /* first, insert the live-outs from this block into our temporary sets */
   std::vector<std::set<Temp>>& live_temps = lives.live_out;
   for (std::set<Temp>::iterator it = live_temps[block->index].begin(); it != live_temps[block->index].end(); ++it)
   {
      if ((*it).type() == vgpr) {
         live_vgprs.insert(*it);
         if (reg_demand_cond)
            vgpr_demand += (*it).size();
      } else {
         live_sgprs.insert(*it);
         if (reg_demand_cond)
            sgpr_demand += (*it).size();
      }
   }

   /* traverse the instructions backwards */
   for (int i = block->instructions.size() -1; i >= 0; i--)
   {
      if (reg_demand_cond)
         register_demand[i] = {sgpr_demand, vgpr_demand};

      Instruction *insn = block->instructions[i].get();
      /* KILL */
      for (unsigned i = 0; i < insn->definitionCount(); ++i)
      {
         auto& definition = insn->getDefinition(i);
         if (definition.isTemp()) {
            if (definition.getTemp().type() == vgpr) {
               size_t n = live_vgprs.erase(definition.getTemp());
               if (reg_demand_cond)
                  vgpr_demand -= definition.size() * n;
             } else {
               size_t n = live_sgprs.erase(definition.getTemp());
               if (reg_demand_cond)
                  sgpr_demand -= definition.size() * n;
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
      } else {
         for (unsigned i = 0; i < insn->operandCount(); ++i)
         {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               if (operand.getTemp().type() == vgpr) {
                  auto d = live_vgprs.insert(operand.getTemp());
                  if (reg_demand_cond && d.second)
                     vgpr_demand += operand.size();
               } else {
                  auto d = live_sgprs.insert(operand.getTemp());
                  if (reg_demand_cond && d.second)
                     sgpr_demand += operand.size();
               }
            }
         }
         if (reg_demand_cond) {
            block->vgpr_demand = std::max(block->vgpr_demand, vgpr_demand);
            block->sgpr_demand = std::max(block->sgpr_demand, sgpr_demand);
         }
      }
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
   assert(!reg_demand_cond || block->linear_predecessors.size() != 0 || (vgpr_demand == 0 && sgpr_demand == 0));
}

template<bool register_demand>
live live_var_analysis(Program* program)
{
   live result;
   result.live_out.resize(program->blocks.size());
   if (register_demand)
      result.register_demand.resize(program->blocks.size());
   std::set<unsigned> worklist;
   uint16_t vgpr_demand = 0;
   uint16_t sgpr_demand = 0;
   /* this implementation assumes that the block idx corresponds to the block's position in program->blocks vector */
   for (auto& block : program->blocks)
      worklist.insert(block->index);
   while (!worklist.empty()) {
      std::set<unsigned>::reverse_iterator b_it = worklist.rbegin();
      unsigned block_idx = *b_it;
      worklist.erase(block_idx);
      process_live_temps_per_block<register_demand>(result, program->blocks[block_idx].get(), worklist);
      vgpr_demand = std::max(vgpr_demand, program->blocks[block_idx]->vgpr_demand);
      sgpr_demand = std::max(sgpr_demand, program->blocks[block_idx]->sgpr_demand);
   }

   /* calculate the program's register demand and number of waves */
   if (register_demand) {
      // TODO: also take shared mem into account
      assert(vgpr_demand <= 256 && sgpr_demand <= 100);
      if (vgpr_demand <= 24 && sgpr_demand <= 46) {
         program->num_waves = 10;
         program->max_sgpr = 46;
         program->max_vgpr = 24;
      } else if (vgpr_demand <= 28 && sgpr_demand <= 54) {
         program->num_waves = 9;
         program->max_sgpr = 54;
         program->max_vgpr = 28;
      } else if (vgpr_demand <= 32 && sgpr_demand <= 62) {
         program->num_waves = 8;
         program->max_sgpr = 62;
         program->max_vgpr = 32;
      } else if (vgpr_demand <= 36 && sgpr_demand <= 70) {
         program->num_waves = 7;
         program->max_sgpr = 70;
         program->max_vgpr = 36;
      } else if (vgpr_demand <= 40 && sgpr_demand <= 78) {
         program->num_waves = 6;
         program->max_sgpr = 78;
         program->max_vgpr = 40;
      } else if (vgpr_demand <= 48 && sgpr_demand <= 94) {
         program->num_waves = 5;
         program->max_sgpr = 94;
         program->max_vgpr = 48;
      } else {
         program->max_sgpr = 100;
         if (vgpr_demand <= 64) {
            program->num_waves = 4;
            program->max_vgpr = 64;
         } else if (vgpr_demand <= 84) {
            program->num_waves = 3;
            program->max_vgpr = 84;
         } else if (vgpr_demand <= 128) {
            program->num_waves = 2;
            program->max_vgpr = 128;
         } else {
            program->num_waves = 1;
            program->max_vgpr = 256;
         }
      }
   }

   return result;
}
template live live_var_analysis<false>(Program* program);
template live live_var_analysis<true>(Program* program);

}

