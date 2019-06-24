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

#include "../vulkan/radv_shader.h"

namespace aco {
namespace {

void process_live_temps_per_block(Program *program, live& lives, Block* block, std::set<unsigned>& worklist)
{
   std::vector<std::pair<uint16_t,uint16_t>>& register_demand = lives.register_demand[block->index];
   uint16_t vgpr_demand = 0;
   uint16_t sgpr_demand = 0;

   register_demand.resize(block->instructions.size());
   block->vgpr_demand = 0;
   block->sgpr_demand = 0;

   std::set<Temp> live_sgprs;
   std::set<Temp> live_vgprs;

   /* add the live_out_exec to live */
   bool exec_live = false;
   if (block->live_out_exec != Temp()) {
      live_sgprs.insert(block->live_out_exec);
      sgpr_demand += 2;
      exec_live = true;
   }

   /* insert the live-outs from this block into our temporary sets */
   std::vector<std::set<Temp>>& live_temps = lives.live_out;
   for (std::set<Temp>::iterator it = live_temps[block->index].begin(); it != live_temps[block->index].end(); ++it)
   {
      bool inserted = false;
      if ((*it).is_linear())
         inserted = live_sgprs.insert(*it).second;
      else
         inserted = live_vgprs.insert(*it).second;
      if (inserted) {
         if (it->type() == vgpr)
            vgpr_demand += it->size();
         else
            sgpr_demand += it->size();
      }
   }

   /* traverse the instructions backwards */
   for (int idx = block->instructions.size() -1; idx >= 0; idx--)
   {
      /* substract the 2 sgprs from exec */
      if (exec_live)
         assert(sgpr_demand >= 2);
      register_demand[idx] = {sgpr_demand - (exec_live ? 2 : 0), vgpr_demand};
      block->vgpr_demand = std::max(block->vgpr_demand, register_demand[idx].second);
      block->sgpr_demand = std::max(block->sgpr_demand, register_demand[idx].first);

      Instruction *insn = block->instructions[idx].get();
      /* KILL */
      for (unsigned i = 0; i < insn->definitionCount(); ++i)
      {
         auto& definition = insn->getDefinition(i);
         if (definition.isTemp()) {
            size_t n = 0;
            if (definition.getTemp().is_linear())
               n = live_sgprs.erase(definition.getTemp());
            else
               n = live_vgprs.erase(definition.getTemp());

            if (n) {
               if (definition.getTemp().type() == vgpr)
                  vgpr_demand -= definition.size();
               else
                  sgpr_demand -= definition.size();
               definition.setKill(false);
            } else {
               if (definition.getTemp().type() == vgpr)
                  register_demand[idx].second += definition.size();
               else
                  register_demand[idx].first += definition.size();
               definition.setKill(true);
            }

            if (definition.isFixed() && definition.physReg() == exec)
               exec_live = false;
         }
      }

      /* GEN */
      if (insn->opcode == aco_opcode::p_phi ||
          insn->opcode == aco_opcode::p_linear_phi) {
         /* directly insert into the predecessors live-out set */
         std::vector<unsigned>& preds = insn->opcode == aco_opcode::p_phi ? block->logical_preds : block->linear_preds;
         for (unsigned i = 0; i < preds.size(); ++i)
         {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               auto it = live_temps[preds[i]].insert(operand.getTemp());
               /* check if we changed an already processed block */
               if (it.second) {
                  operand.setFirstKill(true);
                  worklist.insert(preds[i]);
               }
            }
         }
      } else {
         for (unsigned i = 0; i < insn->operandCount(); ++i)
         {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               bool inserted = false;
               if (operand.getTemp().is_linear())
                  inserted = live_sgprs.insert(operand.getTemp()).second;
               else
                  inserted = live_vgprs.insert(operand.getTemp()).second;

               if (inserted) {
                  operand.setFirstKill(true);
                  for (unsigned j = i + 1; j < insn->operandCount(); ++j) {
                     if (insn->getOperand(j).isTemp() && insn->getOperand(j).tempId() == operand.tempId()) {
                        insn->getOperand(j).setFirstKill(false);
                        insn->getOperand(j).setKill(true);
                     }
                  }

                  if (operand.getTemp().type() == vgpr)
                     vgpr_demand += operand.size();
                  else
                     sgpr_demand += operand.size();
               } else {
                  operand.setKill(false);
               }

               if (operand.isFixed() && operand.physReg() == exec)
                  exec_live = true;
            }
         }
      }
   }

   /* now, we have the live-in sets and need to merge them into the live-out sets */
   for (unsigned pred_idx : block->logical_preds) {
      for (Temp vgpr : live_vgprs) {
         auto it = live_temps[pred_idx].insert(vgpr);
         if (it.second)
            worklist.insert(pred_idx);
      }
   }

   for (unsigned pred_idx : block->linear_preds) {
      for (Temp sgpr : live_sgprs) {
         auto it = live_temps[pred_idx].insert(sgpr);
         if (it.second)
            worklist.insert(pred_idx);
      }
   }

   if (!(block->index != 0 || (live_vgprs.empty() && live_sgprs.empty()))) {
      aco_print_program(program, stderr);
      fprintf(stderr, "These temporaries are never defined or are defined after use:\n");
      for (Temp vgpr : live_vgprs)
         fprintf(stderr, "%%%d\n", vgpr.id());
      for (Temp sgpr : live_sgprs)
         fprintf(stderr, "%%%d\n", sgpr.id());
      abort();
   }

   assert(block->index != 0 || (vgpr_demand == 0 && (sgpr_demand == 0)));
}
} /* end namespace */

void update_vgpr_sgpr_demand(Program* program, unsigned vgpr, unsigned sgpr)
{
   // TODO: also take shared mem into account
   uint16_t total_sgpr_regs = program->chip_class >= GFX8 ? 800 : 512;
   uint16_t max_addressible_sgpr = program->chip_class >= GFX8 ? 102 : 104;
   /* VGPRs are allocated in chunks of 4 */
   uint16_t rounded_vgpr_demand = std::max<uint16_t>(4, (vgpr + 3) & ~3);
   /* SGPRs are allocated in chunks of 16 between 8 and 104. VCC occupies the last 2 registers */
   uint16_t rounded_sgpr_demand = std::min(std::max<uint16_t>(8, (sgpr + 2 + 7) & ~7), max_addressible_sgpr);
   /* this won't compile, register pressure reduction necessary */
   if (vgpr > 256 || sgpr > max_addressible_sgpr) {
      program->num_waves = 0;
      program->max_sgpr = sgpr;
      program->max_vgpr = vgpr;
   } else {
      program->num_waves = std::min<uint16_t>(10,
                                              std::min<uint16_t>(256 / rounded_vgpr_demand,
                                                                 total_sgpr_regs / rounded_sgpr_demand));

      program->max_sgpr = std::min<uint16_t>(((total_sgpr_regs / program->num_waves) & ~7) - 2, max_addressible_sgpr);
      program->max_vgpr = (256 / program->num_waves) & ~3;
   }
}

live live_var_analysis(Program* program,
                       const struct radv_nir_compiler_options *options)
{
   live result;
   result.live_out.resize(program->blocks.size());
   result.register_demand.resize(program->blocks.size());
   std::set<unsigned> worklist;
   uint16_t vgpr_demand = 0;
   uint16_t sgpr_demand = 0;

   /* this implementation assumes that the block idx corresponds to the block's position in program->blocks vector */
   for (Block& block : program->blocks)
      worklist.insert(block.index);
   while (!worklist.empty()) {
      std::set<unsigned>::reverse_iterator b_it = worklist.rbegin();
      unsigned block_idx = *b_it;
      worklist.erase(block_idx);
      process_live_temps_per_block(program, result, &program->blocks[block_idx], worklist);
      vgpr_demand = std::max(vgpr_demand, program->blocks[block_idx].vgpr_demand);
      sgpr_demand = std::max(sgpr_demand, program->blocks[block_idx].sgpr_demand);
   }

   /* calculate the program's register demand and number of waves */
   update_vgpr_sgpr_demand(program, vgpr_demand, sgpr_demand);

   return result;
}

}

