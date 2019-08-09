/*
 * Copyright © 2019 Valve Corporation
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

#include <map>
#include "aco_ir.h"

/*
 * Implements an algorithm to lower to Concentional SSA Form (CSSA).
 * After "Revisiting Out-of-SSA Translation for Correctness, CodeQuality, and Efficiency"
 * by B. Boissinot, A. Darte, F. Rastello, B. Dupont de Dinechin, C. Guillon,
 *
 * By lowering the IR to CSSA, the insertion of parallelcopies is separated from
 * the register coalescing problem. Additionally, correctness is ensured w.r.t. spilling.
 * The algorithm tries to find beneficial insertion points by checking if a basic block
 * is empty and if the variable already has a new definition in a dominating block.
 */


namespace aco {
namespace {

struct phi_info {
   Operand op;
   uint16_t idx;
   aco_ptr<Instruction>& phi;
   int16_t merged = -1;

   phi_info(Operand op, unsigned idx, aco_ptr<Instruction>& phi) noexcept : op(op), idx(idx), phi(phi) {}

   bool operator<(const phi_info& rhs) const {
      if (phi->definitions[0].getTemp() < rhs.phi->definitions[0].getTemp())
         return true;
      if (phi->definitions[0].getTemp() == rhs.phi->definitions[0].getTemp()) {
         if (op.isConstant())
            return !rhs.op.isConstant() || op.constantValue() < rhs.op.constantValue();
         else
            return rhs.op.isTemp() && op.getTemp() < rhs.op.getTemp();
      }
      return false;
   }
};

struct copy_pos {
   unsigned block_idx;
   unsigned last_live_block;
   uint32_t pos;
};

struct cssa_ctx {
   Program* program;
   live& live_vars;
   std::vector<std::vector<phi_info>> phi_infos;
   std::map<phi_info, copy_pos> copy_positions;

   cssa_ctx(Program* program, live& live_vars) : program(program), live_vars(live_vars),
                                                 phi_infos(program->blocks.size()) {}
};

bool block_is_dominated_by(cssa_ctx& ctx, unsigned block_idx, unsigned idom_idx, bool is_linear) {
   Block& block = ctx.program->blocks[block_idx];
   unsigned next_idom = is_linear ? block.linear_idom : block.logical_idom;
   if (next_idom == idom_idx)
      return true;
   else if (next_idom > idom_idx)
      return block_is_dominated_by(ctx, next_idom, idom_idx, is_linear);
   else
      return false;
}


bool collect_phi_info(cssa_ctx& ctx)
{
   bool progress = false;
   for (Block& block : ctx.program->blocks) {
      for (aco_ptr<Instruction>& phi : block.instructions) {
         std::vector<unsigned> preds;
         if (phi->opcode == aco_opcode::p_phi)
            preds = block.logical_preds;
         else if (phi->opcode == aco_opcode::p_linear_phi)
            preds = block.linear_preds;
         else
            break;

         for (unsigned i = 0; i < phi->operands.size(); i++) {
            if (!phi->operands[i].isUndefined() && (phi->operands[i].isConstant() || !phi->operands[i].isKill())) {
               progress = true;
               ctx.phi_infos[preds[i]].emplace_back(phi->operands[i], i, phi);
            }
         }
      }
   }
   return progress;
}


void hoist_copies(cssa_ctx& ctx)
{
   for (unsigned block_idx = 0; block_idx < ctx.program->blocks.size(); block_idx++) {
      if (ctx.phi_infos[block_idx].empty())
         continue;

      std::vector<phi_info> phi_infos = std::move(ctx.phi_infos[block_idx]);
      for (phi_info info : phi_infos) {
         bool is_linear = info.phi->definitions[0].getTemp().is_linear();
         unsigned target_block = block_idx;

         /* check if the current block is empty */
         Block* block = &ctx.program->blocks[block_idx];
         std::vector<aco_ptr<Instruction>>::iterator instr_it = block->instructions.begin();
         while ((*instr_it)->opcode != aco_opcode::p_logical_start)
            ++instr_it;
         ++instr_it;

         bool empty = (*instr_it)->opcode == aco_opcode::p_logical_end;
         unsigned idom_idx = is_linear ? block->linear_idom : block->logical_idom;
         /* if the block is empty and idom is the only predecessor */
         if (empty &&
             ((is_linear && block->linear_preds.size() == 1) ||
              (!is_linear && block->logical_preds.size() == 1))) {
            /* check if register pressure is low enough at idom */
            Block& idom = ctx.program->blocks[idom_idx];
            RegisterDemand& reg_pressure = ctx.live_vars.register_demand[idom_idx][idom.instructions.size() -1];;

            if (info.phi->definitions[0].getTemp().type() == RegType::vgpr &&
                reg_pressure.vgpr + int16_t(info.phi->definitions[0].size()) <= ctx.program->max_reg_demand.vgpr) {
               reg_pressure.vgpr += info.phi->definitions[0].size();
               target_block = idom_idx;
            }

            if (info.phi->definitions[0].getTemp().type() == RegType::sgpr &&
                reg_pressure.sgpr +  int16_t(info.phi->definitions[0].size()) <= ctx.program->max_reg_demand.sgpr) {
               reg_pressure.sgpr += info.phi->definitions[0].size();
               target_block = idom_idx;
            }
         }

         /* check if this phi operand already has an insertion position which we can reuse */
         std::map<phi_info, copy_pos>::iterator it = ctx.copy_positions.find(info);
         if (it != ctx.copy_positions.end()) {
            /* check if the existing copy dominates the current one */
            bool is_dominated = block_is_dominated_by(ctx, target_block, it->second.block_idx, is_linear);

            if (is_dominated) {
               /* check if register pressure between existing copy and current insertion point is low enough */
               bool can_reuse = true;
               uint16_t size = info.phi->definitions[0].size();
               bool is_vgpr = info.phi->definitions[0].getTemp().type() == RegType::vgpr;
               for (unsigned i = it->second.last_live_block; i < target_block; i++) {
                  if ((is_vgpr && ctx.program->blocks[i].register_demand.vgpr + size > ctx.program->max_reg_demand.vgpr) ||
                      (!is_vgpr && ctx.program->blocks[i].register_demand.sgpr + size > ctx.program->max_reg_demand.sgpr)) {
                     can_reuse = false;
                     break;
                  }
               }

               if (can_reuse) {
                  /* update register demand */
                  for (unsigned i = it->second.last_live_block; i < target_block; i++) {
                     if (is_vgpr)
                        ctx.program->blocks[i].register_demand.vgpr += size;
                     else
                        ctx.program->blocks[i].register_demand.sgpr += size;
                  }

                  /* update map and info */
                  it->second.last_live_block = target_block;
                  info.merged = it->second.pos;
                  ctx.phi_infos[it->second.block_idx].emplace_back(info);
                  continue;
               }
            }
         }
         ctx.copy_positions[info] = {target_block, target_block + 1, (uint32_t) ctx.phi_infos[target_block].size()};
         ctx.phi_infos[target_block].emplace_back(info);
      }
   }
}


void emit_parallelcopies(cssa_ctx& ctx)
{
   for (unsigned block_idx = 0; block_idx < ctx.program->blocks.size(); block_idx++) {
      if (ctx.phi_infos[block_idx].empty())
         continue;

      Block& block = ctx.program->blocks[block_idx];
      /* find insertion point */
      std::vector<aco_ptr<Instruction>>::iterator it = block.instructions.end();
      --it;
      while ((*it)->opcode != aco_opcode::p_logical_end)
         --it;

      unsigned num = 0;
      for (phi_info& info : ctx.phi_infos[block_idx])
         if (info.merged == -1)
            num++;

      aco_ptr<Pseudo_instruction> copy{create_instruction<Pseudo_instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, num, num)};
      unsigned idx = 0;
      for (phi_info& info : ctx.phi_infos[block_idx]) {
         if (info.merged != -1) {
            info.phi->operands[info.idx] = ctx.phi_infos[block_idx][info.merged].op;
            continue;
         }
         copy->operands[idx] = info.op;
         Temp tmp = {ctx.program->allocateId(), info.phi->definitions[0].regClass()};
         copy->definitions[idx] = Definition(tmp);
         info.op = Operand(tmp);
         info.phi->operands[info.idx] = info.op;
         idx++;
      }

      block.instructions.insert(it, std::move(copy));
   }
}

} /* end namespace */

void lower_to_cssa(Program* program, live& live_vars, const struct radv_nir_compiler_options *options)
{
   cssa_ctx ctx = {program, live_vars};
   /* collect information about all interfering phi operands */
   bool progress = collect_phi_info(ctx);

   if (!progress)
      return;

   /* find good insertion points */
   hoist_copies(ctx);

   /* emit parallelcopies and rename operands */
   emit_parallelcopies(ctx);

   /* update live variable information */
   live_vars = live_var_analysis(program, options);

}
}
