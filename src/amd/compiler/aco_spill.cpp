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
 */

#ifndef ACO_SPILL_CPP
#define ACO_SPILL_CPP

#include "aco_ir.h"
#include <map>
#include <stack>
#include "../vulkan/radv_shader.h"


/*
 * Implements the spilling algorithm on SSA-form from
 * "Register Spilling and Live-Range Splitting for SSA-Form Programs"
 * by Matthias Braun and Sebastian Hack.
 */

namespace aco {

namespace {

struct remat_info {
   Instruction *instr;
};

struct spill_ctx {
   uint16_t target_vgpr;
   uint16_t target_sgpr;
   Program* program;
   std::vector<std::vector<std::pair<uint16_t,uint16_t>>> register_demand;
   std::vector<std::map<Temp, Temp>> renames;
   std::vector<std::map<Temp, uint32_t>> spills_entry;
   std::vector<std::map<Temp, uint32_t>> spills_exit;
   std::vector<bool> processed;
   std::stack<Block*> loop_header;
   std::vector<std::map<Temp, std::pair<uint32_t, uint32_t>>> next_use_distances_start;
   std::vector<std::map<Temp, std::pair<uint32_t, uint32_t>>> next_use_distances_end;
   std::vector<std::pair<RegClass, std::set<uint32_t>>> interferences;
   std::vector<std::pair<uint32_t, uint32_t>> affinities;
   std::map<uint32_t, unsigned> reload_count;
   std::map<Temp, remat_info> remat;
   std::map<Instruction *, unsigned> remat_use_count;

   spill_ctx(uint16_t target_vgpr, uint16_t target_sgpr, Program* program,
             std::vector<std::vector<std::pair<uint16_t,uint16_t>>> register_demand)
      : target_vgpr(target_vgpr), target_sgpr(target_sgpr), program(program),
        register_demand(register_demand), renames(program->blocks.size()),
        spills_entry(program->blocks.size()), spills_exit(program->blocks.size()),
        processed(program->blocks.size(), false) {}

   uint32_t allocate_spill_id(RegClass rc)
   {
      interferences.emplace_back(rc, std::set<uint32_t>());
      return next_spill_id++;
   }

   uint32_t next_spill_id = 0;
};

int32_t get_dominator(int idx_a, int idx_b, Program* program, bool is_linear)
{

   if (idx_a == -1)
      return idx_b;
   if (idx_b == -1)
      return idx_a;
   if (is_linear) {
      while (idx_a != idx_b) {
         if (idx_a > idx_b)
            idx_a = program->blocks[idx_a]->linear_idom;
         else
            idx_b = program->blocks[idx_b]->linear_idom;
      }
   } else {
      while (idx_a != idx_b) {
         if (idx_a > idx_b)
            idx_a = program->blocks[idx_a]->logical_idom;
         else
            idx_b = program->blocks[idx_b]->logical_idom;
      }
   }
   assert(idx_a != -1);
   return idx_a;
}

void next_uses_per_block(spill_ctx& ctx, unsigned block_idx, std::set<uint32_t>& worklist)
{
   std::unique_ptr<Block>& block = ctx.program->blocks[block_idx];
   std::map<Temp, std::pair<uint32_t, uint32_t>> next_uses = ctx.next_use_distances_end[block_idx];

   /* to compute the next use distance at the beginning of the block, we have to add the block's size */
   for (std::map<Temp, std::pair<uint32_t, uint32_t>>::iterator it = next_uses.begin(); it != next_uses.end();) {
      it->second.second = it->second.second + block->instructions.size();

      /* remove the live out exec mask as we really don't want to spill it */
      if (it->first == block->live_out_exec)
         it = next_uses.erase(it);
      else
         ++it;
   }

   int idx = block->instructions.size() - 1;
   while (idx >= 0) {
      aco_ptr<Instruction>& instr = block->instructions[idx];

      if (instr->opcode == aco_opcode::p_linear_phi ||
          instr->opcode == aco_opcode::p_phi)
         break;

      for (unsigned i = 0; i < instr->num_definitions; i++) {
         if (instr->getDefinition(i).isTemp())
            next_uses.erase(instr->getDefinition(i).getTemp());
      }

      for (unsigned i = 0; i < instr->num_operands; i++) {
         /* omit exec mask */
         if (instr->getOperand(i).isFixed() && instr->getOperand(i).physReg() == exec)
            continue;
         if (instr->getOperand(i).isTemp())
            next_uses[instr->getOperand(i).getTemp()] = {block_idx, idx};
      }
      idx--;
   }

   assert(block_idx != 0 || next_uses.empty());
   ctx.next_use_distances_start[block_idx] = next_uses;
   while (idx >= 0) {
      aco_ptr<Instruction>& instr = block->instructions[idx];
      assert(instr->opcode == aco_opcode::p_linear_phi || instr->opcode == aco_opcode::p_phi);

      for (unsigned i = 0; i < instr->num_operands; i++) {
         unsigned pred_idx = instr->opcode == aco_opcode::p_phi ?
                             block->logical_predecessors[i]->index :
                             block->linear_predecessors[i]->index;
         if (instr->getOperand(i).isTemp()) {
            if (ctx.next_use_distances_end[pred_idx].find(instr->getOperand(i).getTemp()) == ctx.next_use_distances_end[pred_idx].end() ||
                ctx.next_use_distances_end[pred_idx][instr->getOperand(i).getTemp()] != std::pair<uint32_t, uint32_t>{block_idx, 0})
               worklist.insert(pred_idx);
            ctx.next_use_distances_end[pred_idx][instr->getOperand(i).getTemp()] = {block_idx, 0};
         }
      }
      next_uses.erase(instr->getDefinition(0).getTemp());
      idx--;
   }

   /* all remaining live vars must be live-out at the predecessors */
   for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : next_uses) {
      Temp temp = pair.first;
      uint32_t distance = pair.second.second;
      uint32_t dom = pair.second.first;
      std::vector<Block*>& preds = temp.is_linear() ? block->linear_predecessors : block->logical_predecessors;
      for (Block* pred : preds) {
         if (pred->loop_nest_depth > block->loop_nest_depth)
            distance += 0xFFFF;
         if (ctx.next_use_distances_end[pred->index].find(temp) != ctx.next_use_distances_end[pred->index].end()) {
            dom = get_dominator(dom, ctx.next_use_distances_end[pred->index][temp].first, ctx.program, temp.is_linear());
            distance = std::min(ctx.next_use_distances_end[pred->index][temp].second, distance);
         }
         if (ctx.next_use_distances_end[pred->index][temp] != std::pair<uint32_t, uint32_t>{dom, distance})
            worklist.insert(pred->index);
         ctx.next_use_distances_end[pred->index][temp] = {dom, distance};
      }
   }

}

void compute_global_next_uses(spill_ctx& ctx, std::vector<std::set<Temp>>& live_out)
{
   ctx.next_use_distances_start.resize(ctx.program->blocks.size());
   ctx.next_use_distances_end.resize(ctx.program->blocks.size());
   std::set<uint32_t> worklist;
   for (auto& block : ctx.program->blocks)
      worklist.insert(block->index);

   while (!worklist.empty()) {
      std::set<unsigned>::reverse_iterator b_it = worklist.rbegin();
      unsigned block_idx = *b_it;
      worklist.erase(block_idx);
      next_uses_per_block(ctx, block_idx, worklist);
   }
}

bool should_rematerialize(aco_ptr<Instruction>& instr)
{
   /* TODO: rematerialization is only supported for VOP1 and SOP1 */
   if (instr->format != Format::VOP1 && instr->format != Format::SOP1)
      return false;

   for (unsigned i = 0; i < instr->num_operands; i++) {
      /* TODO: rematerialization using temporaries isn't yet supported */
      if (instr->getOperand(i).isTemp())
         return false;
   }

   /* TODO: rematerialization with multiple definitions isn't yet supported */
   if (instr->num_definitions > 1)
      return false;

   return true;
}

aco_ptr<Instruction> do_reload(spill_ctx& ctx, Temp tmp, Temp new_name, uint32_t spill_id)
{
   std::map<Temp, remat_info>::iterator remat = ctx.remat.find(tmp);
   if (remat != ctx.remat.end()) {
      Instruction *instr = remat->second.instr;
      assert((instr->format == Format::VOP1 || instr->format == Format::SOP1) && "unsupported");
      assert(instr->num_definitions == 1 && "unsupported");

      aco_ptr<Instruction> res;
      if (instr->format == Format::VOP1) {
         res.reset(create_instruction<VOP1_instruction>(instr->opcode, instr->format, instr->num_operands, instr->num_definitions));
      } else if (instr->format == Format::SOP1) {
         res.reset(create_instruction<SOP1_instruction>(instr->opcode, instr->format, instr->num_operands, instr->num_definitions));
      }
      for (unsigned i = 0; i < instr->num_operands; i++) {
         res->getOperand(i) = instr->getOperand(i);
         if (instr->getOperand(i).isTemp()) {
            assert(false && "unsupported");
            if (ctx.remat.count(instr->getOperand(i).getTemp()))
               ctx.remat_use_count[ctx.remat[instr->getOperand(i).getTemp()].instr]++;
         }
      }
      res->getDefinition(0) = Definition(new_name);
      return res;
   } else {
      aco_ptr<Instruction> reload{create_instruction<Instruction>(aco_opcode::p_reload, Format::PSEUDO, 1, 1)};
      reload->getOperand(0) = Operand(spill_id);
      reload->getDefinition(0) = Definition(new_name);
      ctx.reload_count[spill_id]++;
      return reload;
   }
}

void get_rematerialize_info(spill_ctx& ctx)
{
   for (auto&& block : ctx.program->blocks) {
      bool logical = false;
      for (aco_ptr<Instruction>& instr : block->instructions) {
         if (instr->opcode == aco_opcode::p_logical_start)
            logical = true;
         else if (instr->opcode == aco_opcode::p_logical_end)
            logical = false;
         if (logical && should_rematerialize(instr)) {
            for (unsigned i = 0; i < instr->num_definitions; i++) {
               if (instr->getDefinition(i).isTemp()) {
                  ctx.remat[instr->getDefinition(i).getTemp()] = (remat_info){instr.get()};
                  ctx.remat_use_count[instr.get()] = 0;
               }
            }
         }
      }
   }
}

std::vector<std::map<Temp, uint32_t>> local_next_uses(spill_ctx& ctx, std::unique_ptr<Block>& block)
{
   std::vector<std::map<Temp, uint32_t>> local_next_uses(block->instructions.size());

   std::map<Temp, uint32_t> next_uses;
   for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_end[block->index]) {
      /* omit live out exec mask */
      if (pair.first == block->live_out_exec)
         continue;

      next_uses[pair.first] = pair.second.second + block->instructions.size();
   }

   for (int idx = block->instructions.size() - 1; idx >= 0; idx--) {
      aco_ptr<Instruction>& instr = block->instructions[idx];
      if (!instr)
         break;
      if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi)
         break;

      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (instr->getOperand(i).isFixed() && instr->getOperand(i).physReg() == exec)
            continue;
         if (instr->getOperand(i).isTemp())
            next_uses[instr->getOperand(i).getTemp()] = idx;
      }
      for (unsigned i = 0; i < instr->num_definitions; i++) {
         if (instr->getDefinition(i).isTemp())
            next_uses.erase(instr->getDefinition(i).getTemp());
      }
      local_next_uses[idx] = next_uses;
   }
   return local_next_uses;
}


std::pair<unsigned, unsigned> init_live_in_vars(spill_ctx& ctx, Block* block, unsigned block_idx)
{
   unsigned spilled_sgprs = 0;
   unsigned spilled_vgprs = 0;

   /* first block, nothing was spilled before */
   if (block_idx == 0)
      return {0, 0};

   /* loop header block */
   if (block->loop_nest_depth > ctx.program->blocks[block_idx - 1]->loop_nest_depth) {
      assert(block->linear_predecessors[0]->index == block_idx - 1);
      assert(block->logical_predecessors[0]->index == block_idx - 1);

      /* create new loop_info */
      ctx.loop_header.emplace(block);

      /* check how many live-through variables should be spilled */
      uint16_t sgpr_demand = 0, vgpr_demand = 0;
      unsigned i = block_idx;
      while (ctx.program->blocks[i]->loop_nest_depth >= block->loop_nest_depth) {
         assert(ctx.program->blocks.size() > i);
         sgpr_demand = std::max(sgpr_demand, ctx.program->blocks[i]->sgpr_demand);
         vgpr_demand = std::max(vgpr_demand, ctx.program->blocks[i]->vgpr_demand);
         i++;
      }
      unsigned loop_end = i;

      /* select live-through vgpr variables */
      while (vgpr_demand - spilled_vgprs > ctx.target_vgpr) {
         unsigned distance = 0;
         Temp to_spill;
         for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_end[block_idx - 1]) {
            if (pair.first.type() == vgpr &&
                pair.second.first >= loop_end &&
                pair.second.second > distance &&
                ctx.spills_entry[block_idx].find(pair.first) == ctx.spills_entry[block_idx].end()) {
               to_spill = pair.first;
               distance = pair.second.second;
            }
         }
         if (distance == 0)
            break;

         uint32_t spill_id;
         if (ctx.spills_exit[block_idx - 1].find(to_spill) == ctx.spills_exit[block_idx - 1].end()) {
            spill_id = ctx.allocate_spill_id(to_spill.regClass());
         } else {
            spill_id = ctx.spills_exit[block_idx - 1][to_spill];
         }

         ctx.spills_entry[block_idx][to_spill] = spill_id;
         spilled_vgprs += to_spill.size();
      }

      /* select live-through sgpr variables */
      while (sgpr_demand - spilled_sgprs > ctx.target_sgpr) {
         unsigned distance = 0;
         Temp to_spill;
         for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_end[block_idx - 1]) {
            if (pair.first.type() == sgpr &&
                pair.second.first >= loop_end &&
                pair.second.second > distance &&
                ctx.spills_entry[block_idx].find(pair.first) == ctx.spills_entry[block_idx].end()) {
               to_spill = pair.first;
               distance = pair.second.second;
            }
         }
         if (distance == 0)
            break;

         uint32_t spill_id;
         if (ctx.spills_exit[block_idx - 1].find(to_spill) == ctx.spills_exit[block_idx - 1].end()) {
            spill_id = ctx.allocate_spill_id(to_spill.regClass());
         } else {
            spill_id = ctx.spills_exit[block_idx - 1][to_spill];
         }

         ctx.spills_entry[block_idx][to_spill] = spill_id;
         spilled_sgprs += to_spill.size();
      }



      /* shortcut */
      if (vgpr_demand - spilled_vgprs <= ctx.target_vgpr &&
          sgpr_demand - spilled_sgprs <= ctx.target_sgpr)
         return {spilled_sgprs, spilled_vgprs};

      /* if reg pressure is too high at beginning of loop, add variables with furthest use */
      unsigned idx = 0;
      while (block->instructions[idx]->opcode == aco_opcode::p_phi || block->instructions[idx]->opcode == aco_opcode::p_linear_phi)
         idx++;

      assert(idx != 0 && "loop without phis: TODO");
      idx--;
      int32_t reg_pressure_sgpr = ctx.register_demand[block_idx][idx].first - spilled_sgprs;
      int32_t reg_pressure_vgpr = ctx.register_demand[block_idx][idx].second - spilled_vgprs;
      while (reg_pressure_sgpr > ctx.target_sgpr) {
         unsigned distance = 0;
         Temp to_spill;
         for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_start[block_idx]) {
            if (pair.first.type() == sgpr &&
                pair.second.second > distance &&
                ctx.spills_entry[block_idx].find(pair.first) == ctx.spills_entry[block_idx].end()) {
               to_spill = pair.first;
               distance = pair.second.second;
            }
         }
         assert(distance != 0);

         ctx.spills_entry[block_idx][to_spill] = ctx.allocate_spill_id(to_spill.regClass());
         spilled_sgprs += to_spill.size();
         reg_pressure_sgpr -= to_spill.size();
      }
      while (reg_pressure_vgpr > ctx.target_vgpr) {
         unsigned distance = 0;
         Temp to_spill;
         for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_start[block_idx]) {
            if (pair.first.type() == vgpr &&
                pair.second.second > distance &&
                ctx.spills_entry[block_idx].find(pair.first) == ctx.spills_entry[block_idx].end()) {
               to_spill = pair.first;
               distance = pair.second.second;
            }
         }
         assert(distance != 0);
         ctx.spills_entry[block_idx][to_spill] = ctx.allocate_spill_id(to_spill.regClass());
         spilled_vgprs += to_spill.size();
         reg_pressure_vgpr -= to_spill.size();
      }

      return {spilled_sgprs, spilled_vgprs};
   }

   /* branch block */
   if (block->linear_predecessors.size() == 1) {
      /* keep variables spilled if they are alive and not used in the current block */
      unsigned pred_idx = block->linear_predecessors[0]->index;
      for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
         if (pair.first.type() == sgpr &&
             ctx.next_use_distances_start[block_idx].find(pair.first) != ctx.next_use_distances_start[block_idx].end() &&
             ctx.next_use_distances_start[block_idx][pair.first].second > block_idx) {
            ctx.spills_entry[block_idx].insert(pair);
            spilled_sgprs += pair.first.size();
         }
      }
      if (block->logical_predecessors.size() == 1) {
         pred_idx = block->logical_predecessors[0]->index;
         for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
            if (pair.first.type() == vgpr &&
                ctx.next_use_distances_start[block_idx].find(pair.first) != ctx.next_use_distances_start[block_idx].end() &&
                ctx.next_use_distances_end[pred_idx][pair.first].second > block_idx) {
               ctx.spills_entry[block_idx].insert(pair);
               spilled_vgprs += pair.first.size();
            }
         }
      }

      /* if register demand is still too high, we just keep all spilled live vars and process the block */
      if (block->sgpr_demand - spilled_sgprs > ctx.target_sgpr) {
         pred_idx = block->linear_predecessors[0]->index;
         for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
            if (pair.first.type() == sgpr &&
                ctx.next_use_distances_start[block_idx].find(pair.first) != ctx.next_use_distances_start[block_idx].end() &&
                ctx.spills_entry[block_idx].insert(pair).second) {
               spilled_sgprs += pair.first.size();
            }
         }
      }
      if (block->vgpr_demand - spilled_vgprs > ctx.target_vgpr && block->logical_predecessors.size() == 1) {
         pred_idx = block->logical_predecessors[0]->index;
         for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
            if (pair.first.type() == vgpr &&
                ctx.next_use_distances_start[block_idx].find(pair.first) != ctx.next_use_distances_start[block_idx].end() &&
                ctx.spills_entry[block_idx].insert(pair).second) {
               spilled_vgprs += pair.first.size();
            }
         }
      }

      return {spilled_sgprs, spilled_vgprs};
   }

   /* else: merge block */
   assert(block->linear_predecessors.size() == 2);
   std::set<Temp> partial_spills;

   /* keep variables spilled on all incoming paths */
   for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_start[block_idx]) {
      std::vector<Block*>& preds = pair.first.type() == vgpr ? block->logical_predecessors : block->linear_predecessors;
      /* If it can be rematerialized, keep the variable spilled if all predecessors do not reload it.
       * Otherwise, if any predecessor reloads it, ensure it's reloaded on all other predecessors.
       * The idea is that it's better in practice to rematerialize redundantly than to create lots of phis. */
      /* TODO: test this idea with more than Dawn of War III shaders (the current pipeline-db doesn't seem to exercise this path much) */
      bool remat = ctx.remat.count(pair.first);
      bool spill = !remat;
      uint32_t spill_id = 0;
      for (Block* pred : preds) {
         /* variable is not even live at the predecessor: probably from a phi */
         if (ctx.next_use_distances_end[pred->index].find(pair.first) == ctx.next_use_distances_end[pred->index].end()) {
            spill = false;
            break;
         }
         if (ctx.spills_exit[pred->index].find(pair.first) == ctx.spills_exit[pred->index].end()) {
            if (!remat)
               spill = false;
         } else {
            partial_spills.insert(pair.first);
            /* it might be that on one incoming path, the variable has a different spill_id, but add_couple_code() will take care of that. */
            spill_id = ctx.spills_exit[pred->index][pair.first];
            if (remat)
               spill = true;
         }
      }
      if (spill) {
         ctx.spills_entry[block_idx][pair.first] = spill_id;
         partial_spills.erase(pair.first);
         if (pair.first.type() == vgpr)
            spilled_vgprs += pair.first.size();
         else
            spilled_sgprs += pair.first.size();
      }
   }

   /* same for phis */
   unsigned idx = 0;
   while (block->instructions[idx]->opcode == aco_opcode::p_linear_phi ||
          block->instructions[idx]->opcode == aco_opcode::p_phi) {
      aco_ptr<Instruction>& phi = block->instructions[idx];
      std::vector<Block*>& preds = phi->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
      bool spill = true;

      for (unsigned i = 0; i < phi->num_operands; i++) {
         if (!phi->getOperand(i).isTemp())
            spill = false;
         else if (ctx.spills_exit[preds[i]->index].find(phi->getOperand(i).getTemp()) == ctx.spills_exit[preds[i]->index].end())
            spill = false;
         else
            partial_spills.insert(phi->getDefinition(0).getTemp());
      }
      if (spill) {
         ctx.spills_entry[block_idx][phi->getDefinition(0).getTemp()] = ctx.allocate_spill_id(phi->getDefinition(0).regClass());
         partial_spills.erase(phi->getDefinition(0).getTemp());
         if (phi->getDefinition(0).getTemp().type() == vgpr)
            spilled_vgprs += phi->getDefinition(0).getTemp().size();
         else
            spilled_sgprs += phi->getDefinition(0).getTemp().size();
      }

      idx++;
   }

   /* if reg pressure at first instruction is still too high, add partially spilled variables */
   int32_t reg_pressure_sgpr = 0, reg_pressure_vgpr = 0;
   if (idx == 0) {
      for (unsigned i = 0; i < block->instructions[idx]->num_definitions; i++) {
         if (block->instructions[idx]->getDefinition(i).isTemp()) {
            if (block->instructions[idx]->getDefinition(i).getTemp().type() == vgpr)
               reg_pressure_vgpr -= block->instructions[idx]->getDefinition(i).size();
            else
               reg_pressure_sgpr -= block->instructions[idx]->getDefinition(i).size();
         }
      }
      for (unsigned i = 0; i < block->instructions[idx]->num_operands; i++) {
         if (block->instructions[idx]->getOperand(i).isTemp() &&
             block->instructions[idx]->getOperand(i).isFirstKill()) {
            if (block->instructions[idx]->getOperand(i).getTemp().type() == vgpr)
               reg_pressure_vgpr += block->instructions[idx]->getOperand(i).size();
            else
               reg_pressure_sgpr += block->instructions[idx]->getOperand(i).size();
         }
      }
   } else {
      idx--;
   }
   reg_pressure_sgpr += ctx.register_demand[block_idx][idx].first - spilled_sgprs;
   reg_pressure_vgpr += ctx.register_demand[block_idx][idx].second - spilled_vgprs;

   while (reg_pressure_sgpr > ctx.target_sgpr) {
      assert(!partial_spills.empty());

      std::set<Temp>::iterator it = partial_spills.begin();
      Temp to_spill = *it;
      unsigned distance = ctx.next_use_distances_start[block_idx][*it].second;
      while (it != partial_spills.end()) {
         assert(ctx.spills_entry[block_idx].find(*it) == ctx.spills_entry[block_idx].end());

         if (it->type() == sgpr && ctx.next_use_distances_start[block_idx][*it].second > distance) {
            distance = ctx.next_use_distances_start[block_idx][*it].second;
            to_spill = *it;
         }
         ++it;
      }
      assert(distance != 0);

      ctx.spills_entry[block_idx][to_spill] = ctx.allocate_spill_id(to_spill.regClass());
      spilled_sgprs += to_spill.size();
      reg_pressure_sgpr -= to_spill.size();
   }

   while (reg_pressure_vgpr > ctx.target_vgpr) {
      assert(!partial_spills.empty());

      std::set<Temp>::iterator it = partial_spills.begin();
      Temp to_spill = *it;
      unsigned distance = ctx.next_use_distances_start[block_idx][*it].second;
      while (it != partial_spills.end()) {
         assert(ctx.spills_entry[block_idx].find(*it) == ctx.spills_entry[block_idx].end());

         if (it->type() == vgpr && ctx.next_use_distances_start[block_idx][*it].second > distance) {
            distance = ctx.next_use_distances_start[block_idx][*it].second;
            to_spill = *it;
         }
         ++it;
      }
      assert(distance != 0);

      ctx.spills_entry[block_idx][to_spill] = ctx.allocate_spill_id(to_spill.regClass());
      spilled_vgprs += to_spill.size();
      reg_pressure_vgpr -= to_spill.size();
   }

   return {spilled_sgprs, spilled_vgprs};
}


void add_coupling_code(spill_ctx& ctx, Block* block, unsigned block_idx)
{
   /* no coupling code necessary */
   if (block->linear_predecessors.size() == 0)
      return;

   std::vector<aco_ptr<Instruction>> instructions;
   /* branch block: TODO take other branch into consideration */
   if (block->linear_predecessors.size() == 1) {
      assert(ctx.processed[block->linear_predecessors[0]->index]);

      if (block->logical_predecessors.size() == 1) {
         unsigned pred_idx = block->logical_predecessors[0]->index;
         for (std::pair<Temp, std::pair<uint32_t, uint32_t>> live : ctx.next_use_distances_start[block_idx]) {
            if (live.first.type() == sgpr)
               continue;
            /* still spilled */
            if (ctx.spills_entry[block_idx].find(live.first) != ctx.spills_entry[block_idx].end())
               continue;

            /* in register at end of predecessor */
            if (ctx.spills_exit[pred_idx].find(live.first) == ctx.spills_exit[pred_idx].end()) {
               std::map<Temp, Temp>::iterator it = ctx.renames[pred_idx].find(live.first);
               if (it != ctx.renames[pred_idx].end())
                  ctx.renames[block_idx].insert(*it);
               continue;
            }

            /* variable is spilled at predecessor and live at current block: create reload instruction */
            Temp new_name = {ctx.program->allocateId(), live.first.regClass()};
            aco_ptr<Instruction> reload = do_reload(ctx, live.first, new_name, ctx.spills_exit[pred_idx][live.first]);
            instructions.emplace_back(std::move(reload));
            ctx.renames[block_idx][live.first] = new_name;
         }
      }

      unsigned pred_idx = block->linear_predecessors[0]->index;
      for (std::pair<Temp, std::pair<uint32_t, uint32_t>> live : ctx.next_use_distances_start[block_idx]) {
         if (live.first.type() == vgpr)
            continue;
         /* still spilled */
         if (ctx.spills_entry[block_idx].find(live.first) != ctx.spills_entry[block_idx].end())
            continue;

         /* in register at end of predecessor */
         if (ctx.spills_exit[pred_idx].find(live.first) == ctx.spills_exit[pred_idx].end()) {
            std::map<Temp, Temp>::iterator it = ctx.renames[pred_idx].find(live.first);
            if (it != ctx.renames[pred_idx].end())
               ctx.renames[block_idx].insert(*it);
            continue;
         }

         /* variable is spilled at predecessor and live at current block: create reload instruction */
         Temp new_name = {ctx.program->allocateId(), live.first.regClass()};
         aco_ptr<Instruction> reload = do_reload(ctx, live.first, new_name, ctx.spills_exit[pred_idx][live.first]);
         instructions.emplace_back(std::move(reload));
         ctx.renames[block_idx][live.first] = new_name;
      }

      /* combine new reload instructions with original block */
      if (!instructions.empty()) {
         unsigned insert_idx = 0;
         while (block->instructions[insert_idx]->opcode == aco_opcode::p_phi ||
                block->instructions[insert_idx]->opcode == aco_opcode::p_linear_phi) {
            insert_idx++;
         }
         ctx.register_demand[block->index].insert(std::next(ctx.register_demand[block->index].begin(), insert_idx),
                                                  instructions.size(), std::pair<uint16_t, uint16_t>(0, 0));
         block->instructions.insert(std::next(block->instructions.begin(), insert_idx),
                                    std::move_iterator<std::vector<aco_ptr<Instruction>>::iterator>(instructions.begin()),
                                    std::move_iterator<std::vector<aco_ptr<Instruction>>::iterator>(instructions.end()));
      }
      return;
   }

   /* loop header and merge blocks: check if all (linear) predecessors have been processed */
   for (Block* pred : block->linear_predecessors)
      assert(ctx.processed[pred->index]);

   /* iterate the phi nodes for which operands to spill at the predecessor */
   for (aco_ptr<Instruction>& phi : block->instructions) {
      if (phi->opcode != aco_opcode::p_phi &&
          phi->opcode != aco_opcode::p_linear_phi)
         break;

      /* if the phi is not spilled, add to instructions */
      if (ctx.spills_entry[block_idx].find(phi->getDefinition(0).getTemp()) == ctx.spills_entry[block_idx].end()) {
         instructions.emplace_back(std::move(phi));
         continue;
      }

      std::vector<Block*>& preds = phi->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
      uint32_t def_spill_id = ctx.spills_entry[block_idx][phi->getDefinition(0).getTemp()];

      for (unsigned i = 0; i < phi->num_operands; i++) {
         unsigned pred_idx = preds[i]->index;

         /* we have to spill constants to the same memory address */
         if (phi->getOperand(i).isConstant()) {
            uint32_t spill_id = ctx.allocate_spill_id(phi->getDefinition(0).regClass());
            for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
               ctx.interferences[def_spill_id].second.emplace(pair.second);
               ctx.interferences[pair.second].second.emplace(def_spill_id);
            }
            ctx.affinities.emplace_back(std::pair<uint32_t, uint32_t>{def_spill_id, spill_id});

            aco_ptr<Instruction> spill{create_instruction<Instruction>(aco_opcode::p_spill, Format::PSEUDO, 2, 0)};
            spill->getOperand(0) = phi->getOperand(i);
            spill->getOperand(1) = Operand(spill_id);
            unsigned idx = preds[i]->instructions.size();
            do {
               assert(idx != 0);
               idx--;
            } while (phi->opcode == aco_opcode::p_phi && preds[i]->instructions[idx]->opcode != aco_opcode::p_logical_end);
            std::vector<aco_ptr<Instruction>>::iterator it = std::next(preds[i]->instructions.begin(), idx);
            preds[i]->instructions.insert(it, std::move(spill));
            continue;
         }
         if (!phi->getOperand(i).isTemp())
            continue;

         /* build interferences between the phi def and all spilled variables at the predecessor blocks */
         for (std::pair<Temp, uint32_t> pair : ctx.spills_exit[pred_idx]) {
            if (phi->getOperand(i).getTemp() == pair.first)
               continue;
            ctx.interferences[def_spill_id].second.emplace(pair.second);
            ctx.interferences[pair.second].second.emplace(def_spill_id);
         }

         /* variable is already spilled at predecessor */
         std::map<Temp, uint32_t>::iterator spilled = ctx.spills_exit[pred_idx].find(phi->getOperand(i).getTemp());
         if (spilled != ctx.spills_exit[pred_idx].end()) {
            if (spilled->second != def_spill_id)
               ctx.affinities.emplace_back(std::pair<uint32_t, uint32_t>{def_spill_id, spilled->second});
            continue;
         }

         /* rename if necessary */
         Temp var = phi->getOperand(i).getTemp();
         std::map<Temp, Temp>::iterator rename_it = ctx.renames[pred_idx].find(var);
         if (rename_it != ctx.renames[pred_idx].end()) {
            var = rename_it->second;
            ctx.renames[pred_idx].erase(rename_it);
         }

         uint32_t spill_id = ctx.allocate_spill_id(phi->getDefinition(0).regClass());
         ctx.affinities.emplace_back(std::pair<uint32_t, uint32_t>{def_spill_id, spill_id});
         aco_ptr<Instruction> spill{create_instruction<Instruction>(aco_opcode::p_spill, Format::PSEUDO, 2, 0)};
         spill->getOperand(0) = Operand(var);
         spill->getOperand(1) = Operand(spill_id);
         unsigned idx = preds[i]->instructions.size();
         do {
            assert(idx != 0);
            idx--;
         } while (phi->opcode == aco_opcode::p_phi && preds[i]->instructions[idx]->opcode != aco_opcode::p_logical_end);
         std::vector<aco_ptr<Instruction>>::iterator it = std::next(preds[i]->instructions.begin(), idx);
         preds[i]->instructions.insert(it, std::move(spill));
         ctx.spills_exit[pred_idx][phi->getOperand(i).getTemp()] = spill_id;
      }

      /* remove phi from instructions */
      phi.reset();
   }

   /* iterate all (other) spilled variables for which to spill at the predecessor */
   // TODO: would be better to have them sorted: first vgprs and first with longest distance
   for (std::pair<Temp, uint32_t> pair : ctx.spills_entry[block_idx]) {
      std::vector<Block*> preds = pair.first.type() == vgpr ? block->logical_predecessors : block->linear_predecessors;

      for (Block* pred : preds) {
         /* add interferences between spilled variable and predecessors exit spills */
         for (std::pair<Temp, uint32_t> exit_spill : ctx.spills_exit[pred->index]) {
            if (exit_spill.first == pair.first)
               continue;
            ctx.interferences[exit_spill.second].second.emplace(pair.second);
            ctx.interferences[pair.second].second.emplace(exit_spill.second);
         }

         /* variable is already spilled at predecessor */
         std::map<Temp, uint32_t>::iterator spilled = ctx.spills_exit[pred->index].find(pair.first);
         if (spilled != ctx.spills_exit[pred->index].end()) {
            if (spilled->second != pair.second)
               ctx.affinities.emplace_back(std::pair<uint32_t, uint32_t>{pair.second, spilled->second});
            continue;
         }

         /* variable is dead at predecessor, it must be from a phi: this works because of CSSA form */ // FIXME: lower_to_cssa()
         if (ctx.next_use_distances_end[pred->index].find(pair.first) == ctx.next_use_distances_end[pred->index].end())
            continue;

         /* variable is in register at predecessor and has to be spilled */
         /* rename if necessary */
         Temp var = pair.first;
         std::map<Temp, Temp>::iterator rename_it = ctx.renames[pred->index].find(var);
         if (rename_it != ctx.renames[pred->index].end()) {
            var = rename_it->second;
            ctx.renames[pred->index].erase(rename_it);
         }

         aco_ptr<Instruction> spill{create_instruction<Instruction>(aco_opcode::p_spill, Format::PSEUDO, 2, 0)};
         spill->getOperand(0) = Operand(var);
         spill->getOperand(1) = Operand(pair.second);
         unsigned idx = pred->instructions.size();
         do {
            assert(idx != 0);
            idx--;
         } while (pair.first.type() == vgpr && pred->instructions[idx]->opcode != aco_opcode::p_logical_end);
         std::vector<aco_ptr<Instruction>>::iterator it = std::next(pred->instructions.begin(), idx);
         pred->instructions.insert(it, std::move(spill));
         ctx.spills_exit[pred->index][pair.first] = pair.second;
      }
   }

   /* iterate phis for which operands to reload */
   for (aco_ptr<Instruction>& phi : instructions) {
      assert(phi->opcode == aco_opcode::p_phi || phi->opcode == aco_opcode::p_linear_phi);
      assert(ctx.spills_entry[block_idx].find(phi->getDefinition(0).getTemp()) == ctx.spills_entry[block_idx].end());

      std::vector<Block*>& preds = phi->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
      for (unsigned i = 0; i < phi->num_operands; i++) {
         if (!phi->getOperand(i).isTemp())
            continue;
         unsigned pred_idx = preds[i]->index;

         /* rename operand */
         if (ctx.spills_exit[pred_idx].find(phi->getOperand(i).getTemp()) == ctx.spills_exit[pred_idx].end()) {
            std::map<Temp, Temp>::iterator it = ctx.renames[pred_idx].find(phi->getOperand(i).getTemp());
            if (it != ctx.renames[pred_idx].end())
               phi->getOperand(i).setTemp(it->second);
            continue;
         }

         Temp tmp = phi->getOperand(i).getTemp();

         /* reload phi operand at end of predecessor block */
         Temp new_name = {ctx.program->allocateId(), tmp.regClass()};

         unsigned idx = preds[i]->instructions.size();
         do {
            assert(idx != 0);
            idx--;
         } while (phi->opcode == aco_opcode::p_phi && preds[i]->instructions[idx]->opcode != aco_opcode::p_logical_end);
         std::vector<aco_ptr<Instruction>>::iterator it = std::next(preds[i]->instructions.begin(), idx);

         aco_ptr<Instruction> reload = do_reload(ctx, tmp, new_name, ctx.spills_exit[pred_idx][tmp]);
         preds[i]->instructions.insert(it, std::move(reload));

         ctx.spills_exit[pred_idx].erase(tmp);
         ctx.renames[pred_idx][tmp] = new_name;
      }
   }

   /* iterate live variables for which to reload */
   // TODO: reload at current block if variable is spilled on all predecessors
   for (std::pair<Temp, std::pair<uint32_t, uint32_t>> pair : ctx.next_use_distances_start[block_idx]) {
      /* skip spilled variables */
      if (ctx.spills_entry[block_idx].find(pair.first) != ctx.spills_entry[block_idx].end())
         continue;
      std::vector<Block*> preds = pair.first.type() == vgpr ? block->logical_predecessors : block->linear_predecessors;

      /* variable is dead at predecessor, it must be from a phi */
      bool is_dead = false;
      for (Block* pred : preds) {
         if (ctx.next_use_distances_end[pred->index].find(pair.first) == ctx.next_use_distances_end[pred->index].end())
            is_dead = true;
      }
      if (is_dead)
         continue;
      for (Block* pred : preds) {
         /* the variable is not spilled at the predecessor */
         if (ctx.spills_exit[pred->index].find(pair.first) == ctx.spills_exit[pred->index].end())
            continue;

         /* variable is spilled at predecessor and has to be reloaded */
         Temp new_name = {ctx.program->allocateId(), pair.first.regClass()};

         unsigned idx = pred->instructions.size();
         do {
            assert(idx != 0);
            idx--;
         } while (pair.first.type() == vgpr && pred->instructions[idx]->opcode != aco_opcode::p_logical_end);
         std::vector<aco_ptr<Instruction>>::iterator it = std::next(pred->instructions.begin(), idx);

         aco_ptr<Instruction> reload = do_reload(ctx, pair.first, new_name, ctx.spills_exit[pred->index][pair.first]);
         pred->instructions.insert(it, std::move(reload));

         ctx.spills_exit[pred->index].erase(pair.first);
         ctx.renames[pred->index][pair.first] = new_name;
      }

      /* check if we have to create a new phi for this variable */
      Temp rename = Temp();
      bool is_same = true;
      for (Block* pred : preds) {
         if (ctx.renames[pred->index].find(pair.first) == ctx.renames[pred->index].end()) {
            if (rename == Temp())
               rename = pair.first;
            else
               is_same = rename == pair.first;
         } else {
            if (rename == Temp())
               rename = ctx.renames[pred->index][pair.first];
            else
               is_same = rename == ctx.renames[pred->index][pair.first];
         }

         if (!is_same)
            break;
      }

      if (!is_same) {
         /* the variable was renamed differently in the predecessors: we have to create a phi */
         aco_opcode opcode = pair.first.type() == vgpr ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
         aco_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
         rename = {ctx.program->allocateId(), pair.first.regClass()};
         for (unsigned i = 0; i < phi->num_operands; i++) {
            Temp tmp;
            if (ctx.renames[preds[i]->index].find(pair.first) != ctx.renames[preds[i]->index].end())
               tmp = ctx.renames[preds[i]->index][pair.first];
            else if (preds[i]->index >= block_idx)
               tmp = rename;
            else
               tmp = pair.first;
            phi->getOperand(i) = Operand(tmp);
         }
         phi->getDefinition(0) = Definition(rename);
         instructions.emplace_back(std::move(phi));
      }

      /* the variable was renamed: add new name to renames */
      if (!(rename == Temp() || rename == pair.first))
         ctx.renames[block_idx][pair.first] = rename;
   }

   /* combine phis with instructions */
   unsigned idx = 0;
   while (!block->instructions[idx]) {
      idx++;
   }

   ctx.register_demand[block->index].erase(ctx.register_demand[block->index].begin(), ctx.register_demand[block->index].begin() + idx);
   ctx.register_demand[block->index].insert(ctx.register_demand[block->index].begin(), instructions.size(), std::pair<uint16_t, uint16_t>(0, 0));

   std::vector<aco_ptr<Instruction>>::iterator start = std::next(block->instructions.begin(), idx);
   instructions.insert(instructions.end(), std::move_iterator<std::vector<aco_ptr<Instruction>>::iterator>(start),
               std::move_iterator<std::vector<aco_ptr<Instruction>>::iterator>(block->instructions.end()));
   block->instructions = std::move(instructions);
}

void process_block(spill_ctx& ctx, unsigned block_idx, std::unique_ptr<Block>& block,
                   std::map<Temp, uint32_t> &current_spills, int spilled_sgprs, int spilled_vgprs)
{
   std::vector<std::map<Temp, uint32_t>> local_next_use_distance;
   std::vector<aco_ptr<Instruction>> instructions;
   unsigned idx = 0;

   /* phis are handled separetely */
   while (block->instructions[idx]->opcode == aco_opcode::p_phi ||
          block->instructions[idx]->opcode == aco_opcode::p_linear_phi) {
      aco_ptr<Instruction>& instr = block->instructions[idx];
      for (unsigned i = 0; i < instr->num_operands; i++) {
         /* prevent it's definining instruction from being DCE'd if it could be rematerialized */
         if (instr->getOperand(i).isTemp() && ctx.remat.count(instr->getOperand(i).getTemp()))
            ctx.remat_use_count[ctx.remat[instr->getOperand(i).getTemp()].instr]++;
      }
      instructions.emplace_back(std::move(instr));
      idx++;
   }

   if (block->vgpr_demand > ctx.target_vgpr || block->sgpr_demand > ctx.target_sgpr)
      local_next_use_distance = local_next_uses(ctx, block);

   while (idx < block->instructions.size()) {
      aco_ptr<Instruction>& instr = block->instructions[idx];

      std::map<Temp, std::pair<Temp, uint32_t>> reloads;
      std::map<Temp, uint32_t> spills;
      /* rename and reload operands */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (!instr->getOperand(i).isTemp())
            continue;
         if (current_spills.find(instr->getOperand(i).getTemp()) == current_spills.end()) {
            /* prevent it's definining instruction from being DCE'd if it could be rematerialized */
            if (ctx.remat.count(instr->getOperand(i).getTemp())) {
               ctx.remat_use_count[ctx.remat[instr->getOperand(i).getTemp()].instr]++;
            }
            /* the Operand is in register: check if it was renamed */
            if (ctx.renames[block_idx].find(instr->getOperand(i).getTemp()) != ctx.renames[block_idx].end())
               instr->getOperand(i).setTemp(ctx.renames[block_idx][instr->getOperand(i).getTemp()]);
            continue;
         }
         /* the Operand is spilled: add it to reloads */
         Temp new_tmp = {ctx.program->allocateId(), instr->getOperand(i).regClass()};
         ctx.renames[block_idx][instr->getOperand(i).getTemp()] = new_tmp;
         reloads[new_tmp] = std::make_pair(instr->getOperand(i).getTemp(), current_spills[instr->getOperand(i).getTemp()]);
         current_spills.erase(instr->getOperand(i).getTemp());
         instr->getOperand(i).setTemp(new_tmp);
         if (new_tmp.type() == vgpr)
            spilled_vgprs -= new_tmp.size();
         else
            spilled_sgprs -= new_tmp.size();
      }

      /* check if register demand is low enough before and after the current instruction */
      if (block->vgpr_demand > ctx.target_vgpr || block->sgpr_demand > ctx.target_sgpr) {

         int sgpr_demand = ctx.register_demand[block_idx][idx].first;
         int vgpr_demand = ctx.register_demand[block_idx][idx].second;
         if (idx == 0) {
            for (unsigned i = 0; i < instr->num_definitions; i++) {
               if (!instr->getDefinition(i).isTemp())
                  continue;
               if (instr->getDefinition(i).getTemp().type() == vgpr)
                  vgpr_demand += instr->getDefinition(i).size();
               else
                  sgpr_demand += instr->getDefinition(i).size();
            }
         } else {
            sgpr_demand = std::max<int>(ctx.register_demand[block_idx][idx - 1].first, sgpr_demand);
            vgpr_demand = std::max<int>(ctx.register_demand[block_idx][idx - 1].second, vgpr_demand);
         }

         assert(!local_next_use_distance.empty());

         /* if reg pressure is too high, spill variable with furthest next use */
         while (std::max(vgpr_demand - spilled_vgprs, 0) > ctx.target_vgpr ||
                std::max(sgpr_demand - spilled_sgprs, 0) > ctx.target_sgpr) {
            unsigned distance = 0;
            Temp to_spill;
            bool do_rematerialize = false;
            if (vgpr_demand - spilled_vgprs > ctx.target_vgpr) {
               for (std::pair<Temp, uint32_t> pair : local_next_use_distance[idx]) {
                  bool can_rematerialize = ctx.remat.count(pair.first);
                  if (pair.first.type() == vgpr &&
                      (pair.second > distance || (can_rematerialize && !do_rematerialize)) &&
                      current_spills.find(pair.first) == current_spills.end() &&
                      ctx.spills_exit[block_idx].find(pair.first) == ctx.spills_exit[block_idx].end()) {
                     to_spill = pair.first;
                     distance = pair.second;
                     do_rematerialize = can_rematerialize;
                  }
               }
            } else {
               for (std::pair<Temp, uint32_t> pair : local_next_use_distance[idx]) {
                  bool can_rematerialize = ctx.remat.count(pair.first);
                  if (pair.first.type() == sgpr &&
                      (pair.second > distance || (can_rematerialize && !do_rematerialize)) &&
                      current_spills.find(pair.first) == current_spills.end() &&
                      ctx.spills_exit[block_idx].find(pair.first) == ctx.spills_exit[block_idx].end()) {
                     to_spill = pair.first;
                     distance = pair.second;
                     do_rematerialize = can_rematerialize;
                  }
               }
            }

            assert(distance != 0);
            uint32_t spill_id = ctx.allocate_spill_id(to_spill.regClass());

            /* add interferences with currently spilled variables */
            for (std::pair<Temp, uint32_t> pair : current_spills) {
               ctx.interferences[spill_id].second.emplace(pair.second);
               ctx.interferences[pair.second].second.emplace(spill_id);
            }
            for (std::pair<Temp, std::pair<Temp, uint32_t>> pair : reloads) {
               ctx.interferences[spill_id].second.emplace(pair.second.second);
               ctx.interferences[pair.second.second].second.emplace(spill_id);
            }

            current_spills[to_spill] = spill_id;
            if (to_spill.type() == vgpr)
               spilled_vgprs += to_spill.size();
            else
               spilled_sgprs += to_spill.size();

            /* rename if necessary */
            if (ctx.renames[block_idx].find(to_spill) != ctx.renames[block_idx].end()) {
               to_spill = ctx.renames[block_idx][to_spill];
            }

            /* add spill to new instructions */
            aco_ptr<Instruction> spill{create_instruction<Instruction>(aco_opcode::p_spill, Format::PSEUDO, 2, 0)};
            spill->getOperand(0) = Operand(to_spill);
            spill->getOperand(1) = Operand(spill_id);
            instructions.emplace_back(std::move(spill));
         }
      }

      /* add reloads and instruction to new instructions */
      for (std::pair<Temp, std::pair<Temp, uint32_t>> pair : reloads) {
         aco_ptr<Instruction> reload = do_reload(ctx, pair.second.first, pair.first, pair.second.second);
         instructions.emplace_back(std::move(reload));
      }
      instructions.emplace_back(std::move(instr));
      idx++;
   }

   block->instructions = std::move(instructions);
   ctx.spills_exit[block_idx].insert(current_spills.begin(), current_spills.end());
}

void spill_block(spill_ctx& ctx, unsigned block_idx)
{
   std::unique_ptr<Block>& block = ctx.program->blocks[block_idx];
   ctx.processed[block_idx] = true;

   /* determine set of variables which are spilled at the beginning of the block */
   std::pair<unsigned, unsigned> spills = init_live_in_vars(ctx, block.get(), block_idx);
   unsigned spilled_sgprs = spills.first;
   unsigned spilled_vgprs = spills.second;

   /* add interferences for spilled variables */
   for (std::pair<Temp, uint32_t> x : ctx.spills_entry[block_idx]) {
      for (std::pair<Temp, uint32_t> y : ctx.spills_entry[block_idx])
         if (x.second != y.second)
            ctx.interferences[x.second].second.emplace(y.second);
   }

   bool is_loop_header = block->loop_nest_depth && ctx.loop_header.top()->index == block_idx;
   if (!is_loop_header) {
      /* add spill/reload code on incoming control flow edges */
      add_coupling_code(ctx, block.get(), block_idx);
   }

   std::map<Temp, uint32_t> current_spills = ctx.spills_entry[block_idx];

   /* remove spills which are not needed in this block */
   std::map<Temp, uint32_t>::iterator spill = current_spills.begin();
   while (spill != current_spills.end()) {
      if (ctx.next_use_distances_start[block_idx][spill->first].first > block_idx) {
         ctx.spills_exit[block_idx].insert(*spill);
         spill = current_spills.erase(spill);
      } else {
         ++spill;
      }
   }

   /* conditions to process this block */
   if (!current_spills.empty() ||
       block->vgpr_demand - spilled_vgprs > ctx.target_vgpr ||
       block->sgpr_demand - spilled_sgprs > ctx.target_sgpr ||
       !ctx.renames[block_idx].empty() ||
       ctx.remat_use_count.size()) {
      process_block(ctx, block_idx, block, current_spills, spilled_sgprs, spilled_vgprs);
   }

   /* check if the next block leaves the current loop */
   if (block->loop_nest_depth == 0 || ctx.program->blocks[block_idx + 1]->loop_nest_depth >= block->loop_nest_depth)
      return;

   Block* loop_header = ctx.loop_header.top();

   /* preserve original renames at end of loop header block */
   std::map<Temp, Temp> renames = std::move(ctx.renames[loop_header->index]);

   /* add coupling code to all loop header predecessors */
   add_coupling_code(ctx, loop_header, loop_header->index);

   /* propagate new renames through loop: i.e. repair the SSA */
   renames.swap(ctx.renames[loop_header->index]);
   for (std::pair<Temp, Temp> rename : renames) {
      for (unsigned idx = loop_header->index; idx <= block_idx; idx++) {
         std::unique_ptr<Block>& current = ctx.program->blocks[idx];
         std::vector<aco_ptr<Instruction>>::iterator instr_it = current->instructions.begin();

         /* first rename phis */
         while (instr_it != current->instructions.end()) {
            aco_ptr<Instruction>& phi = *instr_it;
            if (phi->opcode != aco_opcode::p_phi && phi->opcode != aco_opcode::p_linear_phi)
               break;
            /* no need to rename the loop header phis once again. this happened in add_coupling_code() */
            if (idx == loop_header->index) {
               instr_it++;
               continue;
            }

            for (unsigned i = 0; i < phi->num_operands; i++) {
               if (!phi->getOperand(i).isTemp())
                  continue;
               if (phi->getOperand(i).getTemp() == rename.first)
                  phi->getOperand(i).setTemp(rename.second);
            }
            instr_it++;
         }

         std::map<Temp, std::pair<uint32_t, uint32_t>>::iterator it = ctx.next_use_distances_start[idx].find(rename.first);

         /* variable is not live at beginning of this block */
         if (it == ctx.next_use_distances_start[idx].end())
            continue;

         /* if the variable is live at the block's exit, add rename */
         if (ctx.next_use_distances_end[idx].find(rename.first) != ctx.next_use_distances_end[idx].end())
            ctx.renames[idx].insert(rename);

         /* rename all uses in this block */
         bool renamed = false;
         while (!renamed && instr_it != current->instructions.end()) {
            aco_ptr<Instruction>& instr = *instr_it;
            for (unsigned i = 0; i < instr->num_operands; i++) {
               if (!instr->getOperand(i).isTemp())
                  continue;
               if (instr->getOperand(i).getTemp() == rename.first) {
                  instr->getOperand(i).setTemp(rename.second);
                  /* we can stop with this block as soon as the variable is spilled */
                  if (instr->opcode == aco_opcode::p_spill)
                    renamed = true;
               }
            }
            instr_it++;
         }
      }
   }

   /* remove loop header info from stack */
   ctx.loop_header.pop();
}

void assign_spill_slots(spill_ctx& ctx, unsigned spills_to_vgpr) {
   std::map<uint32_t, uint32_t> sgpr_slot;
   std::map<uint32_t, uint32_t> vgpr_slot;
   std::vector<bool> is_assigned(ctx.interferences.size());

   /* first, handle affinities: just merge all interferences into both spill ids */
   for (std::pair<uint32_t, uint32_t> pair : ctx.affinities) {
      assert(pair.first != pair.second);
      ctx.interferences[pair.first].second.insert(ctx.interferences[pair.second].second.begin(), ctx.interferences[pair.second].second.end());
      ctx.interferences[pair.second].second.insert(ctx.interferences[pair.first].second.begin(), ctx.interferences[pair.first].second.end());
   }
   for (uint32_t i = 0; i < ctx.interferences.size(); i++)
      for (uint32_t id : ctx.interferences[i].second)
         assert(i != id);

   /* for each spill slot, assign as many spill ids as possible */
   std::vector<std::set<uint32_t>> spill_slot_interferences;
   unsigned slot_idx = 0;
   bool done = false;

   /* assign sgpr spill slots */
   while (!done) {
      done = true;
      for (unsigned id = 0; id < ctx.interferences.size(); id++) {
         if (is_assigned[id] || !ctx.reload_count[id])
            continue;
         if (typeOf(ctx.interferences[id].first) != sgpr)
            continue;

         /* check interferences */
         bool interferes = false;
         for (unsigned i = slot_idx; i < slot_idx + sizeOf(ctx.interferences[id].first); i++) {
            if (i == spill_slot_interferences.size())
               spill_slot_interferences.emplace_back(std::set<uint32_t>());
            if (spill_slot_interferences[i].find(id) != spill_slot_interferences[i].end()) {
               interferes = true;
               break;
            }
         }
         if (interferes) {
            done = false;
            continue;
         }

         /* we found a spill id which can be assigned to current spill slot */
         sgpr_slot[id] = slot_idx;
         is_assigned[id] = true;
         for (unsigned i = slot_idx; i < slot_idx + sizeOf(ctx.interferences[id].first); i++)
            spill_slot_interferences[i].insert(ctx.interferences[id].second.begin(), ctx.interferences[id].second.end());
      }
      slot_idx++;
   }

   slot_idx = 0;
   done = false;

   /* assign vgpr spill slots */
   while (!done) {
      done = true;
      for (unsigned id = 0; id < ctx.interferences.size(); id++) {
         if (is_assigned[id] || !ctx.reload_count[id])
            continue;
         if (typeOf(ctx.interferences[id].first) != vgpr)
            continue;

         /* check interferences */
         bool interferes = false;
         for (unsigned i = slot_idx; i < slot_idx + sizeOf(ctx.interferences[id].first); i++) {
            if (i == spill_slot_interferences.size())
               spill_slot_interferences.emplace_back(std::set<uint32_t>());
            /* check for interference and ensure that vector regs are stored next to each other */
            if (spill_slot_interferences[i].find(id) != spill_slot_interferences[i].end() || i / 64 != slot_idx / 64) {
               interferes = true;
               break;
            }
         }
         if (interferes) {
            done = false;
            continue;
         }

         /* we found a spill id which can be assigned to current spill slot */
         vgpr_slot[id] = slot_idx;
         is_assigned[id] = true;
         for (unsigned i = slot_idx; i < slot_idx + sizeOf(ctx.interferences[id].first); i++)
            spill_slot_interferences[i].insert(ctx.interferences[id].second.begin(), ctx.interferences[id].second.end());
      }
      slot_idx++;
   }

   for (unsigned id = 0; id < is_assigned.size(); id++)
      assert(is_assigned[id] || !ctx.reload_count[id]);

   /* hope, we didn't mess up */
   std::vector<Temp> vgpr_spill_temps((spill_slot_interferences.size() + 63) / 64);
   assert(vgpr_spill_temps.size() <= spills_to_vgpr);

   /* replace pseudo instructions with actual hardware instructions */
   unsigned last_top_level_block_idx = 0;
   for (std::unique_ptr<Block>& block : ctx.program->blocks) {

      if (block->kind & block_kind_top_level) {
         last_top_level_block_idx = block->index;

         /* check if any spilled variables use a created linear vgpr, otherwise destroy them */
         for (unsigned i = 0; i < vgpr_spill_temps.size(); i++) {
            if (vgpr_spill_temps[i] == Temp())
               continue;

            bool can_destroy = true;
            for (std::pair<Temp, uint32_t> pair : ctx.spills_entry[block->index]) {

               if (sgpr_slot.find(pair.second) != sgpr_slot.end() &&
                   sgpr_slot[pair.second] / 64 == i) {
                  can_destroy = false;
                  break;
               }
            }
            if (can_destroy) {
               aco_ptr<Instruction> destr{create_instruction<Instruction>(aco_opcode::p_end_linear_vgpr, Format::PSEUDO, 1, 0)};
               destr->getOperand(0) = Operand(vgpr_spill_temps[i]);
               /* find insertion point */
               std::vector<aco_ptr<Instruction>>::iterator after_phi = block->instructions.begin();
               while ((*after_phi)->opcode == aco_opcode::p_linear_phi || (*after_phi)->opcode == aco_opcode::p_phi)
                  after_phi++;
               block->instructions.insert(after_phi, std::move(destr));
               vgpr_spill_temps[i] = Temp();
            }
         }


      }

      std::vector<aco_ptr<Instruction>>::iterator it;
      std::vector<aco_ptr<Instruction>> instructions;
      instructions.reserve(block->instructions.size());
      for (it = block->instructions.begin(); it != block->instructions.end(); ++it) {

         if ((*it)->opcode == aco_opcode::p_spill) {
            uint32_t spill_id = (*it)->getOperand(1).constantValue();

            if (!ctx.reload_count[spill_id]) {
               /* never reloaded, so don't spill */
            } else if (vgpr_slot.find(spill_id) != vgpr_slot.end()) {
               /* spill vgpr */
               ctx.program->config->spilled_vgprs += (*it)->getOperand(0).size();

               assert(false && "vgpr spilling not yet implemented.");
            } else if (sgpr_slot.find(spill_id) != sgpr_slot.end()) {
               ctx.program->config->spilled_sgprs += (*it)->getOperand(0).size();

               uint32_t spill_slot = sgpr_slot[spill_id];

               /* check if the linear vgpr already exists */
               if (vgpr_spill_temps[spill_slot / 64] == Temp()) {
                  Temp linear_vgpr = {ctx.program->allocateId(), v1_linear};
                  vgpr_spill_temps[spill_slot / 64] = linear_vgpr;
                  aco_ptr<Instruction> create{create_instruction<Instruction>(aco_opcode::p_start_linear_vgpr, Format::PSEUDO, 0, 1)};
                  create->getDefinition(0) = Definition(linear_vgpr);
                  /* find the right place to insert this definition */
                  if (last_top_level_block_idx == block->index) {
                     /* insert right before the current instruction */
                     instructions.emplace_back(std::move(create));
                  } else {
                     assert(last_top_level_block_idx < block->index);
                     /* insert before the branch at last top level block */
                     std::vector<aco_ptr<Instruction>>& instructions = ctx.program->blocks[last_top_level_block_idx]->instructions;
                     instructions.insert(std::next(instructions.begin(), instructions.size() - 1), std::move(create));
                  }
               }

               /* spill sgpr: just add the vgpr temp to operands */
               Instruction* spill = create_instruction<Instruction>(aco_opcode::p_spill, Format::PSEUDO, 3, 0);
               spill->getOperand(0) = Operand(vgpr_spill_temps[spill_slot / 64]);
               spill->getOperand(1) = Operand(spill_slot % 64);
               spill->getOperand(2) = (*it)->getOperand(0);
               instructions.emplace_back(aco_ptr<Instruction>(spill));
            } else {
               unreachable("No spill slot assigned for spill id");
            }

         } else if ((*it)->opcode == aco_opcode::p_reload) {
            uint32_t spill_id = (*it)->getOperand(0).constantValue();
            assert(ctx.reload_count[spill_id]);

            if (vgpr_slot.find(spill_id) != vgpr_slot.end()) {
               /* reload vgpr */
               assert(false && "vgpr spilling not yet implemented.");

            } else if (sgpr_slot.find(spill_id) != sgpr_slot.end()) {
               uint32_t spill_slot = sgpr_slot[spill_id];

               /* check if the linear vgpr already exists */
               if (vgpr_spill_temps[spill_slot / 64] == Temp()) {
                  Temp linear_vgpr = {ctx.program->allocateId(), v1_linear};
                  vgpr_spill_temps[spill_slot / 64] = linear_vgpr;
                  aco_ptr<Instruction> create{create_instruction<Instruction>(aco_opcode::p_start_linear_vgpr, Format::PSEUDO, 0, 1)};
                  create->getDefinition(0) = Definition(linear_vgpr);
                  /* find the right place to insert this definition */
                  if (last_top_level_block_idx == block->index) {
                     /* insert right before the current instruction */
                     instructions.emplace_back(std::move(create));
                  } else {
                     assert(last_top_level_block_idx < block->index);
                     /* insert before the branch at last top level block */
                     std::vector<aco_ptr<Instruction>>& instructions = ctx.program->blocks[last_top_level_block_idx]->instructions;
                     instructions.insert(std::next(instructions.begin(), instructions.size() - 1), std::move(create));
                  }
               }

               /* reload sgpr: just add the vgpr temp to operands */
               Instruction* reload = create_instruction<Instruction>(aco_opcode::p_reload, Format::PSEUDO, 2, 1);
               reload->getOperand(0) = Operand(vgpr_spill_temps[spill_slot / 64]);
               reload->getOperand(1) = Operand(spill_slot % 64);
               reload->getDefinition(0) = (*it)->getDefinition(0);
               instructions.emplace_back(aco_ptr<Instruction>(reload));
            } else {
               unreachable("No spill slot assigned for spill id");
            }
         } else if (!ctx.remat_use_count.count(it->get()) || ctx.remat_use_count[it->get()] > 0) {
            instructions.emplace_back(std::move(*it));
         }

      }
      block->instructions = std::move(instructions);
   }
}

} /* end namespace */


void spill(Program* program, live& live_vars, const struct radv_nir_compiler_options *options)
{
   program->config->spilled_vgprs = 0;
   program->config->spilled_sgprs = 0;

   /* no spilling when wave count is already high */
   if (program->num_waves >= 6)
      return;

   /* else, we check if we can improve things a bit */
   uint16_t total_sgpr_regs = options->chip_class >= GFX8 ? 800 : 512;
   uint16_t max_addressible_sgpr = options->chip_class >= GFX8 ? 102 : 104;

   /* calculate target register demand */
   uint16_t max_vgpr = 0, max_sgpr = 0;
   for (std::unique_ptr<Block>& block : program->blocks) {
      max_vgpr = std::max(max_vgpr, block->vgpr_demand);
      max_sgpr = std::max(max_sgpr, block->sgpr_demand);
   }

   uint16_t target_vgpr = 256;
   uint16_t target_sgpr = max_addressible_sgpr;
   unsigned num_waves = 1;
   int spills_to_vgpr = (max_sgpr - max_addressible_sgpr + 63) / 64;

   /* test if it possible to increase occupancy with little spilling */
   for (unsigned num_waves_next = 2; num_waves_next <= 8; num_waves_next++) {
      uint16_t target_vgpr_next = (256 / num_waves_next) & ~3;
      uint16_t target_sgpr_next = std::min<uint16_t>(((total_sgpr_regs / num_waves_next) & ~7) - 2, max_addressible_sgpr);

      /* Currently no vgpr spilling supported.
       * Spill as many sgprs as necessary to not hinder occupancy */
      if (max_vgpr > target_vgpr_next)
         break;
      /* check that we have enough free vgprs to spill sgprs to */
      if (max_sgpr > target_sgpr_next) {
         /* add some buffer in case graph coloring is not perfect ... */
         int spills_to_vgpr_next = (max_sgpr - target_sgpr_next + 63 + 32) / 64;
         if (spills_to_vgpr_next + max_vgpr > target_vgpr_next)
            break;
         spills_to_vgpr = spills_to_vgpr_next;
      }

      target_vgpr = target_vgpr_next;
      target_sgpr = target_sgpr_next;
      num_waves = num_waves_next;
   }

   assert(max_vgpr <= target_vgpr && "VGPR spilling not yet supported.");
   /* nothing to do */
   if (num_waves == program->num_waves)
      return;

   /* initialize ctx */
   spill_ctx ctx(target_vgpr, target_sgpr, program, live_vars.register_demand);
   compute_global_next_uses(ctx, live_vars.live_out);
   get_rematerialize_info(ctx);

   /* create spills and reloads */
   for (unsigned i = 0; i < program->blocks.size(); i++)
      spill_block(ctx, i);

   /* assign spill slots and DCE rematerialized code */
   assign_spill_slots(ctx, spills_to_vgpr);

   /* update live variable information */
   live_vars = live_var_analysis<true>(program, options);

   assert(program->num_waves >= num_waves);
}

}

#endif
