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
#include <unordered_set>
#include <algorithm>

namespace aco {

struct sched_ctx {
   int16_t num_waves;
   int16_t max_vgpr;
   int16_t max_sgpr;
   int16_t last_SMEM_stall;
   unsigned last_SMEM_dep_idx;
};

/* This scheduler is a simple bottom-up pass based on ideas from
 * "A Novel Lightweight Instruction Scheduling Algorithm for Just-In-Time Compiler"
 * from Xiaohua Shi and Peng Guo.
 * The basic approach is to iterate over all instructions. When a memory instruction
 * is encountered it tries to move independent instructions from above and below
 * between the memory instruction and it's first user.
 * The novelty is that this scheduler cares for the current register pressure:
 * Instructions will only be moved if the register pressure won't exceed a certain bound.
 */

void schedule_SMEM(sched_ctx& ctx, std::unique_ptr<Block>& block,
                       std::vector<std::pair<uint16_t,uint16_t>>& register_demand,
                       Instruction* current, unsigned idx = 0)
{
   int window_size = 25 - ctx.num_waves;
   int16_t k = 0;

   /* create the initial set of values which current depends on */
   std::set<Temp> depends_on;
   for (unsigned i = 0; i < current->num_operands; i++) {
      if (current->getOperand(i).isTemp())
         depends_on.insert(current->getOperand(i).getTemp());
   }

   /* maintain how many registers remain free when moving instructions */
   int sgpr_pressure = register_demand[idx].first;
   int vgpr_pressure = register_demand[idx].second;

   /* first, check if we have instructions before current to move down */
   int insert_idx = idx + 1;

   for (int candidate_idx = idx - 1; candidate_idx > (int) idx - window_size; candidate_idx--) {
      assert(candidate_idx >= 0);
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      /* break if we'd make the previous SMEM instruction stall */
      bool can_stall_prev_smem = idx <= ctx.last_SMEM_dep_idx && candidate_idx < ctx.last_SMEM_dep_idx;
      if (can_stall_prev_smem && ctx.last_SMEM_stall >= 0)
         break;

      /* break when encountering another MEM instruction or logical_start */
      if (candidate->isVMEM() || candidate->format == Format::SMEM)
         break;
      if (candidate->opcode == aco_opcode::p_logical_start)
         break;

      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx].second);

      /* if current depends on candidate, add additional dependencies and continue */
      bool can_move_down = true;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isTemp() && depends_on.find(candidate->getDefinition(i).getTemp()) != depends_on.end())
            can_move_down = false;
      }
      if (!can_move_down) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               depends_on.insert(candidate->getOperand(i).getTemp());
         }
         continue;
      }

      bool register_pressure_unknown = false;
      /* check if one of candidate's operands is killed by depending instruction */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && depends_on.find(candidate->getOperand(i).getTemp()) != depends_on.end()) {
            // FIXME: account for difference in register pressure
            register_pressure_unknown = true;
         }
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               depends_on.insert(candidate->getOperand(i).getTemp());
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff = register_demand[candidate_idx].first - register_demand[candidate_idx - 1].first;
      int candidate_vgpr_diff = register_demand[candidate_idx].second - register_demand[candidate_idx - 1].second;
      if (vgpr_pressure - candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure - candidate_sgpr_diff > ctx.max_sgpr)
         break;
      // TODO: we might want to look further to find a sequence of instructions to move down which doesn't exceed reg pressure

      /* move the candidate below the memory load */
      auto begin = std::next(block->instructions.begin(), candidate_idx);
      auto end = std::next(block->instructions.begin(), insert_idx);
      std::rotate(begin, begin + 1, end);

      /* update register pressure */
      for (int i = candidate_idx; i < insert_idx - 1; i++) {
         register_demand[i].first = register_demand[i + 1].first - candidate_sgpr_diff;
         register_demand[i].second = register_demand[i + 1].second - candidate_vgpr_diff;
      }
      vgpr_pressure = vgpr_pressure - candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure - candidate_sgpr_diff;
      assert(candidate_vgpr_diff == (int) register_demand[insert_idx - 1].second - (int) register_demand[insert_idx - 2].second);

      if (candidate_idx < ctx.last_SMEM_dep_idx)
         ctx.last_SMEM_stall++;
      insert_idx--;
      k++;
   }

   /* create the initial set of values which depend on current */
   depends_on.clear();
   for (unsigned i = 0; i < current->num_definitions; i++) {
      if (current->getDefinition(i).isTemp())
         depends_on.insert(current->getDefinition(i).getTemp());
   }
   std::set<Temp> RAR_dependencies;

   /* find the first instruction depending on current or find another MEM */
   insert_idx = idx + 1;

   bool found_dependency = false;
   /* second, check if we have instructions after current to move up */
   for (int candidate_idx = idx + 1; candidate_idx < (int) idx + window_size; candidate_idx++) {
      assert(candidate_idx < block->instructions.size());
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      if (candidate->opcode == aco_opcode::p_logical_end)
         break;

      /* check if candidate depends on current */
      bool is_dependency = false;
      for (unsigned i = 0; !is_dependency && i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && depends_on.find(candidate->getOperand(i).getTemp()) != depends_on.end())
            is_dependency = true;
      }
      if (is_dependency) {
         for (unsigned j = 0; j < candidate->num_definitions; j++) {
            if (candidate->getDefinition(j).isTemp())
               depends_on.insert(candidate->getDefinition(j).getTemp());
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               RAR_dependencies.insert(candidate->getOperand(i).getTemp());
         }
         if (!found_dependency) {
            insert_idx = candidate_idx;
            found_dependency = true;
            /* init register pressure */
            sgpr_pressure = register_demand[insert_idx - 1].first;
            vgpr_pressure = register_demand[insert_idx - 1].second;
         }
      }

      if (candidate->isVMEM() || candidate->format == Format::SMEM)
         break;

      if (!found_dependency) {
         k++;
         continue;
      }

      /* update register pressure */
      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx - 1].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx - 1].second);

      if (is_dependency)
         continue;
      assert(insert_idx != idx);

      // TODO: correctly calculate register pressure for this case
      bool register_pressure_unknown = false;
      /* check if candidate uses/kills an operand which is used by a dependency */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && RAR_dependencies.find(candidate->getOperand(i).getTemp()) != RAR_dependencies.end())
            register_pressure_unknown = true;
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_definitions; i++) {
            if (candidate->getDefinition(i).isTemp())
               RAR_dependencies.insert(candidate->getDefinition(i).getTemp());
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff = register_demand[candidate_idx].first - register_demand[candidate_idx - 1].first;
      int candidate_vgpr_diff = register_demand[candidate_idx].second - register_demand[candidate_idx - 1].second;
      if (vgpr_pressure + candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure + candidate_sgpr_diff > ctx.max_sgpr)
         break;

      /* move the candidate above the insert_idx */
      auto begin = std::next(block->instructions.begin(), insert_idx);
      auto end = std::next(block->instructions.begin(), candidate_idx + 1);
      std::rotate(begin, end - 1, end);

      /* update register pressure */
      for (int i = candidate_idx - 1; i >= insert_idx; i--) {
         register_demand[i].first = register_demand[i - 1].first + candidate_sgpr_diff;
         register_demand[i].second = register_demand[i - 1].second + candidate_vgpr_diff;
      }
      assert(candidate_vgpr_diff == (int) register_demand[insert_idx].second - (int) register_demand[insert_idx - 1].second);
      vgpr_pressure = vgpr_pressure + candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure + candidate_sgpr_diff;
      insert_idx++;
      k++;
   }

   ctx.last_SMEM_dep_idx = found_dependency ? insert_idx : 0;
   ctx.last_SMEM_stall = 10 - ctx.num_waves - k;
}

void schedule_VMEM(sched_ctx& ctx, std::unique_ptr<Block>& block,
                       std::vector<std::pair<uint16_t,uint16_t>>& register_demand,
                       Instruction* current, unsigned idx = 0)
{
   int window_size = 25 - ctx.num_waves;

   /* create the initial set of values which current depends on */
   std::set<Temp> depends_on;
   for (unsigned i = 0; i < current->num_operands; i++) {
      if (current->getOperand(i).isTemp())
         depends_on.insert(current->getOperand(i).getTemp());
   }

   /* maintain how many registers remain free when moving instructions */
   int sgpr_pressure = register_demand[idx].first;
   int vgpr_pressure = register_demand[idx].second;

   /* first, check if we have instructions before current to move down */
   int insert_idx = idx + 1;

   for (int candidate_idx = idx - 1; candidate_idx > (int) idx - window_size; candidate_idx--) {
      assert(candidate_idx >= 0);
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      /* break when encountering another VMEM instruction or logical_start */
      if (candidate->isVMEM())
         break;
      if (candidate->opcode == aco_opcode::p_logical_start)
         break;

      /* break if we'd make the previous SMEM instruction stall */
      bool can_stall_prev_smem = idx <= ctx.last_SMEM_dep_idx && candidate_idx < ctx.last_SMEM_dep_idx;
      if (can_stall_prev_smem && ctx.last_SMEM_stall >= 0)
         break;

      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx].second);

      /* if current depends on candidate, add additional dependencies and continue */
      bool can_move_down = true;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isTemp() && depends_on.find(candidate->getDefinition(i).getTemp()) != depends_on.end())
            can_move_down = false;
      }
      if (!can_move_down) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               depends_on.insert(candidate->getOperand(i).getTemp());
         }
         continue;
      }

      bool register_pressure_unknown = false;
      /* check if one of candidate's operands is killed by depending instruction */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && depends_on.find(candidate->getOperand(i).getTemp()) != depends_on.end()) {
            // FIXME: account for difference in register pressure
            register_pressure_unknown = true;
         }
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               depends_on.insert(candidate->getOperand(i).getTemp());
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff = register_demand[candidate_idx].first - register_demand[candidate_idx - 1].first;
      int candidate_vgpr_diff = register_demand[candidate_idx].second - register_demand[candidate_idx - 1].second;
      if (vgpr_pressure - candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure - candidate_sgpr_diff > ctx.max_sgpr)
         break;
      // TODO: we might want to look further to find a sequence of instructions to move down which doesn't exceed reg pressure

      /* move the candidate below the memory load */
      auto begin = std::next(block->instructions.begin(), candidate_idx);
      auto end = std::next(block->instructions.begin(), insert_idx);
      std::rotate(begin, begin + 1, end);

      /* update register pressure */
      for (int i = candidate_idx; i < insert_idx - 1; i++) {
         register_demand[i].first = register_demand[i + 1].first - candidate_sgpr_diff;
         register_demand[i].second = register_demand[i + 1].second - candidate_vgpr_diff;
      }
      vgpr_pressure = vgpr_pressure - candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure - candidate_sgpr_diff;
      assert(candidate_vgpr_diff == (int) register_demand[insert_idx - 1].second - (int) register_demand[insert_idx - 2].second);
      insert_idx--;
      if (candidate_idx < ctx.last_SMEM_dep_idx)
         ctx.last_SMEM_stall++;
   }

   /* create the initial set of values which depend on current */
   depends_on.clear();
   for (unsigned i = 0; i < current->num_definitions; i++) {
      if (current->getDefinition(i).isTemp())
         depends_on.insert(current->getDefinition(i).getTemp());
   }
   std::set<Temp> RAR_dependencies;

   /* find the first instruction depending on current or find another VMEM */
   insert_idx = idx;

   bool found_dependency = false;
   /* second, check if we have instructions after current to move up */
   for (int candidate_idx = idx + 1; candidate_idx < (int) idx + window_size; candidate_idx++) {
      assert(candidate_idx < block->instructions.size());
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      if (candidate->opcode == aco_opcode::p_logical_end)
         break;

      /* check if candidate depends on current */
      bool is_dependency = candidate->isVMEM();
      for (unsigned i = 0; !is_dependency && i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && depends_on.find(candidate->getOperand(i).getTemp()) != depends_on.end())
            is_dependency = true;
      }
      if (is_dependency) {
         for (unsigned j = 0; j < candidate->num_definitions; j++) {
            if (candidate->getDefinition(j).isTemp())
               depends_on.insert(candidate->getDefinition(j).getTemp());
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               RAR_dependencies.insert(candidate->getOperand(i).getTemp());
         }
         if (!found_dependency) {
            insert_idx = candidate_idx;
            found_dependency = true;
            /* init register pressure */
            sgpr_pressure = register_demand[insert_idx - 1].first;
            vgpr_pressure = register_demand[insert_idx - 1].second;
            continue;
         }
      }

      /* update register pressure */
      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx - 1].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx - 1].second);

      if (is_dependency || !found_dependency)
         continue;
      assert(insert_idx != idx);

      bool register_pressure_unknown = false;
      /* check if candidate uses/kills an operand which is used by a dependency */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && RAR_dependencies.find(candidate->getOperand(i).getTemp()) != RAR_dependencies.end())
            register_pressure_unknown = true;
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_definitions; i++) {
            if (candidate->getDefinition(i).isTemp())
               RAR_dependencies.insert(candidate->getDefinition(i).getTemp());
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff = register_demand[candidate_idx].first - register_demand[candidate_idx - 1].first;
      int candidate_vgpr_diff = register_demand[candidate_idx].second - register_demand[candidate_idx - 1].second;
      if (vgpr_pressure + candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure + candidate_sgpr_diff > ctx.max_sgpr)
         break;

      /* move the candidate above the insert_idx */
      auto begin = std::next(block->instructions.begin(), insert_idx);
      auto end = std::next(block->instructions.begin(), candidate_idx + 1);
      std::rotate(begin, end - 1, end);

      /* update register pressure */
      for (int i = candidate_idx - 1; i >= insert_idx; i--) {
         register_demand[i].first = register_demand[i - 1].first + candidate_sgpr_diff;
         register_demand[i].second = register_demand[i - 1].second + candidate_vgpr_diff;
      }
      assert(candidate_vgpr_diff == (int) register_demand[insert_idx].second - (int) register_demand[insert_idx - 1].second);
      vgpr_pressure = vgpr_pressure + candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure + candidate_sgpr_diff;
      insert_idx++;
   }
}

void schedule_block(sched_ctx& ctx, std::unique_ptr<Block>& block, std::vector<std::pair<uint16_t,uint16_t>>& register_demand)
{
   ctx.last_SMEM_dep_idx = 0;
   ctx.last_SMEM_stall = INT16_MIN;

   /* go through all instructions and find memory loads */
   for (unsigned idx = 0; idx < block->instructions.size(); idx++) {
      Instruction* current = block->instructions[idx].get();

      if (!current->num_definitions)
         continue;

      if (current->isVMEM())
         schedule_VMEM(ctx, block, register_demand, current, idx);
      if (current->format == Format::SMEM)
         schedule_SMEM(ctx, block, register_demand, current, idx);
   }
}


void schedule_program(Program *program, std::vector<std::vector<std::pair<uint16_t,uint16_t>>> register_demand)
{
   sched_ctx ctx;
   ctx.num_waves = program->num_waves;
   ctx.max_vgpr = program->max_vgpr;
   ctx.max_sgpr = program->max_sgpr;

   for (std::unique_ptr<Block>& block : program->blocks)
      schedule_block(ctx, block, register_demand[block->index]);

}

}
