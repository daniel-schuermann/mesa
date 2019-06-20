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

#include "../vulkan/radv_shader.h" // for radv_nir_compiler_options

#define SMEM_WINDOW_SIZE (350 - ctx.num_waves * 35)
#define VMEM_WINDOW_SIZE (1024 - ctx.num_waves * 64)
#define SMEM_MAX_MOVES (80 - ctx.num_waves * 8)
#define VMEM_MAX_MOVES (128 - ctx.num_waves * 4)

namespace aco {

struct sched_ctx {
   std::vector<bool> depends_on;
   std::vector<bool> RAR_dependencies;
   int16_t num_waves;
   int16_t max_vgpr;
   int16_t max_sgpr;
   int16_t last_SMEM_stall;
   int last_SMEM_dep_idx;
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

template <typename T>
void move_element(T& list, size_t idx, size_t before) {
    if (idx < before) {
        auto begin = std::next(list.begin(), idx);
        auto end = std::next(list.begin(), before);
        std::rotate(begin, begin + 1, end);
    } else if (idx > before) {
        auto begin = std::next(list.begin(), before);
        auto end = std::next(list.begin(), idx + 1);
        std::rotate(begin, end - 1, end);
    }
}

static std::pair<int16_t, int16_t> getLiveChanges(aco_ptr<Instruction>& instr)
{
   std::pair<int16_t, int16_t> changes{0, 0};
   for (unsigned i = 0; i < instr->definitionCount(); i++) {
      Definition& def = instr->getDefinition(i);
      if (!def.isTemp() || def.isKill())
         continue;
      if (def.regClass().type() == vgpr)
         changes.second += def.size();
      else
         changes.first += def.size();
   }

   for (unsigned i = 0; i < instr->operandCount(); i++) {
      Operand& op = instr->getOperand(i);
      if (!op.isTemp() || !op.isFirstKill())
         continue;
      if (op.regClass().type() == vgpr)
         changes.second -= op.size();
      else
         changes.first -= op.size();
   }

   return changes;
}

static std::pair<uint16_t, uint16_t> getTempRegisters(aco_ptr<Instruction>& instr)
{
   std::pair<uint16_t, uint16_t> temp_registers{0, 0};
   for (unsigned i = 0; i < instr->definitionCount(); i++) {
      Definition& def = instr->getDefinition(i);
      if (!def.isTemp() || !def.isKill())
         continue;
      if (def.regClass().type() == vgpr)
         temp_registers.second += def.size();
      else
         temp_registers.first += def.size();
   }

   return temp_registers;
}

barrier_interaction get_barrier_interaction(Instruction* instr)
{
   switch (instr->format) {
   case Format::SMEM:
      return static_cast<SMEM_instruction*>(instr)->barrier;
   case Format::MUBUF:
      return static_cast<MUBUF_instruction*>(instr)->barrier;
   case Format::MIMG:
      return static_cast<MIMG_instruction*>(instr)->barrier;
   case Format::FLAT:
   case Format::GLOBAL:
      return barrier_buffer;
   case Format::DS:
      return barrier_shared;
   default:
      return barrier_none;
   }
}

bool can_move_instr(aco_ptr<Instruction>& instr, Instruction* current, int moving_interaction)
{
   /* don't move exports so that they stay closer together */
   if (instr->format == Format::EXP)
      return false;

   /* handle barriers */

   /* TODO: instead of stopping, maybe try to move the barriers and any
    * instructions interacting with them instead? */
   if (instr->format != Format::PSEUDO_BARRIER) {
      if (instr->opcode == aco_opcode::s_barrier) {
         bool can_reorder = false;
         switch (current->format) {
         case Format::SMEM:
            can_reorder = static_cast<SMEM_instruction*>(current)->can_reorder;
            break;
         case Format::MUBUF:
            can_reorder = static_cast<MUBUF_instruction*>(current)->can_reorder;
            break;
         case Format::MIMG:
            can_reorder = static_cast<MIMG_instruction*>(current)->can_reorder;
            break;
         default:
            break;
         }
         return can_reorder && moving_interaction == barrier_none;
      } else {
         return true;
      }
   }

   int interaction = get_barrier_interaction(current);
   interaction |= moving_interaction;

   switch (instr->opcode) {
   case aco_opcode::p_memory_barrier_atomic:
      return !(interaction & barrier_atomic);
   /* For now, buffer and image barriers are treated the same. this is because of
    * dEQP-VK.memory_model.message_passing.core11.u32.coherent.fence_fence.atomicwrite.device.payload_nonlocal.buffer.guard_nonlocal.image.comp
    * which seems to use an image load to determine if the result of a buffer load is valid. So the ordering of the two loads is important.
    * I /think/ we should probably eventually expand the meaning of a buffer barrier so that all buffer operations before it, must stay before it
    * and that both image and buffer operations after it, must stay after it. We should also do the same for image barriers.
    * Or perhaps the problem is that we don't have a combined barrier instruction for both buffers and images, but the CTS test expects us to?
    * Either way, this solution should work. */
   case aco_opcode::p_memory_barrier_buffer:
   case aco_opcode::p_memory_barrier_image:
      return !(interaction & (barrier_image | barrier_buffer));
   case aco_opcode::p_memory_barrier_shared:
      return !(interaction & barrier_shared);
   case aco_opcode::p_memory_barrier_all:
      return interaction == barrier_none;
   default:
      return false;
   }
}

bool can_reorder(Instruction* candidate, bool allow_smem)
{
   switch (candidate->format) {
   case Format::SMEM:
      return allow_smem || static_cast<SMEM_instruction*>(candidate)->can_reorder;
   case Format::MUBUF:
      return static_cast<MUBUF_instruction*>(candidate)->can_reorder;
   case Format::MIMG:
      return static_cast<MIMG_instruction*>(candidate)->can_reorder;
   case Format::FLAT:
   case Format::GLOBAL:
   case Format::SCRATCH:
   case Format::MTBUF:
      return false;
   default:
      return true;
   }
}

void schedule_SMEM(sched_ctx& ctx, std::unique_ptr<Block>& block,
                   std::vector<std::pair<uint16_t,uint16_t>>& register_demand,
                   Instruction* current, int idx)
{
   assert(idx != 0);
   int window_size = SMEM_WINDOW_SIZE;
   int max_moves = SMEM_MAX_MOVES;
   int16_t k = 0;
   bool can_reorder_cur = can_reorder(current, false);

   /* create the initial set of values which current depends on */
   std::fill(ctx.depends_on.begin(), ctx.depends_on.end(), false);
   for (unsigned i = 0; i < current->num_operands; i++) {
      if (current->getOperand(i).isTemp())
         ctx.depends_on[current->getOperand(i).tempId()] = true;
   }

   /* maintain how many registers remain free when moving instructions */
   int sgpr_pressure = register_demand[idx].first;
   int vgpr_pressure = register_demand[idx].second;

   /* first, check if we have instructions before current to move down */
   int insert_idx = idx + 1;
   int moving_interaction = barrier_none;

   for (int candidate_idx = idx - 1; k < max_moves && candidate_idx > (int) idx - window_size; candidate_idx--) {
      assert(candidate_idx >= 0);
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      /* break if we'd make the previous SMEM instruction stall */
      bool can_stall_prev_smem = idx <= ctx.last_SMEM_dep_idx && candidate_idx < ctx.last_SMEM_dep_idx;
      if (can_stall_prev_smem && ctx.last_SMEM_stall >= 0)
         break;

      /* break when encountering another MEM instruction, logical_start or barriers */
      if (!can_reorder(candidate.get(), false) && !can_reorder_cur)
         break;
      if (candidate->opcode == aco_opcode::p_logical_start)
         break;
      if (!can_move_instr(candidate, current, moving_interaction))
         break;

      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx].second);

      /* if current depends on candidate, add additional dependencies and continue */
      bool can_move_down = true;
      bool writes_exec = false;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isTemp() && ctx.depends_on[candidate->getDefinition(i).tempId()])
            can_move_down = false;
         if (candidate->getDefinition(i).isFixed() && candidate->getDefinition(i).physReg() == exec)
            writes_exec = true;
      }
      if (writes_exec)
         break;

      if ((moving_interaction & barrier_shared) && candidate->format == Format::DS)
         can_move_down = false;
      moving_interaction |= get_barrier_interaction(candidate.get());
      if (!can_move_down) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.depends_on[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      bool register_pressure_unknown = false;
      /* check if one of candidate's operands is killed by depending instruction */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && ctx.depends_on[candidate->getOperand(i).tempId()]) {
            // FIXME: account for difference in register pressure
            register_pressure_unknown = true;
         }
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.depends_on[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff, candidate_vgpr_diff, temp_sgpr, temp_vgpr;
      std::tie(candidate_sgpr_diff, candidate_vgpr_diff) = getLiveChanges(candidate);
      std::tie(temp_sgpr, temp_vgpr) = getTempRegisters(candidate);
      if (vgpr_pressure - candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure - candidate_sgpr_diff > ctx.max_sgpr)
         break;
      int temp_sgpr2, temp_vgpr2;
      std::tie(temp_sgpr2, temp_vgpr2) = getTempRegisters(block->instructions[insert_idx - 1]);
      uint16_t new_sgpr_demand, new_vgpr_demand;
      new_sgpr_demand = register_demand[insert_idx - 1].first - temp_sgpr2 + temp_sgpr;
      new_vgpr_demand = register_demand[insert_idx - 1].second - temp_vgpr2 + temp_vgpr;
      if (new_sgpr_demand > ctx.max_sgpr || new_vgpr_demand > ctx.max_vgpr)
         break;
      // TODO: we might want to look further to find a sequence of instructions to move down which doesn't exceed reg pressure

      /* move the candidate below the memory load */
      move_element(block->instructions, candidate_idx, insert_idx);

      /* update register pressure */
      move_element(register_demand, candidate_idx, insert_idx);
      for (int i = candidate_idx; i < insert_idx - 1; i++) {
         register_demand[i].first -= candidate_sgpr_diff;
         register_demand[i].second -= candidate_vgpr_diff;
      }
      register_demand[insert_idx - 1].first = new_sgpr_demand;
      register_demand[insert_idx - 1].second = new_vgpr_demand;
      vgpr_pressure = vgpr_pressure - candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure - candidate_sgpr_diff;

      if (candidate_idx < ctx.last_SMEM_dep_idx)
         ctx.last_SMEM_stall++;
      insert_idx--;
      k++;
   }

   /* create the initial set of values which depend on current */
   std::fill(ctx.depends_on.begin(), ctx.depends_on.end(), false);
   std::fill(ctx.RAR_dependencies.begin(), ctx.RAR_dependencies.end(), false);
   for (unsigned i = 0; i < current->num_definitions; i++) {
      if (current->getDefinition(i).isTemp())
         ctx.depends_on[current->getDefinition(i).tempId()] = true;
   }

   /* find the first instruction depending on current or find another MEM */
   insert_idx = idx + 1;
   moving_interaction = barrier_none;

   bool found_dependency = false;
   /* second, check if we have instructions after current to move up */
   for (int candidate_idx = idx + 1; k < max_moves && candidate_idx < (int) idx + window_size; candidate_idx++) {
      assert(candidate_idx < (int) block->instructions.size());
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      if (candidate->opcode == aco_opcode::p_logical_end)
         break;
      if (!can_move_instr(candidate, current, moving_interaction))
         break;

      bool writes_exec = false;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isFixed() && candidate->getDefinition(i).physReg() == exec)
            writes_exec = true;
      }
      if (writes_exec)
         break;

      /* check if candidate depends on current */
      bool is_dependency = false;
      for (unsigned i = 0; !is_dependency && i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && ctx.depends_on[candidate->getOperand(i).tempId()])
            is_dependency = true;
      }
      if ((moving_interaction & barrier_shared) && candidate->format == Format::DS)
         is_dependency = true;
      moving_interaction |= get_barrier_interaction(candidate.get());
      if (is_dependency) {
         for (unsigned j = 0; j < candidate->num_definitions; j++) {
            if (candidate->getDefinition(j).isTemp())
               ctx.depends_on[candidate->getDefinition(j).tempId()] = true;
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.RAR_dependencies[candidate->getOperand(i).tempId()] = true;
         }
         if (!found_dependency) {
            insert_idx = candidate_idx;
            found_dependency = true;
            /* init register pressure */
            sgpr_pressure = register_demand[insert_idx - 1].first;
            vgpr_pressure = register_demand[insert_idx - 1].second;
         }
      }

      if (!can_reorder(candidate.get(), false) && !can_reorder_cur)
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
         if (candidate->getOperand(i).isTemp() && ctx.RAR_dependencies[candidate->getOperand(i).tempId()])
            register_pressure_unknown = true;
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_definitions; i++) {
            if (candidate->getDefinition(i).isTemp())
               ctx.RAR_dependencies[candidate->getDefinition(i).tempId()] = true;
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.RAR_dependencies[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff, candidate_vgpr_diff, temp_sgpr, temp_vgpr;
      std::tie(candidate_sgpr_diff, candidate_vgpr_diff) = getLiveChanges(candidate);
      std::tie(temp_sgpr, temp_vgpr) = getTempRegisters(candidate);
      if (vgpr_pressure + candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure + candidate_sgpr_diff > ctx.max_sgpr)
         break;
      int temp_sgpr2, temp_vgpr2;
      std::tie(temp_sgpr2, temp_vgpr2) = getTempRegisters(block->instructions[insert_idx - 1]);
      uint16_t new_sgpr_demand, new_vgpr_demand;
      new_sgpr_demand = register_demand[insert_idx - 1].first - temp_sgpr2 + candidate_sgpr_diff + temp_sgpr;
      new_vgpr_demand = register_demand[insert_idx - 1].second - temp_vgpr2 + candidate_vgpr_diff + temp_vgpr;
      if (new_sgpr_demand > ctx.max_sgpr || new_vgpr_demand > ctx.max_vgpr)
         break;

      /* move the candidate above the insert_idx */
      move_element(block->instructions, candidate_idx, insert_idx);

      /* update register pressure */
      move_element(register_demand, candidate_idx, insert_idx);
      for (int i = insert_idx + 1; i <= candidate_idx; i++) {
         register_demand[i].first += candidate_sgpr_diff;
         register_demand[i].second += candidate_vgpr_diff;
      }
      register_demand[insert_idx].first = new_sgpr_demand;
      register_demand[insert_idx].second = new_vgpr_demand;
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
                   Instruction* current, int idx)
{
   assert(idx != 0);
   int window_size = VMEM_WINDOW_SIZE;
   int max_moves = VMEM_MAX_MOVES;
   int16_t k = 0;
   bool can_reorder_cur = can_reorder(current, false);

   /* create the initial set of values which current depends on */
   std::fill(ctx.depends_on.begin(), ctx.depends_on.end(), false);
   for (unsigned i = 0; i < current->num_operands; i++) {
      if (current->getOperand(i).isTemp())
         ctx.depends_on[current->getOperand(i).tempId()] = true;
   }

   /* maintain how many registers remain free when moving instructions */
   int sgpr_pressure = register_demand[idx].first;
   int vgpr_pressure = register_demand[idx].second;

   /* first, check if we have instructions before current to move down */
   int insert_idx = idx + 1;
   int moving_interaction = barrier_none;

   for (int candidate_idx = idx - 1; k < max_moves && candidate_idx > (int) idx - window_size; candidate_idx--) {
      assert(candidate_idx >= 0);
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      /* break when encountering another VMEM instruction, logical_start or barriers */
      if (!can_reorder(candidate.get(), true) && !can_reorder_cur)
         break;
      if (candidate->opcode == aco_opcode::p_logical_start)
         break;
      if (!can_move_instr(candidate, current, moving_interaction))
         break;

      /* break if we'd make the previous SMEM instruction stall */
      bool can_stall_prev_smem = idx <= ctx.last_SMEM_dep_idx && candidate_idx < ctx.last_SMEM_dep_idx;
      if (can_stall_prev_smem && ctx.last_SMEM_stall >= 0)
         break;

      sgpr_pressure = std::max(sgpr_pressure, (int) register_demand[candidate_idx].first);
      vgpr_pressure = std::max(vgpr_pressure, (int) register_demand[candidate_idx].second);

      /* if current depends on candidate, add additional dependencies and continue */
      bool can_move_down = true;
      bool writes_exec = false;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isTemp() && ctx.depends_on[candidate->getDefinition(i).tempId()])
            can_move_down = false;
         if (candidate->getDefinition(i).isFixed() && candidate->getDefinition(i).physReg() == exec)
            writes_exec = true;
      }
      if (writes_exec)
         break;

      if ((moving_interaction & barrier_shared) && candidate->format == Format::DS)
         can_move_down = false;
      moving_interaction |= get_barrier_interaction(candidate.get());
      if (!can_move_down) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.depends_on[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      bool register_pressure_unknown = false;
      /* check if one of candidate's operands is killed by depending instruction */
      for (unsigned i = 0; i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && ctx.depends_on[candidate->getOperand(i).tempId()]) {
            // FIXME: account for difference in register pressure
            register_pressure_unknown = true;
         }
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.depends_on[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff, candidate_vgpr_diff, temp_sgpr, temp_vgpr;
      std::tie(candidate_sgpr_diff, candidate_vgpr_diff) = getLiveChanges(candidate);
      std::tie(temp_sgpr, temp_vgpr) = getTempRegisters(candidate);
      if (vgpr_pressure - candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure - candidate_sgpr_diff > ctx.max_sgpr)
         break;
      int temp_sgpr2, temp_vgpr2;
      std::tie(temp_sgpr2, temp_vgpr2) = getTempRegisters(block->instructions[insert_idx - 1]);
      uint16_t new_sgpr_demand, new_vgpr_demand;
      new_sgpr_demand = register_demand[insert_idx - 1].first - temp_sgpr2 + temp_sgpr;
      new_vgpr_demand = register_demand[insert_idx - 1].second - temp_vgpr2 + temp_vgpr;
      if (new_sgpr_demand > ctx.max_sgpr || new_vgpr_demand > ctx.max_vgpr)
         break;
      // TODO: we might want to look further to find a sequence of instructions to move down which doesn't exceed reg pressure

      /* move the candidate below the memory load */
      move_element(block->instructions, candidate_idx, insert_idx);

      /* update register pressure */
      move_element(register_demand, candidate_idx, insert_idx);
      for (int i = candidate_idx; i < insert_idx - 1; i++) {
         register_demand[i].first -= candidate_sgpr_diff;
         register_demand[i].second -= candidate_vgpr_diff;
      }
      register_demand[insert_idx - 1].first = new_sgpr_demand;
      register_demand[insert_idx - 1].second = new_vgpr_demand;
      vgpr_pressure = vgpr_pressure - candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure - candidate_sgpr_diff;
      insert_idx--;
      k++;
      if (candidate_idx < ctx.last_SMEM_dep_idx)
         ctx.last_SMEM_stall++;
   }

   /* create the initial set of values which depend on current */
   std::fill(ctx.depends_on.begin(), ctx.depends_on.end(), false);
   std::fill(ctx.RAR_dependencies.begin(), ctx.RAR_dependencies.end(), false);
   for (unsigned i = 0; i < current->num_definitions; i++) {
      if (current->getDefinition(i).isTemp())
         ctx.depends_on[current->getDefinition(i).tempId()] = true;
   }

   /* find the first instruction depending on current or find another VMEM */
   insert_idx = idx;
   moving_interaction = barrier_none;

   bool found_dependency = false;
   /* second, check if we have instructions after current to move up */
   for (int candidate_idx = idx + 1; k < max_moves && candidate_idx < (int) idx + window_size; candidate_idx++) {
      assert(candidate_idx < (int) block->instructions.size());
      aco_ptr<Instruction>& candidate = block->instructions[candidate_idx];

      if (candidate->opcode == aco_opcode::p_logical_end)
         break;
      if (!can_move_instr(candidate, current, moving_interaction))
         break;

      bool writes_exec = false;
      for (unsigned i = 0; i < candidate->num_definitions; i++) {
         if (candidate->getDefinition(i).isFixed() && candidate->getDefinition(i).physReg() == exec)
            writes_exec = true;
      }
      if (writes_exec)
         break;

      /* check if candidate depends on current */
      bool is_dependency = !can_reorder(candidate.get(), true) && !can_reorder_cur;
      for (unsigned i = 0; !is_dependency && i < candidate->num_operands; i++) {
         if (candidate->getOperand(i).isTemp() && ctx.depends_on[candidate->getOperand(i).tempId()])
            is_dependency = true;
      }
      if ((moving_interaction & barrier_shared) && candidate->format == Format::DS)
         is_dependency = true;
      moving_interaction |= get_barrier_interaction(candidate.get());
      if (is_dependency) {
         for (unsigned j = 0; j < candidate->num_definitions; j++) {
            if (candidate->getDefinition(j).isTemp())
               ctx.depends_on[candidate->getDefinition(j).tempId()] = true;
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.RAR_dependencies[candidate->getOperand(i).tempId()] = true;
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
         if (candidate->getOperand(i).isTemp() && ctx.RAR_dependencies[candidate->getOperand(i).tempId()])
            register_pressure_unknown = true;
      }
      if (register_pressure_unknown) {
         for (unsigned i = 0; i < candidate->num_definitions; i++) {
            if (candidate->getDefinition(i).isTemp())
               ctx.RAR_dependencies[candidate->getDefinition(i).tempId()] = true;
         }
         for (unsigned i = 0; i < candidate->num_operands; i++) {
            if (candidate->getOperand(i).isTemp())
               ctx.RAR_dependencies[candidate->getOperand(i).tempId()] = true;
         }
         continue;
      }

      /* check if register pressure is low enough: the diff is negative if register pressure is decreased */
      int candidate_sgpr_diff, candidate_vgpr_diff, temp_sgpr, temp_vgpr;
      std::tie(candidate_sgpr_diff, candidate_vgpr_diff) = getLiveChanges(candidate);
      std::tie(temp_sgpr, temp_vgpr) = getTempRegisters(candidate);
      if (vgpr_pressure + candidate_vgpr_diff > ctx.max_vgpr ||
          sgpr_pressure + candidate_sgpr_diff > ctx.max_sgpr)
         break;
      int temp_sgpr2, temp_vgpr2;
      std::tie(temp_sgpr2, temp_vgpr2) = getTempRegisters(block->instructions[insert_idx - 1]);
      uint16_t new_sgpr_demand, new_vgpr_demand;
      new_sgpr_demand = register_demand[insert_idx - 1].first - temp_sgpr2 + candidate_sgpr_diff + temp_sgpr;
      new_vgpr_demand = register_demand[insert_idx - 1].second - temp_vgpr2 + candidate_vgpr_diff + temp_vgpr;
      if (new_sgpr_demand > ctx.max_sgpr || new_vgpr_demand > ctx.max_vgpr)
         break;

      /* move the candidate above the insert_idx */
      move_element(block->instructions, candidate_idx, insert_idx);

      /* update register pressure */
      move_element(register_demand, candidate_idx, insert_idx);
      for (int i = insert_idx + 1; i <= candidate_idx; i++) {
         register_demand[i].first += candidate_sgpr_diff;
         register_demand[i].second += candidate_vgpr_diff;
      }
      register_demand[insert_idx].first = new_sgpr_demand;
      register_demand[insert_idx].second = new_vgpr_demand;
      vgpr_pressure = vgpr_pressure + candidate_vgpr_diff;
      sgpr_pressure = sgpr_pressure + candidate_sgpr_diff;
      insert_idx++;
      k++;
   }
}

void schedule_block(sched_ctx& ctx, Program *program, std::unique_ptr<Block>& block, live& live_vars)
{
   ctx.last_SMEM_dep_idx = 0;
   ctx.last_SMEM_stall = INT16_MIN;

   /* go through all instructions and find memory loads */
   for (unsigned idx = 0; idx < block->instructions.size(); idx++) {
      Instruction* current = block->instructions[idx].get();

      if (!current->num_definitions)
         continue;

      if (current->isVMEM())
         schedule_VMEM(ctx, block, live_vars.register_demand[block->index], current, idx);
      if (current->format == Format::SMEM)
         schedule_SMEM(ctx, block, live_vars.register_demand[block->index], current, idx);
   }

   /* resummarize the block's register demand */
   block->vgpr_demand = 0;
   block->sgpr_demand = 0;
   for (unsigned idx = 0; idx < block->instructions.size(); idx++) {
      block->vgpr_demand = std::max(block->vgpr_demand, live_vars.register_demand[block->index][idx].second);
      block->sgpr_demand = std::max(block->sgpr_demand, live_vars.register_demand[block->index][idx].first);
   }
}


void schedule_program(Program *program, live& live_vars)
{
   sched_ctx ctx;
   ctx.depends_on.resize(program->peekAllocationId());
   ctx.RAR_dependencies.resize(program->peekAllocationId());
   /* Allowing the scheduler to reduce the number of waves to as low as 5
    * improves performance of Thrones of Britannia significantly and doesn't
    * seem to hurt anything else. */
   //TODO: maybe use some sort of heuristic instead
   ctx.num_waves = std::min<uint16_t>(program->num_waves, 5);
   assert(ctx.num_waves);
   uint16_t total_sgpr_regs = program->chip_class >= GFX8 ? 800 : 512;
   uint16_t max_addressible_sgpr = program->chip_class >= GFX8 ? 102 : 104;
   ctx.max_sgpr = std::min<uint16_t>(((total_sgpr_regs / ctx.num_waves) & ~7) - 2, max_addressible_sgpr);
   ctx.max_vgpr = (256 / ctx.num_waves) & ~3;

   for (std::unique_ptr<Block>& block : program->blocks)
      schedule_block(ctx, program, block, live_vars);

   /* update max_vgpr, max_sgpr and num_waves */
   uint16_t sgpr_demand = 0;
   uint16_t vgpr_demand = 0;
   for (std::unique_ptr<Block>& block : program->blocks) {
      sgpr_demand = std::max(block->sgpr_demand, sgpr_demand);
      vgpr_demand = std::max(block->vgpr_demand, vgpr_demand);
   }
   update_vgpr_sgpr_demand(program, vgpr_demand, sgpr_demand);

   /* if enabled, this code asserts that register_demand is updated correctly */
   #if 0
   int prev_num_waves = program->num_waves;
   int prev_max_sgpr = program->max_sgpr;
   int prev_max_vgpr = program->max_vgpr;

   unsigned vgpr_demands[program->blocks.size()];
   unsigned sgpr_demands[program->blocks.size()];
   for (unsigned j = 0; j < program->blocks.size(); j++) {
      vgpr_demands[j] = program->blocks[j]->vgpr_demand;
      sgpr_demands[j] = program->blocks[j]->sgpr_demand;
   }

   struct radv_nir_compiler_options options;
   options.chip_class = program->chip_class;
   live live_vars2 = aco::live_var_analysis<true>(program, &options);

   for (unsigned j = 0; j < program->blocks.size(); j++) {
      Block *b = program->blocks[j].get();
      for (unsigned i = 0; i < b->instructions.size(); i++)
         assert(live_vars.register_demand[b->index][i] == live_vars2.register_demand[b->index][i]);
      assert(b->vgpr_demand == vgpr_demands[j]);
      assert(b->sgpr_demand == sgpr_demands[j]);
   }

   assert(program->max_vgpr == prev_max_vgpr);
   assert(program->max_sgpr == prev_max_sgpr);
   assert(program->num_waves == prev_num_waves);
   #endif
}

}
