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
   std::vector<bool> empty_blocks;
   Program* program;

   ssa_elimination_ctx(Program* program) : empty_blocks(program->blocks.size(), true), program(program) {}
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
               ctx.empty_blocks[preds[i]->index] = false;
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

void find_empty_blocks(ssa_elimination_ctx& ctx)
{
   for (unsigned i = 0; i < ctx.program->blocks.size(); i++) {
      if (!ctx.empty_blocks[i])
         continue;
      std::unique_ptr<Block>& block = ctx.program->blocks[i];

      std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
      for (; it != block->instructions.end(); ++it) {
         aco_opcode op = (*it)->opcode;
         if (op != aco_opcode::p_logical_start &&
             op != aco_opcode::p_logical_end &&
             op != aco_opcode::p_wqm &&
             op != aco_opcode::p_phi &&
             op != aco_opcode::p_linear_phi &&
             op != aco_opcode::p_startpgm &&
             (*it)->format != Format::PSEUDO_BRANCH) {
            ctx.empty_blocks[i] = false;
            break;
         }
      }
   }
}

void try_shrink_branch_block(ssa_elimination_ctx& ctx, std::unique_ptr<Block>& block)
{
   for (aco_ptr<Instruction>& instr : block->instructions) {

      /* TODO: shrink other pattern */
      if (!(instr->opcode == aco_opcode::p_linear_phi ||
            instr->opcode == aco_opcode::s_mov_b64 ||
            instr->opcode == aco_opcode::p_parallelcopy))
         return;

      /* find the parallelcopy instruction which moves the else-mask to exec */
      if (instr->opcode != aco_opcode::p_parallelcopy)
         continue;

      if (!(instr->getDefinition(0).physReg() == exec))
         return;

      for (aco_ptr<Instruction>& restore : block->linear_successors[0]->instructions) {
         if (restore->opcode == aco_opcode::p_phi || restore->opcode == aco_opcode::p_linear_phi)
            continue;

         if (restore->opcode != aco_opcode::s_or_b64)
            return;

         restore->getOperand(1) = instr->getOperand(0);

         assert(block->linear_predecessors[0]->linear_successors.size() == 1 &&
                block->linear_predecessors[1]->linear_successors.size() == 1);
         block->linear_predecessors[0]->linear_successors[0] = block->linear_successors[0];
         block->linear_predecessors[1]->linear_successors[0] = block->linear_successors[0];
         block->linear_successors[0]->linear_predecessors[0] = block->linear_predecessors[0];
         block->linear_successors[0]->linear_predecessors[1] = block->linear_predecessors[1];

         for (unsigned i = 0; i < 2; i++) {
            Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(block->linear_predecessors[0]->instructions.back().get());
            assert(branch->opcode == aco_opcode::p_branch);
            branch->targets[i] = block->linear_successors[0];
         }
         block->instructions.clear();
         block->linear_predecessors.clear();
         block->linear_successors.clear();
         return;
      }
   }
}

void try_remove_simple_block(ssa_elimination_ctx& ctx, std::unique_ptr<Block>& block)
{
   for (aco_ptr<Instruction>& instr : block->instructions) {
      if (instr->format != Format::PSEUDO &&
          instr->format != Format::PSEUDO_BRANCH) {
         return;
      }
   }

   Block* pred = block->linear_predecessors[0];
   Block* succ = block->linear_successors[0];
   Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(pred->instructions.back().get());
   if (branch->opcode == aco_opcode::p_branch) {
      branch->targets[0] = succ;
      branch->targets[1] = succ;
   } else if (branch->targets[0] == block.get()) {
      branch->targets[0] = succ;
   } else if (branch->targets[0] == succ) {
      assert(branch->targets[1] == block.get());
      branch->targets[1] = succ;
      branch->opcode = aco_opcode::p_branch;
   } else if (branch->targets[1] == block.get()) {
      /* check if there is a fall-through path from block to succ */
      bool falls_through = true;
      for (unsigned j = block->index + 1; falls_through && j < succ->index; j++) {
         assert(ctx.program->blocks[j]->index == j);
         if (!ctx.program->blocks[j]->instructions.empty()) {
            assert(ctx.program->blocks[j].get() == branch->targets[0]);
            falls_through = false;
         }
      }
      if (falls_through) {
         branch->targets[1] = succ;
      } else {
         /* This is a (uniform) break or continue block. The branch condition has to be inverted. */
         if (branch->opcode == aco_opcode::p_cbranch_z)
            branch->opcode = aco_opcode::p_cbranch_nz;
         else if (branch->opcode == aco_opcode::p_cbranch_nz)
            branch->opcode = aco_opcode::p_cbranch_z;
         else
            assert(false);
         branch->targets[1] = branch->targets[0];
         branch->targets[0] = succ;
      }
   } else {
      assert(false);
   }

   if (branch->targets[0] == branch->targets[1])
      branch->opcode = aco_opcode::p_branch;

   for (unsigned i = 0; i < pred->linear_successors.size(); i++)
      if (pred->linear_successors[i] == block.get())
         pred->linear_successors[i] = succ;
   for (unsigned i = 0; i < succ->linear_predecessors.size(); i++)
      if (succ->linear_predecessors[i] == block.get())
         succ->linear_predecessors[i] = pred;
   block->instructions.clear();
   block->linear_predecessors.clear();
   block->linear_successors.clear();
}

void jump_threading(ssa_elimination_ctx& ctx)
{
   for (int i = ctx.program->blocks.size() - 1; i >= 0; i--) {
      std::unique_ptr<Block>& block = ctx.program->blocks[i];

      if (block->linear_successors.size() == 2 &&
          block->linear_successors[0] == block->linear_successors[1]) {
         try_shrink_branch_block(ctx, block);
         continue;
      }

      if (!ctx.empty_blocks[i])
         continue;

      // TODO: we should also try to remove blocks with multiple pre- and successors
      if (block->linear_predecessors.size() == 1 && block->linear_successors.size() == 1)
         try_remove_simple_block(ctx, block);
   }
}

} /* end namespace */


void ssa_elimination(Program* program)
{
   ssa_elimination_ctx ctx(program);

   /* Collect information about every phi-instruction */
   collect_phi_info(ctx);

   /* eliminate empty blocks */
   find_empty_blocks(ctx);
   jump_threading(ctx);

   /* insert parallelcopies from SSA elimination */
   insert_parallelcopies(ctx);

}
}
