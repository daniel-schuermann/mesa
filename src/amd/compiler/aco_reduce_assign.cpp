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

#include "aco_ir.h"
#include "aco_builder.h"

/*
 * Insert p_linear_start instructions right before RA to correctly allocate
 * temporaries for reductions that have to disrespect EXEC by executing in
 * WWM.
 */

namespace aco {

void setup_reduce_temp(Program* program)
{
   unsigned last_top_level_block_idx = 0;
   unsigned maxSize = 0;

   std::vector<bool> hasReductions(program->blocks.size());
   for (std::unique_ptr<Block>& block : program->blocks) {
      for (aco_ptr<Instruction>& instr : block->instructions) {
         if (instr->format != Format::PSEUDO_REDUCTION)
            continue;

         maxSize = MAX2(maxSize, instr->getOperand(0).size());
         hasReductions[block->index] = true;
      }
   }

   if (maxSize == 0)
      return;

   assert(maxSize == 1 || maxSize == 2);
   Temp reduceTmp(0, maxSize == 2 ? v2_linear : v1_linear);
   Temp vtmp(0, v1_linear);
   int inserted_at = -1;
   int vtmp_inserted_at = -1;

   for (std::unique_ptr<Block>& block : program->blocks) {
      if (block->is_top_level)
         last_top_level_block_idx = block->index;
      if (!hasReductions[block->index])
         continue;

      std::vector<aco_ptr<Instruction>>::iterator it;
      for (it = block->instructions.begin(); it != block->instructions.end(); ++it) {
         Instruction *instr = (*it).get();
         if (instr->format != Format::PSEUDO_REDUCTION)
            continue;

         ReduceOp op = static_cast<Pseudo_reduction_instruction *>(instr)->reduce_op;

         if ((int)last_top_level_block_idx != inserted_at) {
            reduceTmp = {program->allocateId(), reduceTmp.regClass()};
            aco_ptr<Instruction> create{create_instruction<Instruction>(aco_opcode::p_start_linear_vgpr, Format::PSEUDO, 0, 1)};
            create->getDefinition(0) = Definition(reduceTmp);
            /* find the right place to insert this definition */
            if (last_top_level_block_idx == block->index) {
               /* insert right before the current instruction */
               it = block->instructions.insert(it, std::move(create));
               it++;
               /* inserted_at is intentionally not updated here, so later blocks
                * would insert at the end instead of using this one. */
            } else {
               assert(last_top_level_block_idx < block->index);
               /* insert before the branch at last top level block */
               std::vector<aco_ptr<Instruction>>& instructions = program->blocks[last_top_level_block_idx]->instructions;
               instructions.insert(std::next(instructions.begin(), instructions.size() - 1), std::move(create));
               inserted_at = last_top_level_block_idx;
            }
         }

         /* same as before, except for the vector temporary instead of the reduce temporary */
         bool need_vtmp = op == imul32;
         need_vtmp |= static_cast<Pseudo_reduction_instruction *>(instr)->cluster_size == 32;
         if (need_vtmp && (int)last_top_level_block_idx != vtmp_inserted_at) {
            vtmp = {program->allocateId(), vtmp.regClass()};
            aco_ptr<Instruction> create{create_instruction<Instruction>(aco_opcode::p_start_linear_vgpr, Format::PSEUDO, 0, 1)};
            create->getDefinition(0) = Definition(vtmp);
            if (last_top_level_block_idx == block->index) {
               it = block->instructions.insert(it, std::move(create));
               it++;
            } else {
               assert(last_top_level_block_idx < block->index);
               std::vector<aco_ptr<Instruction>>& instructions = program->blocks[last_top_level_block_idx]->instructions;
               instructions.insert(std::next(instructions.begin(), instructions.size() - 1), std::move(create));
               vtmp_inserted_at = last_top_level_block_idx;
            }
         }

         Temp val = reduceTmp;
         if (val.size() != instr->getOperand(0).size()) {
            val = Temp{program->allocateId(), linearClass(instr->getOperand(0).regClass())};
            aco_ptr<Instruction> split{create_instruction<Instruction>(aco_opcode::p_split_vector, Format::PSEUDO, 1, 2)};
            split->getOperand(0) = Operand(reduceTmp);
            split->getDefinition(0) = Definition(val);
            it = block->instructions.insert(it, std::move(split));
            it++;
         }

         instr->getOperand(1) = Operand(reduceTmp);
         if (need_vtmp)
            instr->getOperand(2) = Operand(vtmp);

         /* scalar temporary */
         Builder bld(program);
         instr->getDefinition(1) = bld.def(s2);

         /* scalar identity temporary */
         if (instr->opcode == aco_opcode::p_exclusive_scan &&
             (op == imin32 || op == imin64 ||
              op == imax32 || op == imax64 ||
              op == fmin32 || op == fmin64 ||
              op == fmax32 || op == fmax64)) {
            instr->getDefinition(2) = bld.def(s1);
         }

         /* vcc clobber */
         if (op == iadd32 && program->chip_class < GFX9)
            instr->getDefinition(4) = Definition(vcc, s2);
      }
   }
}

};

