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
 * Authors:
 *    Daniel Schürmann (daniel.schuermann@campus.tu-berlin.de)
 *
 */


#include <unordered_map>
#include <unordered_set>

#include "aco_ir.h"
#include "aco_dominance.cpp"

/*
 * Implements the algorithm for dominator-tree value numbering
 * from "Value Numbering" by Briggs, Cooper, and Simpson.
 */

namespace aco {

struct InstrHash {
   std::size_t operator()(Instruction* instr) const
   {
      uint64_t hash = (uint64_t) instr->opcode + (uint64_t) instr->format;
      for (unsigned i = 0; i < instr->num_operands; i++) {
         Operand op = instr->getOperand(i);
         uint64_t val = op.isTemp() ? op.tempId() : op.isFixed() ? op.physReg().reg : op.constantValue();
         hash |= val << (i+1) * 8;
      }
      if (instr->isVOP3()) {
         VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr);
         for (unsigned i = 0; i < 3; i++) {
            hash ^= vop3->abs[i] << (i*3 + 0);
            hash ^= vop3->opsel[i] << (i*3 + 1);
            hash ^= vop3->neg[i] << (i*3 + 2);
         }
         hash ^= (vop3->clamp << 28) * 13;
         hash += vop3->omod << 19;
      }
      switch (instr->format) {
      case Format::SMEM:
         break;
      case Format::VINTRP: {
         Interp_instruction* interp = static_cast<Interp_instruction*>(instr);
         hash ^= interp->attribute << 13;
         hash ^= interp->component << 27;
         break;
      }
      case Format::DS:
         break;
      default:
         break;
      }

      return hash;
   }
};

struct InstrPred {
   bool operator()(Instruction* a, Instruction* b) const
   {
      if (a->format != b->format)
         return false;
      if (a->opcode != b->opcode)
         return false;
      if (a->num_operands != b->num_operands || a->num_definitions != b->num_definitions)
         return false; /* possible with pseudo-instructions */
      for (unsigned i = 0; i < a->num_operands; i++) {
         if (a->getOperand(i).isConstant()) {
            if (!b->getOperand(i).isConstant())
               return false;
            if (a->getOperand(i).constantValue() != b->getOperand(i).constantValue())
               return false;
         }
         else if (a->getOperand(i).isTemp()) {
            if (!b->getOperand(i).isTemp())
               return false;
            if (a->getOperand(i).tempId() != b->getOperand(i).tempId())
               return false;
         }
         else if (a->getOperand(i).isUndefined() ^ b->getOperand(i).isUndefined())
            return false;
         if (a->getOperand(i).isFixed()) {
            if (a->getOperand(i).physReg() == exec)
               return false;
            if (!b->getOperand(i).isFixed())
               return false;
            if (!(a->getOperand(i).physReg() == b->getOperand(i).physReg()))
               return false;
         }
      }
      for (unsigned i = 0; i < a->num_definitions; i++) {
         if (a->getDefinition(i).isTemp()) {
            if (!b->getDefinition(i).isTemp())
               return false;
            if (a->getDefinition(i).regClass() != b->getDefinition(i).regClass())
               return false;
         }
         if (a->getDefinition(i).isFixed()) {
            if (!b->getDefinition(i).isFixed())
               return false;
            if (!(a->getDefinition(i).physReg() == b->getDefinition(i).physReg()))
               return false;
         }
      }
      if (a->format == Format::PSEUDO_BRANCH)
         return false;
      if (a->isVOP3()) {
         VOP3A_instruction* a3 = static_cast<VOP3A_instruction*>(a);
         VOP3A_instruction* b3 = static_cast<VOP3A_instruction*>(b);
         for (unsigned i = 0; i < 3; i++) {
            if (a3->abs[i] != b3->abs[i] ||
                a3->opsel[i] != b3->opsel[i] ||
                a3->neg[i] != b3->neg[i])
               return false;
         }
         return a3->clamp == b3->clamp &&
                a3->omod == b3->omod;
      }
      if (a->isDPP()) {
         DPP_instruction* aDPP = static_cast<DPP_instruction*>(a);
         DPP_instruction* bDPP = static_cast<DPP_instruction*>(b);
         return aDPP->dpp_ctrl == bDPP->dpp_ctrl &&
                aDPP->bank_mask == bDPP->bank_mask &&
                aDPP->row_mask == bDPP->row_mask &&
                aDPP->bound_ctrl == bDPP->bound_ctrl &&
                aDPP->abs[0] == bDPP->abs[0] &&
                aDPP->abs[1] == bDPP->abs[1] &&
                aDPP->neg[0] == bDPP->neg[0] &&
                aDPP->neg[1] == bDPP->neg[1];
      }
      switch (a->format) {
         case Format::SOPK: {
            SOPK_instruction* aK = static_cast<SOPK_instruction*>(a);
            SOPK_instruction* bK = static_cast<SOPK_instruction*>(b);
            return aK->imm == bK->imm;
         }
         case Format::SMEM: {
            SMEM_instruction* aS = static_cast<SMEM_instruction*>(a);
            SMEM_instruction* bS = static_cast<SMEM_instruction*>(b);
            return aS->glc == bS->glc && aS->nv == bS->nv;
         }
         case Format::VINTRP: {
            Interp_instruction* aI = static_cast<Interp_instruction*>(a);
            Interp_instruction* bI = static_cast<Interp_instruction*>(b);
            if (aI->attribute != bI->attribute)
               return false;
            if (aI->component != bI->component)
               return false;
            return true;
         }
         case Format::DS: {
            DS_instruction* aDS = static_cast<DS_instruction*>(a);
            DS_instruction* bDS = static_cast<DS_instruction*>(b);
            return aDS->offset0 == bDS->offset0 &&
                   aDS->offset1 == bDS->offset1 &&
                   aDS->gds == bDS->gds;
         }
         /* we want to optimize these in NIR and don't hassle with load-store dependencies */
         case Format::MUBUF:
         case Format::MIMG:
            return false;
         default:
            return true;
      }
   }
};


void process_block(std::unique_ptr<Block>& block,
                   std::unordered_set<Instruction*, InstrHash, InstrPred>& expr_values,
                   std::unordered_map<uint32_t, Operand>& renames,
                   std::set<unsigned>& worklist)
{
   bool process_successors = false;
   bool run = false;
   Instruction* last_sopc = nullptr;
   std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
   std::vector<aco_ptr<Instruction>> new_instructions;
   new_instructions.reserve(block->instructions.size());

   while (it != block->instructions.end()) {
      aco_ptr<Instruction>& instr = *it;
      /* first, rename operands */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (!instr->getOperand(i).isTemp())
            continue;
         std::unordered_map<uint32_t, Operand>::iterator it = renames.find(instr->getOperand(i).tempId());
         if (it != renames.end())
            instr->getOperand(i) = it->second;
      }

      if (instr->opcode == aco_opcode::p_logical_start)
         run = true;
      if (instr->opcode == aco_opcode::p_logical_end)
         run = false;

      if (!instr->num_definitions || !run) {
         new_instructions.emplace_back(std::move(instr));
         ++it;
         continue;
      }

      // TODO: check if phi instructions are meaningless (i.e. all operands are the same)

      std::pair<std::unordered_set<Instruction*, InstrHash, InstrPred>::iterator, bool> res = expr_values.emplace(instr.get());

      /* if there was already an expression with the same value number */
      if (!res.second) {
         Instruction* orig_instr = *(res.first);
         assert(instr->num_definitions == orig_instr->num_definitions);
         for (unsigned i = 0; i < instr->num_definitions; i++) {
            assert(instr->getDefinition(i).regClass() == orig_instr->getDefinition(i).regClass());
            Operand new_op = Operand(orig_instr->getDefinition(i).getTemp());
            if (orig_instr->getDefinition(i).isFixed())
               new_op.setFixed(instr->getDefinition(i).physReg());
            process_successors |= renames.emplace(instr->getDefinition(i).tempId(), new_op).second;
         }
      } else if (instr->isSALU() &&
                 instr->getDefinition(instr->num_definitions - 1).isFixed() &&
                 instr->getDefinition(instr->num_definitions - 1).physReg().reg == scc.reg) {
         /* if the current instructions overwrites scc, we remove the previous scc instruction from the map */
         if (last_sopc)
            expr_values.erase(last_sopc);

         last_sopc = instr->getDefinition(instr->num_definitions - 1).isTemp() ? instr.get() : nullptr;
      }
      if (res.second)
         new_instructions.emplace_back(std::move(instr));
      ++it;
   }
   if (last_sopc)
      expr_values.erase(last_sopc);

   if (process_successors) {
      for (Block* succ : block->logical_successors)
         worklist.insert(succ->index);
   }

   block->instructions.swap(new_instructions);
}

void value_numbering(Program* program)
{
   std::vector<std::unordered_set<Instruction*, InstrHash, InstrPred>> expr_values(program->blocks.size());
   std::unordered_map<uint32_t, Operand> renames;

   /* we only process the logical cfg */
   std::set<unsigned> worklist;
   for (std::unique_ptr<Block>& block : program->blocks)
      if (block->logical_idom != -1)
         worklist.insert(block->index);

   while (!worklist.empty()) {
      std::set<unsigned>::iterator it = worklist.begin();
      unsigned block_idx = *it;
      worklist.erase(it);
      std::unique_ptr<Block>& block = program->blocks[block_idx];
      /* initialize expr_values from idom */
      expr_values[block_idx] = expr_values[block->logical_idom];
      process_block(block, expr_values[block_idx], renames, worklist);
   }
}

}
