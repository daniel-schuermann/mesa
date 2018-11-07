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

#include <map>
#include <unordered_set>

#include "aco_ir.h"

/*
 * Implements the algorithm for dominator-tree value numbering
 * from "Value Numbering" by Briggs, Cooper, and Simpson.
 */

namespace aco {
namespace {

struct InstrHash {
   std::size_t operator()(Instruction* instr) const
   {
      uint64_t hash = (uint64_t) instr->opcode + (uint64_t) instr->format;
      for (unsigned i = 0; i < instr->num_operands; i++) {
         Operand op = instr->getOperand(i);
         uint64_t val = op.isTemp() ? op.tempId() : op.isFixed() ? op.physReg() : op.constantValue();
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
         case Format::VOPC: {
            /* Since the results depend on the exec mask, these shouldn't
             * be value numbered (this is especially useful for subgroupBallot()). */
            return false;
         }
         case Format::SOPK: {
            SOPK_instruction* aK = static_cast<SOPK_instruction*>(a);
            SOPK_instruction* bK = static_cast<SOPK_instruction*>(b);
            return aK->imm == bK->imm;
         }
         case Format::SMEM: {
            SMEM_instruction* aS = static_cast<SMEM_instruction*>(a);
            SMEM_instruction* bS = static_cast<SMEM_instruction*>(b);
            return aS->can_reorder && bS->can_reorder &&
                   aS->glc == bS->glc && aS->nv == bS->nv;
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
         case Format::PSEUDO_REDUCTION:
            return false;
         case Format::MTBUF: {
            /* this is fine since they are only used for vertex input fetches */
            MTBUF_instruction* aM = static_cast<MTBUF_instruction *>(a);
            MTBUF_instruction* bM = static_cast<MTBUF_instruction *>(b);
            return aM->dfmt == bM->dfmt &&
                   aM->nfmt == bM->nfmt &&
                   aM->offset == bM->offset &&
                   aM->offen == bM->offen &&
                   aM->idxen == bM->idxen &&
                   aM->glc == bM->glc &&
                   aM->slc == bM->slc &&
                   aM->tfe == bM->tfe &&
                   aM->disable_wqm == bM->disable_wqm;
         }
         /* we want to optimize these in NIR and don't hassle with load-store dependencies */
         case Format::MUBUF:
         case Format::FLAT:
         case Format::GLOBAL:
         case Format::SCRATCH:
         case Format::DS:
            return false;
         case Format::MIMG: {
            MIMG_instruction* aM = static_cast<MIMG_instruction*>(a);
            MIMG_instruction* bM = static_cast<MIMG_instruction*>(b);
            return aM->can_reorder && bM->can_reorder &&
                   aM->dmask == bM->dmask &&
                   aM->unrm == bM->unrm &&
                   aM->glc == bM->glc &&
                   aM->slc == bM->slc &&
                   aM->tfe == bM->tfe &&
                   aM->da == bM->da &&
                   aM->lwe == bM->lwe &&
                   aM->r128 == bM->r128 &&
                   aM->a16 == bM->a16 &&
                   aM->d16 == bM->d16 &&
                   aM->disable_wqm == bM->disable_wqm;
         }
         default:
            return true;
      }
   }
};


typedef std::unordered_set<Instruction*, InstrHash, InstrPred> expr_set;

void process_block(Block& block,
                   expr_set& expr_values,
                   std::map<uint32_t, Temp>& renames)
{
   bool run = false;
   std::vector<aco_ptr<Instruction>>::iterator it = block.instructions.begin();
   std::vector<aco_ptr<Instruction>> new_instructions;
   new_instructions.reserve(block.instructions.size());
   expr_set phi_values;

   while (it != block.instructions.end()) {
      aco_ptr<Instruction>& instr = *it;
      /* first, rename operands */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (!instr->getOperand(i).isTemp())
            continue;
         std::map<uint32_t, Temp>::iterator it = renames.find(instr->getOperand(i).tempId());
         if (it != renames.end())
            instr->getOperand(i).setTemp(it->second);
      }

      if (!instr->num_definitions || !run) {
         if (instr->opcode == aco_opcode::p_logical_start)
            run = true;
         else if (instr->opcode == aco_opcode::p_logical_end)
            run = false;
         else if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi) {
            std::pair<expr_set::iterator, bool> res = phi_values.emplace(instr.get());
            if (!res.second) {
               Instruction* orig_phi = *(res.first);
               renames.emplace(instr->getDefinition(0).tempId(), orig_phi->getDefinition(0).getTemp()).second;
               ++it;
               continue;
            }
         }
         new_instructions.emplace_back(std::move(instr));
         ++it;
         continue;
      }

      /* simple copy-propagation through renaming */
      if ((instr->opcode == aco_opcode::s_mov_b32 || instr->opcode == aco_opcode::s_mov_b64 || instr->opcode == aco_opcode::v_mov_b32) &&
          !instr->getDefinition(0).isFixed() && instr->getOperand(0).isTemp() && instr->getOperand(0).regClass() == instr->getDefinition(0).regClass() &&
          !instr->isDPP() && !((int)instr->format & (int)Format::SDWA)) {
         renames[instr->getDefinition(0).tempId()] = instr->getOperand(0).getTemp();
      }

      std::pair<expr_set::iterator, bool> res = expr_values.emplace(instr.get());

      /* if there was already an expression with the same value number */
      if (!res.second) {
         Instruction* orig_instr = *(res.first);
         assert(instr->num_definitions == orig_instr->num_definitions);
         for (unsigned i = 0; i < instr->num_definitions; i++) {
            assert(instr->getDefinition(i).regClass() == orig_instr->getDefinition(i).regClass());
            renames.emplace(instr->getDefinition(i).tempId(), orig_instr->getDefinition(i).getTemp()).second;
         }
      } else {
         new_instructions.emplace_back(std::move(instr));
      }
      ++it;
   }

   block.instructions.swap(new_instructions);
}

void rename_phi_operands(Block& block, std::map<uint32_t, Temp>& renames)
{
   for (aco_ptr<Instruction>& phi : block.instructions) {
      if (phi->opcode != aco_opcode::p_phi && phi->opcode != aco_opcode::p_linear_phi)
         break;

      for (unsigned i = 0; i < phi->num_operands; i++) {
         if (!phi->getOperand(i).isTemp())
            continue;
         std::map<uint32_t, Temp>::iterator it = renames.find(phi->getOperand(i).tempId());
         if (it != renames.end())
            phi->getOperand(i).setTemp(it->second);
      }
   }
}
} /* end namespace */


void value_numbering(Program* program)
{
   std::vector<expr_set> expr_values(program->blocks.size());
   std::map<uint32_t, Temp> renames;

   for (Block& block : program->blocks) {
      if (block.logical_idom != -1) {
         /* initialize expr_values from idom */
         expr_values[block.index] = expr_values[block.logical_idom];
         process_block(block, expr_values[block.index], renames);
      } else {
         expr_set empty;
         process_block(block, empty, renames);
      }
   }

   for (Block& block : program->blocks)
      rename_phi_operands(block, renames);
}

}
