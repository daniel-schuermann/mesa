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

#include "aco_ir.h"

namespace aco {
namespace {

struct NOP_ctx {
   /* just initialize these with something less than max NOPs */
   int VALU_wrexec = -10;
   int VALU_wrvcc = -10;
   int VALU_wrsgpr = -10;
   enum chip_class chip_class;
   unsigned vcc_physical;
   NOP_ctx(Program* program) : chip_class(program->chip_class) {
      vcc_physical = program->config->num_sgprs - 2;
   }
};

bool VALU_writes_sgpr(aco_ptr<Instruction>& instr)
{
   if ((uint32_t) instr->format & (uint32_t) Format::VOPC)
      return true;
   if (instr->isVOP3() && instr->num_definitions == 2)
      return true;
   if (instr->opcode == aco_opcode::v_readfirstlane_b32 || instr->opcode == aco_opcode::v_readlane_b32)
      return true;
   return false;
}

bool regs_intersect(PhysReg a_reg, unsigned a_size, PhysReg b_reg, unsigned b_size)
{
   return a_reg.reg > b_reg.reg ?
          (a_reg.reg - b_reg.reg < b_size) :
          (b_reg.reg - a_reg.reg < a_size);
}

int handle_instruction(NOP_ctx& ctx, aco_ptr<Instruction>& instr,
                            std::vector<aco_ptr<Instruction>>& old_instructions,
                            std::vector<aco_ptr<Instruction>>& new_instructions)
{
   int new_idx = new_instructions.size();
   int NOPs = 0;

   // TODO: setreg / getreg / m0 writes
   // TODO: try to schedule the NOP-causing instruction up to reduce the number of stall cycles

   /* break off from prevous SMEM clause if needed */
   if (instr->format == Format::SMEM && ctx.chip_class >= GFX8) {
      bool is_store = instr->num_definitions == 0;
      for (int pred_idx = new_idx - 1; pred_idx >= 0; pred_idx--) {
         aco_ptr<Instruction>& pred = new_instructions[pred_idx];
         if (pred->format != Format::SMEM)
            break;

         /* Don't allow clauses with store instructions since the clause's
          * instructions may use the same address. */
         if (is_store || pred->num_definitions == 0)
            return 1;

         Definition& instr_def = instr->getDefinition(0);
         Definition& pred_def = pred->getDefinition(0);

         /* ISA reference doesn't say anything about this, but best to be safe */
         if (regs_intersect(instr_def.physReg(), instr_def.size(), pred_def.physReg(), pred_def.size()))
            return 1;

         for (unsigned i = 0; i < pred->num_operands; i++) {
            Operand& op = pred->getOperand(i);
            if (op.isConstant() || op.isUndefined() || !op.isFixed())
               continue;
            if (regs_intersect(instr_def.physReg(), instr_def.size(), op.physReg(), op.size()))
               return 1;
         }
         for (unsigned j = 0; j < instr->num_operands; j++) {
            Operand& op = instr->getOperand(j);
            if (op.isConstant() || op.isUndefined() || !op.isFixed())
               continue;
            if (regs_intersect(pred_def.physReg(), pred_def.size(), op.physReg(), op.size()))
               return 1;
         }
      }
   }

   if (instr->isVALU()) {
      if (instr->isDPP()) {
         /* VALU does not forward EXEC to DPP. */
         if (ctx.VALU_wrexec + 5 >= new_idx)
            NOPs = 5 + ctx.VALU_wrexec - new_idx + 1;

         /* VALU DPP reads VGPR written by VALU */
         for (int pred_idx = new_idx - 1; pred_idx >= 0 && pred_idx >= new_idx - 2; pred_idx--) {
            aco_ptr<Instruction>& pred = new_instructions[pred_idx];
            if ((pred->isVALU() || pred->format == Format::VINTRP) &&
                pred->num_definitions &&
                pred->getDefinition(0).physReg() == instr->getOperand(0).physReg()) {
               NOPs = std::max(NOPs, 2 + pred_idx - new_idx + 1);
               break;
            }
         }
      }
      for (unsigned i = 0; i < instr->num_operands; i++) {
         /* VALU which uses VCCZ */
         if (instr->getOperand(i).physReg() == PhysReg{251} &&
             ctx.VALU_wrvcc + 5 >= new_idx)
            NOPs = std::max(NOPs, 5 + ctx.VALU_wrvcc - new_idx + 1);

         /* VALU which uses EXECZ */
         if (instr->getOperand(i).physReg() == PhysReg{252} &&
             ctx.VALU_wrexec + 5 >= new_idx)
            NOPs = std::max(NOPs, 5 + ctx.VALU_wrexec - new_idx + 1);

         /* VALU which reads VCC as a constant */
         if (ctx.VALU_wrvcc + 1 >= new_idx) {
            for (unsigned k = 0; k < instr->getOperand(i).size(); k++) {
               unsigned reg = instr->getOperand(i).physReg().reg + k;
               if (reg == ctx.vcc_physical || reg == ctx.vcc_physical + 1)
                  NOPs = std::max(NOPs, 1);
            }
         }
      }

      switch (instr->opcode) {
         case aco_opcode::v_readlane_b32:
         case aco_opcode::v_writelane_b32: {
            if (ctx.VALU_wrsgpr + 4 < new_idx)
               break;
            PhysReg reg = instr->getOperand(1).physReg();
            for (int pred_idx = new_idx - 1; pred_idx >= 0 && pred_idx >= new_idx - 4; pred_idx--) {
               aco_ptr<Instruction>& pred = new_instructions[pred_idx];
               if (!pred->isVALU() || !VALU_writes_sgpr(pred))
                  continue;
               for (unsigned i = 0; i < pred->num_definitions; i++) {
                  if (pred->getDefinition(i).physReg() == reg)
                     NOPs = std::max(NOPs, 4 + pred_idx - new_idx + 1);
               }
            }
            break;
         }
         case aco_opcode::v_div_fmas_f32:
         case aco_opcode::v_div_fmas_f64: {
            if (ctx.VALU_wrvcc + 4 >= new_idx)
               NOPs = std::max(NOPs, 4 + ctx.VALU_wrvcc - new_idx + 1);
            break;
         }
         default:
            break;
      }

      /* Write VGPRs holding writedata > 64 bit from MIMG/MUBUF instructions */
      if (new_idx > 0) { //FIXME: handle case if the last instruction of a block without branch is such store
         aco_ptr<Instruction>& pred = new_instructions.back();
         if (pred->isVMEM() &&
             pred->num_operands == 3 &&
             pred->getOperand(2).size() > 2 &&
             pred->getOperand(1).size() != 8 &&
             (pred->format != Format::MUBUF || pred->getOperand(2).physReg().reg >= 102)) {
            /* Ops that use a 256-bit T# do not need a wait state.
             * BUFFER_STORE_* operations that use an SGPR for "offset"
             * do not require any wait states. */
            NOPs = std::max(NOPs, 1);
         }
      }

      if (VALU_writes_sgpr(instr)) {
         for (unsigned i = 0; i < instr->num_definitions; i++) {
            if (instr->getDefinition(i).physReg() == vcc)
               ctx.VALU_wrvcc = NOPs ? new_idx : new_idx + 1;
            else if (instr->getDefinition(i).physReg() == exec)
               ctx.VALU_wrexec = NOPs ? new_idx : new_idx + 1;
            else if (instr->getDefinition(i).physReg().reg <= 102)
               ctx.VALU_wrsgpr = NOPs ? new_idx : new_idx + 1;
         }
      }
      return NOPs;
   }

   /* If the VALU writes the SGPR that is used by a VMEM, the user must add five wait states. */
   if (instr->isVMEM() && ctx.VALU_wrsgpr + 5 >= new_idx) {
      for (int pred_idx = new_idx - 1; pred_idx >= 0 && pred_idx >= new_idx - 5; pred_idx--) {
         aco_ptr<Instruction>& pred = new_instructions[pred_idx];
         if (!(pred->isVALU() && VALU_writes_sgpr(pred)))
            continue;

         for (unsigned i = 0; i < pred->num_definitions; i++) {
            if (pred->getDefinition(i).physReg().reg > 102)
               continue;

            if (instr->num_operands > 1 &&
                regs_intersect(instr->getOperand(1).physReg(), instr->getOperand(1).size(),
                               pred->getDefinition(i).physReg(), pred->getDefinition(i).size())) {
                  return 5 + pred_idx - new_idx + 1;
            }

            if (instr->num_operands > 2 &&
                regs_intersect(instr->getOperand(2).physReg(), instr->getOperand(2).size(),
                               pred->getDefinition(i).physReg(), pred->getDefinition(i).size())) {
                  return 5 + pred_idx - new_idx + 1;
            }
         }
      }
   }

   /* SALU writes M0 */
   if (new_idx > 0 && ctx.chip_class >= GFX9 && instr->format == Format::VINTRP) {
      aco_ptr<Instruction>& pred = new_instructions.back();
      if (pred->isSALU() &&
          pred->num_definitions &&
          pred->getDefinition(0).physReg() == m0)
         return 1;
   }

   return 0;
}


void handle_block(NOP_ctx& ctx, std::unique_ptr<Block>& block)
{
   std::vector<aco_ptr<Instruction>> instructions;
   instructions.reserve(block->instructions.size());
   for (unsigned i = 0; i < block->instructions.size(); i++) {
      aco_ptr<Instruction>& instr = block->instructions[i];
      unsigned NOPs = handle_instruction(ctx, instr, block->instructions, instructions);
      if (NOPs) {
         // TODO: try to move the instruction down
         /* create NOP */
         aco_ptr<SOPP_instruction> nop{create_instruction<SOPP_instruction>(aco_opcode::s_nop, Format::SOPP, 0, 0)};
         nop->imm = NOPs - 1;
         instructions.emplace_back(std::move(nop));
      }

      instructions.emplace_back(std::move(instr));
   }

   ctx.VALU_wrvcc -= instructions.size();
   ctx.VALU_wrexec -= instructions.size();
   ctx.VALU_wrsgpr -= instructions.size();
   block->instructions = std::move(instructions);
}

} /* end namespace */


void insert_NOPs(Program* program)
{
   NOP_ctx ctx(program);
   for (std::unique_ptr<Block>& block : program->blocks) {
      if (block->instructions.empty())
         continue;

      handle_block(ctx, block);
   }
}

}
