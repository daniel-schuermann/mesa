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

#include <stdarg.h>
#include <map>

namespace aco {

void validate(Program* program, FILE * output)
{
   bool is_valid = true;
   auto check = [&output, &is_valid](bool check, const char * msg, aco::Instruction * instr) -> void {
      if (!check) {
         fprintf(output, "%s: ", msg);
         aco_print_instr(instr, output);
         fprintf(output, "\n");
         is_valid = false;
      }
   };

   for (auto&& block : program->blocks)
   {
      for (auto&& instr : block->instructions)
      {
         opcode_info op_info = ::opcode_infos[(int)instr->opcode];

         /* check base format */
         Format base_format = instr->format;
         base_format = (Format)((uint32_t)base_format & ~(uint32_t)Format::SDWA);
         base_format = (Format)((uint32_t)base_format & ~(uint32_t)Format::DPP);
         if ((uint32_t)base_format & (uint32_t)Format::VOP1)
            base_format = Format::VOP1;
         else if ((uint32_t)base_format & (uint32_t)Format::VOP2)
            base_format = Format::VOP2;
         else if ((uint32_t)base_format & (uint32_t)Format::VOPC)
            base_format = Format::VOPC;
         else if ((uint32_t)base_format & (uint32_t)Format::VINTRP)
            base_format = Format::VINTRP;
         check(base_format == op_info.format, "Wrong base format for instruction", instr.get());

         /* check VOP3 modifiers */
         if (((uint32_t)instr->format & (uint32_t)Format::VOP3) && instr->format != Format::VOP3) {
            check(base_format == Format::VOP2 ||
                  base_format == Format::VOP1 ||
                  base_format == Format::VOPC ||
                  base_format == Format::VINTRP,
                  "Format cannot have VOP3A/VOP3B applied", instr.get());
         }

         /* check num literals */
         if (instr->isSALU() || instr->isVALU()) {
            unsigned num_literals = 0;
            for (unsigned i = 0; i < instr->num_operands; i++)
            {
               if (instr->getOperand(i).isLiteral()) {
                  check(instr->format == Format::SOP1 ||
                        instr->format == Format::SOP2 ||
                        instr->format == Format::SOPC ||
                        instr->format == Format::VOP1 ||
                        instr->format == Format::VOP2 ||
                        instr->format == Format::VOPC,
                        "Literal applied on wrong instruction format", instr.get());

                  num_literals++;
                  check(!instr->isVALU() || i == 0 || i == 2, "Wrong source position for Literal argument", instr.get());
               }
            }
            check(num_literals <= 1, "Only 1 Literal allowed", instr.get());

            /* check num sgprs for VALU */
            if (instr->isVALU()) {
               check(instr->getDefinition(0).getTemp().type() == vgpr ||
                     (int) instr->format & (int) Format::VOPC ||
                     instr->opcode == aco_opcode::v_readfirstlane_b32 ||
                     instr->opcode == aco_opcode::v_readlane_b32,
                     "Wrong Definition type for VALU instruction", instr.get());
               unsigned num_sgpr = 0;
               unsigned sgpr_idx = instr->num_operands;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == sgpr) {
                     if (sgpr_idx == instr->num_operands || instr->getOperand(sgpr_idx).tempId() != instr->getOperand(i).tempId())
                        num_sgpr++;
                     sgpr_idx = i;
                  }

                  if (instr->getOperand(i).isConstant() && !instr->getOperand(i).isLiteral())
                     check(i == 0 || (int) instr->format & (int) Format::VOP3A, "Wrong source position for SGPR argument", instr.get());
               }
               check(num_sgpr + num_literals <= 1, "Only 1 Literal OR 1 SGPR allowed", instr.get());
            }

            if (instr->format == Format::SOP1 || instr->format == Format::SOP2) {
               check(instr->getDefinition(0).getTemp().type() == sgpr, "Wrong Definition type for SALU instruction", instr.get());
               for (unsigned i = 0; i < instr->num_operands; i++)
                 check(instr->getOperand(i).isConstant() || instr->getOperand(i).isUndefined() || instr->getOperand(i).getTemp().type() <= sgpr,
                       "Wrong Operand type for SALU instruction", instr.get());
            }
         }

         switch (instr->format) {
         case Format::PSEUDO: {
            if (instr->opcode == aco_opcode::p_create_vector) {
               unsigned size = 0;
               for (unsigned i = 0; i < instr->num_operands; i++)
                  size += instr->getOperand(i).size();
               check(size == instr->getDefinition(0).size(), "Definition size does not match operand sizes", instr.get());
               if (instr->getDefinition(0).getTemp().type() == sgpr)
                  for (unsigned i = 0; i < instr->num_operands; i++)
                     check(instr->getOperand(i).isConstant() || instr->getOperand(i).isUndefined() || instr->getOperand(i).getTemp().type() == sgpr,
                           "Wrong Operand type for scalar vector", instr.get());
            } else if (instr->opcode == aco_opcode::p_extract_vector) {
               check(instr->getOperand(0).isTemp() && instr->getOperand(1).isConstant(), "Wrong Operand types", instr.get());
               check(instr->getOperand(1).constantValue() <= instr->getOperand(0).size(), "Index out of range", instr.get());
               check(instr->getDefinition(0).getTemp().type() == vgpr || instr->getOperand(0).getTemp().type() == sgpr,
                     "Cannot extract SGPR value from VGPR vector", instr.get());
            } else if (instr->opcode == aco_opcode::p_parallelcopy) {
               check(instr->num_definitions == instr->num_operands, "Number of Operands does not match number of Definitions", instr.get());
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp())
                     check((instr->getDefinition(i).getTemp().type() == instr->getOperand(i).getTemp().type()) ||
                           (instr->getDefinition(i).getTemp().type() == vgpr && instr->getOperand(i).getTemp().type() == sgpr),
                           "Operand and Definition types do not match", instr.get());
               }
            } else if (instr->opcode == aco_opcode::p_phi) {
               check(instr->num_operands == block->logical_predecessors.size(), "Number of Operands does not match number of predecessors", instr.get());
               check(instr->getDefinition(0).getTemp().type() == vgpr || instr->getDefinition(0).getTemp().regClass() == s2, "Logical Phi Definition must be vgpr or divergent boolean", instr.get());
            } else if (instr->opcode == aco_opcode::p_linear_phi) {
               check(instr->num_operands == block->linear_predecessors.size(), "Number of Operands does not match number of predecessors", instr.get());
            }
            break;
         }
         case Format::SMEM: {
            check(instr->getOperand(0).isTemp() && instr->getOperand(0).getTemp().type() == sgpr, "SMEM operands must be sgpr", instr.get());
            check(instr->getOperand(1).isConstant() || (instr->getOperand(1).isTemp() && instr->getOperand(1).getTemp().type() == sgpr),
                  "SMEM offset must be constant or sgpr", instr.get());
            if (instr->num_definitions)
               check(instr->getDefinition(0).getTemp().type() == sgpr, "SMEM result must be sgpr", instr.get());
            break;
         }
         case Format::MTBUF:
         case Format::MUBUF:
         case Format::MIMG: {
            check(instr->num_operands > 1, "VMEM instructions must have at least one operand", instr.get());
            check(instr->getOperand(0).isUndefined() || (instr->getOperand(0).isTemp() && instr->getOperand(0).getTemp().type() == vgpr),
                  "VADDR must be in vgpr for VMEM instructions", instr.get());
            check(instr->getOperand(1).isTemp() && instr->getOperand(1).getTemp().type() == sgpr, "VMEM resource constant must be sgpr", instr.get());
            check(instr->num_operands < 4 || (instr->getOperand(3).isTemp() && instr->getOperand(3).getTemp().type() == vgpr), "VMEM write data must be vgpr", instr.get());
            break;
         }
         case Format::DS: {
            for (unsigned i = 0; i < instr->num_operands; i++)
               check((instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == vgpr) || instr->getOperand(i).physReg() == m0,
                     "Only VGPRs are valid DS instruction operands", instr.get());
            if (instr->num_definitions)
               check(instr->getDefinition(0).getTemp().type() == vgpr, "DS instruction must return VGPR", instr.get());
            break;
         }
         case Format::EXP: {
            for (unsigned i = 0; i < 4; i++)
               check((!instr->getOperand(i).isConstant() && !instr->getOperand(i).isTemp()) || instr->getOperand(i).getTemp().type() == vgpr,
                     "Only VGPRs are valid Export arguments", instr.get());
            break;
         }
         case Format::FLAT:
            check(instr->getOperand(1).isUndefined(), "Flat instructions don't support SADDR", instr.get());
            /* fallthrough */
         case Format::GLOBAL:
         case Format::SCRATCH: {
            check(instr->getOperand(0).isTemp() && instr->getOperand(0).getTemp().type() == vgpr, "FLAT/GLOBAL/SCRATCH address must be vgpr", instr.get());
            check(instr->getOperand(1).isUndefined() || (instr->getOperand(1).isTemp() && instr->getOperand(1).getTemp().type() == sgpr),
                  "FLAT/GLOBAL/SCRATCH sgpr address must be undefined or sgpr", instr.get());
            if (instr->num_definitions)
               check(instr->getDefinition(0).getTemp().type() == vgpr, "FLAT/GLOBAL/SCRATCH result must be vgpr", instr.get());
            else
               check(instr->getOperand(2).getTemp().type() == vgpr, "FLAT/GLOBAL/SCRATCH data must be vgpr", instr.get());
            break;
         }
         default:
            break;
         }
      }
   }
   assert(is_valid);
}

/* RA validation */
namespace {

struct Location {
   Location() : block(NULL), instr(NULL) {}

   Block *block;
   Instruction *instr; //NULL if it's the block's live-in
};

struct Assignment {
   Location defloc;
   Location firstloc;
   PhysReg reg;
};

bool ra_fail(FILE *output, Location loc, Location loc2, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   char msg[1024];
   vsprintf(msg, fmt, args);
   va_end(args);

   fprintf(stderr, "RA error found at instruction in BB%d:\n", loc.block->index);
   aco_print_instr(loc.instr, stderr);
   fprintf(stderr, "\n%s", msg);
   if (loc2.block) {
      fprintf(stderr, " in BB%d:\n", loc2.block->index);
      aco_print_instr(loc2.instr, stderr);
   }
   fprintf(stderr, "\n\n");

   return true;
}

} /* end namespace */

bool validate_ra(Program *program, const struct radv_nir_compiler_options *options, FILE *output) {
   bool err = false;
   aco::live live_vars = aco::live_var_analysis<true>(program, options);

   std::map<unsigned, Assignment> assignments;
   for (auto& block : program->blocks) {
      Location loc;
      loc.block = block.get();
      for (auto& instr : block->instructions) {
         loc.instr = instr.get();
         for (unsigned i = 0; i < instr->num_operands; i++) {
            Operand& op = instr->getOperand(i);
            if (!op.isTemp())
               continue;
            if (!op.isFixed())
               err |= ra_fail(output, loc, Location(), "Operand %d is not assigned a register", i);
            if (assignments.count(op.tempId()) && assignments[op.tempId()].reg.reg != op.physReg().reg)
               err |= ra_fail(output, loc, assignments.at(op.tempId()).firstloc, "Operand %d has an inconsistent register assignment with instruction", i);
            if (!assignments[op.tempId()].firstloc.block)
               assignments[op.tempId()].firstloc = loc;
            if (!assignments[op.tempId()].defloc.block)
               assignments[op.tempId()].reg.reg = op.physReg().reg;
         }

         for (unsigned i = 0; i < instr->num_definitions; i++) {
            Definition& def = instr->getDefinition(i);
            if (!def.isTemp())
               continue;
            if (!def.isFixed())
               err |= ra_fail(output, loc, Location(), "Definition %d is not assigned a register", i);
            if (assignments[def.tempId()].defloc.block)
               err |= ra_fail(output, loc, assignments.at(def.tempId()).defloc, "Temporary %%%d also defined by instruction", def.tempId());
            if (!assignments[def.tempId()].firstloc.block)
               assignments[def.tempId()].firstloc = loc;
            assignments[def.tempId()].defloc = loc;
            assignments[def.tempId()].reg.reg = def.physReg().reg;
         }
      }
   }

   for (auto& block : program->blocks) {
      Location loc;
      loc.block = block.get();

      std::array<unsigned, 512> regs;
      regs.fill(0);

      std::set<Temp> live;
      live.insert(live_vars.live_out[block->index].begin(), live_vars.live_out[block->index].end());

      for (auto it = block->instructions.rbegin(); it != block->instructions.rend(); ++it) {
         aco_ptr<Instruction>& instr = *it;

         for (unsigned i = 0; i < instr->num_definitions; i++) {
            Definition& def = instr->getDefinition(i);
            if (!def.isTemp())
               continue;
            live.erase(def.getTemp());
         }

         /* don't count phi operands as live-in, since they are actually
          * killed when they are copied at the predecessor */
         if (instr->opcode != aco_opcode::p_phi && instr->opcode != aco_opcode::p_linear_phi) {
            for (unsigned i = 0; i < instr->num_operands; i++) {
               Operand& op = instr->getOperand(i);
               if (!op.isTemp())
                  continue;
               live.insert(op.getTemp());
            }
         }
      }

      for (Temp tmp : live) {
         PhysReg reg = assignments.at(tmp.id()).reg;
         for (unsigned i = 0; i < tmp.size(); i++)
            regs[reg.reg + i] = tmp.id();
      }

      for (auto& instr : block->instructions) {
         loc.instr = instr.get();
         if (instr->opcode != aco_opcode::p_phi && instr->opcode != aco_opcode::p_linear_phi) {
            for (unsigned i = 0; i < instr->num_operands; i++) {
               Operand& op = instr->getOperand(i);
               if (!op.isTemp())
                  continue;
               if (op.isFirstKill()) {
                  for (unsigned j = 0; j < op.getTemp().size(); j++)
                     regs[op.physReg().reg + j] = 0;
               }
            }
         }

         for (unsigned i = 0; i < instr->num_definitions; i++) {
            Definition& def = instr->getDefinition(i);
            if (!def.isTemp())
               continue;
            Temp tmp = def.getTemp();
            PhysReg reg = assignments.at(tmp.id()).reg;
            for (unsigned i = 0; i < tmp.size(); i++) {
               if (regs[reg.reg + i])
                  err |= ra_fail(output, loc, assignments.at(regs[reg.reg + i]).defloc, "Assignment of element %d of %%%d already taken by %%%d from instruction", i, tmp.id(), regs[reg.reg + i]);
               regs[reg.reg + i] = tmp.id();
            }
         }

         for (unsigned i = 0; i < instr->num_definitions; i++) {
            Definition& def = instr->getDefinition(i);
            if (!def.isTemp())
               continue;
            if (def.isKill()) {
               for (unsigned j = 0; j < def.getTemp().size(); j++)
                  regs[def.physReg().reg + j] = 0;
            }
         }
      }
   }

   return err;
}
}
