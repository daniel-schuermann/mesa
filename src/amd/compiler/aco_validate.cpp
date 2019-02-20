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

#include "aco_ir.h"

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
               check(instr->getDefinition(0).size() == 1, "Definition size must be 1", instr.get());
               check(instr->getDefinition(0).getTemp().type() == vgpr || instr->getOperand(0).getTemp().type() == sgpr,
                     "Cannot extract SGPR value from VGPR vector", instr.get());
            } else if (instr->opcode == aco_opcode::p_parallelcopy) {
               check(instr->num_definitions == instr->num_operands, "Number of Operands does not match number of Definitions", instr.get());
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp())
                     check(instr->getDefinition(i).getTemp().type() == instr->getOperand(i).getTemp().type(),
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
         default:
            break;
         }
      }
   }
   assert(is_valid);
}
}
