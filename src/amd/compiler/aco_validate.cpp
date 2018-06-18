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
void validate(Program* program)
{
   for (auto&& block : program->blocks)
   {
      for (auto&& instr : block->instructions)
      {
         /* check num literals */
         if (instr->isSALU() || instr->isVALU()) {
            unsigned num_literals = 0;
            for (unsigned i = 0; i < instr->num_operands; i++)
            {
               if (instr->getOperand(i).isLiteral()) {
                  assert(instr->format == Format::SOP1 ||
                         instr->format == Format::SOP2 ||
                         instr->format == Format::SOPC ||
                         instr->format == Format::VOP1 ||
                         instr->format == Format::VOP2 ||
                         instr->format == Format::VOPC);
                  num_literals++;
                  assert(!instr->isVALU() || i == 0 || i == 2);
               }
            }
            assert(num_literals <= 1);

            /* check num sgprs for VALU */
            if (instr->isVALU()) {
               assert(instr->getDefinition(0).getTemp().type() == vgpr || (int) instr->format & (int) Format::VOPC);
               unsigned num_sgpr = 0;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == sgpr)
                     num_sgpr++;

                  if (instr->getOperand(i).isConstant() && !instr->getOperand(i).isLiteral())
                     assert(i == 0 || (int) instr->format & (int) Format::VOP3A);
               }
               assert(num_sgpr + num_literals <= 1);
            }

            if (instr->format == Format::SOP1 || instr->format == Format::SOP2) {
               assert(instr->getDefinition(0).getTemp().type() == sgpr);
               for (unsigned i = 0; i < instr->num_operands; i++)
                 assert(instr->getOperand(i).isConstant() || instr->getOperand(i).getTemp().type() <= sgpr);
            }
         }

         switch (instr->format) {
         case Format::PSEUDO: {
            if (instr->opcode == aco_opcode::p_create_vector) {
               unsigned size = 0;
               for (unsigned i = 0; i < instr->num_operands; i++)
                  size += instr->getOperand(i).size();
               assert(size == instr->getDefinition(0).size());
               if (instr->getDefinition(0).getTemp().type() == sgpr)
                  for (unsigned i = 0; i < instr->num_operands; i++)
                     assert(instr->getOperand(i).isConstant() || instr->getOperand(i).getTemp().type() == sgpr);
            } else if (instr->opcode == aco_opcode::p_extract_vector) {
               assert(instr->getOperand(0).isTemp() && instr->getOperand(1).isConstant());
               assert(instr->getOperand(1).constantValue() <= instr->getOperand(0).size());
               assert(instr->getDefinition(0).size() == 1);
               assert(instr->getDefinition(0).getTemp().type() == vgpr || instr->getOperand(0).getTemp().type() == sgpr);
            } else if (instr->opcode == aco_opcode::p_parallelcopy) {
               assert(instr->num_definitions == instr->num_operands);
               for (unsigned i = 0; i < instr->num_operands; i++)
                  assert(instr->getDefinition(i).getTemp().type() == instr->getOperand(i).getTemp().type());
            }
            break;
         }
         case Format::EXP: {
            for (unsigned i = 0; i < 4; i++)
               assert(!instr->getOperand(i).isTemp() || instr->getOperand(i).getTemp().type() == vgpr);
            break;
         }
         default:
            break;
         }
      }
   }
}
}
