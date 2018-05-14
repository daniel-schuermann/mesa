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

#include "aco_ir.h"


namespace aco {


void eliminate_pseudo_instr(Program* program)
{
   for (auto&& block : program->blocks)
   {
      std::vector<std::unique_ptr<Instruction>> new_instructions;
      for (auto&& instr : block->instructions)
      {
         if (instr->format != Format::PSEUDO)
         {
            new_instructions.emplace_back(std::move(instr));
            continue;
         }
         std::unique_ptr<Instruction> mov;
         switch (instr->opcode)
         {
         case aco_opcode::p_extract_vector:
            if (instr->getDefinition(0).getTemp().type() == RegType::sgpr)
            {
               assert(instr->getOperand(0).getTemp().type() == RegType::sgpr);
               mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
            } else {
               mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
            }
               mov->getOperand(0) = instr->getOperand(0);
               mov->getOperand(0).setFixed(PhysReg{mov->getOperand(0).physReg().reg + instr->getOperand(1).constantValue()});
               mov->getDefinition(0) = instr->getDefinition(0);
               if (mov->getOperand(0).physReg().reg != mov->getDefinition(0).physReg().reg)
                  new_instructions.emplace_back(std::move(mov));
            break;

         case aco_opcode::p_create_vector:
         {
            if (instr->getDefinition(0).getTemp().type() == RegType::sgpr)
            {
               // TODO
               assert(instr->getOperand(0).getTemp().type() == RegType::sgpr);
               new_instructions.emplace_back(std::move(instr));
               break;
            }

            unsigned def_reg = instr->getDefinition(0).physReg().reg;
            for (unsigned i = 0; i < instr->num_operands; i++)
            {
               /* check if we need to swap: If any other operand has our target reg, we swap */
               for (unsigned j = i + 1; j < instr->num_operands; j++)
               {
                  if (instr->getDefinition(0).physReg().reg + i == instr->getOperand(j).physReg().reg)
                  {
                     mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_swap_b32, Format::VOP1, 2, 2));
                     mov->getOperand(0) = instr->getOperand(i);
                     mov->getOperand(1) = instr->getOperand(j);
                     mov->getDefinition(0) = instr->getDefinition(0);
                     mov->getDefinition(0).setFixed(PhysReg{def_reg + j});
                     mov->getDefinition(1) = instr->getDefinition(0);
                     mov->getDefinition(1).setFixed(PhysReg{def_reg + i});
                     instr->getOperand(j) = instr->getOperand(i);
                     break;
                  }
               }
               if (!mov)
               {
                  mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
                  mov->getOperand(0) = instr->getOperand(i);
                  mov->getDefinition(0) = instr->getDefinition(0);
                  mov->getDefinition(0).setFixed(PhysReg{def_reg + i});
               }
               if (mov->getOperand(0).physReg().reg != mov->getDefinition(0).physReg().reg)
                  new_instructions.emplace_back(std::move(mov));
            }
            break;
         }
         case aco_opcode::p_parallelcopy:
         {
            for (unsigned i = 0; i < instr->num_operands; i++)
            {
               mov.release();
               if (instr->getDefinition(i).getTemp().type() == RegType::sgpr)
               {
                  mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
                  mov->getOperand(0) = instr->getOperand(i);
                  mov->getDefinition(0) = instr->getDefinition(i);
                  if (mov->getOperand(0).physReg().reg != mov->getDefinition(0).physReg().reg)
                     new_instructions.emplace_back(std::move(mov));
               } else {
                  /* check if we need to swap */
                  for (unsigned j = i + 1; j < instr->num_operands; j++)
                  {
                     if (instr->getDefinition(i).physReg().reg == instr->getOperand(j).physReg().reg)
                     {
                        mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_swap_b32, Format::VOP1, 2, 2));
                        mov->getOperand(0) = instr->getOperand(i);
                        mov->getOperand(1) = instr->getOperand(j);
                        mov->getDefinition(0) = instr->getDefinition(j);
                        mov->getDefinition(0).setFixed(instr->getOperand(i).physReg());
                        mov->getDefinition(1) = instr->getDefinition(i);
                        instr->getOperand(j).setFixed(instr->getOperand(i).physReg());
                        break;
                     }
                  }
                  if (!mov)
                  {
                     mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
                     mov->getOperand(0) = instr->getOperand(i);
                     mov->getDefinition(0) = instr->getDefinition(i);
                  }
                  if (mov->getOperand(0).physReg().reg != mov->getDefinition(0).physReg().reg)
                     new_instructions.emplace_back(std::move(mov));
               }

            }
            break;
         }
         default:
            new_instructions.emplace_back(std::move(instr));
            break;
         }


      }
      block->instructions.swap(new_instructions);
   }
}

}
