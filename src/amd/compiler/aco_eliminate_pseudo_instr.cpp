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
#include <algorithm>

#include "aco_ir.h"


namespace aco {


struct copy_operand {
   Operand op;
   Definition def;

   bool operator <(const struct copy_operand& other) const
   {
      return other.def.physReg().reg == op.physReg().reg;
   }
};

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
            std::vector<copy_operand> operands;
            for (unsigned i = 0; i < instr->num_operands; i++)
            {
               for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
               {
                  Operand op = instr->getOperand(i);
                  op.setFixed(PhysReg{op.physReg().reg + j});
                  Definition def = instr->getDefinition(i);
                  def.setFixed(PhysReg{def.physReg().reg + j});
                  operands.emplace_back(copy_operand{op, def});
               }
            }
            std::sort(operands.begin(), operands.end());
            for (unsigned i = 0; i < operands.size(); i++)//copy_operand cp : operands)
            {
               copy_operand cp = operands[i];
               if (cp.def.physReg().reg == cp.op.physReg().reg)
                  continue;
               if (i < operands.size() - 1 && (cp.def.physReg().reg == operands[i+1].op.physReg().reg))
               {
                  if (cp.def.getTemp().type() == RegType::sgpr)
                  {
                     mov.reset(create_instruction<SOP2_instruction>(aco_opcode::s_xor_b32, Format::SOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = Definition(cp.op.physReg(), cp.op.regClass());
                     new_instructions.emplace_back(std::move(mov));
                     mov.reset(create_instruction<SOP2_instruction>(aco_opcode::s_xor_b32, Format::SOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = cp.def;
                     new_instructions.emplace_back(std::move(mov));
                     mov.reset(create_instruction<SOP2_instruction>(aco_opcode::s_xor_b32, Format::SOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = Definition(cp.op.physReg(), cp.op.regClass());
                     new_instructions.emplace_back(std::move(mov));
                  } else {
                     mov.reset(create_instruction<VOP2_instruction>(aco_opcode::v_xor_b32, Format::VOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = Definition(cp.op.physReg(), cp.op.regClass());
                     new_instructions.emplace_back(std::move(mov));
                     mov.reset(create_instruction<VOP2_instruction>(aco_opcode::v_xor_b32, Format::VOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = cp.def;
                     new_instructions.emplace_back(std::move(mov));
                     mov.reset(create_instruction<VOP2_instruction>(aco_opcode::v_xor_b32, Format::VOP2, 2, 1));
                     mov->getOperand(0) = cp.op;
                     mov->getOperand(1) = Operand(cp.def.physReg(), cp.def.regClass());
                     mov->getDefinition(0) = Definition(cp.op.physReg(), cp.op.regClass());
                     new_instructions.emplace_back(std::move(mov));
                  }
                  i++;
               } else {
                  if (cp.def.getTemp().type() == RegType::sgpr)
                  {
                     mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
                  } else {
                     mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
                  }
                  mov->getOperand(0) = cp.op;
                  mov->getDefinition(0) = cp.def;
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
