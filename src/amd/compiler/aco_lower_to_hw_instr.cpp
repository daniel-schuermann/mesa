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
#include <deque>

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

void insert_sorted(std::deque<copy_operand>& operands, struct copy_operand elem)
{
   for (std::deque<copy_operand>::iterator it = operands.begin(); it != operands.end(); it++)
   {
      if (elem < *it)
      {
         operands.insert(it, elem);
         return;
      }
   }
   operands.emplace_back(elem);
}

void handle_operands(std::deque<copy_operand>& operands, std::vector<std::unique_ptr<Instruction>>& new_instructions)
{
   std::unique_ptr<Instruction> mov;
   for (unsigned i = 0; i < operands.size(); i++)
   {
      copy_operand cp = operands[i];
      if (cp.def.physReg().reg == cp.op.physReg().reg)
         continue;

      for (unsigned j = i + 1; j < operands.size(); j++)
      {
         if (cp.def.physReg().reg == operands[j].op.physReg().reg)
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
            }
            operands[j].op.setFixed(cp.op.physReg());
            break;
         }
      }
      if (!mov)
      {
         if (cp.def.getTemp().type() == RegType::sgpr)
         {
            mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
         } else {
            mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
         }
         mov->getOperand(0) = cp.op;
         mov->getDefinition(0) = cp.def;
      }
      new_instructions.emplace_back(std::move(mov));
   }
}

void lower_to_hw_instr(Program* program)
{
   for (auto&& block : program->blocks)
   {
      std::vector<std::unique_ptr<Instruction>> new_instructions;
      for (auto&& instr : block->instructions)
      {
         std::unique_ptr<Instruction> mov;
         if (instr->format == Format::PSEUDO) {
            if (instr->num_definitions == 0)
               continue;
            switch (instr->opcode)
            {
            case aco_opcode::p_extract_vector:
            {
               unsigned reg = instr->getOperand(0).physReg().reg + instr->getOperand(1).constantValue();
               RegClass rc = instr->getDefinition(0).regClass();
               if (reg == instr->getDefinition(0).physReg().reg)
                  break;

               if (instr->getDefinition(0).getTemp().type() == RegType::sgpr)
               {
                  assert(instr->getOperand(0).getTemp().type() == RegType::sgpr);
                  mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
               } else {
                  mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
               }
               mov->getOperand(0) = Operand(PhysReg{reg}, rc);
               mov->getDefinition(0) = instr->getDefinition(0);
               new_instructions.emplace_back(std::move(mov));
               break;
            }
            case aco_opcode::p_create_vector:
            {
               std::deque<copy_operand> operands;
               RegClass rc_def = (RegClass) (((int) v1 & (int) instr->getDefinition(0).regClass()) | 1);
               unsigned reg_idx = 0;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  if (instr->getOperand(i).isConstant()) {
                     Definition def = Definition(PhysReg{instr->getDefinition(0).physReg().reg + reg_idx}, rc_def);
                     insert_sorted(operands, copy_operand{instr->getOperand(i), def});
                     reg_idx++;
                     continue;
                  }

                  RegClass rc_op = (RegClass) (((int) v1 & (int) instr->getOperand(i).regClass()) | 1);
                  for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  {
                     Operand op = Operand(PhysReg{instr->getOperand(i).physReg().reg + j}, rc_op);
                     Definition def = Definition(PhysReg{instr->getDefinition(0).physReg().reg + reg_idx}, rc_def);
                     insert_sorted(operands, copy_operand{op, def});
                     reg_idx++;
                  }
               }
               handle_operands(operands, new_instructions);
               break;
            }
            case aco_opcode::p_parallelcopy:
            {
               std::deque<copy_operand> operands;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  assert(!instr->getOperand(i).isConstant());
                  RegClass rc = (RegClass) (((int) v1 & (int) instr->getDefinition(i).regClass()) | 1);
                  for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  {
                     Operand op = Operand(PhysReg{instr->getOperand(i).physReg().reg + j}, rc);
                     Definition def = Definition(PhysReg{instr->getDefinition(i).physReg().reg + j}, rc);
                     insert_sorted(operands, copy_operand{op, def});
                  }
               }
               handle_operands(operands, new_instructions);
               break;
            }
            default:
               new_instructions.emplace_back(std::move(instr));
               break;
            }
         } else {
            new_instructions.emplace_back(std::move(instr));
         }

      }
      block->instructions.swap(new_instructions);
   }
}

}
