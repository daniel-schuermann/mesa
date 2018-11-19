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

#include <map>

#include "aco_ir.h"


namespace aco {


struct copy_operation {
   Operand op;
   Definition def;
   unsigned uses;
};

void handle_operands(std::map<PhysReg, copy_operation>& copy_map, std::vector<aco_ptr<Instruction>>& new_instructions, chip_class chip_class)
{
   aco_ptr<Instruction> mov;
   std::map<PhysReg, copy_operation>::iterator it = copy_map.begin();
   std::map<PhysReg, copy_operation>::iterator target;

   /* count the number of uses for each dst reg */
   while (it != copy_map.end()) {
      if (it->second.op.isConstant()) {
         ++it;
         continue;
      }
      /* if src and dst reg are the same, remove operation */
      if (it->first == it->second.op.physReg()) {
         it = copy_map.erase(it);
         continue;
      }
      /* check if the operand reg may be overwritten by another copy operation */
      target = copy_map.find(it->second.op.physReg());
      if (target != copy_map.end()) {
         target->second.uses++;
      }
      ++it;
   }

   /* first, handle paths in the location transfer graph */
   it = copy_map.begin();
   while (it != copy_map.end()) {
      if (it->second.uses == 0) {
         /* the target reg is not used as operand for any other copy */
         if (it->second.def.getTemp().type() == RegType::sgpr)
            mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
         else
            mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
         mov->getOperand(0) = it->second.op;
         mov->getDefinition(0) = it->second.def;
         new_instructions.emplace_back(std::move(mov));

         /* reduce the number of uses of the operand reg by one */
         if (it->second.op.isFixed()) {
            target = copy_map.find(it->second.op.physReg());
            if (target != copy_map.end())
               target->second.uses--;
         }

         copy_map.erase(it);
         it = copy_map.begin();
         continue;
      } else {
         /* the target reg is used as operand, check the next entry */
         ++it;
      }
   }

   if (copy_map.empty())
      return;

   /* all target regs are needed as operand somewhere which means, all entries are part of a cycle */
   for (it = copy_map.begin(); it != copy_map.end(); ++it) {
      assert(it->second.op.isFixed());
      if (it->first == it->second.op.physReg())
         continue;

      /* to resolve the cycle, we have to swap the src reg with the dst reg */
      copy_operation swap = it->second;
      assert(swap.op.regClass() == swap.def.regClass());
      Operand def_as_op = Operand(swap.def.physReg(), swap.def.regClass());
      Definition op_as_def = Definition(swap.op.physReg(), swap.op.regClass());
      if (chip_class >= GFX9 && swap.def.getTemp().type() == RegType::vgpr) {
         mov.reset(create_instruction<Instruction>(aco_opcode::v_swap_b32, Format::VOP1, 2, 2));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = swap.def;
         mov->getDefinition(1) = op_as_def;
         new_instructions.emplace_back(std::move(mov));
      } else {
         aco_opcode opcode = swap.def.getTemp().type() == RegType::sgpr ? aco_opcode::s_xor_b32 : aco_opcode::v_xor_b32;
         Format format = swap.def.getTemp().type() == RegType::sgpr ? Format::SOP2 : Format::VOP2;
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = op_as_def;
         new_instructions.emplace_back(std::move(mov));
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = swap.def;
         new_instructions.emplace_back(std::move(mov));
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = op_as_def;
         new_instructions.emplace_back(std::move(mov));
      }

      /* change the operand reg of the target's use */
      assert(swap.uses == 1);
      target = it;
      for (++target; target != copy_map.end(); ++target) {
         if (target->second.op.physReg() == it->first) {
            target->second.op.setFixed(swap.op.physReg());
            break;
         }
      }
   }
}

void lower_to_hw_instr(Program* program)
{
   //for (auto&& block : program->blocks)
   for (std::vector<std::unique_ptr<Block>>::reverse_iterator it = program->blocks.rbegin(); it != program->blocks.rend(); ++it)
   {
      Block* block = it->get();
      std::vector<aco_ptr<Instruction>> new_instructions;
      for (auto&& instr : block->instructions)
      {
         aco_ptr<Instruction> mov;
         if (instr->format == Format::PSEUDO) {
            switch (instr->opcode)
            {
            case aco_opcode::p_extract_vector:
            {
               unsigned reg = instr->getOperand(0).physReg().reg + instr->getOperand(1).constantValue();
               RegClass rc = getRegClass(instr->getOperand(0).getTemp().type(), 1);
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
               std::map<PhysReg, copy_operation> copy_operations;
               RegClass rc_def = getRegClass(instr->getDefinition(0).getTemp().type(), 1);
               unsigned reg_idx = 0;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  if (instr->getOperand(i).isConstant()) {
                     PhysReg reg = {instr->getDefinition(0).physReg().reg + reg_idx};
                     Definition def = Definition(reg, rc_def);
                     copy_operations[reg] = {instr->getOperand(i), def, 0};
                     reg_idx++;
                     continue;
                  }

                  RegClass rc_op = getRegClass(instr->getOperand(i).getTemp().type(), 1);
                  for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  {
                     Operand op = Operand(PhysReg{instr->getOperand(i).physReg().reg + j}, rc_op);
                     Definition def = Definition(PhysReg{instr->getDefinition(0).physReg().reg + reg_idx}, rc_def);
                     copy_operations[def.physReg()] = {op, def, 0};
                     reg_idx++;
                  }
               }
               handle_operands(copy_operations, new_instructions, program->chip_class);
               break;
            }
            case aco_opcode::p_split_vector:
            {
               std::map<PhysReg, copy_operation> copy_operations;
               RegClass rc_op = instr->getOperand(0).isConstant() ? s1 : getRegClass(typeOf(instr->getOperand(0).regClass()), 1);
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  unsigned k = instr->getDefinition(i).size();
                  RegClass rc_def = getRegClass(instr->getDefinition(i).getTemp().type(), 1);
                  for (unsigned j = 0; j < k; j++) {
                     Operand op = Operand(PhysReg{instr->getOperand(0).physReg().reg + (i*k+j)}, rc_op);
                     Definition def = Definition(PhysReg{instr->getDefinition(i).physReg().reg + j}, rc_def);
                     copy_operations[def.physReg()] = {op, def, 0};
                  }
               }
               handle_operands(copy_operations, new_instructions, program->chip_class);
               break;
            }
            case aco_opcode::p_parallelcopy:
            {
               std::map<PhysReg, copy_operation> copy_operations;
               for (unsigned i = 0; i < instr->num_operands; i++)
               {
                  Operand operand = instr->getOperand(i);
                  if (operand.isConstant() || operand.size() == 1) {
                     assert(instr->getDefinition(i).size() == 1);
                     copy_operations[instr->getDefinition(i).physReg()] = {operand, instr->getDefinition(i), 0};
                  } else {
                     RegClass def_rc = getRegClass(typeOf(instr->getDefinition(i).regClass()), 1);
                     RegClass op_rc = getRegClass(operand.getTemp().type(), 1);
                     for (unsigned j = 0; j < operand.size(); j++)
                     {
                        Operand op = Operand({instr->getOperand(i).physReg().reg + j}, op_rc);
                        Definition def = Definition(PhysReg{instr->getDefinition(i).physReg().reg + j}, def_rc);
                        copy_operations[def.physReg()] = {op, def, 0};
                     }
                  }
               }
               handle_operands(copy_operations, new_instructions, program->chip_class);
               break;
            }
            case aco_opcode::p_discard_if:
            {
               assert(instr->getOperand(0).regClass() == s2);
               aco_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2)};
               sop2->getOperand(0) = Operand(exec, s2);
               sop2->getOperand(1) = instr->getOperand(0);
               sop2->getDefinition(0) = Definition(exec, s2);
               sop2->getDefinition(1) = Definition(PhysReg{253}, b);
               new_instructions.emplace_back(std::move(sop2));

               aco_ptr<SOPP_instruction> branch{create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc1, Format::SOPP, 1, 0)};
               branch->getOperand(0) = Operand(exec, s2);
               branch->imm = 3; /* (8 + 4 dwords) / 4 */
               new_instructions.emplace_back(std::move(branch));

               aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
               for (unsigned i = 0; i < 4; i++)
                  exp->getOperand(i) = Operand();
               exp->enabled_mask = 0;
               exp->compressed = false;
               exp->done = true;
               exp->valid_mask = true;
               exp->dest = 9; /* NULL */
               new_instructions.emplace_back(std::move(exp));

               aco_ptr<SOPP_instruction> endpgm{create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)};
               new_instructions.emplace_back(std::move(endpgm));
               break;
            }
            default:
               new_instructions.emplace_back(std::move(instr));
               break;
            }
         } else if (instr->format == Format::PSEUDO_BRANCH) {
            Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(instr.get());
            /* check if all blocks from current to target are empty */
            bool can_remove = block->index < branch->targets[0]->index;
            for (unsigned i = block->index + 1; i < branch->targets[0]->index; i++) {
               if (!program->blocks[i]->instructions.empty()) {
                  can_remove = false;
                  break;
               }
            }
            if (can_remove)
               continue;

            aco_ptr<SOPP_instruction> sopp;
            switch (instr->opcode) {
               case aco_opcode::p_branch:
                  sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_branch, Format::SOPP, 0, 0));
                  break;
               case aco_opcode::p_cbranch_nz:
                  if (branch->getOperand(0).physReg() == exec)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_execnz, Format::SOPP, 0, 0));
                  else if (branch->getOperand(0).physReg() == vcc)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_vccnz, Format::SOPP, 0, 0));
                  else {
                     assert(branch->getOperand(0).physReg() == PhysReg{253});
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc1, Format::SOPP, 0, 0));
                  }
                  break;
               case aco_opcode::p_cbranch_z:
                  if (branch->getOperand(0).physReg() == exec)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_execz, Format::SOPP, 0, 0));
                  else if (branch->getOperand(0).physReg() == vcc)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_vccz, Format::SOPP, 0, 0));
                  else {
                     assert(branch->getOperand(0).physReg() == PhysReg{253});
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc0, Format::SOPP, 0, 0));
                  }
                  break;
               default:
                  unreachable("Unknown Pseudo branch instruction!");
            }
            sopp->block = branch->targets[0];
            new_instructions.emplace_back(std::move(sopp));

         // FIXME: do this while RA?
         } else if (instr->format == Format::VOPC) {
            /* check if the register allocator was able to assign vcc */
            if (!(instr->getDefinition(0).physReg() == vcc)) {
               /* check if the first operand was a literal */
               if (instr->getOperand(0).physReg().reg == 255) {
                  aco_ptr<SOP1_instruction> mov{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1)};
                  mov->getOperand(0) = instr->getOperand(0);
                  mov->getDefinition(0) = Definition(instr->getDefinition(0).physReg(), s1);
                  instr->getOperand(0) = Operand(instr->getDefinition(0).physReg(), s1);
                  new_instructions.emplace_back(std::move(mov));
               }

               /* change the instruction to VOP3 to enable an arbitrary register pair as dst */
               aco_ptr<Instruction> tmp = std::move(instr);
               Format format = (Format) ((int) tmp->format | (int) Format::VOP3A);
               instr.reset(create_instruction<VOP3A_instruction>(tmp->opcode, format, tmp->num_operands, tmp->num_definitions));
               for (unsigned i = 0; i < instr->num_operands; i++)
                  instr->getOperand(i) = tmp->getOperand(i);
               for (unsigned i = 0; i < instr->num_definitions; i++)
                  instr->getDefinition(i) = tmp->getDefinition(i);
            }
            new_instructions.emplace_back(std::move(instr));
         } else if (instr->format == Format::VOP2) {
            // TODO: what about literals?!
            if (instr->num_operands == 3 && !(instr->getOperand(2).physReg() == vcc || instr->getOperand(2).physReg().reg == 255 || instr->opcode == aco_opcode::v_mac_f32)) {
               /* change the instruction to VOP3 to enable an arbitrary register pair as dst */
               aco_ptr<Instruction> tmp = std::move(instr);
               Format format = (Format) ((int) tmp->format | (int) Format::VOP3A);
               instr.reset(create_instruction<VOP3A_instruction>(tmp->opcode, format, tmp->num_operands, tmp->num_definitions));
               for (unsigned i = 0; i < instr->num_operands; i++)
                  instr->getOperand(i) = tmp->getOperand(i);
               for (unsigned i = 0; i < instr->num_definitions; i++)
                  instr->getDefinition(i) = tmp->getDefinition(i);
            }

            new_instructions.emplace_back(std::move(instr));
         } else {
            new_instructions.emplace_back(std::move(instr));
         }

      }
      block->instructions.swap(new_instructions);
   }
}

}
