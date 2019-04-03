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

struct lower_context {
   std::vector<aco_ptr<Instruction>> instructions;
};

enum dpp_ctrl {
	_dpp_quad_perm = 0x000,
	_dpp_row_sl = 0x100,
	_dpp_row_sr = 0x110,
	_dpp_row_rr = 0x120,
	dpp_wf_sl1 = 0x130,
	dpp_wf_rl1 = 0x134,
	dpp_wf_sr1 = 0x138,
	dpp_wf_rr1 = 0x13C,
	dpp_row_mirror = 0x140,
	dpp_row_half_mirror = 0x141,
	dpp_row_bcast15 = 0x142,
	dpp_row_bcast31 = 0x143
};

static inline dpp_ctrl
dpp_quad_perm(unsigned lane0, unsigned lane1, unsigned lane2, unsigned lane3)
{
	assert(lane0 < 4 && lane1 < 4 && lane2 < 4 && lane3 < 4);
	return (dpp_ctrl)(lane0 | (lane1 << 2) | (lane2 << 4) | (lane3 << 6));
}

static inline dpp_ctrl
dpp_row_sl(unsigned amount)
{
	assert(amount > 0 && amount < 16);
	return (dpp_ctrl)(((unsigned) _dpp_row_sl) | amount);
}

static inline dpp_ctrl
dpp_row_sr(unsigned amount)
{
	assert(amount > 0 && amount < 16);
	return (dpp_ctrl)(((unsigned) _dpp_row_sr) | amount);
}

void emit_dpp_op(lower_context *ctx, PhysReg dst, PhysReg src0, PhysReg src1,
                 aco_opcode op, unsigned dpp_ctrl, unsigned row_mask, unsigned bank_mask,
                 bool bound_ctrl_zero)
{
   Format format = (Format) ((uint32_t) Format::VOP2 | (uint32_t) Format::DPP);
   aco_ptr<DPP_instruction> dpp{create_instruction<DPP_instruction>(op, format, 3, 1)};
   dpp->getOperand(0) = Operand(src0, v1);
   dpp->getOperand(1) = Operand(src1, v1);
   dpp->getDefinition(0) = Definition(dst, v1);
   dpp->dpp_ctrl = dpp_ctrl;
   dpp->row_mask = row_mask;
   dpp->bank_mask = bank_mask;
   dpp->bound_ctrl = bound_ctrl_zero;
   ctx->instructions.emplace_back(std::move(dpp));
}

void invert_exec(lower_context *ctx)
{
   aco_ptr<Instruction> inv{create_instruction<SOP2_instruction>(aco_opcode::s_xor_b64, Format::SOP2, 2, 1)};

   inv->getOperand(0) = Operand(exec, s2);
   inv->getOperand(1) = Operand((uint64_t)-1);
   inv->getDefinition(0) = Definition(exec, s2);
   ctx->instructions.emplace_back(std::move(inv));
}

void emit_reduce(lower_context *ctx, aco_opcode op, ReduceOp reduce_op, unsigned cluster_size, PhysReg tmp,
                 PhysReg stmp, Operand src, Definition dst)
{
   assert(reduce_op == ReduceOp::umin32);
   assert(op == aco_opcode::p_reduce);
   assert(cluster_size == 0);

   /* First, copy the source to tmp and set inactive lanes to the identity */
   for (unsigned k = 0; k < src.size(); k++) {
      aco_ptr<Instruction> mov{create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1)};
      mov->getOperand(0) = Operand(PhysReg{src.physReg().reg + k}, v1);
      mov->getDefinition(0) = Definition(PhysReg{tmp.reg + k}, v1);
      ctx->instructions.emplace_back(std::move(mov));
   }

   invert_exec(ctx);

   for (unsigned k = 0; k < src.size(); k++) {
      aco_ptr<Instruction> mov{create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1)};
      mov->getOperand(0) = Operand((uint32_t) -1); // TODO get identity for op!
      mov->getDefinition(0) = Definition(PhysReg{tmp.reg + k}, v1);
      ctx->instructions.emplace_back(std::move(mov));
   }

   invert_exec(ctx);

   // note: this clobbers SCC!
   aco_ptr<SOP1_instruction> set_exec{create_instruction<SOP1_instruction>(aco_opcode::s_or_saveexec_b64, Format::SOP1, 1, 3)};
   set_exec->getOperand(0) = Operand((uint64_t) -1);
   set_exec->getDefinition(0) = Definition(stmp, s2);
   set_exec->getDefinition(1) = Definition(scc, b);
   set_exec->getDefinition(2) = Definition(exec, s2);
   ctx->instructions.emplace_back(std::move(set_exec));

   // TODO: generalize this!
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_quad_perm(1, 0, 3, 2), 0xf, 0xf, false);
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_quad_perm(2, 3, 0, 1), 0xf, 0xf, false);
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_row_half_mirror, 0xf, 0xf, false);
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_row_mirror, 0xf, 0xf, false);
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_row_bcast15, 0xf, 0xf, false);
   emit_dpp_op(ctx, tmp, tmp, tmp, aco_opcode::v_min_u32,
               dpp_row_bcast31, 0xf, 0xf, false);

   aco_ptr<Instruction> restore{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1)};
   restore->getOperand(0) = Operand(stmp, s2);
   restore->getDefinition(0) = Definition(exec, s2);
   ctx->instructions.emplace_back(std::move(restore));

   for (unsigned k = 0; k < src.size(); k++) {
      aco_ptr<VOP3A_instruction> readlane{create_instruction<VOP3A_instruction>(aco_opcode::v_readlane_b32, Format::VOP3A, 2, 1)};
      readlane->getOperand(0) = Operand(PhysReg{tmp.reg + k}, v1);
      readlane->getOperand(1) = Operand((uint32_t)63);
      readlane->getDefinition(0) = Definition(PhysReg{dst.physReg().reg + k}, s1);
      ctx->instructions.emplace_back(std::move(readlane));
   }
}

struct copy_operation {
   Operand op;
   Definition def;
   unsigned uses;
   unsigned size;
};

void handle_operands(std::map<PhysReg, copy_operation>& copy_map, lower_context* ctx, chip_class chip_class)
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

   /* coalesce 32-bit sgpr copies to 64-bit copies */
   it = copy_map.begin();
   while (it != copy_map.end()) {
      if (it->second.def.getTemp().type() != RegType::sgpr || !it->first.reg ||
          it->second.uses || it->second.size != 1 || it->second.op.isConstant() ||
          it->first.reg % 2 != 1 || it->second.op.physReg().reg % 2 != 1) {
         ++it;
         continue;
      }

      std::map<PhysReg, copy_operation>::iterator second = copy_map.find(PhysReg{it->first.reg - 1});
      if (second == copy_map.end() || second->second.uses || second->second.size != 1 ||
          second->second.op.physReg().reg + 1 != it->second.op.physReg().reg ||
          second->second.op.isConstant()) {
         ++it;
         continue;
      }

      second->second.size = 2;
      it = copy_map.erase(it);
   }

   /* first, handle paths in the location transfer graph */
   it = copy_map.begin();
   while (it != copy_map.end()) {
      if (it->second.uses == 0) {
         /* the target reg is not used as operand for any other copy */
         if (it->second.def.physReg().reg == scc.reg) {
            mov.reset(create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lg_i32, Format::SOPC, 2, 1));
            mov->getOperand(1) = Operand((uint32_t) 0);
            mov->getOperand(0) = it->second.op;
            mov->getDefinition(0) = it->second.def;
            ctx->instructions.emplace_back(std::move(mov));
         } else if (it->second.size == 2 && it->second.def.getTemp().type() == RegType::sgpr) {
            mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
            mov->getOperand(0) = it->second.op;
            mov->getDefinition(0) = it->second.def;
            ctx->instructions.emplace_back(std::move(mov));
         } else if (it->second.def.getTemp().type() == RegType::sgpr) {
            ctx->instructions.emplace_back(std::move(create_s_mov(it->second.def, it->second.op)));
         } else {
            mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
            mov->getOperand(0) = it->second.op;
            mov->getDefinition(0) = it->second.def;
            ctx->instructions.emplace_back(std::move(mov));
         }

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
   bool constants = false;
   for (it = copy_map.begin(); it != copy_map.end(); ++it) {
      assert(it->second.op.isFixed());
      if (it->first == it->second.op.physReg())
         continue;
      /* do constants later */
      if (it->second.op.isConstant()) {
         constants = true;
         continue;
      }

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
         ctx->instructions.emplace_back(std::move(mov));
      } else {
         aco_opcode opcode = swap.def.getTemp().type() == RegType::sgpr ? aco_opcode::s_xor_b32 : aco_opcode::v_xor_b32;
         Format format = swap.def.getTemp().type() == RegType::sgpr ? Format::SOP2 : Format::VOP2;
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = op_as_def;
         ctx->instructions.emplace_back(std::move(mov));
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = swap.def;
         ctx->instructions.emplace_back(std::move(mov));
         mov.reset(create_instruction<Instruction>(opcode, format, 2, 1));
         mov->getOperand(0) = swap.op;
         mov->getOperand(1) = def_as_op;
         mov->getDefinition(0) = op_as_def;
         ctx->instructions.emplace_back(std::move(mov));
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

   /* copy constants into a registers which were operands */
   if (constants) {
      for (it = copy_map.begin(); it != copy_map.end(); ++it) {
         if (!it->second.op.isConstant())
            continue;
         if (it->second.def.getTemp().type() == RegType::sgpr) {
            ctx->instructions.emplace_back(std::move(create_s_mov(it->second.def, it->second.op)));
         } else {
            mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
            mov->getOperand(0) = it->second.op;
            mov->getDefinition(0) = it->second.def;
            ctx->instructions.emplace_back(std::move(mov));
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
      lower_context ctx;
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
               RegClass rc_def = getRegClass(instr->getDefinition(0).getTemp().type(), 1);
               if (reg == instr->getDefinition(0).physReg().reg)
                  break;

               std::map<PhysReg, copy_operation> copy_operations;
               for (unsigned i = 0; i < instr->getDefinition(0).size(); i++) {
                  Definition def = Definition(PhysReg{instr->getDefinition(0).physReg().reg + i}, rc_def);
                  copy_operations[def.physReg()] = {Operand(PhysReg{reg + i}, rc), def, 0, 1};
               }
               handle_operands(copy_operations, &ctx, program->chip_class);
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
                     copy_operations[def.physReg()] = {op, def, 0, 1};
                     reg_idx++;
                  }
               }
               handle_operands(copy_operations, &ctx, program->chip_class);
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
                     copy_operations[def.physReg()] = {op, def, 0, 1};
                  }
               }
               handle_operands(copy_operations, &ctx, program->chip_class);
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
                        copy_operations[def.physReg()] = {op, def, 0, 1};
                     }
                  }
               }
               handle_operands(copy_operations, &ctx, program->chip_class);
               break;
            }
            case aco_opcode::p_discard_if:
            {
               aco_opcode opcode;
               Temp branch_cond;
               if (instr->getOperand(0).regClass() == s2) {
                  aco_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2)};
                  sop2->getOperand(0) = Operand(exec, s2);
                  sop2->getOperand(1) = instr->getOperand(0);
                  sop2->getDefinition(0) = Definition(exec, s2);
                  branch_cond = {program->allocateId(), s1};
                  sop2->getDefinition(1) = Definition(branch_cond.id(), scc, s1);
                  ctx.instructions.emplace_back(std::move(sop2));
                  opcode = aco_opcode::s_cbranch_scc1;
               } else {
                  assert(instr->getOperand(0).isFixed() && instr->getOperand(0).physReg() == scc);
                  opcode = aco_opcode::s_cbranch_scc0;
                  branch_cond = instr->getOperand(0).getTemp();
               }

               aco_ptr<SOPP_instruction> branch{create_instruction<SOPP_instruction>(opcode, Format::SOPP, 1, 0)};
               branch->getOperand(0) = Operand(branch_cond);
               branch->getOperand(0).setFixed(scc);
               branch->imm = 3; /* (8 + 4 dwords) / 4 */
               ctx.instructions.emplace_back(std::move(branch));

               aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
               for (unsigned i = 0; i < 4; i++)
                  exp->getOperand(i) = Operand();
               exp->enabled_mask = 0;
               exp->compressed = false;
               exp->done = true;
               exp->valid_mask = true;
               exp->dest = 9; /* NULL */
               ctx.instructions.emplace_back(std::move(exp));

               aco_ptr<SOPP_instruction> endpgm{create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)};
               ctx.instructions.emplace_back(std::move(endpgm));
               break;
            }
            case aco_opcode::p_spill:
            {
               assert(instr->getOperand(0).regClass() == v1_linear);
               for (unsigned i = 0; i < instr->getOperand(2).size(); i++) {
                  aco_ptr<VOP3A_instruction> writelane{create_instruction<VOP3A_instruction>(aco_opcode::v_writelane_b32, Format::VOP3A, 2, 1)};
                  writelane->getOperand(0) = Operand(PhysReg{instr->getOperand(2).physReg().reg + i}, s1);
                  writelane->getOperand(1) = Operand(instr->getOperand(1).constantValue() + i);
                  writelane->getDefinition(0) = Definition(instr->getOperand(0).getTemp());
                  writelane->getDefinition(0).setFixed(instr->getOperand(0).physReg());
                  ctx.instructions.emplace_back(std::move(writelane));
               }
               break;
            }
            case aco_opcode::p_reload:
            {
               assert(instr->getOperand(0).regClass() == v1_linear);
               for (unsigned i = 0; i < instr->getDefinition(0).size(); i++) {
                  aco_ptr<VOP3A_instruction> readlane{create_instruction<VOP3A_instruction>(aco_opcode::v_readlane_b32, Format::VOP3A, 2, 1)};
                  readlane->getOperand(0) = instr->getOperand(0);
                  readlane->getOperand(1) = Operand(instr->getOperand(1).constantValue() + i);
                  readlane->getDefinition(0) = Definition(PhysReg{instr->getDefinition(0).physReg().reg + i}, s1);
                  ctx.instructions.emplace_back(std::move(readlane));
               }
               break;
            }
            case aco_opcode::p_wqm:
            {
               assert(instr->getOperand(0).physReg() == instr->getDefinition(0).physReg());
               break;
            }
            case aco_opcode::p_as_uniform:
            {
               assert(typeOf(instr->getOperand(0).regClass()) == RegType::vgpr);
               assert(typeOf(instr->getDefinition(0).regClass()) == RegType::sgpr);
               assert(instr->getOperand(0).size() == instr->getDefinition(0).size());
               for (unsigned i = 0; i < instr->getDefinition(0).size(); i++) {
                  aco_ptr<VOP1_instruction> readfirstlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
                  readfirstlane->getOperand(0) = instr->getOperand(0);
                  readfirstlane->getOperand(0).setFixed(PhysReg{instr->getOperand(0).physReg().reg + i});
                  readfirstlane->getDefinition(0) = instr->getDefinition(0);
                  readfirstlane->getDefinition(0).setFixed(PhysReg{instr->getDefinition(0).physReg().reg + i});
                  ctx.instructions.emplace_back(std::move(readfirstlane));
               }
               break;
            }
            default:
               break;
            }
         } else if (instr->format == Format::PSEUDO_BRANCH) {
            Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(instr.get());
            /* check if all blocks from current to target are empty */
            bool can_remove = block->index < branch->targets[0]->index;
            for (unsigned i = block->index + 1; i < branch->targets[0]->index; i++) {
               if (program->blocks[i]->instructions.size() > 2) {
                  can_remove = false;
                  break;
               }
               for (aco_ptr<Instruction>& instr : program->blocks[i]->instructions) {
                  if (instr->opcode != aco_opcode::p_logical_start &&
                      instr->opcode != aco_opcode::p_logical_end) {
                     can_remove = false;
                     break;
                  }
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
                     assert(branch->getOperand(0).physReg() == scc);
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc1, Format::SOPP, 0, 0));
                  }
                  break;
               case aco_opcode::p_cbranch_z:
                  if (branch->getOperand(0).physReg() == exec)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_execz, Format::SOPP, 0, 0));
                  else if (branch->getOperand(0).physReg() == vcc)
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_vccz, Format::SOPP, 0, 0));
                  else {
                     assert(branch->getOperand(0).physReg() == scc);
                     sopp.reset(create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc0, Format::SOPP, 0, 0));
                  }
                  break;
               default:
                  unreachable("Unknown Pseudo branch instruction!");
            }
            sopp->block = branch->targets[0];
            ctx.instructions.emplace_back(std::move(sopp));

         } else if (instr->format == Format::PSEUDO_REDUCTION) {
            Pseudo_reduction_instruction* reduce = static_cast<Pseudo_reduction_instruction*>(instr.get());
            emit_reduce(&ctx, reduce->opcode, reduce->reduce_op, reduce->cluster_size,
                        reduce->getOperand(1).physReg(), // tmp
                        reduce->getDefinition(1).physReg(), // stmp
                        reduce->getOperand(0), reduce->getDefinition(0));
         } else {
            ctx.instructions.emplace_back(std::move(instr));
         }

      }
      block->instructions.swap(ctx.instructions);
   }
}

}
