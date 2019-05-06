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
#include <math.h>

#include "aco_ir.h"
#include "aco_builder.h"
#include "util/u_math.h"


namespace aco {

struct lower_context {
   Program *program;
   std::vector<aco_ptr<Instruction>> instructions;
};

void emit_dpp_op(lower_context *ctx, PhysReg dst, PhysReg src0, PhysReg src1, PhysReg vtmp, PhysReg wrtmp,
                 aco_opcode op, Format format, bool clobber_vcc, unsigned dpp_ctrl,
                 unsigned row_mask, unsigned bank_mask, bool bound_ctrl_zero,
                 Operand identity=Operand()) /* for VOP3 with sparse writes */
{
   if (format == Format::VOP3) {
      Builder bld(ctx->program, &ctx->instructions);

      if (!identity.isUndefined())
         bld.vop1(aco_opcode::v_mov_b32, Definition(vtmp, v1), identity);

      bld.vop1_dpp(aco_opcode::v_mov_b32, Definition(vtmp, v1), Operand(src0, v1),
                   dpp_ctrl, row_mask, bank_mask, bound_ctrl_zero);

      if (clobber_vcc)
         bld.vop3(op, Definition(dst, v1), Definition(vcc, s2), Operand(vtmp, v1), Operand(src1, v1));
      else
         bld.vop3(op, Definition(dst, v1), Operand(vtmp, v1), Operand(src1, v1));
   } else {
      assert(format == Format::VOP2 || format == Format::VOP1);
      aco_ptr<DPP_instruction> dpp{create_instruction<DPP_instruction>(
         op, (Format) ((uint32_t) format | (uint32_t) Format::DPP),
         format == Format::VOP2 ? 2 : 1, clobber_vcc ? 2 : 1)};
      dpp->getOperand(0) = Operand(src0, v1);
      if (format == Format::VOP2)
         dpp->getOperand(1) = Operand(src1, v1);
      dpp->getDefinition(0) = Definition(dst, v1);
      if (clobber_vcc)
         dpp->getDefinition(1) = Definition(vcc, s2);
      dpp->dpp_ctrl = dpp_ctrl;
      dpp->row_mask = row_mask;
      dpp->bank_mask = bank_mask;
      dpp->bound_ctrl = bound_ctrl_zero;
      ctx->instructions.emplace_back(std::move(dpp));
   }
}

uint32_t get_reduction_identity(ReduceOp op)
{
   switch (op) {
   case iadd32:
   case iadd64:
   case fadd32:
   case fadd64:
   case ior32:
   case ior64:
   case ixor32:
   case ixor64:
   case umax32:
   case umax64:
      return 0;
   case imul32:
   case imul64:
      return 1;
   case fmul32:
   case fmul64:
      return 0x3f800000u; /* 1.0 */
   case imin32:
   case imin64:
      return INT32_MAX;
   case imax32:
   case imax64:
      return INT32_MIN;
   case umin32:
   case umin64:
   case iand32:
   case iand64:
      return UINT32_MAX;
   case fmin32:
   case fmin64:
      return 0x7f800000u; /* infinity */
   case fmax32:
   case fmax64:
      return 0xff800000u; /* negative infinity */
   }
   unreachable("Invalid reduction operation");
}

aco_opcode get_reduction_opcode(lower_context *ctx, ReduceOp op, bool *clobber_vcc, Format *format)
{
   *clobber_vcc = false;
   *format = Format::VOP2;
   switch (op) {
   case iadd32:
      *clobber_vcc = ctx->program->chip_class < GFX9;
      return ctx->program->chip_class < GFX9 ? aco_opcode::v_add_co_u32 : aco_opcode::v_add_u32;
   case imul32:
      *format = Format::VOP3;
      return aco_opcode::v_mul_lo_u32;
   case fadd32:
      return aco_opcode::v_add_f32;
   case fmul32:
      return aco_opcode::v_mul_f32;
   case imax32:
      return aco_opcode::v_max_i32;
   case imin32:
      return aco_opcode::v_min_i32;
   case umin32:
      return aco_opcode::v_min_u32;
   case umax32:
      return aco_opcode::v_max_u32;
   case fmin32:
      return aco_opcode::v_min_f32;
   case fmax32:
      return aco_opcode::v_max_f32;
   case iand32:
      return aco_opcode::v_and_b32;
   case ixor32:
      return aco_opcode::v_xor_b32;
   case ior32:
      return aco_opcode::v_or_b32;
   case iadd64:
   case imul64:
   case fadd64:
   case fmul64:
   case imin64:
   case imax64:
   case umin64:
   case umax64:
   case fmin64:
   case fmax64:
   case iand64:
   case ior64:
   case ixor64:
      assert(false);
      break;
   }
   unreachable("Invalid reduction operation");
   return aco_opcode::v_min_u32;
}

void emit_vopn(lower_context *ctx, PhysReg dst, PhysReg src0, PhysReg src1,
               aco_opcode op, Format format, bool clobber_vcc)
{
   aco_ptr<Instruction> instr;
   switch (format) {
   case Format::VOP2:
      instr.reset(create_instruction<VOP2_instruction>(op, format, 2, clobber_vcc ? 2 : 1));
      break;
   case Format::VOP3:
      instr.reset(create_instruction<VOP3A_instruction>(op, format, 2, clobber_vcc ? 2 : 1));
      break;
   default:
      assert(false);
   }
   instr->getOperand(0) = Operand(src0, v1);
   instr->getOperand(1) = Operand(src1, v1);
   instr->getDefinition(0) = Definition(dst, v1);
   if (clobber_vcc)
      instr->getDefinition(1) = Definition(vcc, s2);
   ctx->instructions.emplace_back(std::move(instr));
}

void emit_reduction(lower_context *ctx, aco_opcode op, ReduceOp reduce_op, unsigned cluster_size, PhysReg tmp,
                    PhysReg stmp, PhysReg vtmp, PhysReg sitmp, Operand src, Definition dst)
{
   assert(cluster_size == 64 || op == aco_opcode::p_reduce);

   Builder bld(ctx->program, &ctx->instructions);

   PhysReg wrtmp{0}; /* should never be needed */

   Format format;
   bool should_clobber_vcc;
   aco_opcode reduce_opcode = get_reduction_opcode(ctx, reduce_op, &should_clobber_vcc, &format);
   Operand identity = Operand(get_reduction_identity(reduce_op));
   Operand vcndmask_identity = identity;

   /* First, copy the source to tmp and set inactive lanes to the identity */
   // note: this clobbers SCC!
   bld.sop1(aco_opcode::s_or_saveexec_b64, Definition(stmp, s2), Definition(scc, s1), Definition(exec, s2), Operand(UINT64_MAX), Operand(exec, s2));

   /* p_exclusive_scan needs it to be a sgpr or inline constant for the v_writelane_b32 */
   if (identity.isLiteral() && op == aco_opcode::p_exclusive_scan) {
      bld.sop1(aco_opcode::s_mov_b32, Definition(sitmp, s1), identity);
      identity = Operand(sitmp, s1);

      bld.vop1(aco_opcode::v_mov_b32, Definition(PhysReg{tmp.reg + src.size() - 1}, v1), identity);
      vcndmask_identity = Operand(PhysReg{tmp.reg + src.size() - 1}, v1);
   } else if (identity.isLiteral()) {
      bld.vop1(aco_opcode::v_mov_b32, Definition(PhysReg{tmp.reg + src.size() - 1}, v1), identity);
      vcndmask_identity = Operand(PhysReg{tmp.reg + src.size() - 1}, v1);
   }

   for (unsigned k = 0; k < src.size(); k++) {
      bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(PhysReg{tmp.reg + k}, v1),
                   vcndmask_identity, Operand(PhysReg{src.physReg().reg + k}, v1),
                   Operand(stmp, s2));
   }

   bool exec_restored = false;
   bool dst_written = false;
   switch (op) {
   case aco_opcode::p_reduce:
      if (cluster_size == 1) break;
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_quad_perm(1, 0, 3, 2), 0xf, 0xf, false);
      if (cluster_size == 2) break;
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_quad_perm(2, 3, 0, 1), 0xf, 0xf, false);
      if (cluster_size == 4) break;
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_half_mirror, 0xf, 0xf, false);
      if (cluster_size == 8) break;
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_mirror, 0xf, 0xf, false);
      if (cluster_size == 16) break;
      if (cluster_size == 32) {
         bld.ds(aco_opcode::ds_swizzle_b32, Definition(vtmp, v1), Operand(tmp, s1), ds_pattern_bitmode(0x1f, 0, 0x10));
         bld.sop1(aco_opcode::s_mov_b64, Definition(exec, s2), Operand(stmp, s2));
         exec_restored = true;
         emit_vopn(ctx, dst.physReg(), vtmp, tmp, reduce_opcode, format, should_clobber_vcc);
         dst_written = true;
      } else {
         assert(cluster_size == 64);
         emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                     dpp_row_bcast15, 0xa, 0xf, false);
         emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                     dpp_row_bcast31, 0xc, 0xf, false);
      }
      break;
   case aco_opcode::p_exclusive_scan:
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, aco_opcode::v_mov_b32, Format::VOP1, false,
                  dpp_wf_sr1, 0xf, 0xf, true);
      if (!identity.isConstant() || identity.constantValue()) { /* bound_ctrl should take case of this overwise */
         assert((identity.isConstant() && !identity.isLiteral()) || identity.physReg() == sitmp);
         bld.vop3(aco_opcode::v_writelane_b32, Definition(tmp, v1),
                  identity, Operand(0u));
      }
      /* fall through */
   case aco_opcode::p_inclusive_scan:
      assert(cluster_size == 64);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_sr(1), 0xf, 0xf, false, identity);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_sr(2), 0xf, 0xf, false, identity);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_sr(4), 0xf, 0xf, false, identity);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_sr(8), 0xf, 0xf, false, identity);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_bcast15, 0xa, 0xf, false, identity);
      emit_dpp_op(ctx, tmp, tmp, tmp, vtmp, wrtmp, reduce_opcode, format, should_clobber_vcc,
                  dpp_row_bcast31, 0xc, 0xf, false, identity);
      break;
   default:
      unreachable("Invalid reduction mode");
   }

   if (!exec_restored)
      bld.sop1(aco_opcode::s_mov_b64, Definition(exec, s2), Operand(stmp, s2));

   if (op == aco_opcode::p_reduce && cluster_size == 64) {
      for (unsigned k = 0; k < src.size(); k++) {
         bld.vop3(aco_opcode::v_readlane_b32, Definition(PhysReg{dst.physReg().reg + k}, s1),
                  Operand(PhysReg{tmp.reg + k}, v1), Operand(63u));
      }
   } else if (!(dst.physReg() == tmp) && !dst_written) {
      for (unsigned k = 0; k < src.size(); k++) {
         bld.vop1(aco_opcode::v_mov_b32, Definition(PhysReg{dst.physReg().reg + k}, s1),
                  Operand(PhysReg{tmp.reg + k}, v1));
      }
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

   /* first, handle paths in the location transfer graph */
   it = copy_map.begin();
   while (it != copy_map.end()) {

      /* the target reg is not used as operand for any other copy */
      if (it->second.uses == 0) {

         /* try to coalesce 32-bit sgpr copies to 64-bit copies */
         if (it->second.def.getTemp().type() == RegType::sgpr && it->second.size == 1 &&
             !it->second.op.isConstant() && it->first.reg % 2 == it->second.op.physReg().reg % 2) {

            PhysReg other_def_reg = PhysReg{it->first.reg % 2 ? it->first.reg - 1 : it->first.reg + 1};
            PhysReg other_op_reg = PhysReg{it->first.reg % 2 ? it->second.op.physReg().reg - 1 : it->second.op.physReg().reg + 1};
            std::map<PhysReg, copy_operation>::iterator other = copy_map.find(other_def_reg);

            if (other != copy_map.end() && !other->second.uses && other->second.size == 1 &&
                other->second.op.physReg() == other_op_reg && !other->second.op.isConstant()) {
               std::map<PhysReg, copy_operation>::iterator to_erase = it->first.reg % 2 ? it : other;
               it = it->first.reg % 2 ? other : it;
               copy_map.erase(to_erase);
               it->second.size = 2;
            }
         }

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
         if (!it->second.op.isConstant()) {
            for (unsigned i = 0; i < it->second.size; i++) {
               target = copy_map.find(PhysReg{it->second.op.physReg().reg + i});
               if (target != copy_map.end())
                  target->second.uses--;
            }
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
      ctx.program = program;
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
                     copy_operations[reg] = {instr->getOperand(i), def, 0, 1};
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
                     copy_operations[instr->getDefinition(i).physReg()] = {operand, instr->getDefinition(i), 0, 1};
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
               // TODO: optimize uniform conditions
               Definition branch_cond = instr->getDefinition(instr->num_definitions - 1);
               Operand discard_cond = instr->getOperand(instr->num_operands - 1);
               aco_ptr<Instruction> sop2;
               /* backwards, to finally branch on the global exec mask */
               for (int i = instr->num_operands - 2; i >= 0; i--) {
                  sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2));
                  sop2->getOperand(0) = instr->getOperand(i); /* old mask */
                  sop2->getOperand(1) = discard_cond;
                  sop2->getDefinition(0) = instr->getDefinition(i); /* new mask */
                  sop2->getDefinition(1) = branch_cond; /* scc */
                  ctx.instructions.emplace_back(std::move(sop2));
               }

               aco_ptr<SOPP_instruction> branch{create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_scc1, Format::SOPP, 1, 0)};
               branch->getOperand(0) = Operand(branch_cond.getTemp());
               branch->getOperand(0).setFixed(scc);
               branch->imm = program->wb_smem_l1_on_end ? 5 : 3; /* (8 + (wb_smem ? 8 : 0) + 4 dwords) / 4 */
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

               if (program->wb_smem_l1_on_end) {
                  aco_ptr<SMEM_instruction> smem{create_instruction<SMEM_instruction>(aco_opcode::s_dcache_wb, Format::SMEM, 0, 0)};
                  ctx.instructions.emplace_back(std::move(smem));
               }

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
            for (unsigned i = block->index + 1; can_remove && i < branch->targets[0]->index; i++) {
               if (program->blocks[i]->instructions.size())
                  can_remove = false;
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
            emit_reduction(&ctx, reduce->opcode, reduce->reduce_op, reduce->cluster_size,
                           reduce->getOperand(1).physReg(), // tmp
                           reduce->getDefinition(1).physReg(), // stmp
                           reduce->getOperand(2).physReg(), // vtmp
                           reduce->getDefinition(2).physReg(), // sitmp
                           reduce->getOperand(0), reduce->getDefinition(0));
         } else {
            ctx.instructions.emplace_back(std::move(instr));
         }

      }
      block->instructions.swap(ctx.instructions);
   }
}

}
