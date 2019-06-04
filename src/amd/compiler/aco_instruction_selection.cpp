/*
 * Copyright © 2018 Valve Corporation
 * Copyright © 2018 Google
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

#include <algorithm>
#include <map>
#include <set>
#include <stack>

#include "aco_ir.h"
#include "aco_builder.h"
#include "aco_interface.h"
#include "aco_instruction_selection_setup.cpp"

namespace aco {
namespace {

class loop_info_RAII {
   isel_context* ctx;
   Block* entry_old;
   Block* exit_old;
   bool divergent_cont_old;
   bool divergent_branch_old;
   bool divergent_if_old;

public:
   loop_info_RAII(isel_context* ctx, Block* loop_entry, Block* loop_exit)
      : ctx(ctx),
        entry_old(ctx->cf_info.parent_loop.entry), exit_old(ctx->cf_info.parent_loop.exit),
        divergent_cont_old(ctx->cf_info.parent_loop.has_divergent_continue),
        divergent_branch_old(ctx->cf_info.parent_loop.has_divergent_branch),
        divergent_if_old(ctx->cf_info.parent_if.is_divergent)
   {
      ctx->cf_info.parent_loop.entry = loop_entry;
      ctx->cf_info.parent_loop.exit = loop_exit;
      ctx->cf_info.parent_loop.has_divergent_continue = false;
      ctx->cf_info.parent_loop.has_divergent_branch = false;
      ctx->cf_info.parent_if.is_divergent = false;
      ctx->cf_info.loop_nest_depth = ctx->cf_info.loop_nest_depth + 1;
   }

   ~loop_info_RAII()
   {
      ctx->cf_info.parent_loop.entry = entry_old;
      ctx->cf_info.parent_loop.exit = exit_old;
      ctx->cf_info.parent_loop.has_divergent_continue = divergent_cont_old;
      ctx->cf_info.parent_loop.has_divergent_branch = divergent_branch_old;
      ctx->cf_info.parent_if.is_divergent = divergent_if_old;
      ctx->cf_info.loop_nest_depth = ctx->cf_info.loop_nest_depth - 1;
   }
};

class if_info_RAII {
   isel_context* ctx;
   Block* merge_old;
   bool divergent_old;

public:
   if_info_RAII(isel_context* ctx, Block* merge_block)
      : ctx(ctx), merge_old(ctx->cf_info.parent_if.merge_block),
        divergent_old(ctx->cf_info.parent_if.is_divergent)
   {
      ctx->cf_info.parent_if.merge_block = merge_block;
      ctx->cf_info.parent_if.is_divergent = true;
   }

   ~if_info_RAII()
   {
      ctx->cf_info.parent_if.merge_block = merge_old;
      ctx->cf_info.parent_if.is_divergent = divergent_old;
   }
};

static void visit_cf_list(struct isel_context *ctx,
                          struct exec_list *list);

static void add_logical_edge(Block *pred, Block *succ)
{
   pred->logical_successors.push_back(succ);
   succ->logical_predecessors.push_back(pred);
}

static void add_linear_edge(Block *pred, Block *succ)
{
   pred->linear_successors.push_back(succ);
   succ->linear_predecessors.push_back(pred);
}

static void add_edge(Block *pred, Block *succ)
{
   add_logical_edge(pred, succ);
   add_linear_edge(pred, succ);
}

static void append_logical_start(Block *b)
{
   Builder(NULL, b).pseudo(aco_opcode::p_logical_start);
}

static void append_logical_end(Block *b)
{
   Builder(NULL, b).pseudo(aco_opcode::p_logical_end);
}

Temp get_ssa_temp(struct isel_context *ctx, nir_ssa_def *def)
{
   RegClass rc = ctx->reg_class[def->index];
   auto it = ctx->allocated.find(def->index);
   if (it != ctx->allocated.end())
      return Temp{it->second, rc};
   uint32_t id = ctx->program->allocateId();
   ctx->allocated.insert({def->index, id});
   return Temp{id, rc};
}

Temp emit_v_add32(isel_context *ctx, Temp dst, Operand a, Operand b, bool carry_out=false)
{
   if (!b.isTemp() || typeOf(b.regClass()) != RegType::vgpr)
      std::swap(a, b);
   assert(b.isTemp() && typeOf(b.regClass()) == RegType::vgpr); // in case two SGPRs are given

   Builder bld(ctx->program, ctx->block);
   if (ctx->options->chip_class < GFX9 || carry_out) {
      return bld.vop2(aco_opcode::v_add_co_u32, Definition(dst),
                      bld.hint_vcc(bld.def(s2)), a, b).def(1).getTemp();
   } else {
      bld.vop2(aco_opcode::v_add_u32, Definition(dst), a, b);
      return Temp();
   }
}

Temp emit_v_sub32(isel_context *ctx, Temp dst, Operand a, Operand b, bool carry_out = false, Operand borrow = Operand(s2))
{
   if (!borrow.isUndefined() || ctx->options->chip_class < GFX9)
      carry_out = true;

   bool reverse = !b.isTemp() || typeOf(b.regClass()) != RegType::vgpr;
   if (reverse)
      std::swap(a, b);
   assert(b.isTemp() && typeOf(b.regClass()) == RegType::vgpr);

   aco_opcode op;
   Temp carry;
   if (carry_out) {
      carry = {ctx->program->allocateId(), s2};
      if (borrow.isUndefined())
         op = reverse ? aco_opcode::v_subrev_co_u32 : aco_opcode::v_sub_co_u32;
      else
         op = reverse ? aco_opcode::v_subbrev_co_u32 : aco_opcode::v_subb_co_u32;
   } else {
      op = reverse ? aco_opcode::v_subrev_u32 : aco_opcode::v_sub_u32;
   }

   int num_ops = borrow.isUndefined() ? 2 : 3;
   int num_defs = carry_out ? 2 : 1;
   aco_ptr<Instruction> sub{create_instruction<VOP2_instruction>(op, Format::VOP2, num_ops, num_defs)};
   sub->getOperand(0) = Operand(a);
   sub->getOperand(1) = Operand(b);
   if (!borrow.isUndefined())
      sub->getOperand(2) = borrow;
   sub->getDefinition(0) = Definition(dst);
   if (carry_out) {
      sub->getDefinition(1) = Definition(carry);
      sub->getDefinition(1).setHint(vcc);
   }
   ctx->block->instructions.emplace_back(std::move(sub));

   return carry;
}

void emit_v_mov(isel_context *ctx, Temp src, Temp dst)
{
   Builder bld(ctx->program, ctx->block);
   if (dst.size() == 1)
   {
      bld.vop1(aco_opcode::v_mov_b32, Definition(dst), src);
   } else {
      bld.pseudo(aco_opcode::p_create_vector, Definition(dst), src);
   }
}

Temp emit_wqm(isel_context *ctx, Temp src, Temp dst=Temp(0, s1))
{
   Builder bld(ctx->program, ctx->block);

   if (!dst.id())
      dst = bld.tmp(src.regClass());

   if (ctx->stage != MESA_SHADER_FRAGMENT) {
      if (!dst.id())
         return src;

      if (src.type() == vgpr || src.size() > 1)
         emit_v_mov(ctx, src, dst);
      else
         bld.sop1(aco_opcode::s_mov_b32, Definition(dst), src);
      return dst;
   }

   bld.pseudo(aco_opcode::p_wqm, Definition(dst), src);
   ctx->program->needs_wqm = true;
   return dst;
}

Temp as_vgpr(isel_context *ctx, Temp val)
{
   if (val.type() == RegType::sgpr) {
      Temp tmp = {ctx->program->allocateId(), getRegClass(vgpr, val.size())};
      emit_v_mov(ctx, val, tmp);
      return tmp;
   }
   assert(val.type() == RegType::vgpr);
   return val;
}

void emit_extract_vector(isel_context* ctx, Temp src, uint32_t idx, Temp dst)
{
   Builder bld(ctx->program, ctx->block);
   bld.pseudo(aco_opcode::p_extract_vector, Definition(dst), src, Operand(idx));
}


Temp emit_extract_vector(isel_context* ctx, Temp src, uint32_t idx, RegClass dst_rc)
{
   /* no need to extract the whole vector */
   if (src.regClass() == dst_rc) {
      assert(idx == 0);
      return src;
   }
   assert(src.size() > idx);
   auto it = ctx->allocated_vec.find(src.id());
   if (it != ctx->allocated_vec.end()) {
      if (it->second[idx].regClass() == dst_rc) {
         return it->second[idx];
      } else if (sizeOf(dst_rc) == sizeOf(it->second[idx].regClass())) {
         assert(typeOf(dst_rc) == vgpr && it->second[idx].type() == sgpr);
         Temp dst = {ctx->program->allocateId(), dst_rc};
         emit_v_mov(ctx, it->second[idx], dst);
         return dst;
      }
   }

   Temp dst = {ctx->program->allocateId(), dst_rc};
   if (src.size() == sizeOf(dst_rc)) {
      assert(idx == 0);
      emit_v_mov(ctx, src, dst);
   } else {
      emit_extract_vector(ctx, src, idx, dst);
   }
   return dst;
}

void emit_split_vector(isel_context* ctx, Temp vec_src, unsigned num_components)
{
   if (num_components == 1)
      return;
   if (ctx->allocated_vec.find(vec_src.id()) != ctx->allocated_vec.end())
      return;
   aco_ptr<Instruction> split{create_instruction<Instruction>(aco_opcode::p_split_vector, Format::PSEUDO, 1, num_components)};
   split->getOperand(0) = Operand(vec_src);
   std::array<Temp,4> elems;
   for (unsigned i = 0; i < num_components; i++) {
      elems[i] = {ctx->program->allocateId(), getRegClass(vec_src.type(), vec_src.size() / num_components)};
      split->getDefinition(i) = Definition(elems[i]);
   }
   ctx->block->instructions.emplace_back(std::move(split));
   ctx->allocated_vec.emplace(vec_src.id(), elems);
}

/* This vector expansion uses a mask to determine which elements in the new vector
 * come from the original vector. The other elements are undefined. */
void expand_vector(isel_context* ctx, Temp vec_src, Temp dst, unsigned num_components, unsigned mask)
{
   emit_split_vector(ctx, vec_src, util_bitcount(mask));

   if (vec_src == dst)
      return;

   Builder bld(ctx->program, ctx->block);
   if (num_components == 1) {
      if (dst.type() == sgpr)
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), vec_src);
      else
         emit_v_mov(ctx, vec_src, dst);
      return;
   }

   unsigned component_size = dst.size() / num_components;
   std::array<Temp,4> elems;

   aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, num_components, 1)};
   vec->getDefinition(0) = Definition(dst);
   unsigned k = 0;
   for (unsigned i = 0; i < num_components; i++) {
      if (mask & (1 << i)) {
         Temp src = emit_extract_vector(ctx, vec_src, k++, getRegClass(vec_src.type(), component_size));
         if (dst.type() == sgpr)
            src = bld.as_uniform(src);
         vec->getOperand(i) = Operand(src);
      } else {
         vec->getOperand(i) = Operand(getRegClass(dst.type(), 1));
      }
      elems[i] = vec->getOperand(i).getTemp();
   }
   ctx->block->instructions.emplace_back(std::move(vec));
   ctx->allocated_vec.emplace(dst.id(), elems);
}

Temp as_divergent_bool(isel_context *ctx, Temp val, bool vcc_hint)
{
   if (val.regClass() == s2) {
      return val;
   } else {
      assert(val.regClass() == s1);
      Builder bld(ctx->program, ctx->block);
      Definition& def = bld.sop2(aco_opcode::s_cselect_b64, bld.def(s2),
                                 Operand((uint32_t) -1), Operand(0u), bld.scc(val)).def(0);
      if (vcc_hint)
         def.setHint(vcc);
      return def.getTemp();
   }
}

Temp as_uniform_bool(isel_context *ctx, Temp val)
{
   if (val.regClass() == s1) {
      return val;
   } else {
      assert(val.regClass() == s2);
      Builder bld(ctx->program, ctx->block);
      return bld.sopc(aco_opcode::s_cmp_lg_u64, bld.def(s1, scc), Operand(0u), Operand(val));
   }
}

Temp get_alu_src(struct isel_context *ctx, nir_alu_src src)
{
   if (src.src.ssa->num_components == 1 && src.swizzle[0] == 0)
      return get_ssa_temp(ctx, src.src.ssa);

   Temp vec = get_ssa_temp(ctx, src.src.ssa);
   assert(vec.size() % src.src.ssa->num_components == 0);
   return emit_extract_vector(ctx, vec, src.swizzle[0], getRegClass(vec.type(), vec.size() / src.src.ssa->num_components));
}

Temp convert_pointer_to_64_bit(isel_context *ctx, Temp ptr)
{
   if (ptr.size() == 2)
      return ptr;
   Builder bld(ctx->program, ctx->block);
   return bld.pseudo(aco_opcode::p_create_vector, bld.def(s2),
                     ptr, Operand((unsigned)ctx->options->address32_hi));
}

void emit_sop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst, bool writes_scc)
{
   aco_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(op, Format::SOP2, 2, writes_scc ? 2 : 1)};
   sop2->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
   sop2->getOperand(1) = Operand(get_alu_src(ctx, instr->src[1]));
   sop2->getDefinition(0) = Definition(dst);
   if (writes_scc)
      sop2->getDefinition(1) = Definition(ctx->program->allocateId(), scc, s1);
   ctx->block->instructions.emplace_back(std::move(sop2));
}

void emit_vop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst, bool commutative, bool swap_srcs=false)
{
   Builder bld(ctx->program, ctx->block);
   Temp src0 = get_alu_src(ctx, instr->src[swap_srcs ? 1 : 0]);
   Temp src1 = get_alu_src(ctx, instr->src[swap_srcs ? 0 : 1]);
   if (src1.type() == sgpr) {
      if (commutative && src0.type() == vgpr) {
         Temp t = src0;
         src0 = src1;
         src1 = t;
      } else if (src0.type() == vgpr &&
                 op != aco_opcode::v_madmk_f32 &&
                 op != aco_opcode::v_madak_f32 &&
                 op != aco_opcode::v_madmk_f16 &&
                 op != aco_opcode::v_madak_f16) {
         /* If the instruction is not commutative, we emit a VOP3A instruction */
         bld.vop2_e64(op, Definition(dst), src0, src1);
         return;
      } else {
         Temp mov_dst = Temp(ctx->program->allocateId(), getRegClass(vgpr, src1.size()));
         emit_v_mov(ctx, src1, mov_dst);
         src1 = mov_dst;
      }
   }
   bld.vop2(op, Definition(dst), src0, src1);
}

void emit_vop3a_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   Temp src0 = get_alu_src(ctx, instr->src[0]);
   Temp src1 = get_alu_src(ctx, instr->src[1]);
   Temp src2 = get_alu_src(ctx, instr->src[2]);

   /* ensure that the instruction has at most 1 sgpr operand
    * The optimizer will inline constants for us */
   if (src0.type() == sgpr && src1.type() == sgpr)
      src0 = as_vgpr(ctx, src0);
   if (src1.type() == sgpr && src2.type() == sgpr)
      src1 = as_vgpr(ctx, src1);
   if (src2.type() == sgpr && src0.type() == sgpr)
      src2 = as_vgpr(ctx, src2);

   Builder bld(ctx->program, ctx->block);
   bld.vop3(op, Definition(dst), src0, src1, src2);
}

void emit_vop1_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   Builder bld(ctx->program, ctx->block);
   bld.vop1(op, Definition(dst), get_alu_src(ctx, instr->src[0]));
}

void emit_vopc_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   Temp src0 = get_alu_src(ctx, instr->src[0]);
   Temp src1 = get_alu_src(ctx, instr->src[1]);
   aco_ptr<Instruction> vopc;
   if (src1.type() == sgpr) {
      if (src0.type() == vgpr) {
         /* to swap the operands, we might also have to change the opcode */
         switch (op) {
            case aco_opcode::v_cmp_lt_f32:
               op = aco_opcode::v_cmp_gt_f32;
               break;
            case aco_opcode::v_cmp_ge_f32:
               op = aco_opcode::v_cmp_le_f32;
               break;
            case aco_opcode::v_cmp_lt_i32:
               op = aco_opcode::v_cmp_gt_i32;
               break;
            case aco_opcode::v_cmp_ge_i32:
               op = aco_opcode::v_cmp_le_i32;
               break;
            case aco_opcode::v_cmp_lt_u32:
               op = aco_opcode::v_cmp_gt_u32;
               break;
            case aco_opcode::v_cmp_ge_u32:
               op = aco_opcode::v_cmp_le_u32;
               break;
            case aco_opcode::v_cmp_lt_f64:
               op = aco_opcode::v_cmp_gt_f64;
               break;
            case aco_opcode::v_cmp_ge_f64:
               op = aco_opcode::v_cmp_le_f64;
               break;
            case aco_opcode::v_cmp_lt_i64:
               op = aco_opcode::v_cmp_gt_i64;
               break;
            case aco_opcode::v_cmp_ge_i64:
               op = aco_opcode::v_cmp_le_i64;
               break;
            case aco_opcode::v_cmp_lt_u64:
               op = aco_opcode::v_cmp_gt_u64;
               break;
            case aco_opcode::v_cmp_ge_u64:
               op = aco_opcode::v_cmp_le_u64;
               break;
            default: /* eq and ne are commutative */
               break;
         }
         Temp t = src0;
         src0 = src1;
         src1 = t;
      } else {
         src1 = as_vgpr(ctx, src1);
      }
   }
   Builder bld(ctx->program, ctx->block);
   bld.vopc(op, Definition(dst), src0, src1).def(0).setHint(vcc);
}

void emit_comparison(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   if (dst.regClass() == s2) {
      emit_vopc_instruction(ctx, instr, op, dst);
      if (!ctx->divergent_vals[instr->dest.dest.ssa.index])
         emit_split_vector(ctx, dst, 2);
   } else if (dst.regClass() == s1) {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      assert(src0.type() == sgpr && src1.type() == sgpr);

      Builder bld(ctx->program, ctx->block);
      bld.sopc(op, bld.scc(Definition(dst)), src0, src1);

   } else {
      assert(false);
   }
}

void emit_boolean_logic(isel_context *ctx, nir_alu_instr *instr, aco_opcode op32, aco_opcode op64, Temp dst)
{
   Builder bld(ctx->program, ctx->block);
   Temp src0 = get_alu_src(ctx, instr->src[0]);
   Temp src1 = get_alu_src(ctx, instr->src[1]);
   if (dst.regClass() == s2) {
      bld.sop2(op64, Definition(dst), bld.def(s1, scc),
               as_divergent_bool(ctx, src0, false), as_divergent_bool(ctx, src1, false));
   } else {
      assert(dst.regClass() == s1);
      bld.sop2(op32, bld.def(s1), bld.scc(Definition(dst)),
               as_uniform_bool(ctx, src0), as_uniform_bool(ctx, src1));
   }
}


void emit_bcsel(isel_context *ctx, nir_alu_instr *instr, Temp dst)
{
   Builder bld(ctx->program, ctx->block);
   Temp cond = get_alu_src(ctx, instr->src[0]);
   Temp then = get_alu_src(ctx, instr->src[1]);
   Temp els = get_alu_src(ctx, instr->src[2]);

   if (dst.type() == vgpr) {
      cond = as_divergent_bool(ctx, cond, true);

      aco_ptr<Instruction> bcsel;
      if (dst.size() == 1) {
         then = as_vgpr(ctx, then);
         els = as_vgpr(ctx, els);

         bld.vop2(aco_opcode::v_cndmask_b32, Definition(dst), els, then, cond);
      } else if (dst.size() == 2) {
         emit_split_vector(ctx, then, 2);
         emit_split_vector(ctx, els, 2);

         Temp dst0 = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                              emit_extract_vector(ctx, els, 0, v1),
                              emit_extract_vector(ctx, then, 0, v1),
                              cond);

         Temp dst1 = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                              emit_extract_vector(ctx, els, 1, v1),
                              emit_extract_vector(ctx, then, 1, v1),
                              cond);

         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), dst0, dst1);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      return;
   }

   if (instr->dest.dest.ssa.bit_size != 1) { /* uniform condition and values in sgpr */
      if (dst.regClass() == s1 || dst.regClass() == s2) {
         assert((then.regClass() == s1 || then.regClass() == s2) && els.regClass() == then.regClass());
         aco_opcode op = dst.regClass() == s1 ? aco_opcode::s_cselect_b32 : aco_opcode::s_cselect_b64;
         bld.sop2(op, Definition(dst), then, els, bld.scc(as_uniform_bool(ctx, cond)));
      } else {
         fprintf(stderr, "Unimplemented uniform bcsel bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      return;
   }

   /* boolean bcsel */
   assert(instr->dest.dest.ssa.bit_size == 1);

   if (dst.regClass() == s1)
      cond = as_uniform_bool(ctx, cond);

   if (cond.regClass() == s1) { /* uniform selection */
      aco_opcode op;
      if (dst.regClass() == s2) {
         op = aco_opcode::s_cselect_b64;
         then = as_divergent_bool(ctx, then, false);
         els = as_divergent_bool(ctx, els, false);
      } else {
         assert(dst.regClass() == s1);
         op = aco_opcode::s_cselect_b32;
         then = as_uniform_bool(ctx, then);
         els = as_uniform_bool(ctx, els);
      }
      bld.sop2(op, Definition(dst), then, els, bld.scc(cond));
      return;
   }

   /* divergent boolean bcsel
    * this implements bcsel on bools: dst = s0 ? s1 : s2
    * are going to be: dst = (s0 & s1) | (~s0 & s2) */
   assert (dst.regClass() == s2);
   then = as_divergent_bool(ctx, then, false);
   els = as_divergent_bool(ctx, els, false);

   if (cond.id() != then.id())
      then = bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), cond, then);

   if (cond.id() == els.id())
      bld.sop1(aco_opcode::s_mov_b64, Definition(dst), then);
   else
      bld.sop2(aco_opcode::s_or_b64, Definition(dst), bld.def(s1, scc), then,
               bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.def(s1, scc), els, cond));
}

void visit_alu_instr(isel_context *ctx, nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa) {
      fprintf(stderr, "nir alu dst not in ssa: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }
   Builder bld(ctx->program, ctx->block);
   Temp dst = get_ssa_temp(ctx, &instr->dest.dest.ssa);
   switch(instr->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      std::array<Temp,4> elems;
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.dest.ssa.num_components, 1)};
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; ++i) {
         elems[i] = get_alu_src(ctx, instr->src[i]);
         vec->getOperand(i) = Operand{elems[i]};
      }
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
      ctx->allocated_vec.emplace(dst.id(), elems);
      break;
   }
   case nir_op_imov: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      aco_ptr<Instruction> mov;
      if (dst.regClass() == s1) {
         if (src.regClass() == v1)
            bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), src);
         else if (src.regClass() == s1)
            bld.sop1(aco_opcode::s_mov_b32, Definition(dst), src);
         else
            unreachable("wrong src register class for nir_op_imov");
      } else if (dst.regClass() == s2) {
         bld.sop1(aco_opcode::s_mov_b64, Definition(dst), src);
      } else if (dst.regClass() == v1) {
         bld.vop1(aco_opcode::v_mov_b32, Definition(dst), src);
      } else if (dst.regClass() == v2) {
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), src);
      } else {
         nir_print_instr(&instr->instr, stderr);
         unreachable("Should have been lowered to scalar.");
      }
      break;
   }
   case nir_op_fmov: {
      if (dst.regClass() == s1) {
         bld.sop1(aco_opcode::s_mov_b32, Definition(dst), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == v1) {
         bld.vop1(aco_opcode::v_mov_b32, Definition(dst), get_alu_src(ctx, instr->src[0]));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_inot: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      /* uniform booleans */
      if (instr->dest.dest.ssa.bit_size == 1 && dst.regClass() == s1) {
         if (src.regClass() == s1) {
            /* in this case, src is either 1 or 0 */
            bld.sop2(aco_opcode::s_xor_b32, bld.def(s1), bld.scc(Definition(dst)), Operand(1u), src);
         } else {
            /* src is either exec_mask or 0 */
            assert(src.regClass() == s2);
            bld.sopc(aco_opcode::s_cmp_eq_u64, bld.scc(Definition(dst)), Operand(0u), src);
         }
      } else if (dst.regClass() == v1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_not_b32, dst);
      } else if (dst.type() == sgpr) {
         aco_opcode opcode = dst.size() == 1 ? aco_opcode::s_not_b32 : aco_opcode::s_not_b64;
         bld.sop1(opcode, Definition(dst), bld.def(s1, scc), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ineg: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == v1) {
         emit_v_sub32(ctx, dst, Operand((uint32_t) 0), Operand(src));
      } else if (dst.regClass() == v2) {
         emit_split_vector(ctx, src, 2);
         Temp lower = bld.tmp(v1);
         Temp borrow = emit_v_sub32(ctx, lower, Operand((uint32_t) 0), Operand(emit_extract_vector(ctx, src, 0, v1)), true);
         Temp upper = bld.tmp(v1);
         emit_v_sub32(ctx, upper, Operand((uint32_t) 0), Operand(emit_extract_vector(ctx, src, 1, v1)), false, Operand(borrow));
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), lower, upper);
      } else if (dst.regClass() == s1) {
         bld.sop2(aco_opcode::s_mul_i32, Definition(dst), Operand((uint32_t) -1), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_iabs: {
      if (dst.regClass() == s1) {
         bld.sop1(aco_opcode::s_abs_i32, Definition(dst), bld.def(s1, scc), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == v1) {
         Temp tmp = bld.tmp(v1);
         Temp src = get_alu_src(ctx, instr->src[0]);
         emit_v_sub32(ctx, tmp, Operand(0u), Operand(src));
         bld.vop2(aco_opcode::v_max_i32, Definition(dst), src, tmp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_isign: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s1) {
         Temp tmp = bld.sop2(aco_opcode::s_ashr_i32, bld.def(s1), bld.def(s1, scc), src, Operand(31u));
         Temp gtz = bld.sopc(aco_opcode::s_cmp_gt_i32, bld.def(s1, scc), src, Operand(0u));
         bld.sop2(aco_opcode::s_add_i32, Definition(dst), bld.def(s1, scc), gtz, tmp);
      } else if (dst.regClass() == v1) {
         Temp tmp = bld.vop2(aco_opcode::v_ashrrev_i32, bld.def(v1), Operand(31u), src);
         Temp gtz = bld.tmp(s2);
         bld.vopc(aco_opcode::v_cmp_ge_i32, Definition(gtz), Operand(0u), src).def(0).setHint(vcc);
         bld.vop2(aco_opcode::v_cndmask_b32, Definition(dst), Operand(1u), tmp, gtz);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imax: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_max_i32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_max_i32, dst, true);
      } else if (dst.regClass() == v2) {
         // TODO: if the result is only used for compares, we should lower it in NIR
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         assert(src0.size() == 2 && src1.size() == 2);
         Temp cmp = {ctx->program->allocateId(), s2};
         emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lt_u64, cmp);
         emit_split_vector(ctx, src0, 2);
         emit_split_vector(ctx, src1, 2);
         Temp dst0 = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                              emit_extract_vector(ctx, src0, 0, v1),
                              emit_extract_vector(ctx, src1, 0, v1),
                              cmp);
         Temp dst1 = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                              emit_extract_vector(ctx, src0, 1, v1),
                              emit_extract_vector(ctx, src1, 1, v1),
                              cmp);
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), dst0, dst1);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umax: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_max_u32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_max_u32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imin: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_min_i32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_min_i32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umin: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_min_u32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_min_u32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ior: {
      if (instr->dest.dest.ssa.bit_size == 1) {
         emit_boolean_logic(ctx, instr, aco_opcode::s_or_b32, aco_opcode::s_or_b64, dst);
      } else if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_or_b32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_or_b32, dst, true);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_or_b64, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_iand: {
      if (instr->dest.dest.ssa.bit_size == 1) {
         emit_boolean_logic(ctx, instr, aco_opcode::s_and_b32, aco_opcode::s_and_b64, dst);
      } else if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_and_b32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_and_b32, dst, true);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_and_b64, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ixor: {
      if (instr->dest.dest.ssa.bit_size == 1) {
         emit_boolean_logic(ctx, instr, aco_opcode::s_xor_b32, aco_opcode::s_xor_b64, dst);
      } else if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_xor_b32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_xor_b32, dst, true);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_xor_b64, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ushr: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_lshrrev_b32, dst, false, true);
      } else if (dst.regClass() == v2) {
         bld.vop3(aco_opcode::v_lshrrev_b64, Definition(dst),
                  get_alu_src(ctx, instr->src[1]), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_lshr_b64, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_lshr_b32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ishl: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_lshlrev_b32, dst, false, true);
      } else if (dst.regClass() == v2) {
         bld.vop3(aco_opcode::v_lshlrev_b64, Definition(dst),
                  get_alu_src(ctx, instr->src[1]), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_lshl_b32, dst, true);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_lshl_b64, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ishr: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_ashrrev_i32, dst, false, true);
      } else if (dst.regClass() == v2) {
         bld.vop3(aco_opcode::v_ashrrev_i64, Definition(dst),
                  get_alu_src(ctx, instr->src[1]), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_ashr_i32, dst, true);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_ashr_i64, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_find_lsb: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (src.regClass() == s1) {
         bld.sop1(aco_opcode::s_ff1_i32_b32, Definition(dst), src);
      } else if (src.regClass() == v1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_ffbl_b32, dst);
      } else if (src.regClass() == s2) {
         bld.sop1(aco_opcode::s_ff1_i32_b64, Definition(dst), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ufind_msb:
   case nir_op_ifind_msb: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (src.regClass() == s1 || src.regClass() == s2) {
         aco_opcode op = src.regClass() == s2 ?
                         (instr->op == nir_op_ufind_msb ? aco_opcode::s_flbit_i32_b64 : aco_opcode::s_flbit_i32_i64) :
                         (instr->op == nir_op_ufind_msb ? aco_opcode::s_flbit_i32_b32 : aco_opcode::s_flbit_i32);
         Temp msb_rev = bld.sop1(op, bld.def(s1), src);

         Builder::Result sub = bld.sop2(aco_opcode::s_sub_u32, bld.def(s1), bld.def(s1, scc),
                                        Operand(src.size() * 32u - 1u), msb_rev);
         Temp msb = sub.def(0).getTemp();
         Temp carry = sub.def(1).getTemp();

         bld.sop2(aco_opcode::s_cselect_b32, Definition(dst), Operand((uint32_t)-1), msb, carry);
      } else if (src.regClass() == v1) {
         aco_opcode op = instr->op == nir_op_ufind_msb ? aco_opcode::v_ffbh_u32 : aco_opcode::v_ffbh_i32;
         Temp msb_rev = bld.tmp(v1);
         emit_vop1_instruction(ctx, instr, op, msb_rev);
         Temp msb = bld.tmp(v1);
         Temp carry = emit_v_sub32(ctx, msb, Operand((uint32_t) 31), Operand(msb_rev), true);
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), msb, Operand((uint32_t)-1), carry);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_bitfield_reverse: {
      if (dst.regClass() == s1) {
         bld.sop1(aco_opcode::s_brev_b32, Definition(dst), get_alu_src(ctx, instr->src[0]));
      } else if (dst.regClass() == v1) {
         bld.vop1(aco_opcode::v_bfrev_b32, Definition(dst), get_alu_src(ctx, instr->src[0]));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_iadd: {
      if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_add_i32, dst, true);
         break;
      } else if (dst.regClass() == s2) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         Temp src00 = emit_extract_vector(ctx, src0, 0, s1);
         Temp src10 = emit_extract_vector(ctx, src1, 0, s1);
         Temp src01 = emit_extract_vector(ctx, src0, 1, s1);
         Temp src11 = emit_extract_vector(ctx, src1, 1, s1);
         Temp carry = bld.tmp(s1);
         Temp dst0 = bld.sop2(aco_opcode::s_add_u32, bld.def(s1), bld.scc(Definition(carry)), src00, src10);
         Temp dst1 = bld.sop2(aco_opcode::s_addc_u32, bld.def(s1), bld.def(s1, scc), src01, src11, bld.scc(carry));
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), dst0, dst1);
         break;
      }
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      aco_ptr<Instruction> add;
      if (dst.regClass() == v1) {
         emit_v_add32(ctx, dst, Operand(src0), Operand(src1));
      } else if (dst.regClass() == v2) {
         assert(src0.size() == 2 && src1.size() == 2);
         emit_split_vector(ctx, src0, 2);
         emit_split_vector(ctx, src1, 2);
         Temp src00 = emit_extract_vector(ctx, src0, 0, getRegClass(src0.type(), 1));
         Temp src10 = emit_extract_vector(ctx, src1, 0, getRegClass(src1.type(), 1));

         Temp dst0 = bld.tmp(v1), dst1 = bld.tmp(v1);
         Temp carry = emit_v_add32(ctx, dst0, Operand(src00), Operand(src10), true);
         bld.vop2(aco_opcode::v_addc_co_u32, Definition(dst1), bld.def(s2),
                  emit_extract_vector(ctx, src0, 1, v1),
                  emit_extract_vector(ctx, src1, 1, v1),
                  carry).def(1).setHint(vcc);

         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), dst0, dst1);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_uadd_sat: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      if (dst.regClass() == s1) {
         Temp tmp = bld.tmp(s1), carry = bld.tmp(s1);
         bld.sop2(aco_opcode::s_add_u32, Definition(tmp), bld.scc(Definition(carry)),
                  src0, src1);
         bld.sop2(aco_opcode::s_cselect_b32, Definition(dst), Operand((uint32_t) -1), tmp, bld.scc(carry));
      } else if (dst.regClass() == v1) {
         if (ctx->options->chip_class >= GFX9) {
            aco_ptr<VOP3A_instruction> add{create_instruction<VOP3A_instruction>(aco_opcode::v_add_u32, asVOP3(Format::VOP2), 2, 1)};
            add->getOperand(0) = Operand(src0);
            add->getOperand(1) = Operand(src1);
            add->getDefinition(0) = Definition(dst);
            add->clamp = 1;
            ctx->block->instructions.emplace_back(std::move(add));
         } else {
            if (src1.regClass() != v1)
               std::swap(src0, src1);
            assert(src1.regClass() == v1);
            Temp tmp = bld.tmp(v1), carry = bld.tmp(v1);
            bld.vop2(aco_opcode::v_add_co_u32, Definition(tmp), Definition(carry),
                     src0, src1).def(1).setHint(vcc);
            bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), tmp, Operand((uint32_t) -1), carry);
         }
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_uadd_carry: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      if (dst.regClass() == s1) {
         bld.sop2(aco_opcode::s_add_u32, bld.def(s1), bld.scc(Definition(dst)), src0, src1);
      } else if (dst.regClass() == v1) {
         Temp carry = emit_v_add32(ctx, bld.tmp(v1), Operand(src0), Operand(src1), true);
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), Operand(1u), carry);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_isub: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_sub_i32, dst, true);
      } else if (dst.regClass() == v1) {
         emit_v_sub32(ctx, dst, Operand(src0), Operand(src1));
      } else if (dst.regClass() == v2) {
         emit_split_vector(ctx, src0, 2);
         emit_split_vector(ctx, src1, 2);
         Temp lower = bld.tmp(v1);
         Temp borrow = emit_v_sub32(ctx, lower, Operand(emit_extract_vector(ctx, src0, 0, v1)),
                                    Operand(emit_extract_vector(ctx, src1, 0, v1)), true);
         Temp upper = bld.tmp(v1);
         emit_v_sub32(ctx, upper, Operand(emit_extract_vector(ctx, src0, 1, v1)),
                      Operand(emit_extract_vector(ctx, src1, 1, v1)), false, Operand(borrow));
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), lower, upper);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_usub_borrow: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      if (dst.regClass() == s1) {
         bld.sop2(aco_opcode::s_sub_u32, bld.def(s1), bld.scc(Definition(dst)), src0, src1);
      } else if (dst.regClass() == v1) {
         Temp tmp = {ctx->program->allocateId(), v1};
         Temp borrow = emit_v_sub32(ctx, tmp, Operand(src0), Operand(src1), true);
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), Operand(1u), borrow);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imul: {
      if (dst.regClass() == v1) {
         bld.vop3(aco_opcode::v_mul_lo_u32, Definition(dst),
                  get_alu_src(ctx, instr->src[0]), get_alu_src(ctx, instr->src[1]));
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_mul_i32, dst, false);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umul_high: {
      if (dst.regClass() == v1) {
         bld.vop3(aco_opcode::v_mul_hi_u32, Definition(dst), get_alu_src(ctx, instr->src[0]), get_alu_src(ctx, instr->src[1]));
      } else if (dst.regClass() == s1) {
         Temp tmp = bld.vop3(aco_opcode::v_mul_hi_u32, bld.def(v1), get_alu_src(ctx, instr->src[0]),
                             as_vgpr(ctx, get_alu_src(ctx, instr->src[1])));
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imul_high: {
      if (dst.regClass() == v1) {
         bld.vop3(aco_opcode::v_mul_hi_i32, Definition(dst), get_alu_src(ctx, instr->src[0]), get_alu_src(ctx, instr->src[1]));
      } else if (dst.regClass() == s1) {
         Temp tmp = bld.vop3(aco_opcode::v_mul_hi_i32, bld.def(v1), get_alu_src(ctx, instr->src[0]),
                             as_vgpr(ctx, get_alu_src(ctx, instr->src[1])));
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmul: {
      if (dst.size() == 1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_mul_f32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fadd: {
      if (dst.size() == 1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_add_f32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsub: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);
      if (dst.size() == 1) {
         if (src1.type() == vgpr || src0.type() != vgpr)
            emit_vop2_instruction(ctx, instr, aco_opcode::v_sub_f32, dst, false);
         else
            emit_vop2_instruction(ctx, instr, aco_opcode::v_subrev_f32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmod:
   case nir_op_frem: {
      if (dst.size() == 1) {
         Temp rcp = bld.vop1(aco_opcode::v_rcp_f32, bld.def(v1), get_alu_src(ctx, instr->src[1]));
         Temp mul = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), get_alu_src(ctx, instr->src[0]), rcp);

         aco_opcode op = instr->op == nir_op_fmod ? aco_opcode::v_floor_f32 : aco_opcode::v_trunc_f32;
         Temp floor = bld.vop1(op, bld.def(v1), mul);

         mul = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), get_alu_src(ctx, instr->src[1]), floor);
         bld.vop2(aco_opcode::v_sub_f32, Definition(dst), get_alu_src(ctx, instr->src[0]), mul);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmax: {
      if (dst.size() == 1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_max_f32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmin: {
      if (dst.size() == 1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_min_f32, dst, true);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmax3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_max3_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmin3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_min3_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmed3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_med3_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umax3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_max3_u32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umin3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_min3_u32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_umed3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_med3_u32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imax3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_max3_i32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imin3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_min3_i32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imed3: {
      if (dst.size() == 1) {
         emit_vop3a_instruction(ctx, instr, aco_opcode::v_med3_i32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_cube_face_coord: {
      Temp in = get_ssa_temp(ctx, instr->src[0].src.ssa);
      emit_split_vector(ctx, in, 3);
      Temp src[3] = { emit_extract_vector(ctx, in, 0, v1),
                      emit_extract_vector(ctx, in, 1, v1),
                      emit_extract_vector(ctx, in, 2, v1) };
      Temp ma = bld.vop3(aco_opcode::v_cubema_f32, bld.def(v1), src[0], src[1], src[2]);
      ma = bld.vop1(aco_opcode::v_rcp_f32, bld.def(v1), ma);
      Temp sc = bld.vop3(aco_opcode::v_cubesc_f32, bld.def(v1), src[0], src[1], src[2]);
      Temp tc = bld.vop3(aco_opcode::v_cubetc_f32, bld.def(v1), src[0], src[1], src[2]);
      sc = bld.vop2(aco_opcode::v_madak_f32, bld.def(v1), sc, ma, Operand(0x3f000000u/*0.5*/));
      tc = bld.vop2(aco_opcode::v_madak_f32, bld.def(v1), tc, ma, Operand(0x3f000000u/*0.5*/));
      bld.pseudo(aco_opcode::p_create_vector, Definition(dst), sc, tc);
      break;
   }
   case nir_op_cube_face_index: {
      Temp in = get_ssa_temp(ctx, instr->src[0].src.ssa);
      emit_split_vector(ctx, in, 3);
      Temp src[3] = { emit_extract_vector(ctx, in, 0, v1),
                      emit_extract_vector(ctx, in, 1, v1),
                      emit_extract_vector(ctx, in, 2, v1) };
      bld.vop3(aco_opcode::v_cubeid_f32, Definition(dst), src[0], src[1], src[2]);
      break;
   }
   case nir_op_bcsel: {
      emit_bcsel(ctx, instr, dst);
      break;
   }
   case nir_op_frsq: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_rsq_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fneg: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.size() == 1) {
         bld.vop2(aco_opcode::v_xor_b32, Definition(dst), Operand(0x80000000u), as_vgpr(ctx, src));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fabs: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.size() == 1) {
         bld.vop2(aco_opcode::v_and_b32, Definition(dst), Operand(0x7FFFFFFFu), as_vgpr(ctx, src));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsat: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.size() == 1) {
         bld.vop3(aco_opcode::v_med3_f32, Definition(dst), Operand(0u), Operand(0x3f800000u), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_flog2: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_log_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_frcp: {
      if (dst.size() == 1)
         emit_vop1_instruction(ctx, instr, aco_opcode::v_rcp_f32, dst);
      else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fexp2: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_exp_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsqrt: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_sqrt_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ffract: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_fract_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ffloor: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_floor_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fceil: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_ceil_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ftrunc: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_trunc_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fround_even: {
      if (dst.size() == 1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_rndne_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsin:
   case nir_op_fcos: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      aco_ptr<Instruction> norm;
      if (dst.size() == 1) {
         Temp tmp;
         Operand half_pi(0x3e22f983u);
         if (src.type() == sgpr)
            tmp = bld.vop2_e64(aco_opcode::v_mul_f32, bld.def(v1), half_pi, src);
         else
            tmp = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), half_pi, src);

         aco_opcode opcode = instr->op == nir_op_fsin ? aco_opcode::v_sin_f32 : aco_opcode::v_cos_f32;
         bld.vop1(opcode, Definition(dst), tmp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ldexp: {
      if (dst.size() == 1) {
         bld.vop3(aco_opcode::v_ldexp_f32, Definition(dst),
                  as_vgpr(ctx, get_alu_src(ctx, instr->src[0])),
                  get_alu_src(ctx, instr->src[1]));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_frexp_sig: {
      if (dst.size() == 1) {
         bld.vop1(aco_opcode::v_frexp_mant_f32, Definition(dst),
                  get_alu_src(ctx, instr->src[0]));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_frexp_exp: {
      if (dst.size() == 1) {
         bld.vop1(aco_opcode::v_frexp_exp_i32_f32, Definition(dst),
                  get_alu_src(ctx, instr->src[0]));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsign: {
      Temp src = as_vgpr(ctx, get_alu_src(ctx, instr->src[0]));
      if (dst.size() == 1) {
         Temp temp = bld.tmp(s2);
         bld.vopc(aco_opcode::v_cmp_nlt_f32, Definition(temp), Operand(0u), src).def(0).setHint(vcc);
         src = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), Operand(0x3f800000u), src, temp);

         temp = bld.tmp(s2);
         bld.vopc(aco_opcode::v_cmp_le_f32, Definition(temp), Operand(0u), src).def(0).setHint(vcc);
         bld.vop2(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0xbf800000u), src, temp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_i2i64: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (src.size() == 1) {
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), src, Operand(0u));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_i2f32: {
      assert(dst.size() == 1);
      emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_f32_i32, dst);
      break;
   }
   case nir_op_u2f32: {
      assert(dst.size() == 1);
      emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_f32_u32, dst);
      break;
   }
   case nir_op_f2i32: {
      if (dst.regClass() == s1) {
         Temp tmp = bld.tmp(v1);
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, tmp);
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
      } else {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, dst);
      }
      break;
   }
   case nir_op_f2u32: {
      if (dst.regClass() == s1) {
         Temp tmp = bld.tmp(v1);
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_u32_f32, tmp);
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
      } else {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_u32_f32, dst);
      }
      break;
   }
   case nir_op_b2f32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s1) {
         aco_ptr<Instruction> sop2;
         src = as_uniform_bool(ctx, src);
         bld.sop2(aco_opcode::s_mul_i32, Definition(dst), Operand(0x3f800000u), src);
      } else if (dst.regClass() == v1) {
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), Operand(0x3f800000u),
                      as_divergent_bool(ctx, src, true));
      } else {
         unreachable("Wrong destination register class for nir_op_b2f32.");
      }
      break;
   }
   case nir_op_u2u32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (instr->src[0].src.ssa->bit_size == 16) {
         if (dst.regClass() == s1) {
            bld.sop2(aco_opcode::s_and_b32, Definition(dst), bld.def(s1, scc), Operand(0xFFFFu), src);
         } else {
            // TODO: do better with SDWA
            bld.vop2(aco_opcode::v_and_b32, Definition(dst), Operand(0xFFFFu), src);
         }
      } else if (instr->src[0].src.ssa->bit_size == 64) {
         /* we can actually just say dst = src, as it would map the lower register */
         emit_extract_vector(ctx, src, 0, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_b2i32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s1) {
         if (src.regClass() == s1) {
            aco_ptr<Instruction> mov = create_s_mov(Definition(dst), Operand(src));
            ctx->block->instructions.emplace_back(std::move(mov));
         } else {
            // TODO: in a post-RA optimization, we can check if src is in VCC, and directly use VCCNZ
            assert(src.regClass() == s2);
            src = emit_extract_vector(ctx, src, 0, s1);
            bld.sop2(aco_opcode::s_and_b32, Definition(dst), bld.def(s1, scc), Operand(1u), src);
         }
      } else {
         assert(dst.regClass() == v1 && src.regClass() == s2);
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), Operand(1u), src);
      }
      break;
   }
   case nir_op_i2b1: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s2) {
         assert(src.regClass() == v1 || src.regClass() == v2);
         bld.vopc(src.size() == 2 ? aco_opcode::v_cmp_lg_u64 : aco_opcode::v_cmp_lg_u32,
                  Definition(dst), Operand(0u), src).def(0).setHint(vcc);
      } else {
         assert(src.regClass() == s1 && dst.regClass() == s1);
         bld.sopc(aco_opcode::s_cmp_lg_u32, bld.scc(Definition(dst)), Operand(0u), src);
      }
      break;
   }
   case nir_op_pack_64_2x32_split: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);

      bld.pseudo(aco_opcode::p_create_vector, Definition(dst), src0, src1);
      break;
   }
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      emit_split_vector(ctx, src, 2);
      emit_extract_vector(ctx, src, instr->op == nir_op_unpack_64_2x32_split_x ? 0 : 1, dst);
      break;
   }
   case nir_op_pack_half_2x16: {
      Temp src = get_ssa_temp(ctx, instr->src[0].src.ssa);
      emit_split_vector(ctx, src, 2);

      if (dst.regClass() == v1) {
         Temp src0 = emit_extract_vector(ctx, src, 0, v1);
         Temp src1 = emit_extract_vector(ctx, src, 1, v1);
         bld.vop3(aco_opcode::v_cvt_pkrtz_f16_f32, Definition(dst), src0, src1);

      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_unpack_half_2x16: {
      if (dst.regClass() == v2) {
         Temp src = get_alu_src(ctx, instr->src[0]);
         Builder bld(ctx->program, ctx->block);
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    bld.vop1(aco_opcode::v_cvt_f32_f16, bld.def(v1),
                             bld.vop3(aco_opcode::v_bfe_u32, bld.def(v1), src, Operand(0u), Operand(16u))),
                    bld.vop1(aco_opcode::v_cvt_f32_f16, bld.def(v1),
                             bld.vop3(aco_opcode::v_bfe_u32, bld.def(v1), src, Operand(16u), Operand(16u))));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fquantize2f16: {
      Temp f16 = bld.vop1(aco_opcode::v_cvt_f16_f32, bld.def(v1), get_alu_src(ctx, instr->src[0]));

      Temp mask = bld.tmp(s1);
      aco_ptr<Instruction> mov = create_s_mov(Definition(mask), Operand((uint32_t) 0x36F)); /* value is NOT negative/positive denormal value */
      ctx->block->instructions.emplace_back(std::move(mov));

      Temp cmp_res = bld.tmp(s2);
      bld.vopc_e64(aco_opcode::v_cmp_class_f16, Definition(cmp_res), f16, mask).def(0).setHint(vcc);

      Temp f32 = bld.vop1(aco_opcode::v_cvt_f32_f16, bld.def(v1), f16);

      bld.vop2(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), f32, cmp_res);
      break;
   }
   case nir_op_bfm: {
      Temp bits = get_alu_src(ctx, instr->src[0]);
      Temp offset = get_alu_src(ctx, instr->src[1]);

      if (dst.regClass() == s1) {
         bld.sop2(aco_opcode::s_bfm_b32, Definition(dst), bits, offset);
      } else if (dst.regClass() == v1) {
         bld.vop3(aco_opcode::v_bfm_b32, Definition(dst), bits, offset);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_bitfield_select: {
      /* (mask & insert) | (~mask & base) */
      Temp bitmask = get_alu_src(ctx, instr->src[0]);
      Temp insert = get_alu_src(ctx, instr->src[1]);
      Temp base = get_alu_src(ctx, instr->src[2]);

      /* dst = (insert & bitmask) | (base & ~bitmask) */
      if (dst.regClass() == s1) {
         aco_ptr<Instruction> sop2;
         nir_const_value* const_bitmask = nir_src_as_const_value(instr->src[0].src);
         nir_const_value* const_insert = nir_src_as_const_value(instr->src[1].src);
         Operand lhs;
         if (const_insert && const_bitmask) {
            lhs = Operand(const_insert->u32 & const_bitmask->u32);
         } else {
            insert = bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), insert, bitmask);
            lhs = Operand(insert);
         }

         Operand rhs;
         nir_const_value* const_base = nir_src_as_const_value(instr->src[2].src);
         if (const_base && const_bitmask) {
            rhs = Operand(const_base->u32 & ~const_bitmask->u32);
         } else {
            base = bld.sop2(aco_opcode::s_andn2_b32, bld.def(s1), bld.def(s1, scc), base, bitmask);
            rhs = Operand(base);
         }

         bld.sop2(aco_opcode::s_or_b32, Definition(dst), bld.def(s1, scc), rhs, lhs);

      } else if (dst.regClass() == v1) {
         if (base.type() == sgpr && (bitmask.type() == sgpr || (insert.type() == sgpr)))
            base = as_vgpr(ctx, base);
         if (insert.type() == sgpr && bitmask.type() == sgpr)
            insert = as_vgpr(ctx, insert);

         bld.vop3(aco_opcode::v_bfi_b32, Definition(dst), bitmask, insert, base);

      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ubfe:
   case nir_op_ibfe: {
      Temp base = get_alu_src(ctx, instr->src[0]);
      Temp offset = get_alu_src(ctx, instr->src[1]);
      Temp bits = get_alu_src(ctx, instr->src[2]);

      if (dst.type() == sgpr) {
         Operand extract;
         nir_const_value* const_offset = nir_src_as_const_value(instr->src[1].src);
         nir_const_value* const_bits = nir_src_as_const_value(instr->src[2].src);
         if (const_offset && const_bits) {
            uint32_t const_extract = (const_bits->u32 << 16) | const_offset->u32;
            extract = Operand(const_extract);
         } else {
            Operand width;
            if (const_bits) {
               width = Operand(const_bits->u32 << 16);
            } else {
               width = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), bits, Operand(16u));
            }
            extract = bld.sop2(aco_opcode::s_or_b32, bld.def(s1), bld.def(s1, scc), offset, width);
         }

         aco_opcode opcode;
         if (dst.regClass() == s1) {
            if (instr->op == nir_op_ubfe)
               opcode = aco_opcode::s_bfe_u32;
            else
               opcode = aco_opcode::s_bfe_i32;
         } else if (dst.regClass() == s2) {
            if (instr->op == nir_op_ubfe)
               opcode = aco_opcode::s_bfe_u64;
            else
               opcode = aco_opcode::s_bfe_i64;
         } else {
            unreachable("Unsupported BFE bit size");
         }

         bld.sop2(opcode, Definition(dst), bld.def(s1, scc), base, extract);

      } else {
         aco_opcode opcode;
         if (dst.regClass() == v1) {
            if (instr->op == nir_op_ubfe)
               opcode = aco_opcode::v_bfe_u32;
            else
               opcode = aco_opcode::v_bfe_i32;
         } else {
            unreachable("Unsupported BFE bit size");
         }

         emit_vop3a_instruction(ctx, instr, opcode, dst);
      }
      break;
   }
   case nir_op_bit_count: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (src.regClass() == s1) {
         bld.sop1(aco_opcode::s_bcnt1_i32_b32, Definition(dst), bld.def(s1, scc), src);
      } else if (src.regClass() == v1) {
         bld.vop3(aco_opcode::v_bcnt_u32_b32, Definition(dst), src, Operand(0u));
      } else if (src.regClass() == v2) {
         bld.vop3(aco_opcode::v_bcnt_u32_b32, Definition(dst),
                  emit_extract_vector(ctx, src, 1, v1),
                  bld.vop3(aco_opcode::v_bcnt_u32_b32, bld.def(v1),
                           emit_extract_vector(ctx, src, 0, v1), Operand(0u)));
      } else if (src.regClass() == s2) {
         bld.sop1(aco_opcode::s_bcnt1_i32_b64, Definition(dst), bld.def(s1, scc), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_flt: {
      if (instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_f32, dst);
      else if (instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_f64, dst);
      break;
   }
   case nir_op_fge: {
      if (instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_f32, dst);
      else if (instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_f64, dst);
      break;
   }
   case nir_op_feq: {
      if (instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_eq_f32, dst);
      else if (instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_eq_f64, dst);
      break;
   }
   case nir_op_fne: {
      if (instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_neq_f32, dst);
      else if (instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_neq_f64, dst);
      break;
   }
   case nir_op_ilt: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_i32, dst);
      else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::s_cmp_lt_i32, dst);
      else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_i64, dst);
      break;
   }
   case nir_op_ige: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_i32, dst);
      else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::s_cmp_ge_i32, dst);
      else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_i64, dst);
      break;
   }
   case nir_op_ieq: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32) {
         emit_comparison(ctx, instr, aco_opcode::v_cmp_eq_i32, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32) {
         emit_comparison(ctx, instr, aco_opcode::s_cmp_eq_i32, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 64) {
         emit_comparison(ctx, instr, aco_opcode::s_cmp_eq_u64, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         bld.sopc(aco_opcode::s_cmp_eq_i32, bld.scc(Definition(dst)),
                  as_uniform_bool(ctx, src0), as_uniform_bool(ctx, src1));
      } else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         bld.sop2(aco_opcode::s_xnor_b64, Definition(dst), bld.def(s1, scc),
                  as_divergent_bool(ctx, src0, false), as_divergent_bool(ctx, src1, false));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ine: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32) {
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lg_i32, dst);
      } else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 64) {
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lg_i64, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32) {
         emit_comparison(ctx, instr, aco_opcode::s_cmp_lg_i32, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 64) {
         emit_comparison(ctx, instr, aco_opcode::s_cmp_lg_u64, dst);
      } else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         bld.sopc(aco_opcode::s_cmp_lg_i32, bld.scc(Definition(dst)),
                  as_uniform_bool(ctx, src0), as_uniform_bool(ctx, src1));
      } else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         bld.sop2(aco_opcode::s_xor_b64, Definition(dst), bld.def(s1, scc),
                  as_divergent_bool(ctx, src0, false), as_divergent_bool(ctx, src1, false));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ult: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_u32, dst);
      else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::s_cmp_lt_u32, dst);
      else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_lt_u64, dst);
      break;
   }
   case nir_op_uge: {
      if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_u32, dst);
      else if (dst.regClass() == s1 && instr->src[0].src.ssa->bit_size == 32)
         emit_comparison(ctx, instr, aco_opcode::s_cmp_ge_u32, dst);
      else if (dst.regClass() == s2 && instr->src[0].src.ssa->bit_size == 64)
         emit_comparison(ctx, instr, aco_opcode::v_cmp_ge_u64, dst);
      break;
   }
   case nir_op_fddx:
   case nir_op_fddy:
   case nir_op_fddx_fine:
   case nir_op_fddy_fine:
   case nir_op_fddx_coarse:
   case nir_op_fddy_coarse: {
      Definition tl = bld.def(v1);
      uint16_t dpp_ctrl;
      if (instr->op == nir_op_fddx_fine) {
         bld.vop1_dpp(aco_opcode::v_mov_b32, tl, get_alu_src(ctx, instr->src[0]), dpp_quad_perm(0, 0, 2, 2));
         dpp_ctrl = dpp_quad_perm(1, 1, 3, 3);
      } else if (instr->op == nir_op_fddy_fine) {
         bld.vop1_dpp(aco_opcode::v_mov_b32, tl, get_alu_src(ctx, instr->src[0]), dpp_quad_perm(0, 1, 0, 1));
         dpp_ctrl = dpp_quad_perm(2, 3, 2, 3);
      } else {
         bld.vop1_dpp(aco_opcode::v_mov_b32, tl, get_alu_src(ctx, instr->src[0]), dpp_quad_perm(0, 0, 0, 0));
         if (instr->op == nir_op_fddx || instr->op == nir_op_fddx_coarse)
            dpp_ctrl = dpp_quad_perm(1, 1, 1, 1);
         else
            dpp_ctrl = dpp_quad_perm(2, 2, 2, 2);
      }

      Definition tmp = bld.def(v1);
      bld.vop2_dpp(aco_opcode::v_sub_f32, tmp, get_alu_src(ctx, instr->src[0]), tl.getTemp(), dpp_ctrl);
      emit_wqm(ctx, tmp.getTemp(), dst);
      break;
   }
   case nir_op_urcp: {
      if (dst.regClass() == v1 || dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         aco_ptr<Instruction> instr;

         Temp f_src0 = bld.vop1(aco_opcode::v_cvt_f32_u32, bld.def(v1), src0);
         Temp rcp = bld.vop1(aco_opcode::v_rcp_iflag_f32, bld.def(v1), f_src0);
         Temp f_dst = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), Operand(0x4f800000u), rcp);

         Temp tmp = dst.regClass() == s1 ? bld.tmp(v1) : dst;
         bld.vop1(aco_opcode::v_cvt_u32_f32, Definition(tmp), f_dst);

         if (dst.regClass() == s1)
            bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   default:
      fprintf(stderr, "Unknown NIR ALU instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
   }
}

void visit_load_const(isel_context *ctx, nir_load_const_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->def);

   // TODO: we really want to have the resulting type as this would allow for 64bit literals
   // which get truncated the lsb if double and msb if int
   // for now, we only use s_mov_b64 with 64bit inline constants
   assert(instr->def.num_components == 1 && "Vector load_const should be lowered to scalar.");
   assert(dst.type() == sgpr);

   if (dst.size() == 1)
   {
      aco_ptr<Instruction> mov = create_s_mov(Definition(dst), Operand(instr->value[0].u32));
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      assert(dst.size() != 1);
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1)};
      if (instr->def.bit_size == 64)
         for (unsigned i = 0; i < dst.size(); i++)
            vec->getOperand(i) = Operand{(uint32_t)(instr->value[0].u64 >> i * 32)};
      else {
         for (unsigned i = 0; i < dst.size(); i++)
            vec->getOperand(i) = Operand{instr->value[i].u32};
      }
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
   }
}

void visit_store_output(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned write_mask = nir_intrinsic_write_mask(instr);
   Operand values[4];
   Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
   for (unsigned i = 0; i < 4; ++i) {
      if (write_mask & (1 << i)) {
         Temp tmp = emit_extract_vector(ctx, src, i, v1);
         values[i] = Operand(tmp);
      } else {
         values[i] = Operand(v1);
      }
   }

   unsigned index = nir_intrinsic_base(instr) / 4;
   unsigned target, col_format;
   unsigned enabled_channels = 0xF;
   aco_opcode compr_op = (aco_opcode)0;

   assert(index != FRAG_RESULT_COLOR);

   /* Unlike vertex shader exports, it's fine to use multiple exports to
    * export separate channels of one target. So shaders which export both
    * FRAG_RESULT_SAMPLE_MASK and FRAG_RESULT_DEPTH should work fine.
    * TODO: combine the exports in those cases and create better code
    */

   if (index == FRAG_RESULT_SAMPLE_MASK) {

      if (ctx->program->info->info.ps.writes_z) {
         target = V_008DFC_SQ_EXP_MRTZ;
         enabled_channels = 0x4;
         col_format = (unsigned) -1;

         values[2] = values[0];
         values[0] = Operand(v1);
      } else {
         aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
         exp->valid_mask = false;
         exp->done = false;
         exp->compressed = true;
         exp->dest = V_008DFC_SQ_EXP_MRTZ;
         exp->enabled_mask = 0xc;
         for (int i = 0; i < 4; i++)
            exp->getOperand(i) = Operand(v1);
         exp->getOperand(1) = Operand(values[0]);
         ctx->block->instructions.emplace_back(std::move(exp));
         return;
      }

   } else if (index == FRAG_RESULT_DEPTH) {

      target = V_008DFC_SQ_EXP_MRTZ;
      enabled_channels = 0x1;
      col_format = (unsigned) -1;

   } else if (index == FRAG_RESULT_STENCIL) {

      assert(!ctx->program->info->info.ps.writes_z && "unimplemented");

      aco_ptr<Instruction> shift{create_instruction<VOP2_instruction>(aco_opcode::v_lshlrev_b32, Format::VOP2, 2, 1)};
      shift->getOperand(0) = Operand((uint32_t) 16);
      shift->getOperand(1) = values[0];
      Temp tmp = {ctx->program->allocateId(), v1};
      shift->getDefinition(0) = Definition(tmp);
      ctx->block->instructions.emplace_back(std::move(shift));

      aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
      exp->valid_mask = false; // TODO
      exp->done = false; // TODO
      exp->compressed = true;
      exp->dest = V_008DFC_SQ_EXP_MRTZ;
      exp->enabled_mask = 0x3;
      exp->getOperand(0) = Operand(tmp);
      for (int i = 1; i < 4; i++)
         exp->getOperand(i) = Operand(v1);
      ctx->block->instructions.emplace_back(std::move(exp));
      return;

   } else {
      index = index - FRAG_RESULT_DATA0;
      nir_const_value* offset = nir_src_as_const_value(instr->src[1]);
      assert(offset && "Non-const offsets on exports not yet supported");
      index += offset->u32;
      target = V_008DFC_SQ_EXP_MRT + index;
      col_format = (ctx->options->key.fs.col_format >> (4 * index)) & 0xf;
   }
   bool is_int8 = (ctx->options->key.fs.is_int8 >> index) & 1;
   bool is_int10 = (ctx->options->key.fs.is_int10 >> index) & 1;
   assert(!is_int8 && !is_int10);

   switch (col_format)
   {
   case V_028714_SPI_SHADER_ZERO:
      enabled_channels = 0; /* writemask */
      target = V_008DFC_SQ_EXP_NULL;
      break;

   case V_028714_SPI_SHADER_32_R:
      enabled_channels = 1;
      break;

   case V_028714_SPI_SHADER_32_GR:
      enabled_channels = 0x3;
      break;

   case V_028714_SPI_SHADER_32_AR:
      enabled_channels = 0x9;
      break;

   case V_028714_SPI_SHADER_FP16_ABGR:
      enabled_channels = 0x5;
      compr_op = aco_opcode::v_cvt_pkrtz_f16_f32;
      break;

   case V_028714_SPI_SHADER_UNORM16_ABGR:
      enabled_channels = 0x5;
      compr_op = aco_opcode::v_cvt_pknorm_u16_f32;
      break;

   case V_028714_SPI_SHADER_SNORM16_ABGR:
      enabled_channels = 0x5;
      compr_op = aco_opcode::v_cvt_pknorm_i16_f32;
      break;

   case V_028714_SPI_SHADER_UINT16_ABGR:
      enabled_channels = 0x5;
      compr_op = aco_opcode::v_cvt_pk_u16_u32;
      break;

   case V_028714_SPI_SHADER_SINT16_ABGR:
      enabled_channels = 0x5;
      compr_op = aco_opcode::v_cvt_pk_i16_i32;
      break;

   case V_028714_SPI_SHADER_32_ABGR:
      enabled_channels = 0xF;
      break;

   default:
      break;
   }

   if (target == V_008DFC_SQ_EXP_NULL)
      return;

   if ((bool)compr_op)
   {
      for (int i = 0; i < 2; i++)
      {
         /* check if at least one of the values to be compressed is enabled */
         unsigned enabled = (write_mask >> (i*2) | write_mask >> (i*2+1)) & 0x1;
         if (enabled) {
            enabled_channels |= enabled << (i*2);
            aco_ptr<VOP3A_instruction> compr{create_instruction<VOP3A_instruction>(compr_op, Format::VOP3A, 2, 1)};
            Temp tmp{ctx->program->allocateId(), v1};
            compr->getOperand(0) = values[i*2];
            compr->getOperand(1) = values[i*2+1];
            compr->getDefinition(0) = Definition(tmp);
            values[i] = Operand(tmp);
            ctx->block->instructions.emplace_back(std::move(compr));
         } else {
            values[i] = Operand(v1);
         }
      }
   }

   aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
   exp->valid_mask = false; // TODO
   exp->done = false; // TODO
   exp->compressed = (bool) compr_op;
   exp->dest = target;
   exp->enabled_mask = enabled_channels;
   if ((bool) compr_op) {
      for (int i = 0; i < 2; i++)
         exp->getOperand(i) = enabled_channels & (3 << (i * 2)) ? values[i] : Operand(v1);
      exp->getOperand(2) = Operand(v1);
      exp->getOperand(3) = Operand(v1);
   } else {
      for (int i = 0; i < 4; i++)
         exp->getOperand(i) = enabled_channels & (1 << i) ? values[i] : Operand(v1);
   }

   ctx->block->instructions.emplace_back(std::move(exp));
}

void emit_interp_instr(isel_context *ctx, unsigned idx, unsigned component, Temp src, Temp dst, Temp prim_mask)
{
   Temp coord1 = emit_extract_vector(ctx, src, 0, v1);
   Temp coord2 = emit_extract_vector(ctx, src, 1, v1);

   Builder bld(ctx->program, ctx->block);
   Temp tmp = bld.vintrp(aco_opcode::v_interp_p1_f32, bld.def(v1), coord1, bld.m0(prim_mask), idx, component);
   bld.vintrp(aco_opcode::v_interp_p2_f32, Definition(dst), coord2, bld.m0(prim_mask), tmp, idx, component);
}

void emit_load_frag_coord(isel_context *ctx, Temp dst, unsigned num_components)
{
   aco_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, num_components, 1));
   for (unsigned i = 0; i < num_components; i++)
      vec->getOperand(i) = Operand(ctx->fs_inputs[fs_input::frag_pos_0 + i]);

   if (ctx->fs_vgpr_args[fs_input::frag_pos_3]) {
      assert(num_components == 4);
      Builder bld(ctx->program, ctx->block);
      vec->getOperand(3) = bld.vop1(aco_opcode::v_rcp_f32, bld.def(v1), ctx->fs_inputs[fs_input::frag_pos_3]);
   }

   vec->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(vec));
   emit_split_vector(ctx, dst, num_components);
   return;
}

void visit_load_interpolated_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   if (nir_intrinsic_base(instr) == VARYING_SLOT_POS) {
      emit_load_frag_coord(ctx, dst, instr->dest.ssa.num_components);
      return;
   }

   assert(nir_intrinsic_base(instr) != VARYING_SLOT_CLIP_DIST0);
   uint64_t base = nir_intrinsic_base(instr) / 4;

   nir_const_value* offset = nir_src_as_const_value(instr->src[1]);
   if (offset)
      base += offset->u32;

   Temp coords = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned idx = util_bitcount64(ctx->input_mask & ((1ull << base) - 1ull));
   unsigned component = nir_intrinsic_component(instr);
   Temp prim_mask = ctx->prim_mask;

   if (!offset) {
      /* the lower 15bit of the prim_mask contain the offset into LDS
       * while the upper bits contain the number of prims */
      Temp offset_src = get_ssa_temp(ctx, instr->src[1].ssa);
      assert(offset_src.regClass() == s1 && "TODO: divergent offsets...");
      Builder bld(ctx->program, ctx->block);
      Temp stride = bld.sop2(aco_opcode::s_lshr_b32, bld.def(s1), bld.def(s1, scc), prim_mask, Operand(16u));
      stride = bld.sop1(aco_opcode::s_bcnt1_i32_b32, bld.def(s1), bld.def(s1, scc), stride);
      stride = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), stride, Operand(48u));
      offset_src = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), stride, offset_src);
      prim_mask = bld.sop2(aco_opcode::s_add_i32, bld.def(s1, m0), bld.def(s1, scc), offset_src, prim_mask);
   }

   if (instr->dest.ssa.num_components == 1) {
      emit_interp_instr(ctx, idx, component, coords, dst, prim_mask);
   } else {
      aco_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.ssa.num_components, 1));
      for (unsigned i = 0; i < instr->dest.ssa.num_components; i++)
      {
         Temp tmp = {ctx->program->allocateId(), v1};
         emit_interp_instr(ctx, idx, component+i, coords, tmp, prim_mask);
         vec->getOperand(i) = Operand(tmp);
      }
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
   }
}

void visit_load_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   if (ctx->stage == MESA_SHADER_VERTEX) {

      Temp vertex_buffers = ctx->vertex_buffers;
      if (vertex_buffers.size() == 1) {
         vertex_buffers = convert_pointer_to_64_bit(ctx, vertex_buffers);
         ctx->vertex_buffers = vertex_buffers;
      }

      unsigned offset = (nir_intrinsic_base(instr) / 4 - VERT_ATTRIB_GENERIC0) * 16;
      Temp list = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), vertex_buffers, Operand(offset));

      Temp index = {ctx->program->allocateId(), v1};
      if (ctx->options->key.vs.instance_rate_inputs & (1u << offset)) {
         fprintf(stderr, "Unimplemented: instance rate inputs\n");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
         abort();
      } else {
         emit_v_add32(ctx, index, Operand(ctx->base_vertex), Operand(ctx->vertex_id));
      }

      aco_opcode opcode;
      switch (dst.size()) {
      case 1:
         opcode = aco_opcode::buffer_load_format_x;
         break;
      case 2:
         opcode = aco_opcode::buffer_load_format_xy;
         break;
      case 3:
         opcode = aco_opcode::buffer_load_format_xyz;
         break;
      case 4:
         opcode = aco_opcode::buffer_load_format_xyzw;
         break;
      default:
         unreachable("Unimplemented load_input vector size");
      }

      aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(opcode, Format::MUBUF, 3, 1)};
      mubuf->getOperand(0) = Operand(index);
      mubuf->getOperand(1) = Operand(list); /* resource constant */
      mubuf->getOperand(2) = Operand((uint32_t) 0); /* soffset */
      mubuf->getDefinition(0) = Definition(dst);
      mubuf->idxen = true;
      mubuf->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));

      unsigned alpha_adjust = (ctx->options->key.vs.alpha_adjust >> (offset * 2)) & 3;
      if (alpha_adjust != RADV_ALPHA_ADJUST_NONE) {
         fprintf(stderr, "Unimplemented alpha adjust\n");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
         abort();
      }

   } else if (ctx->stage == MESA_SHADER_FRAGMENT) {
      unsigned base = nir_intrinsic_base(instr) / 4;
      Temp prim_mask = ctx->prim_mask;
      nir_const_value* offset = nir_src_as_const_value(instr->src[0]);
      if (offset) {
         base += offset->u32;
      } else {
         /* the lower 15bit of the prim_mask contain the offset into LDS
          * while the upper bits contain the number of prims */
         Temp offset_src = get_ssa_temp(ctx, instr->src[0].ssa);
         assert(offset_src.regClass() == s1 && "TODO: divergent offsets...");
         Builder bld(ctx->program, ctx->block);
         Temp stride = bld.sop2(aco_opcode::s_lshr_b32, bld.def(s1), bld.def(s1, scc), prim_mask, Operand(16u));
         stride = bld.sop1(aco_opcode::s_bcnt1_i32_b32, bld.def(s1), bld.def(s1, scc), stride);
         stride = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), stride, Operand(48u));
         offset_src = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), stride, offset_src);
         prim_mask = bld.sop2(aco_opcode::s_add_i32, bld.def(s1, m0), bld.def(s1, scc), offset_src, prim_mask);
      }

      unsigned idx = util_bitcount64(ctx->input_mask & ((1ull << base) - 1ull));
      unsigned component = nir_intrinsic_component(instr);
      Operand P0;
      P0.setFixed(PhysReg{2});

      if (dst.size() == 1) {
         bld.vintrp(aco_opcode::v_interp_mov_f32, Definition(dst), P0, bld.m0(prim_mask), idx, component);
      } else {
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1)};
         for (unsigned i = 0; i < dst.size(); i++)
            vec->getOperand(i) = bld.vintrp(aco_opcode::v_interp_mov_f32, bld.def(v1), P0, bld.m0(prim_mask), idx, component + i);
         vec->getDefinition(0) = Definition(dst);
         bld.insert(std::move(vec));
      }

   } else {
      unreachable("Shader stage not implemented");
   }
}

Temp load_desc_ptr(isel_context *ctx, unsigned desc_set)
{
   if (ctx->program->info->need_indirect_descriptor_sets) {
      Builder bld(ctx->program, ctx->block);
      Temp ptr64 = convert_pointer_to_64_bit(ctx, ctx->descriptor_sets[0]);
      return bld.smem(aco_opcode::s_load_dword, bld.def(s1), ptr64, Operand(desc_set << 2));//, false, false, false);
   }

   return ctx->descriptor_sets[desc_set];
}


void visit_load_resource(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Temp index = bld.as_uniform(get_ssa_temp(ctx, instr->src[0].ssa));
   unsigned desc_set = nir_intrinsic_desc_set(instr);
   unsigned binding = nir_intrinsic_binding(instr);

   Temp desc_ptr;
   radv_pipeline_layout *pipeline_layout = ctx->options->layout;
   radv_descriptor_set_layout *layout = pipeline_layout->set[desc_set].layout;
   unsigned offset = layout->binding[binding].offset;
   unsigned stride;
   if (layout->binding[binding].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       layout->binding[binding].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      unsigned idx = pipeline_layout->set[desc_set].dynamic_offset_start + layout->binding[binding].dynamic_offset_offset;
      desc_ptr = ctx->push_constants;
      offset = pipeline_layout->push_constant_size + 16 * idx;
      stride = 16;
   } else {
      desc_ptr = load_desc_ptr(ctx, desc_set);
      stride = layout->binding[binding].size;
   }

   nir_const_value* nir_const_index = nir_src_as_const_value(instr->src[0]);
   unsigned const_index = nir_const_index ? nir_const_index->u32 : 0;
   if (stride != 1) {
      if (nir_const_index) {
         const_index = const_index * stride;
      } else {
         index = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), Operand(stride), Operand(index));
      }
   }
   if (offset) {
      if (nir_const_index) {
         const_index = const_index + offset;
      } else {
         index = bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc), Operand(offset), Operand(index));
      }
   }

   if (nir_const_index && const_index == 0) {
      index = desc_ptr;
   } else {
      index = bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc),
                       nir_const_index ? Operand(const_index) : Operand(index),
                       Operand(desc_ptr));
   }

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   bld.sop1(aco_opcode::s_mov_b32, Definition(dst), index);
}

void visit_load_ubo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp rsrc = get_ssa_temp(ctx, instr->src[0].ssa);
   nir_const_value* const_offset = nir_src_as_const_value(instr->src[1]);

   Builder bld(ctx->program, ctx->block);

   nir_intrinsic_instr* idx_instr = nir_instr_as_intrinsic(instr->src[0].ssa->parent_instr);
   unsigned desc_set = nir_intrinsic_desc_set(idx_instr);
   unsigned binding = nir_intrinsic_binding(idx_instr);
   radv_descriptor_set_layout *layout = ctx->options->layout->set[desc_set].layout;

   if (layout->binding[binding].type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT) {
      uint32_t desc_type = S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
                           S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
                           S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
                           S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
                           S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
                           S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32);
      Temp upper_dwords = bld.pseudo(aco_opcode::p_create_vector, bld.def(s3),
                                     Operand(S_008F04_BASE_ADDRESS_HI(ctx->options->address32_hi)),
                                     Operand(0xFFFFFFFFu),
                                     Operand(desc_type));
      rsrc = bld.pseudo(aco_opcode::p_create_vector, bld.def(s4),
                        rsrc, upper_dwords);
   } else {
      rsrc = convert_pointer_to_64_bit(ctx, rsrc);
      rsrc = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), rsrc, Operand(0u));
   }

   if (dst.type() == sgpr) {
      aco_opcode op;
      switch(dst.size()) {
      case 1:
         op = aco_opcode::s_buffer_load_dword;
         break;
      case 2:
         op = aco_opcode::s_buffer_load_dwordx2;
         break;
      case 3:
      case 4:
         op = aco_opcode::s_buffer_load_dwordx4;
         break;
      case 8:
         op = aco_opcode::s_buffer_load_dwordx8;
         break;
      case 16:
         op = aco_opcode::s_buffer_load_dwordx16;
         break;
      default:
         unreachable("Forbidden regclass in load_ubo instruction.");
      }
      aco_ptr<SMEM_instruction> load{create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1)};
      load->getOperand(0) = Operand(rsrc);

      if (const_offset && const_offset->u32 < 0xFFFFF)
         load->getOperand(1) = Operand(const_offset->u32);
      else
         load->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      load->getDefinition(0) = Definition(dst);
      load->can_reorder = true;

      if (dst.size() == 3) {
      /* trim vector */
         Temp vec = {ctx->program->allocateId(), s4};
         load->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(load));
         emit_split_vector(ctx, vec, 4);

         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    emit_extract_vector(ctx, vec, 0, s1),
                    emit_extract_vector(ctx, vec, 1, s1),
                    emit_extract_vector(ctx, vec, 2, s1));
      } else {
         ctx->block->instructions.emplace_back(std::move(load));
      }

   } else { /* vgpr dst */
      aco_opcode op;
      switch(dst.size()) {
      case 1:
         op = aco_opcode::buffer_load_dword;
         break;
      case 2:
         op = aco_opcode::buffer_load_dwordx2;
         break;
      case 3:
         op = aco_opcode::buffer_load_dwordx3;
         break;
      case 4:
         op = aco_opcode::buffer_load_dwordx4;
         break;
      default:
         unreachable("Unimplemented regclass in load_ubo instruction.");
      }

      aco_ptr<MUBUF_instruction> mubuf;
      mubuf.reset(create_instruction<MUBUF_instruction>(op, Format::MUBUF, 3, 1));
      mubuf->getOperand(0) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      mubuf->getOperand(1) = Operand(rsrc);
      mubuf->getOperand(2) = Operand((uint32_t) 0);
      mubuf->getDefinition(0) = Definition(dst);
      mubuf->offen = true;
      mubuf->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));
   }

   emit_split_vector(ctx, dst, instr->dest.ssa.num_components);
}

void visit_load_push_constant(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   unsigned offset = nir_intrinsic_base(instr);
   nir_const_value *index_cv = nir_src_as_const_value(instr->src[0]);
   if (index_cv && instr->dest.ssa.bit_size == 32) {

      unsigned count = instr->dest.ssa.num_components;
      unsigned start = (offset + index_cv->u32) / 4u;
      start -= ctx->base_inline_push_consts;
      if (start + count <= ctx->num_inline_push_consts) {
         std::array<Temp,NIR_MAX_VEC_COMPONENTS> elems;
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, count, 1)};
         for (unsigned i = 0; i < count; ++i) {
            elems[i] = ctx->inline_push_consts[start + i];
            vec->getOperand(i) = Operand{elems[i]};
         }
         vec->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vec));
         ctx->allocated_vec.emplace(dst.id(), elems);
         return;
      }
   }

   Temp index = bld.as_uniform(get_ssa_temp(ctx, instr->src[0].ssa));
   if (offset != 0) // TODO check if index != 0 as well
      index = bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc), Operand(offset), index);
   Temp ptr = convert_pointer_to_64_bit(ctx, ctx->push_constants);

   aco_opcode op;
   RegClass rc;
   switch (dst.size()) {
   case 1:
      op = aco_opcode::s_load_dword;
      rc = s1;
      break;
   case 2:
      op = aco_opcode::s_load_dwordx2;
      rc = s2;
      break;
   case 3:
   case 4:
      op = aco_opcode::s_load_dwordx4;
      rc = s4;
      break;
   default:
      unreachable("unimplemented or forbidden load_push_constant.");
   }

   Temp vec = dst.size() == 3 ? bld.tmp(rc) : dst;
   bld.smem(op, Definition(vec), ptr, index);
   emit_split_vector(ctx, vec, vec.size());

   if (dst.size() == 3) {
      bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                 emit_extract_vector(ctx, vec, 0, s1),
                 emit_extract_vector(ctx, vec, 1, s1),
                 emit_extract_vector(ctx, vec, 2, s1));
   }
}

void visit_discard_if(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Temp src = as_divergent_bool(ctx, get_ssa_temp(ctx, instr->src[0].ssa), false);
   src = bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2));
   bld.pseudo(aco_opcode::p_discard_if, src);
   ctx->block->kind |= block_kind_uses_discard_if;
   return;
}

void visit_discard(isel_context* ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);

   /* it can currently happen that NIR doesn't remove the unreachable code */
   if (!nir_instr_is_last(&instr->instr)) {
      bld.pseudo(aco_opcode::p_discard_if, Operand(exec, s2));
      ctx->block->kind |= block_kind_uses_discard_if;
      return;
   }

   /* we handle discards the same way as jump instructions */
   append_logical_end(ctx->block);

   if (ctx->block->loop_nest_depth) {
      /* in loops, discard behaves like break */
      Block *linear_target = ctx->cf_info.parent_loop.exit;
      ctx->block->kind |= block_kind_discard;

      if (!ctx->cf_info.parent_if.is_divergent &&
          !ctx->cf_info.parent_loop.has_divergent_continue) {
         /* uniform discard - loop ends here */
         ctx->block->kind |= block_kind_uniform;
         ctx->cf_info.has_branch = true;
         bld.branch(aco_opcode::p_branch, linear_target);
         add_linear_edge(ctx->block, linear_target);
         return;
      }

      /* we add a break right behind the discard() instructions */
      ctx->block->kind |= block_kind_break;

      /* remove critical edges from linear CFG */
      Block* break_block = ctx->program->createAndInsertBlock();
      break_block->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      break_block->kind |= block_kind_uniform;
      add_linear_edge(ctx->block, break_block);
      Block* continue_block = ctx->program->createAndInsertBlock();
      continue_block->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      add_linear_edge(ctx->block, continue_block);
      bld.branch(aco_opcode::p_branch, break_block, continue_block);

      bld.reset(break_block);
      add_linear_edge(break_block, linear_target);
      bld.branch(aco_opcode::p_branch, linear_target);

      append_logical_start(continue_block);
      ctx->block = continue_block;

   } else {
      /* not inside loop */

      if (!ctx->cf_info.parent_if.is_divergent) {
         /* program just ends here */
         ctx->block->kind |= block_kind_uniform;
         ctx->cf_info.has_branch = true; /* not really, but doesn't need one */
         bld.exp(aco_opcode::exp, Operand(v1), Operand(v1), Operand(v1), Operand(v1),
                 0 /* enabled mask */, 9 /* dest */,
                 false /* compressed */, true/* done */, true /* valid mask */);
         bld.sopp(aco_opcode::s_endpgm);

      } else {
         ctx->block->kind |= block_kind_discard;
         /* branch and linear edge is added by visit_if() */

      }
   }

}

enum aco_descriptor_type {
   ACO_DESC_IMAGE,
   ACO_DESC_FMASK,
   ACO_DESC_SAMPLER,
   ACO_DESC_BUFFER,
};

enum aco_image_dim {
   aco_image_1d,
   aco_image_2d,
   aco_image_3d,
   aco_image_cube, // includes cube arrays
   aco_image_1darray,
   aco_image_2darray,
   aco_image_2dmsaa,
   aco_image_2darraymsaa,
};

static enum aco_image_dim
get_sampler_dim(isel_context *ctx, enum glsl_sampler_dim dim, bool is_array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      if (ctx->options->chip_class >= GFX9)
         return is_array ? aco_image_2darray : aco_image_2d;
      return is_array ? aco_image_1darray : aco_image_1d;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return is_array ? aco_image_2darray : aco_image_2d;
   case GLSL_SAMPLER_DIM_3D:
      return aco_image_3d;
   case GLSL_SAMPLER_DIM_CUBE:
      return aco_image_cube;
   case GLSL_SAMPLER_DIM_MS:
      return is_array ? aco_image_2darraymsaa : aco_image_2dmsaa;
   case GLSL_SAMPLER_DIM_SUBPASS:
      return aco_image_2darray;
   case GLSL_SAMPLER_DIM_SUBPASS_MS:
      return aco_image_2darraymsaa;
   default:
      unreachable("bad sampler dim");
   }
}

static bool
should_declare_array(isel_context *ctx, enum glsl_sampler_dim sampler_dim, bool is_array) {
   if (sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;
   aco_image_dim dim = get_sampler_dim(ctx, sampler_dim, is_array);
   return dim == aco_image_cube ||
          dim == aco_image_1darray ||
          dim == aco_image_2darray ||
          dim == aco_image_2darraymsaa;
}

Temp get_sampler_desc(isel_context *ctx, nir_deref_instr *deref_instr,
                      enum aco_descriptor_type desc_type,
                      const nir_tex_instr *tex_instr, bool image, bool write)
{
/* FIXME: we should lower the deref with some new nir_intrinsic_load_desc
   std::unordered_map<uint64_t, Temp>::iterator it = ctx->tex_desc.find((uint64_t) desc_type << 32 | deref_instr->dest.ssa.index);
   if (it != ctx->tex_desc.end())
      return it->second;
*/
   Temp index;
   bool index_set = false;
   unsigned constant_index = 0;
   unsigned descriptor_set;
   unsigned base_index;
   Builder bld(ctx->program, ctx->block);

   if (!deref_instr) {
      assert(tex_instr && !image);
      descriptor_set = 0;
      base_index = tex_instr->sampler_index;
   } else {
      while(deref_instr->deref_type != nir_deref_type_var) {
         unsigned array_size = glsl_get_aoa_size(deref_instr->type);
         if (!array_size)
            array_size = 1;

         assert(deref_instr->deref_type == nir_deref_type_array);
         nir_const_value *const_value = nir_src_as_const_value(deref_instr->arr.index);
         if (const_value) {
            constant_index += array_size * const_value->u32;
         } else {
            Temp indirect = bld.as_uniform(get_ssa_temp(ctx, deref_instr->arr.index.ssa));

            if (array_size != 1)
               indirect = bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), Operand(array_size), indirect);

            if (!index_set) {
               index = indirect;
               index_set = true;
            } else {
               index = bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc), index, indirect);
            }
         }

         deref_instr = nir_src_as_deref(deref_instr->parent);
      }
      descriptor_set = deref_instr->var->data.descriptor_set;
      base_index = deref_instr->var->data.binding;
   }

   Temp list = load_desc_ptr(ctx, descriptor_set);
   list = convert_pointer_to_64_bit(ctx, list);

   struct radv_descriptor_set_layout *layout = ctx->options->layout->set[descriptor_set].layout;
   struct radv_descriptor_set_binding_layout *binding = layout->binding + base_index;
   unsigned offset = binding->offset;
   unsigned stride = binding->size;
   aco_opcode opcode;
   RegClass type;

   assert(base_index < layout->binding_count);

   switch (desc_type) {
   case ACO_DESC_IMAGE:
      type = s8;
      opcode = aco_opcode::s_load_dwordx8;
      break;
   case ACO_DESC_FMASK:
      type = s8;
      opcode = aco_opcode::s_load_dwordx8;
      offset += 32;
      break;
   case ACO_DESC_SAMPLER:
      type = s4;
      opcode = aco_opcode::s_load_dwordx4;
      if (binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         offset += radv_combined_image_descriptor_sampler_offset(binding);
      break;
   case ACO_DESC_BUFFER:
      type = s4;
      opcode = aco_opcode::s_load_dwordx4;
      break;
   default:
      unreachable("invalid desc_type\n");
   }

   offset += constant_index * stride;

   if (desc_type == ACO_DESC_SAMPLER && binding->immutable_samplers_offset &&
      (!index_set || binding->immutable_samplers_equal)) {
      if (binding->immutable_samplers_equal)
         constant_index = 0;

      const uint32_t *samplers = radv_immutable_samplers(layout, binding);
      return bld.pseudo(aco_opcode::p_create_vector, bld.def(s4),
                        Operand(samplers[constant_index * 4 + 0]),
                        Operand(samplers[constant_index * 4 + 1]),
                        Operand(samplers[constant_index * 4 + 2]),
                        Operand(samplers[constant_index * 4 + 3]));
   }

   Operand off;
   if (!index_set) {
      off = Operand(offset);
   } else {
      off = Operand((Temp)bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc), Operand(offset),
                                   bld.sop2(aco_opcode::s_mul_i32, bld.def(s1), Operand(stride), index)));
   }

   return bld.smem(opcode, bld.def(type), list, off);
}

static int image_type_to_components_count(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
      return 1;
   case GLSL_SAMPLER_DIM_1D:
      return array ? 2 : 1;
   case GLSL_SAMPLER_DIM_2D:
      return array ? 3 : 2;
   case GLSL_SAMPLER_DIM_MS:
      return array ? 4 : 3;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      return 3;
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_SUBPASS:
      return 2;
   case GLSL_SAMPLER_DIM_SUBPASS_MS:
      return 3;
   default:
      break;
   }
   return 0;
}


/* Adjust the sample index according to FMASK.
 *
 * For uncompressed MSAA surfaces, FMASK should return 0x76543210,
 * which is the identity mapping. Each nibble says which physical sample
 * should be fetched to get that sample.
 *
 * For example, 0x11111100 means there are only 2 samples stored and
 * the second sample covers 3/4 of the pixel. When reading samples 0
 * and 1, return physical sample 0 (determined by the first two 0s
 * in FMASK), otherwise return physical sample 1.
 *
 * The sample index should be adjusted as follows:
 *   sample_index = (fmask >> (sample_index * 4)) & 0xF;
 */
static Temp adjust_sample_index_using_fmask(isel_context *ctx, bool da, Temp coords, Operand sample_index, Temp fmask_desc_ptr)
{
   Builder bld(ctx->program, ctx->block);
   Temp fmask = bld.tmp(v1);

   aco_ptr<MIMG_instruction> load{create_instruction<MIMG_instruction>(aco_opcode::image_load, Format::MIMG, 2, 1)};
   load->getOperand(0) = Operand(coords);
   load->getOperand(1) = Operand(fmask_desc_ptr);
   load->getDefinition(0) = Definition(fmask);
   load->glc = false;
   load->dmask = 0x1;
   load->unrm = true;
   load->da = da;
   load->can_reorder = true; /* fmask images shouldn't be modified */
   ctx->block->instructions.emplace_back(std::move(load));

   Operand sample_index4;
   if (sample_index.isConstant() && sample_index.constantValue() < 16) {
      sample_index4 = Operand(sample_index.constantValue() << 2);
   } else if (sample_index.regClass() == s1) {
      sample_index4 = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), sample_index, Operand(2u));
   } else {
      assert(sample_index.regClass() == v1);
      sample_index4 = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), Operand(2u), sample_index);
   }

   Temp final_sample;
   if (sample_index4.isConstant() && sample_index4.constantValue() == 0)
      final_sample = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(15u), fmask);
   else if (sample_index4.isConstant() && sample_index4.constantValue() == 28)
      final_sample = bld.vop2(aco_opcode::v_lshrrev_b32, bld.def(v1), Operand(28u), fmask);
   else
      final_sample = bld.vop3(aco_opcode::v_bfe_u32, bld.def(v1), fmask, sample_index4, Operand(4u));

   /* Don't rewrite the sample index if WORD1.DATA_FORMAT of the FMASK
    * resource descriptor is 0 (invalid),
    */
   Temp compare = bld.tmp(s2);
   bld.vopc_e64(aco_opcode::v_cmp_lg_u32, Definition(compare),
                Operand(0u), emit_extract_vector(ctx, fmask_desc_ptr, 1, s1)).def(0).setHint(vcc);

   Temp sample_index_v = bld.vop1(aco_opcode::v_mov_b32, bld.def(v1), sample_index);

   /* Replace the MSAA sample index. */
   return bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), sample_index_v, final_sample, compare);
}

static Temp get_image_coords(isel_context *ctx, const nir_intrinsic_instr *instr, const struct glsl_type *type)
{

   Temp src0 = get_ssa_temp(ctx, instr->src[1].ssa);
   enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   bool is_array = glsl_sampler_type_is_array(type);
   bool add_frag_pos = (dim == GLSL_SAMPLER_DIM_SUBPASS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   assert(!add_frag_pos && "Input attachments should be lowered.");
   bool is_ms = (dim == GLSL_SAMPLER_DIM_MS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   bool gfx9_1d = ctx->options->chip_class >= GFX9 && dim == GLSL_SAMPLER_DIM_1D;
   int count = image_type_to_components_count(dim, is_array);
   std::vector<Operand> coords(count);

   if (is_ms) {
      Operand sample_index;
      nir_const_value *sample_cv = nir_src_as_const_value(instr->src[2]);
      if (sample_cv)
         sample_index = Operand(sample_cv->u32);
      else
         sample_index = Operand(emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[2].ssa), 0, v1));

      if (instr->intrinsic == nir_intrinsic_image_deref_load) {
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, is_array ? 3 : 2, 1)};
         for (unsigned i = 0; i < vec->num_operands; i++)
            vec->getOperand(i) = Operand(emit_extract_vector(ctx, src0, i, v1));
         Temp fmask_load_address = {ctx->program->allocateId(), is_array ? v3 : v2};
         vec->getDefinition(0) = Definition(fmask_load_address);
         ctx->block->instructions.emplace_back(std::move(vec));

         Temp fmask_desc_ptr = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_FMASK, nullptr, false, false);
         sample_index = Operand(adjust_sample_index_using_fmask(ctx, is_array, fmask_load_address, sample_index, fmask_desc_ptr));
      }
      count--;
      coords[count] = sample_index;
   }

   if (count == 1 && !gfx9_1d)
      return emit_extract_vector(ctx, src0, 0, v1);

   if (gfx9_1d) {
      coords[0] = Operand(emit_extract_vector(ctx, src0, 0, v1));
      coords.resize(coords.size() + 1);
      coords[1] = Operand((uint32_t) 0);
      if (is_array)
         coords[2] = Operand(emit_extract_vector(ctx, src0, 1, v1));
   } else {
      for (int i = 0; i < count; i++)
         coords[i] = Operand(emit_extract_vector(ctx, src0, i, v1));
   }

   aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, coords.size(), 1)};
   for (unsigned i = 0; i < coords.size(); i++)
      vec->getOperand(i) = coords[i];
   Temp res = {ctx->program->allocateId(), getRegClass(vgpr, coords.size())};
   vec->getDefinition(0) = Definition(res);
   ctx->block->instructions.emplace_back(std::move(vec));
   return res;
}


void visit_image_load(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      unsigned mask = nir_ssa_def_components_read(&instr->dest.ssa);
      unsigned num_channels = util_last_bit(mask);
      Temp rsrc = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_BUFFER, nullptr, true, true);
      Temp vindex = emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[1].ssa), 0, v1);

      aco_opcode opcode;
      switch (num_channels) {
      case 1:
         opcode = aco_opcode::buffer_load_format_x;
         break;
      case 2:
         opcode = aco_opcode::buffer_load_format_xy;
         break;
      case 3:
         opcode = aco_opcode::buffer_load_format_xyz;
         break;
      case 4:
         opcode = aco_opcode::buffer_load_format_xyzw;
         break;
      default:
         unreachable(">4 channel buffer image load");
      }
      aco_ptr<MUBUF_instruction> load{create_instruction<MUBUF_instruction>(opcode, Format::MUBUF, 3, 1)};
      load->getOperand(0) = Operand(vindex);
      load->getOperand(1) = Operand(rsrc);
      load->getOperand(2) = Operand((uint32_t) 0);
      Temp tmp;
      if (num_channels == instr->dest.ssa.num_components && dst.type() == vgpr)
         tmp = dst;
      else
         tmp = {ctx->program->allocateId(), getRegClass(RegType::vgpr, num_channels)};
      load->getDefinition(0) = Definition(tmp);
      load->idxen = true;
      load->barrier = barrier_image;
      ctx->block->instructions.emplace_back(std::move(load));

      expand_vector(ctx, tmp, dst, instr->dest.ssa.num_components, (1 << num_channels) - 1);
      return;
   }

   Temp coords = get_image_coords(ctx, instr, type);
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);
   //aco_image_dim img_dim = get_image_dim(ctx, glsl_get_sampler_dim(type), glsl_sampler_type_is_array(type));

   unsigned dmask = nir_ssa_def_components_read(&instr->dest.ssa);
   unsigned num_components = util_bitcount(dmask);
   Temp tmp;
   if (num_components == instr->dest.ssa.num_components && dst.type() == vgpr)
      tmp = dst;
   else
      tmp = {ctx->program->allocateId(), getRegClass(RegType::vgpr, num_components)};

   aco_ptr<MIMG_instruction> load{create_instruction<MIMG_instruction>(aco_opcode::image_load, Format::MIMG, 2, 1)};
   load->getOperand(0) = Operand(coords);
   load->getOperand(1) = Operand(resource);
   load->getDefinition(0) = Definition(tmp);
   load->glc = var->data.image.access & (ACCESS_VOLATILE | ACCESS_COHERENT) ? 1 : 0;
   load->dmask = dmask;
   load->unrm = true;
   load->da = should_declare_array(ctx, dim, glsl_sampler_type_is_array(type));
   load->barrier = barrier_image;
   ctx->block->instructions.emplace_back(std::move(load));

   expand_vector(ctx, tmp, dst, instr->dest.ssa.num_components, dmask);
   return;
}

void visit_image_store(isel_context *ctx, nir_intrinsic_instr *instr)
{
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[3].ssa));

   bool glc = ctx->options->chip_class == SI || var->data.image.access & (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_READABLE) ? 1 : 0;

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      Temp rsrc = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_BUFFER, nullptr, true, true);
      Temp vindex = emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[1].ssa), 0, v1);
      aco_opcode opcode;
      switch (data.size()) {
      case 1:
         opcode = aco_opcode::buffer_store_format_x;
         break;
      case 2:
         opcode = aco_opcode::buffer_store_format_xy;
         break;
      case 3:
         opcode = aco_opcode::buffer_store_format_xyz;
         break;
      case 4:
         opcode = aco_opcode::buffer_store_format_xyzw;
         break;
      default:
         unreachable(">4 channel buffer image store");
      }
      aco_ptr<MUBUF_instruction> store{create_instruction<MUBUF_instruction>(opcode, Format::MUBUF, 4, 0)};
      store->getOperand(0) = Operand(vindex);
      store->getOperand(1) = Operand(rsrc);
      store->getOperand(2) = Operand((uint32_t) 0);
      store->getOperand(3) = Operand(data);
      store->idxen = true;
      store->glc = glc;
      store->disable_wqm = true;
      store->barrier = barrier_image;
      ctx->program->needs_exact = true;
      ctx->block->instructions.emplace_back(std::move(store));
      return;
   }

   assert(data.type() == vgpr);
   Temp coords = get_image_coords(ctx, instr, type);
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);

   aco_ptr<MIMG_instruction> store{create_instruction<MIMG_instruction>(aco_opcode::image_store, Format::MIMG, 4, 0)};
   store->getOperand(0) = Operand(coords);
   store->getOperand(1) = Operand(resource);
   store->getOperand(2) = Operand(s4);
   store->getOperand(3) = Operand(data);
   store->glc = glc;
   store->dmask = (1 << data.size()) - 1;
   store->unrm = true;
   store->da = should_declare_array(ctx, dim, glsl_sampler_type_is_array(type));
   store->disable_wqm = true;
   store->barrier = barrier_image;
   ctx->program->needs_exact = true;
   ctx->block->instructions.emplace_back(std::move(store));
   return;
}

void visit_image_atomic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   /* return the previous value if dest is ever used */
   bool return_previous = false;
   nir_foreach_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }
   nir_foreach_if_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }

   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   bool is_unsigned = glsl_get_sampler_result_type(type) == GLSL_TYPE_UINT;
   Builder bld(ctx->program, ctx->block);

   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[3].ssa));
   assert(data.size() == 1 && "64bit ssbo atomics not yet implemented.");

   if (instr->intrinsic == nir_intrinsic_image_deref_atomic_comp_swap)
      data = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2), get_ssa_temp(ctx, instr->src[4].ssa), data);

   aco_opcode buf_op, image_op;
   switch (instr->intrinsic) {
      case nir_intrinsic_image_deref_atomic_add:
         buf_op = aco_opcode::buffer_atomic_add;
         image_op = aco_opcode::image_atomic_add;
         break;
      case nir_intrinsic_image_deref_atomic_min:
         buf_op = aco_opcode::buffer_atomic_umin;
         image_op = is_unsigned ? aco_opcode::image_atomic_umin : aco_opcode::image_atomic_smin;
         break;
      case nir_intrinsic_image_deref_atomic_max:
         buf_op = aco_opcode::buffer_atomic_umax;
         image_op = is_unsigned ? aco_opcode::image_atomic_umax : aco_opcode::image_atomic_smax;
         break;
      case nir_intrinsic_image_deref_atomic_and:
         buf_op = aco_opcode::buffer_atomic_and;
         image_op = aco_opcode::image_atomic_and;
         break;
      case nir_intrinsic_image_deref_atomic_or:
         buf_op = aco_opcode::buffer_atomic_or;
         image_op = aco_opcode::image_atomic_or;
         break;
      case nir_intrinsic_image_deref_atomic_xor:
         buf_op = aco_opcode::buffer_atomic_xor;
         image_op = aco_opcode::image_atomic_xor;
         break;
      case nir_intrinsic_image_deref_atomic_exchange:
         buf_op = aco_opcode::buffer_atomic_swap;
         image_op = aco_opcode::image_atomic_swap;
         break;
      case nir_intrinsic_image_deref_atomic_comp_swap:
         buf_op = aco_opcode::buffer_atomic_cmpswap;
         image_op = aco_opcode::image_atomic_cmpswap;
         break;
      default:
         unreachable("visit_image_atomic should only be called with nir_intrinsic_image_deref_atomic_* instructions.");
   }

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      Temp vindex = emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[1].ssa), 0, v1);
      Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_BUFFER, nullptr, true, true);
      //assert(ctx->options->chip_class < GFX9 && "GFX9 stride size workaround not yet implemented.");
      aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(buf_op, Format::MUBUF, 4, return_previous ? 1 : 0)};
      mubuf->getOperand(0) = Operand(vindex);
      mubuf->getOperand(1) = Operand(resource);
      mubuf->getOperand(2) = Operand((uint32_t)0);
      mubuf->getOperand(3) = Operand(data);
      if (return_previous)
         mubuf->getDefinition(0) = Definition(dst);
      mubuf->offset = 0;
      mubuf->idxen = true;
      mubuf->glc = return_previous;
      mubuf->disable_wqm = true;
      mubuf->barrier = barrier_image;
      ctx->program->needs_exact = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));
      return;
   }

   Temp coords = get_image_coords(ctx, instr, type);
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);
   aco_ptr<MIMG_instruction> mimg{create_instruction<MIMG_instruction>(image_op, Format::MIMG, 4, return_previous ? 1 : 0)};
   mimg->getOperand(0) = Operand(coords);
   mimg->getOperand(1) = Operand(resource);
   mimg->getOperand(2) = Operand(s4); /* no sampler */
   mimg->getOperand(3) = Operand(data);
   if (return_previous)
      mimg->getDefinition(0) = Definition(dst);
   mimg->glc = return_previous;
   mimg->dmask = (1 << data.size()) - 1;
   mimg->unrm = true;
   mimg->da = should_declare_array(ctx, dim, glsl_sampler_type_is_array(type));
   mimg->disable_wqm = true;
   mimg->barrier = barrier_image;
   ctx->program->needs_exact = true;
   ctx->block->instructions.emplace_back(std::move(mimg));
   return;
}

void get_buffer_size(isel_context *ctx, Temp desc, Temp dst, bool in_elements)
{
   if (in_elements && ctx->options->chip_class == VI) {
      Builder bld(ctx->program, ctx->block);

      Temp stride = emit_extract_vector(ctx, desc, 1, s1);
      stride = bld.sop2(aco_opcode::s_bfe_u32, bld.def(s1), bld.def(s1, scc), stride, Operand((5u << 16) | 16u));
      stride = bld.vop1(aco_opcode::v_cvt_f32_ubyte0, bld.def(v1), stride);
      stride = bld.vop1(aco_opcode::v_rcp_iflag_f32, bld.def(v1), stride);

      Temp size = emit_extract_vector(ctx, desc, 2, s1);
      size = bld.vop1(aco_opcode::v_cvt_f32_u32, bld.def(v1), size);

      Temp res = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), size, stride);
      res = bld.vop1(aco_opcode::v_cvt_u32_f32, bld.def(v1), res);
      bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), res);

      // TODO: we can probably calculate this faster on the scalar unit to do: size / stride{1,2,4,8,12,16}
      /* idea
       * for 1,2,4,8,16, the result is just (stride >> S_FF1_I32_B32)
       * in case 12 (or 3?), we have to divide by 3:
       * set v_skip in case it's 12 (if we also have to take care of 3, shift first)
       * use v_mul_hi_u32 with magic number to divide
       * we need some pseudo merge opcode to overwrite the original SALU result with readfirstlane
       * disable v_skip
       * total: 6 SALU + 2 VALU instructions vs 1 SALU + 6 VALU instructions
       */

   } else {
      emit_extract_vector(ctx, desc, 2, dst);
   }
}

void visit_image_size(isel_context *ctx, nir_intrinsic_instr *instr)
{
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   Builder bld(ctx->program, ctx->block);

   if (glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF) {
      Temp desc = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_BUFFER, NULL, true, false);
      return get_buffer_size(ctx, desc, get_ssa_temp(ctx, &instr->dest.ssa), true);
   }

   /* LOD */
   Temp lod = bld.vop1(aco_opcode::v_mov_b32, bld.def(v1), Operand(0u));

   /* Resource */
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, NULL, true, false);

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_ptr<MIMG_instruction> mimg{create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1)};
   mimg->getOperand(0) = Operand(lod);
   mimg->getOperand(1) = Operand(resource);
   unsigned& dmask = mimg->dmask;
   mimg->dmask = (1 << instr->dest.ssa.num_components) - 1;
   mimg->da = glsl_sampler_type_is_array(type);
   mimg->can_reorder = true;
   Definition& def = mimg->getDefinition(0);
   ctx->block->instructions.emplace_back(std::move(mimg));

   if (glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_CUBE &&
       glsl_sampler_type_is_array(type)) {

      assert(instr->dest.ssa.num_components == 3);
      Temp tmp = {ctx->program->allocateId(), v3};
      def = Definition(tmp);
      emit_split_vector(ctx, tmp, 3);

      /* divide 3rd value by 6 by multiplying with magic number */
      Temp c = {ctx->program->allocateId(), s1};
      aco_ptr<Instruction> mov{create_s_mov(Definition(c), Operand((uint32_t) 0x2AAAAAAB))};
      ctx->block->instructions.emplace_back(std::move(mov));

      Temp by_6 = bld.vop3(aco_opcode::v_mul_hi_i32, bld.def(v1), emit_extract_vector(ctx, tmp, 2, v1), c);

      bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                 emit_extract_vector(ctx, tmp, 0, v1),
                 emit_extract_vector(ctx, tmp, 1, v1),
                 by_6);

   } else if (ctx->options->chip_class >= GFX9 &&
              glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_1D &&
              glsl_sampler_type_is_array(type)) {
      assert(instr->dest.ssa.num_components == 2);
      def = Definition(dst);
      dmask = 0x5;
   } else {
      def = Definition(dst);
   }

   emit_split_vector(ctx, dst, instr->dest.ssa.num_components);
}

void visit_load_ssbo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   unsigned num_components = instr->num_components;
   unsigned num_bytes = num_components * instr->dest.ssa.bit_size / 8;

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp rsrc = convert_pointer_to_64_bit(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
   rsrc = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), rsrc, Operand(0u));

   bool glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT);
   aco_opcode op;
   if (dst.type() == vgpr || (glc && ctx->options->chip_class < VI)) {
      Temp offset;
      if (ctx->options->chip_class < VI)
         offset = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
      else
         offset = get_ssa_temp(ctx, instr->src[1].ssa);

      switch (num_bytes) {
         case 4:
            op = aco_opcode::buffer_load_dword;
            break;
         case 8:
            op = aco_opcode::buffer_load_dwordx2;
            break;
         case 12:
            op = aco_opcode::buffer_load_dwordx3;
            break;
         case 16:
            op = aco_opcode::buffer_load_dwordx4;
            break;
         default:
            unreachable("Load SSBO not implemented for this size.");
      }
      aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(op, Format::MUBUF, 3, 1)};
      mubuf->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand(v1);
      mubuf->getOperand(1) = Operand(rsrc);
      mubuf->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
      mubuf->offen = (offset.type() == vgpr);
      mubuf->glc = glc;
      mubuf->barrier = barrier_buffer;

      if (dst.type() == sgpr) {
         Temp vec = bld.tmp(vgpr, dst.size());
         mubuf->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(mubuf));
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), vec);
      } else {
         mubuf->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(mubuf));
      }
   } else {
      switch (num_bytes) {
         case 4:
            op = aco_opcode::s_buffer_load_dword;
            break;
         case 8:
            op = aco_opcode::s_buffer_load_dwordx2;
            break;
         case 12:
         case 16:
            op = aco_opcode::s_buffer_load_dwordx4;
            break;
         default:
            unreachable("Load SSBO not implemented for this size.");
      }
      aco_ptr<SMEM_instruction> load{create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1)};
      load->getOperand(0) = Operand(rsrc);
      load->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      assert(load->getOperand(1).getTemp().type() == sgpr);
      load->getDefinition(0) = Definition(dst);
      load->glc = glc;
      load->barrier = barrier_buffer;
      assert(ctx->options->chip_class >= VI || !glc);

      if (dst.size() == 3) {
      /* trim vector */
         Temp vec = {ctx->program->allocateId(), s4};
         load->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(load));
         emit_split_vector(ctx, vec, 4);

         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    emit_extract_vector(ctx, vec, 0, s1),
                    emit_extract_vector(ctx, vec, 1, s1),
                    emit_extract_vector(ctx, vec, 2, s1));
      } else {
         ctx->block->instructions.emplace_back(std::move(load));
      }

   }
   emit_split_vector(ctx, dst, num_components);
}

void visit_store_ssbo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Temp data = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned elem_size_bytes = instr->src[0].ssa->bit_size / 8;
   unsigned writemask = nir_intrinsic_write_mask(instr);

   Temp offset;
   if (ctx->options->chip_class < VI)
      offset = as_vgpr(ctx,get_ssa_temp(ctx, instr->src[2].ssa));
   else
      offset = get_ssa_temp(ctx, instr->src[2].ssa);

   Temp rsrc = convert_pointer_to_64_bit(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   rsrc = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), rsrc, Operand(0u));

   bool smem = !ctx->divergent_vals[instr->src[2].ssa->index] &&
               ctx->options->chip_class >= VI;
   if (smem)
      offset = bld.as_uniform(offset);
   bool smem_nonfs = smem && ctx->stage != MESA_SHADER_FRAGMENT;

   while (writemask) {
      int start, count;
      u_bit_scan_consecutive_range(&writemask, &start, &count);
      if (count == 3 && smem) {
         writemask |= 1u << (start + 2);
         count = 2;
      }
      int num_bytes = count * elem_size_bytes;

      // TODO: we can only store 4 DWords at the same time. Fix for 64bit vectors
      // TODO: check alignment of sub-dword stores
      // TODO: split 3 bytes. there is no store instruction for that

      Temp write_data;
      if (count != instr->num_components) {
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, count, 1)};
         for (int i = 0; i < count; i++) {
            Temp elem = emit_extract_vector(ctx, data, start + i, getRegClass(data.type(), 1));
            vec->getOperand(i) = Operand(smem_nonfs ? bld.as_uniform(elem) : elem);
         }
         write_data = {ctx->program->allocateId(), getRegClass(smem_nonfs ? sgpr : data.type(), count)};
         vec->getDefinition(0) = Definition(write_data);
         ctx->block->instructions.emplace_back(std::move(vec));
      } else if (!smem && data.type() != vgpr) {
         assert(num_bytes % 4 == 0);
         write_data = {ctx->program->allocateId(), getRegClass(vgpr, num_bytes / 4)};
         emit_v_mov(ctx, data, write_data);
      } else if (smem_nonfs && data.type() == vgpr) {
         assert(num_bytes % 4 == 0);
         write_data = bld.as_uniform(data);
      } else {
         write_data = data;
      }

      aco_opcode vmem_op, smem_op;
      switch (num_bytes) {
         case 4:
            vmem_op = aco_opcode::buffer_store_dword;
            smem_op = aco_opcode::s_buffer_store_dword;
            break;
         case 8:
            vmem_op = aco_opcode::buffer_store_dwordx2;
            smem_op = aco_opcode::s_buffer_store_dwordx2;
            break;
         case 12:
            vmem_op = aco_opcode::buffer_store_dwordx3;
            smem_op = aco_opcode::last_opcode;
            assert(!smem);
            break;
         case 16:
            vmem_op = aco_opcode::buffer_store_dwordx4;
            smem_op = aco_opcode::s_buffer_store_dwordx4;
            break;
         default:
            unreachable("Store SSBO not implemented for this size.");
      }
      if (ctx->stage == MESA_SHADER_FRAGMENT)
         smem_op = aco_opcode::p_fs_buffer_store_smem;

      if (smem) {
         aco_ptr<SMEM_instruction> store{create_instruction<SMEM_instruction>(smem_op, Format::SMEM, 3, 0)};
         store->getOperand(0) = Operand(rsrc);
         if (start) {
            Temp off = bld.sop2(aco_opcode::s_add_i32, bld.def(s1), bld.def(s1, scc),
                                offset, Operand(start * elem_size_bytes));
            store->getOperand(1) = Operand(off);
         } else {
            store->getOperand(1) = Operand(offset);
         }
         if (smem_op != aco_opcode::p_fs_buffer_store_smem)
            store->getOperand(1).setFixed(m0);
         store->getOperand(2) = Operand(write_data);
         store->glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_READABLE);
         store->disable_wqm = true;
         store->barrier = barrier_buffer;
         ctx->block->instructions.emplace_back(std::move(store));
         ctx->program->wb_smem_l1_on_end = true;
         if (smem_op == aco_opcode::p_fs_buffer_store_smem) {
            ctx->block->kind |= block_kind_needs_lowering;
            ctx->program->needs_exact = true;
         }
      } else {
         aco_ptr<MUBUF_instruction> store{create_instruction<MUBUF_instruction>(vmem_op, Format::MUBUF, 4, 0)};
         store->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand(v1);
         store->getOperand(1) = Operand(rsrc);
         store->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
         store->getOperand(3) = Operand(write_data);
         store->offset = start * elem_size_bytes;
         store->offen = (offset.type() == vgpr);
         store->glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_READABLE);
         store->disable_wqm = true;
         store->barrier = barrier_buffer;
         ctx->program->needs_exact = true;
         ctx->block->instructions.emplace_back(std::move(store));
      }
   }
}

void visit_atomic_ssbo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   /* return the previous value if dest is ever used */
   bool return_previous = false;
   nir_foreach_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }
   nir_foreach_if_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }

   Builder bld(ctx->program, ctx->block);
   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[2].ssa));
   assert(data.size() == 1 && "64bit ssbo atomics not yet implemented.");

   if (instr->intrinsic == nir_intrinsic_ssbo_atomic_comp_swap)
      data = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2), get_ssa_temp(ctx, instr->src[3].ssa), data);

   Temp offset;
   if (ctx->options->chip_class < VI)
      offset = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   else
      offset = get_ssa_temp(ctx, instr->src[1].ssa);

   Temp rsrc = convert_pointer_to_64_bit(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
   rsrc = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), rsrc, Operand(0u));

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_opcode op;
   switch (instr->intrinsic) {
      case nir_intrinsic_ssbo_atomic_add:
         op = aco_opcode::buffer_atomic_add;
         break;
      case nir_intrinsic_ssbo_atomic_imin:
         op = aco_opcode::buffer_atomic_smin;
         break;
      case nir_intrinsic_ssbo_atomic_umin:
         op = aco_opcode::buffer_atomic_umin;
         break;
      case nir_intrinsic_ssbo_atomic_imax:
         op = aco_opcode::buffer_atomic_smax;
         break;
      case nir_intrinsic_ssbo_atomic_umax:
         op = aco_opcode::buffer_atomic_umax;
         break;
      case nir_intrinsic_ssbo_atomic_and:
         op = aco_opcode::buffer_atomic_and;
         break;
      case nir_intrinsic_ssbo_atomic_or:
         op = aco_opcode::buffer_atomic_or;
         break;
      case nir_intrinsic_ssbo_atomic_xor:
         op = aco_opcode::buffer_atomic_xor;
         break;
      case nir_intrinsic_ssbo_atomic_exchange:
         op = aco_opcode::buffer_atomic_swap;
         break;
      case nir_intrinsic_ssbo_atomic_comp_swap:
         op = aco_opcode::buffer_atomic_cmpswap;
         break;
      default:
         unreachable("visit_atomic_ssbo should only be called with nir_intrinsic_ssbo_atomic_* instructions.");
   }

   aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(op, Format::MUBUF, 4, return_previous ? 1 : 0)};
   mubuf->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand(v1);
   mubuf->getOperand(1) = Operand(rsrc);
   mubuf->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
   mubuf->getOperand(3) = Operand(data);
   if (return_previous)
      mubuf->getDefinition(0) = Definition(dst);
   mubuf->offset = 0;
   mubuf->offen = (offset.type() == vgpr);
   mubuf->glc = return_previous;
   mubuf->disable_wqm = true;
   mubuf->barrier = barrier_buffer;
   ctx->program->needs_exact = true;
   ctx->block->instructions.emplace_back(std::move(mubuf));
}

void visit_get_buffer_size(isel_context *ctx, nir_intrinsic_instr *instr) {

   Temp index = convert_pointer_to_64_bit(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
   Builder bld(ctx->program, ctx->block);
   Temp desc = bld.smem(aco_opcode::s_load_dwordx4, bld.def(s4), index, Operand(0u));
   get_buffer_size(ctx, desc, get_ssa_temp(ctx, &instr->dest.ssa), false);
}

void visit_load_global(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   unsigned num_components = instr->num_components;
   unsigned num_bytes = num_components * instr->dest.ssa.bit_size / 8;

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp addr = get_ssa_temp(ctx, instr->src[0].ssa);

   bool glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT);
   aco_opcode op;
   if (dst.type() == vgpr || (glc && ctx->options->chip_class < VI)) {
      bool global = ctx->options->chip_class >= GFX9;
      aco_opcode op;
      switch (num_bytes) {
      case 4:
         op = global ? aco_opcode::global_load_dword : aco_opcode::flat_load_dword;
         break;
      case 8:
         op = global ? aco_opcode::global_load_dwordx2 : aco_opcode::flat_load_dwordx2;
         break;
      case 12:
         op = global ? aco_opcode::global_load_dwordx3 : aco_opcode::flat_load_dwordx3;
         break;
      case 16:
         op = global ? aco_opcode::global_load_dwordx4 : aco_opcode::flat_load_dwordx4;
         break;
      default:
         unreachable("load_global not implemented for this size.");
      }
      aco_ptr<FLAT_instruction> flat{create_instruction<FLAT_instruction>(op, global ? Format::GLOBAL : Format::FLAT, 2, 1)};
      flat->getOperand(0) = Operand(addr);
      flat->getOperand(1) = Operand(s1);
      flat->glc = glc;

      if (dst.type() == sgpr) {
         Temp vec = bld.tmp(vgpr, dst.size());
         flat->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(flat));
         bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), vec);
      } else {
         flat->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(flat));
      }
      emit_split_vector(ctx, dst, num_components);
   } else {
      switch (num_bytes) {
         case 4:
            op = aco_opcode::s_load_dword;
            break;
         case 8:
            op = aco_opcode::s_load_dwordx2;
            break;
         case 12:
         case 16:
            op = aco_opcode::s_load_dwordx4;
            break;
         default:
            unreachable("load_global not implemented for this size.");
      }
      aco_ptr<SMEM_instruction> load{create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1)};
      load->getOperand(0) = Operand(addr);
      load->getOperand(1) = Operand(0u);
      load->getDefinition(0) = Definition(dst);
      load->glc = glc;
      load->barrier = barrier_buffer;
      assert(ctx->options->chip_class >= VI || !glc);

      if (dst.size() == 3) {
         /* trim vector */
         Temp vec = {ctx->program->allocateId(), s4};
         load->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(load));
         emit_split_vector(ctx, vec, 4);

         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    emit_extract_vector(ctx, vec, 0, s1),
                    emit_extract_vector(ctx, vec, 1, s1),
                    emit_extract_vector(ctx, vec, 2, s1));
      } else {
         ctx->block->instructions.emplace_back(std::move(load));
      }
   }
}

void visit_store_global(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   unsigned elem_size_bytes = instr->src[0].ssa->bit_size / 8;

   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
   Temp addr = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));

   unsigned writemask = nir_intrinsic_write_mask(instr);
   while (writemask) {
      int start, count;
      u_bit_scan_consecutive_range(&writemask, &start, &count);
      unsigned num_bytes = count * elem_size_bytes;

      Temp write_data = data;
      if (count != instr->num_components) {
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, count, 1)};
         for (int i = 0; i < count; i++)
            vec->getOperand(i) = Operand(emit_extract_vector(ctx, data, start + i, v1));
         write_data = {ctx->program->allocateId(), getRegClass(vgpr, count)};
         vec->getDefinition(0) = Definition(write_data);
         ctx->block->instructions.emplace_back(std::move(vec));
      }

      unsigned offset = start * elem_size_bytes;
      if (offset > 0 && ctx->options->chip_class < GFX9) {
         Temp addr0 = bld.tmp(v1), addr1 = bld.tmp(v1);
         Temp new_addr0 = bld.tmp(v1), new_addr1 = bld.tmp(v1);
         Temp carry = bld.tmp(s2);
         bld.pseudo(aco_opcode::p_split_vector, Definition(addr0), Definition(addr1), addr);

         bld.vop2(aco_opcode::v_add_co_u32, Definition(new_addr0), bld.hint_vcc(Definition(carry)),
                  Operand(offset), addr0);
         bld.vop2(aco_opcode::v_addc_co_u32, Definition(new_addr1), bld.def(s2),
                  Operand(0u), addr1,
                  carry).def(1).setHint(vcc);

         addr = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2), new_addr0, new_addr1);

         offset = 0;
      }

      bool glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_READABLE);
      bool global = ctx->options->chip_class >= GFX9;
      aco_opcode op;
      switch (num_bytes) {
      case 4:
         op = global ? aco_opcode::global_store_dword : aco_opcode::flat_store_dword;
         break;
      case 8:
         op = global ? aco_opcode::global_store_dwordx2 : aco_opcode::flat_store_dwordx2;
         break;
      case 12:
         op = global ? aco_opcode::global_store_dwordx3 : aco_opcode::flat_store_dwordx3;
         break;
      case 16:
         op = global ? aco_opcode::global_store_dwordx4 : aco_opcode::flat_store_dwordx4;
         break;
      default:
         unreachable("store_global not implemented for this size.");
      }
      aco_ptr<FLAT_instruction> flat{create_instruction<FLAT_instruction>(op, global ? Format::GLOBAL : Format::FLAT, 3, 0)};
      flat->getOperand(0) = Operand(addr);
      flat->getOperand(1) = Operand(s1);
      flat->getOperand(2) = Operand(data);
      flat->glc = glc;
      flat->offset = offset;
      ctx->block->instructions.emplace_back(std::move(flat));
   }
}

void emit_memory_barrier(isel_context *ctx, nir_intrinsic_instr *instr) {
   Builder bld(ctx->program, ctx->block);
   switch(instr->intrinsic) {
      case nir_intrinsic_group_memory_barrier:
      case nir_intrinsic_memory_barrier:
         bld.barrier(aco_opcode::p_memory_barrier_all);
         break;
      case nir_intrinsic_memory_barrier_atomic_counter:
         bld.barrier(aco_opcode::p_memory_barrier_atomic);
         break;
      case nir_intrinsic_memory_barrier_buffer:
         bld.barrier(aco_opcode::p_memory_barrier_buffer);
         break;
      case nir_intrinsic_memory_barrier_image:
         bld.barrier(aco_opcode::p_memory_barrier_image);
         break;
      case nir_intrinsic_memory_barrier_shared:
         bld.barrier(aco_opcode::p_memory_barrier_shared);
         break;
      default:
         unreachable("Unimplemented memory barrier intrinsic");
         break;
   }
}

Operand load_lds_size_m0(isel_context *ctx)
{
   if (ctx->options->chip_class >= GFX9) //m0 does not need to be initialized on GFX9+
      return Operand(m0, s1);
   Builder bld(ctx->program, ctx->block);
   return bld.m0((Temp)bld.sopk(aco_opcode::s_movk_i32, bld.def(s1, m0), 0xffff));
}


void visit_load_shared(isel_context *ctx, nir_intrinsic_instr *instr)
{
   // TODO: implement sparse reads using ds_read2_b32 and nir_ssa_def_components_read()
   Operand m = load_lds_size_m0(ctx);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   assert(instr->dest.ssa.bit_size == 32 && "Bitsize not supported in load_shared.");
   Temp address = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
   Builder bld(ctx->program, ctx->block);

   unsigned bytes_read = 0;
   unsigned result_size = 0;
   Temp result[instr->num_components];

   unsigned align = nir_intrinsic_align_mul(instr) ? nir_intrinsic_align(instr) : instr->dest.ssa.bit_size / 8;

   while (bytes_read < instr->num_components * 4) {
      unsigned todo = instr->num_components * 4 - bytes_read;
      bool aligned8 = bytes_read % 8 == 0 && align % 8 == 0;
      bool aligned16 = bytes_read % 16 == 0 && align % 16 == 0;

      aco_opcode op;
      unsigned size = 0;
      if (todo >= 16 && aligned16) {
         op = aco_opcode::ds_read_b128;
         size = 4;
      } else if (todo >= 12 && aligned16) {
         op = aco_opcode::ds_read_b96;
         size = 3;
      } else if (todo >= 8) {
         op = aligned8 ? aco_opcode::ds_read_b64 : aco_opcode::ds_read2_b32;
         size = 2;
      } else if (todo >= 4) {
         op = aco_opcode::ds_read_b32;
         size = 1;
      } else {
         assert(false);
      }

      unsigned offset = nir_intrinsic_base(instr) + bytes_read;
      unsigned max_offset = op == aco_opcode::ds_read2_b32 ? 1019 : 65535;
      Temp address_offset = address;
      if (offset > max_offset) {
         Temp new_addr{ctx->program->allocateId(), v1};
         emit_v_add32(ctx, new_addr, Operand((uint32_t) nir_intrinsic_base(instr)), Operand(address_offset));
         address_offset = new_addr;
         offset = bytes_read;
      }
      assert(offset <= max_offset); /* bytes_read shouldn't be large enough for this to happen */

      if (op == aco_opcode::ds_read2_b32)
         result[result_size++] = bld.ds(op, bld.def(getRegClass(vgpr, size)), address_offset, m, offset >> 2, (offset >> 2) + 1);
      else
         result[result_size++] = bld.ds(op, bld.def(getRegClass(vgpr, size)), address_offset, m, offset);
      bytes_read += size * 4;
   }

   aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, result_size, 1)};
   for (unsigned i = 0; i < result_size; i++)
      vec->getOperand(i) = Operand(result[i]);
   if (dst.type() == vgpr) {
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
   } else {
      Temp tmp{ctx->program->allocateId(), getRegClass(vgpr, dst.size())};
      vec->getDefinition(0) = Definition(tmp);
      ctx->block->instructions.emplace_back(std::move(vec));

      bld.pseudo(aco_opcode::p_as_uniform, Definition(dst), tmp);
   }
   emit_split_vector(ctx, dst, instr->num_components);

   return;
}

void ds_write_helper(isel_context *ctx, Operand m, Temp address, Temp data, unsigned offset0, unsigned offset1, unsigned align)
{
   Builder bld(ctx->program, ctx->block);
   unsigned bytes_written = 0;
   while (bytes_written < data.size() * 4) {
      unsigned todo = data.size() * 4 - bytes_written;
      bool aligned8 = bytes_written % 8 == 0 && align % 8 == 0;
      bool aligned16 = bytes_written % 16 == 0 && align % 16 == 0;

      aco_opcode op;
      unsigned size = 0;
      if (todo >= 16 && aligned16) {
         op = aco_opcode::ds_write_b128;
         size = 4;
      } else if (todo >= 12 && aligned16) {
         op = aco_opcode::ds_write_b96;
         size = 3;
      } else if (todo >= 8) {
         op = aligned8 ? aco_opcode::ds_write_b64 : aco_opcode::ds_write2_b32;
         size = 2;
      } else if (todo >= 4) {
         op = aco_opcode::ds_write_b32;
         size = 1;
      } else {
         assert(false);
      }

      bool write2 = op == aco_opcode::ds_write2_b32;
      unsigned offset = offset0 + offset1 + bytes_written;
      unsigned max_offset = write2 ? 1020 : 65535;
      Temp address_offset = address;
      if (offset > max_offset) {
         Temp new_addr{ctx->program->allocateId(), v1};
         emit_v_add32(ctx, new_addr, Operand(offset0), Operand(address_offset));
         address_offset = new_addr;
         offset = offset1 + bytes_written;
      }
      assert(offset <= max_offset); /* offset1 shouldn't be large enough for this to happen */

      if (write2) {
         Temp val0 = emit_extract_vector(ctx, data, bytes_written >> 2, v1);
         Temp val1 = emit_extract_vector(ctx, data, (bytes_written >> 2) + 1, v1);
         bld.ds(op, address_offset, val0, val1, m, offset >> 2, (offset >> 2) + 1);
      } else {
         Temp val = emit_extract_vector(ctx, data, bytes_written >> 2, getRegClass(vgpr, size));
         bld.ds(op, address_offset, val, m, offset);
      }

      bytes_written += size * 4;
   }
}

void visit_store_shared(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned offset = nir_intrinsic_base(instr);
   unsigned writemask = nir_intrinsic_write_mask(instr);
   Operand m = load_lds_size_m0(ctx);
   Temp data = get_ssa_temp(ctx, instr->src[0].ssa);
   Temp address = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   unsigned elem_size_bytes = instr->src[0].ssa->bit_size / 8;
   assert(elem_size_bytes == 4 && "Only 32bit store_shared currently supported.");

   /* we need at most two stores for 32bit variables */
   int start[2], count[2];
   u_bit_scan_consecutive_range(&writemask, &start[0], &count[0]);
   u_bit_scan_consecutive_range(&writemask, &start[1], &count[1]);
   assert(writemask == 0);

   /* one combined store is sufficient */
   if (count[0] == count[1]) {
      Temp address_offset = address;
      if ((offset >> 2) + start[1] > 255) {
         Temp new_addr{ctx->program->allocateId(), v1};
         emit_v_add32(ctx, new_addr, Operand(offset), Operand(address_offset));
         address_offset = new_addr;
         offset = 0;
      }

      assert(count[0] == 1);
      Temp val0 = emit_extract_vector(ctx, data, start[0], v1);
      Temp val1 = emit_extract_vector(ctx, data, start[1], v1);
      Builder bld(ctx->program, ctx->block);
      bld.ds(aco_opcode::ds_write2_b32, address_offset, val0, val1, m,
             (offset >> 2) + start[0], (offset >> 2) + start[1]);
      return;
   }

   unsigned align = nir_intrinsic_align_mul(instr) ? nir_intrinsic_align(instr) : elem_size_bytes;
   for (unsigned i = 0; i < 2; i++) {
      if (count[i] == 0)
         continue;

      Temp write_data = emit_extract_vector(ctx, data, start[i], getRegClass(vgpr, count[i]));
      ds_write_helper(ctx, m, address, write_data, offset, start[i] * elem_size_bytes, align);
   }
   return;
}

void visit_shared_atomic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned offset = nir_intrinsic_base(instr);
   Operand m = load_lds_size_m0(ctx);
   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   Temp address = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[0].ssa));

   unsigned num_operands = 3;
   aco_opcode op32, op64, op32_rtn, op64_rtn;
   switch(instr->intrinsic) {
      case nir_intrinsic_shared_atomic_add:
         op32 = aco_opcode::ds_add_u32;
         op64 = aco_opcode::ds_add_u64;
         op32_rtn = aco_opcode::ds_add_rtn_u32;
         op64_rtn = aco_opcode::ds_add_rtn_u64;
         break;
      case nir_intrinsic_shared_atomic_imin:
         op32 = aco_opcode::ds_min_i32;
         op64 = aco_opcode::ds_min_i64;
         op32_rtn = aco_opcode::ds_min_rtn_i32;
         op64_rtn = aco_opcode::ds_min_rtn_i64;
         break;
      case nir_intrinsic_shared_atomic_umin:
         op32 = aco_opcode::ds_min_u32;
         op64 = aco_opcode::ds_min_u64;
         op32_rtn = aco_opcode::ds_min_rtn_u32;
         op64_rtn = aco_opcode::ds_min_rtn_u64;
         break;
      case nir_intrinsic_shared_atomic_imax:
         op32 = aco_opcode::ds_max_i32;
         op64 = aco_opcode::ds_max_i64;
         op32_rtn = aco_opcode::ds_max_rtn_i32;
         op64_rtn = aco_opcode::ds_max_rtn_i64;
         break;
      case nir_intrinsic_shared_atomic_umax:
         op32 = aco_opcode::ds_max_u32;
         op64 = aco_opcode::ds_max_u64;
         op32_rtn = aco_opcode::ds_max_rtn_u32;
         op64_rtn = aco_opcode::ds_max_rtn_u64;
         break;
      case nir_intrinsic_shared_atomic_and:
         op32 = aco_opcode::ds_and_b32;
         op64 = aco_opcode::ds_and_b64;
         op32_rtn = aco_opcode::ds_and_rtn_b32;
         op64_rtn = aco_opcode::ds_and_rtn_b64;
         break;
      case nir_intrinsic_shared_atomic_or:
         op32 = aco_opcode::ds_or_b32;
         op64 = aco_opcode::ds_or_b64;
         op32_rtn = aco_opcode::ds_or_rtn_b32;
         op64_rtn = aco_opcode::ds_or_rtn_b64;
         break;
      case nir_intrinsic_shared_atomic_xor:
         op32 = aco_opcode::ds_xor_b32;
         op64 = aco_opcode::ds_xor_b64;
         op32_rtn = aco_opcode::ds_xor_rtn_b32;
         op64_rtn = aco_opcode::ds_xor_rtn_b64;
         break;
      case nir_intrinsic_shared_atomic_exchange:
         op32 = aco_opcode::ds_write_b32;
         op64 = aco_opcode::ds_write_b64;
         op32_rtn = aco_opcode::ds_wrxchg_rtn_b32;
         op64_rtn = aco_opcode::ds_wrxchg2_rtn_b64;
         break;
      case nir_intrinsic_shared_atomic_comp_swap:
         op32 = aco_opcode::ds_cmpst_b32;
         op64 = aco_opcode::ds_cmpst_b64;
         op32_rtn = aco_opcode::ds_cmpst_rtn_b32;
         op64_rtn = aco_opcode::ds_cmpst_rtn_b64;
         num_operands = 4;
         break;
      default:
         unreachable("Unhandled shared atomic intrinsic");
   }

   /* return the previous value if dest is ever used */
   bool return_previous = false;
   nir_foreach_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }
   nir_foreach_if_use_safe(use_src, &instr->dest.ssa) {
      return_previous = true;
      break;
   }

   aco_opcode op;
   if (data.size() == 1) {
      assert(instr->dest.ssa.bit_size == 32);
      op = return_previous ? op32_rtn : op32;
   } else {
      assert(instr->dest.ssa.bit_size == 64);
      op = return_previous ? op64_rtn : op64;
   }

   if (offset > 65535) {
      Temp new_addr{ctx->program->allocateId(), v1};
      emit_v_add32(ctx, new_addr, Operand(offset), Operand(address));
      address = new_addr;
      offset = 0;
   }

   aco_ptr<DS_instruction> ds;
   ds.reset(create_instruction<DS_instruction>(op, Format::DS, num_operands, return_previous ? 1 : 0));
   ds->getOperand(0) = Operand(address);
   ds->getOperand(1) = Operand(data);
   if (num_operands == 4)
      ds->getOperand(2) = Operand(get_ssa_temp(ctx, instr->src[2].ssa));
   ds->getOperand(num_operands - 1) = m;
   ds->offset0 = offset;
   if (return_previous)
      ds->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(ds));
}

void visit_load_sample_mask_in(isel_context *ctx, nir_intrinsic_instr *instr) {
   uint8_t log2_ps_iter_samples;
   if (ctx->program->info->info.ps.force_persample) {
      log2_ps_iter_samples =
         util_logbase2(ctx->options->key.fs.num_samples);
   } else {
      log2_ps_iter_samples = ctx->options->key.fs.log2_ps_iter_samples;
   }

   /* The bit pattern matches that used by fixed function fragment
    * processing. */
   static const unsigned ps_iter_masks[] = {
      0xffff, /* not used */
      0x5555,
      0x1111,
      0x0101,
      0x0001,
   };
   assert(log2_ps_iter_samples < ARRAY_SIZE(ps_iter_masks));

   Builder bld(ctx->program, ctx->block);

   Temp sample_id = bld.vop3(aco_opcode::v_bfe_u32, bld.def(v1), ctx->fs_inputs[fs_input::ancillary], Operand(8u), Operand(4u));
   Temp ps_iter_mask = bld.vop1(aco_opcode::v_mov_b32, bld.def(v1), Operand(ps_iter_masks[log2_ps_iter_samples]));
   Temp mask = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), sample_id, ps_iter_mask);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   bld.vop2(aco_opcode::v_and_b32, Definition(dst), mask, ctx->fs_inputs[fs_input::sample_coverage]);
}

Temp emit_boolean_reduce(isel_context *ctx, nir_op op, unsigned cluster_size, Temp src)
{
   Builder bld(ctx->program, ctx->block);

   if (cluster_size == 1) {
      return src;
   } if (op == nir_op_iand && cluster_size == 4) {
      //subgroupClusteredAnd(val, 4) -> ~wqm(exec & ~val)
      Temp tmp = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.def(s1, scc), Operand(exec, s2), src);
      return bld.sop1(aco_opcode::s_not_b64, bld.def(s2), bld.def(s1, scc),
                      bld.sop1(aco_opcode::s_wqm_b64, bld.def(s2), bld.def(s1, scc), tmp));
   } else if (op == nir_op_ior && cluster_size == 4) {
      //subgroupClusteredOr(val, 4) -> wqm(val & exec)
      return bld.sop1(aco_opcode::s_wqm_b64, bld.def(s2), bld.def(s1, scc),
                      bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2)));
   } else if (op == nir_op_iand && cluster_size == 64) {
      //subgroupAnd(val) -> (exec & ~val) == 0
      Temp tmp = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.def(s1, scc), Operand(exec, s2), src).def(1).getTemp();
      return bld.sopc(aco_opcode::s_cmp_eq_u32, bld.def(s1, scc), tmp, Operand(0u));
   } else if (op == nir_op_ior && cluster_size == 64) {
      //subgroupOr(val) -> (val & exec) != 0
      return bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2)).def(1).getTemp();
   } else if (op == nir_op_ixor && cluster_size == 64) {
      //subgroupXor(val) -> s_bcnt1_i32_b64(val & exec) & 1
      Temp tmp = bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2));
      tmp = bld.sop1(aco_opcode::s_bcnt1_i32_b64, bld.def(s2), bld.def(s1, scc), tmp);
      return bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), tmp, Operand(1u)).def(1).getTemp();
   } else {
      //subgroupClustered{And,Or,Xor}(val, n) ->
      //lane_id = v_mbcnt_hi_u32_b32(-1, v_mbcnt_lo_u32_b32(-1, 0))
      //cluster_offset = ~(n - 1) & lane_id
      //cluster_mask = ((1 << n) - 1)
      //subgroupClusteredAnd():
      //   return ((val | ~exec) >> cluster_offset) & cluster_mask == cluster_mask
      //subgroupClusteredOr():
      //   return ((val & exec) >> cluster_offset) & cluster_mask != 0
      //subgroupClusteredXor():
      //   return v_bnt_u32_b32(((val & exec) >> cluster_offset) & cluster_mask, 0) & 1 != 0
      Temp lane_id = bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32, bld.def(v1), Operand((uint32_t) -1),
                              bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, bld.def(v1), Operand((uint32_t) -1), Operand(0u)));
      Temp cluster_offset = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(~uint32_t(cluster_size - 1)), lane_id);

      Temp tmp;
      if (op == nir_op_iand)
         tmp = bld.sop2(aco_opcode::s_orn2_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2));
      else
         tmp = bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2));

      uint32_t cluster_mask = cluster_size == 32 ? -1 : (1u << cluster_size) - 1u;
      tmp = bld.vop3(aco_opcode::v_lshrrev_b64, bld.def(v2), cluster_offset, tmp);
      tmp = emit_extract_vector(ctx, tmp, 0, v1);
      if (cluster_mask != 0xffffffff)
         tmp = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(cluster_mask), tmp);

      Definition cmp_def = Definition();
      if (op == nir_op_iand) {
         cmp_def = bld.vopc(aco_opcode::v_cmp_eq_u32, bld.def(s2), Operand(cluster_mask), tmp).def(0);
      } else if (op == nir_op_ior) {
         cmp_def = bld.vopc(aco_opcode::v_cmp_lg_u32, bld.def(s2), Operand(0u), tmp).def(0);
      } else if (op == nir_op_ixor) {
         tmp = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(1u),
                        bld.vop3(aco_opcode::v_bcnt_u32_b32, bld.def(v1), tmp, Operand(0u)));
         cmp_def = bld.vopc(aco_opcode::v_cmp_lg_u32, bld.def(s2), Operand(0u), tmp).def(0);
      }
      cmp_def.setHint(vcc);
      return cmp_def.getTemp();
   }
}

Temp emit_boolean_exclusive_scan(isel_context *ctx, nir_op op, Temp src)
{
   Builder bld(ctx->program, ctx->block);

   //subgroupExclusiveAnd(val) -> mbcnt(exec & ~val) == 0
   //subgroupExclusiveOr(val) -> mbcnt(val & exec) != 0
   //subgroupExclusiveXor(val) -> mbcnt(val & exec) & 1 != 0
   Temp tmp;
   if (op == nir_op_iand)
      tmp = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.def(s1, scc), Operand(exec, s2), src);
   else
      tmp = bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2));

   Builder::Result lohi = bld.pseudo(aco_opcode::p_split_vector, bld.def(s1), bld.def(s1), tmp);
   Temp lo = lohi.def(0).getTemp();
   Temp hi = lohi.def(1).getTemp();
   Temp mbcnt = bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32, bld.def(v1), hi,
                         bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, bld.def(v1), lo, Operand(0u)));

   Definition cmp_def = Definition();
   if (op == nir_op_iand)
      cmp_def = bld.vopc(aco_opcode::v_cmp_eq_u32, bld.def(s2), Operand(0u), mbcnt).def(0);
   else if (op == nir_op_ior)
      cmp_def = bld.vopc(aco_opcode::v_cmp_lg_u32, bld.def(s2), Operand(0u), mbcnt).def(0);
   else if (op == nir_op_ixor)
      cmp_def = bld.vopc(aco_opcode::v_cmp_lg_u32, bld.def(s2), Operand(0u),
                         bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(1u), mbcnt)).def(0);
   cmp_def.setHint(vcc);
   return cmp_def.getTemp();
}

Temp emit_boolean_inclusive_scan(isel_context *ctx, nir_op op, Temp src)
{
   Builder bld(ctx->program, ctx->block);

   //subgroupInclusiveAnd(val) -> subgroupExclusiveAnd(val) && val
   //subgroupInclusiveOr(val) -> subgroupExclusiveOr(val) || val
   //subgroupInclusiveXor(val) -> subgroupExclusiveXor(val) ^^ val
   Temp tmp = emit_boolean_exclusive_scan(ctx, op, src);
   if (op == nir_op_iand)
      return bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), tmp, src);
   else if (op == nir_op_ior)
      return bld.sop2(aco_opcode::s_or_b64, bld.def(s2), bld.def(s1, scc), tmp, src);
   else if (op == nir_op_ixor)
      return bld.sop2(aco_opcode::s_xor_b64, bld.def(s2), bld.def(s1, scc), tmp, src);

   assert(false);
   return Temp();
}

void emit_uniform_subgroup(isel_context *ctx, nir_intrinsic_instr *instr, Temp src)
{
   Builder bld(ctx->program, ctx->block);
   Definition dst(get_ssa_temp(ctx, &instr->dest.ssa));
   if (typeOf(src.regClass()) == vgpr) {
      bld.pseudo(aco_opcode::p_as_uniform, dst, src);
   } else if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
      bld.sopc(aco_opcode::s_cmp_lg_u64, bld.scc(dst), Operand(0u), Operand(src));
   } else if (src.regClass() == s1) {
      bld.sop1(aco_opcode::s_mov_b32, dst, src);
   } else if (src.regClass() == s2) {
      bld.sop1(aco_opcode::s_mov_b64, dst, src);
   } else {
      fprintf(stderr, "Unimplemented NIR instr bit size: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
   }
}

void emit_interp_center(isel_context *ctx, Temp dst, Temp pos1, Temp pos2)
{
   Builder bld(ctx->program, ctx->block);
   Temp p1 = ctx->fs_inputs[fs_input::persp_center_p1];
   Temp p2 = ctx->fs_inputs[fs_input::persp_center_p2];

   /* Build DD X/Y */
   Temp tl_1 = bld.vop1_dpp(aco_opcode::v_mov_b32, bld.def(v1), p1, dpp_quad_perm(0, 0, 0, 0));
   Temp ddx_1 = bld.vop2_dpp(aco_opcode::v_sub_f32, bld.def(v1), p1, tl_1, dpp_quad_perm(1, 1, 1, 1));
   Temp ddy_1 = bld.vop2_dpp(aco_opcode::v_sub_f32, bld.def(v1), p1, tl_1, dpp_quad_perm(2, 2, 2, 2));
   Temp tl_2 = bld.vop1_dpp(aco_opcode::v_mov_b32, bld.def(v1), p2, dpp_quad_perm(0, 0, 0, 0));
   Temp ddx_2 = bld.vop2_dpp(aco_opcode::v_sub_f32, bld.def(v1), p2, tl_2, dpp_quad_perm(1, 1, 1, 1));
   Temp ddy_2 = bld.vop2_dpp(aco_opcode::v_sub_f32, bld.def(v1), p2, tl_2, dpp_quad_perm(2, 2, 2, 2));

   /* res_k = p_k + ddx_k * pos1 + ddy_k * pos2 */
   Temp tmp1 = bld.vop3(aco_opcode::v_mad_f32, bld.def(v1), ddx_1, pos1, p1);
   Temp tmp2 = bld.vop3(aco_opcode::v_mad_f32, bld.def(v1), ddx_2, pos1, p2);
   tmp1 = bld.vop3(aco_opcode::v_mad_f32, bld.def(v1), ddy_1, pos2, tmp1);
   tmp2 = bld.vop3(aco_opcode::v_mad_f32, bld.def(v1), ddy_2, pos2, tmp2);
   Temp wqm1 = bld.tmp(v1);
   emit_wqm(ctx, tmp1, wqm1);
   Temp wqm2 = bld.tmp(v1);
   emit_wqm(ctx, tmp2, wqm2);
   bld.pseudo(aco_opcode::p_create_vector, Definition(dst), wqm1, wqm2);
   return;
}

void visit_intrinsic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   switch(instr->intrinsic) {
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid: {
      glsl_interp_mode mode = (glsl_interp_mode)nir_intrinsic_interp_mode(instr);
      fs_input input = get_interp_input(instr->intrinsic, mode);

      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      if (input == fs_input::max_inputs) {
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    Operand(0u), Operand(0u));
      } else {
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst),
                    ctx->fs_inputs[input],
                    ctx->fs_inputs[input + 1]);
      }
      emit_split_vector(ctx, dst, 2);
      break;
   }
   case nir_intrinsic_load_barycentric_at_sample: {
      uint32_t sample_pos_offset = RING_PS_SAMPLE_POSITIONS * 16;
      switch (ctx->options->key.fs.num_samples) {
         case 2: sample_pos_offset += 1 << 3; break;
         case 4: sample_pos_offset += 3 << 3; break;
         case 8: sample_pos_offset += 7 << 3; break;
         default: break;
      }
      Temp sample_pos;
      Temp addr = get_ssa_temp(ctx, instr->src[0].ssa);
      nir_const_value* const_addr = nir_src_as_const_value(instr->src[0]);
      if (addr.type() == sgpr) {
         Operand offset;
         if (const_addr) {
            sample_pos_offset += const_addr->u32 << 3;
            offset = Operand(sample_pos_offset);
         } else if (ctx->options->chip_class >= GFX9) {
            offset = bld.sop2(aco_opcode::s_lshl3_add_u32, bld.def(s1), bld.def(s1, scc), addr, Operand(sample_pos_offset));
         } else {
            offset = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), addr, Operand(3u));
            offset = bld.sop2(aco_opcode::s_add_u32, bld.def(s1), bld.def(s1, scc), addr, Operand(sample_pos_offset));
         }
         addr = ctx->ring_offsets;
         sample_pos = bld.smem(aco_opcode::s_load_dwordx2, bld.def(s2), addr, Operand(offset));

      } else if (ctx->options->chip_class >= GFX9) {
         addr = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), Operand(3u), addr);
         sample_pos = bld.global(aco_opcode::global_load_dwordx2, bld.def(v2), addr, ctx->ring_offsets, sample_pos_offset);
      } else {
         /* addr += ctx->ring_offsets + sample_pos_offset */
         Temp tmp0 = bld.tmp(s1);
         Temp tmp1 = bld.tmp(s1);
         bld.pseudo(aco_opcode::p_split_vector, Definition(tmp0), Definition(tmp1), ctx->ring_offsets);
         Definition scc_tmp = bld.def(s1, scc);
         tmp0 = bld.sop2(aco_opcode::s_add_u32, bld.def(s1), scc_tmp, tmp0, Operand(sample_pos_offset));
         tmp1 = bld.sop2(aco_opcode::s_addc_u32, bld.def(s1), bld.def(s1, scc), tmp1, Operand(0u), scc_tmp.getTemp());
         addr = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), Operand(3u), addr);
         Temp pck0 = bld.tmp(v1);
         Temp carry = emit_v_add32(ctx, pck0, Operand(tmp0), Operand(addr), true);
         tmp1 = as_vgpr(ctx, tmp1);
         Temp pck1 = bld.vop2_e64(aco_opcode::v_addc_co_u32, bld.def(v1), bld.hint_vcc(bld.def(s2)), tmp1, Operand(0u), carry);
         addr = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2), pck0, pck1);

         /* sample_pos = flat_load_dwordx2 addr */
         sample_pos = bld.flat(aco_opcode::flat_load_dwordx2, bld.def(v2), addr, Operand(s1));
      }

      /* sample_pos -= 0.5 */
      Temp pos1 = bld.tmp(getRegClass(sample_pos.type(), 1));
      Temp pos2 = bld.tmp(getRegClass(sample_pos.type(), 1));
      bld.pseudo(aco_opcode::p_split_vector, Definition(pos1), Definition(pos2), sample_pos);
      pos1 = bld.vop2_e64(aco_opcode::v_sub_f32, bld.def(v1), pos1, Operand(0x3f000000u));
      pos2 = bld.vop2_e64(aco_opcode::v_sub_f32, bld.def(v1), pos2, Operand(0x3f000000u));

      emit_interp_center(ctx, get_ssa_temp(ctx, &instr->dest.ssa), pos1, pos2);
      break;
   }
   case nir_intrinsic_load_barycentric_at_offset: {
      Temp offset = get_ssa_temp(ctx, instr->src[0].ssa);
      emit_split_vector(ctx, offset, 2);
      RegClass rc = getRegClass(offset.type(), 1);
      Temp pos1 = emit_extract_vector(ctx, offset, 0, rc);
      Temp pos2 = emit_extract_vector(ctx, offset, 1, rc);
      emit_interp_center(ctx, get_ssa_temp(ctx, &instr->dest.ssa), pos1, pos2);
      break;
   }
   case nir_intrinsic_load_front_face: {
      bld.vopc(aco_opcode::v_cmp_lg_u32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)),
               Operand(0u), ctx->fs_inputs[fs_input::front_face]).def(0).setHint(vcc);
      break;
   }
   case nir_intrinsic_load_view_index:
   case nir_intrinsic_load_layer_id: {
      unsigned base = VARYING_SLOT_LAYER / 4;
      unsigned idx = util_bitcount64(ctx->input_mask & ((1ull << base) - 1ull));
      Operand P0;
      P0.setFixed(PhysReg{2});
      bld.vintrp(aco_opcode::v_interp_mov_f32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)),
                 P0, bld.m0(ctx->prim_mask), idx, 0);
      break;
   }
   case nir_intrinsic_load_frag_coord: {
      emit_load_frag_coord(ctx, get_ssa_temp(ctx, &instr->dest.ssa), 4);
      break;
   }
   case nir_intrinsic_load_sample_pos: {
      bld.pseudo(aco_opcode::p_create_vector, Definition(get_ssa_temp(ctx, &instr->dest.ssa)),
                 bld.vop1(aco_opcode::v_fract_f32, bld.def(v1), ctx->fs_inputs[fs_input::frag_pos_0]),
                 bld.vop1(aco_opcode::v_fract_f32, bld.def(v1), ctx->fs_inputs[fs_input::frag_pos_1]));
      break;
   }
   case nir_intrinsic_load_interpolated_input:
      visit_load_interpolated_input(ctx, instr);
      break;
   case nir_intrinsic_store_output:
      visit_store_output(ctx, instr);
      break;
   case nir_intrinsic_load_input:
      visit_load_input(ctx, instr);
      break;
   case nir_intrinsic_load_ubo:
      visit_load_ubo(ctx, instr);
      break;
   case nir_intrinsic_load_push_constant:
      visit_load_push_constant(ctx, instr);
      break;
   case nir_intrinsic_vulkan_resource_index:
      visit_load_resource(ctx, instr);
      break;
   case nir_intrinsic_discard:
      visit_discard(ctx, instr);
      break;
   case nir_intrinsic_discard_if:
      visit_discard_if(ctx, instr);
      break;
   case nir_intrinsic_load_shared:
      visit_load_shared(ctx, instr);
      break;
   case nir_intrinsic_store_shared:
      visit_store_shared(ctx, instr);
      break;
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
      visit_shared_atomic(ctx, instr);
      break;
   case nir_intrinsic_image_deref_load:
      visit_image_load(ctx, instr);
      break;
   case nir_intrinsic_image_deref_store:
      visit_image_store(ctx, instr);
      break;
   case nir_intrinsic_image_deref_atomic_add:
   case nir_intrinsic_image_deref_atomic_min:
   case nir_intrinsic_image_deref_atomic_max:
   case nir_intrinsic_image_deref_atomic_and:
   case nir_intrinsic_image_deref_atomic_or:
   case nir_intrinsic_image_deref_atomic_xor:
   case nir_intrinsic_image_deref_atomic_exchange:
   case nir_intrinsic_image_deref_atomic_comp_swap:
      visit_image_atomic(ctx, instr);
      break;
   case nir_intrinsic_image_deref_size:
      visit_image_size(ctx, instr);
      break;
   case nir_intrinsic_load_ssbo:
      visit_load_ssbo(ctx, instr);
      break;
   case nir_intrinsic_store_ssbo:
      visit_store_ssbo(ctx, instr);
      break;
   case nir_intrinsic_load_global:
      visit_load_global(ctx, instr);
      break;
   case nir_intrinsic_store_global:
      visit_store_global(ctx, instr);
      break;
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
      visit_atomic_ssbo(ctx, instr);
      break;
   case nir_intrinsic_get_buffer_size:
      visit_get_buffer_size(ctx, instr);
      break;
   case nir_intrinsic_barrier: {
      unsigned* bsize = ctx->program->info->cs.block_size;
      unsigned workgroup_size = bsize[0] * bsize[1] * bsize[2];
      if (workgroup_size > 64)
         bld.sopp(aco_opcode::s_barrier);
      break;
   }
   case nir_intrinsic_group_memory_barrier:
   case nir_intrinsic_memory_barrier:
   case nir_intrinsic_memory_barrier_atomic_counter:
   case nir_intrinsic_memory_barrier_buffer:
   case nir_intrinsic_memory_barrier_image:
   case nir_intrinsic_memory_barrier_shared:
      emit_memory_barrier(ctx, instr);
      break;
   case nir_intrinsic_load_num_work_groups:
   case nir_intrinsic_load_work_group_id:
   case nir_intrinsic_load_local_invocation_id: {
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      Temp* ids;
      if (instr->intrinsic == nir_intrinsic_load_num_work_groups)
         ids = ctx->num_workgroups;
      else if (instr->intrinsic == nir_intrinsic_load_work_group_id)
         ids = ctx->workgroup_ids;
      else
         ids = ctx->local_invocation_ids;
      bld.pseudo(aco_opcode::p_create_vector, Definition(dst), ids[0], ids[1], ids[2]);
      emit_split_vector(ctx, dst, 3);
      break;
   }
   case nir_intrinsic_load_local_invocation_index: {
      Temp id = bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32, bld.def(v1), Operand((uint32_t) -1),
                         bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, bld.def(v1), Operand((uint32_t) -1), Operand(0u)));
      Temp tg_num = bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), Operand(0xfc0u), ctx->tg_size);
      bld.vop2(aco_opcode::v_or_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), tg_num, id);
      break;
   }
   case nir_intrinsic_load_subgroup_id: {
      if (ctx->stage == MESA_SHADER_COMPUTE) {
         Temp tg_num = bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), Operand(0xfc0u), ctx->tg_size);
         bld.sop2(aco_opcode::s_lshr_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), bld.def(s1, scc), tg_num, Operand(0x6u));
      } else {
         bld.sop1(aco_opcode::s_mov_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), Operand(0x0u));
      }
      break;
   }
   case nir_intrinsic_load_subgroup_invocation: {
      bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), Operand((uint32_t) -1),
               bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, bld.def(v1), Operand((uint32_t) -1), Operand(0u)));
      break;
   }
   case nir_intrinsic_load_num_subgroups: {
      if (ctx->stage == MESA_SHADER_COMPUTE)
         bld.sop2(aco_opcode::s_and_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), bld.def(s1, scc), Operand(0x3fu), ctx->tg_size);
      else
         bld.sop1(aco_opcode::s_mov_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)), Operand(0x1u));
      break;
   }
   case nir_intrinsic_ballot: {
      Definition tmp = bld.def(s2);
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      if (instr->src[0].ssa->bit_size == 1 && src.regClass() == s2) {
         bld.sop2(aco_opcode::s_and_b64, tmp, Operand(exec, s2), src);
      } else if (instr->src[0].ssa->bit_size == 1 && src.regClass() == s1) {
         bld.sop2(aco_opcode::s_cselect_b64, tmp, Operand(exec, s2), Operand(0u), bld.scc(src));
      } else if (instr->src[0].ssa->bit_size == 32 && src.regClass() == v1) {
         bld.vopc(aco_opcode::v_cmp_lg_u32, tmp, Operand(0u), src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp.getTemp(), get_ssa_temp(ctx, &instr->dest.ssa));
      break;
   }
   case nir_intrinsic_shuffle: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      if (!ctx->divergent_vals[instr->dest.ssa.index]) {
         emit_uniform_subgroup(ctx, instr, src);
      } else {
         Temp tid = get_ssa_temp(ctx, instr->src[1].ssa);
         assert(tid.regClass() == v1);
         Definition dst = bld.def(src.regClass());
         if (src.regClass() == v1) {
            tid = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), Operand(2u), tid);
            bld.ds(aco_opcode::ds_bpermute_b32, dst, tid, src);
         } else if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
            Temp tmp = bld.vop3(aco_opcode::v_lshrrev_b64, bld.def(v2), tid, src);
            tmp = emit_extract_vector(ctx, tmp, 0, v1);
            tmp = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(1u), tmp);
            bld.vopc(aco_opcode::v_cmp_lg_u32, dst, Operand(0u), tmp);
         } else {
            fprintf(stderr, "Unimplemented NIR instr bit size: ");
            nir_print_instr(&instr->instr, stderr);
            fprintf(stderr, "\n");
         }
         emit_wqm(ctx, dst.getTemp(), get_ssa_temp(ctx, &instr->dest.ssa));
      }
      break;
   }
   case nir_intrinsic_load_sample_id: {
      bld.vop3(aco_opcode::v_bfe_u32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)),
               ctx->fs_inputs[ancillary], Operand(8u), Operand(4u));
      break;
   }
   case nir_intrinsic_load_sample_mask_in: {
      visit_load_sample_mask_in(ctx, instr);
      break;
   }
   case nir_intrinsic_read_first_invocation: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      Definition tmp = bld.def(s1);
      if (src.regClass() == v1) {
         bld.vop1(aco_opcode::v_readfirstlane_b32, tmp, src);
      } else if (src.regClass() == s1) {
         bld.sop1(aco_opcode::s_mov_b32, tmp, src);
      } else if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
         bld.sopc(aco_opcode::s_bitcmp1_b64, bld.scc(tmp), src,
                  bld.sop1(aco_opcode::s_ff1_i32_b64, bld.def(s1), Operand(exec, s2)));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp.getTemp(), get_ssa_temp(ctx, &instr->dest.ssa));
      break;
   }
   case nir_intrinsic_read_invocation: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      Temp lane = get_ssa_temp(ctx, instr->src[1].ssa);
      assert(lane.regClass() == s1);
      Definition tmp = bld.def(s1);
      if (src.regClass() == v1) {
         bld.vop3(aco_opcode::v_readlane_b32, tmp, src, lane);
      } else if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
         bld.sopc(aco_opcode::s_bitcmp1_b64, bld.scc(tmp), src, lane);
      } else if (src.regClass() == s1) {
         bld.sop1(aco_opcode::s_mov_b32, tmp, src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp.getTemp(), get_ssa_temp(ctx, &instr->dest.ssa));
      break;
   }
   case nir_intrinsic_vote_all: {
      Temp src = as_divergent_bool(ctx, get_ssa_temp(ctx, instr->src[0].ssa), false);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      assert(src.regClass() == s2);
      assert(dst.regClass() == s1);

      Definition tmp = bld.def(s1);
      bld.sopc(aco_opcode::s_cmp_eq_u64, bld.scc(tmp),
               bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.def(s1, scc), src, Operand(exec, s2)),
               Operand(exec, s2));
      emit_wqm(ctx, tmp.getTemp(), dst);
      break;
   }
   case nir_intrinsic_vote_any: {
      Temp src = as_divergent_bool(ctx, get_ssa_temp(ctx, instr->src[0].ssa), false);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      assert(src.regClass() == s2);
      assert(dst.regClass() == s1);

      Definition tmp = bld.def(s1);
      bld.sop2(aco_opcode::s_and_b64, bld.def(s2), bld.scc(tmp), src, Operand(exec, s2));
      emit_wqm(ctx, tmp.getTemp(), dst);
      break;
   }
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      nir_op op = (nir_op) nir_intrinsic_reduction_op(instr);
      unsigned cluster_size = instr->intrinsic == nir_intrinsic_reduce ?
         nir_intrinsic_cluster_size(instr) : 0;
      cluster_size = util_next_power_of_two(MIN2(cluster_size ? cluster_size : 64, 64));

      if (!ctx->divergent_vals[instr->src[0].ssa->index] && (op == nir_op_ior || op == nir_op_iand)) {
         emit_uniform_subgroup(ctx, instr, src);
      } else if (instr->dest.ssa.bit_size == 1) {
         if (op == nir_op_imul || op == nir_op_umin || op == nir_op_imin)
            op = nir_op_iand;
         else if (op == nir_op_iadd)
            op = nir_op_ixor;
         else if (op == nir_op_umax || op == nir_op_imax)
            op = nir_op_ior;
         assert(op == nir_op_iand || op == nir_op_ior || op == nir_op_ixor);

         switch (instr->intrinsic) {
         case nir_intrinsic_reduce:
            emit_wqm(ctx, emit_boolean_reduce(ctx, op, cluster_size, src), dst);
            break;
         case nir_intrinsic_exclusive_scan:
            emit_wqm(ctx, emit_boolean_exclusive_scan(ctx, op, src), dst);
            break;
         case nir_intrinsic_inclusive_scan:
            emit_wqm(ctx, emit_boolean_inclusive_scan(ctx, op, src), dst);
            break;
         default:
            assert(false);
         }
      } else if (cluster_size == 1) {
         emit_v_mov(ctx, src, dst);
      } else {
         src = as_vgpr(ctx, src);

         ReduceOp reduce_op;
         switch (op) {
         #define CASE(name) case nir_op_##name: reduce_op = (src.regClass() == v1) ? name##32 : name##64; break;
            CASE(iadd)
            CASE(imul)
            CASE(fadd)
            CASE(fmul)
            CASE(imin)
            CASE(umin)
            CASE(fmin)
            CASE(imax)
            CASE(umax)
            CASE(fmax)
            CASE(iand)
            CASE(ior)
            CASE(ixor)
            default:
               unreachable("unknown reduction op");
         #undef CASE
         }

         aco_opcode aco_op;
         switch (instr->intrinsic) {
            case nir_intrinsic_reduce: aco_op = aco_opcode::p_reduce; break;
            case nir_intrinsic_inclusive_scan: aco_op = aco_opcode::p_inclusive_scan; break;
            case nir_intrinsic_exclusive_scan: aco_op = aco_opcode::p_exclusive_scan; break;
            default:
               unreachable("unknown reduce intrinsic");
         }

         aco_ptr<Pseudo_reduction_instruction> reduce{create_instruction<Pseudo_reduction_instruction>(aco_op, Format::PSEUDO_REDUCTION, 3, 5)};
         reduce->getOperand(0) = Operand(src);
         // filled in by aco_reduce_assign.cpp, used internally as part of the
         // reduce sequence
         reduce->getOperand(1) = Operand(dst.size() == 2 ? v2_linear : v1_linear);
         reduce->getOperand(2) = Operand(v1_linear);

         Temp tmp_dst = bld.tmp(dst.regClass());
         reduce->getDefinition(0) = Definition(tmp_dst);
         reduce->getDefinition(1) = bld.def(s2); // used internally
         reduce->getDefinition(2) = Definition();
         reduce->getDefinition(3) = Definition(scc, s1);
         reduce->getDefinition(4) = Definition();
         reduce->reduce_op = reduce_op;
         reduce->cluster_size = cluster_size;
         ctx->block->instructions.emplace_back(std::move(reduce));

         emit_wqm(ctx, tmp_dst, dst);
      }
      break;
   }
   case nir_intrinsic_quad_broadcast: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      if (!ctx->divergent_vals[instr->dest.ssa.index]) {
         emit_uniform_subgroup(ctx, instr, src);
      } else {
         Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
         Definition tmp = bld.def(dst.regClass());
         unsigned lane = nir_src_as_const_value(instr->src[1])->u32;
         if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
            uint32_t half_mask = 0x11111111u << lane;
            Temp mask_tmp = bld.pseudo(aco_opcode::p_create_vector, bld.def(s2), Operand(half_mask), Operand(half_mask));
            bld.sop1(aco_opcode::s_wqm_b64, tmp,
                     bld.sop2(aco_opcode::s_and_b64, bld.def(s2), mask_tmp,
                              bld.sop2(aco_opcode::s_and_b64, bld.def(s2), src, Operand(exec, s2))));
         } else if (instr->dest.ssa.bit_size == 32) {
            bld.vop1_dpp(aco_opcode::v_mov_b32, tmp, src,
                         dpp_quad_perm(lane, lane, lane, lane));
         } else {
            fprintf(stderr, "Unimplemented NIR instr bit size: ");
            nir_print_instr(&instr->instr, stderr);
            fprintf(stderr, "\n");
         }
         emit_wqm(ctx, tmp.getTemp(), dst);
      }
      break;
   }
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_swizzle_amd: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      if (!ctx->divergent_vals[instr->dest.ssa.index]) {
         emit_uniform_subgroup(ctx, instr, src);
         break;
      }
      uint16_t dpp_ctrl = 0;
      switch (instr->intrinsic) {
      case nir_intrinsic_quad_swap_horizontal:
         dpp_ctrl = dpp_quad_perm(1, 0, 3, 2);
         break;
      case nir_intrinsic_quad_swap_vertical:
         dpp_ctrl = dpp_quad_perm(2, 3, 0, 1);
         break;
      case nir_intrinsic_quad_swap_diagonal:
         dpp_ctrl = dpp_quad_perm(3, 2, 1, 0);
         break;
      case nir_intrinsic_quad_swizzle_amd: {
         dpp_ctrl = nir_intrinsic_swizzle_mask(instr);
         break;
      }
      default:
         break;
      }

      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      Definition tmp = bld.def(dst.regClass());
      if (instr->dest.ssa.bit_size == 1 && src.regClass() == s2) {
         src = bld.vop2_e64(aco_opcode::v_cndmask_b32, bld.def(v1), Operand(0u), Operand((uint32_t)-1), src);
         bld.vopc_dpp(aco_opcode::v_cmp_lg_u32, tmp, Operand(0u), src, dpp_ctrl);
      } else if (instr->dest.ssa.bit_size == 32) {
         bld.vop1_dpp(aco_opcode::v_mov_b32, tmp, src, dpp_ctrl);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp.getTemp(), dst);
      break;
   }
   case nir_intrinsic_masked_swizzle_amd: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      if (!ctx->divergent_vals[instr->dest.ssa.index]) {
         emit_uniform_subgroup(ctx, instr, src);
         break;
      }
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      Temp tmp = bld.tmp(dst.regClass());
      uint32_t mask = nir_intrinsic_swizzle_mask(instr);
      if (dst.regClass() == v1) {
         bld.ds(aco_opcode::ds_swizzle_b32, Definition(tmp), src, mask, 0, false);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp, dst);
      break;
   }
   case nir_intrinsic_write_invocation_amd: {
      Temp src = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[0].ssa));
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      Temp tmp = bld.tmp(dst.regClass());
      if (dst.regClass() == v1) {
         Temp val = bld.as_uniform(get_ssa_temp(ctx, instr->src[1].ssa));
         Temp lane = bld.as_uniform(get_ssa_temp(ctx, instr->src[2].ssa));
         /* src2 is ignored for writelane. RA assigns the same reg for dst */
         bld.vop3(aco_opcode::v_writelane_b32, Definition(tmp), val, lane, src);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      emit_wqm(ctx, tmp, dst);
      break;
   }
   case nir_intrinsic_mbcnt_amd: {
      Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
      emit_split_vector(ctx, src, 2);
      RegClass rc = getRegClass(src.type(), 1);
      Temp mask_lo = bld.as_uniform(emit_extract_vector(ctx, src, 0, rc));
      Temp tmp = bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, bld.def(v1), mask_lo, Operand(0u));
      Temp mask_hi = bld.as_uniform(emit_extract_vector(ctx, src, 1, rc));
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      Temp wqm_tmp = bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32, bld.def(v1), mask_hi, tmp);
      emit_wqm(ctx, wqm_tmp, dst);
      break;
   }
   case nir_intrinsic_load_helper_invocation: {
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      bld.pseudo(aco_opcode::p_is_helper, Definition(dst));
      ctx->block->kind |= block_kind_needs_lowering;
      ctx->program->needs_exact = true;
      break;
   }
   case nir_intrinsic_first_invocation: {
      emit_wqm(ctx, bld.sop1(aco_opcode::s_ff1_i32_b64, bld.def(s1), Operand(exec, s2)),
               get_ssa_temp(ctx, &instr->dest.ssa));
      break;
   }
   case nir_intrinsic_shader_clock:
      bld.smem(aco_opcode::s_memtime, Definition(get_ssa_temp(ctx, &instr->dest.ssa)));
      break;
   default:
      fprintf(stderr, "Unimplemented intrinsic instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();

      break;
   }
}


void tex_fetch_ptrs(isel_context *ctx, nir_tex_instr *instr,
                    Temp *res_ptr, Temp *samp_ptr, Temp *fmask_ptr,
                    enum glsl_base_type *stype)
{
   nir_deref_instr *texture_deref_instr = NULL;
   nir_deref_instr *sampler_deref_instr = NULL;

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_texture_deref:
         texture_deref_instr = nir_src_as_deref(instr->src[i].src);
         break;
      case nir_tex_src_sampler_deref:
         sampler_deref_instr = nir_src_as_deref(instr->src[i].src);
         break;
      default:
         break;
      }
   }

   *stype = glsl_get_sampler_result_type(texture_deref_instr->type);

   if (!sampler_deref_instr)
      sampler_deref_instr = texture_deref_instr;

   if (instr->sampler_dim  == GLSL_SAMPLER_DIM_BUF)
      *res_ptr = get_sampler_desc(ctx, texture_deref_instr, ACO_DESC_BUFFER, instr, false, false);
   else
      *res_ptr = get_sampler_desc(ctx, texture_deref_instr, ACO_DESC_IMAGE, instr, false, false);
   if (samp_ptr) {
      *samp_ptr = get_sampler_desc(ctx, sampler_deref_instr, ACO_DESC_SAMPLER, instr, false, false);
      if (instr->sampler_dim < GLSL_SAMPLER_DIM_RECT && ctx->options->chip_class < VI) {
         fprintf(stderr, "Unimplemented sampler descriptor: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
         abort();
         // TODO: build samp_ptr = and(samp_ptr, res_ptr)
      }
   }
   if (fmask_ptr && (instr->op == nir_texop_txf_ms ||
                     instr->op == nir_texop_samples_identical))
      *fmask_ptr = get_sampler_desc(ctx, texture_deref_instr, ACO_DESC_FMASK, instr, false, false);
}

void build_cube_select(isel_context *ctx, Temp ma, Temp id, Temp deriv,
                       Temp *out_ma, Temp *out_sc, Temp *out_tc)
{
   Builder bld(ctx->program, ctx->block);

   Temp deriv_x = emit_extract_vector(ctx, deriv, 0, v1);
   Temp deriv_y = emit_extract_vector(ctx, deriv, 1, v1);
   Temp deriv_z = emit_extract_vector(ctx, deriv, 2, v1);

   Operand neg_one(0xbf800000u);
   Operand one(0x3f800000u);
   Operand two(0x40000000u);
   Operand four(0x40800000u);

   Temp is_ma_positive = bld.vopc(aco_opcode::v_cmp_le_f32, bld.hint_vcc(bld.def(s2)), Operand(0u), ma);
   Temp sgn_ma = bld.vop2_e64(aco_opcode::v_cndmask_b32, bld.def(v1), neg_one, one, is_ma_positive);
   Temp neg_sgn_ma = bld.vop2(aco_opcode::v_sub_f32, bld.def(v1), Operand(0u), sgn_ma);

   Temp is_ma_z = bld.vopc(aco_opcode::v_cmp_le_f32, bld.hint_vcc(bld.def(s2)), four, id);
   Temp is_ma_y = bld.vopc(aco_opcode::v_cmp_le_f32, bld.def(s2), two, id);
   is_ma_y = bld.sop2(aco_opcode::s_andn2_b64, bld.hint_vcc(bld.def(s2)), is_ma_y, is_ma_z);
   Temp is_not_ma_x = bld.sop2(aco_opcode::s_or_b64, bld.hint_vcc(bld.def(s2)), is_ma_z, is_ma_y);

   // select sc
   Temp tmp = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), deriv_z, deriv_x, is_not_ma_x);
   Temp sgn = bld.vop2_e64(aco_opcode::v_cndmask_b32, bld.def(v1),
                       bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), neg_sgn_ma, sgn_ma, is_ma_z),
                       one, is_ma_y);
   *out_sc = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), tmp, sgn);

   // select tc
   tmp = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), deriv_y, deriv_z, is_ma_y);
   sgn = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), neg_one, sgn_ma, is_ma_y);
   *out_tc = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), tmp, sgn);

   // select ma
   tmp = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                  bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), deriv_x, deriv_y, is_ma_y),
                  deriv_z, is_ma_z);
   tmp = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(0x7fffffffu), tmp);
   *out_ma = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), two, tmp);
}

void prepare_cube_coords(isel_context *ctx, Temp* coords, Temp* ddx, Temp* ddy, bool is_deriv, bool is_array)
{
   Builder bld(ctx->program, ctx->block);
   Temp coord_args[4], ma, tc, sc, id;
   aco_ptr<Instruction> tmp;
   emit_split_vector(ctx, *coords, is_array ? 4 : 3);
   for (unsigned i = 0; i < (is_array ? 4 : 3); i++)
      coord_args[i] = emit_extract_vector(ctx, *coords, i, v1);

   if (is_array) {
      coord_args[3] = bld.vop1(aco_opcode::v_rndne_f32, bld.def(v1), coord_args[3]);

      // see comment in ac_prepare_cube_coords()
      if (ctx->options->chip_class <= VI)
         coord_args[3] = bld.vop2(aco_opcode::v_max_f32, bld.def(v1), Operand(0u), coord_args[3]);
   }

   ma = bld.vop3(aco_opcode::v_cubema_f32, bld.def(v1), coord_args[0], coord_args[1], coord_args[2]);

   aco_ptr<VOP3A_instruction> vop3a{create_instruction<VOP3A_instruction>(aco_opcode::v_rcp_f32, asVOP3(Format::VOP1), 1, 1)};
   vop3a->getOperand(0) = Operand(ma);
   vop3a->abs[0] = true;
   Temp invma = bld.tmp(v1);
   vop3a->getDefinition(0) = Definition(invma);
   ctx->block->instructions.emplace_back(std::move(vop3a));

   sc = bld.vop3(aco_opcode::v_cubesc_f32, bld.def(v1), coord_args[0], coord_args[1], coord_args[2]);
   if (!is_deriv)
      sc = bld.vop2(aco_opcode::v_madak_f32, bld.def(v1), sc, invma, Operand(0x3fc00000u/*1.5*/));

   tc = bld.vop3(aco_opcode::v_cubetc_f32, bld.def(v1), coord_args[0], coord_args[1], coord_args[2]);
   if (!is_deriv)
      tc = bld.vop2(aco_opcode::v_madak_f32, bld.def(v1), tc, invma, Operand(0x3fc00000u/*1.5*/));

   id = bld.vop3(aco_opcode::v_cubeid_f32, bld.def(v1), coord_args[0], coord_args[1], coord_args[2]);

   if (is_deriv) {
      sc = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), sc, invma);
      tc = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), tc, invma);

      for (unsigned i = 0; i < 2; i++) {
         // see comment in ac_prepare_cube_coords()
         Temp deriv_ma;
         Temp deriv_sc, deriv_tc;
         build_cube_select(ctx, ma, id, i ? *ddy : *ddx,
                           &deriv_ma, &deriv_sc, &deriv_tc);

         deriv_ma = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), deriv_ma, invma);

         Temp x = bld.vop2(aco_opcode::v_sub_f32, bld.def(v1),
                               bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), deriv_sc, invma),
                               bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), deriv_ma, sc));
         Temp y = bld.vop2(aco_opcode::v_sub_f32, bld.def(v1),
                               bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), deriv_tc, invma),
                               bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), deriv_ma, tc));
         *(i ? ddy : ddx) = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2), x, y);
      }

      sc = bld.vop2(aco_opcode::v_add_f32, bld.def(v1), Operand(0x3fc00000u/*1.5*/), sc);
      tc = bld.vop2(aco_opcode::v_add_f32, bld.def(v1), Operand(0x3fc00000u/*1.5*/), tc);
   }

   if (is_array)
      id = bld.vop2(aco_opcode::v_madmk_f32, bld.def(v1), coord_args[3], id, Operand(0x41000000u/*8.0*/));
   *coords = bld.pseudo(aco_opcode::p_create_vector, bld.def(v3), sc, tc, id);

}

Temp apply_round_slice(isel_context *ctx, Temp coords, unsigned idx)
{
   Temp coord_vec[3];
   emit_split_vector(ctx, coords, coords.size());
   for (unsigned i = 0; i < coords.size(); i++)
      coord_vec[i] = emit_extract_vector(ctx, coords, i, v1);

   Builder bld(ctx->program, ctx->block);
   coord_vec[idx] = bld.vop1(aco_opcode::v_rndne_f32, bld.def(v1), coord_vec[idx]);

   aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, coords.size(), 1)};
   for (unsigned i = 0; i < coords.size(); i++)
      vec->getOperand(i) = Operand(coord_vec[i]);
   Temp res = {ctx->program->allocateId(), getRegClass(vgpr, coords.size())};
   vec->getDefinition(0) = Definition(res);
   ctx->block->instructions.emplace_back(std::move(vec));
   return res;
}

void visit_tex(isel_context *ctx, nir_tex_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   bool has_bias = false, has_lod = false, level_zero = false, has_compare = false,
        has_offset = false, has_ddx = false, has_ddy = false, has_derivs = false, has_sample_index = false;
   Temp resource, sampler, fmask_ptr, bias = Temp(), coords, compare = Temp(), sample_index = Temp(),
        lod = Temp(), offset = Temp(), ddx = Temp(), ddy = Temp(), derivs = Temp();
   nir_const_value *sample_index_cv = NULL;
   enum glsl_base_type stype;
   tex_fetch_ptrs(ctx, instr, &resource, &sampler, &fmask_ptr, &stype);

   bool tg4_integer_workarounds = ctx->options->chip_class <= VI && instr->op == nir_texop_tg4 &&
                                  (stype == GLSL_TYPE_UINT || stype == GLSL_TYPE_INT);
   bool tg4_integer_cube_workaround = tg4_integer_workarounds &&
                                      instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE;

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_coord:
         coords = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[i].src.ssa));
         break;
      case nir_tex_src_bias:
         if (instr->op == nir_texop_txb) {
            bias = get_ssa_temp(ctx, instr->src[i].src.ssa);
            has_bias = true;
         }
         break;
      case nir_tex_src_lod: {
         nir_const_value *val = nir_src_as_const_value(instr->src[i].src);

         if (val && val->i32 == 0) {
            level_zero = true;
         } else {
            lod = get_ssa_temp(ctx, instr->src[i].src.ssa);
            has_lod = true;
         }
         break;
      }
      case nir_tex_src_comparator:
         if (instr->is_shadow) {
            compare = get_ssa_temp(ctx, instr->src[i].src.ssa);
            has_compare = true;
         }
         break;
      case nir_tex_src_offset:
         offset = get_ssa_temp(ctx, instr->src[i].src.ssa);
         //offset_src = i;
         has_offset = true;
         break;
      case nir_tex_src_ddx:
         ddx = get_ssa_temp(ctx, instr->src[i].src.ssa);
         has_ddx = true;
         break;
      case nir_tex_src_ddy:
         ddy = get_ssa_temp(ctx, instr->src[i].src.ssa);
         has_ddy = true;
         break;
      case nir_tex_src_ms_index:
         sample_index = get_ssa_temp(ctx, instr->src[i].src.ssa);
         sample_index_cv = nir_src_as_const_value(instr->src[i].src);
         has_sample_index = true;
         break;
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      default:
         break;
      }
   }
// TODO: all other cases: structure taken from ac_nir_to_llvm.c
   if (instr->op == nir_texop_txs && instr->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return get_buffer_size(ctx, resource, get_ssa_temp(ctx, &instr->dest.ssa), true);

   if (instr->op == nir_texop_texture_samples) {
      Temp dword3 = emit_extract_vector(ctx, resource, 3, s1);

      Temp samples_log2 = bld.sop2(aco_opcode::s_bfe_u32, bld.def(s1), bld.def(s1, scc), dword3, Operand(16u | 4u<<16));
      Temp samples = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), Operand(1u), samples_log2);
      Temp type = bld.sop2(aco_opcode::s_bfe_u32, bld.def(s1), bld.def(s1, scc), dword3, Operand(28u | 4u<<16 /* offset=28, width=4 */));
      Temp is_msaa = bld.sopc(aco_opcode::s_cmp_ge_u32, bld.def(s1, scc), type, Operand(14u));

      bld.sop2(aco_opcode::s_cselect_b32, Definition(get_ssa_temp(ctx, &instr->dest.ssa)),
               samples, Operand(1u), bld.scc(is_msaa));
      return;
   }

   if (has_offset && instr->op != nir_texop_txf) {
      aco_ptr<Instruction> tmp_instr;
      Temp acc, pack = Temp();
      if (offset.type() == sgpr) {
         for (unsigned i = 0; i < offset.size(); i++) {
            acc = emit_extract_vector(ctx, offset, i, s1);
            acc = bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), acc, Operand(0x3Fu));

            if (i == 0) {
               pack = acc;
            } else {
               acc = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), acc, Operand(8u * i));
               pack = bld.sop2(aco_opcode::s_or_b32, bld.def(s1), bld.def(s1, scc), pack, acc);
            }
         }
      } else {
         for (unsigned i = 0; i < offset.size(); i++) {
            acc = emit_extract_vector(ctx, offset, i, v1);
            acc = bld.vop2(aco_opcode::v_and_b32, bld.def(v1), Operand(0x3Fu), acc);

            if (i == 0) {
               pack = acc;
            } else {
               acc = bld.vop2(aco_opcode::v_lshlrev_b32, bld.def(v1), Operand(8u * i), acc);
               pack = bld.vop2(aco_opcode::v_or_b32, bld.def(v1), pack, acc);
            }
         }
      }
      offset = pack;
   }

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && instr->coord_components)
      prepare_cube_coords(ctx, &coords, &ddx, &ddy, instr->op == nir_texop_txd, instr->is_array && instr->op != nir_texop_lod);

   /* pack derivatives */
   if (has_ddx || has_ddy) {
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_1D && ctx->options->chip_class >= GFX9) {
         derivs = bld.pseudo(aco_opcode::p_create_vector, bld.def(v4),
                             ddx, Operand(0u), ddy, Operand(0u));
      } else {
         derivs = bld.pseudo(aco_opcode::p_create_vector, bld.def(vgpr, ddx.size() + ddy.size()), ddx, ddy);
      }
      has_derivs = true;
   }

   if (instr->coord_components > 1 &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
       instr->is_array &&
       instr->op != nir_texop_txf)
      coords = apply_round_slice(ctx, coords, 1);

   if (instr->coord_components > 2 &&
      (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_MS ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) &&
       instr->is_array &&
       instr->op != nir_texop_txf && instr->op != nir_texop_txf_ms)
      coords = apply_round_slice(ctx, coords, 2);

   if (ctx->options->chip_class >= GFX9 &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
       instr->op != nir_texop_lod && instr->coord_components) {
      assert(coords.size() > 0 && coords.size() < 3);

      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, coords.size() + 1, 1)};
      vec->getOperand(0) = Operand(emit_extract_vector(ctx, coords, 0, v1));
      vec->getOperand(1) = instr->op == nir_texop_txf ? Operand((uint32_t) 0) : Operand((uint32_t) 0x3f000000);
      if (coords.size() > 1)
         vec->getOperand(2) = Operand(emit_extract_vector(ctx, coords, 1, v1));
      coords = {ctx->program->allocateId(), getRegClass(RegType::vgpr, coords.size() + 1)};
      vec->getDefinition(0) = Definition(coords);
      ctx->block->instructions.emplace_back(std::move(vec));
   }

   bool da = should_declare_array(ctx, instr->sampler_dim, instr->is_array);

   if (instr->op == nir_texop_samples_identical)
      resource = fmask_ptr;

   else if ((instr->sampler_dim == GLSL_SAMPLER_DIM_MS ||
             instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) &&
            instr->op != nir_texop_txs) {
      assert(has_sample_index);
      Operand op(sample_index);
      if (sample_index_cv)
         op = Operand(sample_index_cv->u32);
      sample_index = adjust_sample_index_using_fmask(ctx, da, coords, op, fmask_ptr);
   }

   if (has_offset && instr->op == nir_texop_txf) {
      Temp split_coords[coords.size()];
      emit_split_vector(ctx, coords, coords.size());
      for (unsigned i = 0; i < coords.size(); i++)
         split_coords[i] = emit_extract_vector(ctx, coords, i, v1);

      unsigned i = 0;
      for (; i < std::min(offset.size(), instr->coord_components); i++) {
         Temp off = emit_extract_vector(ctx, offset, i, v1);
         Temp dst{ctx->program->allocateId(), v1};
         emit_v_add32(ctx, dst, Operand(split_coords[i]), Operand(off));
         split_coords[i] = dst;
      }

      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, coords.size(), 1)};
      for (unsigned i = 0; i < coords.size(); i++)
         vec->getOperand(i) = Operand(split_coords[i]);
      coords = {ctx->program->allocateId(), coords.regClass()};
      vec->getDefinition(0) = Definition(coords);
      ctx->block->instructions.emplace_back(std::move(vec));

      has_offset = false;
   }

   /* Build tex instruction */
   unsigned dmask = nir_ssa_def_components_read(&instr->dest.ssa);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp tmp_dst = dst;
   if ((util_bitcount(dmask) != instr->dest.ssa.num_components && instr->op != nir_texop_tg4) ||
       tg4_integer_cube_workaround || dst.type() == sgpr || instr->op == nir_texop_samples_identical)
      tmp_dst = Temp{ctx->program->allocateId(), getRegClass(vgpr, util_bitcount(dmask))};

   /* gather4 selects the component by dmask and always returns vec4 */
   if (instr->op == nir_texop_tg4) {
      assert(instr->dest.ssa.num_components == 4);
      if (instr->is_shadow)
         dmask = 1;
      else
         dmask = 1 << instr->component;
   } else if (instr->op == nir_texop_query_levels) {
      dmask = 1 << 3;
   } else if (ctx->options->chip_class >= GFX9 &&
              instr->op == nir_texop_txs &&
              instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
              instr->is_array) {
      dmask = (dmask & 0x1) | ((dmask & 0x2) << 1);
   }

   aco_ptr<MIMG_instruction> tex;
   if (instr->op == nir_texop_txs || instr->op == nir_texop_query_levels) {
      if (!has_lod)
         lod = bld.vop1(aco_opcode::v_mov_b32, bld.def(v1), Operand(0u));

      bool div_by_6 = instr->op == nir_texop_txs &&
                      instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
                      instr->is_array &&
                      (dmask & (1 << 2));
      if (tmp_dst.id() == dst.id() && div_by_6)
         tmp_dst = {ctx->program->allocateId(), tmp_dst.regClass()};

      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1));
      tex->getOperand(0) = Operand(as_vgpr(ctx,lod));
      tex->getOperand(1) = Operand(resource);
      tex->dmask = dmask;
      tex->da = da;
      tex->getDefinition(0) = Definition(tmp_dst);
      tex->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(tex));

      if (div_by_6) {
         /* divide 3rd value by 6 by multiplying with magic number */
         emit_split_vector(ctx, tmp_dst, tmp_dst.size());
         Temp c = {ctx->program->allocateId(), s1};
         aco_ptr<Instruction> mov{create_s_mov(Definition(c), Operand((uint32_t) 0x2AAAAAAB))};
         ctx->block->instructions.emplace_back(std::move(mov));
         Temp by_6 = bld.vop3(aco_opcode::v_mul_hi_i32, bld.def(v1), emit_extract_vector(ctx, tmp_dst, 2, v1), c);
         assert(instr->dest.ssa.num_components == 3);
         Temp tmp = dst.type() == vgpr ? dst : bld.tmp(v3);
         tmp_dst = bld.pseudo(aco_opcode::p_create_vector, Definition(tmp),
                              emit_extract_vector(ctx, tmp_dst, 0, v1),
                              emit_extract_vector(ctx, tmp_dst, 1, v1),
                              by_6);

      }

      expand_vector(ctx, tmp_dst, dst, instr->dest.ssa.num_components, dmask);
      return;
   }

   Temp tg4_compare_cube_wa64 = Temp();

   if (tg4_integer_workarounds) {
      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1));
      tex->getOperand(0) = bld.vop1(aco_opcode::v_mov_b32, bld.def(v1), Operand(0u));
      tex->getOperand(1) = Operand(resource);
      tex->dmask = 0x3;
      tex->da = da;
      Temp size = bld.tmp(v2);
      tex->getDefinition(0) = Definition(size);
      tex->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(tex));
      emit_split_vector(ctx, size, size.size());

      Temp half_texel[2];
      for (unsigned i = 0; i < 2; i++) {
         half_texel[i] = emit_extract_vector(ctx, size, i, v1);
         half_texel[i] = bld.vop1(aco_opcode::v_cvt_f32_i32, bld.def(v1), half_texel[i]);
         half_texel[i] = bld.vop1(aco_opcode::v_rcp_iflag_f32, bld.def(v1), half_texel[i]);
         half_texel[i] = bld.vop2(aco_opcode::v_mul_f32, bld.def(v1), Operand(0xbf000000/*-0.5*/), half_texel[i]);
      }

      Temp orig_coords[2] = {
         emit_extract_vector(ctx, coords, 0, v1),
         emit_extract_vector(ctx, coords, 1, v1)};
      Temp new_coords[2] = {
         bld.vop2(aco_opcode::v_add_f32, bld.def(v1), orig_coords[0], half_texel[0]),
         bld.vop2(aco_opcode::v_add_f32, bld.def(v1), orig_coords[1], half_texel[1])
      };

      if (tg4_integer_cube_workaround) {
         // see comment in ac_nir_to_llvm.c's lower_gather4_integer()
         Temp desc[resource.size()];
         aco_ptr<Instruction> split{create_instruction<Instruction>(aco_opcode::p_split_vector,
                                                                    Format::PSEUDO, 1, resource.size())};
         split->getOperand(0) = Operand(resource);
         for (unsigned i = 0; i < resource.size(); i++) {
            desc[i] = bld.tmp(s1);
            split->getDefinition(i) = Definition(desc[i]);
         }
         ctx->block->instructions.emplace_back(std::move(split));

         Temp dfmt = bld.sop2(aco_opcode::s_bfe_u32, bld.def(s1), bld.def(s1, scc), desc[1], Operand(20u | (6u << 16)));
         Temp compare_cube_wa = bld.sopc(aco_opcode::s_cmp_eq_u32, bld.def(s1, scc), dfmt,
                                         Operand((uint32_t)V_008F14_IMG_DATA_FORMAT_8_8_8_8));

         Temp nfmt;
         if (stype == GLSL_TYPE_UINT) {
            nfmt = bld.sop2(aco_opcode::s_cselect_b32, bld.def(s1),
                            Operand((uint32_t)V_008F14_IMG_NUM_FORMAT_USCALED),
                            Operand((uint32_t)V_008F14_IMG_NUM_FORMAT_UINT),
                            bld.scc(compare_cube_wa));
         } else {
            nfmt = bld.sop2(aco_opcode::s_cselect_b32, bld.def(s1),
                            Operand((uint32_t)V_008F14_IMG_NUM_FORMAT_SSCALED),
                            Operand((uint32_t)V_008F14_IMG_NUM_FORMAT_SINT),
                            bld.scc(compare_cube_wa));
         }
         tg4_compare_cube_wa64 = as_divergent_bool(ctx, compare_cube_wa, true);
         nfmt = bld.sop2(aco_opcode::s_lshl_b32, bld.def(s1), bld.def(s1, scc), nfmt, Operand(26u));

         desc[1] = bld.sop2(aco_opcode::s_and_b32, bld.def(s1), bld.def(s1, scc), desc[1],
                            Operand((uint32_t)C_008F14_NUM_FORMAT_GFX6));
         desc[1] = bld.sop2(aco_opcode::s_or_b32, bld.def(s1), bld.def(s1, scc), desc[1], nfmt);

         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector,
                                                                  Format::PSEUDO, resource.size(), 1)};
         for (unsigned i = 0; i < resource.size(); i++)
            vec->getOperand(i) = Operand(desc[i]);
         resource = bld.tmp(resource.regClass());
         vec->getDefinition(0) = Definition(resource);
         ctx->block->instructions.emplace_back(std::move(vec));

         new_coords[0] = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                                  new_coords[0], orig_coords[0], tg4_compare_cube_wa64);
         new_coords[1] = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                                  new_coords[1], orig_coords[1], tg4_compare_cube_wa64);
      }

      if (coords.size() == 3) {
         coords = bld.pseudo(aco_opcode::p_create_vector, bld.def(v3),
                             new_coords[0], new_coords[1],
                             emit_extract_vector(ctx, coords, 2, v1));
      } else {
         assert(coords.size() == 2);
         coords = bld.pseudo(aco_opcode::p_create_vector, bld.def(v2),
                             new_coords[0], new_coords[1]);
      }
   }

   if (!(has_ddx && has_ddy) && !has_lod && !level_zero &&
       instr->sampler_dim != GLSL_SAMPLER_DIM_MS &&
       instr->sampler_dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
      coords = emit_wqm(ctx, coords);

   std::vector<Operand> args;
   if (has_offset)
      args.emplace_back(Operand(offset));
   if (has_bias)
      args.emplace_back(Operand(bias));
   if (has_compare)
      args.emplace_back(Operand(compare));
   if (has_derivs)
      args.emplace_back(Operand(derivs));
   args.emplace_back(Operand(coords));
   if (has_sample_index)
      args.emplace_back(Operand(sample_index));
   if (has_lod)
      args.emplace_back(lod);

   Operand arg;
   if (args.size() > 1) {
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, args.size(), 1)};
      unsigned size = 0;
      for (unsigned i = 0; i < args.size(); i++) {
         size += args[i].size();
         vec->getOperand(i) = args[i];
      }
      RegClass rc = getRegClass(vgpr, size);
      Temp tmp = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(tmp);
      ctx->block->instructions.emplace_back(std::move(vec));
      arg = Operand(tmp);
   } else {
      assert(args[0].isTemp());
      arg = Operand(as_vgpr(ctx, args[0].getTemp()));
   }

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      //FIXME: if (ctx->abi->gfx9_stride_size_workaround) return ac_build_buffer_load_format_gfx9_safe()

      assert(coords.size() == 1);
      unsigned last_bit = util_last_bit(nir_ssa_def_components_read(&instr->dest.ssa));
      aco_opcode op;
      switch (last_bit) {
      case 1:
         op = aco_opcode::buffer_load_format_x; break;
      case 2:
         op = aco_opcode::buffer_load_format_xy; break;
      case 3:
         op = aco_opcode::buffer_load_format_xyz; break;
      case 4:
         op = aco_opcode::buffer_load_format_xyzw; break;
      default:
         unreachable("Tex instruction loads more than 4 components.");
      }

      /* if the instruction return value matches exactly the nir dest ssa, we can use it directly */
      if (last_bit == instr->dest.ssa.num_components && dst.type() == vgpr)
         tmp_dst = dst;
      else
         tmp_dst = {ctx->program->allocateId(), getRegClass(vgpr, last_bit)};

      aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(op, Format::MUBUF, 3, 1)};
      mubuf->getOperand(0) = Operand(coords);
      mubuf->getOperand(1) = Operand(resource);
      mubuf->getOperand(2) = Operand((uint32_t) 0);
      mubuf->getDefinition(0) = Definition(tmp_dst);
      mubuf->idxen = true;
      mubuf->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));

      expand_vector(ctx, tmp_dst, dst, instr->dest.ssa.num_components, (1 << last_bit) - 1);
      return;
   }


   if (instr->op == nir_texop_txf ||
       instr->op == nir_texop_txf_ms ||
       instr->op == nir_texop_samples_identical) {
      aco_opcode op = level_zero || instr->sampler_dim == GLSL_SAMPLER_DIM_MS ? aco_opcode::image_load : aco_opcode::image_load_mip;
      tex.reset(create_instruction<MIMG_instruction>(op, Format::MIMG, 2, 1));
      tex->getOperand(0) = Operand(arg);
      tex->getOperand(1) = Operand(resource);
      tex->dmask = dmask;
      tex->unrm = true;
      tex->da = da;
      tex->getDefinition(0) = Definition(tmp_dst);
      tex->can_reorder = true;
      ctx->block->instructions.emplace_back(std::move(tex));

      if (instr->op == nir_texop_samples_identical) {
         assert(dmask == 1 && dst.regClass() == v1);
         assert(dst.id() != tmp_dst.id());

         Temp tmp = bld.tmp(s2);
         bld.vopc(aco_opcode::v_cmp_eq_u32, Definition(tmp), Operand(0u), tmp_dst).def(0).setHint(vcc);
         bld.vop2_e64(aco_opcode::v_cndmask_b32, Definition(dst), Operand(0u), Operand((uint32_t)-1), tmp);

      } else {
         expand_vector(ctx, tmp_dst, dst, instr->dest.ssa.num_components, dmask);
      }
      return;
   }

   // TODO: would be better to do this by adding offsets, but needs the opcodes ordered.
   aco_opcode opcode = aco_opcode::image_sample;
   if (has_offset) { /* image_sample_*_o */
      if (has_compare) {
         opcode = aco_opcode::image_sample_c_o;
         if (has_derivs)
            opcode = aco_opcode::image_sample_c_d_o;
         if (has_bias)
            opcode = aco_opcode::image_sample_c_b_o;
         if (level_zero)
            opcode = aco_opcode::image_sample_c_lz_o;
         if (has_lod)
            opcode = aco_opcode::image_sample_c_l_o;
      } else {
         opcode = aco_opcode::image_sample_o;
         if (has_derivs)
            opcode = aco_opcode::image_sample_d_o;
         if (has_bias)
            opcode = aco_opcode::image_sample_b_o;
         if (level_zero)
            opcode = aco_opcode::image_sample_lz_o;
         if (has_lod)
            opcode = aco_opcode::image_sample_l_o;
      }
   } else { /* no offset */
      if (has_compare) {
         opcode = aco_opcode::image_sample_c;
         if (has_derivs)
            opcode = aco_opcode::image_sample_c_d;
         if (has_bias)
            opcode = aco_opcode::image_sample_c_b;
         if (level_zero)
            opcode = aco_opcode::image_sample_c_lz;
         if (has_lod)
            opcode = aco_opcode::image_sample_c_l;
      } else {
         opcode = aco_opcode::image_sample;
         if (has_derivs)
            opcode = aco_opcode::image_sample_d;
         if (has_bias)
            opcode = aco_opcode::image_sample_b;
         if (level_zero)
            opcode = aco_opcode::image_sample_lz;
         if (has_lod)
            opcode = aco_opcode::image_sample_l;
      }
   }

   if (instr->op == nir_texop_tg4) {
      if (has_offset) {
         opcode = aco_opcode::image_gather4_lz_o;
         if (has_compare)
            opcode = aco_opcode::image_gather4_c_lz_o;
      } else {
         opcode = aco_opcode::image_gather4_lz;
         if (has_compare)
            opcode = aco_opcode::image_gather4_c_lz;
      }
   } else if (instr->op == nir_texop_lod) {
      opcode = aco_opcode::image_get_lod;
   }

   tex.reset(create_instruction<MIMG_instruction>(opcode, Format::MIMG, 3, 1));
   tex->getOperand(0) = arg;
   tex->getOperand(1) = Operand(resource);
   tex->getOperand(2) = Operand(sampler);
   tex->dmask = dmask;
   tex->da = da;
   tex->getDefinition(0) = Definition(tmp_dst);
   tex->can_reorder = true;
   ctx->block->instructions.emplace_back(std::move(tex));

   if (tg4_integer_cube_workaround) {
      assert(tmp_dst.id() != dst.id());
      assert(tmp_dst.size() == dst.size() && dst.size() == 4);

      emit_split_vector(ctx, tmp_dst, tmp_dst.size());
      Temp val[4];
      for (unsigned i = 0; i < dst.size(); i++) {
         val[i] = emit_extract_vector(ctx, tmp_dst, i, v1);
         Temp cvt_val;
         if (stype == GLSL_TYPE_UINT)
            cvt_val = bld.vop1(aco_opcode::v_cvt_u32_f32, bld.def(v1), val[i]);
         else
            cvt_val = bld.vop1(aco_opcode::v_cvt_i32_f32, bld.def(v1), val[i]);
         val[i] = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1), val[i], cvt_val, tg4_compare_cube_wa64);
      }
      Temp tmp = dst.regClass() == v4 ? dst : bld.tmp(v4);
      tmp_dst = bld.pseudo(aco_opcode::p_create_vector, Definition(tmp),
                           val[0], val[1], val[2], val[3]);
   }
   unsigned mask = instr->op == nir_texop_tg4 ? 0xF : dmask;
   expand_vector(ctx, tmp_dst, dst, instr->dest.ssa.num_components, mask);

}


void visit_phi(isel_context *ctx, nir_phi_instr *instr)
{
   aco_ptr<Instruction> phi;
   unsigned num_src = exec_list_length(&instr->srcs);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_opcode opcode = ctx->divergent_vals[instr->dest.ssa.index] ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
   if (instr->dest.ssa.bit_size == 1 && opcode == aco_opcode::p_phi && num_src == 1) {
      nir_phi_src *src = (nir_phi_src*)exec_list_get_head(&instr->srcs);
      if (get_ssa_temp(ctx, src->src.ssa).regClass() == s2)
         opcode = aco_opcode::p_linear_phi; /* just a linear phi will do and will avoid expensive lowering */
   }

   std::map<unsigned, nir_ssa_def*> phi_src;
   bool all_undef = true;
   nir_foreach_phi_src(src, instr) {
      phi_src[src->pred->index] = src->src.ssa;
      if (src->src.ssa->parent_instr->type != nir_instr_type_ssa_undef)
         all_undef = false;
   }
   if (all_undef) {
      Builder bld(ctx->program, ctx->block);
      if (dst.regClass() == s1) {
         bld.sop1(aco_opcode::s_mov_b32, Definition(dst), Operand(s1));
      } else if (dst.regClass() == v1) {
         bld.vop1(aco_opcode::v_mov_b32, Definition(dst), Operand(v1));
      } else {
         bld.pseudo(aco_opcode::p_create_vector, Definition(dst), Operand(dst.regClass()));
      }
      return;
   }

   /* try to scalarize vector phis */
   if (dst.size() > 1) {
      // TODO: scalarize linear phis on divergent ifs
      bool can_scalarize = (opcode == aco_opcode::p_phi || !(ctx->block->kind & block_kind_merge));
      std::array<Temp, 4> new_vec;
      for (std::pair<const unsigned, nir_ssa_def*>& pair : phi_src) {
         Temp src = get_ssa_temp(ctx, pair.second);
         if (ctx->allocated_vec.find(src.id()) == ctx->allocated_vec.end()) {
            can_scalarize = false;
            break;
         }
      }
      if (can_scalarize) {
         unsigned num_components = instr->dest.ssa.num_components;
         assert(dst.size() % num_components == 0);
         RegClass rc = getRegClass(dst.type(), dst.size() / num_components);

         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, num_components, 1)};
         for (unsigned k = 0; k < num_components; k++) {
            phi.reset(create_instruction<Instruction>(opcode, Format::PSEUDO, num_src, 1));
            std::map<unsigned, nir_ssa_def*>::iterator it = phi_src.begin();
            for (unsigned i = 0; i < num_src; i++) {
               Temp src = get_ssa_temp(ctx, it->second);
               phi->getOperand(i) = Operand(ctx->allocated_vec[src.id()][k]);
               ++it;
            }
            Temp phi_dst = {ctx->program->allocateId(), rc};
            phi->getDefinition(0) = Definition(phi_dst);
            ctx->block->instructions.emplace(ctx->block->instructions.begin(), std::move(phi));
            new_vec[k] = phi_dst;
            vec->getOperand(k) = Operand(phi_dst);
         }
         vec->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vec));
         ctx->allocated_vec.emplace(dst.id(), new_vec);
         return;
      }
   }

   phi.reset(create_instruction<Instruction>(opcode, Format::PSEUDO, num_src, 1));

   /* if we have a linear phi on a divergent if, we know that one src is undef */
   if (opcode == aco_opcode::p_linear_phi && ctx->block->kind & block_kind_merge) {
      Block* block;
      /* we place the phi either in the between-block or in the current block */
      if (phi_src.begin()->second->parent_instr->type != nir_instr_type_ssa_undef) {
         assert((++phi_src.begin())->second->parent_instr->type == nir_instr_type_ssa_undef);
         block = ctx->block->linear_predecessors[1]->linear_predecessors[0];
         assert(block->kind & block_kind_invert);
         phi->getOperand(0) = Operand(get_ssa_temp(ctx, phi_src.begin()->second));
      } else {
         assert((++phi_src.begin())->second->parent_instr->type != nir_instr_type_ssa_undef);
         block = ctx->block;
         phi->getOperand(0) = Operand(get_ssa_temp(ctx, (++phi_src.begin())->second));
      }
      phi->getOperand(1) = Operand(dst.regClass());
      phi->getDefinition(0) = Definition(dst);
      block->instructions.emplace(block->instructions.begin(), std::move(phi));
      return;
   }

   std::map<unsigned, nir_ssa_def*>::iterator it = phi_src.begin();
   for (unsigned i = 0; i < num_src; i++) {
      phi->getOperand(i) = Operand(get_ssa_temp(ctx, it->second));
      ++it;
   }
   phi->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace(ctx->block->instructions.begin(), std::move(phi));
}


void visit_undef(isel_context *ctx, nir_ssa_undef_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->def);
   aco_ptr<Instruction> undef;
   assert(dst.type() == sgpr);

   if (dst.size() == 1) {
      undef.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
   } else {
      undef.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 1, 1));
   }
   undef->getOperand(0) = Operand(dst.regClass());
   undef->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(undef));
}

void visit_jump(isel_context *ctx, nir_jump_instr *instr)
{
   Builder bld(ctx->program, ctx->block);
   Block *logical_target;
   append_logical_end(ctx->block);

   switch (instr->type) {
   case nir_jump_break:
      logical_target = ctx->cf_info.parent_loop.exit;
      add_logical_edge(ctx->block, logical_target);
      ctx->block->kind |= block_kind_break;

      if (!ctx->cf_info.parent_if.is_divergent &&
          !ctx->cf_info.parent_loop.has_divergent_continue) {
         /* uniform break - directly jump out of the loop */
         ctx->block->kind |= block_kind_uniform;
         ctx->cf_info.has_branch = true;
         bld.branch(aco_opcode::p_branch, logical_target);
         add_linear_edge(ctx->block, logical_target);
         return;
      }
      ctx->cf_info.parent_loop.has_divergent_branch = true;
      break;
   case nir_jump_continue:
      logical_target = ctx->cf_info.parent_loop.entry;
      add_logical_edge(ctx->block, logical_target);
      ctx->block->kind |= block_kind_continue;

      if (ctx->cf_info.parent_if.is_divergent) {
         /* for potential uniform breaks after this continue,
            we must ensure that they are handled correctly */
         ctx->cf_info.parent_loop.has_divergent_continue = true;
         ctx->cf_info.parent_loop.has_divergent_branch = true;
      } else {
         /* uniform continue - directly jump to the loop header */
         ctx->block->kind |= block_kind_uniform;
         ctx->cf_info.has_branch = true;
         bld.branch(aco_opcode::p_branch, logical_target);
         add_linear_edge(ctx->block, logical_target);
         return;
      }
      break;
   default:
      fprintf(stderr, "Unknown NIR jump instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }

   /* remove critical edges from linear CFG */
   Block* break_block = ctx->program->createAndInsertBlock();
   break_block->loop_nest_depth = ctx->cf_info.loop_nest_depth;
   break_block->kind |= block_kind_uniform;
   add_linear_edge(ctx->block, break_block);
   Block* continue_block = ctx->program->createAndInsertBlock();
   continue_block->loop_nest_depth = ctx->cf_info.loop_nest_depth;
   add_linear_edge(ctx->block, continue_block);
   bld.branch(aco_opcode::p_branch, break_block, continue_block);

   bld.reset(break_block);
   add_linear_edge(break_block, logical_target);
   bld.branch(aco_opcode::p_branch, logical_target);

   append_logical_start(continue_block);
   ctx->block = continue_block;
   return;
}

void visit_block(isel_context *ctx, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
         visit_alu_instr(ctx, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_load_const:
         visit_load_const(ctx, nir_instr_as_load_const(instr));
         break;
      case nir_instr_type_intrinsic:
         visit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
         break;
      case nir_instr_type_tex:
         visit_tex(ctx, nir_instr_as_tex(instr));
         break;
      case nir_instr_type_phi:
         visit_phi(ctx, nir_instr_as_phi(instr));
         break;
      case nir_instr_type_ssa_undef:
         visit_undef(ctx, nir_instr_as_ssa_undef(instr));
         break;
      case nir_instr_type_deref:
         break;
      case nir_instr_type_jump:
         visit_jump(ctx, nir_instr_as_jump(instr));
         break;
      default:
         fprintf(stderr, "Unknown NIR instr type: ");
         nir_print_instr(instr, stderr);
         fprintf(stderr, "\n");
         //abort();
      }
   }
}



static void visit_loop(isel_context *ctx, nir_loop *loop)
{
   append_logical_end(ctx->block);
   ctx->block->kind |= block_kind_loop_preheader | block_kind_uniform;

   Block* loop_entry = ctx->program->createAndInsertBlock();
   loop_entry->loop_nest_depth = ctx->cf_info.loop_nest_depth + 1;
   loop_entry->kind |= block_kind_loop_header;
   Block* loop_exit = new Block();
   loop_exit->loop_nest_depth = ctx->cf_info.loop_nest_depth;
   loop_exit->kind |= (block_kind_loop_exit | (ctx->block->kind & block_kind_top_level));

   Builder bld(ctx->program, ctx->block);
   bld.branch(aco_opcode::p_branch, loop_entry);
   add_edge(ctx->block, loop_entry);
   ctx->block = loop_entry;

   /* emit loop body */
   loop_info_RAII loop_raii(ctx, loop_entry, loop_exit);
   append_logical_start(ctx->block);
   visit_cf_list(ctx, &loop->body);

   if (!ctx->cf_info.has_branch && !ctx->cf_info.parent_loop.has_divergent_branch) {
      append_logical_end(ctx->block);
      add_edge(ctx->block, loop_entry);
      ctx->block->kind |= (block_kind_continue | block_kind_uniform);
      bld.reset(ctx->block);
      bld.branch(aco_opcode::p_branch, loop_entry);
   } else {
      /* fixup phis in loop header */
      for (auto&& instr : loop_entry->instructions) {
         if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi) {
            /* the last operand should be the one that needs to be removed */
            instr->num_operands--;
         } else {
            break;
         }
      }
   }

   ctx->cf_info.has_branch = false;

   // TODO: if the loop has not a single exit, we must add one °°
   /* emit loop successor block */
   loop_exit->index = ctx->program->blocks.size();
   ctx->block = loop_exit;
   ctx->program->blocks.emplace_back(loop_exit);
   append_logical_start(ctx->block);

   #if 0
   // TODO: check if it is beneficial to not branch on continues
   /* trim linear phis in loop header */
   for (auto&& instr : loop_entry->instructions) {
      if (instr->opcode == aco_opcode::p_linear_phi) {
         aco_ptr<Instruction> new_phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, loop_entry->linear_predecessors.size(), 1)};
         new_phi->getDefinition(0) = instr->getDefinition(0);
         for (unsigned i = 0; i < new_phi->num_operands; i++)
            new_phi->getOperand(i) = instr->getOperand(i);
         /* check that the remaining operands are all the same */
         for (unsigned i = new_phi->num_operands; i < instr->num_operands; i++)
            assert(instr->getOperand(i).tempId() == instr->getOperand(new_phi->num_operands -1).tempId());
         instr.swap(new_phi);
      } else if (instr->opcode == aco_opcode::p_phi) {
         continue;
      } else {
         break;
      }
   }
   #endif
}

static void visit_if(isel_context *ctx, nir_if *if_stmt)
{
   Temp cond = get_ssa_temp(ctx, if_stmt->condition.ssa);
   Builder bld(ctx->program, ctx->block);
   aco_ptr<Pseudo_branch_instruction> branch;

   if (!ctx->divergent_vals[if_stmt->condition.ssa->index]) { /* uniform condition */
      /**
       * Uniform conditionals are represented in the following way*) :
       *
       * The linear and logical CFG:
       *                        BB_IF
       *                        /    \
       *       BB_THEN (logical)      BB_ELSE (logical)
       *                        \    /
       *                        BB_ENDIF
       *
       * *) Exceptions may be due to break and continue statements within loops
       *    If a break/continue happens within uniform control flow, it branches
       *    to the loop exit/entry block. Otherwise, it branches to the next
       *    merge block.
       **/

      Block* BB_if = ctx->block;
      BB_if->kind |= block_kind_uniform;
      Block* BB_then = ctx->program->createAndInsertBlock();
      BB_then->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_else = new Block();
      BB_else->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_endif = new Block();
      BB_endif->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      BB_endif->kind |= BB_if->kind & block_kind_top_level;
      Block* parent_if_merge_block = ctx->cf_info.parent_if.merge_block;
      ctx->cf_info.parent_if.merge_block = BB_endif;

      append_logical_end(BB_if);

      if (cond.regClass() == s2) {
         // TODO: in a post-RA optimizer, we could check if the condition is in VCC and omit this instruction
         cond = as_uniform_bool(ctx, cond);
      }

      /* emit branch */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(cond);
      branch->getOperand(0).setFixed(scc);
      branch->targets[0] = BB_else;
      branch->targets[1] = BB_then;
      BB_if->instructions.emplace_back(std::move(branch));
      add_edge(BB_if, BB_then);
      add_edge(BB_if, BB_else);

      /** emit then block */
      append_logical_start(BB_then);
      ctx->block = BB_then;
      visit_cf_list(ctx, &if_stmt->then_list);
      BB_then = ctx->block;
      bool then_branch = ctx->cf_info.has_branch;
      bool then_branch_divergent = ctx->cf_info.parent_loop.has_divergent_branch;

      if (!then_branch) {
         append_logical_end(BB_then);
         /* branch from then block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_then->instructions.emplace_back(std::move(branch));
         add_linear_edge(BB_then, BB_endif);
         if (!then_branch_divergent)
            add_logical_edge(BB_then, BB_endif);
         BB_then->kind |= block_kind_uniform;
      }

      ctx->cf_info.has_branch = false;
      ctx->cf_info.parent_loop.has_divergent_branch = false;

      /** emit else block */
      BB_else->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else);
      append_logical_start(BB_else);
      ctx->block = BB_else;
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else = ctx->block;

      if (!ctx->cf_info.has_branch) {
         append_logical_end(BB_else);
         /* branch from then block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_else->instructions.emplace_back(std::move(branch));
         add_linear_edge(BB_else, BB_endif);
         if (!ctx->cf_info.parent_loop.has_divergent_branch)
            add_logical_edge(BB_else, BB_endif);
         BB_else->kind |= block_kind_uniform;
      }

      ctx->cf_info.has_branch &= then_branch;
      ctx->cf_info.parent_loop.has_divergent_branch &= then_branch_divergent;

      /** emit endif merge block */
      if (!ctx->cf_info.has_branch) {
         BB_endif->index = ctx->program->blocks.size();
         ctx->program->blocks.emplace_back(BB_endif);

         append_logical_start(BB_endif);
         ctx->block = BB_endif;
      } else {
         delete BB_endif;
      }
      ctx->cf_info.parent_if.merge_block = parent_if_merge_block;

   } else { /* non-uniform condition */
      /**
       * To maintain a logical and linear CFG without critical edges,
       * non-uniform conditionals are represented in the following way*) :
       *
       * The linear CFG:
       *                        BB_IF
       *                        /    \
       *       BB_THEN (logical)      BB_THEN (linear)
       *                        \    /
       *                        BB_BETWEEN (linear)
       *                        /    \
       *       BB_ELSE (logical)      BB_ELSE (linear)
       *                        \    /
       *                        BB_ENDIF
       *
       * The logical CFG:
       *                        BB_IF
       *                        /    \
       *       BB_THEN (logical)      BB_ELSE (logical)
       *                        \    /
       *                        BB_ENDIF
       *
       * *) Exceptions may be due to break and continue statements within loops
       **/

      Block* BB_if = ctx->block;
      BB_if->kind |= block_kind_branch;
      Block* BB_then_logical = ctx->program->createAndInsertBlock();
      BB_then_logical->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_then_linear = new Block();
      BB_then_linear->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      BB_then_linear->kind |= block_kind_uniform;
      Block* BB_between = new Block();
      BB_between->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      /* Invert blocks are intentionally not marked as top level because they
       * are not part of the logical cfg. */
      BB_between->kind |= block_kind_invert;
      Block* BB_else_logical = new Block();
      BB_else_logical->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_else_linear = new Block();
      BB_else_linear->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      BB_else_linear->kind |= block_kind_uniform;
      Block* BB_endif = new Block();
      BB_endif->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      BB_endif->kind |= (block_kind_merge | (BB_if->kind & block_kind_top_level));

      append_logical_end(BB_if);

      /* branch to linear then block */
      assert(cond.regClass() == s2);
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(cond);
      branch->targets[0] = BB_then_linear;
      branch->targets[1] = BB_then_logical;
      BB_if->instructions.push_back(std::move(branch));
      add_edge(BB_if, BB_then_logical);
      add_linear_edge(BB_if, BB_then_linear);
      add_logical_edge(BB_if, BB_else_logical);
      if_info_RAII if_raii(ctx, BB_between);

      /** emit logical then block */
      ctx->block = BB_then_logical;
      append_logical_start(BB_then_logical);
      visit_cf_list(ctx, &if_stmt->then_list);
      BB_then_logical = ctx->block;
      append_logical_end(BB_then_logical);

      /* branch from logical then block to between block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_between;
      BB_then_logical->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_then_logical, BB_between);
      if (!ctx->cf_info.parent_loop.has_divergent_branch)
         add_logical_edge(BB_then_logical, BB_endif);
      BB_then_logical->kind |= block_kind_uniform;

      assert(!ctx->cf_info.has_branch);
      bool then_branch_divergent = ctx->cf_info.parent_loop.has_divergent_branch;
      ctx->cf_info.parent_loop.has_divergent_branch = false;

      /** emit linear then block */
      BB_then_linear->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_then_linear);

      /* branch from linear then block to between block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_between;
      BB_then_linear->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_then_linear, BB_between);

      /** emit in-between merge block */
      BB_between->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_between);

      /* branch to linear else block (skip else) */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_nz, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(cond);
      branch->targets[0] = BB_else_linear;
      branch->targets[1] = BB_else_logical;
      BB_between->instructions.push_back(std::move(branch));
      add_linear_edge(BB_between, BB_else_logical);
      add_linear_edge(BB_between, BB_else_linear);

      /** emit logical else block */
      BB_else_logical->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else_logical);
      ctx->cf_info.parent_if.merge_block = BB_endif;
      ctx->block = BB_else_logical;
      append_logical_start(BB_else_logical);
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else_logical = ctx->block;
      append_logical_end(BB_else_logical);

      /* branch from logical else block to endif block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_endif;
      BB_else_logical->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_else_logical, BB_endif);
      if (!ctx->cf_info.parent_loop.has_divergent_branch)
         add_logical_edge(BB_else_logical, BB_endif);
      BB_else_logical->kind |= block_kind_uniform;

      assert(!ctx->cf_info.has_branch);
      ctx->cf_info.parent_loop.has_divergent_branch &= then_branch_divergent;

      /** emit linear else block */
      BB_else_linear->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else_linear);

      /* branch from linear else block to endif block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_endif;
      BB_else_linear->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_else_linear, BB_endif);

      /** emit endif merge block */
      BB_endif->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_endif);

      append_logical_start(BB_endif);
      ctx->block = BB_endif;
   }
}

static void visit_cf_list(isel_context *ctx,
                          struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         visit_block(ctx, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         visit_if(ctx, nir_cf_node_as_if(node));
         break;
      case nir_cf_node_loop:
         visit_loop(ctx, nir_cf_node_as_loop(node));
         break;
      default:
         unreachable("unimplemented cf list type");
      }
   }
}
} /* end namespace */

aco_ptr<Instruction> create_s_mov(Definition dst, Operand src) {
   if (src.isLiteral()) {
      uint32_t v = src.constantValue();
      if (v >= 0xffff8000 || v <= 0x7fff) {
         aco_ptr<Instruction> mov(create_instruction<SOPK_instruction>(aco_opcode::s_movk_i32, Format::SOPK, 0, 1));
         static_cast<SOPK_instruction*>(mov.get())->imm = v & 0xFFFF;
         mov->getDefinition(0) = dst;
         return mov;
      }
   }
   aco_ptr<Instruction> mov(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
   mov->getOperand(0) = src;
   mov->getDefinition(0) = dst;
   return mov;
}

void handle_bc_optimize(isel_context *ctx)
{
   /* needed when SPI_PS_IN_CONTROL.BC_OPTIMIZE_DISABLE is set to 0 */
   Builder bld(ctx->program, ctx->block);
   uint32_t spi_ps_input_ena = ctx->program->config->spi_ps_input_ena;
   bool uses_center = G_0286CC_PERSP_CENTER_ENA(spi_ps_input_ena) || G_0286CC_LINEAR_CENTER_ENA(spi_ps_input_ena);
   bool uses_centroid = G_0286CC_PERSP_CENTROID_ENA(spi_ps_input_ena) || G_0286CC_LINEAR_CENTROID_ENA(spi_ps_input_ena);
   if (uses_center && uses_centroid) {
      Temp sel = bld.vopc_e64(aco_opcode::v_cmp_lt_i32, bld.hint_vcc(bld.def(s2)), ctx->prim_mask, Operand(0u));

      if (G_0286CC_PERSP_CENTROID_ENA(spi_ps_input_ena)) {
         for (unsigned i = 0; i < 2; i++) {
            Temp new_coord = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                                      ctx->fs_inputs[fs_input::persp_centroid_p1 + i],
                                      ctx->fs_inputs[fs_input::persp_center_p1 + i],
                                      sel);
            ctx->fs_inputs[fs_input::persp_centroid_p1 + i] = new_coord;
         }
      }

      if (G_0286CC_LINEAR_CENTROID_ENA(spi_ps_input_ena)) {
         for (unsigned i = 0; i < 2; i++) {
            Temp new_coord = bld.vop2(aco_opcode::v_cndmask_b32, bld.def(v1),
                                      ctx->fs_inputs[fs_input::linear_centroid_p1 + i],
                                      ctx->fs_inputs[fs_input::linear_center_p1 + i],
                                      sel);
            ctx->fs_inputs[fs_input::linear_centroid_p1 + i] = new_coord;
         }
      }
   }
}

std::unique_ptr<Program> select_program(struct nir_shader *nir,
                                        ac_shader_config* config,
                                        struct radv_shader_variant_info *info,
                                        struct radv_nir_compiler_options *options)
{
   std::unique_ptr<Program> program{new Program};
   isel_context ctx = setup_isel_context(program.get(), nir, config, info, options);

   append_logical_start(ctx.block);

   if (ctx.stage == MESA_SHADER_FRAGMENT)
      handle_bc_optimize(&ctx);

   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   visit_cf_list(&ctx, &func->impl->body);

   append_logical_end(ctx.block);
   ctx.block->kind |= block_kind_uniform;
   Builder bld(ctx.program, ctx.block);
   if (ctx.program->wb_smem_l1_on_end)
      bld.smem(aco_opcode::s_dcache_wb, false);
   bld.sopp(aco_opcode::s_endpgm);

   return program;
}
}
