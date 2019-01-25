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
#include "aco_interface.h"
#include "aco_instruction_selection_setup.cpp"

namespace aco {
namespace {

class loop_info_RAII {
   isel_context* ctx;
   Temp orig_old;
   Temp active_old;
   Block* entry_old;
   Block* exit_old;
   bool divergent_cont_old;
   bool divergent_break_old;
   bool divergent_if_old;

public:
   loop_info_RAII(isel_context* ctx, Block* loop_entry, Block* loop_exit, Temp orig_exec, Temp active_mask)
      : ctx(ctx), orig_old(ctx->cf_info.parent_loop.orig_exec), active_old(ctx->cf_info.parent_loop.active_mask),
        entry_old(ctx->cf_info.parent_loop.entry), exit_old(ctx->cf_info.parent_loop.exit),
        divergent_cont_old(ctx->cf_info.parent_loop.has_divergent_continue),
        divergent_break_old(ctx->cf_info.parent_loop.has_divergent_break),
        divergent_if_old(ctx->cf_info.parent_if.is_divergent)
   {
      ctx->cf_info.parent_loop.entry = loop_entry;
      ctx->cf_info.parent_loop.exit = loop_exit;
      ctx->cf_info.parent_loop.orig_exec = orig_exec;
      ctx->cf_info.parent_loop.active_mask = active_mask;
      ctx->cf_info.parent_loop.has_divergent_continue = false;
      ctx->cf_info.parent_loop.has_divergent_break = false;
      ctx->cf_info.parent_if.is_divergent = false;
      ctx->cf_info.loop_nest_depth = ctx->cf_info.loop_nest_depth + 1;
   }

   ~loop_info_RAII()
   {
      ctx->cf_info.parent_loop.entry = entry_old;
      ctx->cf_info.parent_loop.exit = exit_old;
      ctx->cf_info.parent_loop.orig_exec = orig_old;
      ctx->cf_info.parent_loop.active_mask = active_old;
      ctx->cf_info.parent_loop.has_divergent_continue = divergent_cont_old;
      ctx->cf_info.parent_loop.has_divergent_break = divergent_break_old;
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
   if (b.isTemp() && typeOf(b.regClass()) != RegType::vgpr)
      std::swap(a, b);
   assert(typeOf(b.regClass()) == RegType::vgpr); // in case two SGPRs are given

   if (ctx->options->chip_class < GFX9)
      carry_out = true;

   aco_opcode op = carry_out ? aco_opcode::v_add_co_u32 : aco_opcode::v_add_u32;
   int num_defs = carry_out ? 2 : 1;

   Temp carry;

   aco_ptr<VOP2_instruction> add{create_instruction<VOP2_instruction>(op, Format::VOP2, 2, num_defs)};
   add->getOperand(0) = Operand(a);
   add->getOperand(1) = Operand(b);
   add->getDefinition(0) = Definition(dst);
   if (ctx->options->chip_class < GFX9) {
      carry = {ctx->program->allocateId(), s2};
      add->getDefinition(1) = Definition(carry);
      add->getDefinition(1).setHint(vcc);
   }
   ctx->block->instructions.emplace_back(std::move(add));

   return carry;
}

void emit_v_mov(isel_context *ctx, Temp src, Temp dst)
{
   aco_ptr<Instruction> mov;
   if (dst.size() == 1)
   {
      mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
      mov->getDefinition(0) = Definition(dst);
      mov->getOperand(0) = Operand(src);
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 1, 1)};
      vec->getOperand(0) = Operand(src);
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
   }
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
   aco_ptr<Instruction> extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   extract->getOperand(0) = Operand(src);
   extract->getOperand(1) = Operand(idx);
   extract->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(extract));
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
      } else {
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
      aco_ptr<Instruction> tmp{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      tmp->getOperand(0) = Operand(ptr);
      tmp->getOperand(1) = Operand((unsigned)ctx->options->address32_hi);
      Temp ptr64 = {ctx->program->allocateId(), s2};
      tmp->getDefinition(0) = Definition(ptr64);
      ctx->block->instructions.emplace_back(std::move(tmp));
      return ptr64;
}

void emit_sop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst, bool scc)
{
   aco_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(op, Format::SOP2, 2, scc ? 2 : 1)};
   sop2->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
   sop2->getOperand(1) = Operand(get_alu_src(ctx, instr->src[1]));
   sop2->getDefinition(0) = Definition(dst);
   if (scc)
      sop2->getDefinition(1) = Definition(PhysReg{253}, b);
   ctx->block->instructions.emplace_back(std::move(sop2));
}

void emit_sopc_instruction_output32(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   aco_ptr<SOPC_instruction> cmp{create_instruction<SOPC_instruction>(op, Format::SOPC, 2, 1)};
   cmp->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
   cmp->getOperand(1) = Operand(get_alu_src(ctx, instr->src[1]));
   Temp scc = {ctx->program->allocateId(), b};
   cmp->getDefinition(0) = Definition(scc);
   cmp->getDefinition(0).setFixed({253}); /* scc */
   ctx->block->instructions.emplace_back(std::move(cmp));
   aco_ptr<SOP2_instruction> to_sgpr{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
   to_sgpr->getOperand(0) = Operand(0xFFFFFFFF);
   to_sgpr->getOperand(1) = Operand((uint32_t) 0);
   to_sgpr->getOperand(2) = Operand(scc);
   to_sgpr->getOperand(2).setFixed({253}); /* scc */
   to_sgpr->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(to_sgpr));
}

void emit_vop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst, bool commutative, bool swap_srcs=false)
{
   Temp src0 = get_alu_src(ctx, instr->src[swap_srcs ? 1 : 0]);
   Temp src1 = get_alu_src(ctx, instr->src[swap_srcs ? 0 : 1]);
   aco_ptr<Instruction> vop2{create_instruction<VOP2_instruction>(op, Format::VOP2, 2, 1)};
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
         Format format = (Format) ((int) Format::VOP2 | (int) Format::VOP3A);
         vop2.reset(create_instruction<VOP3A_instruction>(op, format, 2, 1));
      } else {
         Temp mov_dst = Temp(ctx->program->allocateId(), getRegClass(vgpr, src1.size()));
         emit_v_mov(ctx, src1, mov_dst);
         src1 = mov_dst;
      }
   }
   vop2->getOperand(0) = Operand{src0};
   vop2->getOperand(1) = Operand{src1};
   vop2->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(vop2));
}

void emit_vop1_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   aco_ptr<VOP1_instruction> vop1{create_instruction<VOP1_instruction>(op, Format::VOP1, 1, 1)};
   vop1->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
   vop1->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(vop1));
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
            default: /* eq and ne are commutative */
               break;
         }
         Temp t = src0;
         src0 = src1;
         src1 = t;
      } else {
         Temp vgpr_src = {ctx->program->allocateId(), getRegClass(vgpr, src1.size())};
         emit_v_mov(ctx, src1, vgpr_src);
         src1 = vgpr_src;
      }
   }
   vopc.reset(create_instruction<VOPC_instruction>(op, Format::VOPC, 2, 1));
   vopc->getOperand(0) = Operand(src0);
   vopc->getOperand(1) = Operand(src1);
   vopc->getDefinition(0) = Definition(dst);
   vopc->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(vopc));
}

void emit_vopc_instruction_output32(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   Temp tmp{ctx->program->allocateId(), s2};

   emit_vopc_instruction(ctx, instr, op, tmp);

   if (dst.regClass() == v1) {
      aco_ptr<Instruction> bcsel{create_instruction<VOP3A_instruction>(aco_opcode::v_cndmask_b32, static_cast<Format>((int)Format::VOP2 | (int)Format::VOP3A), 3, 1)};
      bcsel->getOperand(0) = Operand((uint32_t) 0);
      bcsel->getOperand(1) = Operand((uint32_t) -1);
      bcsel->getOperand(2) = Operand{tmp};
      bcsel->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(bcsel));
   } else {
      Temp scc_tmp{ctx->program->allocateId(), b};
      aco_ptr<Instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lg_u64, Format::SOPC, 2, 1)};
      cmp->getOperand(0) = Operand{tmp};
      cmp->getOperand(1) = Operand((uint32_t) 0);
      cmp->getDefinition(0) = Definition{scc_tmp};
      cmp->getDefinition(0).setFixed({253}); /* scc */
      ctx->block->instructions.emplace_back(std::move(cmp));

      aco_ptr<Instruction> cselect{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
      cselect->getOperand(0) = Operand((uint32_t) -1);
      cselect->getOperand(1) = Operand((uint32_t) 0);
      cselect->getOperand(2) = Operand{scc_tmp};
      cselect->getOperand(2).setFixed({253}); /* scc */
      cselect->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(cselect));
   }
}


Temp extract_uniform_cond32(isel_context *ctx, Temp cond32)
{
   Temp cond = Temp{ctx->program->allocateId(), b};

   aco_ptr<Instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lg_u32, Format::SOPC, 2, 1)};
   cmp->getOperand(0) = Operand{cond32};
   cmp->getOperand(1) = Operand((uint32_t) 0);
   cmp->getDefinition(0) = Definition{cond};
   cmp->getDefinition(0).setFixed(PhysReg{253}); /* scc */
   ctx->block->instructions.emplace_back(std::move(cmp));

   return cond;
}

Temp extract_divergent_cond32(isel_context *ctx, Temp cond32)
{
   Temp cond = Temp{ctx->program->allocateId(), s2};

   aco_ptr<Instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_lg_u32, Format::VOPC, 2, 1)};
   cmp->getOperand(0) = Operand((uint32_t) 0);
   if (cond32.type() == sgpr) {
      Temp vgpr_cond = {ctx->program->allocateId(), v1};
      emit_v_mov(ctx, cond32, vgpr_cond);
      cond32 = vgpr_cond;
   }
   cmp->getOperand(1) = Operand{cond32};
   cmp->getDefinition(0) = Definition{cond};
   cmp->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(cmp));

   return cond;
}

void emit_quad_swizzle(isel_context *ctx, Temp src, Temp dst,
                       unsigned lane0, unsigned lane1, unsigned lane2, unsigned lane3)
{
   unsigned quad_mask = lane0 | (lane1 << 2) | (lane2 << 4) | (lane3 << 6);
   aco_ptr<DPP_instruction> dpp;
   Format format = (Format) ((uint32_t) Format::VOP1 | (uint32_t) Format::DPP);
   dpp.reset(create_instruction<DPP_instruction>(aco_opcode::v_mov_b32, format, 1, 1));
   dpp->dpp_ctrl = quad_mask;
   dpp->row_mask = 0xF;
   dpp->bank_mask = 0xF;
   dpp->getOperand(0) = Operand(src);
   dpp->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(dpp));
}

void emit_bcsel(isel_context *ctx, nir_alu_instr *instr, Temp dst)
{
   Temp cond32 = get_alu_src(ctx, instr->src[0]);
   Temp then = get_alu_src(ctx, instr->src[1]);
   Temp els = get_alu_src(ctx, instr->src[2]);

   if (dst.type() == vgpr) {
      Temp cond;
      if (cond32.type() == vgpr) {
         cond = extract_divergent_cond32(ctx, cond32);
      } else {
         Temp scc_tmp = extract_uniform_cond32(ctx, cond32);
         aco_ptr<Instruction> cselect{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b64, Format::SOP2, 3, 1)};
         cselect->getOperand(0) = Operand((uint32_t) -1);
         cselect->getOperand(1) = Operand((uint32_t) 0);
         cselect->getOperand(2) = Operand{scc_tmp};
         cselect->getOperand(2).setFixed({253});
         cond = Temp{ctx->program->allocateId(), s2};
         cselect->getDefinition(0) = Definition(cond);
         cselect->getDefinition(0).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(cselect));
      }
      if (dst.type() == vgpr) {
         aco_ptr<Instruction> bcsel;
         if (dst.size() == 1) {
            then = as_vgpr(ctx, then);
            els = as_vgpr(ctx, els);

            bcsel.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
            bcsel->getOperand(0) = Operand{els};
            bcsel->getOperand(1) = Operand{then};
            bcsel->getOperand(2) = Operand{cond};
            bcsel->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(bcsel));
         } else if (dst.regClass() == v2) {
            emit_split_vector(ctx, then, 2);
            emit_split_vector(ctx, els, 2);

            bcsel.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
            bcsel->getOperand(0) = Operand{emit_extract_vector(ctx, els, 0, v1)};
            bcsel->getOperand(1) = Operand{emit_extract_vector(ctx, then, 0, v1)};
            bcsel->getOperand(2) = Operand{cond};
            Temp dst0 = {ctx->program->allocateId(), v1};
            bcsel->getDefinition(0) = Definition(dst0);
            ctx->block->instructions.emplace_back(std::move(bcsel));

            bcsel.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
            bcsel->getOperand(0) = Operand{emit_extract_vector(ctx, els, 1, v1)};
            bcsel->getOperand(1) = Operand{emit_extract_vector(ctx, then, 1, v1)};
            bcsel->getOperand(2) = Operand{cond};
            Temp dst1 = {ctx->program->allocateId(), v1};
            bcsel->getDefinition(0) = Definition(dst1);
            ctx->block->instructions.emplace_back(std::move(bcsel));

            bcsel.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1));
            bcsel->getOperand(0) = Operand(dst0);
            bcsel->getOperand(1) = Operand(dst1);
            bcsel->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(bcsel));
         } else {
            fprintf(stderr, "Unimplemented NIR instr bit size: ");
            nir_print_instr(&instr->instr, stderr);
            fprintf(stderr, "\n");
         }
      } else { /* dst.type() == sgpr */
         unreachable("Are 1-bit Bools enabled... ?");

         /* this implements bcsel on bools: dst = s0 ? s1 : s2
          * are going to be: dst = (s0 & s1) | (~s0 & s2) */
         assert(cond.regClass() == s2 && then.regClass() == s2 && els.regClass() == s2);

         aco_ptr<SOP2_instruction> sop2;
         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b64, Format::SOP2, 2, 2));
         sop2->getOperand(0) = Operand(cond);
         sop2->getOperand(1) = Operand(then);
         then = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(then);
         sop2->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(sop2));

         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2));
         sop2->getOperand(0) = Operand(els);
         sop2->getOperand(1) = Operand(cond);
         els = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(els);
         sop2->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(sop2));

         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 2));
         sop2->getOperand(0) = Operand(then);
         sop2->getOperand(1) = Operand(els);
         then = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(dst);
         sop2->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(sop2));
      }
   } else { /* condition is uniform */
      Temp cond;
      if (cond32.type() == vgpr) {
         cond = extract_divergent_cond32(ctx, cond32);
      } else {
         cond = extract_uniform_cond32(ctx, cond32);
      }
      if (cond.regClass() == s2) {
         cond = emit_extract_vector(ctx, cond, 0, s1);
         aco_ptr<SOPK_instruction> sopk{create_instruction<SOPK_instruction>(aco_opcode::s_cmpk_lg_u32, Format::SOPK, 1, 1)};
         sopk->getOperand(0) = Operand(cond);
         sopk->imm = 0;
         cond = {ctx->program->allocateId(), b};
         sopk->getDefinition(0) = Definition(cond);
         sopk->getDefinition(0).setFixed(PhysReg{253}); /* scc */
         ctx->block->instructions.emplace_back(std::move(sopk));
      }
      assert(cond.regClass() == b);
      if (dst.regClass() == s1 || dst.regClass() == s2) {
         assert((then.regClass() == s1 || then.regClass() == s2) && els.regClass() == then.regClass());
         aco_ptr<SOP2_instruction> select{create_instruction<SOP2_instruction>(dst.size() == 2 ? aco_opcode::s_cselect_b64 : aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
         select->getOperand(0) = Operand(then);
         select->getOperand(1) = Operand(els);
         select->getOperand(2) = Operand(cond);
         select->getOperand(2).setFixed({253}); /* scc */
         select->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(select));
      } else {
         fprintf(stderr, "Unimplemented uniform bcsel bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
   }
}

void emit_udiv(isel_context* ctx, Temp src0, Temp src1, Temp dst)
{
   // FIXME: this algorithm is wrong in the general case, but works most of the time.
   aco_ptr<Instruction> instr;

   instr.reset(create_instruction<VOP1_instruction>(aco_opcode::v_cvt_f32_u32, Format::VOP1, 1, 1));
   instr->getOperand(0) = Operand(src0);
   Temp f_src0 = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(f_src0);
   ctx->block->instructions.emplace_back(std::move(instr));

   instr.reset(create_instruction<VOP1_instruction>(aco_opcode::v_cvt_f32_u32, Format::VOP1, 1, 1));
   instr->getOperand(0) = Operand(src1);
   Temp f_src1 = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(f_src1);
   ctx->block->instructions.emplace_back(std::move(instr));

   instr.reset(create_instruction<VOP1_instruction>(aco_opcode::v_rcp_iflag_f32, Format::VOP1, 1, 1));
   instr->getOperand(0) = Operand(f_src1);
   Temp rcp = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(rcp);
   ctx->block->instructions.emplace_back(std::move(instr));

   instr.reset(create_instruction<VOP2_instruction>(aco_opcode::v_mul_f32, Format::VOP2, 2, 1));
   instr->getOperand(0) = Operand(f_src0);
   instr->getOperand(1) = Operand(rcp);
   Temp f_dst = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(f_dst);
   ctx->block->instructions.emplace_back(std::move(instr));

   instr.reset(create_instruction<VOP1_instruction>(aco_opcode::v_cvt_u32_f32, Format::VOP1, 1, 1));
   instr->getOperand(0) = Operand(f_dst);
   instr->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(instr));

}

void visit_alu_instr(isel_context *ctx, nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa) {
      fprintf(stderr, "nir alu dst not in ssa: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }
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
   case nir_op_imov:
   case nir_op_fmov: {
      aco_ptr<Instruction> mov;
      if (dst.regClass() == s1) {
         mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
      } else if (dst.regClass() == v1) {
         mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      mov->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
      mov->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(mov));
      break;
   }
   case nir_op_inot: {
      if (dst.regClass() == v1) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_not_b32, dst);
      } else if (dst.type() == sgpr) {
         aco_opcode opcode = dst.size() == 1 ? aco_opcode::s_not_b32 : aco_opcode::s_not_b64;
         aco_ptr<Instruction> sop1{create_instruction<SOP1_instruction>(opcode, Format::SOP1, 1, 2)};
         sop1->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         sop1->getDefinition(0) = Definition(dst);
         sop1->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(sop1));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ineg: {
      if (dst.regClass() == v1) {
         aco_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_mul_lo_u32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0xFFFFFFFF);
         vop2->getOperand(1) = Operand(get_alu_src(ctx, instr->src[0]));
         vop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop2));
      } else if (dst.regClass() == s1) {
         aco_ptr<SOPK_instruction> sopk{create_instruction<SOPK_instruction>(aco_opcode::s_mulk_i32, Format::SOPK, 1, 1)};
         sopk->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
         sopk->getDefinition(0) = Definition(dst);
         sopk->imm = 0xFFFF;
         ctx->block->instructions.emplace_back(std::move(sopk));
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
         aco_ptr<Instruction> bcsel;
         bcsel.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
         bcsel->getOperand(0) = Operand(emit_extract_vector(ctx, src0, 0, v1));
         bcsel->getOperand(1) = Operand(emit_extract_vector(ctx, src1, 0, v1));
         bcsel->getOperand(2) = Operand(cmp);
         Temp dst0 = {ctx->program->allocateId(), v1};
         bcsel->getDefinition(0) = Definition(dst0);
         ctx->block->instructions.emplace_back(std::move(bcsel));
         bcsel.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
         bcsel->getOperand(0) = Operand(emit_extract_vector(ctx, src0, 1, v1));
         bcsel->getOperand(1) = Operand(emit_extract_vector(ctx, src1, 1, v1));
         bcsel->getOperand(2) = Operand(cmp);
         Temp dst1 = {ctx->program->allocateId(), v1};
         bcsel->getDefinition(0) = Definition(dst1);
         ctx->block->instructions.emplace_back(std::move(bcsel));
         bcsel.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1));
         bcsel->getOperand(0) = Operand(dst0);
         bcsel->getOperand(1) = Operand(dst1);
         bcsel->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bcsel));
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
      if (dst.regClass() == v1) {
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
      if (dst.regClass() == v1) {
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
      if (dst.regClass() == v1) {
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
         aco_ptr<VOP2_instruction> shl{create_instruction<VOP2_instruction>(aco_opcode::v_lshrrev_b32, Format::VOP2, 2, 1)};
         shl->getOperand(0) = Operand{get_alu_src(ctx, instr->src[1])};
         shl->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
         shl->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(shl));
      } else if (dst.regClass() == s1) {
         aco_ptr<SOP2_instruction> shl{create_instruction<SOP2_instruction>(aco_opcode::s_lshr_b32, Format::SOP2, 2, 2)};
         shl->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         shl->getOperand(1) = Operand{get_alu_src(ctx, instr->src[1])};
         shl->getDefinition(0) = Definition(dst);
         Temp t = {ctx->program->allocateId(), b};
         shl->getDefinition(1) = Definition(t);
         shl->getDefinition(1).setFixed(PhysReg{253}); /* scc */
         ctx->block->instructions.emplace_back(std::move(shl));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ishl: {
      if (dst.regClass() == v1) {
         aco_ptr<VOP2_instruction> shl{create_instruction<VOP2_instruction>(aco_opcode::v_lshlrev_b32, Format::VOP2, 2, 1)};
         shl->getOperand(0) = Operand{get_alu_src(ctx, instr->src[1])};
         shl->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
         shl->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(shl));
      } else if (dst.regClass() == s1) {
         aco_ptr<SOP2_instruction> shl{create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 2)};
         shl->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         shl->getOperand(1) = Operand{get_alu_src(ctx, instr->src[1])};
         shl->getDefinition(0) = Definition(dst);
         Temp t = {ctx->program->allocateId(), b};
         shl->getDefinition(1) = Definition(t);
         shl->getDefinition(1).setFixed(PhysReg{253}); /* scc */
         ctx->block->instructions.emplace_back(std::move(shl));
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
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_ashr_i32, dst, true);
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

         Temp dst0 = {ctx->program->allocateId(), v1};
         Temp carry = emit_v_add32(ctx, dst0, Operand(src00), Operand(src10), true);

         add.reset(create_instruction<VOP2_instruction>(aco_opcode::v_addc_co_u32, Format::VOP2, 3, 2));
         Temp src01 = emit_extract_vector(ctx, src0, 1, getRegClass(src0.type(), 1));
         Temp src11 = emit_extract_vector(ctx, src1, 1, getRegClass(src1.type(), 1));
         add->getOperand(0) = Operand(src01);
         add->getOperand(1) = Operand(src11);
         add->getOperand(2) = Operand(carry);
         Temp dst1 = {ctx->program->allocateId(), v1};
         add->getDefinition(0) = Definition(dst1);
         Temp tmp = {ctx->program->allocateId(), s2};
         add->getDefinition(1) = Definition(tmp);
         add->getDefinition(1).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(add));

         add.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1));
         add->getOperand(0) = Operand(dst0);
         add->getOperand(1) = Operand(dst1);
         add->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(add));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_isub: {
      if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_sub_i32, dst, false);
         break;
      }
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);

      if (dst.regClass() == v1 && ctx->options->chip_class >= GFX9) {
         if (src1.type() == vgpr)
            emit_vop2_instruction(ctx, instr, aco_opcode::v_sub_u32, dst, false);
         else
            emit_vop2_instruction(ctx, instr, aco_opcode::v_subrev_u32, dst, true);
         break;
      }
      aco_opcode op = aco_opcode::v_sub_co_u32;
      aco_opcode opc = aco_opcode::v_subb_co_u32;
      if (src1.type() != vgpr) {
         op = aco_opcode::v_subrev_co_u32;
         opc = aco_opcode::v_subbrev_co_u32;
         Temp t = src0;
         src0 = src1;
         src1 = t;
      }

      aco_ptr<Instruction> sub;
      if (dst.regClass() == v1) {
         sub.reset(create_instruction<VOP2_instruction>(op, Format::VOP2, 2, 2));
         sub->getOperand(0) = Operand(src0);
         sub->getOperand(1) = Operand(src1);
         sub->getDefinition(0) = Definition(dst);
         Temp tmp = {ctx->program->allocateId(), s2};
         sub->getDefinition(1) = Definition(tmp);
         sub->getDefinition(1).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(sub));
      } else if (dst.regClass() == v2) {
         assert(src0.size() == 2 && src1.size() == 2);
         emit_split_vector(ctx, src0, 2);
         emit_split_vector(ctx, src1, 2);
         Temp src00 = emit_extract_vector(ctx, src0, 0, getRegClass(src0.type(), 1));
         Temp src10 = emit_extract_vector(ctx, src1, 0, getRegClass(src1.type(), 1));

         sub.reset(create_instruction<VOP2_instruction>(op, Format::VOP2, 2, 2));
         sub->getOperand(0) = Operand(src00);
         sub->getOperand(1) = Operand(src10);
         Temp dst0 = {ctx->program->allocateId(), v1};
         sub->getDefinition(0) = Definition(dst0);
         Temp tmp = {ctx->program->allocateId(), s2};
         sub->getDefinition(1) = Definition(tmp);
         sub->getDefinition(1).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(sub));

         sub.reset(create_instruction<VOP2_instruction>(opc, Format::VOP2, 3, 2));
         Temp src01 = emit_extract_vector(ctx, src0, 1, getRegClass(src0.type(), 1));
         Temp src11 = emit_extract_vector(ctx, src1, 1, getRegClass(src1.type(), 1));
         sub->getOperand(0) = Operand(src01);
         sub->getOperand(1) = Operand(src11);
         sub->getOperand(2) = Operand(tmp);
         Temp dst1 = {ctx->program->allocateId(), v1};
         sub->getDefinition(0) = Definition(dst1);
         tmp = {ctx->program->allocateId(), s2};
         sub->getDefinition(1) = Definition(tmp);
         sub->getDefinition(1).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(sub));

         sub.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1));
         sub->getOperand(0) = Operand(dst0);
         sub->getOperand(1) = Operand(dst1);
         sub->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(sub));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_imul: {
      if (dst.regClass() == v1) {
         aco_ptr<VOP3A_instruction> mul{create_instruction<VOP3A_instruction>(aco_opcode::v_mul_lo_u32, Format::VOP3A, 2, 1)};
         mul->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         mul->getOperand(1) = Operand{get_alu_src(ctx, instr->src[1])};
         mul->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(mul));
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
         aco_ptr<VOP3A_instruction> mul{create_instruction<VOP3A_instruction>(aco_opcode::v_mul_hi_u32, Format::VOP3A, 2, 1)};
         mul->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         mul->getOperand(1) = Operand{get_alu_src(ctx, instr->src[1])};
         mul->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(mul));
      } else if (dst.regClass() == s1) {
         aco_ptr<VOP3A_instruction> mul{create_instruction<VOP3A_instruction>(aco_opcode::v_mul_hi_u32, Format::VOP3A, 2, 1)};
         mul->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         mul->getOperand(1) = Operand{as_vgpr(ctx, get_alu_src(ctx, instr->src[1]))};
         Temp vgpr_dst = {ctx->program->allocateId(), v1};
         mul->getDefinition(0) = Definition(vgpr_dst);
         ctx->block->instructions.emplace_back(std::move(mul));
         aco_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
         readlane->getOperand(0) = Operand(vgpr_dst);
         readlane->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(readlane));
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
   case nir_op_fmod: {
      if (dst.size() == 1) {
         Temp rcp = {ctx->program->allocateId(), v1};
         aco_ptr<Instruction> inst;
         inst.reset(create_instruction<VOP1_instruction>(aco_opcode::v_rcp_f32, Format::VOP1, 1, 1));
         inst->getOperand(0) = Operand(get_alu_src(ctx, instr->src[1]));
         inst->getDefinition(0) = Definition(rcp);
         ctx->block->instructions.emplace_back(std::move(inst));

         Temp mul = {ctx->program->allocateId(), v1};
         inst.reset(create_instruction<VOP2_instruction>(aco_opcode::v_mul_f32, Format::VOP2, 2, 1));
         inst->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
         inst->getOperand(1) = Operand(rcp);
         inst->getDefinition(0) = Definition(mul);
         ctx->block->instructions.emplace_back(std::move(inst));

         Temp floor = {ctx->program->allocateId(), v1};
         inst.reset(create_instruction<VOP1_instruction>(aco_opcode::v_floor_f32, Format::VOP1, 1, 1));
         inst->getOperand(0) = Operand(mul);
         inst->getDefinition(0) = Definition(floor);
         ctx->block->instructions.emplace_back(std::move(inst));

         mul = {ctx->program->allocateId(), v1};
         inst.reset(create_instruction<VOP2_instruction>(aco_opcode::v_mul_f32, Format::VOP2, 2, 1));
         inst->getOperand(0) = Operand(get_alu_src(ctx, instr->src[1]));
         inst->getOperand(1) = Operand(floor);
         inst->getDefinition(0) = Definition(mul);
         ctx->block->instructions.emplace_back(std::move(inst));

         inst.reset(create_instruction<VOP2_instruction>(aco_opcode::v_subrev_f32, Format::VOP2, 2, 1));
         inst->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
         inst->getOperand(1) = Operand(mul);
         inst->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(inst));
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
   case nir_op_b32csel: {
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
         aco_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_sub_f32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0);
         vop2->getOperand(1) = Operand(as_vgpr(ctx, src));
         vop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop2));
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
         aco_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_and_b32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0x7FFFFFFF);
         vop2->getOperand(1) = Operand(as_vgpr(ctx, src));
         vop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop2));
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
         aco_ptr<VOP3A_instruction> vop3{create_instruction<VOP3A_instruction>(aco_opcode::v_med3_f32, Format::VOP3A, 3, 1)};
         vop3->getOperand(0) = Operand((uint32_t) 0);
         vop3->getOperand(1) = Operand((uint32_t) 0x3f800000);
         vop3->getOperand(2) = Operand(src);
         vop3->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop3));
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
         if (src.type() == sgpr) {
            Format format = (Format) ((int) Format::VOP3A | (int) Format::VOP2);
            norm.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_mul_f32, format, 2, 1));
         } else
            norm.reset(create_instruction<VOP2_instruction>(aco_opcode::v_mul_f32, Format::VOP2, 2, 1));
         norm->getOperand(0) = Operand((uint32_t) 0x3e22f983); /* 1/2*PI */
         norm->getOperand(1) = Operand(src);
         Temp tmp = Temp(ctx->program->allocateId(), v1);
         norm->getDefinition(0) = Definition(tmp);
         ctx->block->instructions.emplace_back(std::move(norm));

         aco_opcode opcode = instr->op == nir_op_fsin ? aco_opcode::v_sin_f32 : aco_opcode::v_cos_f32;
         aco_ptr<VOP1_instruction> vop1{create_instruction<VOP1_instruction>(opcode, Format::VOP1, 1, 1)};
         vop1->getOperand(0) = Operand(tmp);
         vop1->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop1));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fsign: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      assert(src.type() == vgpr);
      if (dst.size() == 1) {
         aco_ptr<VOPC_instruction> vopc{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_nlt_f32, Format::VOPC, 2, 1)};
         vopc->getOperand(0) = Operand((uint32_t) 0);
         vopc->getOperand(1) = Operand(src);
         Temp temp = Temp(ctx->program->allocateId(), s2);
         vopc->getDefinition(0) = Definition(temp);
         vopc->getDefinition(0).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(vopc));

         aco_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0x3f800000);
         vop2->getOperand(1) = Operand(src);
         vop2->getOperand(2) = Operand(temp);
         src = Temp(ctx->program->allocateId(), v1);
         vop2->getDefinition(0) = Definition(src);
         ctx->block->instructions.emplace_back(std::move(vop2));

         vopc.reset(create_instruction<VOPC_instruction>(aco_opcode::v_cmp_le_f32, Format::VOPC, 2, 1));
         vopc->getOperand(0) = Operand((uint32_t) 0);
         vopc->getOperand(1) = Operand(src);
         temp = Temp(ctx->program->allocateId(), s2);
         vopc->getDefinition(0) = Definition(temp);
         vopc->getDefinition(0).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(vopc));

         vop2.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
         vop2->getOperand(0) = Operand((uint32_t) 0xbf800000);
         vop2->getOperand(1) = Operand(src);
         vop2->getOperand(2) = Operand(temp);
         vop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop2));
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
         Temp tmp = {ctx->program->allocateId(), v1};
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, tmp);
         aco_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
         readlane->getOperand(0) = Operand(tmp);
         readlane->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(readlane));
      } else {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, dst);
      }
      break;
   }
   case nir_op_f2u32: {
      if (dst.regClass() == s1) {
         Temp tmp = {ctx->program->allocateId(), v1};
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_u32_f32, tmp);
         aco_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
         readlane->getOperand(0) = Operand(tmp);
         readlane->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(readlane));
      } else {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_u32_f32, dst);
      }
      break;
   }
   case nir_op_b2f32: {
      Temp cond32 = get_alu_src(ctx, instr->src[0]);
      aco_ptr<VOP3A_instruction> cndmask{create_instruction<VOP3A_instruction>(aco_opcode::v_and_b32, (Format) ((int) Format::VOP3A | (int) Format::VOP2), 2, 1)};
      cndmask->getOperand(0) = Operand(cond32);
      cndmask->getOperand(1) = Operand((uint32_t) 0x3f800000); /* 1.0 */
      cndmask->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(cndmask));
      break;
   }
   case nir_op_u2u32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (src.size() == 2) {
         /* we could actually just say dst = src, as it would map the lower register */
         emit_extract_vector(ctx, src, 0, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_i2b32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s1) {
         if (src.regClass() == s1) {
            aco_ptr<Instruction> mov{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1)};
            mov->getOperand(0) = Operand(src);
            mov->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(mov));

         } else {
            fprintf(stderr, "Unimplemented NIR instr bit size: ");
            nir_print_instr(&instr->instr, stderr);
            fprintf(stderr, "\n");
         }
      } else {
         assert(dst.regClass() == v1);
         emit_v_mov(ctx, get_alu_src(ctx, instr->src[0]), dst);
      }
      break;
   }
   case nir_op_b2i32: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      if (dst.regClass() == s1) {
         assert(src.regClass() == s1);
         Temp scc_tmp = extract_uniform_cond32(ctx, src);
         aco_ptr<Instruction> cselect{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
         cselect->getOperand(0) = Operand((uint32_t) 1);
         cselect->getOperand(1) = Operand((uint32_t) 0);
         cselect->getOperand(2) = Operand{scc_tmp};
         cselect->getOperand(2).setFixed({253});
         cselect->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(cselect));
      } else if (dst.regClass() == v1) {
         assert(src.regClass() == v1);
         Temp tmp = extract_divergent_cond32(ctx, src);
         Format format = (Format) ((uint32_t) Format::VOP2 | (uint32_t) Format::VOP3A);
         aco_ptr<Instruction> bcsel{create_instruction<VOP3A_instruction>(aco_opcode::v_cndmask_b32, format, 3, 1)};
         bcsel->getOperand(0) = Operand((uint32_t) 0);
         bcsel->getOperand(1) = Operand((uint32_t) 1);
         bcsel->getOperand(2) = Operand{tmp};
         bcsel->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bcsel));

      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_pack_64_2x32_split: {
      Temp src0 = get_alu_src(ctx, instr->src[0]);
      Temp src1 = get_alu_src(ctx, instr->src[1]);

      aco_ptr<Instruction> tmp{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      tmp->getOperand(0) = Operand(src0);
      tmp->getOperand(1) = Operand(src1);
      tmp->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(tmp));
      break;
   }
   case nir_op_pack_half_2x16: {
      Temp src = get_ssa_temp(ctx, instr->src[0].src.ssa);
      emit_split_vector(ctx, src, 2);

      if (dst.regClass() == v1) {
         Temp src0 = emit_extract_vector(ctx, src, 0, v1);
         Temp src1 = emit_extract_vector(ctx, src, 1, v1);
         aco_ptr<Instruction> vop3{create_instruction<VOP3A_instruction>(aco_opcode::v_cvt_pkrtz_f16_f32, Format::VOP3A, 2, 1)};
         vop3->getOperand(0) = Operand(src0);
         vop3->getOperand(1) = Operand(src1);
         vop3->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop3));

      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_bfm: {
      Temp bits = get_alu_src(ctx, instr->src[0]);
      Temp offset = get_alu_src(ctx, instr->src[1]);

      if (dst.regClass() == s1) {
         aco_ptr<Instruction> bfm{create_instruction<SOP2_instruction>(aco_opcode::s_bfm_b32, Format::SOP2, 2, 1)};
         bfm->getOperand(0) = Operand(bits);
         bfm->getOperand(1) = Operand(offset);
         bfm->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bfm));
      } else if (dst.regClass() == v1) {
         aco_ptr<Instruction> bfm{create_instruction<VOP3A_instruction>(aco_opcode::v_bfm_b32, Format::VOP3A, 2, 1)};
         bfm->getOperand(0) = Operand(bits);
         bfm->getOperand(1) = Operand(offset);
         bfm->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bfm));
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
         Temp scc_tmp;
         nir_const_value* const_bitmask = nir_src_as_const_value(instr->src[0].src);
         nir_const_value* const_insert = nir_src_as_const_value(instr->src[1].src);
         Operand lhs;
         if (const_insert && const_bitmask) {
            lhs = Operand(const_insert->u32[0] & const_bitmask->u32[0]);
         } else {
            sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b32, Format::SOP2, 2, 2));
            sop2->getOperand(0) = Operand(insert);
            sop2->getOperand(1) = Operand(bitmask);
            insert = {ctx->program->allocateId(), s1};
            sop2->getDefinition(0) = Definition(insert);
            scc_tmp = {ctx->program->allocateId(), b};
            sop2->getDefinition(1) = Definition(scc_tmp);
            sop2->getDefinition(1).setFixed(PhysReg{253});
            ctx->block->instructions.emplace_back(std::move(sop2));
            lhs = Operand(insert);
         }

         Operand rhs;
         nir_const_value* const_base = nir_src_as_const_value(instr->src[3].src);
         if (const_base && const_bitmask) {
            rhs = Operand(const_base->u32[0] & ~const_bitmask->u32[0]);
         } else {
            sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b32, Format::SOP2, 2, 2));
            sop2->getOperand(0) = Operand(base);
            sop2->getOperand(1) = Operand(bitmask);
            base = {ctx->program->allocateId(), s1};
            sop2->getDefinition(0) = Definition(base);
            scc_tmp = {ctx->program->allocateId(), b};
            sop2->getDefinition(1) = Definition(scc_tmp);
            sop2->getDefinition(1).setFixed(PhysReg{253});
            ctx->block->instructions.emplace_back(std::move(sop2));
            rhs = Operand(base);
         }

         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b32, Format::SOP2, 2, 2));
         sop2->getOperand(0) = rhs;
         sop2->getOperand(1) = lhs;
         sop2->getDefinition(0) = Definition(dst);
         scc_tmp = {ctx->program->allocateId(), b};
         sop2->getDefinition(1) = Definition(scc_tmp);
         sop2->getDefinition(1).setFixed(PhysReg{253});
         ctx->block->instructions.emplace_back(std::move(sop2));

      } else if (dst.regClass() == v1) {
         if (base.type() == sgpr && (bitmask.type() == sgpr || (insert.type() == sgpr)))
            base = as_vgpr(ctx, base);
         if (insert.type() == sgpr && bitmask.type() == sgpr)
            insert = as_vgpr(ctx, insert);

         aco_ptr<Instruction> bfi{create_instruction<VOP3A_instruction>(aco_opcode::v_bfi_b32, Format::VOP3A, 3, 1)};
         bfi->getOperand(0) = Operand(bitmask);
         bfi->getOperand(1) = Operand(insert);
         bfi->getOperand(2) = Operand(base);
         bfi->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bfi));

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
            uint32_t const_extract = (const_bits->u32[0] << 16) | const_offset->u32[0];
            extract = Operand(const_extract);
         } else {
            Operand width;
            if (const_bits) {
               width = Operand(const_bits->u32[0] << 16);
            } else {
               aco_ptr<Instruction> shift{create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 2)};
               shift->getOperand(0) = Operand(bits);
               shift->getOperand(1) = Operand((uint32_t) 16);
               Temp tmp = {ctx->program->allocateId(), s1};
               shift->getDefinition(0) = Definition(tmp);
               Temp scc_tmp = {ctx->program->allocateId(), b};
               shift->getDefinition(1) = Definition(scc_tmp);
               shift->getDefinition(1).setFixed(PhysReg{253});
               ctx->block->instructions.emplace_back(std::move(shift));
               width = Operand(tmp);
            }
            aco_ptr<Instruction> sop2{create_instruction<SOP2_instruction>(aco_opcode::s_or_b32, Format::SOP2, 2, 2)};
            sop2->getOperand(0) = Operand(offset);
            sop2->getOperand(1) = width;
            Temp tmp = {ctx->program->allocateId(), s1};
            sop2->getDefinition(0) = Definition(tmp);
            Temp scc_tmp = {ctx->program->allocateId(), b};
            sop2->getDefinition(1) = Definition(scc_tmp);
            sop2->getDefinition(1).setFixed(PhysReg{253});
            ctx->block->instructions.emplace_back(std::move(sop2));
            extract = Operand(tmp);
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

         aco_ptr<Instruction> sop2{create_instruction<SOP2_instruction>(opcode, Format::SOP2, 2, 1)};
         sop2->getOperand(0) = Operand(base);
         sop2->getOperand(1) = extract;
         sop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(sop2));

      } else {
         /* secure that the instruction has at most 1 sgpr operand
          * The optimizer will inline constants for us */
         if (base.type() == sgpr && offset.type() == sgpr)
            base = as_vgpr(ctx, base);
         if (base.type() == sgpr && bits.type() == sgpr)
            base = as_vgpr(ctx, base);
         if (offset.type() == sgpr && bits.type() == sgpr)
            offset = as_vgpr(ctx, offset);

         aco_opcode opcode;
         if (dst.regClass() == v1) {
            if (instr->op == nir_op_ubfe)
               opcode = aco_opcode::v_bfe_u32;
            else
               opcode = aco_opcode::v_bfe_i32;
         } else {
            unreachable("Unsupported BFE bit size");
         }

         aco_ptr<Instruction> bfe{create_instruction<VOP3A_instruction>(opcode, Format::VOP3A, 3, 1)};
         bfe->getOperand(0) = Operand(base);
         bfe->getOperand(1) = Operand(offset);
         bfe->getOperand(2) = Operand(bits);
         bfe->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(bfe));
      }
      break;
   }
   case nir_op_feq32: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_eq_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fne32: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lg_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_flt32: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lt_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fge32: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_ge_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ieq32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_eq_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_eq_i32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_eq_i32, dst);
         }
      }
      break;
   }
   case nir_op_ine32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lg_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lg_i32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_lg_i32, dst);
         }
      }
      break;
   }
   case nir_op_ilt32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lt_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lt_i32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_lt_i32, dst);
         }
      }
      break;
   }
   case nir_op_ige32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_ge_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_ge_i32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_ge_i32, dst);
         }
      }
      break;
   }
   case nir_op_ult32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lt_u32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lt_u32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_lt_u32, dst);
         }
      }
      break;
   }
   case nir_op_uge32: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_ge_u32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_ge_u32, dst_tmp);
            emit_extract_vector(ctx, dst_tmp, 0, dst);
         } else {
            emit_sopc_instruction_output32(ctx, instr, aco_opcode::s_cmp_ge_u32, dst);
         }
      }
      break;
   }
   case nir_op_fddx:
   case nir_op_fddy: {
      Temp tl = {ctx->program->allocateId(), v1};
      emit_quad_swizzle(ctx, get_alu_src(ctx, instr->src[0]), tl, 0, 0, 0, 0);
      Format format = (Format) ((uint32_t) Format::VOP2 | (uint32_t) Format::DPP);
      aco_ptr<DPP_instruction> sub{create_instruction<DPP_instruction>(aco_opcode::v_sub_f32, format, 2, 1)};
      sub->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
      sub->getOperand(1) = Operand(tl);
      sub->getDefinition(0) = Definition(dst);
      sub->dpp_ctrl = instr->op == nir_op_fddx ? 0x55 : 0xAA;
      sub->row_mask = 0xF;
      sub->bank_mask = 0xF;
      ctx->block->instructions.emplace_back(std::move(sub));
      break;
   }
   case nir_op_idiv:
   case nir_op_udiv: {
      if (dst.regClass() == v1) {
         emit_udiv(ctx, get_alu_src(ctx, instr->src[0]), get_alu_src(ctx, instr->src[1]), dst);
      } else {
         Temp tmp = {ctx->program->allocateId(), v1};
         emit_udiv(ctx, get_alu_src(ctx, instr->src[0]), get_alu_src(ctx, instr->src[1]), tmp);
         aco_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
         readlane->getOperand(0) = Operand(tmp);
         readlane->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(readlane));
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
   // TODO: we really want to have the resulting type as this would allow for 64bit literals
   // which get truncated the lsb if double and msb if int
   // for now, we only use s_mov_b64 with 64bit inline constants
   assert(instr->def.num_components == 1 && "Vector load_const should be lowered to scalar.");
   Temp dst = get_ssa_temp(ctx, &instr->def);
   assert(dst.type() == sgpr);

   if (dst.size() == 1)
   {
      aco_ptr<Instruction> mov;
      aco_opcode op = dst.size() == 2 ? aco_opcode::s_mov_b64 : aco_opcode::s_mov_b32;
      mov.reset(create_instruction<Instruction>(op, Format::SOP1, 1, 1));
      mov->getDefinition(0) = Definition(dst);
      mov->getOperand(0) = Operand(instr->value.u32[0]);
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      assert(dst.size() != 1);
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1)};
      for (unsigned i = 0; i < dst.size(); i++)
         vec->getOperand(i) = Operand{instr->value.u32[i]};
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
         values[i] = Operand();
      }
   }

   unsigned index = nir_intrinsic_base(instr) / 4;
   index = index - FRAG_RESULT_DATA0;
   unsigned target = V_008DFC_SQ_EXP_MRT + index;
   unsigned col_format = (ctx->options->key.fs.col_format >> (4 * index)) & 0xf;
   //bool is_int8 = (ctx->options->key.fs.is_int8 >> index) & 1;
   //bool is_int10 = (ctx->options->key.fs.is_int10 >> index) & 1;
   unsigned enabled_channels = 0xF;
   aco_opcode compr_op = (aco_opcode)0;

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
      enabled_channels = 0;//0x5;
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

   default:
   case V_028714_SPI_SHADER_32_ABGR:
      break;
   }

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
            values[i] = Operand();
         }
      }
      values[2] = Operand();
      values[3] = Operand();
   }

   aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
   exp->valid_mask = false; // TODO
   exp->done = false; // TODO
   exp->compressed = (bool) compr_op;
   exp->dest = target;
   exp->enabled_mask = enabled_channels;
   for (int i = 0; i < 4; i++)
      exp->getOperand(i) = values[i];

   ctx->block->instructions.emplace_back(std::move(exp));
}

void emit_interp_instr(isel_context *ctx, unsigned idx, unsigned component, Temp src, Temp dst)
{
   Temp coord1 = emit_extract_vector(ctx, src, 0, v1);
   Temp coord2 = emit_extract_vector(ctx, src, 1, v1);

   Temp tmp{ctx->program->allocateId(), v1};
   aco_ptr<Interp_instruction> p1{create_instruction<Interp_instruction>(aco_opcode::v_interp_p1_f32, Format::VINTRP, 2, 1)};
   p1->getOperand(0) = Operand(coord1);
   p1->getOperand(1) = Operand(ctx->prim_mask);
   p1->getOperand(1).setFixed(m0);
   p1->getDefinition(0) = Definition(tmp);
   p1->attribute = idx;
   p1->component = component;
   aco_ptr<Interp_instruction> p2{create_instruction<Interp_instruction>(aco_opcode::v_interp_p2_f32, Format::VINTRP, 3, 1)};
   p2->getOperand(0) = Operand(coord2);
   p2->getOperand(1) = Operand(ctx->prim_mask);
   p2->getOperand(1).setFixed(m0);
   p2->getOperand(2) = Operand(tmp);
   p2->getDefinition(0) = Definition(dst);
   p2->attribute = idx;
   p2->component = component;

   ctx->block->instructions.emplace_back(std::move(p1));
   ctx->block->instructions.emplace_back(std::move(p2));
}

void visit_load_interpolated_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   if (nir_intrinsic_base(instr) == VARYING_SLOT_POS) {
      aco_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.ssa.num_components, 1));
      for (unsigned i = 0; i < instr->dest.ssa.num_components; i++)
         vec->getOperand(i) = Operand(ctx->fs_inputs[fs_input::frag_pos_0 + i]);
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
      return;
   }

   uint64_t base = nir_intrinsic_base(instr) / 4;
   unsigned idx = util_bitcount64(ctx->input_mask & ((1ull << base) - 1ull));
   unsigned component = nir_intrinsic_component(instr);

   if (instr->dest.ssa.num_components == 1) {
      emit_interp_instr(ctx, idx, component, get_ssa_temp(ctx, instr->src[0].ssa), get_ssa_temp(ctx, &instr->dest.ssa));
   } else {
      aco_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.ssa.num_components, 1));
      for (unsigned i = 0; i < instr->dest.ssa.num_components; i++)
      {
         Temp tmp = {ctx->program->allocateId(), v1};
         emit_interp_instr(ctx, idx, component+i, get_ssa_temp(ctx, instr->src[0].ssa), tmp);
         vec->getOperand(i) = Operand(tmp);
      }
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
   }
}

void visit_load_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   if (ctx->stage == MESA_SHADER_VERTEX) {

      Temp vertex_buffers = ctx->vertex_buffers;
      if (vertex_buffers.size() == 1) {
         vertex_buffers = convert_pointer_to_64_bit(ctx, vertex_buffers);
         ctx->vertex_buffers = vertex_buffers;
      }

      unsigned offset = (nir_intrinsic_base(instr) / 4 - VERT_ATTRIB_GENERIC0) * 16;
      aco_ptr<Instruction> load;
      load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
      load->getOperand(0) = Operand(vertex_buffers);
      load->getOperand(1) = Operand((uint32_t) offset);
      Temp list = {ctx->program->allocateId(), s4};
      load->getDefinition(0) = Definition(list);
      ctx->block->instructions.emplace_back(std::move(load));

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
      unsigned idx = util_bitcount64(ctx->input_mask & ((1ull << base) - 1ull));
      unsigned component = nir_intrinsic_component(instr);

      aco_ptr<Interp_instruction> mov{create_instruction<Interp_instruction>(aco_opcode::v_interp_mov_f32, Format::VINTRP, 2, 1)};
      mov->getOperand(0) = Operand();
      mov->getOperand(0).setFixed(PhysReg{2});
      mov->getOperand(1) = Operand(ctx->prim_mask);
      mov->getOperand(1).setFixed(m0);
      mov->getDefinition(0) = Definition(dst);
      mov->attribute = idx;
      mov->component = component;
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      unreachable("Shader stage not implemented");
   }
}

void visit_load_resource(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp index = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned desc_set = nir_intrinsic_desc_set(instr);
   unsigned binding = nir_intrinsic_binding(instr);

   Temp desc_ptr = ctx->descriptor_sets[desc_set];
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
   } else
      stride = layout->binding[binding].size;

   nir_const_value* nir_const_index = nir_src_as_const_value(instr->src[0]);
   unsigned const_index = nir_const_index ? nir_const_index->u32[0] : 0;
   if (stride != 1) {
      if (nir_const_index) {
         const_index = const_index * stride;
      } else {
         aco_ptr<Instruction> tmp;
         tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1));
         tmp->getOperand(0) = Operand(stride);
         tmp->getOperand(1) = Operand(index);
         index = {ctx->program->allocateId(), index.regClass()};
         tmp->getDefinition(0) = Definition(index);
         ctx->block->instructions.emplace_back(std::move(tmp));
      }
   }
   if (offset) {
      if (nir_const_index) {
         const_index = const_index + offset;
      } else {
         aco_ptr<Instruction> tmp;
         tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 2));
         tmp->getOperand(0) = Operand(offset);
         tmp->getOperand(1) = Operand(index);
         index = {ctx->program->allocateId(), index.regClass()};
         tmp->getDefinition(0) = Definition(index);
         tmp->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(tmp));
      }
   }

   aco_ptr<Instruction> tmp;
   tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 2));
   tmp->getOperand(0) = nir_const_index ? Operand(const_index) : Operand(index);
   tmp->getOperand(1) = Operand(desc_ptr);
   index = {ctx->program->allocateId(), index.regClass()};
   tmp->getDefinition(0) = Definition(index);
   tmp->getDefinition(1) = Definition(PhysReg{253}, b);
   ctx->block->instructions.emplace_back(std::move(tmp));

   index = convert_pointer_to_64_bit(ctx, index);
   ctx->allocated.insert({instr->dest.ssa.index, index.id()});


}

void visit_load_ubo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp rsrc = get_ssa_temp(ctx, instr->src[0].ssa);
   nir_const_value* const_offset = nir_src_as_const_value(instr->src[1]);

   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(rsrc);
   load->getOperand(1) = Operand((uint32_t) 0);
   rsrc = {ctx->program->allocateId(), s4};
   load->getDefinition(0) = Definition(rsrc);
   ctx->block->instructions.emplace_back(std::move(load));

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
      load.reset(create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1));
      load->getOperand(0) = Operand(rsrc);

      if (const_offset && const_offset->u32[0] < 0xFFFFF)
         load->getOperand(1) = Operand(const_offset->u32[0]);
      else
         load->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      load->getDefinition(0) = Definition(dst);

      if (dst.size() == 3) {
      /* trim vector */
         Temp vec = {ctx->program->allocateId(), s4};
         load->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(load));
         emit_split_vector(ctx, vec, 4);

         aco_ptr<Instruction> trimmed{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1)};
         for (unsigned i = 0; i < 3; i++) {
            Temp tmp = emit_extract_vector(ctx, vec, i, s1);
            trimmed->getOperand(i) = Operand(tmp);
         }
         trimmed->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(trimmed));
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
      ctx->block->instructions.emplace_back(std::move(mubuf));
   }

   emit_split_vector(ctx, dst, instr->dest.ssa.num_components);
}

void visit_load_push_constant(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp index = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned offset = nir_intrinsic_base(instr);
   if (offset != 0) { // TODO check if index != 0 as well
      aco_ptr<SOP2_instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 2)};
      add->getOperand(0) = Operand(offset);
      add->getOperand(1) = Operand(index);
      index = {ctx->program->allocateId(), s1};
      add->getDefinition(0) = Definition(index);
      add->getDefinition(1) = Definition(PhysReg{253}, b);
      ctx->block->instructions.emplace_back(std::move(add));
   }
   Temp ptr = ctx->push_constants;
   if (ptr.size() == 1) {
      ptr = convert_pointer_to_64_bit(ctx, ptr);
   }

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
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

   aco_ptr<SMEM_instruction> load{create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1)};
   load->getOperand(0) = Operand(ptr);
   load->getOperand(1) = Operand(index);
   Temp vec = dst.size() == 3 ? Temp{ctx->program->allocateId(), rc} : dst;
   load->getDefinition(0) = Definition(vec);
   ctx->block->instructions.emplace_back(std::move(load));

   emit_split_vector(ctx, vec, vec.size());

   if (dst.size() == 3) {
      aco_ptr<Instruction> trimmed{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1)};
      for (unsigned i = 0; i < dst.size(); i++) {
         trimmed->getOperand(i) = Operand(emit_extract_vector(ctx, vec, i, s1));
      }
      trimmed->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(trimmed));
   }
}

void visit_discard_if(isel_context *ctx, nir_intrinsic_instr *instr)
{
   /**
    * s_andn2_b64 exec, exec, vcc
    * s_cbranch_execnz Label
    * exp null off, off, off, off done vm
    * s_endpgm
    * Label
    */
   Temp cond32 = get_ssa_temp(ctx, instr->src[0].ssa);
   Temp cond = Temp{ctx->program->allocateId(), s2};

   aco_ptr<Instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_lg_u32, Format::VOPC, 2, 1)};
   cmp->getOperand(0) = Operand((uint32_t) 0);
   cmp->getOperand(1) = Operand{cond32};
   cmp->getDefinition(0) = Definition{cond};
   cmp->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(cmp));

   aco_ptr<Instruction> discard{create_instruction<Instruction>(aco_opcode::p_discard_if, Format::PSEUDO, 1, 1)};
   discard->getOperand(0) = Operand(cond);
   discard->getDefinition(0) = Definition{ctx->program->allocateId(), b};
   discard->getDefinition(0).setFixed(PhysReg{253});
   ctx->block->instructions.emplace_back(std::move(discard));
   return;
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
            constant_index += array_size * const_value->u32[0];
         } else {
            Temp indirect = get_ssa_temp(ctx, deref_instr->arr.index.ssa);
            /* check if index is in sgpr */
            if (indirect.type() == vgpr) {
               aco_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
               readlane->getOperand(0) = Operand(indirect);
               indirect = {ctx->program->allocateId(), s1};
               readlane->getDefinition(0) = Definition(indirect);
               ctx->block->instructions.emplace_back(std::move(readlane));
            }

            if (array_size != 1) {
               aco_ptr<Instruction> mul{create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1)};
               indirect = {ctx->program->allocateId(), s1};
               mul->getDefinition(0) = Definition(indirect);
               mul->getOperand(0) = Operand(array_size);
               mul->getOperand(1) = Operand(indirect);
               ctx->block->instructions.emplace_back(std::move(mul));
            }

            if (!index_set) {
               index = indirect;
               index_set = true;
            } else {
               aco_ptr<Instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 2)};
               add->getDefinition(0) = Definition{ctx->program->allocateId(), s1};
               add->getDefinition(1) = Definition(PhysReg{253}, b);
               add->getOperand(0) = Operand(index);
               add->getOperand(1) = Operand(indirect);
               ctx->block->instructions.emplace_back(std::move(add));
               index = add->getDefinition(0).getTemp();
            }
         }

         deref_instr = nir_src_as_deref(deref_instr->parent);
      }
      descriptor_set = deref_instr->var->data.descriptor_set;
      base_index = deref_instr->var->data.binding;
   }

   Temp list = ctx->descriptor_sets[descriptor_set];
   if (list.size() == 1) {
      list = convert_pointer_to_64_bit(ctx, list);
      //ctx->descriptor_sets[descriptor_set] = list;
   }

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
         offset += 64;
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
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 4, 1)};
      vec->getOperand(0) = Operand(samplers[constant_index * 4 + 0]);
      vec->getOperand(1) = Operand(samplers[constant_index * 4 + 1]);
      vec->getOperand(2) = Operand(samplers[constant_index * 4 + 2]);
      vec->getOperand(3) = Operand(samplers[constant_index * 4 + 3]);
      Temp res = {ctx->program->allocateId(), s4};
      vec->getDefinition(0) = Definition(res);
      ctx->block->instructions.emplace_back(std::move(vec));
      return res;
   }

   Operand off;
   if (!index_set) {
      off = Operand(offset);
   } else {
      aco_ptr<SOP2_instruction> mul{create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1)};
      mul->getOperand(0) = Operand(stride);
      mul->getOperand(1) = Operand(index);
      Temp t = {ctx->program->allocateId(), s1};
      mul->getDefinition(0) = Definition(t);
      ctx->block->instructions.emplace_back(std::move(mul));
      aco_ptr<SOP2_instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 2)};
      add->getOperand(0) = Operand(offset);
      add->getOperand(1) = Operand(t);
      t = {ctx->program->allocateId(), s1};
      add->getDefinition(0) = Definition(t);
      add->getDefinition(1) = Definition(PhysReg{253}, b);
      ctx->block->instructions.emplace_back(std::move(add));
      off = Operand(t);
   }

   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(opcode, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(list);
   load->getOperand(1) = off;
   Temp t = {ctx->program->allocateId(), type};
   load->getDefinition(0) = Definition(t);
   ctx->block->instructions.emplace_back(std::move(load));
   return t;
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
static Temp adjust_sample_index_using_fmask(isel_context *ctx, Temp coords, Temp sample_index, Temp fmask_desc_ptr)
{
   Temp fmask = {ctx->program->allocateId(), v1};

   aco_ptr<MIMG_instruction> load{create_instruction<MIMG_instruction>(aco_opcode::image_load, Format::MIMG, 2, 1)};
   load->getOperand(0) = Operand(coords);
   load->getOperand(1) = Operand(fmask_desc_ptr);
   load->getDefinition(0) = Definition(fmask);
   load->glc = false;
   load->dmask = 0x1;
   load->unrm = true;
   ctx->block->instructions.emplace_back(std::move(load));

   Temp sample_index4 = {ctx->program->allocateId(), sample_index.regClass()};
   if (sample_index.regClass() == s1) {
      aco_ptr<Instruction> instr(create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 2));
      instr->getOperand(0) = Operand(sample_index);
      instr->getOperand(1) = Operand((uint32_t) 2);
      instr->getDefinition(0) = Definition(sample_index4);
      Temp t = {ctx->program->allocateId(), b};
      instr->getDefinition(1) = Definition(t);
      instr->getDefinition(1).setFixed(PhysReg{253}); /* scc */
      ctx->block->instructions.emplace_back(std::move(instr));
   } else {
      assert(sample_index.regClass() == v1);

      aco_ptr<Instruction> instr(create_instruction<VOP2_instruction>(aco_opcode::v_lshlrev_b32, Format::VOP2, 2, 1));
      instr->getOperand(0) = Operand((uint32_t) 2);
      instr->getOperand(1) = Operand(sample_index);
      instr->getDefinition(0) = Definition(sample_index4);
      ctx->block->instructions.emplace_back(std::move(instr));
   }

   aco_ptr<Instruction> instr(create_instruction<VOP3A_instruction>(aco_opcode::v_bfe_u32, Format::VOP3A, 3, 1));
   instr->getOperand(0) = Operand(fmask);
   instr->getOperand(1) = Operand(sample_index4);
   instr->getOperand(2) = Operand((uint32_t) 4);
   Temp final_sample = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(final_sample);
   ctx->block->instructions.emplace_back(std::move(instr));

   /* Don't rewrite the sample index if WORD1.DATA_FORMAT of the FMASK
    * resource descriptor is 0 (invalid),
    */
   Temp fmask_word1 = emit_extract_vector(ctx, fmask_desc_ptr, 1, s1);

   Format format = (Format) ((uint32_t) Format::VOPC | (uint32_t) Format::VOP3A);
   instr.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cmp_lg_u32, format, 2, 1));
   instr->getOperand(0) = Operand((uint32_t) 0);
   instr->getOperand(1) = Operand(fmask_word1);
   Temp compare = {ctx->program->allocateId(), s2};
   instr->getDefinition(0) = Definition(compare);
   instr->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(instr));

   Temp sample_index_v = {ctx->program->allocateId(), v1};
   emit_v_mov(ctx, sample_index, sample_index_v);

   /* Replace the MSAA sample index. */
   instr.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
   instr->getOperand(0) = Operand(sample_index_v); // else
   instr->getOperand(1) = Operand(final_sample);
   instr->getOperand(2) = Operand(compare);
   sample_index = {ctx->program->allocateId(), v1};
   instr->getDefinition(0) = Definition(sample_index);
   ctx->block->instructions.emplace_back(std::move(instr));

   return sample_index;
}

static Temp get_image_coords(isel_context *ctx, const nir_intrinsic_instr *instr, const struct glsl_type *type)
{

   Temp src0 = get_ssa_temp(ctx, instr->src[1].ssa);
   enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   bool is_array = glsl_sampler_type_is_array(type);
   bool add_frag_pos = (dim == GLSL_SAMPLER_DIM_SUBPASS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   bool is_ms = (dim == GLSL_SAMPLER_DIM_MS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   bool gfx9_1d = ctx->options->chip_class >= GFX9 && dim == GLSL_SAMPLER_DIM_1D;
   int count = image_type_to_components_count(dim, is_array);

   if (count == 1 && !gfx9_1d)
      return emit_extract_vector(ctx, src0, 0, v1);

   std::vector<Operand> coords;

   if (add_frag_pos) {
      unreachable("add_frag_pos not implemented.");
   }

   if (gfx9_1d) {
      coords.emplace_back(Operand(emit_extract_vector(ctx, src0, 0, v1)));
      coords.emplace_back(Operand((uint32_t) 0));
      if (is_array)
         coords.emplace_back(Operand(emit_extract_vector(ctx, src0, 1, v1)));
   } else {
      for (int i = 0; i < count; i++)
         coords.emplace_back(Operand(emit_extract_vector(ctx, src0, i, v1)));
   }

   if (is_ms) {
      unreachable("is_ms not implemented.");
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
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      unreachable("image load with GLSL_SAMPLER_DIM_BUF not yet implemented\n");
      return;
   }

   Temp coords = get_image_coords(ctx, instr, type);
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);
   //aco_image_dim img_dim = get_image_dim(ctx, glsl_get_sampler_dim(type), glsl_sampler_type_is_array(type));
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_ptr<MIMG_instruction> load{create_instruction<MIMG_instruction>(aco_opcode::image_load, Format::MIMG, 2, 1)};
   load->getOperand(0) = Operand(coords);
   load->getOperand(1) = Operand(resource);
   load->getDefinition(0) = Definition(dst);
   load->glc = var->data.image.access & (ACCESS_VOLATILE | ACCESS_COHERENT) ? 1 : 0;
   load->dmask = 0xF;
   load->unrm = true;
   ctx->block->instructions.emplace_back(std::move(load));
   emit_split_vector(ctx, dst, 4);
   return;
}

void visit_image_store(isel_context *ctx, nir_intrinsic_instr *instr)
{
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
   Temp data = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[3].ssa));

   bool glc = ctx->options->chip_class == SI || var->data.image.access & (ACCESS_VOLATILE | ACCESS_COHERENT) ? 1 : 0;

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      Temp rsrc = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_BUFFER, nullptr, true, true);
      Temp vindex = emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[1].ssa), 0, v1);

      aco_ptr<MUBUF_instruction> store{create_instruction<MUBUF_instruction>(aco_opcode::buffer_store_format_xyzw, Format::MUBUF, 4, 0)};
      store->getOperand(0) = Operand(vindex);
      store->getOperand(1) = Operand(rsrc);
      store->getOperand(2) = Operand((uint32_t) 0);
      store->getOperand(3) = Operand(data);
      store->idxen = true;
      store->glc = glc;
      ctx->block->instructions.emplace_back(std::move(store));
      return;
   }

   assert(data.type() == vgpr);
   Temp coords = get_image_coords(ctx, instr, type);
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);

   aco_ptr<MIMG_instruction> store{create_instruction<MIMG_instruction>(aco_opcode::image_store, Format::MIMG, 3, 0)};
   store->getOperand(0) = Operand(coords);
   store->getOperand(1) = Operand(resource);
   store->getOperand(2) = Operand(data);
   store->glc = glc;
   store->dmask = 0xF;
   store->unrm = true;
   ctx->block->instructions.emplace_back(std::move(store));
   return;
}

void visit_image_atomic(isel_context *ctx, nir_intrinsic_instr *instr) {

   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   bool is_unsigned = glsl_get_sampler_result_type(type) == GLSL_TYPE_UINT;

   Temp data = get_ssa_temp(ctx, instr->src[3].ssa);
   assert(data.size() == 1 && "64bit ssbo atomics not yet implemented.");
   if (instr->intrinsic == nir_intrinsic_image_deref_atomic_comp_swap) {
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(data);
      vec->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[4].ssa));
      data = {ctx->program->allocateId(), v2};
      vec->getDefinition(0) = Definition(data);
      ctx->block->instructions.emplace_back(std::move(vec));
   } else {
      data = as_vgpr(ctx, data);
   }

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

   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);

   if (glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF) {
      Temp vindex = emit_extract_vector(ctx, get_ssa_temp(ctx, instr->src[1].ssa), 0, v1);
      Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, nullptr, true, true);
      assert(ctx->options->chip_class < GFX9 && "GFX9 stride size workaround not yet implemented.");
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
      ctx->block->instructions.emplace_back(std::move(mubuf));
      return;
   }

   Temp coords = get_image_coords(ctx, instr, type);
   aco_ptr<MIMG_instruction> mimg{create_instruction<MIMG_instruction>(image_op, Format::MIMG, 3, return_previous ? 1 : 0)};
   mimg->getOperand(0) = Operand(coords);
   mimg->getOperand(1) = Operand(resource);
   mimg->getOperand(2) = Operand(data);
   if (return_previous)
      mimg->getDefinition(0) = Definition(dst);
   mimg->glc = return_previous;
   mimg->dmask = (1 << data.size()) - 1;
   mimg->unrm = true;
   ctx->block->instructions.emplace_back(std::move(mimg));
   return;
}

void visit_image_size(isel_context *ctx, nir_intrinsic_instr *instr)
{
   const nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
   const struct glsl_type *type = glsl_without_array(var->type);
   if (glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF)
      assert(false && "image_deref_size: get buffer size");
      /*return get_buffer_size(ctx, get_image_descriptor(ctx, instr, AC_DESC_BUFFER, false), true);*/

   /* LOD */
   aco_ptr<VOP1_instruction> mov{create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1)};
   mov->getOperand(0) = Operand((uint32_t) 0);
   Temp lod = Temp{ctx->program->allocateId(), v1};
   mov->getDefinition(0) = Definition(lod);
   ctx->block->instructions.emplace_back(std::move(mov));

   /* Resource */
   Temp resource = get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr), ACO_DESC_IMAGE, NULL, true, false);

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_ptr<MIMG_instruction> mimg{create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1)};
   mimg->getOperand(0) = Operand(lod);
   mimg->getOperand(1) = Operand(resource);
   mimg->dmask = (1 << instr->dest.ssa.num_components) - 1;
   Definition& def = mimg->getDefinition(0);
   ctx->block->instructions.emplace_back(std::move(mimg));

   if (glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_CUBE &&
       glsl_sampler_type_is_array(type)) {
      Temp tmp = {ctx->program->allocateId(), v4};
      def = Definition(tmp);
      /* TODO: split vector and divide 3nd value by 6 */
      assert(false && "Unimplemented image_deref_size");

   } else if (ctx->options->chip_class >= GFX9 &&
              glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_1D &&
              glsl_sampler_type_is_array(type)) {
      Temp tmp = {ctx->program->allocateId(), v4};
      def = Definition(tmp);
      /* TODO: split vector, extract 3nd value and insert as 2nd */
      assert(false && "Unimplemented image_deref_size");

   } else {
      def = Definition(dst);
   }

   emit_split_vector(ctx, dst, instr->dest.ssa.num_components);
}

void visit_load_ssbo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned num_components = instr->num_components;
   unsigned num_bytes = num_components * instr->dest.ssa.bit_size / 8;

   Temp rsrc = {ctx->program->allocateId(), s4};
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);

   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(get_ssa_temp(ctx, instr->src[0].ssa));
   load->getOperand(1) = Operand((uint32_t) 0);
   load->getDefinition(0) = Definition(rsrc);
   ctx->block->instructions.emplace_back(std::move(load));

   aco_opcode op;
   if (dst.type() == vgpr) {
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
      mubuf->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand();
      mubuf->getOperand(1) = Operand(rsrc);
      mubuf->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
      mubuf->getDefinition(0) = Definition(dst);
      mubuf->offen = (offset.type() == vgpr);
      mubuf->glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT);
      ctx->block->instructions.emplace_back(std::move(mubuf));
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
      load.reset(create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1));
      load->getOperand(0) = Operand(rsrc);
      load->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      assert(load->getOperand(1).getTemp().type() == sgpr);
      load->getDefinition(0) = Definition(dst);

      if (dst.size() == 3) {
      /* trim vector */
         Temp vec = {ctx->program->allocateId(), s4};
         load->getDefinition(0) = Definition(vec);
         ctx->block->instructions.emplace_back(std::move(load));
         emit_split_vector(ctx, vec, 4);

         aco_ptr<Instruction> trimmed{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1)};
         for (unsigned i = 0; i < 3; i++) {
            Temp tmp = emit_extract_vector(ctx, vec, i, s1);
            trimmed->getOperand(i) = Operand(tmp);
         }
         trimmed->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(trimmed));
      } else {
         ctx->block->instructions.emplace_back(std::move(load));
      }

   }
   emit_split_vector(ctx, dst, num_components);
}

void visit_store_ssbo(isel_context *ctx, nir_intrinsic_instr *instr)
{

   Temp data = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned elem_size_bytes = instr->src[0].ssa->bit_size / 8;
   unsigned writemask = nir_intrinsic_write_mask(instr);

   Temp offset;
   if (ctx->options->chip_class < VI)
      offset = as_vgpr(ctx,get_ssa_temp(ctx, instr->src[2].ssa));
   else
      offset = get_ssa_temp(ctx, instr->src[2].ssa);

   Temp rsrc = {ctx->program->allocateId(), s4};

   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
   load->getOperand(1) = Operand((uint32_t) 0);
   load->getDefinition(0) = Definition(rsrc);
   ctx->block->instructions.emplace_back(std::move(load));

   while (writemask) {
      int start, count;
      u_bit_scan_consecutive_range(&writemask, &start, &count);
      int num_bytes = count * elem_size_bytes;

      // TODO: we can only store 4 DWords at the same time. Fix for 64bit vectors
      // TODO: check alignment of sub-dword stores
      // TODO: split 3 bytes. there is no store instruction for that

      Temp write_data;
      if (count != instr->num_components) {
         aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, count, 1)};
         for (int i = 0; i < count; i++)
            vec->getOperand(i) = Operand(emit_extract_vector(ctx, data, start + i, v1));
         write_data = {ctx->program->allocateId(), getRegClass(vgpr, count)};
         vec->getDefinition(0) = Definition(write_data);
         ctx->block->instructions.emplace_back(std::move(vec));
      } else if (data.type() != vgpr) {
         assert(num_bytes % 4 == 0);
         write_data = {ctx->program->allocateId(), getRegClass(vgpr, num_bytes / 4)};
         emit_v_mov(ctx, data, write_data);
      } else {
         write_data = data;
      }

      aco_opcode op;
      switch (num_bytes) {
         case 4:
            op = aco_opcode::buffer_store_dword;
            break;
         case 8:
            op = aco_opcode::buffer_store_dwordx2;
            break;
         case 12:
            op = aco_opcode::buffer_store_dwordx3;
            break;
         case 16:
            op = aco_opcode::buffer_store_dwordx4;
            break;
         default:
            unreachable("Store SSBO not implemented for this size.");
      }

      aco_ptr<MUBUF_instruction> store{create_instruction<MUBUF_instruction>(op, Format::MUBUF, 4, 0)};
      store->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand();
      store->getOperand(1) = Operand(rsrc);
      store->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
      store->getOperand(3) = Operand(write_data);
      store->offset = start;
      store->offen = (offset.type() == vgpr);
      store->glc = nir_intrinsic_access(instr) & (ACCESS_VOLATILE | ACCESS_COHERENT);
      ctx->block->instructions.emplace_back(std::move(store));
   }
}

void visit_atomic_ssbo(isel_context *ctx, nir_intrinsic_instr *instr) {

   Temp data = get_ssa_temp(ctx, instr->src[2].ssa);
   assert(data.size() == 1 && "64bit ssbo atomics not yet implemented.");
   if (instr->intrinsic == nir_intrinsic_ssbo_atomic_comp_swap) {
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(data);
      vec->getOperand(1) = Operand(get_ssa_temp(ctx, instr->src[3].ssa));
      data = {ctx->program->allocateId(), v2};
      vec->getDefinition(0) = Definition(data);
      ctx->block->instructions.emplace_back(std::move(vec));
   } else {
      data = as_vgpr(ctx, data);
   }

   Temp offset;
   if (ctx->options->chip_class < VI)
      offset = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   else
      offset = get_ssa_temp(ctx, instr->src[1].ssa);

   Temp rsrc = {ctx->program->allocateId(), s4};
   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(get_ssa_temp(ctx, instr->src[0].ssa));
   load->getOperand(1) = Operand((uint32_t) 0);
   load->getDefinition(0) = Definition(rsrc);
   ctx->block->instructions.emplace_back(std::move(load));

   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
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
   mubuf->getOperand(0) = offset.type() == vgpr ? Operand(offset) : Operand();
   mubuf->getOperand(1) = Operand(rsrc);
   mubuf->getOperand(2) = offset.type() == sgpr ? Operand(offset) : Operand((uint32_t) 0);
   mubuf->getOperand(3) = Operand(data);
   if (return_previous)
      mubuf->getDefinition(0) = Definition(dst);
   mubuf->offset = 0;
   mubuf->offen = (offset.type() == vgpr);
   mubuf->glc = return_previous;
   ctx->block->instructions.emplace_back(std::move(mubuf));
}

void visit_get_buffer_size(isel_context *ctx, nir_intrinsic_instr *instr) {

   Temp index = get_ssa_temp(ctx, instr->src[0].ssa);
   aco_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(index);
   load->getOperand(1) = Operand((uint32_t) 0);
   Temp desc = {ctx->program->allocateId(), s4};
   load->getDefinition(0) = Definition(desc);
   ctx->block->instructions.emplace_back(std::move(load));

   emit_extract_vector(ctx, desc, 2, get_ssa_temp(ctx, &instr->dest.ssa));
}

void emit_memory_barrier(isel_context *ctx, nir_intrinsic_instr *instr) {
   aco_ptr<Instruction> barrier;
   aco_opcode op;
   switch(instr->intrinsic) {
      case nir_intrinsic_group_memory_barrier:
      case nir_intrinsic_memory_barrier:
         op = aco_opcode::p_memory_barrier_all;
         break;
      case nir_intrinsic_memory_barrier_atomic_counter:
         op = aco_opcode::p_memory_barrier_atomic;
         break;
      case nir_intrinsic_memory_barrier_buffer:
         op = aco_opcode::p_memory_barrier_buffer;
         break;
      case nir_intrinsic_memory_barrier_image:
         op = aco_opcode::p_memory_barrier_image;
         break;
      case nir_intrinsic_memory_barrier_shared:
         op = aco_opcode::p_memory_barrier_shared;
         break;
      default:
         unreachable("Unimplemented memory barrier intrinsic");
         break;
   }
   barrier.reset(create_instruction<Pseudo_barrier_instruction>(op, Format::PSEUDO_BARRIER, 0, 0));
   ctx->block->instructions.emplace_back(std::move(barrier));
}

Temp load_lds_size_m0(isel_context *ctx)
{
   aco_ptr<Instruction> instr{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1)};
   instr->getOperand(0) = Operand(0xFFFFFFFF);
   Temp dst = {ctx->program->allocateId(), s1};
   instr->getDefinition(0) = Definition(dst);
   instr->getDefinition(0).setFixed(m0);
   ctx->block->instructions.emplace_back(std::move(instr));
   return dst;
}


void visit_load_shared(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp m = load_lds_size_m0(ctx);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   assert(instr->dest.ssa.bit_size == 32 && "Bitsize not supported in load_shared.");
   Temp address = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[0].ssa));

   unsigned offset = instr->const_index[0];
   aco_opcode op;
   switch (instr->num_components) {
      case 1:
         op = aco_opcode::ds_read_b32;
         break;
      case 2:
         op = aco_opcode::ds_read_b64;
         break;
      case 3:
         op = aco_opcode::ds_read_b96;
         break;
      case 4:
         op = aco_opcode::ds_read_b128;
         break;
      default:
         unreachable("unreachable");
   }

   aco_ptr<DS_instruction> ds{create_instruction<DS_instruction>(op, Format::DS, 2, 1)};
   ds->getOperand(0) = Operand(address);
   ds->getOperand(1) = Operand(m);
   ds->getOperand(1).setFixed(m0);
   ds->getDefinition(0) = Definition(dst);
   ds->offset0 = offset;
   ctx->block->instructions.emplace_back(std::move(ds));
   return;
}

void visit_store_shared(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned offset = instr->const_index[0];
   unsigned writemask = instr->const_index[1];
   Temp m = load_lds_size_m0(ctx);
   Temp data = get_ssa_temp(ctx, instr->src[0].ssa);
   Temp address = as_vgpr(ctx, get_ssa_temp(ctx, instr->src[1].ssa));
   unsigned elem_size_bytes = instr->src[0].ssa->bit_size / 8;
   assert(elem_size_bytes == 4 && "Only 32bit store_shared currently supported.");

   /* we need at most two stores for 32bit variables */
   int start[2], count[2];
   u_bit_scan_consecutive_range(&writemask, &start[0], &count[0]);
   u_bit_scan_consecutive_range(&writemask, &start[1], &count[1]);
   assert(writemask == 0);

   aco_ptr<DS_instruction> ds;
   /* one combined store is sufficient */
   if (count[0] == count[1]) {
      assert(count[0] == 1);
      ds.reset(create_instruction<DS_instruction>(aco_opcode::ds_write2_b32, Format::DS, 4, 0));
      ds->getOperand(0) = Operand(address);
      ds->getOperand(1) = Operand(emit_extract_vector(ctx, data, start[0], v1));
      ds->getOperand(2) = Operand(emit_extract_vector(ctx, data, start[1], v1));
      ds->getOperand(3) = Operand(m);
      ds->getOperand(3).setFixed(m0);
      ds->offset0 = (offset >> 2) + start[0];
      ds->offset1 = (offset >> 2) + start[1];
      ctx->block->instructions.emplace_back(std::move(ds));
      return;
   }

   for (unsigned i = 0; i < 2; i++) {
      if (count[i] == 0)
         continue;

      aco_opcode op;
      if (count[i] == 1)
         op = aco_opcode::ds_write_b32;
      else if (count[i] == 2)
         op = aco_opcode::ds_write_b64;
      else if (count[i] == 3)
         op = aco_opcode::ds_write_b96;
      else if (count[i] == 4)
         op = aco_opcode::ds_write_b128;
      else
         unreachable("Unhandled LDS write size");

      Temp write_data = emit_extract_vector(ctx, data, start[i], getRegClass(vgpr, count[i]));
      ds.reset(create_instruction<DS_instruction>(op, Format::DS, 3, 0));
      ds->getOperand(0) = Operand(address);
      ds->getOperand(1) = Operand(write_data);
      ds->getOperand(2) = Operand(m);
      ds->getOperand(2).setFixed(m0);
      ds->offset0 = offset + (start[i] * elem_size_bytes);
      ctx->block->instructions.emplace_back(std::move(ds));
   }
   return;
}

void visit_shared_atomic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned offset = instr->const_index[0];
   Temp m = load_lds_size_m0(ctx);
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

   aco_ptr<DS_instruction> ds;
   ds.reset(create_instruction<DS_instruction>(op, Format::DS, num_operands, return_previous ? 1 : 0));
   ds->getOperand(0) = Operand(address);
   ds->getOperand(1) = Operand(data);
   if (num_operands == 4)
      ds->getOperand(2) = Operand(get_ssa_temp(ctx, instr->src[2].ssa));
   ds->getOperand(num_operands - 1) = Operand(m);
   ds->getOperand(num_operands - 1).setFixed(m0);
   ds->offset0 = offset;
   if (return_previous)
      ds->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(ds));
}

void visit_intrinsic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   switch(instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel: {
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(ctx->fs_inputs[fs_input::persp_center_p1]);
      vec->getOperand(1) = Operand(ctx->fs_inputs[fs_input::persp_center_p2]);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
      emit_split_vector(ctx, dst, 2);
      break;
   }
   case nir_intrinsic_load_front_face: {
      emit_v_mov(ctx, ctx->fs_inputs[fs_input::front_face], get_ssa_temp(ctx, &instr->dest.ssa));
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
      if (workgroup_size > 64) {
         aco_ptr<Instruction> barrier{create_instruction<SOPP_instruction>(aco_opcode::s_barrier, Format::SOPP, 0, 0)};
         ctx->block->instructions.emplace_back(std::move(barrier));
      }
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
      aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1)};
      Temp* ids;
      if (instr->intrinsic == nir_intrinsic_load_num_work_groups)
         ids = ctx->num_workgroups;
      else if (instr->intrinsic == nir_intrinsic_load_work_group_id)
         ids = ctx->workgroup_ids;
      else
         ids = ctx->local_invocation_ids;
      vec->getOperand(0) = Operand(ids[0]);
      vec->getOperand(1) = Operand(ids[1]);
      vec->getOperand(2) = Operand(ids[2]);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
      emit_split_vector(ctx, dst, 3);
      break;
   }
   case nir_intrinsic_load_local_invocation_index: {
      aco_ptr<Instruction> mbcnt{create_instruction<VOP3A_instruction>(aco_opcode::v_mbcnt_lo_u32_b32, Format::VOP3A, 2, 1)};
      mbcnt->getOperand(0) = Operand((uint32_t) -1);
      mbcnt->getOperand(1) = Operand((uint32_t) 0);
      Temp tmp = {ctx->program->allocateId(), v1};
      mbcnt->getDefinition(0) = Definition(tmp);
      ctx->block->instructions.emplace_back(std::move(mbcnt));
      mbcnt.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_mbcnt_hi_u32_b32, Format::VOP3A, 2, 1));
      mbcnt->getOperand(0) = Operand((uint32_t) -1);
      mbcnt->getOperand(1) = Operand(tmp);
      Temp id = {ctx->program->allocateId(), v1};
      mbcnt->getDefinition(0) = Definition(id);
      ctx->block->instructions.emplace_back(std::move(mbcnt));
      mbcnt.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b32, Format::SOP2, 2, 1));
      mbcnt->getOperand(0) = Operand((uint32_t) 0xfc0);
      mbcnt->getOperand(1) = Operand(ctx->tg_size);
      Temp tg_num = {ctx->program->allocateId(), s1};
      mbcnt->getDefinition(0) = Definition(tg_num);
      ctx->block->instructions.emplace_back(std::move(mbcnt));
      mbcnt.reset(create_instruction<VOP2_instruction>(aco_opcode::v_or_b32, Format::VOP2, 2, 1));
      mbcnt->getOperand(0) = Operand(tg_num);
      mbcnt->getOperand(1) = Operand(id);
      Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
      mbcnt->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(mbcnt));
      break;
   }
   case nir_intrinsic_load_sample_id: {
      aco_ptr<Instruction> bfe{create_instruction<VOP3A_instruction>(aco_opcode::v_bfe_u32, Format::VOP3A, 3, 1)};
      bfe->getOperand(0) = Operand(ctx->fs_inputs[ancillary]);
      bfe->getOperand(1) = Operand((uint32_t) 8);
      bfe->getOperand(2) = Operand((uint32_t) 4);
      bfe->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(bfe));
      break;
   }
   default:
      fprintf(stderr, "Unimplemented intrinsic instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();

      break;
   }
}


void tex_fetch_ptrs(isel_context *ctx, nir_tex_instr *instr,
                           Temp *res_ptr, Temp *samp_ptr, Temp *fmask_ptr)
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

void prepare_cube_coords(isel_context *ctx, Temp* coords, bool is_deriv, bool is_array, bool is_lod)
{

   if (is_array && !is_lod)
      fprintf(stderr, "Unimplemented tex instr type: cube coords1");

   Temp coord_args[3], ma, tc, sc, id;
   aco_ptr<Instruction> tmp;
   emit_split_vector(ctx, *coords, 3);
   for (unsigned i = 0; i < 3; i++)
      coord_args[i] = emit_extract_vector(ctx, *coords, i, v1);

   tmp.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cubema_f32, Format::VOP3A, 3, 1));
   for (unsigned i = 0; i < 3; i++)
      tmp->getOperand(i) = Operand(coord_args[i]);
   ma = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(ma);
   ctx->block->instructions.emplace_back(std::move(tmp));
   aco_ptr<VOP3A_instruction> vop3a{create_instruction<VOP3A_instruction>(aco_opcode::v_rcp_f32, (Format) ((uint16_t) Format::VOP3A | (uint16_t) Format::VOP1), 1, 1)};
   vop3a->getOperand(0) = Operand(ma);
   vop3a->abs[0] = true;
   ma = {ctx->program->allocateId(), v1};
   vop3a->getDefinition(0) = Definition(ma);
   ctx->block->instructions.emplace_back(std::move(vop3a));
   tmp.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cubesc_f32, Format::VOP3A, 3, 1));
   for (unsigned i = 0; i < 3; i++)
      tmp->getOperand(i) = Operand(coord_args[i]);
   sc = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(sc);
   ctx->block->instructions.emplace_back(std::move(tmp));
   tmp.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madak_f32, Format::VOP2, 3, 1));
   tmp->getOperand(0) = Operand(sc);
   tmp->getOperand(1) = Operand(ma);
   tmp->getOperand(2) = Operand((uint32_t) 0x3fc00000); /* 1.5 */
   sc = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(sc);
   ctx->block->instructions.emplace_back(std::move(tmp));
   tmp.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cubetc_f32, Format::VOP3A, 3, 1));
   for (unsigned i = 0; i < 3; i++)
      tmp->getOperand(i) = Operand(coord_args[i]);
   tc = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(tc);
   ctx->block->instructions.emplace_back(std::move(tmp));
   tmp.reset(create_instruction<VOP2_instruction>(aco_opcode::v_madak_f32, Format::VOP2, 3, 1));
   tmp->getOperand(0) = Operand(tc);
   tmp->getOperand(1) = Operand(ma);
   tmp->getOperand(2) = Operand((uint32_t) 0x3fc00000); /* 1.5 */
   tc = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(tc);
   ctx->block->instructions.emplace_back(std::move(tmp));
   tmp.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cubeid_f32, Format::VOP3A, 3, 1));
   for (unsigned i = 0; i < 3; i++)
      tmp->getOperand(i) = Operand(coord_args[i]);
   id = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(id);
   ctx->block->instructions.emplace_back(std::move(tmp));
   tmp.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1));
   tmp->getOperand(0) = Operand(sc);
   tmp->getOperand(1) = Operand(tc);
   tmp->getOperand(2) = Operand(id);
   *coords = {ctx->program->allocateId(), v3};
   tmp->getDefinition(0) = Definition(*coords);
   ctx->block->instructions.emplace_back(std::move(tmp));

   if (is_deriv || is_array)
      fprintf(stderr, "Unimplemented tex instr type: cube coords2");

}

Temp apply_round_slice(isel_context *ctx, Temp coords, unsigned idx)
{
   Temp coord_vec[3];
   emit_split_vector(ctx, coords, coords.size());
   for (unsigned i = 0; i < coords.size(); i++)
      coord_vec[i] = emit_extract_vector(ctx, coords, i, v1);

   aco_ptr<VOP1_instruction> rne{create_instruction<VOP1_instruction>(aco_opcode::v_rndne_f32, Format::VOP1, 1, 1)};
   rne->getOperand(0) = Operand(coord_vec[idx]);
   coord_vec[idx] = {ctx->program->allocateId(), v1};
   rne->getDefinition(0) = Definition(coord_vec[idx]);
   ctx->block->instructions.emplace_back(std::move(rne));

   aco_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, coords.size(), 1)};
   for (unsigned i = 0; i < coords.size(); i++)
      vec->getOperand(i) = Operand(coord_vec[i]);
   Temp res = {ctx->program->allocateId(), coords.regClass()};
   vec->getDefinition(0) = Definition(res);
   ctx->block->instructions.emplace_back(std::move(vec));
   return res;
}

void visit_tex(isel_context *ctx, nir_tex_instr *instr)
{
   bool has_bias = false, has_lod = false, level_zero = false, has_compare = false,
        has_offset = false, has_ddx = false, has_ddy = false, has_derivs = false, has_sample_index = false;
   Temp resource, sampler, fmask_ptr, bias, coords, compare, sample_index,
        lod = Temp(), offset = Temp(), ddx = Temp(), ddy = Temp(), derivs = Temp();
   tex_fetch_ptrs(ctx, instr, &resource, &sampler, &fmask_ptr);

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_coord:
         coords = get_ssa_temp(ctx, instr->src[i].src.ssa);
         break;
      case nir_tex_src_bias:
         if (instr->op == nir_texop_txb) {
            bias = get_ssa_temp(ctx, instr->src[i].src.ssa);
            has_bias = true;
         }
         break;
      case nir_tex_src_lod: {
         nir_const_value *val = nir_src_as_const_value(instr->src[i].src);

         if (val && val->i32[0] == 0) {
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
      unreachable("Unimplemented tex instr type");

   if (instr->op == nir_texop_texture_samples)
      unreachable("Unimplemented tex instr type");

   if (has_offset && instr->op != nir_texop_txf) {
      aco_ptr<Instruction> tmp_instr;
      Temp acc, pack = Temp();
      for (unsigned i = 0; i < offset.size(); i++) {
         acc = emit_extract_vector(ctx, offset, i, s1);

         tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b32, Format::SOP2, 2, 2));
         tmp_instr->getOperand(0) = Operand(acc);
         tmp_instr->getOperand(1) = Operand((uint32_t) 0x3F);
         acc = {ctx->program->allocateId(), s1};
         tmp_instr->getDefinition(0) = Definition(acc);
         tmp_instr->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(tmp_instr));

         if (i == 0) {
            pack = acc;
         } else {
            tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 2));
            tmp_instr->getOperand(0) = Operand(pack);
            tmp_instr->getOperand(1) = Operand((uint32_t) 8 * i);
            acc = {ctx->program->allocateId(), s1};
            tmp_instr->getDefinition(0) = Definition(acc);
            tmp_instr->getDefinition(1) = Definition(PhysReg{253}, b);
            ctx->block->instructions.emplace_back(std::move(tmp_instr));

            tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b32, Format::SOP2, 2, 2));
            tmp_instr->getOperand(0) = Operand(pack);
            tmp_instr->getOperand(1) = Operand(acc);
            pack = {ctx->program->allocateId(), s1};
            tmp_instr->getDefinition(0) = Definition(pack);
            tmp_instr->getDefinition(1) = Definition(PhysReg{253}, b);
            ctx->block->instructions.emplace_back(std::move(tmp_instr));
         }
      }
      offset = pack;
   }

   /* pack derivatives */
   if (has_ddx || has_ddy) {
      aco_ptr<Instruction> pack_derivs;
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_1D && ctx->options->chip_class >= GFX9) {
         pack_derivs.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 4, 1));
         pack_derivs->getOperand(0) = Operand((uint32_t) 0);
         pack_derivs->getOperand(1) = Operand(ddx);
         pack_derivs->getOperand(2) = Operand((uint32_t) 0);
         pack_derivs->getOperand(3) = Operand(ddy);
         derivs = {ctx->program->allocateId(), v4};
      } else {
         pack_derivs.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1));
         pack_derivs->getOperand(0) = Operand(ddx);
         pack_derivs->getOperand(1) = Operand(ddy);
         derivs = {ctx->program->allocateId(), getRegClass(vgpr, ddx.size() + ddy.size())};
      }
      pack_derivs->getDefinition(0) = Definition(derivs);
      ctx->block->instructions.emplace_back(std::move(pack_derivs));
      has_derivs = true;
   }

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && instr->coord_components)
      prepare_cube_coords(ctx, &coords, instr->op == nir_texop_txd, instr->is_array, instr->op == nir_texop_lod);

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
       instr->op != nir_texop_lod) {
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

   if (instr->op == nir_texop_samples_identical)
      resource = fmask_ptr;

   else if (instr->sampler_dim == GLSL_SAMPLER_DIM_MS &&
       instr->op != nir_texop_txs) {
      assert(has_sample_index);
      sample_index = adjust_sample_index_using_fmask(ctx, coords, sample_index, fmask_ptr);
   }

   if (has_offset && instr->op == nir_texop_txf)
      unreachable("Unimplemented tex instr type");

   bool da = false;
   if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      aco_image_dim dim = get_sampler_dim(ctx, instr->sampler_dim, instr->is_array);

      da = dim == aco_image_cube ||
           dim == aco_image_1darray ||
           dim == aco_image_2darray ||
           dim == aco_image_2darraymsaa;
   }

   /* Build tex instruction */
   // TODO: use nir_ssa_def_components_read(&instr->dest.ssa), but then dst size doesn't match the instruction's return value
   unsigned dmask = (1 << instr->dest.ssa.num_components) - 1;
   /* gather4 selects the component by dmask */
   if (instr->op == nir_texop_tg4) {
      if (instr->is_shadow)
         dmask = 1;
      else
         dmask = 1 << instr->component;
   }


   aco_ptr<MIMG_instruction> tex;
   if (instr->op == nir_texop_txs) {
      if (!has_lod) {
         aco_ptr<VOP1_instruction> mov{create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1)};
         mov->getOperand(0) = Operand((uint32_t) 0);
         lod = Temp{ctx->program->allocateId(), v1};
         mov->getDefinition(0) = Definition(lod);
         ctx->block->instructions.emplace_back(std::move(mov));
      }
      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1));
      tex->getOperand(0) = Operand(lod);
      tex->getOperand(1) = Operand(resource);
      tex->dmask = dmask;
      tex->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(tex));
      return;
   }

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
      unsigned last_bit = util_last_bit(dmask);
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
      aco_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(op, Format::MUBUF, 3, 1)};
      mubuf->getOperand(0) = Operand(coords);
      mubuf->getOperand(1) = Operand(resource);
      mubuf->getOperand(2) = Operand((uint32_t) 0);
      mubuf->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      mubuf->idxen = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));
      emit_split_vector(ctx, get_ssa_temp(ctx, &instr->dest.ssa), instr->dest.ssa.num_components);
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
      Temp dst = instr->op == nir_texop_samples_identical ? Temp{ctx->program->allocateId(), v1} : get_ssa_temp(ctx, &instr->dest.ssa);
      tex->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(tex));

      if (instr->op == nir_texop_samples_identical) {
         assert(dmask == 1);

         aco_ptr<Instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_eq_u32, Format::VOPC, 2, 1)};
         cmp->getOperand(0) = Operand((uint32_t) 0);
         cmp->getOperand(1) = Operand(dst);
         Temp tmp = {ctx->program->allocateId(), s2};
         cmp->getDefinition(0) = Definition(tmp);
         cmp->getDefinition(0).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(cmp));

         aco_ptr<Instruction> bcsel{create_instruction<VOP3A_instruction>(aco_opcode::v_cndmask_b32, static_cast<Format>((int)Format::VOP2 | (int)Format::VOP3A), 3, 1)};
         bcsel->getOperand(0) = Operand((uint32_t) 0);
         bcsel->getOperand(1) = Operand((uint32_t) -1);
         bcsel->getOperand(2) = Operand{tmp};
         bcsel->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
         ctx->block->instructions.emplace_back(std::move(bcsel));
      } else {
         emit_split_vector(ctx, get_ssa_temp(ctx, &instr->dest.ssa), instr->dest.ssa.num_components);
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
            opcode = aco_opcode::image_sample_c_d;
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
   }

   tex.reset(create_instruction<MIMG_instruction>(opcode, Format::MIMG, 3, 1));
   tex->getOperand(0) = arg;
   tex->getOperand(1) = Operand(resource);
   tex->getOperand(2) = Operand(sampler);
   tex->dmask = dmask;
   tex->da = da;
   tex->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(tex));

   emit_split_vector(ctx, get_ssa_temp(ctx, &instr->dest.ssa), instr->dest.ssa.num_components);

   if (instr->op == nir_texop_query_levels)
      unreachable("Unimplemented tex instr type");
   else if (instr->is_shadow && instr->is_new_style_shadow &&
            instr->op != nir_texop_txs && instr->op != nir_texop_lod &&
            instr->op != nir_texop_tg4)
      unreachable("Unimplemented tex instr type");
   else if (instr->op == nir_texop_txs &&
            instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
            instr->is_array)
      unreachable("Unimplemented tex instr type");
   else if (ctx->options->chip_class >= GFX9 &&
            instr->op == nir_texop_txs &&
            instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
            instr->is_array)
      unreachable("Unimplemented tex instr type");

}


void visit_phi(isel_context *ctx, nir_phi_instr *instr)
{
   aco_ptr<Instruction> phi;
   unsigned num_src = exec_list_length(&instr->srcs);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   aco_opcode opcode = ctx->divergent_vals[instr->dest.ssa.index] ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
   phi.reset(create_instruction<Instruction>(opcode, Format::PSEUDO, num_src, 1));
   std::map<unsigned, nir_ssa_def*> phi_src;
   nir_foreach_phi_src(src, instr)
      phi_src[src->pred->index] = src->src.ssa;

   /* if we have a linear phi on a divergent if, we know that one src is undef */
   if (opcode == aco_opcode::p_linear_phi && ctx->block->logical_predecessors[0] != ctx->block->linear_predecessors[0]) {
      assert(num_src == 2);
      Block* block;
      /* we place the phi either in the between-block or in the current block */
      if (phi_src.begin()->second->parent_instr->type != nir_instr_type_ssa_undef) {
         assert((++phi_src.begin())->second->parent_instr->type == nir_instr_type_ssa_undef);
         block = ctx->block->linear_predecessors[1]->linear_predecessors[0];
         phi->getOperand(0) = Operand(get_ssa_temp(ctx, phi_src.begin()->second));
      } else {
         assert((++phi_src.begin())->second->parent_instr->type != nir_instr_type_ssa_undef);
         block = ctx->block;
         phi->getOperand(0) = Operand(get_ssa_temp(ctx, (++phi_src.begin())->second));
      }
      phi->getOperand(1) = Operand();
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

   if (dst.size() == 1) {
      undef.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
   } else {
      undef.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1));
   }
   for (unsigned i = 0; i < dst.size(); i++)
      undef->getOperand(i) = Operand();
   undef->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(undef));
}

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
   b->instructions.push_back(
      aco_ptr<Instruction>(create_instruction<Instruction>(aco_opcode::p_logical_start,
                                                                   Format::PSEUDO, 0, 0)));
}

static void append_logical_end(Block *b)
{
   b->instructions.push_back(
      aco_ptr<Instruction>(create_instruction<Instruction>(aco_opcode::p_logical_end,
                                                                   Format::PSEUDO, 0, 0)));
}

void visit_jump(isel_context *ctx, nir_jump_instr *instr)
{
   Block *logical_target, *linear_target;
   aco_ptr<Instruction> aco_instr;
   aco_ptr<Pseudo_branch_instruction> branch;

   append_logical_end(ctx->block);
   switch (instr->type) {
   case nir_jump_break: {
      logical_target = ctx->cf_info.parent_loop.exit;
      add_logical_edge(ctx->block, logical_target);
      ctx->cf_info.has_break = true;

      if (ctx->cf_info.parent_if.is_divergent) {
         linear_target = ctx->cf_info.parent_if.merge_block;
         ctx->cf_info.parent_loop.has_divergent_break = true;

         /* remove current exec mask from active */
         aco_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2));
         aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
         aco_instr->getOperand(1) = Operand(exec, s2);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         aco_instr->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         aco_instr->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(aco_instr));

         /* set exec zero */
         aco_ptr<Instruction> restore_exec;
         restore_exec.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
         restore_exec->getOperand(0) = Operand((uint32_t) 0);
         restore_exec->getDefinition(0) = Definition(exec, s2);
         ctx->block->instructions.emplace_back(std::move(restore_exec));

      } else if (ctx->cf_info.parent_loop.has_divergent_continue) {
         Block* break_block = ctx->program->createAndInsertBlock();
         break_block->loop_nest_depth = ctx->cf_info.loop_nest_depth;

         /* there might be still active lanes due to previous continue */
         aco_instr.reset(create_instruction<SOP1_instruction>(aco_opcode::s_andn2_saveexec_b64, Format::SOP1, 2, 2));
         aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
         aco_instr->getOperand(1) = Operand(exec, s2);
         Temp temp = {ctx->program->allocateId(), s2};
         aco_instr->getDefinition(0) = Definition(temp);
         aco_instr->getDefinition(1) = Definition(PhysReg{253}, b); /* scc */
         ctx->block->instructions.emplace_back(std::move(aco_instr));

         /* branch to loop entry if still lanes are active */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_nz, Format::PSEUDO_BRANCH, 1, 0));
         branch->getOperand(0) = Operand(PhysReg{253}, b); /* scc */
         branch->targets[0] = ctx->cf_info.parent_loop.entry;
         branch->targets[1] = break_block;
         ctx->block->instructions.emplace_back(std::move(branch));
         add_linear_edge(ctx->block, ctx->cf_info.parent_loop.entry);
         add_linear_edge(ctx->block, break_block);

         ctx->block = break_block;

         /* branch out of the loop */
         linear_target = ctx->cf_info.parent_loop.exit;
      } else {
         /* uniform break - directly jump out of the loop */
         linear_target = ctx->cf_info.parent_loop.exit;
      }

      break;
   }
   case nir_jump_continue:
      logical_target = ctx->cf_info.parent_loop.entry;
      add_logical_edge(ctx->block, logical_target);
      ctx->cf_info.has_continue = true;

      if (ctx->cf_info.parent_if.is_divergent) {
         linear_target = ctx->cf_info.parent_if.merge_block;
         /* for potential uniform breaks after this continue,
            we must ensure that they are handled correctly */
         ctx->cf_info.parent_loop.has_divergent_continue = true;

         /* set exec zero */
         aco_instr.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
         aco_instr->getOperand(0) = Operand((uint32_t) 0);
         aco_instr->getDefinition(0) = Definition(exec, s2);
         ctx->block->instructions.emplace_back(std::move(aco_instr));
      } else {
         /* uniform continue - directly jump to the loop entry block */
         linear_target = logical_target;

         if (ctx->cf_info.parent_loop.has_divergent_break) {
            /* restore exec with all continues */
            aco_instr.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
            aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
            aco_instr->getDefinition(0) = Definition(exec, s2);
            ctx->block->instructions.emplace_back(std::move(aco_instr));
         }
      }
      break;
   default:
      fprintf(stderr, "Unknown NIR jump instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }

   /* branch to linear target */
   branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
   branch->targets[0] = linear_target;
   ctx->block->instructions.emplace_back(std::move(branch));

   add_linear_edge(ctx->block, linear_target);
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
   aco_ptr<Pseudo_branch_instruction> branch;
   append_logical_end(ctx->block);
   /* save original exec */
   aco_ptr<Instruction> save_exec;
   save_exec.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
   save_exec->getOperand(0) = Operand{exec, s2};
   Temp orig_exec = {ctx->program->allocateId(), s2};
   save_exec->getDefinition(0) = Definition(orig_exec);
   ctx->block->instructions.emplace_back(std::move(save_exec));

   Block* loop_entry = ctx->program->createAndInsertBlock();
   loop_entry->loop_nest_depth = ctx->cf_info.loop_nest_depth + 1;
   Block* loop_exit = new Block();
   loop_exit->loop_nest_depth = ctx->cf_info.loop_nest_depth;
   branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
   branch->targets[0] = loop_entry;
   ctx->block->instructions.emplace_back(std::move(branch));
   add_edge(ctx->block, loop_entry);
   ctx->block = loop_entry;

   /* save current exec as active mask */
   save_exec.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
   save_exec->getOperand(0) = Operand{exec, s2};
   Temp active_mask = {ctx->program->allocateId(), s2};
   save_exec->getDefinition(0) = Definition(active_mask);
   ctx->block->instructions.emplace_back(std::move(save_exec));

   /* emit loop body */
   loop_info_RAII loop_raii(ctx, loop_entry, loop_exit, orig_exec, active_mask);
   append_logical_start(ctx->block);
   visit_cf_list(ctx, &loop->body);

   aco_ptr<Instruction> restore;

   if (ctx->cf_info.has_break) {
      ctx->cf_info.has_break = false;
   } else {
      append_logical_end(ctx->block);
      if (ctx->cf_info.parent_loop.has_divergent_continue) {
         /* restore all 'continue' lanes */
         restore.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 2));
         restore->getOperand(0) = Operand{exec, s2};
         if (ctx->cf_info.parent_loop.has_divergent_break)
            restore->getOperand(1) = Operand(ctx->cf_info.parent_loop.active_mask);
         else
            restore->getOperand(1) = Operand(ctx->cf_info.parent_loop.orig_exec);
         restore->getDefinition(0) = Definition{exec, s2};
         restore->getDefinition(1) = Definition(PhysReg{253}, b);
         ctx->block->instructions.emplace_back(std::move(restore));
      }

      add_logical_edge(ctx->block, loop_entry);

      if (ctx->cf_info.parent_loop.has_divergent_break) {
         Block* loop_continue = ctx->program->createAndInsertBlock();
         loop_continue->loop_nest_depth = ctx->cf_info.loop_nest_depth;
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
         branch->getOperand(0) = Operand{exec, s2};
         branch->targets[0] = loop_exit;
         branch->targets[1] = loop_continue;
         ctx->block->instructions.emplace_back(std::move(branch));
         add_linear_edge(ctx->block, loop_exit);
         add_linear_edge(ctx->block, loop_continue);
         ctx->block = loop_continue;
      }

      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = loop_entry;
      ctx->block->instructions.emplace_back(std::move(branch));
      add_linear_edge(ctx->block, loop_entry);
   }

   /* emit loop successor block */
   loop_exit->index = ctx->program->blocks.size();
   ctx->block = loop_exit;
   ctx->program->blocks.emplace_back(loop_exit);
   /* restore original exec */
   if (ctx->cf_info.parent_loop.has_divergent_break || ctx->cf_info.parent_loop.has_divergent_continue) {
      restore.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 2));
      restore->getOperand(0) = Operand{exec, s2};
      restore->getOperand(1) = Operand(ctx->cf_info.parent_loop.orig_exec);
      restore->getDefinition(0) = Definition{exec, s2};
      restore->getDefinition(1) = Definition(PhysReg{253}, b);
      ctx->block->instructions.emplace_back(std::move(restore));
   }

   append_logical_start(ctx->block);

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
}

static void visit_if(isel_context *ctx, nir_if *if_stmt)
{
   Temp cond32 = get_ssa_temp(ctx, if_stmt->condition.ssa);
   aco_ptr<Pseudo_branch_instruction> branch;

   if (cond32.type() == RegType::sgpr) { /* uniform condition */
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
      Block* BB_then = ctx->program->createAndInsertBlock();
      BB_then->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_else = new Block();
      BB_else->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_endif = new Block();
      BB_endif->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* parent_if_merge_block = ctx->cf_info.parent_if.merge_block;
      ctx->cf_info.parent_if.merge_block = BB_endif;

      /** emit conditional statement */
      Temp cond = extract_uniform_cond32(ctx, cond32);
      append_logical_end(BB_if);

      /* emit branch */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(cond);
      branch->getOperand(0).setFixed({253});
      branch->targets[0] = BB_else;
      branch->targets[1] = BB_then;
      BB_if->instructions.emplace_back(std::move(branch));
      add_edge(BB_if, BB_then);
      add_edge(BB_if, BB_else);

      /* remember active lanes mask just in case */
      Temp active_mask_if = ctx->cf_info.parent_loop.active_mask;

      /** emit then block */
      append_logical_start(BB_then);
      ctx->block = BB_then;
      visit_cf_list(ctx, &if_stmt->then_list);
      BB_then = ctx->block;

      if (!ctx->cf_info.has_break && !ctx->cf_info.has_continue) {
         append_logical_end(BB_then);
         /* branch from then block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_then->instructions.emplace_back(std::move(branch));
         add_edge(BB_then, BB_endif);
      }
      Temp active_mask_then = ctx->cf_info.parent_loop.active_mask;
      bool break_then = active_mask_then.id() != active_mask_if.id();

      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /** emit else block */
      BB_else->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else);
      append_logical_start(BB_else);
      ctx->block = BB_else;
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else = ctx->block;

      ctx->cf_info.parent_loop.active_mask = active_mask_if;

      if (!ctx->cf_info.has_break && !ctx->cf_info.has_continue) {
         append_logical_end(BB_else);
         /* branch from then block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_else->instructions.emplace_back(std::move(branch));
         add_edge(BB_else, BB_endif);
      }
      Temp active_mask_else = ctx->cf_info.parent_loop.active_mask;
      bool break_else = active_mask_else.id() != active_mask_if.id();

      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /** emit endif merge block */
      BB_endif->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_endif);

      /* emit linear phi for active mask */
      if (break_then || break_else) {

         aco_ptr<Instruction> phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1)};
         phi->getOperand(0) = Operand(break_then ? active_mask_then : active_mask_if);
         phi->getOperand(1) = Operand(break_else ? active_mask_else : active_mask_if);
         Temp active_mask = {ctx->program->allocateId(), s2};
         phi->getDefinition(0) = Definition(active_mask);
         BB_endif->instructions.emplace_back(std::move(phi));
         ctx->cf_info.parent_loop.active_mask = active_mask;
      }
      append_logical_start(BB_endif);
      ctx->block = BB_endif;
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
      Block* BB_then_logical = ctx->program->createAndInsertBlock();
      BB_then_logical->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_then_linear = new Block();
      BB_then_linear->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_between = new Block();
      BB_between->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_else_logical = new Block();
      BB_else_logical->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_else_linear = new Block();
      BB_else_linear->loop_nest_depth = ctx->cf_info.loop_nest_depth;
      Block* BB_endif = new Block();
      BB_endif->loop_nest_depth = ctx->cf_info.loop_nest_depth;

      /** emit conditional statement */
      Temp cond = extract_divergent_cond32(ctx, cond32);
      append_logical_end(BB_if);

      /* create the exec mask for then branch */
      aco_ptr<SOP1_instruction> set_exec{create_instruction<SOP1_instruction>(aco_opcode::s_and_saveexec_b64, Format::SOP1, 1, 2)};
      set_exec->getOperand(0) = Operand(cond);
      Temp orig_exec = {ctx->program->allocateId(), s2};
      set_exec->getDefinition(0) = Definition(orig_exec);
      set_exec->getDefinition(1) = Definition(PhysReg{253}, b);
      BB_if->instructions.push_back(std::move(set_exec));

      /* create the exec mask for else branch */
      aco_ptr<SOP2_instruction> nand{create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 2)};
      nand->getOperand(0) = Operand(orig_exec);
      nand->getOperand(1) = Operand(cond);
      Temp else_mask = {ctx->program->allocateId(), s2};
      nand->getDefinition(0) = Definition(else_mask);
      nand->getDefinition(1) = Definition(PhysReg{253}, b);
      BB_if->instructions.push_back(std::move(nand));

      /* branch to linear then block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(exec, s2);
      branch->targets[0] = BB_then_linear;
      branch->targets[1] = BB_then_logical;
      BB_if->instructions.push_back(std::move(branch));
      add_edge(BB_if, BB_then_logical);
      add_linear_edge(BB_if, BB_then_linear);
      add_logical_edge(BB_if, BB_else_logical);
      if_info_RAII if_raii(ctx, BB_between);

      /* remember active lanes mask just in case */
      Temp active_mask = ctx->cf_info.parent_loop.active_mask;

      /** emit logical then block */
      ctx->block = BB_then_logical;
      append_logical_start(BB_then_logical);
      visit_cf_list(ctx, &if_stmt->then_list);
      BB_then_logical = ctx->block;

      Temp active_mask_new = ctx->cf_info.parent_loop.active_mask;

      if (!ctx->cf_info.has_break && !ctx->cf_info.has_continue) {
         append_logical_end(BB_then_logical);
         /* branch from logical then block to between block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_between;
         BB_then_logical->instructions.emplace_back(std::move(branch));
         add_linear_edge(BB_then_logical, BB_between);
         add_logical_edge(BB_then_logical, BB_endif);
      }

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

      if (active_mask.id() != active_mask_new.id()) {
         /* emit linear phi for active mask */
         aco_ptr<Instruction> phi;
         phi.reset(create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1));
         phi->getOperand(0) = Operand(active_mask_new);
         phi->getOperand(1) = Operand(active_mask);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         phi->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         BB_between->instructions.push_back(std::move(phi));
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /* invert exec mask */
      aco_ptr<Instruction> mov;
      mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
      mov->getOperand(0) = Operand(exec, s2);
      Temp then_mask = {ctx->program->allocateId(), s2};
      mov->getDefinition(0) = Definition(then_mask);
      BB_between->instructions.push_back(std::move(mov));
      mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
      mov->getOperand(0) = Operand(else_mask);
      mov->getDefinition(0) = Definition(exec, s2);
      BB_between->instructions.push_back(std::move(mov));

      /* branch to linear else block (skip else) */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(PhysReg{126}, s2);
      branch->targets[0] = BB_else_linear;
      branch->targets[1] = BB_else_logical;
      BB_between->instructions.push_back(std::move(branch));
      add_linear_edge(BB_between, BB_else_linear);
      add_linear_edge(BB_between, BB_else_logical);

      active_mask = ctx->cf_info.parent_loop.active_mask;

      /** emit logical else block */
      BB_else_logical->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else_logical);
      ctx->cf_info.parent_if.merge_block = BB_endif;
      ctx->block = BB_else_logical;
      append_logical_start(BB_else_logical);
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else_logical = ctx->block;

      active_mask_new = ctx->cf_info.parent_loop.active_mask;

      if (!ctx->cf_info.has_break && !ctx->cf_info.has_continue) {
         append_logical_end(BB_else_logical);
         /* branch from logical else block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_else_logical->instructions.emplace_back(std::move(branch));
         add_edge(BB_else_logical, BB_endif);
      }

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

      if (active_mask.id() != active_mask_new.id()) {
         /* emit linear phi for active mask */
         aco_ptr<Instruction> phi;
         phi.reset(create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1));
         phi->getOperand(0) = Operand(active_mask_new);
         phi->getOperand(1) = Operand(active_mask);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         phi->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         BB_endif->instructions.push_back(std::move(phi));
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /* restore original exec mask */
      aco_ptr<SOP2_instruction> restore{create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 2)};
      restore->getOperand(0) = Operand(exec, s2);
      restore->getOperand(1) = Operand(then_mask);
      restore->getDefinition(0) = Definition(exec, s2);
      restore->getDefinition(1) = Definition(PhysReg{253}, b);
      BB_endif->instructions.emplace_back(std::move(restore));

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

std::unique_ptr<Program> select_program(struct nir_shader *nir,
                                        ac_shader_config* config,
                                        struct radv_shader_variant_info *info,
                                        struct radv_nir_compiler_options *options)
{
   std::unique_ptr<Program> program{new Program};
   isel_context ctx = setup_isel_context(program.get(), nir, config, info, options);

   // TODO: this is more a workaround until we have proper wqm handling
   if (ctx.stage == MESA_SHADER_FRAGMENT) {
      aco_ptr<Instruction> wqm{create_instruction<Instruction>(aco_opcode::s_wqm_b64, Format::SOP1, 1, 2)};
      wqm->getOperand(0) = Operand(PhysReg{126}, s2);
      wqm->getDefinition(0) = Definition(PhysReg{126}, s2);
      wqm->getDefinition(1) = Definition(PhysReg{253}, b);
      ctx.block->instructions.push_back(std::move(wqm));
   }

   append_logical_start(ctx.block);

   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   visit_cf_list(&ctx, &func->impl->body);

   append_logical_end(ctx.block);
   ctx.block->instructions.push_back(aco_ptr<SOPP_instruction>(create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)));

   return program;
}
}
