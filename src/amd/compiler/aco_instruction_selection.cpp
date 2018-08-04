#include <algorithm>
#include <unordered_map>
#include <set>
#include <stack>

#include "aco_ir.h"
#include "aco_builder.h"
#include "aco_interface.h"
#include "nir/nir.h"
#include "common/sid.h"
#include "vulkan/radv_shader.h"

#include "gallium/auxiliary/util/u_math.h"
namespace aco {
namespace {

enum fs_input {
   persp_sample,
   persp_center_p1,
   persp_center_p2,
   persp_centroid,
   persp_pull_model,
   linear_sample,
   linear_center,
   linear_centroid,
   line_stipple,
   frag_pos_0,
   frag_pos_1,
   frag_pos_2,
   frag_pos_3,
   front_face,
   ancillary,
   sample_coverage,
   fixed_pt,
   max_inputs,
};

struct loop_info {
   Block* loop_header;
   Block* loop_footer;
   Temp continues;
};

struct isel_context {
   struct radv_nir_compiler_options *options;
   Program *program;
   Block *block;
   bool *divergent_vals;
   std::unique_ptr<RegClass[]> reg_class;
   std::unordered_map<unsigned, unsigned> allocated;
   gl_shader_stage stage;
   struct {
      bool has_continue;
      bool has_break;
      struct {
         Block* entry;
         Block* exit;
         Temp active_mask;
         Temp orig_exec;
         bool has_divergent_continue;
         bool has_divergent_break;
      } parent_loop;
      struct {
         Block* merge_block;
         bool is_divergent;
      } parent_if;
   } cf_info;

   /* FS inputs */
   bool fs_vgpr_args[fs_input::max_inputs];
   Temp fs_inputs[fs_input::max_inputs];
   Temp prim_mask;
   Temp descriptor_sets[RADV_UD_MAX_SETS];
   Temp push_constants;
   Temp ring_offsets;
   Temp sample_pos_offset;
   //std::unordered_map<uint64_t, Temp> tex_desc;

   /* VS inputs */
   Temp vertex_buffers;
   Temp base_vertex;
   Temp start_instance;
   Temp draw_id;
   Temp view_index;
   Temp es2gs_offset;
   Temp vertex_id;
   Temp rel_auto_id;
   Temp instance_id;
   Temp vs_prim_id;

   uint32_t input_mask;
};

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


void emit_v_mov(isel_context *ctx, Temp src, Temp dst)
{
   std::unique_ptr<Instruction> mov;
   if (dst.size() == 1)
   {
      mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
      mov->getDefinition(0) = Definition(dst);
      mov->getOperand(0) = Operand(src);
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 1, 1)};
      vec->getOperand(0) = Operand(src);
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
   }
}

Temp emit_extract_vector(isel_context* ctx, Temp src, uint32_t idx, RegClass dst_rc)
{
   /* no need to extract the whole vector */
   if (src.regClass() == dst_rc) {
      assert(idx == 0);
      return src;
   }

   Temp dst = {ctx->program->allocateId(), dst_rc};
   if (src.size() == sizeOf(dst_rc)) {
      assert(idx == 0);
      emit_v_mov(ctx, src, dst);
   } else {
      std::unique_ptr<Instruction> extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
      extract->getOperand(0) = Operand(src);
      extract->getOperand(1) = Operand(idx);
      extract->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(extract));
   }
   return dst;
}

Temp get_alu_src(struct isel_context *ctx, nir_alu_src src)
{
   if (src.src.ssa->num_components == 1 && src.swizzle[0] == 0)
      return get_ssa_temp(ctx, src.src.ssa);

   Temp vec = get_ssa_temp(ctx, src.src.ssa);
   return emit_extract_vector(ctx, vec, src.swizzle[0], getRegClass(vec.type(), 1));
}

Temp convert_pointer_to_64_bit(isel_context *ctx, Temp ptr)
{
      std::unique_ptr<Instruction> tmp{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      tmp->getOperand(0) = Operand(ptr);
      tmp->getOperand(1) = Operand((unsigned)ctx->options->address32_hi);
      Temp ptr64 = {ctx->program->allocateId(), s2};
      tmp->getDefinition(0) = Definition(ptr64);
      ctx->block->instructions.emplace_back(std::move(tmp));
      return ptr64;
}

void emit_sop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   std::unique_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(op, Format::SOP2, 2, 1)};
   sop2->getOperand(0) = Operand(get_alu_src(ctx, instr->src[0]));
   sop2->getOperand(1) = Operand(get_alu_src(ctx, instr->src[1]));
   sop2->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(sop2));
}

void emit_vop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst, bool commutative)
{
   Temp src0 = get_alu_src(ctx, instr->src[0]);
   Temp src1 = get_alu_src(ctx, instr->src[1]);
   std::unique_ptr<Instruction> vop2{create_instruction<VOP2_instruction>(op, Format::VOP2, 2, 1)};
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
   std::unique_ptr<VOP1_instruction> vop1{create_instruction<VOP1_instruction>(op, Format::VOP1, 1, 1)};
   vop1->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
   vop1->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(vop1));
}

void emit_vopc_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op, Temp dst)
{
   Temp src0 = get_alu_src(ctx, instr->src[0]);
   Temp src1 = get_alu_src(ctx, instr->src[1]);
   std::unique_ptr<Instruction> vopc;
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
      std::unique_ptr<Instruction> bcsel{create_instruction<VOP3A_instruction>(aco_opcode::v_cndmask_b32, static_cast<Format>((int)Format::VOP2 | (int)Format::VOP3A), 3, 1)};
      bcsel->getOperand(0) = Operand{0x0};
      bcsel->getOperand(1) = Operand{0xFFFFFFFf};
      bcsel->getOperand(2) = Operand{tmp};
      bcsel->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(bcsel));
   } else {
      Temp scc_tmp{ctx->program->allocateId(), b};
      std::unique_ptr<Instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lg_u64, Format::SOPC, 2, 1)};
      cmp->getOperand(0) = Operand{tmp};
      cmp->getOperand(1) = Operand{0};
      cmp->getDefinition(0) = Definition{scc_tmp};
      ctx->block->instructions.emplace_back(std::move(cmp));

      std::unique_ptr<Instruction> cselect{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
      cselect->getOperand(0) = Operand{0xFFFFFFFF};
      cselect->getOperand(1) = Operand{0};
      cselect->getOperand(2) = Operand{scc_tmp};
      cselect->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(cselect));
   }
}


Temp extract_uniform_cond32(isel_context *ctx, Temp cond32)
{
   Temp cond = Temp{ctx->program->allocateId(), b};

   std::unique_ptr<Instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lg_u32, Format::SOPC, 2, 1)};
   cmp->getOperand(0) = Operand{cond32};
   cmp->getOperand(1) = Operand{0};
   cmp->getDefinition(0) = Definition{cond};
   cmp->getDefinition(0).setFixed(PhysReg{253}); /* scc */
   ctx->block->instructions.emplace_back(std::move(cmp));

   return cond;
}

Temp extract_divergent_cond32(isel_context *ctx, Temp cond32)
{
   Temp cond = Temp{ctx->program->allocateId(), s2};

   std::unique_ptr<Instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_lg_u32, Format::VOPC, 2, 1)};
   cmp->getOperand(0) = Operand{0};
   cmp->getOperand(1) = Operand{cond32};
   cmp->getDefinition(0) = Definition{cond};
   cmp->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(cmp));

   return cond;
}

void emit_quad_swizzle(isel_context *ctx, Temp src, Temp dst,
                       unsigned lane0, unsigned lane1, unsigned lane2, unsigned lane3)
{
   // TODO: we can do better using DPP instructions
   unsigned quad_mask = lane0 | (lane1 << 2) | (lane2 << 4) | (lane3 << 6);
   std::unique_ptr<DS_instruction> ds{create_instruction<DS_instruction>(aco_opcode::ds_swizzle_b32, Format::DS, 1, 1)};
   ds->getOperand(0) = Operand(src);
   ds->getDefinition(0) = Definition(dst);
   ds->offset0 = (1 << 15) | quad_mask;
   ctx->block->instructions.emplace_back(std::move(ds));
}

void emit_bcsel(isel_context *ctx, nir_alu_instr *instr, Temp dst)
{
   Temp cond32 = get_alu_src(ctx, instr->src[0]);
   Temp then = get_alu_src(ctx, instr->src[1]);
   Temp els = get_alu_src(ctx, instr->src[2]);

   Temp cond;
   if (cond32.type() == vgpr) {
      cond = extract_divergent_cond32(ctx, cond32);
   } else {
      cond = extract_uniform_cond32(ctx, cond32);
   }

   if (ctx->divergent_vals[instr->dest.dest.ssa.index]) {
      if (dst.type() == vgpr) {
         if (dst.size() == 1) {
            if (then.type() != vgpr) {
               Temp mov_dst = Temp(ctx->program->allocateId(), getRegClass(vgpr, then.size()));
               emit_v_mov(ctx, then, mov_dst);
               then = mov_dst;
            }
            if (els.type() != vgpr) {
               Temp mov_dst = Temp(ctx->program->allocateId(), getRegClass(vgpr, els.size()));
               emit_v_mov(ctx, els, mov_dst);
               els = mov_dst;
            }
            std::unique_ptr<Instruction> bcsel{create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1)};
            bcsel->getOperand(0) = Operand{els};
            bcsel->getOperand(1) = Operand{then};
            bcsel->getOperand(2) = Operand{cond};
            bcsel->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(bcsel));
         } else {
            fprintf(stderr, "Unimplemented NIR instr bit size: ");
            nir_print_instr(&instr->instr, stderr);
            fprintf(stderr, "\n");
         }
      } else { /* dst.type() == sgpr */
         /* this implements bcsel on bools: dst = s0 ? s1 : s2
          * are going to be: dst = (s0 & s1) | (~s0 & s2) */
         assert(cond.regClass() == s2 && then.regClass() == s2 && els.regClass() == s2);

         std::unique_ptr<SOP2_instruction> sop2;
         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b64, Format::SOP2, 2, 1));
         sop2->getOperand(0) = Operand(cond);
         sop2->getOperand(1) = Operand(then);
         then = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(then);
         ctx->block->instructions.emplace_back(std::move(sop2));

         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 1));
         sop2->getOperand(0) = Operand(els);
         sop2->getOperand(1) = Operand(cond);
         els = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(els);
         ctx->block->instructions.emplace_back(std::move(sop2));

         sop2.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1));
         sop2->getOperand(0) = Operand(then);
         sop2->getOperand(1) = Operand(els);
         then = Temp(ctx->program->allocateId(), s2);
         sop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(sop2));
      }
   } else { /* condition is uniform */
      if (cond.regClass() == s2) {
         cond = emit_extract_vector(ctx, cond, 0, s1);
         std::unique_ptr<SOPK_instruction> sopk{create_instruction<SOPK_instruction>(aco_opcode::s_cmpk_lg_u32, Format::SOPK, 1, 1)};
         sopk->getOperand(0) = Operand(cond);
         sopk->imm = 0;
         cond = {ctx->program->allocateId(), b};
         sopk->getDefinition(0) = Definition(cond);
         sopk->getDefinition(0).setFixed(PhysReg{253}); /* scc */
         ctx->block->instructions.emplace_back(std::move(sopk));
      }
      assert(cond.regClass() == b);
      if (dst.size() == 1) {
         assert(then.regClass() == s1 && els.regClass() == s1);
         std::unique_ptr<SOP2_instruction> select{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
         select->getOperand(0) = Operand(then);
         select->getOperand(1) = Operand(els);
         select->getOperand(2) = Operand(cond);
         select->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(select));
      } else {
         fprintf(stderr, "Unimplemented uniform bcsel bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
   }
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
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.dest.ssa.num_components, 1)};
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; ++i) {
         vec->getOperand(i) = Operand{get_alu_src(ctx, instr->src[i])};
      }
      vec->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(vec));
      break;
   }
   case nir_op_imov:
   case nir_op_fmov: {
      std::unique_ptr<Instruction> mov;
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
         std::unique_ptr<Instruction> sop1{create_instruction<SOP1_instruction>(opcode, Format::SOP1, 1, 1)};
         sop1->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
         sop1->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(sop1));
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
         emit_sop2_instruction(ctx, instr, aco_opcode::s_or_b32, dst);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_or_b64, dst);
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
         emit_sop2_instruction(ctx, instr, aco_opcode::s_and_b32, dst);
      } else if (dst.regClass() == s2) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_and_b64, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ishl: {
      if (dst.regClass() == v1) {
         std::unique_ptr<VOP2_instruction> shl{create_instruction<VOP2_instruction>(aco_opcode::v_lshlrev_b32, Format::VOP2, 2, 1)};
         shl->getOperand(0) = Operand{get_alu_src(ctx, instr->src[1])};
         shl->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
         shl->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(shl));
      } else if (dst.regClass() == s1) {
         std::unique_ptr<SOP2_instruction> shl{create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 2)};
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
   case nir_op_iadd: {
      if (dst.regClass() == v1) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_add_u32, dst, true);
      } else if (dst.regClass() == s1) {
         emit_sop2_instruction(ctx, instr, aco_opcode::s_add_i32, dst);
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
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_sub_f32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0);
         if (src.type() == sgpr) {
            Temp old_src = src;
            src = {ctx->program->allocateId(), v1};
            emit_v_mov(ctx, old_src, src);
         }
         vop2->getOperand(1) = Operand(src);
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
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_and_b32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0x7FFFFFFF);
         if (src.type() == sgpr) {
            Temp old_src = src;
            src = {ctx->program->allocateId(), v1};
            emit_v_mov(ctx, old_src, src);
         }
         vop2->getOperand(1) = Operand(src);
         vop2->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(vop2));
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
   case nir_op_fsin:
   case nir_op_fcos: {
      Temp src = get_alu_src(ctx, instr->src[0]);
      std::unique_ptr<Instruction> norm;
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
         std::unique_ptr<VOP1_instruction> vop1{create_instruction<VOP1_instruction>(opcode, Format::VOP1, 1, 1)};
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
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_and_b32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0x8FFFFFFF);
         vop2->getOperand(1) = Operand(src);
         Temp tmp = Temp(ctx->program->allocateId(), v1);
         vop2->getDefinition(0) = Definition(tmp);
         ctx->block->instructions.emplace_back(std::move(vop2));
         vop2.reset(create_instruction<VOP2_instruction>(aco_opcode::v_or_b32, Format::VOP2, 2, 1));
         vop2->getOperand(0) = Operand((uint32_t) 0x3f800000);
         vop2->getOperand(1) = Operand(tmp);
         tmp = Temp(ctx->program->allocateId(), v1);
         vop2->getDefinition(0) = Definition(tmp);
         ctx->block->instructions.emplace_back(std::move(vop2));
         std::unique_ptr<VOPC_instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_neq_f32, Format::VOPC, 2, 1)};
         cmp->getOperand(0) = Operand((unsigned) 0);
         cmp->getOperand(1) = Operand(src);
         Temp cmp_res = Temp(ctx->program->allocateId(), s2);
         cmp->getDefinition(0) = Definition(cmp_res);
         cmp->getDefinition(0).setHint(vcc);
         ctx->block->instructions.emplace_back(std::move(cmp));
         vop2.reset(create_instruction<VOP2_instruction>(aco_opcode::v_cndmask_b32, Format::VOP2, 3, 1));
         vop2->getOperand(0) = Operand((uint32_t) 0);
         vop2->getOperand(1) = Operand(tmp); // then
         vop2->getOperand(2) = Operand(cmp_res);
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
         std::unique_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
         readlane->getOperand(0) = Operand(tmp);
         readlane->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(readlane));
      } else {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, dst);
      }
      break;
   }
   case nir_op_b2f: {
      Temp cond32 = get_alu_src(ctx, instr->src[0]);
      if (dst.size() == 1) {
         std::unique_ptr<VOP3A_instruction> cndmask{create_instruction<VOP3A_instruction>(aco_opcode::v_and_b32, (Format) ((int) Format::VOP3A | (int) Format::VOP2), 2, 1)};
         cndmask->getOperand(0) = Operand(cond32);
         cndmask->getOperand(1) = Operand((uint32_t) 0x3f800000); /* 1.0 */
         cndmask->getDefinition(0) = Definition(dst);
         ctx->block->instructions.emplace_back(std::move(cndmask));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_feq: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_eq_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fne: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lg_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_flt: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lt_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fge: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_ge_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ieq: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_eq_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            std::unique_ptr<Instruction> cmp{create_instruction<VOP3A_instruction>(aco_opcode::v_cmp_eq_i32, (Format) ((int) Format::VOP3A | (int) Format::VOPC), 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            cmp->getDefinition(0) = Definition(dst_tmp);
            ctx->block->instructions.emplace_back(std::move(cmp));
            cmp.reset(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
            cmp->getOperand(0) = Operand(dst_tmp);
            cmp->getOperand(1) = Operand{0};
            cmp->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(cmp));
         } else {
            std::unique_ptr<SOPC_instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_eq_i32, Format::SOPC, 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            Temp scc = {ctx->program->allocateId(), b};
            cmp->getDefinition(0) = Definition(scc);
            cmp->getDefinition(0).setFixed({253}); /* scc */
            ctx->block->instructions.emplace_back(std::move(cmp));
            std::unique_ptr<SOP2_instruction> to_sgpr{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
            to_sgpr->getOperand(0) = Operand(0xFFFFFFFF);
            to_sgpr->getOperand(1) = Operand(0);
            to_sgpr->getOperand(2) = Operand(scc);
            to_sgpr->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(to_sgpr));
         }
      }
      break;
   }
   case nir_op_ilt: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_lt_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            std::unique_ptr<Instruction> cmp{create_instruction<VOP3A_instruction>(aco_opcode::v_cmp_lt_i32, (Format) ((int) Format::VOP3A | (int) Format::VOPC), 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            cmp->getDefinition(0) = Definition(dst_tmp);
            ctx->block->instructions.emplace_back(std::move(cmp));
            cmp.reset(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
            cmp->getOperand(0) = Operand(dst_tmp);
            cmp->getOperand(1) = Operand{0};
            cmp->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(cmp));
         } else {
            std::unique_ptr<SOPC_instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_lt_i32, Format::SOPC, 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            Temp scc = {ctx->program->allocateId(), b};
            cmp->getDefinition(0) = Definition(scc);
            cmp->getDefinition(0).setFixed({253}); /* scc */
            ctx->block->instructions.emplace_back(std::move(cmp));
            std::unique_ptr<SOP2_instruction> to_sgpr{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
            to_sgpr->getOperand(0) = Operand(0xFFFFFFFF);
            to_sgpr->getOperand(1) = Operand(0);
            to_sgpr->getOperand(2) = Operand(scc);
            to_sgpr->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(to_sgpr));
         }
      }
      break;
   }
   case nir_op_ige: {
      if (dst.regClass() == v1) {
         emit_vopc_instruction_output32(ctx, instr, aco_opcode::v_cmp_ge_i32, dst);
      } else if (dst.regClass() == s1) {
         Temp src0 = get_alu_src(ctx, instr->src[0]);
         Temp src1 = get_alu_src(ctx, instr->src[1]);
         if (src0.regClass() == v1 || src1.regClass() == v1) {
            Temp dst_tmp = {ctx->program->allocateId(), s2};
            std::unique_ptr<Instruction> cmp{create_instruction<VOP3A_instruction>(aco_opcode::v_cmp_ge_i32, (Format) ((int) Format::VOP3A | (int) Format::VOPC), 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            cmp->getDefinition(0) = Definition(dst_tmp);
            ctx->block->instructions.emplace_back(std::move(cmp));
            cmp.reset(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
            cmp->getOperand(0) = Operand(dst_tmp);
            cmp->getOperand(1) = Operand{0};
            cmp->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(cmp));
         } else {
            std::unique_ptr<SOPC_instruction> cmp{create_instruction<SOPC_instruction>(aco_opcode::s_cmp_ge_i32, Format::SOPC, 2, 1)};
            cmp->getOperand(0) = Operand(src0);
            cmp->getOperand(1) = Operand(src1);
            Temp scc = {ctx->program->allocateId(), b};
            cmp->getDefinition(0) = Definition(scc);
            cmp->getDefinition(0).setFixed({253}); /* scc */
            ctx->block->instructions.emplace_back(std::move(cmp));
            std::unique_ptr<SOP2_instruction> to_sgpr{create_instruction<SOP2_instruction>(aco_opcode::s_cselect_b32, Format::SOP2, 3, 1)};
            to_sgpr->getOperand(0) = Operand(0xFFFFFFFF);
            to_sgpr->getOperand(1) = Operand(0);
            to_sgpr->getOperand(2) = Operand(scc);
            to_sgpr->getDefinition(0) = Definition(dst);
            ctx->block->instructions.emplace_back(std::move(to_sgpr));
         }
      }
      break;
   }
   case nir_op_fddx:
   case nir_op_fddy: {
      Temp tl = {ctx->program->allocateId(), v1};
      Temp trbl = {ctx->program->allocateId(), v1};
      emit_quad_swizzle(ctx, get_alu_src(ctx, instr->src[0]), tl, 0, 0, 0, 0);
      if (instr->op == nir_op_fddx)
         emit_quad_swizzle(ctx, get_alu_src(ctx, instr->src[0]), trbl, 1, 1, 1, 1);
      else
         emit_quad_swizzle(ctx, get_alu_src(ctx, instr->src[0]), trbl, 2, 2, 2, 2);
      std::unique_ptr<Instruction> sub{create_instruction<VOP2_instruction>(aco_opcode::v_sub_f32, Format::VOP2, 2, 1)};
      sub->getOperand(0) = Operand(trbl);
      sub->getOperand(1) = Operand(tl);
      sub->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(sub));
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
   if (instr->def.bit_size != 32) {
      fprintf(stderr, "Unsupported load_const instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }

   std::unique_ptr<Instruction> mov;
   if (instr->def.num_components == 1)
   {
      if (typeOf(ctx->reg_class[instr->def.index]) == vgpr) {
         mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
      } else {
         mov.reset(create_instruction<Instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
      }
      mov->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->def));
      mov->getOperand(0) = Operand{instr->value.u32[0]};
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->def.num_components, 1)};
      for (unsigned i = 0; i < instr->def.num_components; i++)
         vec->getOperand(i) = Operand{instr->value.u32[i]};
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->def));
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
            std::unique_ptr<VOP3A_instruction> compr{create_instruction<VOP3A_instruction>(compr_op, Format::VOP3A, 2, 1)};
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

   std::unique_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
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
   std::unique_ptr<Interp_instruction> p1{create_instruction<Interp_instruction>(aco_opcode::v_interp_p1_f32, Format::VINTRP, 2, 1)};
   p1->getOperand(0) = Operand(coord1);
   p1->getOperand(1) = Operand(ctx->prim_mask);
   p1->getOperand(1).setFixed(m0);
   p1->getDefinition(0) = Definition(tmp);
   p1->attribute = idx;
   p1->component = component;
   std::unique_ptr<Interp_instruction> p2{create_instruction<Interp_instruction>(aco_opcode::v_interp_p2_f32, Format::VINTRP, 3, 1)};
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
      std::unique_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.ssa.num_components, 1));
      for (unsigned i = 0; i < instr->dest.ssa.num_components; i++)
         vec->getOperand(i) = Operand(ctx->fs_inputs[fs_input::frag_pos_0 + i]);
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
      return;
   }

   unsigned base = nir_intrinsic_base(instr) / 4 - VARYING_SLOT_VAR0;
   unsigned idx = util_bitcount(ctx->input_mask & ((1u << base) - 1));

   unsigned component = nir_intrinsic_component(instr);

   if (instr->dest.ssa.num_components == 1) {
      emit_interp_instr(ctx, idx, component, get_ssa_temp(ctx, instr->src[0].ssa), get_ssa_temp(ctx, &instr->dest.ssa));
   } else {
      std::unique_ptr<Instruction> vec(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.ssa.num_components, 1));
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
      std::unique_ptr<Instruction> load;
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
         std::unique_ptr<VOP2_instruction> add{create_instruction<VOP2_instruction>(aco_opcode::v_add_u32, Format::VOP2, 2, 1)};
         add->getOperand(0) = Operand(ctx->base_vertex);
         add->getOperand(1) = Operand(ctx->vertex_id);
         add->getDefinition(0) = Definition(index);
         ctx->block->instructions.emplace_back(std::move(add));
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

      std::unique_ptr<MUBUF_instruction> mubuf{create_instruction<MUBUF_instruction>(opcode, Format::MUBUF, 3, 1)};
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
      unsigned base = nir_intrinsic_base(instr) / 4 - VARYING_SLOT_VAR0;
      unsigned idx = util_bitcount(ctx->input_mask & ((1u << base) - 1));
      unsigned component = nir_intrinsic_component(instr);

      std::unique_ptr<Interp_instruction> mov{create_instruction<Interp_instruction>(aco_opcode::v_interp_mov_f32, Format::VINTRP, 2, 1)};
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
         std::unique_ptr<Instruction> tmp;
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
         std::unique_ptr<Instruction> tmp;
         tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
         tmp->getOperand(0) = Operand(offset);
         tmp->getOperand(1) = Operand(index);
         index = {ctx->program->allocateId(), index.regClass()};
         tmp->getDefinition(0) = Definition(index);
         ctx->block->instructions.emplace_back(std::move(tmp));
      }
   }

   std::unique_ptr<Instruction> tmp;
   tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
   tmp->getOperand(0) = nir_const_index ? Operand(const_index) : Operand(index);
   tmp->getOperand(1) = Operand(desc_ptr);
   index = {ctx->program->allocateId(), index.regClass()};
   tmp->getDefinition(0) = Definition(index);
   ctx->block->instructions.emplace_back(std::move(tmp));

   index = convert_pointer_to_64_bit(ctx, index);
   ctx->allocated.insert({instr->dest.ssa.index, index.id()});


}

void visit_load_ubo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp rsrc = get_ssa_temp(ctx, instr->src[0].ssa);
   nir_const_value* const_offset = nir_src_as_const_value(instr->src[1]);

   std::unique_ptr<Instruction> load;
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

         std::unique_ptr<Instruction> trimmed{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1)};
         std::unique_ptr<Instruction> extract;
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
      fprintf(stderr, "Unimplemented vector load\n");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
      #if 0
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

      std::unique_ptr<MUBUF_instruction> mubuf;
      mubuf.reset(create_instruction<MUBUF_instruction>(op, Format::MUBUF, 3, 1));
      mubuf->getOperand(0) = Operand(get_ssa_temp(ctx, instr->src[1].ssa));
      mubuf->getOperand(1) = Operand(rsrc);
      mubuf->getOperand(2) = Operand(0);
      mubuf->getDefinition(0) = Definition(dst);
      mubuf->offen = true;
      ctx->block->instructions.emplace_back(std::move(mubuf));
      #endif
   }
}

void visit_load_push_constant(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp index = get_ssa_temp(ctx, instr->src[0].ssa);
   unsigned offset = nir_intrinsic_base(instr);
   if (offset != 0) { // TODO check if index != 0 as well
      std::unique_ptr<SOP2_instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1)};
      add->getOperand(0) = Operand(offset);
      add->getOperand(1) = Operand(index);
      index = {ctx->program->allocateId(), s1};
      add->getDefinition(0) = Definition(index);
      ctx->block->instructions.emplace_back(std::move(add));
   }
   Temp ptr = ctx->push_constants;
   if (ptr.size() == 1) {
      ptr = convert_pointer_to_64_bit(ctx, ptr);
      ctx->push_constants = ptr;
   }

   unsigned range = nir_intrinsic_range(instr);
   aco_opcode op;
   switch (range) {
   case 4:
      op = aco_opcode::s_load_dword;
      break;
   case 8:
      op = aco_opcode::s_load_dwordx2;
      break;
   case 16:
      op = aco_opcode::s_load_dwordx4;
      break;
   default:
      unreachable("unimplemented or forbidden load_push_constant.");
   }
   std::unique_ptr<SMEM_instruction> load{create_instruction<SMEM_instruction>(op, Format::SMEM, 2, 1)};
   load->getOperand(0) = Operand(ptr);
   load->getOperand(1) = Operand(index);
   load->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(load));
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
   ctx->program->info->fs.can_discard = true;
   Temp cond32 = get_ssa_temp(ctx, instr->src[0].ssa);
   Temp cond = Temp{ctx->program->allocateId(), s2};

    std::unique_ptr<Instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_lg_u32, Format::VOPC, 2, 1)};
    cmp->getOperand(0) = Operand{0};
    cmp->getOperand(1) = Operand{cond32};
    cmp->getDefinition(0) = Definition{cond};
    cmp->getDefinition(0).setHint(vcc);
    ctx->block->instructions.emplace_back(std::move(cmp));

   std::unique_ptr<SOP2_instruction> sop2{create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 1)};
   sop2->getOperand(0) = Operand(exec, s2);
   sop2->getOperand(1) = Operand(cond);
   sop2->getDefinition(0) = Definition(exec, s2);
   ctx->block->instructions.emplace_back(std::move(sop2));

   std::unique_ptr<SOPP_instruction> branch{create_instruction<SOPP_instruction>(aco_opcode::s_cbranch_execnz, Format::SOPP, 1, 0)};
   branch->getOperand(0) = Operand(exec, s2);
   branch->imm = 3; /* (8 + 4 dwords) / 4 */
   ctx->block->instructions.emplace_back(std::move(branch));

   std::unique_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
   for (unsigned i = 0; i < 4; i++)
      exp->getOperand(i) = Operand();
   exp->enabled_mask = 0;
   exp->compressed = false;
   exp->done = true;
   exp->valid_mask = true;
   exp->dest = 9; /* NULL */
   ctx->block->instructions.emplace_back(std::move(exp));

   std::unique_ptr<SOPP_instruction> endpgm{create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)};
   ctx->block->instructions.emplace_back(std::move(endpgm));
}

void visit_intrinsic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   switch(instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel: {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(ctx->fs_inputs[fs_input::persp_center_p1]);
      vec->getOperand(1) = Operand(ctx->fs_inputs[fs_input::persp_center_p2]);
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
      break;
   }
   case nir_intrinsic_load_front_face: {
      std::unique_ptr<VOPC_instruction> cmp{create_instruction<VOPC_instruction>(aco_opcode::v_cmp_lg_u32, Format::VOPC, 2, 1)};
      cmp->getOperand(0) = Operand((uint32_t) 0);
      cmp->getOperand(1) = Operand(ctx->fs_inputs[fs_input::front_face]);
      cmp->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
      cmp->getDefinition(0).setHint(vcc);
      ctx->block->instructions.emplace_back(std::move(cmp));
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
   default:
      fprintf(stderr, "Unimplemented intrinsic instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();

      break;
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
               std::unique_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
               readlane->getOperand(0) = Operand(indirect);
               indirect = {ctx->program->allocateId(), s1};
               readlane->getDefinition(0) = Definition(indirect);
               ctx->block->instructions.emplace_back(std::move(readlane));
            }

            if (array_size != 1) {
               std::unique_ptr<Instruction> mul{create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1)};
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
               std::unique_ptr<Instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1)};
               add->getDefinition(0) = Definition{ctx->program->allocateId(), s1};
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
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 4, 1)};
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
      std::unique_ptr<SOP2_instruction> mul{create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1)};
      mul->getOperand(0) = Operand(stride);
      mul->getOperand(1) = Operand(index);
      Temp t = {ctx->program->allocateId(), s1};
      mul->getDefinition(0) = Definition(t);
      ctx->block->instructions.emplace_back(std::move(mul));
      std::unique_ptr<SOP2_instruction> add{create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1)};
      add->getOperand(0) = Operand(offset);
      add->getOperand(1) = Operand(t);
      t = {ctx->program->allocateId(), s1};
      add->getDefinition(0) = Definition(t);
      ctx->block->instructions.emplace_back(std::move(add));
      off = Operand(t);
   }

   std::unique_ptr<Instruction> load;
   load.reset(create_instruction<SMEM_instruction>(opcode, Format::SMEM, 2, 1));
   load->getOperand(0) = Operand(list);
   load->getOperand(1) = off;
   Temp t = {ctx->program->allocateId(), type};
   load->getDefinition(0) = Definition(t);
   ctx->block->instructions.emplace_back(std::move(load));
   //ctx->tex_desc.insert({(uint64_t) desc_type << 32 | deref_instr->dest.ssa.index, t});
   return t;
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
      fprintf(stderr, "Unimplemented tex instr type: ");

   Temp coord_args[3], ma, tc, sc, id;
   std::unique_ptr<Instruction> tmp;
   for (unsigned i = 0; i < 3; i++)
      coord_args[i] = emit_extract_vector(ctx, *coords, i, v1);

   tmp.reset(create_instruction<VOP3A_instruction>(aco_opcode::v_cubema_f32, Format::VOP3A, 3, 1));
   for (unsigned i = 0; i < 3; i++)
      tmp->getOperand(i) = Operand(coord_args[i]);
   ma = {ctx->program->allocateId(), v1};
   tmp->getDefinition(0) = Definition(ma);
   ctx->block->instructions.emplace_back(std::move(tmp));
   std::unique_ptr<VOP3A_instruction> vop3a{create_instruction<VOP3A_instruction>(aco_opcode::v_rcp_f32, (Format) ((uint16_t) Format::VOP3A | (uint16_t) Format::VOP1), 1, 1)};
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
      fprintf(stderr, "Unimplemented tex instr type: ");

}

Temp apply_round_slice(isel_context *ctx, Temp coords, unsigned idx)
{
   Temp coord_vec[3];
   for (unsigned i = 0; i < 3; i++)
      coord_vec[i] = emit_extract_vector(ctx, coords, i, v1);

   std::unique_ptr<VOP1_instruction> rne{create_instruction<VOP1_instruction>(aco_opcode::v_rndne_f32, Format::VOP1, 1, 1)};
   rne->getOperand(0) = Operand(coord_vec[idx]);
   coord_vec[idx] = {ctx->program->allocateId(), v1};
   rne->getDefinition(0) = Definition(coord_vec[idx]);
   ctx->block->instructions.emplace_back(std::move(rne));

   std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 3, 1)};
   for (unsigned i = 0; i < 3; i++)
      vec->getOperand(i) = Operand(coord_vec[i]);
   Temp res = {ctx->program->allocateId(), v3};
   vec->getDefinition(0) = Definition(res);
   ctx->block->instructions.emplace_back(std::move(vec));
   return res;
}

void visit_tex(isel_context *ctx, nir_tex_instr *instr)
{
   bool has_bias = false, has_lod = false, level_zero = false, has_compare = false,
        has_offset = false, has_ddx = false, has_ddy = false, has_derivs = false;
   Temp resource, sampler, fmask_ptr, bias, coords, compare, lod = Temp(), offset = Temp(), ddx, ddy, derivs;
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
         assert(false && "Unimplemented tex instr type\n");
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      default:
         break;
      }
   }
// TODO: all other cases: structure taken from ac_nir_to_llvm.c
   if (instr->op == nir_texop_txs && instr->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      assert(false && "Unimplemented tex instr type\n");

   if (instr->op == nir_texop_texture_samples)
      assert(false && "Unimplemented tex instr type\n");

   if (has_offset && instr->op != nir_texop_txf) {
      std::unique_ptr<Instruction> tmp_instr;
      Temp acc, pack = Temp();
      for (unsigned i = 0; i < offset.size(); i++) {
         acc = emit_extract_vector(ctx, offset, i, s1);

         tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_and_b32, Format::SOP2, 2, 1));
         tmp_instr->getOperand(0) = Operand(acc);
         tmp_instr->getOperand(1) = Operand((uint32_t) 0x3F);
         acc = {ctx->program->allocateId(), s1};
         tmp_instr->getDefinition(0) = Definition(acc);
         ctx->block->instructions.emplace_back(std::move(tmp_instr));

         if (i == 0) {
            pack = acc;
         } else {
            tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_lshl_b32, Format::SOP2, 2, 1));
            tmp_instr->getOperand(0) = Operand(pack);
            tmp_instr->getOperand(1) = Operand((uint32_t) 8 * i);
            acc = {ctx->program->allocateId(), s1};
            tmp_instr->getDefinition(0) = Definition(acc);
            ctx->block->instructions.emplace_back(std::move(tmp_instr));

            tmp_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b32, Format::SOP2, 2, 1));
            tmp_instr->getOperand(0) = Operand(pack);
            tmp_instr->getOperand(1) = Operand(acc);
            pack = {ctx->program->allocateId(), s1};
            tmp_instr->getDefinition(0) = Definition(pack);
            ctx->block->instructions.emplace_back(std::move(tmp_instr));
         }
      }
      offset = pack;
   }

   /* pack derivatives */
   if (has_ddx || has_ddy) {
      std::unique_ptr<Instruction> pack_derivs;
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_1D && ctx->options->chip_class >= GFX9) {
         pack_derivs.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 4, 1));
         pack_derivs->getOperand(0) = Operand(0);
         pack_derivs->getOperand(1) = Operand(ddx);
         pack_derivs->getOperand(2) = Operand(0);
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
      assert(false && "Unimplemented tex instr type\n");

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
       instr->op != nir_texop_lod)
      assert(false && "Unimplemented tex instr type\n");

   if (instr->op == nir_texop_samples_identical)
      assert(false && "Unimplemented tex instr type\n");

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_MS &&
       instr->op != nir_texop_txs)
      assert(false && "Unimplemented tex instr type\n");

   if (instr->op == nir_texop_tg4)
      assert(false && "Unimplemented tex instr type\n");

   if (has_offset && instr->op == nir_texop_txf)
      assert(false && "Unimplemented tex instr type\n");

   bool da = false;
   if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      aco_image_dim dim = get_sampler_dim(ctx, instr->sampler_dim, instr->is_array);

      da = dim == aco_image_cube ||
           dim == aco_image_1darray ||
           dim == aco_image_2darray ||
           dim == aco_image_2darraymsaa;
   }

   /* Build tex instruction */
   unsigned dmask = (1 << instr->dest.ssa.num_components) - 1;

   std::unique_ptr<MIMG_instruction> tex;
   if (instr->op == nir_texop_txs) {
      if (!has_lod) {
         std::unique_ptr<VOP1_instruction> mov{create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1)};
         mov->getOperand(0) = Operand((uint32_t) 0);
         lod = Temp{ctx->program->allocateId(), v1};
         mov->getDefinition(0) = Definition(lod);
         ctx->block->instructions.emplace_back(std::move(mov));
      }
      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_get_resinfo, Format::MIMG, 2, 1));
      tex->getOperand(0) = Operand(lod);
      tex->getOperand(1) = Operand(resource);
      tex->dmask = dmask;
      tex->getDefinition(0) = get_ssa_temp(ctx, &instr->dest.ssa);
      ctx->block->instructions.emplace_back(std::move(tex));
      return;
   }

   Temp arg = coords;

   if (has_derivs) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(derivs);
      vec->getOperand(1) = Operand(arg);
      RegClass rc = getRegClass(vgpr, derivs.size() + arg.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
   }

   if (has_compare) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(compare);
      vec->getOperand(1) = Operand(arg);
      RegClass rc = (RegClass) ((int) compare.regClass() + arg.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
   }

   if (has_bias) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(bias);
      vec->getOperand(1) = Operand(arg);
      RegClass rc = (RegClass) ((int) bias.regClass() + arg.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
   }

   if (has_offset) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(offset);
      vec->getOperand(1) = Operand(arg);
      RegClass rc = (RegClass) ((int) arg.regClass() + offset.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
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

   tex.reset(create_instruction<MIMG_instruction>(opcode, Format::MIMG, 3, 1));
   tex->getOperand(0) = Operand{arg};
   tex->getOperand(1) = Operand(resource);
   tex->getOperand(2) = Operand(sampler);
   tex->dmask = dmask;
   tex->da = da;
   tex->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(tex));

   if (instr->op == nir_texop_query_levels)
      assert(false && "Unimplemented tex instr type\n");
   else if (instr->op == nir_texop_txs &&
            instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
            instr->is_array)
      assert(false && "Unimplemented tex instr type\n");
   else if (ctx->options->chip_class >= GFX9 &&
            instr->op == nir_texop_txs &&
            instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
            instr->is_array)
      assert(false && "Unimplemented tex instr type\n");

}


void visit_phi(isel_context *ctx, nir_phi_instr *instr)
{
   // FIXME: are we sure that the order of phi src corresponds to our order of block predecessors?
   std::unique_ptr<Instruction> phi;
   unsigned num_src = exec_list_length(&instr->srcs);
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   aco_opcode opcode = dst.type() == vgpr ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
   phi.reset(create_instruction<Instruction>(opcode, Format::PSEUDO, num_src, 1));
   std::set<unsigned> block_idx;

   nir_foreach_phi_src(src, instr)
      block_idx.insert(src->pred->index);

   nir_foreach_phi_src(src, instr)
      phi->getOperand(std::distance(block_idx.begin(), block_idx.find(src->pred->index))) = Operand(get_ssa_temp(ctx, src->src.ssa));

   phi->getDefinition(0) = Definition(dst);
   ctx->block->instructions.emplace_back(std::move(phi));
}


void visit_undef(isel_context *ctx, nir_ssa_undef_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->def);
   std::unique_ptr<Instruction> undef;

   if (dst.size() == 1) {
      undef.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
   } else {
      undef.reset(create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, dst.size(), 1));
   }
   for (unsigned i = 0; i < dst.size(); i++)
      undef->getOperand(i) = Operand((unsigned) 0);
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
      std::unique_ptr<Instruction>(create_instruction<Instruction>(aco_opcode::p_logical_start,
                                                                   Format::PSEUDO, 0, 0)));
}

static void append_logical_end(Block *b)
{
   b->instructions.push_back(
      std::unique_ptr<Instruction>(create_instruction<Instruction>(aco_opcode::p_logical_end,
                                                                   Format::PSEUDO, 0, 0)));
}

void visit_jump(isel_context *ctx, nir_jump_instr *instr)
{
   Block *logical_target, *linear_target;
   std::unique_ptr<Instruction> aco_instr;
   std::unique_ptr<Pseudo_branch_instruction> branch;

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
         aco_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 1));
         aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
         aco_instr->getOperand(1) = Operand(exec, s2);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         aco_instr->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         ctx->block->instructions.emplace_back(std::move(aco_instr));

         /* set exec zero */
         std::unique_ptr<Instruction> restore_exec;
         restore_exec.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
         restore_exec->getOperand(0) = Operand(0);
         restore_exec->getDefinition(0) = Definition(exec, s2);
         ctx->block->instructions.emplace_back(std::move(restore_exec));

      } else if (ctx->cf_info.parent_loop.has_divergent_continue) {

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
         ctx->block->instructions.emplace_back(std::move(branch));
         add_linear_edge(ctx->block, ctx->cf_info.parent_loop.entry);

         /* restore the exec mask and branch out of the loop */
         linear_target = ctx->cf_info.parent_loop.exit;
         aco_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1));
         aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.orig_exec);
         aco_instr->getOperand(1) = Operand(temp);
         aco_instr->getDefinition(0) = Definition(exec, s2);
         ctx->block->instructions.emplace_back(std::move(aco_instr));

      } else {
         /* uniform break - directly jump out of the loop */
         linear_target = ctx->cf_info.parent_loop.exit;
         aco_instr.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1));
         aco_instr->getOperand(0) = Operand(ctx->cf_info.parent_loop.orig_exec);
         aco_instr->getOperand(1) = Operand(exec, s2);
         aco_instr->getDefinition(0) = Definition(exec, s2);
         ctx->block->instructions.emplace_back(std::move(aco_instr));
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
   append_logical_end(ctx->block);
   /* save original exec */
   std::unique_ptr<Instruction> save_exec;
   save_exec.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
   save_exec->getOperand(0) = Operand{exec, s2};
   Temp orig_exec = {ctx->program->allocateId(), s2};
   save_exec->getDefinition(0) = Definition(orig_exec);
   ctx->block->instructions.emplace_back(std::move(save_exec));

   Block* loop_entry = ctx->program->createAndInsertBlock();
   Block* loop_exit = new Block();
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
   append_logical_end(ctx->block); // FIXME the loop might end with a break?

   /* restore all 'continue' lanes */
   std::unique_ptr<Instruction> restore;
   if (ctx->cf_info.parent_loop.has_divergent_break) {
      restore.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1));
      restore->getOperand(0) = Operand{exec, s2};
      restore->getOperand(1) = Operand(ctx->cf_info.parent_loop.active_mask);
      restore->getDefinition(0) = Definition{exec, s2};
      ctx->block->instructions.emplace_back(std::move(restore));
   }

   /* jump back to loop_entry */
   std::unique_ptr<Pseudo_branch_instruction> branch{create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_nz, Format::PSEUDO_BRANCH, 1, 0)};
   branch->getOperand(0) = Operand{exec, s2};
   branch->targets[0] = loop_entry;
   branch->targets[1] = loop_exit;
   ctx->block->instructions.emplace_back(std::move(branch));
   add_edge(ctx->block, loop_entry);
   add_linear_edge(ctx->block, loop_exit);

   /* emit loop successor block */
   loop_exit->index = ctx->program->blocks.size();
   ctx->block = loop_exit;
   ctx->program->blocks.emplace_back(loop_exit);
   /* restore original exec */
   restore.reset(create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1));
   restore->getOperand(0) = Operand{exec, s2};
   restore->getOperand(1) = Operand(ctx->cf_info.parent_loop.orig_exec);
   restore->getDefinition(0) = Definition{exec, s2};
   ctx->block->instructions.emplace_back(std::move(restore));

   append_logical_start(ctx->block);
}

static void visit_if(isel_context *ctx, nir_if *if_stmt)
{
   Temp cond32 = get_ssa_temp(ctx, if_stmt->condition.ssa);
   std::unique_ptr<Pseudo_branch_instruction> branch;

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
      Block* BB_else = new Block();
      Block* BB_endif = new Block();
      Block* parent_if_merge_block = ctx->cf_info.parent_if.merge_block;
      ctx->cf_info.parent_if.merge_block = BB_endif;
      Temp active_mask_if, active_mask_then, active_mask_else;
      bool break_then = false, break_else = false;

      /** emit conditional statement */
      append_logical_end(BB_if);
      Temp cond = extract_uniform_cond32(ctx, cond32);

      /* emit branch */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(cond);
      branch->targets[0] = BB_else;
      branch->targets[1] = BB_then;
      BB_if->instructions.emplace_back(std::move(branch));
      add_edge(BB_if, BB_then);
      add_edge(BB_if, BB_else);

      /* remember active lanes mask just in case */
      active_mask_if = ctx->cf_info.parent_loop.active_mask;

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
      } else if (ctx->cf_info.has_break && ctx->cf_info.parent_if.is_divergent) {
         break_then = true;
         active_mask_then = ctx->cf_info.parent_loop.active_mask;
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /** emit else block */
      BB_else->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else);
      append_logical_start(BB_else);
      ctx->block = BB_else;
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else = ctx->block;

      if (!ctx->cf_info.has_break && !ctx->cf_info.has_continue) {
         append_logical_end(BB_else);
         /* branch from then block to endif block */
         branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
         branch->targets[0] = BB_endif;
         BB_else->instructions.emplace_back(std::move(branch));
         add_edge(BB_else, BB_endif);
      } else if (ctx->cf_info.has_break && ctx->cf_info.parent_if.is_divergent) {
         break_else = true;
         active_mask_else = ctx->cf_info.parent_loop.active_mask;
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /** emit endif merge block */
      BB_endif->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_endif);

      /* emit linear phi for active mask */
      if (break_then || break_else) {
         std::unique_ptr<Instruction> phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1)};
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
      Block* BB_then_linear = new Block();
      Block* BB_between = new Block();
      Block* BB_else_logical = new Block();
      Block* BB_else_linear = new Block();
      Block* BB_endif = new Block();

      /** emit conditional statement */
      append_logical_end(BB_if);
      Temp cond = extract_divergent_cond32(ctx, cond32);

      /* create the exec mask for then branch */
      std::unique_ptr<SOP1_instruction> set_exec{create_instruction<SOP1_instruction>(aco_opcode::s_and_saveexec_b64, Format::SOP1, 1, 1)};
      set_exec->getOperand(0) = Operand(cond);
      Temp orig_exec = {ctx->program->allocateId(), s2};
      set_exec->getDefinition(0) = Definition(orig_exec);
      BB_if->instructions.push_back(std::move(set_exec));

      /* create the exec mask for else branch */
      std::unique_ptr<SOP2_instruction> nand{create_instruction<SOP2_instruction>(aco_opcode::s_andn2_b64, Format::SOP2, 2, 1)};
      nand->getOperand(0) = Operand(orig_exec);
      nand->getOperand(1) = Operand(cond);
      Temp else_mask = {ctx->program->allocateId(), s2};
      nand->getDefinition(0) = Definition(else_mask);
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
      append_logical_start(BB_then_linear);
      /* nothing in here */
      append_logical_end(BB_then_linear);

      /* branch from linear then block to between block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_between;
      BB_then_linear->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_then_linear, BB_between);


      /** emit in-between merge block */
      BB_between->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_between);

      if (ctx->cf_info.has_break) {
         /* emit linear phi for active & inactive mask */
         std::unique_ptr<Instruction> phi;
         phi.reset(create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1));
         phi->getOperand(1) = Operand(active_mask);
         phi->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         phi->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         BB_between->instructions.push_back(std::move(phi));
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /* invert exec mask */
      std::unique_ptr<Instruction> mov;
      mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
      mov->getOperand(0) = Operand(exec, s2);
      Temp then_mask = {ctx->program->allocateId(), s2};
      mov->getDefinition(0) = Definition(then_mask);
      BB_between->instructions.push_back(std::move(mov));
      mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1));
      mov->getOperand(0) = Operand(else_mask);
      mov->getDefinition(0) = Definition(exec, s2);
      BB_between->instructions.push_back(std::move(mov));
      append_logical_start(BB_between);
      append_logical_end(BB_between);

      /* branch to linear else block (skip else) */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_cbranch_z, Format::PSEUDO_BRANCH, 1, 0));
      branch->getOperand(0) = Operand(PhysReg{126}, s2);
      branch->targets[0] = BB_else_linear;
      branch->targets[1] = BB_else_logical;
      BB_between->instructions.push_back(std::move(branch));
      add_linear_edge(BB_between, BB_else_linear);
      add_linear_edge(BB_between, BB_else_logical);

      /** emit logical else block */
      BB_else_logical->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_else_logical);
      ctx->cf_info.parent_if.merge_block = BB_endif;
      ctx->block = BB_else_logical;
      append_logical_start(BB_else_logical);
      visit_cf_list(ctx, &if_stmt->else_list);
      BB_else_logical = ctx->block;

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
      append_logical_start(BB_else_linear);
      /* nothing in here */
      append_logical_end(BB_else_linear);

      /* branch from linear else block to endif block */
      branch.reset(create_instruction<Pseudo_branch_instruction>(aco_opcode::p_branch, Format::PSEUDO_BRANCH, 0, 0));
      branch->targets[0] = BB_endif;
      BB_else_linear->instructions.emplace_back(std::move(branch));
      add_linear_edge(BB_else_linear, BB_endif);

      /** emit endif merge block */
      BB_endif->index = ctx->program->blocks.size();
      ctx->program->blocks.emplace_back(BB_endif);

      if (ctx->cf_info.has_break) {
         /* emit linear phi for active & inactive mask */
         std::unique_ptr<Instruction> phi;
         phi.reset(create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, 2, 1));
         phi->getOperand(0) = Operand(ctx->cf_info.parent_loop.active_mask);
         phi->getOperand(1) = Operand(active_mask);
         ctx->cf_info.parent_loop.active_mask = {ctx->program->allocateId(), s2};
         phi->getDefinition(0) = Definition(ctx->cf_info.parent_loop.active_mask);
         BB_endif->instructions.push_back(std::move(phi));
      }
      ctx->cf_info.has_break = false;
      ctx->cf_info.has_continue = false;

      /* restore original exec mask */
      std::unique_ptr<SOP2_instruction> restore{create_instruction<SOP2_instruction>(aco_opcode::s_or_b64, Format::SOP2, 2, 1)};
      restore->getOperand(0) = Operand(exec, s2);
      restore->getOperand(1) = Operand(then_mask);
      restore->getDefinition(0) = Definition(exec, s2);
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


void init_context(isel_context *ctx, nir_function_impl *impl)
{
   std::unique_ptr<RegClass[]> reg_class{new RegClass[impl->ssa_alloc]};
   memset(&ctx->fs_vgpr_args, false, sizeof(ctx->fs_vgpr_args));

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch(instr->type) {
         case nir_instr_type_alu: {
            nir_alu_instr *alu_instr = nir_instr_as_alu(instr);
            unsigned size =  alu_instr->dest.dest.ssa.num_components;
            if (alu_instr->dest.dest.ssa.bit_size == 64)
               size *= 2;
            RegType type = sgpr;
            switch(alu_instr->op) {
               case nir_op_fmul:
               case nir_op_fadd:
               case nir_op_fsub:
               case nir_op_fmax:
               case nir_op_fmin:
               case nir_op_fneg:
               case nir_op_fabs:
               case nir_op_fsign:
               case nir_op_frcp:
               case nir_op_frsq:
               case nir_op_fsqrt:
               case nir_op_fexp2:
               case nir_op_flog2:
               case nir_op_ffract:
               case nir_op_ffloor:
               case nir_op_fsin:
               case nir_op_fcos:
               case nir_op_u2f32:
               case nir_op_i2f32:
               case nir_op_b2f:
                  type = vgpr;
                  break;
               case nir_op_flt:
               case nir_op_fge:
               case nir_op_feq:
               case nir_op_fne:
               case nir_op_ilt:
               case nir_op_ige:
               case nir_op_ieq:
               case nir_op_ine:
               case nir_op_ult:
               case nir_op_uge:
               case nir_op_bcsel:
                  type = ctx->divergent_vals[alu_instr->dest.dest.ssa.index] ? vgpr : sgpr;
                  break;
               default:
                  for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                     if (typeOf(reg_class[alu_instr->src[i].src.ssa->index]) == vgpr)
                        type = vgpr;
                  }
                  break;
            }
            reg_class[alu_instr->dest.dest.ssa.index] = getRegClass(type, size);
            break;
         }
         case nir_instr_type_load_const: {
            unsigned size = nir_instr_as_load_const(instr)->def.num_components;
            if (nir_instr_as_load_const(instr)->def.bit_size == 64)
               size *= 2;
            reg_class[nir_instr_as_load_const(instr)->def.index] = getRegClass(sgpr, size);
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
            unsigned size =  intrinsic->dest.ssa.num_components;
            if (intrinsic->dest.ssa.bit_size == 64)
               size *= 2;
            RegType type = sgpr;
            switch(intrinsic->intrinsic) {
               case nir_intrinsic_load_front_face:
                  type = sgpr;
                  size = 2;
                  break;
               case nir_intrinsic_load_input:
               case nir_intrinsic_load_vertex_id:
               case nir_intrinsic_load_vertex_id_zero_base:
               case nir_intrinsic_load_barycentric_pixel:
               case nir_intrinsic_load_interpolated_input:
                  type = vgpr;
                  break;
               case nir_intrinsic_vulkan_resource_index:
                  type = sgpr;
                  size = 2;
                  break;
               case nir_intrinsic_load_ubo:
                  type = ctx->divergent_vals[intrinsic->dest.ssa.index] ? vgpr : sgpr;
                  break;
               default:
                  for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs; i++) {
                     if (typeOf(reg_class[intrinsic->src[i].ssa->index]) == vgpr)
                        type = vgpr;
                  }
                  break;
            }
            if (nir_intrinsic_infos[intrinsic->intrinsic].has_dest)
               reg_class[intrinsic->dest.ssa.index] = getRegClass(type, size);

            switch(intrinsic->intrinsic) {
               case nir_intrinsic_load_barycentric_pixel:
                  ctx->fs_vgpr_args[fs_input::persp_center_p1] = true;
                  break;
               case nir_intrinsic_load_front_face:
                  ctx->fs_vgpr_args[fs_input::front_face] = true;
                  break;
               case nir_intrinsic_load_interpolated_input:
                  if (nir_intrinsic_base(intrinsic) == VARYING_SLOT_POS) {
                     for (unsigned i = 0; i < intrinsic->dest.ssa.num_components; i++)
                        ctx->fs_vgpr_args[fs_input::frag_pos_0 + i] = true;
                  }
                  break;
               default:
                  break;
            }

            break;
         }
         case nir_instr_type_tex: {
            unsigned size = nir_instr_as_tex(instr)->dest.ssa.num_components;
            if (nir_instr_as_tex(instr)->dest.ssa.bit_size == 64)
               size *= 2;
            reg_class[nir_instr_as_tex(instr)->dest.ssa.index] = getRegClass(vgpr, size);
            break;
         }
         case nir_instr_type_parallel_copy: {
            nir_foreach_parallel_copy_entry(entry, nir_instr_as_parallel_copy(instr)) {
               reg_class[entry->dest.ssa.index] = reg_class[entry->src.ssa->index];
            }
            break;
         }
         case nir_instr_type_ssa_undef: {
            unsigned size = nir_instr_as_ssa_undef(instr)->def.num_components;
            if (nir_instr_as_ssa_undef(instr)->def.bit_size == 64)
               size *= 2;
            reg_class[nir_instr_as_ssa_undef(instr)->def.index] = getRegClass(sgpr, size);
            break;
         }
         case nir_instr_type_phi: {
            nir_phi_instr* phi = nir_instr_as_phi(instr);
            RegType type;
            if (ctx->divergent_vals[phi->dest.ssa.index])
               type = vgpr;
            else
               type = sgpr;
            unsigned size = phi->dest.ssa.num_components;
            if (phi->dest.ssa.bit_size == 64)
               size *= 2;
            reg_class[phi->dest.ssa.index] = getRegClass(type, size);
            break;
         }
         default:
            break;
         }
      }
   }
   ctx->reg_class.reset(reg_class.release());
}

struct user_sgpr_info {
   uint8_t num_sgpr;
   uint8_t user_sgpr_idx;
   bool need_ring_offsets;
   bool indirect_all_descriptor_sets;
};

static void allocate_user_sgprs(isel_context *ctx,
                                /* TODO bool has_previous_stage, gl_shader_stage previous_stage, */
                                bool needs_view_index, user_sgpr_info& user_sgpr_info)
{
   memset(&user_sgpr_info, 0, sizeof(struct user_sgpr_info));
   uint32_t user_sgpr_count = 0;

   /* until we sort out scratch/global buffers always assign ring offsets for gs/vs/es */
   if (ctx->stage == MESA_SHADER_GEOMETRY ||
       ctx->stage == MESA_SHADER_VERTEX ||
       ctx->stage == MESA_SHADER_TESS_CTRL ||
       ctx->stage == MESA_SHADER_TESS_EVAL
       /*|| ctx->is_gs_copy_shader */)
      user_sgpr_info.need_ring_offsets = true;

   if (ctx->stage == MESA_SHADER_FRAGMENT &&
       ctx->program->info->info.ps.needs_sample_positions)
      user_sgpr_info.need_ring_offsets = true;

   /* 2 user sgprs will nearly always be allocated for scratch/rings */
   if (ctx->options->supports_spill || user_sgpr_info.need_ring_offsets) {
      user_sgpr_count += 2;
   }

   switch (ctx->stage) {
   case MESA_SHADER_VERTEX:
   /* if (!ctx->is_gs_copy_shader) */ {
         if (ctx->program->info->info.vs.has_vertex_buffers)
            user_sgpr_count++;
         user_sgpr_count += ctx->program->info->info.vs.needs_draw_id ? 3 : 2;
      }
      break;
   case MESA_SHADER_FRAGMENT:
      user_sgpr_count += ctx->program->info->info.ps.needs_sample_positions;
      break;
   default:
      unreachable("Shader stage not implemented");
   }

   if (needs_view_index)
      user_sgpr_count++;

   if (ctx->program->info->info.loads_push_constants)
      user_sgpr_count += 1; /* we use 32bit pointers */

   uint32_t available_sgprs = ctx->options->chip_class >= GFX9 ? 32 : 16;
   uint32_t num_desc_set = util_bitcount(ctx->program->info->info.desc_set_used_mask);
   user_sgpr_info.num_sgpr = user_sgpr_count + num_desc_set;

   if (available_sgprs < user_sgpr_info.num_sgpr)
      user_sgpr_info.indirect_all_descriptor_sets = true;
}

#define MAX_ARGS 23
struct arg_info {
   RegClass types[MAX_ARGS];
   Temp *assign[MAX_ARGS];
   PhysReg reg[MAX_ARGS];
   unsigned array_params_mask;
   uint8_t count;
   uint8_t sgpr_count;
   uint8_t num_sgprs_used;
   uint8_t num_vgprs_used;
};

static void
add_arg(arg_info *info, RegClass type, Temp *param_ptr, unsigned reg)
{
   assert(info->count < MAX_ARGS);

   info->assign[info->count] = param_ptr;
   info->types[info->count] = type;

   if (typeOf(type) == sgpr) {
      info->num_sgprs_used += sizeOf(type);
      info->sgpr_count++;
      info->reg[info->count] = fixed_sgpr(reg);
   } else {
      assert(typeOf(type) == vgpr);
      info->num_vgprs_used += sizeOf(type);
      info->reg[info->count] = fixed_vgpr(reg);
   }
   info->count++;
}

static inline void
add_array_arg(arg_info *info, RegClass type, Temp *param_ptr, unsigned reg)
{
   info->array_params_mask |= (1 << info->count);
   add_arg(info, type, param_ptr, reg);
}

static void
set_loc(struct radv_userdata_info *ud_info, uint8_t *sgpr_idx, uint8_t num_sgprs,
        int32_t indirect_offset)
{
   ud_info->sgpr_idx = *sgpr_idx;
   ud_info->num_sgprs = num_sgprs;
   ud_info->indirect = indirect_offset > 0;
   ud_info->indirect_offset = indirect_offset;
   *sgpr_idx += num_sgprs;
}

static void
set_loc_shader(isel_context *ctx, int idx, uint8_t *sgpr_idx,
               uint8_t num_sgprs)
{
   struct radv_userdata_info *ud_info = &ctx->program->info->user_sgprs_locs.shader_data[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, num_sgprs, 0);
}

static void
set_loc_shader_ptr(isel_context *ctx, int idx, uint8_t *sgpr_idx)
{
   bool use_32bit_pointers = idx != AC_UD_SCRATCH_RING_OFFSETS;

   set_loc_shader(ctx, idx, sgpr_idx, use_32bit_pointers ? 1 : 2);
}

static void
set_loc_desc(isel_context *ctx, int idx,  uint8_t *sgpr_idx,
             uint32_t indirect_offset)
{
   struct radv_userdata_info *ud_info =
      &ctx->program->info->user_sgprs_locs.descriptor_sets[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, 1, indirect_offset);
}

static void
declare_global_input_sgprs(isel_context *ctx,
                           /* bool has_previous_stage, gl_shader_stage previous_stage, */
                           user_sgpr_info *user_sgpr_info,
                           struct arg_info *args,
                           Temp *desc_sets)
{
   unsigned num_sets = ctx->options->layout ? ctx->options->layout->num_sets : 0;
   unsigned stage_mask = 1 << ctx->stage;

   //if (has_previous_stage)
   //   stage_mask |= 1 << previous_stage;

   /* 1 for each descriptor set */
   if (!user_sgpr_info->indirect_all_descriptor_sets) {
      for (unsigned i = 0; i < num_sets; ++i) {
         if ((ctx->program->info->info.desc_set_used_mask & (1 << i)) &&
             ctx->options->layout->set[i].layout->shader_stages & stage_mask) {
            add_array_arg(args, s1, &desc_sets[i], user_sgpr_info->user_sgpr_idx);
            set_loc_desc(ctx, i, &user_sgpr_info->user_sgpr_idx, 0);
         }
      }
   } else {
      add_array_arg(args, s1, desc_sets, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_INDIRECT_DESCRIPTOR_SETS, &user_sgpr_info->user_sgpr_idx);
      for (unsigned i = 0; i < num_sets; ++i) {
         if ((ctx->program->info->info.desc_set_used_mask & (1 << i)) &&
             ctx->options->layout->set[i].layout->shader_stages & stage_mask)
            set_loc_desc(ctx, i, &user_sgpr_info->user_sgpr_idx, i * 8);
      }
      ctx->program->info->need_indirect_descriptor_sets = true;
   }

   if (ctx->program->info->info.loads_push_constants) {
      /* 1 for push constants and dynamic descriptors */
      add_array_arg(args, s1, &ctx->push_constants, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_PUSH_CONSTANTS, &user_sgpr_info->user_sgpr_idx);
   }
}

static void
declare_vs_input_vgprs(isel_context *ctx, struct arg_info *args)
{
   unsigned vgpr_idx = 0;
   add_arg(args, v1, &ctx->vertex_id, vgpr_idx++);
/* if (!ctx->is_gs_copy_shader) */ {
      if (ctx->options->key.vs.as_ls) {
         add_arg(args, v1, &ctx->rel_auto_id, vgpr_idx++);
         add_arg(args, v1, &ctx->instance_id, vgpr_idx++);
      } else {
         add_arg(args, v1, &ctx->instance_id, vgpr_idx++);
         add_arg(args, v1, &ctx->vs_prim_id, vgpr_idx++);
      }
      add_arg(args, v1, NULL, vgpr_idx); /* unused */
   }
}

static bool needs_view_index_sgpr(isel_context *ctx)
{
   switch (ctx->stage) {
   case MESA_SHADER_VERTEX:
      if (ctx->program->info->info.needs_multiview_view_index ||
          (!ctx->options->key.vs.as_es && !ctx->options->key.vs.as_ls && ctx->options->key.has_multiview_view_index))
         return true;
      break;
   case MESA_SHADER_TESS_EVAL:
      if (ctx->program->info->info.needs_multiview_view_index || (!ctx->options->key.tes.as_es && ctx->options->key.has_multiview_view_index))
         return true;
      break;
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_TESS_CTRL:
      if (ctx->program->info->info.needs_multiview_view_index)
         return true;
      break;
   default:
      break;
   }
   return false;
}

void add_startpgm(struct isel_context *ctx)
{
   user_sgpr_info user_sgpr_info;
   bool needs_view_index = needs_view_index_sgpr(ctx);
   allocate_user_sgprs(ctx, needs_view_index, user_sgpr_info);

   assert(!user_sgpr_info.indirect_all_descriptor_sets && "Not yet implemented.");
   arg_info args = {};

   if (user_sgpr_info.need_ring_offsets && !ctx->options->supports_spill)
      add_arg(&args, s2, &ctx->ring_offsets, 0);

   if (ctx->options->supports_spill || user_sgpr_info.need_ring_offsets) {
      set_loc_shader_ptr(ctx, AC_UD_SCRATCH_RING_OFFSETS, &user_sgpr_info.user_sgpr_idx);
   }

   unsigned vgpr_idx = 0;
   switch (ctx->stage) {
   case MESA_SHADER_VERTEX: {
      declare_global_input_sgprs(ctx, &user_sgpr_info, &args, ctx->descriptor_sets);

      if (ctx->program->info->info.vs.has_vertex_buffers) {
         add_arg(&args, s1, &ctx->vertex_buffers, user_sgpr_info.user_sgpr_idx);
         set_loc_shader_ptr(ctx, AC_UD_VS_VERTEX_BUFFERS, &user_sgpr_info.user_sgpr_idx);
      }
      add_arg(&args, s1, &ctx->base_vertex, user_sgpr_info.user_sgpr_idx);
      add_arg(&args, s1, &ctx->start_instance, user_sgpr_info.user_sgpr_idx + 1);
      if (ctx->program->info->info.vs.needs_draw_id) {
         add_arg(&args, s1, &ctx->draw_id, user_sgpr_info.user_sgpr_idx + 2);
         set_loc_shader(ctx, AC_UD_VS_BASE_VERTEX_START_INSTANCE, &user_sgpr_info.user_sgpr_idx, 3);
      } else
         set_loc_shader(ctx, AC_UD_VS_BASE_VERTEX_START_INSTANCE, &user_sgpr_info.user_sgpr_idx, 2);

      if (needs_view_index) {
         add_arg(&args, s1, &ctx->view_index, user_sgpr_info.user_sgpr_idx);
         set_loc_shader(ctx, AC_UD_VIEW_INDEX, &user_sgpr_info.user_sgpr_idx, 1);
      }
      if (ctx->options->key.vs.as_es)
         add_arg(&args, s1, &ctx->es2gs_offset, user_sgpr_info.user_sgpr_idx);

      declare_vs_input_vgprs(ctx, &args);
      break;
   }
   case MESA_SHADER_FRAGMENT: {
      declare_global_input_sgprs(ctx, &user_sgpr_info, &args, ctx->descriptor_sets);

      assert(user_sgpr_info.user_sgpr_idx == user_sgpr_info.num_sgpr);
      add_arg(&args, s1, &ctx->prim_mask, user_sgpr_info.user_sgpr_idx);

      ctx->program->config->spi_ps_input_addr = 0;
      ctx->program->config->spi_ps_input_ena = 0;
      if (ctx->fs_vgpr_args[fs_input::persp_sample]) {
         add_arg(&args, v2, &ctx->fs_inputs[fs_input::persp_sample], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_SAMPLE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_SAMPLE_ENA(1);
         vgpr_idx += 2;
      }
      if (ctx->fs_vgpr_args[fs_input::persp_center_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_center_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_center_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_CENTER_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_CENTER_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::persp_centroid]) {
         add_arg(&args, v2, &ctx->fs_inputs[fs_input::persp_centroid], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_CENTROID_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_CENTROID_ENA(1);
         vgpr_idx += 2;
      }
      if (ctx->fs_vgpr_args[fs_input::persp_pull_model]) {
         add_arg(&args, v3, &ctx->fs_inputs[fs_input::persp_pull_model], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_PULL_MODEL_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_PULL_MODEL_ENA(1);
         vgpr_idx += 3;
      }
      if (ctx->fs_vgpr_args[fs_input::linear_sample]) {
         add_arg(&args, v2, &ctx->fs_inputs[fs_input::linear_sample], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_SAMPLE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_SAMPLE_ENA(1);
         vgpr_idx += 2;
      }
      if (ctx->fs_vgpr_args[fs_input::linear_center]) {
         add_arg(&args, v2, &ctx->fs_inputs[fs_input::linear_center], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_CENTER_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_CENTER_ENA(1);
         vgpr_idx += 2;
      }
      if (ctx->fs_vgpr_args[fs_input::linear_centroid]) {
         add_arg(&args, v2, &ctx->fs_inputs[fs_input::linear_centroid], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_CENTROID_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_CENTROID_ENA(1);
         vgpr_idx += 2;
      }
      if (ctx->fs_vgpr_args[fs_input::line_stipple]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::line_stipple], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINE_STIPPLE_TEX_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINE_STIPPLE_TEX_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::frag_pos_0]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::frag_pos_0], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_POS_X_FLOAT_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_POS_X_FLOAT_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::frag_pos_1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::frag_pos_1], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_POS_Y_FLOAT_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_POS_Y_FLOAT_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::frag_pos_2]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::frag_pos_2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_POS_Z_FLOAT_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_POS_Z_FLOAT_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::frag_pos_3]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::frag_pos_3], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_POS_W_FLOAT_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_POS_W_FLOAT_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::front_face]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::front_face], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_FRONT_FACE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_FRONT_FACE_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::ancillary]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::ancillary], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_ANCILLARY_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_ANCILLARY_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::sample_coverage]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::sample_coverage], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_SAMPLE_COVERAGE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_SAMPLE_COVERAGE_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::fixed_pt]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::fixed_pt], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_POS_FIXED_PT_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_POS_FIXED_PT_ENA(1);
      }
      break;
   }
   default:
      unreachable("Shader stage not implemented");
   }

   ctx->program->info->num_input_vgprs = 0;
   ctx->program->info->num_input_sgprs = ctx->options->supports_spill ? 2 : 0;
   ctx->program->info->num_input_sgprs += args.num_sgprs_used;
   ctx->program->info->num_user_sgprs = user_sgpr_info.num_sgpr;

   if (ctx->stage != MESA_SHADER_FRAGMENT)
      ctx->program->info->num_input_vgprs = args.num_vgprs_used;

   std::unique_ptr<Instruction> startpgm{create_instruction<Instruction>(aco_opcode::p_startpgm, Format::PSEUDO, 0, args.count)};
   for (unsigned i = 0; i < args.count; i++) {
      if (args.assign[i]) {
         *args.assign[i] = Temp{ctx->program->allocateId(), args.types[i]};
         startpgm->getDefinition(i) = *args.assign[i];
         startpgm->getDefinition(i).setFixed(args.reg[i]);
      }
   }

   ctx->block->instructions.push_back(std::move(startpgm));

   std::unique_ptr<Instruction> wqm{create_instruction<Instruction>(aco_opcode::s_wqm_b64, Format::SOP1, 1, 1)};
   wqm->getOperand(0) = Operand(PhysReg{126}, s2);
   wqm->getDefinition(0) = Definition(PhysReg{126}, s2);
   ctx->block->instructions.push_back(std::move(wqm));

   append_logical_start(ctx->block);
}

}

int
type_size(const struct glsl_type *type)
{
   return glsl_count_attribute_slots(type, false);
}

std::unique_ptr<Program> select_program(struct nir_shader *nir,
                                        ac_shader_config* config,
                                        struct radv_shader_variant_info *info,
                                        struct radv_nir_compiler_options *options)
{
   std::unique_ptr<Program> program{new Program};
   program->config = config;
   program->info = info;

   for (unsigned i = 0; i < RADV_UD_MAX_SETS; ++i)
      program->info->user_sgprs_locs.descriptor_sets[i].sgpr_idx = -1;
   for (unsigned i = 0; i < AC_UD_MAX_UD; ++i)
      program->info->user_sgprs_locs.shader_data[i].sgpr_idx = -1;

   isel_context ctx = {};
   ctx.program = program.get();
   ctx.options = options;
   ctx.stage = nir->info.stage;
   nir_lower_io(nir, (nir_variable_mode)(nir_var_shader_in | nir_var_shader_out), type_size, (nir_lower_io_options)0);
   nir_opt_cse(nir);
   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   nir_index_ssa_defs(func->impl);
   ctx.divergent_vals = nir_divergence_analysis(nir);
   init_context(&ctx, func->impl);

   nir_print_shader(nir, stderr);

   ctx.program->blocks.push_back(std::unique_ptr<Block>{new Block});
   ctx.block = ctx.program->blocks.back().get();
   ctx.block->index = 0;

   if (ctx.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_variable(variable, &nir->inputs)
      {
         int idx = variable->data.location - VARYING_SLOT_VAR0;
         ctx.input_mask |= 1ull << idx;
      }
      program->info->fs.num_interp = util_bitcount(ctx.input_mask);
      program->info->fs.input_mask = ctx.input_mask;
   }

   add_startpgm(&ctx);

   visit_cf_list(&ctx, &func->impl->body);

   append_logical_end(ctx.block);
   ctx.block->instructions.push_back(std::unique_ptr<SOPP_instruction>(create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)));

   return program;
}
}
