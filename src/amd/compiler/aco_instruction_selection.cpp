#include <algorithm>
#include <unordered_map>

#include "aco_ir.h"
#include "aco_builder.h"
#include "aco_interface.h"
#include "nir/nir.h"
#include "common/sid.h"
#include "vulkan/radv_shader.h"

#include "gallium/auxiliary/util/u_math.h"
namespace aco {
namespace {
struct isel_context {
   struct radv_nir_compiler_options *options;
   Program *program;
   Block *block;
   bool *divergent_vals;
   std::unique_ptr<RegClass[]> reg_class;
   std::unordered_map<unsigned, unsigned> allocated;

   Temp barycentric_coords;
   Temp prim_mask;
   Temp descriptor_sets[RADV_UD_MAX_SETS];
   Temp push_constants;
   Temp ring_offsets;
   Temp sample_pos_offset;
   Temp persp_sample;
   Temp persp_center;
   Temp persp_centroid;
   Temp linear_sample;
   Temp linear_center;
   Temp linear_centroid;
   Temp frag_pos[4];
   Temp front_face;
   Temp ancillary;
   Temp sample_coverage;

   uint32_t input_mask;
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

Temp get_alu_src(struct isel_context *ctx, nir_alu_src src)
{
   if (src.src.ssa->num_components == 1 && src.swizzle[0] == 0)
      return get_ssa_temp(ctx, src.src.ssa);

   Temp tmp{ctx->program->allocateId(), v1};
   std::unique_ptr<Instruction> extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   extract->getOperand(0) = Operand(get_ssa_temp(ctx, src.src.ssa));
   extract->getOperand(1) = Operand((uint32_t) src.swizzle[0]);
   extract->getDefinition(0) = Definition(tmp);
   ctx->block->instructions.emplace_back(std::move(extract));
   return tmp;
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
         vopc.reset(create_instruction<VOPC_instruction>(op, Format::VOPC, 2, 1));
      } else {
         // TODO: Handle both cases SGPR.
         abort();
      }
   } else {
      vopc.reset(create_instruction<VOPC_instruction>(op, Format::VOPC, 2, 1));
   }
   vopc->getOperand(0) = Operand(src0);
   vopc->getOperand(1) = Operand(src1);
   vopc->getDefinition(0) = Definition(dst);
   vopc->getDefinition(0).setHint(vcc);
   ctx->block->instructions.emplace_back(std::move(vopc));
}

void emit_bcsel(isel_context *ctx, nir_alu_instr *instr, Temp dst)
{
   Temp cond = get_alu_src(ctx, instr->src[0]);
   Temp then = get_alu_src(ctx, instr->src[1]);
   Temp els = get_alu_src(ctx, instr->src[2]);
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
   } else {
      fprintf(stderr, "Unimplemented uniform bcsel: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
   }
}

void visit_alu_instr(isel_context *ctx, nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa)
      abort();
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
      if (dst.size() == 1) {
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_sub_f32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0);
         vop2->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
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
      if (dst.size() == 1) {
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_and_b32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0x7FFFFFFF);
         vop2->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
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
      emit_vop1_instruction(ctx, instr, aco_opcode::v_cvt_i32_f32, dst);
      break;
   }
   case nir_op_flt: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lt_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fge: {
      if (instr->src[0].src.ssa->bit_size == 32) {
         emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_ge_f32, dst);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_ilt: {
      if (dst.regClass() == s2) {
         emit_vopc_instruction(ctx, instr, aco_opcode::v_cmp_lt_i32, dst);
      } else {
         fprintf(stderr, "Unimplemented: scalar cmp instr: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
   }
   default:
      fprintf(stderr, "Unknown NIR instr type: ");
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
   Operand values[4];
   Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
   for (unsigned i = 0; i < 4; ++i) {
      Temp tmp{ctx->program->allocateId(), v1};
      std::unique_ptr<Instruction> extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));

      extract->getOperand(0) = Operand(src);
      extract->getOperand(1) = Operand(i);
      extract->getDefinition(0) = Definition(tmp);

      ctx->block->instructions.emplace_back(std::move(extract));
      values[i] = Operand(tmp);
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

   default:
   case V_028714_SPI_SHADER_32_ABGR:
      break;
   }

   if ((bool)compr_op)
   {
      for (int i = 0; i < 2; i++)
      {
         std::unique_ptr<VOP3A_instruction> compr{create_instruction<VOP3A_instruction>(compr_op, Format::VOP3A, 2, 1)};
         Temp tmp{ctx->program->allocateId(), v1};
         compr->getOperand(0) = values[i*2];
         compr->getOperand(1) = values[i*2+1];
         compr->getDefinition(0) = Definition(tmp);
         values[i] = Operand(tmp);
         ctx->block->instructions.emplace_back(std::move(compr));
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
   Temp coord1{ctx->program->allocateId(), v1};
   Temp coord2{ctx->program->allocateId(), v1};

   std::unique_ptr<Instruction> coord1_extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   coord1_extract->getOperand(0) = Operand{src};
   coord1_extract->getOperand(1) = Operand((uint32_t)0);
   coord1_extract->getDefinition(0) = Definition{coord1};

   std::unique_ptr<Instruction> coord2_extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   coord2_extract->getOperand(0) = Operand{src};
   coord2_extract->getOperand(1) = Operand((uint32_t)1);
   coord2_extract->getDefinition(0) = Definition{coord2};

   ctx->block->instructions.emplace_back(std::move(coord1_extract));
   ctx->block->instructions.emplace_back(std::move(coord2_extract));

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
   unsigned base = nir_intrinsic_base(instr) / 4 - VARYING_SLOT_VAR0;
   unsigned idx = util_bitcount(ctx->input_mask & ((1u << base) - 1));
   unsigned component = nir_intrinsic_component(instr);

   std::unique_ptr<Interp_instruction> mov{create_instruction<Interp_instruction>(aco_opcode::v_interp_mov_f32, Format::VINTRP, 2, 1)};
   mov->getOperand(0) = Operand();
   mov->getOperand(0).setFixed(PhysReg{2});
   mov->getOperand(1) = Operand(ctx->prim_mask);
   mov->getOperand(1).setFixed(m0);
   mov->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   mov->attribute = idx;
   mov->component = component;
   ctx->block->instructions.emplace_back(std::move(mov));
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
		//TODO desc_ptr = ctx->abi.push_constants;
      offset = pipeline_layout->push_constant_size + 16 * idx;
      stride = 16;
   } else
      stride = layout->binding[binding].size;

   if (stride != 1) {
      std::unique_ptr<Instruction> tmp;
      tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1));
      tmp->getOperand(0) = Operand(stride);
      tmp->getOperand(1) = Operand(index);
      index = {ctx->program->allocateId(), index.regClass()};
      tmp->getDefinition(0) = Definition(index);
      ctx->block->instructions.emplace_back(std::move(tmp));
   }
   if (offset) {
      std::unique_ptr<Instruction> tmp;
      tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
      tmp->getOperand(0) = Operand(offset);
      tmp->getOperand(1) = Operand(index);
      index = {ctx->program->allocateId(), index.regClass()};
      tmp->getDefinition(0) = Definition(index);
      ctx->block->instructions.emplace_back(std::move(tmp));
   }

   std::unique_ptr<Instruction> tmp;
   tmp.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
   tmp->getOperand(0) = Operand(index);
   tmp->getOperand(1) = Operand(desc_ptr);
   tmp->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   ctx->block->instructions.emplace_back(std::move(tmp));

}

void visit_load_ubo(isel_context *ctx, nir_intrinsic_instr *instr)
{
   Temp dst = get_ssa_temp(ctx, &instr->dest.ssa);
   Temp rsrc = get_ssa_temp(ctx, instr->src[0].ssa);
   Temp offset = get_ssa_temp(ctx, instr->src[1].ssa);

   if (dst.type() == sgpr) {
      if (rsrc.size() == 1) {
         std::unique_ptr<Instruction> tmp{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
         tmp->getOperand(0) = Operand(rsrc);
         tmp->getOperand(1) = Operand((unsigned)0);
         rsrc = {ctx->program->allocateId(), s2};
         tmp->getDefinition(0) = Definition(rsrc);
         ctx->block->instructions.emplace_back(std::move(tmp));
      }
      std::unique_ptr<Instruction> load;
      load.reset(create_instruction<SMEM_instruction>(aco_opcode::s_load_dwordx4, Format::SMEM, 2, 1));
      load->getOperand(0) = Operand(rsrc);
      load->getOperand(1) = Operand((uint32_t) 0);
      rsrc = {ctx->program->allocateId(), s4};
      load->getDefinition(0) = Definition(rsrc);
      ctx->block->instructions.emplace_back(std::move(load));

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
      load->getOperand(1) = Operand(offset);
      load->getDefinition(0) = Definition(dst);
      ctx->block->instructions.emplace_back(std::move(load));

   } else {
      fprintf(stderr, "Unsupported: ubo vector load: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      abort();
   }

}

void visit_intrinsic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   switch(instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
      ctx->allocated[instr->dest.ssa.index] = ctx->barycentric_coords.id();
      break;
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
   case nir_intrinsic_vulkan_resource_index:
      fprintf(stderr, "Untested implementation: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      visit_load_resource(ctx, instr);
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

Temp get_sampler_desc(isel_context *ctx, const nir_deref_var *deref,
                      enum aco_descriptor_type desc_type,
                      const nir_tex_instr *tex_instr, bool image, bool write)
{
   Temp index;
   bool index_set = false;
   unsigned constant_index = 0;
   unsigned descriptor_set;
   unsigned base_index;

   if (!deref) {
      assert(tex_instr && !image);
      descriptor_set = 0;
      base_index = tex_instr->sampler_index;
   } else {
      const nir_deref *tail = &deref->deref;
      while (tail->child) {
         const nir_deref_array *child = nir_deref_as_array(tail->child);
         unsigned array_size = glsl_get_aoa_size(tail->child->type);

         if (!array_size)
            array_size = 1;

         assert(child->deref_array_type != nir_deref_array_type_wildcard);

         if (child->deref_array_type == nir_deref_array_type_indirect) {
            /* check if index is in sgpr */
            Temp indirect_tmp = get_ssa_temp(ctx, child->indirect.ssa);
            if (indirect_tmp.type() == vgpr) {
               std::unique_ptr<VOP1_instruction> readlane{create_instruction<VOP1_instruction>(aco_opcode::v_readfirstlane_b32, Format::VOP1, 1, 1)};
               readlane->getOperand(0) = Operand(indirect_tmp);
               indirect_tmp = {ctx->program->allocateId(), s1};
               readlane->getDefinition(0) = Definition(indirect_tmp);
               ctx->block->instructions.emplace_back(std::move(readlane));
            }
            if (array_size != 1) {
               std::unique_ptr<Instruction> indirect;
               indirect.reset(create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1));
               indirect_tmp = {ctx->program->allocateId(), s1};
               indirect->getDefinition(0) = Definition(indirect_tmp);
               indirect->getOperand(0) = Operand(array_size);
               indirect->getOperand(1) = Operand(indirect_tmp);
               ctx->block->instructions.emplace_back(std::move(indirect));
            }
            if (!index_set) {
               index = indirect_tmp;
               index_set = true;
            } else {
               std::unique_ptr<Instruction> add;
               add.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
               add->getDefinition(0) = Definition{ctx->program->allocateId(), s1};
               add->getOperand(0) = Operand(index);
               add->getOperand(1) = Operand(indirect_tmp);

               ctx->block->instructions.emplace_back(std::move(add));
               index = add->getDefinition(0).getTemp();
            }
         }

         constant_index += child->base_offset * array_size;
         tail = &child->deref;
      }
      descriptor_set = deref->var->data.descriptor_set;

      if (deref->var->data.bindless) {
         base_index = deref->var->data.driver_location;
      } else {
         base_index = deref->var->data.binding;
      }
   }

   Temp list = ctx->descriptor_sets[descriptor_set];
   if (list.size() == 1) {
      std::unique_ptr<Instruction> tmp{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      tmp->getOperand(0) = Operand(list);
      tmp->getOperand(1) = Operand((unsigned)ctx->options->address32_hi);
      list = {ctx->program->allocateId(), s2};
      tmp->getDefinition(0) = Definition(list);
      ctx->block->instructions.emplace_back(std::move(tmp));
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
/*
      const uint32_t *samplers = radv_immutable_samplers(layout, binding);

      // TODO!
		LLVMValueRef constants[] = {
			LLVMConstInt(ctx->ac.i32, samplers[constant_index * 4 + 0], 0),
			LLVMConstInt(ctx->ac.i32, samplers[constant_index * 4 + 1], 0),
			LLVMConstInt(ctx->ac.i32, samplers[constant_index * 4 + 2], 0),
			LLVMConstInt(ctx->ac.i32, samplers[constant_index * 4 + 3], 0),
		};
		return ac_build_gather_values(&ctx->ac, constants, 4);*/
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
   return t;
}

void tex_fetch_ptrs(isel_context *ctx, nir_tex_instr *instr,
                           Temp *res_ptr, Temp *samp_ptr, Temp *fmask_ptr)
{
   if (instr->sampler_dim  == GLSL_SAMPLER_DIM_BUF)
      *res_ptr = get_sampler_desc(ctx, instr->texture, ACO_DESC_BUFFER, instr, false, false);
   else
      *res_ptr = get_sampler_desc(ctx, instr->texture, ACO_DESC_IMAGE, instr, false, false);
   if (samp_ptr) {
      if (instr->sampler)
         *samp_ptr = get_sampler_desc(ctx, instr->sampler, ACO_DESC_SAMPLER, instr, false, false);
      else
         *samp_ptr = get_sampler_desc(ctx, instr->texture, ACO_DESC_SAMPLER, instr, false, false);
      if (instr->sampler_dim < GLSL_SAMPLER_DIM_RECT && ctx->options->chip_class < VI) {
         // TODO: build samp_ptr = and(samp_ptr, res_ptr)
      }
   }
   if (fmask_ptr && !instr->sampler && (instr->op == nir_texop_txf_ms ||
                                        instr->op == nir_texop_samples_identical))
      *fmask_ptr = get_sampler_desc(ctx, instr->texture, ACO_DESC_FMASK, instr, false, false);
}

void prepare_cube_coords(isel_context *ctx, Temp* coords, bool is_deriv, bool is_array, bool is_lod)
{

   if (is_array && !is_lod)
      fprintf(stderr, "Unimplemented tex instr type: ");

   Temp coord_args[3], ma, tc, sc, id;
   std::unique_ptr<Instruction> tmp;
   for (unsigned i = 0; i < 3; i++) {
      tmp.reset(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
      tmp->getOperand(0) = Operand(*coords);
      tmp->getOperand(1) = Operand(i);
      coord_args[i] = {ctx->program->allocateId(), v1};
      tmp->getDefinition(0) = Definition(coord_args[i]);
      ctx->block->instructions.emplace_back(std::move(tmp));
   }
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

   if (is_deriv)
      fprintf(stderr, "Unimplemented tex instr type: ");

}

void visit_tex(isel_context *ctx, nir_tex_instr *instr)
{
   bool has_bias = false, has_lod = false;// level_zero = false;
   Temp resource, sampler, fmask_ptr, bias, coords, lod = Temp();
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
            //level_zero = true;
         } else {
            lod = get_ssa_temp(ctx, instr->src[i].src.ssa);
            has_lod = true;
         }
         break;
      }
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      default:
         break;
      }
   }
// TODO: all other cases: structure taken from ac_nir_to_llvm.c
   if (instr->op == nir_texop_txs && instr->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      fprintf(stderr, "Unimplemented tex instr type: ");

   if (instr->op == nir_texop_texture_samples)
      fprintf(stderr, "Unimplemented tex instr type: ");

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && instr->coord_components)
      prepare_cube_coords(ctx, &coords, instr->op == nir_texop_txd, instr->is_array, instr->op == nir_texop_lod);

   if (instr->coord_components > 1 &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
       instr->is_array &&
       instr->op != nir_texop_txf)
      fprintf(stderr, "Unimplemented tex instr type: ");

   if (instr->coord_components > 2 &&
      (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_MS ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) &&
       instr->is_array &&
       instr->op != nir_texop_txf && instr->op != nir_texop_txf_ms)
   fprintf(stderr, "Unimplemented tex instr type: ");

   if (ctx->options->chip_class >= GFX9 &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
       instr->op != nir_texop_lod)
   fprintf(stderr, "Unimplemented tex instr type: ");

   if (instr->op == nir_texop_samples_identical)
      fprintf(stderr, "Unimplemented tex instr type: ");

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_MS &&
       instr->op != nir_texop_txs)
   fprintf(stderr, "Unimplemented tex instr type: ");


   unsigned dmask = 0xf;
   if (instr->op == nir_texop_tg4)
      fprintf(stderr, "Unimplemented tex instr type: ");

   bool da = false;
   if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      aco_image_dim dim = get_sampler_dim(ctx, instr->sampler_dim, instr->is_array);

      da = dim == aco_image_cube ||
           dim == aco_image_1darray ||
           dim == aco_image_2darray ||
           dim == aco_image_2darraymsaa;
   }

   Temp arg = coords;
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

   if (has_bias) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(bias);
      vec->getOperand(1) = Operand(coords);
      RegClass rc = (RegClass) ((int) bias.regClass() + coords.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
   }

   if (instr->op == nir_texop_txb)
      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_sample_b, Format::MIMG, 3, 1));
   else
      tex.reset(create_instruction<MIMG_instruction>(aco_opcode::image_sample, Format::MIMG, 3, 1));
   tex->getOperand(0) = Operand{arg};
   tex->getOperand(1) = Operand(resource);
   tex->getOperand(2) = Operand(sampler);
   tex->dmask = dmask;
   tex->da = da;
   tex->getDefinition(0) = get_ssa_temp(ctx, &instr->dest.ssa);
   ctx->block->instructions.emplace_back(std::move(tex));
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
      case nir_instr_type_ssa_undef:
         visit_undef(ctx, nir_instr_as_ssa_undef(instr));
         break;
      default:
         fprintf(stderr, "Unknown NIR instr type: ");
         nir_print_instr(instr, stderr);
         fprintf(stderr, "\n");
         //abort();
      }
   }
}

static void visit_if(isel_context *ctx, nir_if *if_stmt)
{
   /* Disabled for now till I find time to untangle the assumptions. */
#if 0
   Temp cond = get_ssa_temp(ctx, if_stmt->condition.ssa);

   Block* aco_then = ctx->program->createAndInsertBlock();
   aco_then->logical_predecessors.push_back(ctx->block);
   ctx->block->logical_successors.push_back(aco_then);

   if (cond.type() == RegType::scc) { /* uniform condition */
      Block* aco_else = ctx->program->createAndInsertBlock();
      aco_else->logical_predecessors.push_back(ctx->block);
      ctx->block->logical_successors.push_back(aco_else);

      /* branch to the new block if condition is false */
      Builder(ctx->program, ctx->block).s_cbranch_scc0(Operand(cond), aco_else);

      /* emit then block */
      ctx->block = aco_then;
      visit_cf_list(ctx, &if_stmt->then_list);

      if (exec_list_is_empty(&if_stmt->else_list)) {
         /* if there is no else-list, take the
          * created aco_else block to continue */
         ctx->block = aco_else;
      } else {
         Block* aco_cont = ctx->program->createAndInsertBlock();
         aco_else->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_else);

         /* at the end of the then block, jump to the cont block */
         Builder(ctx->program, ctx->block).s_branch(aco_cont);

         /* emit else block */
         ctx->block = aco_else;
         visit_cf_list(ctx, &if_stmt->else_list);

         ctx->block = aco_cont;
      }
      aco_then->logical_successors.push_back(ctx->block);
      ctx->block->logical_predecessors.push_back(aco_then);

   } else { /* non-uniform condition */

      if (exec_list_is_empty(&if_stmt->else_list)) {
         Block* aco_cont = ctx->program->createAndInsertBlock();
         ctx->block->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(ctx->block);
         aco_then->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_then);

         /* without else-list, directly branch on condition */
         Builder(ctx->program, ctx->block).s_cbranch_vccz(Operand(cond), aco_cont);

         ctx->block = aco_then;
         Builder B(ctx->program, aco_then);
         /* set the exec mask inside then-block */
         Instruction* orig_exec = B.s_and_saveexec_b64(Operand(cond));

         /* emit then block */
         visit_cf_list(ctx, &if_stmt->then_list);

         /* restore exec mask */
         Instruction* restore = B.s_mov_b64(orig_exec->asOperand(0));
         restore->getDefinition(0).setFixed(PhysReg{126});
         ctx->block = aco_cont;

      } else {
         Block* aco_T = ctx->program->createAndInsertBlock();
         Block* aco_else = ctx->program->createAndInsertBlock();
         Block* aco_cont = ctx->program->createAndInsertBlock();
         ctx->block->logical_successors.push_back(aco_else);
         aco_else->logical_predecessors.push_back(ctx->block);
         aco_then->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_then);
         aco_else->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_else);
         Builder B(ctx->program, ctx->block);
         /* with else-list, first set exec mask */
         Instruction* orig_exec = B.s_and_saveexec_b64(Operand(cond));

         /* branch on exec mask to T block */
         B.s_cbranch_execz(aco_T);

         /* emit then block */
         ctx->block = aco_then;
         visit_cf_list(ctx, &if_stmt->then_list);

         ctx->block = aco_T;
         B = Builder(ctx->program, aco_T);
         /* negate exec mask */
         Instruction* else_exec = B.s_xor_b64(Operand(PhysReg{126}, s2), orig_exec->asOperand(0));
         else_exec->getDefinition(0).setFixed(PhysReg{126});

         /* branch on exec mask to cont block */
         B.s_cbranch_execz(aco_cont);

         /* emit else block */
         ctx->block = aco_else;
         visit_cf_list(ctx, &if_stmt->else_list);

         ctx->block = aco_cont;
         /* restore original exec mask */
         Instruction* restore = Builder(ctx->program, aco_cont).s_mov_b64(orig_exec->asOperand(0));
         restore->getDefinition(0).setFixed(PhysReg{126});
      }
   }
#endif
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
      default:
         unreachable("unimplemented cf list type");
      }
   }
}


std::unique_ptr<RegClass[]> init_reg_class(isel_context *ctx, nir_function_impl *impl)
{
   std::unique_ptr<RegClass[]> reg_class{new RegClass[impl->ssa_alloc]};

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
               case nir_op_frcp:
               case nir_op_frsq:
               case nir_op_fsqrt:
               case nir_op_fexp2:
               case nir_op_flog2:
               case nir_op_ffract:
               case nir_op_u2f32:
               case nir_op_i2f32:
                  type = vgpr;
                  break;
               case nir_op_flt:
               case nir_op_fge:
               case nir_op_feq:
               case nir_op_fne:
                  type = sgpr;
                  size = 2;
                  break;
               case nir_op_ilt:
               case nir_op_ige:
               case nir_op_ieq:
               case nir_op_ine:
               case nir_op_ult:
               case nir_op_uge:
                  if ((typeOf(reg_class[alu_instr->src[0].src.ssa->index]) == vgpr) ||
                      (typeOf(reg_class[alu_instr->src[1].src.ssa->index]) == vgpr)) {
                     size = 2;
                     type = sgpr;
                  } else {
                     type = scc;
                  }
                  break;
               case nir_op_bcsel:
                  if ((typeOf(reg_class[alu_instr->src[1].src.ssa->index]) == vgpr) ||
                      (typeOf(reg_class[alu_instr->src[2].src.ssa->index]) == vgpr)) {
                     type = vgpr;
                  /* if the arguments are not vgpr, but divergent, it must be bools */
                  } else if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index] &&
                             ctx->divergent_vals[alu_instr->src[1].src.ssa->index] &&
                             ctx->divergent_vals[alu_instr->src[2].src.ssa->index]) {
                     type = sgpr;
                     size = 2;
                  } else if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index]){
                     type = vgpr;
                  } else {
                     type = sgpr;
                  }
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
               case nir_intrinsic_load_input:
               case nir_intrinsic_load_vertex_id:
               case nir_intrinsic_load_vertex_id_zero_base:
               case nir_intrinsic_load_barycentric_pixel:
               case nir_intrinsic_load_interpolated_input:
                  type = vgpr;
                  break;
               case nir_intrinsic_vulkan_resource_index:
                  type = sgpr;
                  break;
               case nir_intrinsic_load_ubo:
                  type = ctx->divergent_vals[intrinsic->src[0].ssa->index] ? vgpr : sgpr;
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
   return reg_class;
}

struct user_sgpr_info {
   uint8_t num_sgpr;
   uint8_t user_sgpr_idx;
   bool need_ring_offsets;
   bool indirect_all_descriptor_sets;
};

static void allocate_user_sgprs(isel_context *ctx, gl_shader_stage stage,
                                /* TODO bool has_previous_stage, gl_shader_stage previous_stage, */
                                bool needs_view_index, user_sgpr_info& user_sgpr_info)
{
   memset(&user_sgpr_info, 0, sizeof(struct user_sgpr_info));
   uint32_t user_sgpr_count = 0;
#if 0
   /* until we sort out scratch/global buffers always assign ring offsets for gs/vs/es */
   if (stage == MESA_SHADER_GEOMETRY ||
       stage == MESA_SHADER_VERTEX ||
       stage == MESA_SHADER_TESS_CTRL ||
       stage == MESA_SHADER_TESS_EVAL ||
       ctx->is_gs_copy_shader)
      user_sgpr_info->need_ring_offsets = true;
#endif
   if (stage == MESA_SHADER_FRAGMENT &&
       ctx->program->info->info.ps.needs_sample_positions)
      user_sgpr_info.need_ring_offsets = true;

   /* 2 user sgprs will nearly always be allocated for scratch/rings */
   if (ctx->options->supports_spill || user_sgpr_info.need_ring_offsets) {
      user_sgpr_count += 2;
   }

   switch (stage) {
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
declare_global_input_sgprs(isel_context *ctx, gl_shader_stage stage,
                           /* bool has_previous_stage, gl_shader_stage previous_stage, */
                           user_sgpr_info *user_sgpr_info,
                           struct arg_info *args,
                           Temp *desc_sets)
{
   unsigned num_sets = ctx->options->layout ? ctx->options->layout->num_sets : 0;
   unsigned stage_mask = 1 << stage;

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

void add_startpgm(struct isel_context *ctx, gl_shader_stage stage)
{
   user_sgpr_info user_sgpr_info;
   allocate_user_sgprs(ctx, stage, false, user_sgpr_info);

   assert(!user_sgpr_info.indirect_all_descriptor_sets && "Not yet implemented.");
   arg_info args = {};

   if (user_sgpr_info.need_ring_offsets && !ctx->options->supports_spill)
      add_arg(&args, s2, &ctx->ring_offsets, 0);

   if (ctx->options->supports_spill || user_sgpr_info.need_ring_offsets) {
      set_loc_shader_ptr(ctx, AC_UD_SCRATCH_RING_OFFSETS, &user_sgpr_info.user_sgpr_idx);
   }

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      declare_global_input_sgprs(ctx, stage, &user_sgpr_info, &args, ctx->descriptor_sets);

      if (ctx->program->info->info.ps.needs_sample_positions) {
         add_arg(&args, s1, &ctx->sample_pos_offset, user_sgpr_info.user_sgpr_idx);
         set_loc_shader(ctx, AC_UD_PS_SAMPLE_POS_OFFSET, &user_sgpr_info.user_sgpr_idx, 1);
      }
      assert(user_sgpr_info.user_sgpr_idx == user_sgpr_info.num_sgpr);
      add_arg(&args, s1, &ctx->prim_mask, user_sgpr_info.user_sgpr_idx);
      add_arg(&args, v2, &ctx->barycentric_coords, 0);
      #if 0
      add_arg(&args, v2, &ctx->persp_sample);
      add_arg(&args, v2, &ctx->persp_center);
      add_arg(&args, v2, &ctx->persp_centroid);
      add_arg(&args, v3, NULL); /* persp pull model */
      add_arg(&args, v2, &ctx->linear_sample);
      add_arg(&args, v2, &ctx->linear_center);
      add_arg(&args, v2, &ctx->linear_centroid);
      add_arg(&args, v1, NULL);  /* line stipple tex */
      add_arg(&args, v1, &ctx->frag_pos[0]);
      add_arg(&args, v1, &ctx->frag_pos[1]);
      add_arg(&args, v1, &ctx->frag_pos[2]);
      add_arg(&args, v1, &ctx->frag_pos[3]);
      add_arg(&args, v1, &ctx->front_face);
      add_arg(&args, v1, &ctx->ancillary);
      add_arg(&args, v1, &ctx->sample_coverage);
      add_arg(&args, v1, NULL);  /* fixed pt */
      #endif
      break;
   default:
      unreachable("Shader stage not implemented");
   }

   ctx->program->info->num_input_vgprs = 0;
   ctx->program->info->num_input_sgprs = ctx->options->supports_spill ? 2 : 0;
   ctx->program->info->num_input_sgprs += args.num_sgprs_used;
   ctx->program->info->num_user_sgprs = user_sgpr_info.num_sgpr;

   if (stage != MESA_SHADER_FRAGMENT)
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
   nir_lower_io(nir, (nir_variable_mode)(nir_var_shader_in | nir_var_shader_out), type_size, (nir_lower_io_options)0);
   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   nir_index_ssa_defs(func->impl);
   ctx.divergent_vals = nir_divergence_analysis(nir);
   ctx.reg_class = init_reg_class(&ctx, func->impl);

   nir_print_shader(nir, stderr);

   ctx.program->blocks.push_back(std::unique_ptr<Block>{new Block});
   ctx.block = ctx.program->blocks.back().get();
   ctx.block->index = 0;

   nir_foreach_variable(variable, &nir->inputs)
   {
      int idx = variable->data.location - VARYING_SLOT_VAR0;
      ctx.input_mask |= 1ull << idx;
   }

   add_startpgm(&ctx, nir->info.stage);

   visit_cf_list(&ctx, &func->impl->body);

   ctx.block->instructions.push_back(std::unique_ptr<SOPP_instruction>(create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)));

   program->info->fs.num_interp = util_bitcount(ctx.input_mask);
   program->info->fs.input_mask = ctx.input_mask;

   return program;
}
}
