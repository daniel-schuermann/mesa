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
   bool *uniform_vals;
   std::unique_ptr<bool[]> use_vgpr;
   std::unordered_map<unsigned, unsigned> allocated;

   Temp barycentric_coords;
   Temp prim_mask;
   Temp descriptor_sets[RADV_UD_MAX_SETS];

   uint32_t input_mask;
};

static void visit_cf_list(struct isel_context *ctx,
                          struct exec_list *list);

RegClass get_ssa_reg_class(struct isel_context *ctx, nir_ssa_def *def)
{
   if (def->bit_size != 32) {
      fprintf(stderr, "Unsupported bit size for ssa-def %d: has %d bit\n", def->index, def->bit_size);
      abort();
   }
   unsigned v = def->num_components;
   if (ctx->use_vgpr[def->index])
      v |= 1 << 5;
   return (RegClass)v;
}

Temp get_ssa_temp(struct isel_context *ctx, nir_ssa_def *def)
{
   RegClass rc = get_ssa_reg_class(ctx, def);
   auto it = ctx->allocated.find(def->index);
   if (it != ctx->allocated.end())
      return Temp{it->second, rc};
   uint32_t id = ctx->program->allocateId();
   ctx->allocated.insert({def->index, id});
   return Temp{id, rc};
}

Temp get_alu_src(struct isel_context *ctx, nir_alu_src src)
{
   if (src.src.ssa->num_components == 1 || src.swizzle[0] == 0)
      return get_ssa_temp(ctx, src.src.ssa);

   Temp tmp{ctx->program->allocateId(), v1};
   std::unique_ptr<Instruction> extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   extract->getOperand(0) = Operand(get_ssa_temp(ctx, src.src.ssa));
   extract->getOperand(1) = Operand((uint32_t) src.swizzle[0]);
   extract->getDefinition(0) = Definition(tmp);
   ctx->block->instructions.emplace_back(std::move(extract));
   return tmp;
}


void emit_vop2_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op)
{
   std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(op, Format::VOP2, 2, 1)};
   vop2->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
   vop2->getOperand(1) = Operand{get_alu_src(ctx, instr->src[1])};
   vop2->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.dest.ssa));
   ctx->block->instructions.emplace_back(std::move(vop2));
}

void emit_vop1_instruction(isel_context *ctx, nir_alu_instr *instr, aco_opcode op)
{
   std::unique_ptr<VOP1_instruction> vop1{create_instruction<VOP1_instruction>(op, Format::VOP1, 1, 1)};
   vop1->getOperand(0) = Operand{get_alu_src(ctx, instr->src[0])};
   vop1->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.dest.ssa));
   ctx->block->instructions.emplace_back(std::move(vop1));
}

void visit_alu_instr(isel_context *ctx, nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa)
      abort();
   switch(instr->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->dest.dest.ssa.num_components, 1)};
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; ++i) {
         vec->getOperand(i) = Operand{get_alu_src(ctx, instr->src[i])};
      }
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
      break;
   }
   case nir_op_fmul: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_mul_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fadd: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_add_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fmax: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop2_instruction(ctx, instr, aco_opcode::v_max_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_frsq: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_rsq_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fneg: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         std::unique_ptr<VOP2_instruction> vop2{create_instruction<VOP2_instruction>(aco_opcode::v_sub_f32, Format::VOP2, 2, 1)};
         vop2->getOperand(0) = Operand((uint32_t) 0);
         vop2->getOperand(1) = Operand{get_alu_src(ctx, instr->src[0])};
         vop2->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.dest.ssa));
         ctx->block->instructions.emplace_back(std::move(vop2));
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_flog2: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_log_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
   }
   case nir_op_fexp2: {
      if (instr->dest.dest.ssa.bit_size == 32) {
         emit_vop1_instruction(ctx, instr, aco_opcode::v_exp_f32);
      } else {
         fprintf(stderr, "Unimplemented NIR instr bit size: ");
         nir_print_instr(&instr->instr, stderr);
         fprintf(stderr, "\n");
      }
      break;
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
      if (ctx->use_vgpr[instr->def.index]) {
         mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
      } else {
         mov.reset(create_instruction<Instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
      }
      mov->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->def));
      mov->getOperand(0) = Operand{instr->value.u32[0]};
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, instr->def.num_components, 1)};
      Temp t;
      for (unsigned i = 0; i < instr->def.num_components; i++)
      {
         if (ctx->use_vgpr[instr->def.index]) {
            mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
            t = Temp(ctx->program->allocateId(), v1);
         } else {
            mov.reset(create_instruction<Instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
            t = Temp(ctx->program->allocateId(), s1);
         }
         mov->getDefinition(0) = Definition(t);
         mov->getOperand(0) = Operand{instr->value.u32[0]};
         ctx->block->instructions.emplace_back(std::move(mov));
         vec->getOperand(i) = Operand(t);
      }
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
   // FIXME
   unsigned col_format = V_028714_SPI_SHADER_FP16_ABGR;//(ctx->options->key.fs.col_format >> (4 * index)) & 0xf;
   //bool is_int8 = (ctx->options->key.fs.is_int8 >> index) & 1;
   //bool is_int10 = (ctx->options->key.fs.is_int10 >> index) & 1;
   unsigned enabled_channels;
   aco_opcode compr_op;

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
         values[2*i] = Operand(tmp);
         values[2*i+1] = Operand();
         ctx->block->instructions.emplace_back(std::move(compr));
      }
   }

   std::unique_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
   exp->valid_mask = true; // TODO
   exp->done = true; // TODO
   exp->compressed = (bool) compr_op;
   exp->dest = target;
   exp->enabled_mask = enabled_channels;
   for (int i = 0; i < 4; i++)
      exp->getOperand(i) = values[i];

   ctx->block->instructions.emplace_back(std::move(exp));
}

void visit_load_interpolated_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr) / 4 - VARYING_SLOT_VAR0;
   unsigned idx = util_bitcount(ctx->input_mask & ((1u << base) - 1));

   unsigned component = nir_intrinsic_component(instr);
   Temp coord1{ctx->program->allocateId(), v1};
   Temp coord2{ctx->program->allocateId(), v1};

   std::unique_ptr<Instruction> coord1_extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   coord1_extract->getOperand(0) = Operand{get_ssa_temp(ctx, instr->src[0].ssa)};
   coord1_extract->getOperand(1) = Operand((uint32_t)0);

   coord1_extract->getDefinition(0) = Definition{coord1};
   std::unique_ptr<Instruction> coord2_extract(create_instruction<Instruction>(aco_opcode::p_extract_vector, Format::PSEUDO, 2, 1));
   coord2_extract->getOperand(0) = Operand{get_ssa_temp(ctx, instr->src[0].ssa)};
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
   p2->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));
   p2->attribute = idx;
   p2->component = component;

   ctx->block->instructions.emplace_back(std::move(p1));
   ctx->block->instructions.emplace_back(std::move(p2));
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
   default:
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
            std::unique_ptr<Instruction> indirect;
            indirect.reset(create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1));
            indirect->getDefinition(0) = Definition{ctx->program->allocateId(), s1};
            indirect->getOperand(0) = Operand(array_size);
            indirect->getOperand(1) = Operand{get_ssa_temp(ctx, child->indirect.ssa)};
            ctx->block->instructions.emplace_back(std::move(indirect));

            if (!index_set) {
               index = indirect->getDefinition(0).getTemp();
               index_set = true;
            } else {
               std::unique_ptr<Instruction> add;
               add.reset(create_instruction<SOP2_instruction>(aco_opcode::s_add_i32, Format::SOP2, 2, 1));
               add->getDefinition(0) = Definition{ctx->program->allocateId(), s1};
               add->getOperand(0) = Operand(index);
               add->getOperand(1) = Operand(indirect->getDefinition(0).getTemp());

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

   //LLVMValueRef list = ctx->descriptor_sets[descriptor_set];
   Temp list = ctx->descriptor_sets[descriptor_set];

   struct radv_descriptor_set_layout *layout = ctx->options->layout->set[descriptor_set].layout;
   struct radv_descriptor_set_binding_layout *binding = layout->binding + base_index;
   unsigned offset = binding->offset;
   unsigned stride = binding->size;
   unsigned type_size;
   aco_opcode opcode;
   RegClass type;

   assert(base_index < layout->binding_count);

   switch (desc_type) {
   case ACO_DESC_IMAGE:
      type = s8;
      opcode = aco_opcode::s_load_dwordx8;
      type_size = 32;
      break;
   case ACO_DESC_FMASK:
      type = s8;
      opcode = aco_opcode::s_load_dwordx8;
      offset += 32;
      type_size = 32;
      break;
   case ACO_DESC_SAMPLER:
      type = s4;
      opcode = aco_opcode::s_load_dwordx4;
      if (binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         offset += 64;

      type_size = 16;
      break;
   case ACO_DESC_BUFFER:
      type = s4;
      opcode = aco_opcode::s_load_dwordx4;
      type_size = 16;
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

   assert(stride % type_size == 0);
   Operand off;
   if (!index_set) {
      off = Operand(offset);
   } else {
      std::unique_ptr<SOP2_instruction> mul{create_instruction<SOP2_instruction>(aco_opcode::s_mul_i32, Format::SOP2, 2, 1)};
      mul->getOperand(0) = Operand(stride / type_size);
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

void visit_tex(isel_context *ctx, nir_tex_instr *instr)
{
   bool has_bias = false;
   Temp resource, sampler, fmask_ptr, bias, coords = Temp();
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

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && coords.id())
      fprintf(stderr, "Unimplemented tex instr type: ");

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

   Temp arg = get_ssa_temp(ctx, instr->src[0].src.ssa);

   if (has_bias) {
      std::unique_ptr<Instruction> vec{create_instruction<Instruction>(aco_opcode::p_create_vector, Format::PSEUDO, 2, 1)};
      vec->getOperand(0) = Operand(bias);
      vec->getOperand(1) = Operand(coords);
      RegClass rc = (RegClass) ((int) bias.regClass() + coords.size());
      arg = Temp{ctx->program->allocateId(), rc};
      vec->getDefinition(0) = Definition(arg);
      ctx->block->instructions.emplace_back(std::move(vec));
   }
   std::unique_ptr<MIMG_instruction> tex;
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


std::unique_ptr<bool[]> init_reg_type(nir_function_impl *impl)
{
   std::unique_ptr<bool[]> use_vgpr{new bool[impl->ssa_alloc]};
   for(unsigned i = 0; i < impl->ssa_alloc; ++i)
      use_vgpr[i] = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch(instr->type) {
         case nir_instr_type_alu: {
            nir_alu_instr *alu_instr = nir_instr_as_alu(instr);
            bool emit_vgpr = false;
            switch(alu_instr->op) {
               case nir_op_fmul:
               case nir_op_fadd:
               case nir_op_fsub:
               case nir_op_fneg:
               case nir_op_fabs:
                  emit_vgpr = true;
                  break;
               default:
                  for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                     if (use_vgpr[alu_instr->src[i].src.ssa->index])
                        emit_vgpr = true;
                  }
                  break;
            }
            use_vgpr[alu_instr->dest.dest.ssa.index] = emit_vgpr;
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
            bool emit_vgpr = false;
            switch(intrinsic->intrinsic) {
               case nir_intrinsic_load_input:
               case nir_intrinsic_load_vertex_id:
               case nir_intrinsic_load_vertex_id_zero_base:
               case nir_intrinsic_load_barycentric_pixel:
               case nir_intrinsic_load_interpolated_input:
                  emit_vgpr = true;
                  break;
               default:
                  for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs; i++) {
                     if (use_vgpr[intrinsic->src[i].ssa->index])
                        emit_vgpr = true;
                  }
                  break;
            }
            if (nir_intrinsic_infos[intrinsic->intrinsic].has_dest) {
               use_vgpr[intrinsic->dest.ssa.index] = emit_vgpr;
            }
            break;
         }
         case nir_instr_type_tex:
            use_vgpr[nir_instr_as_tex(instr)->dest.ssa.index] = true;
            break;
         default:
            break;
         }
      }
   }
   return use_vgpr;
}

int
type_size(const struct glsl_type *type)
{
        return glsl_count_attribute_slots(type, false);
}

void add_startpgm(struct isel_context *ctx)
{
   unsigned num_descriptor_sets = util_bitcount(ctx->program->info->info.desc_set_used_mask);
   std::unique_ptr<Instruction> startpgm{create_instruction<Instruction>(aco_opcode::p_startpgm, Format::PSEUDO, 0, 2 + num_descriptor_sets)};

   ctx->barycentric_coords = Temp{ctx->program->allocateId(), v2};

   startpgm->getDefinition(0) = Definition{ctx->barycentric_coords};
   startpgm->getDefinition(0).setFixed(fixed_vgpr(0));

   unsigned user_sgpr = 2;
   unsigned descriptor_set_cnt = 0;
   for (unsigned i = 0; i < RADV_UD_MAX_SETS; ++i) {
      if (!((1u << i) & ctx->program->info->info.desc_set_used_mask))
         continue;

      ctx->descriptor_sets[i] = Temp{ctx->program->allocateId(), s2};

      startpgm->getDefinition(1 + descriptor_set_cnt) = Definition{ctx->descriptor_sets[i]};
      startpgm->getDefinition(1 + descriptor_set_cnt).setFixed(fixed_sgpr(user_sgpr));

      ctx->program->info->user_sgprs_locs.descriptor_sets[0].sgpr_idx = user_sgpr;
      ctx->program->info->user_sgprs_locs.descriptor_sets[0].num_sgprs = 2;

      ++descriptor_set_cnt;
      user_sgpr += 2;
   }

   ctx-> program->info->num_user_sgprs = user_sgpr;

   ctx->prim_mask = Temp{ctx->program->allocateId(), s1};
   startpgm->getDefinition(1 + num_descriptor_sets) = Definition{ctx->prim_mask};
   startpgm->getDefinition(1 + num_descriptor_sets).setFixed(fixed_sgpr(user_sgpr));

   ctx->block->instructions.push_back(std::move(startpgm));
}

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
   ctx.uniform_vals = nir_uniform_analysis(nir);
   ctx.use_vgpr = init_reg_type(func->impl);

   nir_print_shader(nir, stderr);

   ctx.program->blocks.push_back(std::unique_ptr<Block>{new Block});
   ctx.block = ctx.program->blocks.back().get();
   ctx.block->index = 0;

   nir_foreach_variable(variable, &nir->inputs)
   {
      int idx = variable->data.location - VARYING_SLOT_VAR0;
      ctx.input_mask |= 1ull << idx;
   }

   add_startpgm(&ctx);

   visit_cf_list(&ctx, &func->impl->body);

   ctx.block->instructions.push_back(std::unique_ptr<SOPP_instruction>(create_instruction<SOPP_instruction>(aco_opcode::s_endpgm, Format::SOPP, 0, 0)));

   program->info->fs.num_interp = util_bitcount(ctx.input_mask);
   program->info->fs.input_mask = ctx.input_mask;

   return program;
}
}
