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
 */

#include <unordered_map>
#include "aco_ir.h"
#include "nir/nir.h"
#include "vulkan/radv_shader.h"
#include "common/sid.h"

#include "util/u_math.h"

namespace aco {

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

struct isel_context {
   struct radv_nir_compiler_options *options;
   Program *program;
   Block *block;
   bool *divergent_vals;
   std::unique_ptr<RegClass[]> reg_class;
   std::unordered_map<unsigned, unsigned> allocated;
   std::unordered_map<unsigned, std::array<Temp,4>> allocated_vec;
   gl_shader_stage stage;
   struct {
      bool has_continue;
      bool has_break;
      uint16_t loop_nest_depth = 0;
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

   /* CS inputs */
   Temp num_workgroups[3];
   Temp workgroup_ids[3];
   Temp tg_size;
   Temp local_invocation_ids[3];

   uint64_t input_mask;
};

void init_context(isel_context *ctx, nir_function_impl *impl)
{
   std::unique_ptr<RegClass[]> reg_class{new RegClass[impl->ssa_alloc]()};
   memset(&ctx->fs_vgpr_args, false, sizeof(ctx->fs_vgpr_args));

   bool done = false;
   while (!done) {
      done = true;
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
                  case nir_op_fmod:
                  case nir_op_fneg:
                  case nir_op_fabs:
                  case nir_op_fsat:
                  case nir_op_fsign:
                  case nir_op_frcp:
                  case nir_op_frsq:
                  case nir_op_fsqrt:
                  case nir_op_fexp2:
                  case nir_op_flog2:
                  case nir_op_ffract:
                  case nir_op_ffloor:
                  case nir_op_fceil:
                  case nir_op_fsin:
                  case nir_op_fcos:
                  case nir_op_u2f32:
                  case nir_op_i2f32:
                  case nir_op_b2f32:
                  case nir_op_fddx:
                  case nir_op_fddy:
                     type = vgpr;
                     break;
                  case nir_op_flt32:
                  case nir_op_fge32:
                  case nir_op_feq32:
                  case nir_op_fne32:
                  case nir_op_ilt32:
                  case nir_op_ige32:
                  case nir_op_ieq32:
                  case nir_op_ine32:
                  case nir_op_ult32:
                  case nir_op_uge32:
                  case nir_op_f2i32:
                  case nir_op_f2u32:
                  case nir_op_i2b32:
                  case nir_op_b2i32:
                     type = ctx->divergent_vals[alu_instr->dest.dest.ssa.index] ? vgpr : sgpr;
                     break;
                  case nir_op_b32csel:
                     if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index])
                        type = vgpr;
                     // TODO: with 1-bit bools, the regClass changes!
                     /* fallthrough */
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
               if (!nir_intrinsic_infos[intrinsic->intrinsic].has_dest)
                  break;
               unsigned size =  intrinsic->dest.ssa.num_components;
               if (intrinsic->dest.ssa.bit_size == 64)
                  size *= 2;
               RegType type = sgpr;
               switch(intrinsic->intrinsic) {
                  case nir_intrinsic_load_work_group_id:
                  case nir_intrinsic_load_num_work_groups:
                  case nir_intrinsic_get_buffer_size:
                     type = sgpr;
                     break;
                  case nir_intrinsic_load_front_face:
                  case nir_intrinsic_load_sample_id:
                  case nir_intrinsic_load_input:
                  case nir_intrinsic_load_vertex_id:
                  case nir_intrinsic_load_vertex_id_zero_base:
                  case nir_intrinsic_load_barycentric_pixel:
                  case nir_intrinsic_load_interpolated_input:
                  case nir_intrinsic_load_local_invocation_id:
                  case nir_intrinsic_load_local_invocation_index:
                  case nir_intrinsic_load_shared:
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
                  case nir_intrinsic_image_deref_load:
                  case nir_intrinsic_image_deref_atomic_add:
                  case nir_intrinsic_image_deref_atomic_min:
                  case nir_intrinsic_image_deref_atomic_max:
                  case nir_intrinsic_image_deref_atomic_and:
                  case nir_intrinsic_image_deref_atomic_or:
                  case nir_intrinsic_image_deref_atomic_xor:
                  case nir_intrinsic_image_deref_atomic_exchange:
                  case nir_intrinsic_image_deref_atomic_comp_swap:
                  case nir_intrinsic_image_deref_size:
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
                     type = vgpr;
                     break;
                  case nir_intrinsic_vulkan_resource_index:
                     type = sgpr;
                     size = 2;
                     break;
                  case nir_intrinsic_load_ubo:
                  case nir_intrinsic_load_ssbo:
                     type = ctx->divergent_vals[intrinsic->dest.ssa.index] ? vgpr : sgpr;
                     break;
                  default:
                     for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs; i++) {
                        if (typeOf(reg_class[intrinsic->src[i].ssa->index]) == vgpr)
                           type = vgpr;
                     }
                     break;
               }
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
                        uint8_t mask = nir_ssa_def_components_read(&intrinsic->dest.ssa);
                        for (unsigned i = 0; i < 4; i++) {
                           if (mask & (1 << i))
                              ctx->fs_vgpr_args[fs_input::frag_pos_0 + i] = true;

                        }
                     }
                     break;
                  case nir_intrinsic_load_sample_id:
                     ctx->fs_vgpr_args[fs_input::ancillary] = true;
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
               unsigned size = phi->dest.ssa.num_components;
               if (phi->dest.ssa.bit_size == 64)
                  size *= 2;
               RegType type;
               if (ctx->divergent_vals[phi->dest.ssa.index]) {
                  type = vgpr;
               } else {
                  type = sgpr;
                  nir_foreach_phi_src (src, phi) {
                     if (reg_class[src->src.ssa->index] == getRegClass(vgpr, size))
                        type = vgpr;
                     else if (reg_class[src->src.ssa->index] != getRegClass(sgpr, size))
                        done = false;
                  }
               }

               reg_class[phi->dest.ssa.index] = getRegClass(type, size);
               break;
            }
            default:
               break;
            }
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
   case MESA_SHADER_COMPUTE:
      if (ctx->program->info->info.cs.uses_grid_size)
         user_sgpr_count += 3;
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
        bool indirect)
{
   ud_info->sgpr_idx = *sgpr_idx;
   ud_info->num_sgprs = num_sgprs;
   ud_info->indirect = indirect;
   *sgpr_idx += num_sgprs;
}

static void
set_loc_shader(isel_context *ctx, int idx, uint8_t *sgpr_idx,
               uint8_t num_sgprs)
{
   struct radv_userdata_info *ud_info = &ctx->program->info->user_sgprs_locs.shader_data[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, num_sgprs, false);
}

static void
set_loc_shader_ptr(isel_context *ctx, int idx, uint8_t *sgpr_idx)
{
   bool use_32bit_pointers = idx != AC_UD_SCRATCH_RING_OFFSETS;

   set_loc_shader(ctx, idx, sgpr_idx, use_32bit_pointers ? 1 : 2);
}

static void
set_loc_desc(isel_context *ctx, int idx,  uint8_t *sgpr_idx,
             bool indirect)
{
   struct radv_userdata_locations *locs = &ctx->program->info->user_sgprs_locs;
   struct radv_userdata_info *ud_info = &locs->descriptor_sets[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, 1, indirect);

   if (!indirect)
      locs->descriptor_sets_enabled |= 1 << idx;
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
            set_loc_desc(ctx, i, &user_sgpr_info->user_sgpr_idx, false);
         }
      }
   } else {
      unreachable("Fix access to indirect descriptor sets.");
      add_array_arg(args, s1, desc_sets, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_INDIRECT_DESCRIPTOR_SETS, &user_sgpr_info->user_sgpr_idx);
      /*
      for (unsigned i = 0; i < num_sets; ++i) {
         if ((ctx->program->info->info.desc_set_used_mask & (1 << i)) &&
             ctx->options->layout->set[i].layout->shader_stages & stage_mask)
            set_loc_desc(ctx, i, &user_sgpr_info->user_sgpr_idx, i * 8);
      }
      */
      ctx->program->info->need_indirect_descriptor_sets = true;
   }

   if (ctx->program->info->info.loads_push_constants) {
      /* 1 for push constants and dynamic descriptors */
      add_array_arg(args, s1, &ctx->push_constants, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_PUSH_CONSTANTS, &user_sgpr_info->user_sgpr_idx);
   }

   if (ctx->program->info->info.so.num_outputs) {
      unreachable("Streamout not yet supported.");
      //add_arg(args, s4, &ctx->streamout_buffers, user_sgpr_info->user_sgpr_idx);
      //set_loc_shader_ptr(ctx, AC_UD_STREAMOUT_BUFFERS, &user_sgpr_info->user_sgpr_idx);
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

      bool needs_interp_mode = !(ctx->program->config->spi_ps_input_addr & 0x7F) ||
                               (G_0286CC_POS_W_FLOAT_ENA(ctx->program->config->spi_ps_input_addr)
                                && !(ctx->program->config->spi_ps_input_addr & 0xF));
      unsigned interp_mode = needs_interp_mode ? S_0286CC_PERSP_CENTER_ENA(1) : 0;
      ctx->program->config->spi_ps_input_addr |= interp_mode;
      ctx->program->config->spi_ps_input_ena |= interp_mode;
      break;
   }
   case MESA_SHADER_COMPUTE: {
      declare_global_input_sgprs(ctx, &user_sgpr_info, &args, ctx->descriptor_sets);

      if (ctx->program->info->info.cs.uses_grid_size) {
         add_arg(&args, s1, &ctx->num_workgroups[0], user_sgpr_info.user_sgpr_idx);
         add_arg(&args, s1, &ctx->num_workgroups[1], user_sgpr_info.user_sgpr_idx + 1);
         add_arg(&args, s1, &ctx->num_workgroups[2], user_sgpr_info.user_sgpr_idx + 2);
         set_loc_shader(ctx, AC_UD_CS_GRID_SIZE, &user_sgpr_info.user_sgpr_idx, 3);
      }
      assert(user_sgpr_info.user_sgpr_idx == user_sgpr_info.num_sgpr);
      unsigned idx = user_sgpr_info.user_sgpr_idx;
      for (unsigned i = 0; i < 3; i++) {
         if (ctx->program->info->info.cs.uses_block_id[i])
            add_arg(&args, s1, &ctx->workgroup_ids[i], idx++);
      }

      if (ctx->program->info->info.cs.uses_local_invocation_idx)
         add_arg(&args, s1, &ctx->tg_size, idx++);

      add_arg(&args, v1, &ctx->local_invocation_ids[0], vgpr_idx++);
      add_arg(&args, v1, &ctx->local_invocation_ids[1], vgpr_idx++);
      add_arg(&args, v1, &ctx->local_invocation_ids[2], vgpr_idx++);
      break;
   }
   default:
      unreachable("Shader stage not implemented");
   }

   ctx->program->info->num_input_vgprs = 0;
   ctx->program->info->num_input_sgprs = ctx->options->supports_spill ? 2 : 0;
   ctx->program->info->num_input_sgprs += args.num_sgprs_used;
   ctx->program->info->num_user_sgprs = user_sgpr_info.num_sgpr;
   ctx->program->info->num_input_vgprs = args.num_vgprs_used;

   aco_ptr<Instruction> startpgm{create_instruction<Instruction>(aco_opcode::p_startpgm, Format::PSEUDO, 0, args.count)};
   for (unsigned i = 0; i < args.count; i++) {
      if (args.assign[i]) {
         *args.assign[i] = Temp{ctx->program->allocateId(), args.types[i]};
         startpgm->getDefinition(i) = *args.assign[i];
         startpgm->getDefinition(i).setFixed(args.reg[i]);
      }
   }
   ctx->block->instructions.push_back(std::move(startpgm));
}

int
type_size(const struct glsl_type *type)
{
   // TODO: don't we need type->std430_base_alignment() here?
   return glsl_count_attribute_slots(type, false);
}

/* Taken from src/intel/compiler/brw_fs.cpp
   TODO: this might better go to core */
int
type_size_scalar(const struct glsl_type *type)
{
   unsigned int size, i;

   switch (type->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_BOOL:
      return type->components();
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
      return DIV_ROUND_UP(type->components(), 2);
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      return DIV_ROUND_UP(type->components(), 4);
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return type->components() * 2;
   case GLSL_TYPE_ARRAY:
      return type_size_scalar(type->fields.array) * type->length;
   case GLSL_TYPE_STRUCT:
      size = 0;
      for (i = 0; i < type->length; i++) {
         size += type_size_scalar(type->fields.structure[i].type);
      }
      return size;
   case GLSL_TYPE_SAMPLER:
      /* Samplers take up no register space, since they're baked in at
       * link time.
       */
      return 0;
   case GLSL_TYPE_ATOMIC_UINT:
      return 0;
   case GLSL_TYPE_SUBROUTINE:
      return 1;
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_FUNCTION:
      unreachable("not reached");
   }

   return 0;
}

int
shared_var_size(const struct glsl_type *type)
{
   return type_size_scalar(type)*4;
}

unsigned
total_shared_var_size(const struct glsl_type *type)
{
   if (type->is_array())
      return type->arrays_of_arrays_size() * shared_var_size(type->without_array());
   else
      return shared_var_size(type);
}

void
setup_variables(isel_context *ctx, nir_shader *nir)
{
   switch (ctx->stage) {
   case MESA_SHADER_FRAGMENT: {
      nir_foreach_variable(variable, &nir->outputs)
      {
         int idx = variable->data.location + variable->data.index;
         variable->data.driver_location = idx * 4;
      }
      nir_foreach_variable(variable, &nir->inputs)
      {
         int idx = variable->data.location;
         variable->data.driver_location = idx * 4;
         unsigned attrib_count = glsl_count_attribute_slots(variable->type, false);
         if (idx >= VARYING_SLOT_VAR0 || idx == VARYING_SLOT_PNTC ||
             idx == VARYING_SLOT_PRIMITIVE_ID || idx == VARYING_SLOT_LAYER)
            ctx->input_mask |= ((1ull << attrib_count) - 1ull) << idx;
      }
      ctx->program->info->fs.num_interp = util_bitcount64(ctx->input_mask);
      ctx->program->info->fs.input_mask = ctx->input_mask >> VARYING_SLOT_VAR0;
      ctx->program->info->fs.can_discard = nir->info.fs.uses_discard;
      ctx->program->info->fs.early_fragment_test = nir->info.fs.early_fragment_tests;
      break;
   }
   case MESA_SHADER_COMPUTE: {
      unsigned lds_size_bytes = 0;
      nir_foreach_variable(variable, &nir->shared)
      {
         lds_size_bytes += total_shared_var_size(variable->type);
      }
      const unsigned lds_allocation_size_unit = 4 * 64;
      ctx->program->config->lds_size = (lds_size_bytes + lds_allocation_size_unit - 1) / lds_allocation_size_unit;
      ctx->program->info->cs.block_size[0] = nir->info.cs.local_size[0];
      ctx->program->info->cs.block_size[1] = nir->info.cs.local_size[1];
      ctx->program->info->cs.block_size[2] = nir->info.cs.local_size[2];
      break;
   }
   default:
      unreachable("Unhandled shader stage.");
   }

   ctx->program->config->float_mode = V_00B028_FP_64_DENORMS;
}

isel_context
setup_isel_context(Program* program, nir_shader *nir,
                   ac_shader_config* config,
                   radv_shader_variant_info *info,
                   radv_nir_compiler_options *options)
{
   program->config = config;
   program->info = info;
   program->chip_class = options->chip_class;
   program->stage = nir->info.stage;
   for (unsigned i = 0; i < RADV_UD_MAX_SETS; ++i)
      program->info->user_sgprs_locs.descriptor_sets[i].sgpr_idx = -1;
   for (unsigned i = 0; i < AC_UD_MAX_UD; ++i)
      program->info->user_sgprs_locs.shader_data[i].sgpr_idx = -1;

   isel_context ctx = {};
   ctx.program = program;
   ctx.options = options;
   ctx.stage = nir->info.stage;

   /* the variable setup has to be done before lower_io / CSE */
   setup_variables(&ctx, nir);

   nir_lower_load_const_to_scalar(nir);
   nir_lower_io(nir, (nir_variable_mode)(nir_var_shader_in | nir_var_shader_out), type_size, (nir_lower_io_options)0);
   nir_lower_io(nir, nir_var_shared, shared_var_size, (nir_lower_io_options)0);
   nir_copy_prop(nir);
   nir_opt_idiv_const(nir, 32);
   nir_opt_shrink_load(nir);
   nir_opt_cse(nir);
   nir_opt_dce(nir);
   nir_opt_sink(nir);
   nir_opt_move_load_ubo(nir);

   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   nir_index_ssa_defs(func->impl);

   if (options->dump_preoptir) {
      fprintf(stderr, "NIR shader before instruction selection:\n");
      nir_print_shader(nir, stderr);
   }
   ctx.divergent_vals = nir_divergence_analysis(nir);
   init_context(&ctx, func->impl);

   ctx.program->blocks.push_back(std::unique_ptr<Block>{new Block});
   ctx.block = ctx.program->blocks.back().get();
   ctx.block->index = 0;
   ctx.block->loop_nest_depth = 0;

   add_startpgm(&ctx);

   return ctx;
}

}
