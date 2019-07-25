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
#include "nir.h"
#include "vulkan/radv_shader.h"
#include "sid.h"
#include "ac_exp_param.h"

#include "util/u_math.h"

#define MAX_INLINE_PUSH_CONSTS 8

namespace aco {

enum fs_input {
   persp_sample_p1,
   persp_sample_p2,
   persp_center_p1,
   persp_center_p2,
   persp_centroid_p1,
   persp_centroid_p2,
   persp_pull_model,
   linear_sample_p1,
   linear_sample_p2,
   linear_center_p1,
   linear_center_p2,
   linear_centroid_p1,
   linear_centroid_p2,
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

struct vs_output_state {
   uint8_t mask[VARYING_SLOT_VAR31 + 1];
   Temp outputs[VARYING_SLOT_VAR31 + 1][4];
};

struct isel_context {
   struct radv_nir_compiler_options *options;
   Program *program;
   Block *block;
   bool *divergent_vals;
   std::unique_ptr<RegClass[]> reg_class;
   std::unordered_map<unsigned, unsigned> allocated;
   /* used to avoid splitting SCC live ranges and pointless conversions */
   std::unordered_map<unsigned, unsigned> allocated_bool32;
   std::unordered_map<unsigned, std::array<Temp,4>> allocated_vec;
   gl_shader_stage stage;
   struct {
      bool has_branch;
      bool has_continue;
      bool has_break;
      uint16_t loop_nest_depth = 0;
      struct {
         unsigned header_idx;
         Block* exit;
         bool has_divergent_continue = false;
         bool has_divergent_branch = false;
      } parent_loop;
      struct {
         bool is_divergent = false;
      } parent_if;
   } cf_info;

   /* FS inputs */
   bool fs_vgpr_args[fs_input::max_inputs];
   Temp fs_inputs[fs_input::max_inputs];
   Temp prim_mask = Temp(0, s1);
   Temp descriptor_sets[RADV_UD_MAX_SETS];
   Temp push_constants = Temp(0, s1);
   Temp inline_push_consts[MAX_INLINE_PUSH_CONSTS];
   unsigned num_inline_push_consts = 0;
   unsigned base_inline_push_consts = 0;
   Temp ring_offsets = Temp(0, s2);

   /* VS inputs */
   Temp vertex_buffers = Temp(0, s1);
   Temp base_vertex = Temp(0, s1);
   Temp start_instance = Temp(0, s1);
   Temp draw_id = Temp(0, s1);
   Temp view_index = Temp(0, s1);
   Temp es2gs_offset = Temp(0, s1);
   Temp vertex_id = Temp(0, v1);
   Temp rel_auto_id = Temp(0, v1);
   Temp instance_id = Temp(0, v1);
   Temp vs_prim_id = Temp(0, v1);
   bool needs_instance_id;

   /* CS inputs */
   Temp num_workgroups[3] = {Temp(0, s1), Temp(0, s1), Temp(0, s1)};
   Temp workgroup_ids[3] = {Temp(0, s1), Temp(0, s1), Temp(0, s1)};
   Temp tg_size = Temp(0, s1);
   Temp local_invocation_ids[3] = {Temp(0, v1), Temp(0, v1), Temp(0, v1)};

   /* VS output information */
   unsigned num_clip_distances;
   unsigned num_cull_distances;
   vs_output_state vs_output;

   /* Streamout */
   Temp streamout_buffers = Temp(0, s1);
   Temp streamout_write_idx = Temp(0, s1);
   Temp streamout_config = Temp(0, s1);
   Temp streamout_offset[4] = {Temp(0, s1), Temp(0, s1), Temp(0, s1), Temp(0, s1)};
};

fs_input get_interp_input(nir_intrinsic_op intrin, enum glsl_interp_mode interp)
{
   switch (interp) {
   case INTERP_MODE_SMOOTH:
   case INTERP_MODE_NONE:
      if (intrin == nir_intrinsic_load_barycentric_pixel ||
          intrin == nir_intrinsic_load_barycentric_at_sample ||
          intrin == nir_intrinsic_load_barycentric_at_offset)
         return fs_input::persp_center_p1;
      else if (intrin == nir_intrinsic_load_barycentric_centroid)
         return fs_input::persp_centroid_p1;
      else if (intrin == nir_intrinsic_load_barycentric_sample)
         return fs_input::persp_sample_p1;
      break;
   case INTERP_MODE_NOPERSPECTIVE:
      if (intrin == nir_intrinsic_load_barycentric_pixel)
         return fs_input::linear_center_p1;
      else if (intrin == nir_intrinsic_load_barycentric_centroid)
         return fs_input::linear_centroid_p1;
      else if (intrin == nir_intrinsic_load_barycentric_sample)
         return fs_input::linear_sample_p1;
      break;
   default:
      break;
   }
   return fs_input::max_inputs;
}

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
                  case nir_op_fmax3:
                  case nir_op_fmin3:
                  case nir_op_fmed3:
                  case nir_op_fmod:
                  case nir_op_frem:
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
                  case nir_op_ftrunc:
                  case nir_op_fround_even:
                  case nir_op_fsin:
                  case nir_op_fcos:
                  case nir_op_f2f32:
                  case nir_op_f2f64:
                  case nir_op_u2f32:
                  case nir_op_u2f64:
                  case nir_op_i2f32:
                  case nir_op_i2f64:
                  case nir_op_pack_half_2x16:
                  case nir_op_unpack_half_2x16_split_x:
                  case nir_op_unpack_half_2x16_split_y:
                  case nir_op_fddx:
                  case nir_op_fddy:
                  case nir_op_fddx_fine:
                  case nir_op_fddy_fine:
                  case nir_op_fddx_coarse:
                  case nir_op_fddy_coarse:
                  case nir_op_fquantize2f16:
                  case nir_op_ldexp:
                  case nir_op_frexp_sig:
                  case nir_op_frexp_exp:
                  case nir_op_cube_face_index:
                  case nir_op_cube_face_coord:
                     type = vgpr;
                     break;
                  case nir_op_flt:
                  case nir_op_fge:
                  case nir_op_feq:
                  case nir_op_fne:
                     size = 2;
                     break;
                  case nir_op_ilt:
                  case nir_op_ige:
                  case nir_op_ult:
                  case nir_op_uge:
                     size = alu_instr->src[0].src.ssa->bit_size == 64 ? 2 : 1;
                     /* fallthrough */
                  case nir_op_ieq:
                  case nir_op_ine:
                  case nir_op_i2b1:
                     if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index]) {
                        size = 2;
                     } else {
                        for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                           if (reg_class[alu_instr->src[i].src.ssa->index].type() == vgpr)
                              size = 2;
                        }
                     }
                     break;
                  case nir_op_f2i64:
                  case nir_op_f2u64:
                  case nir_op_b2i32:
                  case nir_op_b2f32:
                  case nir_op_f2i32:
                  case nir_op_f2u32:
                     type = ctx->divergent_vals[alu_instr->dest.dest.ssa.index] ? vgpr : sgpr;
                     break;
                  case nir_op_bcsel:
                     if (alu_instr->dest.dest.ssa.bit_size == 1) {
                        if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index])
                           size = 2;
                        else if (reg_class[alu_instr->src[1].src.ssa->index] == s2 &&
                                 reg_class[alu_instr->src[2].src.ssa->index] == s2)
                           size = 2;
                        else
                           size = 1;
                     } else {
                        if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index]) {
                           type = vgpr;
                        } else {
                           if (reg_class[alu_instr->src[1].src.ssa->index].type() == vgpr ||
                               reg_class[alu_instr->src[2].src.ssa->index].type() == vgpr) {
                              type = vgpr;
                           }
                        }
                        if (alu_instr->src[1].src.ssa->num_components == 1 && alu_instr->src[2].src.ssa->num_components == 1) {
                           assert(reg_class[alu_instr->src[1].src.ssa->index].size() == reg_class[alu_instr->src[2].src.ssa->index].size());
                           size = reg_class[alu_instr->src[1].src.ssa->index].size();
                        }
                     }
                     break;
                  case nir_op_mov:
                     if (alu_instr->dest.dest.ssa.bit_size == 1) {
                        size = reg_class[alu_instr->src[0].src.ssa->index].size();
                     } else {
                        type = ctx->divergent_vals[alu_instr->dest.dest.ssa.index] ? vgpr : sgpr;
                     }
                     break;
                  case nir_op_inot:
                  case nir_op_ixor:
                     if (alu_instr->dest.dest.ssa.bit_size == 1) {
                        size = ctx->divergent_vals[alu_instr->dest.dest.ssa.index] ? 2 : 1;
                        break;
                     } else {
                        /* fallthrough */
                     }
                  default:
                     if (alu_instr->dest.dest.ssa.bit_size == 1) {
                        if (ctx->divergent_vals[alu_instr->dest.dest.ssa.index]) {
                           size = 2;
                        } else {
                           size = 2;
                           for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                              if (reg_class[alu_instr->src[i].src.ssa->index] == s1) {
                                 size = 1;
                                 break;
                              }
                           }
                        }
                     } else {
                        for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                           if (reg_class[alu_instr->src[i].src.ssa->index].type() == vgpr)
                              type = vgpr;
                        }
                     }
                     break;
               }
               reg_class[alu_instr->dest.dest.ssa.index] = RegClass(type, size);
               break;
            }
            case nir_instr_type_load_const: {
               unsigned size = nir_instr_as_load_const(instr)->def.num_components;
               if (nir_instr_as_load_const(instr)->def.bit_size == 64)
                  size *= 2;
               reg_class[nir_instr_as_load_const(instr)->def.index] = RegClass(sgpr, size);
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
                  case nir_intrinsic_load_push_constant:
                  case nir_intrinsic_load_work_group_id:
                  case nir_intrinsic_load_num_work_groups:
                  case nir_intrinsic_load_subgroup_id:
                  case nir_intrinsic_load_num_subgroups:
                  case nir_intrinsic_load_first_vertex:
                  case nir_intrinsic_load_base_instance:
                  case nir_intrinsic_get_buffer_size:
                  case nir_intrinsic_vote_all:
                  case nir_intrinsic_vote_any:
                  case nir_intrinsic_read_first_invocation:
                  case nir_intrinsic_read_invocation:
                  case nir_intrinsic_first_invocation:
                  case nir_intrinsic_vulkan_resource_index:
                     type = sgpr;
                     break;
                  case nir_intrinsic_ballot:
                     type = sgpr;
                     size = 2;
                     break;
                  case nir_intrinsic_load_sample_id:
                  case nir_intrinsic_load_sample_mask_in:
                  case nir_intrinsic_load_input:
                  case nir_intrinsic_load_vertex_id:
                  case nir_intrinsic_load_vertex_id_zero_base:
                  case nir_intrinsic_load_barycentric_sample:
                  case nir_intrinsic_load_barycentric_pixel:
                  case nir_intrinsic_load_barycentric_centroid:
                  case nir_intrinsic_load_barycentric_at_sample:
                  case nir_intrinsic_load_barycentric_at_offset:
                  case nir_intrinsic_load_interpolated_input:
                  case nir_intrinsic_load_frag_coord:
                  case nir_intrinsic_load_sample_pos:
                  case nir_intrinsic_load_layer_id:
                  case nir_intrinsic_load_local_invocation_id:
                  case nir_intrinsic_load_local_invocation_index:
                  case nir_intrinsic_load_subgroup_invocation:
                  case nir_intrinsic_write_invocation_amd:
                  case nir_intrinsic_mbcnt_amd:
                  case nir_intrinsic_load_instance_id:
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
                  case nir_intrinsic_shuffle:
                  case nir_intrinsic_quad_broadcast:
                  case nir_intrinsic_quad_swap_horizontal:
                  case nir_intrinsic_quad_swap_vertical:
                  case nir_intrinsic_quad_swap_diagonal:
                  case nir_intrinsic_quad_swizzle_amd:
                  case nir_intrinsic_masked_swizzle_amd:
                  case nir_intrinsic_inclusive_scan:
                  case nir_intrinsic_exclusive_scan:
                     if (!ctx->divergent_vals[intrinsic->dest.ssa.index]) {
                        type = sgpr;
                     } else if (intrinsic->src[0].ssa->bit_size == 1) {
                        type = sgpr;
                        size = 2;
                     } else {
                        type = vgpr;
                     }
                     break;
                  case nir_intrinsic_load_view_index:
                     type = ctx->stage == MESA_SHADER_FRAGMENT ? vgpr : sgpr;
                     break;
                  case nir_intrinsic_load_front_face:
                  case nir_intrinsic_load_helper_invocation:
                  case nir_intrinsic_is_helper_invocation:
                     type = sgpr;
                     size = 2;
                     break;
                  case nir_intrinsic_reduce:
                     if (nir_intrinsic_cluster_size(intrinsic) == 0 ||
                         !ctx->divergent_vals[intrinsic->dest.ssa.index]) {
                        type = sgpr;
                     } else if (intrinsic->src[0].ssa->bit_size == 1) {
                        type = sgpr;
                        size = 2;
                     } else {
                        type = vgpr;
                     }
                     break;
                  case nir_intrinsic_load_ubo:
                  case nir_intrinsic_load_ssbo:
                  case nir_intrinsic_load_global:
                     type = ctx->divergent_vals[intrinsic->dest.ssa.index] ? vgpr : sgpr;
                     break;
                  /* due to copy propagation, the swizzled imov is removed if num dest components == 1 */
                  case nir_intrinsic_load_shared:
                     if (ctx->divergent_vals[intrinsic->dest.ssa.index])
                        type = vgpr;
                     else
                        type = sgpr;
                     break;
                  default:
                     for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs; i++) {
                        if (reg_class[intrinsic->src[i].ssa->index].type() == vgpr)
                           type = vgpr;
                     }
                     break;
               }
               reg_class[intrinsic->dest.ssa.index] = RegClass(type, size);

               switch(intrinsic->intrinsic) {
                  case nir_intrinsic_load_barycentric_sample:
                  case nir_intrinsic_load_barycentric_pixel:
                  case nir_intrinsic_load_barycentric_centroid:
                  case nir_intrinsic_load_barycentric_at_sample:
                  case nir_intrinsic_load_barycentric_at_offset: {
                     glsl_interp_mode mode = (glsl_interp_mode)nir_intrinsic_interp_mode(intrinsic);
                     ctx->fs_vgpr_args[get_interp_input(intrinsic->intrinsic, mode)] = true;
                     break;
                  }
                  case nir_intrinsic_load_front_face:
                     ctx->fs_vgpr_args[fs_input::front_face] = true;
                     break;
                  case nir_intrinsic_load_frag_coord:
                  case nir_intrinsic_load_sample_pos: {
                     uint8_t mask = nir_ssa_def_components_read(&intrinsic->dest.ssa);
                     for (unsigned i = 0; i < 4; i++) {
                        if (mask & (1 << i))
                           ctx->fs_vgpr_args[fs_input::frag_pos_0 + i] = true;

                     }
                     break;
                  }
                  case nir_intrinsic_load_sample_id:
                     ctx->fs_vgpr_args[fs_input::ancillary] = true;
                     break;
                  case nir_intrinsic_load_sample_mask_in:
                     ctx->fs_vgpr_args[fs_input::ancillary] = true;
                     ctx->fs_vgpr_args[fs_input::sample_coverage] = true;
                     break;
                  default:
                     break;
               }
               break;
            }
            case nir_instr_type_tex: {
               nir_tex_instr* tex = nir_instr_as_tex(instr);
               unsigned size = tex->dest.ssa.num_components;

               if (tex->dest.ssa.bit_size == 64)
                  size *= 2;
               if (tex->op == nir_texop_texture_samples)
                  assert(!ctx->divergent_vals[tex->dest.ssa.index]);
               if (ctx->divergent_vals[tex->dest.ssa.index])
                  reg_class[tex->dest.ssa.index] = RegClass(vgpr, size);
               else
                  reg_class[tex->dest.ssa.index] = RegClass(sgpr, size);
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
               reg_class[nir_instr_as_ssa_undef(instr)->def.index] = RegClass(sgpr, size);
               break;
            }
            case nir_instr_type_phi: {
               nir_phi_instr* phi = nir_instr_as_phi(instr);
               RegType type;
               unsigned size = phi->dest.ssa.num_components;

               if (phi->dest.ssa.bit_size == 1) {
                  assert(size == 1 && "multiple components not yet supported on boolean phis.");
                  type = sgpr;
                  size *= ctx->divergent_vals[phi->dest.ssa.index] ? 2 : 1;
                  reg_class[phi->dest.ssa.index] = RegClass(type, size);
                  break;
               }

               if (ctx->divergent_vals[phi->dest.ssa.index]) {
                  type = vgpr;
               } else {
                  type = sgpr;
                  nir_foreach_phi_src (src, phi) {
                     if (reg_class[src->src.ssa->index].type() == RegType::vgpr)
                        type = vgpr;
                     if (reg_class[src->src.ssa->index].type() == 0)
                        done = false;
                  }
               }

               size *= phi->dest.ssa.bit_size == 64 ? 2 : 1;
               RegClass rc = RegClass(type, size);
               if (rc != reg_class[phi->dest.ssa.index]) {
                  done = false;
               } else {
                  nir_foreach_phi_src(src, phi)
                     assert(reg_class[src->src.ssa->index].size() == rc.size());
               }
               reg_class[phi->dest.ssa.index] = rc;
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
   uint8_t remaining_sgprs;
   uint8_t user_sgpr_idx;
   bool need_ring_offsets;
   bool indirect_all_descriptor_sets;
};

static void allocate_inline_push_consts(isel_context *ctx,
                                        user_sgpr_info& user_sgpr_info)
{
   uint8_t remaining_sgprs = user_sgpr_info.remaining_sgprs;

   /* Only supported if shaders use push constants. */
   if (ctx->program->info->info.min_push_constant_used == UINT8_MAX)
      return;

   /* Only supported if shaders don't have indirect push constants. */
   if (ctx->program->info->info.has_indirect_push_constants)
      return;

   /* Only supported for 32-bit push constants. */
   //TODO: it's possible that some day, the load/store vectorization could make this inaccurate
   if (!ctx->program->info->info.has_only_32bit_push_constants)
      return;

   uint8_t num_push_consts =
      (ctx->program->info->info.max_push_constant_used -
       ctx->program->info->info.min_push_constant_used) / 4;

   /* Check if the number of user SGPRs is large enough. */
   if (num_push_consts < remaining_sgprs) {
      ctx->program->info->info.num_inline_push_consts = num_push_consts;
   } else {
      ctx->program->info->info.num_inline_push_consts = remaining_sgprs;
   }

   /* Clamp to the maximum number of allowed inlined push constants. */
   if (ctx->program->info->info.num_inline_push_consts > MAX_INLINE_PUSH_CONSTS)
      ctx->program->info->info.num_inline_push_consts = MAX_INLINE_PUSH_CONSTS;

   if (ctx->program->info->info.num_inline_push_consts == num_push_consts &&
       !ctx->program->info->info.loads_dynamic_offsets) {
      /* Disable the default push constants path if all constants are
       * inlined and if shaders don't use dynamic descriptors.
       */
      ctx->program->info->info.loads_push_constants = false;
      user_sgpr_info.num_sgpr--;
      user_sgpr_info.remaining_sgprs++;
   }

   ctx->program->info->info.base_inline_push_consts =
      ctx->program->info->info.min_push_constant_used / 4;

   user_sgpr_info.num_sgpr += ctx->program->info->info.num_inline_push_consts;
   user_sgpr_info.remaining_sgprs -= ctx->program->info->info.num_inline_push_consts;
}

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
      //user_sgpr_count += ctx->program->info->info.ps.needs_sample_positions;
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

   if (ctx->program->info->info.so.num_outputs)
      user_sgpr_count += 1; /* we use 32bit pointers */

   uint32_t available_sgprs = ctx->options->chip_class >= GFX9 && ctx->stage != MESA_SHADER_COMPUTE ? 32 : 16;
   uint32_t remaining_sgprs = available_sgprs - user_sgpr_count;
   uint32_t num_desc_set = util_bitcount(ctx->program->info->info.desc_set_used_mask);

   if (available_sgprs < user_sgpr_count + num_desc_set) {
      user_sgpr_info.indirect_all_descriptor_sets = true;
      user_sgpr_info.num_sgpr = user_sgpr_count + 1;
      user_sgpr_info.remaining_sgprs = remaining_sgprs - 1;
   } else {
      user_sgpr_info.num_sgpr = user_sgpr_count + num_desc_set;
      user_sgpr_info.remaining_sgprs = remaining_sgprs - num_desc_set;
   }

   allocate_inline_push_consts(ctx, user_sgpr_info);
}

#define MAX_ARGS 64
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
add_arg(arg_info *info, RegClass rc, Temp *param_ptr, unsigned reg)
{
   assert(info->count < MAX_ARGS);

   info->assign[info->count] = param_ptr;
   info->types[info->count] = rc;

   if (rc.type() == sgpr) {
      info->num_sgprs_used += rc.size();
      info->sgpr_count++;
      info->reg[info->count] = PhysReg{reg};
   } else {
      assert(rc.type() == vgpr);
      info->num_vgprs_used += rc.size();
      info->reg[info->count] = PhysReg{reg + 256};
   }
   info->count++;
}

static void
set_loc(struct radv_userdata_info *ud_info, uint8_t *sgpr_idx, uint8_t num_sgprs)
{
   ud_info->sgpr_idx = *sgpr_idx;
   ud_info->num_sgprs = num_sgprs;
   *sgpr_idx += num_sgprs;
}

static void
set_loc_shader(isel_context *ctx, int idx, uint8_t *sgpr_idx,
               uint8_t num_sgprs)
{
   struct radv_userdata_info *ud_info = &ctx->program->info->user_sgprs_locs.shader_data[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, num_sgprs);
}

static void
set_loc_shader_ptr(isel_context *ctx, int idx, uint8_t *sgpr_idx)
{
   bool use_32bit_pointers = idx != AC_UD_SCRATCH_RING_OFFSETS;

   set_loc_shader(ctx, idx, sgpr_idx, use_32bit_pointers ? 1 : 2);
}

static void
set_loc_desc(isel_context *ctx, int idx,  uint8_t *sgpr_idx)
{
   struct radv_userdata_locations *locs = &ctx->program->info->user_sgprs_locs;
   struct radv_userdata_info *ud_info = &locs->descriptor_sets[idx];
   assert(ud_info);

   set_loc(ud_info, sgpr_idx, 1);
   locs->descriptor_sets_enabled |= 1 << idx;
}

static void
declare_global_input_sgprs(isel_context *ctx,
                           /* bool has_previous_stage, gl_shader_stage previous_stage, */
                           user_sgpr_info *user_sgpr_info,
                           struct arg_info *args,
                           Temp *desc_sets)
{
   /* 1 for each descriptor set */
   if (!user_sgpr_info->indirect_all_descriptor_sets) {
      uint32_t mask = ctx->program->info->info.desc_set_used_mask;
      while (mask) {
         int i = u_bit_scan(&mask);
         add_arg(args, s1, &desc_sets[i], user_sgpr_info->user_sgpr_idx);
         set_loc_desc(ctx, i, &user_sgpr_info->user_sgpr_idx);
      }
      /* NIR->LLVM might have set this to true if RADV_DEBUG=compiletime */
      ctx->program->info->need_indirect_descriptor_sets = false;
   } else {
      add_arg(args, s1, desc_sets, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_INDIRECT_DESCRIPTOR_SETS, &user_sgpr_info->user_sgpr_idx);
      ctx->program->info->need_indirect_descriptor_sets = true;
   }

   if (ctx->program->info->info.loads_push_constants) {
      /* 1 for push constants and dynamic descriptors */
      add_arg(args, s1, &ctx->push_constants, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_PUSH_CONSTANTS, &user_sgpr_info->user_sgpr_idx);
   }

   if (ctx->program->info->info.num_inline_push_consts) {
      unsigned count = ctx->program->info->info.num_inline_push_consts;
      for (unsigned i = 0; i < count; i++)
         add_arg(args, s1, &ctx->inline_push_consts[i], user_sgpr_info->user_sgpr_idx + i);
      set_loc_shader(ctx, AC_UD_INLINE_PUSH_CONSTANTS, &user_sgpr_info->user_sgpr_idx, count);

      ctx->num_inline_push_consts = ctx->program->info->info.num_inline_push_consts;
      ctx->base_inline_push_consts = ctx->program->info->info.base_inline_push_consts;
   }

   if (ctx->program->info->info.so.num_outputs) {
      add_arg(args, s1, &ctx->streamout_buffers, user_sgpr_info->user_sgpr_idx);
      set_loc_shader_ptr(ctx, AC_UD_STREAMOUT_BUFFERS, &user_sgpr_info->user_sgpr_idx);
   }
}

static void
declare_vs_input_vgprs(isel_context *ctx, struct arg_info *args)
{
   unsigned vgpr_idx = 0;
   add_arg(args, v1, &ctx->vertex_id, vgpr_idx++);
/* if (!ctx->is_gs_copy_shader) */ {
      if (ctx->options->key.vs.out.as_ls) {
         add_arg(args, v1, &ctx->rel_auto_id, vgpr_idx++);
         add_arg(args, v1, &ctx->instance_id, vgpr_idx++);
      } else {
         add_arg(args, v1, &ctx->instance_id, vgpr_idx++);
         add_arg(args, v1, &ctx->vs_prim_id, vgpr_idx++);
      }
      add_arg(args, v1, NULL, vgpr_idx); /* unused */
   }
}

static void
declare_streamout_sgprs(isel_context *ctx, struct arg_info *args, unsigned *idx)
{
   /* Streamout SGPRs. */
   if (ctx->program->info->info.so.num_outputs) {
      assert(ctx->stage == MESA_SHADER_VERTEX ||
             ctx->stage == MESA_SHADER_TESS_EVAL);

      if (ctx->stage != MESA_SHADER_TESS_EVAL) {
         add_arg(args, s1, &ctx->streamout_config, (*idx)++);
      } else {
         args->assign[args->count - 1] = &ctx->streamout_config;
         args->types[args->count - 1] = s1;
      }

      add_arg(args, s1, &ctx->streamout_write_idx, (*idx)++);
   }

   /* A streamout buffer offset is loaded if the stride is non-zero. */
   for (unsigned i = 0; i < 4; i++) {
      if (!ctx->program->info->info.so.strides[i])
         continue;

      add_arg(args, s1, &ctx->streamout_offset[i], (*idx)++);
   }
}

static bool needs_view_index_sgpr(isel_context *ctx)
{
   switch (ctx->stage) {
   case MESA_SHADER_VERTEX:
      if (ctx->program->info->info.needs_multiview_view_index ||
          (!ctx->options->key.vs.out.as_es && !ctx->options->key.vs.out.as_ls && ctx->options->key.has_multiview_view_index))
         return true;
      break;
   case MESA_SHADER_TESS_EVAL:
      if (ctx->program->info->info.needs_multiview_view_index || (!ctx->options->key.tes.out.as_es && ctx->options->key.has_multiview_view_index))
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
   arg_info args = {};

   if (user_sgpr_info.need_ring_offsets/* && !ctx->options->supports_spill*/)
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

      assert(user_sgpr_info.user_sgpr_idx == user_sgpr_info.num_sgpr);
      unsigned idx = user_sgpr_info.user_sgpr_idx;
      if (ctx->options->key.vs.out.as_es)
         add_arg(&args, s1, &ctx->es2gs_offset, idx++);
      else
         declare_streamout_sgprs(ctx, &args, &idx);

      declare_vs_input_vgprs(ctx, &args);
      break;
   }
   case MESA_SHADER_FRAGMENT: {
      declare_global_input_sgprs(ctx, &user_sgpr_info, &args, ctx->descriptor_sets);

      assert(user_sgpr_info.user_sgpr_idx == user_sgpr_info.num_sgpr);
      add_arg(&args, s1, &ctx->prim_mask, user_sgpr_info.user_sgpr_idx);

      ctx->program->config->spi_ps_input_addr = 0;
      ctx->program->config->spi_ps_input_ena = 0;
      if (ctx->fs_vgpr_args[fs_input::persp_sample_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_sample_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_sample_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_SAMPLE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_SAMPLE_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::persp_center_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_center_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_center_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_CENTER_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_CENTER_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::persp_centroid_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_centroid_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::persp_centroid_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_CENTROID_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_CENTROID_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::persp_pull_model]) {
         add_arg(&args, v3, &ctx->fs_inputs[fs_input::persp_pull_model], vgpr_idx);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_PERSP_PULL_MODEL_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_PERSP_PULL_MODEL_ENA(1);
         vgpr_idx += 3;
      }
      if (ctx->fs_vgpr_args[fs_input::linear_sample_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_sample_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_sample_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_SAMPLE_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_SAMPLE_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::linear_center_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_center_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_center_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_CENTER_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_CENTER_ENA(1);
      }
      if (ctx->fs_vgpr_args[fs_input::linear_centroid_p1]) {
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_centroid_p1], vgpr_idx++);
         add_arg(&args, v1, &ctx->fs_inputs[fs_input::linear_centroid_p2], vgpr_idx++);
         ctx->program->config->spi_ps_input_addr |= S_0286CC_LINEAR_CENTROID_ENA(1);
         ctx->program->config->spi_ps_input_ena |= S_0286CC_LINEAR_CENTROID_ENA(1);
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

   aco_ptr<Pseudo_instruction> startpgm{create_instruction<Pseudo_instruction>(aco_opcode::p_startpgm, Format::PSEUDO, 0, args.count + 1)};
   for (unsigned i = 0; i < args.count; i++) {
      if (args.assign[i]) {
         *args.assign[i] = Temp{ctx->program->allocateId(), args.types[i]};
         startpgm->getDefinition(i) = Definition(*args.assign[i]);
         startpgm->getDefinition(i).setFixed(args.reg[i]);
      }
   }
   startpgm->getDefinition(args.count) = Definition{ctx->program->allocateId(), exec, s2};
   ctx->block->instructions.push_back(std::move(startpgm));
}

int
type_size(const struct glsl_type *type, bool bindless)
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

void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   *size = shared_var_size(type);
   *align = 1;
}

int
get_align(nir_variable_mode mode, bool is_store, unsigned bit_size, unsigned num_components)
{
   /* TODO: ACO doesn't have good support for non-32-bit reads/writes yet */
   if (bit_size != 32)
      return -1;

   switch (mode) {
   case nir_var_mem_ubo:
   case nir_var_mem_ssbo:
   case nir_var_mem_push_const:
   case nir_var_mem_shared:
      /* TODO: what are the alignment requirements for LDS? */
      return num_components <= 4 ? 4 : -1;
   default:
      return -1;
   }
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
setup_vs_variables(isel_context *ctx, nir_shader *nir)
{
   assert(!ctx->program->info->vs.as_ls);
   assert(!ctx->program->info->vs.as_es);

   nir_foreach_variable(variable, &nir->inputs)
   {
      variable->data.driver_location = variable->data.location * 4;
   }
   nir_foreach_variable(variable, &nir->outputs)
   {
      variable->data.driver_location = variable->data.location * 4;
   }

   radv_vs_output_info *outinfo = &ctx->program->info->vs.outinfo;

   memset(outinfo->vs_output_param_offset, AC_EXP_PARAM_UNDEFINED,
          sizeof(outinfo->vs_output_param_offset));

   ctx->num_clip_distances = 0;
   ctx->num_cull_distances = 0;
   ctx->needs_instance_id = ctx->program->info->info.vs.needs_instance_id;

   bool export_clip_dists = ctx->options->key.vs_common_out.export_clip_dists;

   outinfo->param_exports = 0;
   outinfo->writes_pointsize = false;
   outinfo->writes_layer = false;
   outinfo->writes_viewport_index = false;
   int pos_written = 0x1;

   nir_foreach_variable(variable, &nir->outputs)
   {
      int idx = variable->data.location;
      unsigned slots = variable->type->count_attribute_slots(false);
	   if (variable->data.compact) {
		   unsigned component_count = variable->data.location_frac + variable->type->length;
		   slots = (component_count + 3) / 4;
	   }

      if (idx >= VARYING_SLOT_VAR0 || idx == VARYING_SLOT_LAYER || idx == VARYING_SLOT_PRIMITIVE_ID ||
          ((idx == VARYING_SLOT_CLIP_DIST0 || idx == VARYING_SLOT_CLIP_DIST1) && export_clip_dists)) {
         for (unsigned i = 0; i < slots; i++) {
            if (outinfo->vs_output_param_offset[idx + i] == AC_EXP_PARAM_UNDEFINED)
               outinfo->vs_output_param_offset[idx + i] = outinfo->param_exports++;
         }
      }

      if (idx == VARYING_SLOT_PSIZ) {
         outinfo->writes_pointsize = true;
         pos_written |= 1 << 1;
      } else if (idx == VARYING_SLOT_LAYER) {
         outinfo->writes_layer = true;
         pos_written |= 1 << 1;
      } else if (idx == VARYING_SLOT_VIEWPORT) {
         outinfo->writes_viewport_index = true;
         pos_written |= 1 << 1;
      } else if (idx == VARYING_SLOT_CLIP_DIST0) {
         ctx->num_clip_distances = nir->info.clip_distance_array_size;
         ctx->num_cull_distances = nir->info.cull_distance_array_size;
      }
   }
   if (ctx->options->key.has_multiview_view_index) {
      assert(!outinfo->writes_layer);
      outinfo->writes_layer = true;
      outinfo->vs_output_param_offset[VARYING_SLOT_LAYER] = outinfo->param_exports++;
      pos_written |= 1 << 1;
   }

   assert(outinfo->vs_output_param_offset[VARYING_SLOT_PRIMITIVE_ID] == AC_EXP_PARAM_UNDEFINED);
   outinfo->export_prim_id = false;
   if (ctx->options->key.vs.out.export_prim_id) {
		outinfo->export_prim_id = true;
      outinfo->vs_output_param_offset[VARYING_SLOT_PRIMITIVE_ID] = outinfo->param_exports++;
   }

   assert(ctx->num_clip_distances + ctx->num_cull_distances <= 8);

   if (ctx->num_clip_distances + ctx->num_cull_distances > 0)
      pos_written |= 1 << 2;
   if (ctx->num_clip_distances + ctx->num_cull_distances > 4)
      pos_written |= 1 << 3;

   outinfo->pos_exports = util_bitcount(pos_written);

   outinfo->clip_dist_mask = (1 << ctx->num_clip_distances) - 1;
   outinfo->cull_dist_mask = (1 << ctx->num_cull_distances) - 1;
   outinfo->cull_dist_mask <<= ctx->num_clip_distances;

   ctx->program->info->vs.export_prim_id = ctx->options->key.vs.out.export_prim_id;
   ctx->program->info->vs.as_ls = false;
   ctx->program->info->vs.as_es = false;
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

      ctx->program->info->fs.can_discard = nir->info.fs.uses_discard;
      ctx->program->info->fs.early_fragment_test = nir->info.fs.early_fragment_tests;
      break;
   }
   case MESA_SHADER_COMPUTE: {
      unsigned lds_size_bytes = 0;
      nir_foreach_variable(variable, &nir->shared)
      {
         variable->data.driver_location = lds_size_bytes;
         lds_size_bytes += total_shared_var_size(variable->type);
      }
      unsigned lds_allocation_size_unit = 4 * 64;
      if (ctx->program->chip_class >= GFX7)
         lds_allocation_size_unit = 4 * 128;
      ctx->program->config->lds_size = (lds_size_bytes + lds_allocation_size_unit - 1) / lds_allocation_size_unit;
      ctx->program->info->cs.block_size[0] = nir->info.cs.local_size[0];
      ctx->program->info->cs.block_size[1] = nir->info.cs.local_size[1];
      ctx->program->info->cs.block_size[2] = nir->info.cs.local_size[2];
      break;
   }
   case MESA_SHADER_VERTEX: {
      setup_vs_variables(ctx, nir);
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
   program->family = options->family;
   program->sgpr_limit = options->chip_class >= GFX8 ? 102 : 104;
   if (options->family == CHIP_TONGA)
      program->sgpr_limit = 94; /* workaround hardware bug */

   program->stage = nir->info.stage;
   for (unsigned i = 0; i < RADV_UD_MAX_SETS; ++i)
      program->info->user_sgprs_locs.descriptor_sets[i].sgpr_idx = -1;
   for (unsigned i = 0; i < AC_UD_MAX_UD; ++i)
      program->info->user_sgprs_locs.shader_data[i].sgpr_idx = -1;

   isel_context ctx = {};
   ctx.program = program;
   ctx.options = options;
   ctx.stage = nir->info.stage;

   for (unsigned i = 0; i < fs_input::max_inputs; ++i)
      ctx.fs_inputs[i] = Temp(0, v1);
   ctx.fs_inputs[fs_input::persp_pull_model] = Temp(0, v3);
   for (unsigned i = 0; i < RADV_UD_MAX_SETS; ++i)
      ctx.descriptor_sets[i] = Temp(0, s1);
   for (unsigned i = 0; i < MAX_INLINE_PUSH_CONSTS; ++i)
      ctx.inline_push_consts[i] = Temp(0, s1);
   for (unsigned i = 0; i <= VARYING_SLOT_VAR31; ++i) {
      for (unsigned j = 0; j < 4; ++j)
         ctx.vs_output.outputs[i][j] = Temp(0, v1);
   }

   /* the variable setup has to be done before lower_io / CSE */
   setup_variables(&ctx, nir);

   /* optimize and lower memory operations */
   nir_lower_to_explicit(nir, nir_var_mem_shared, shared_var_info);
   if (nir_opt_load_store_vectorize(nir,
                                    (nir_variable_mode)(nir_var_mem_ssbo | nir_var_mem_ubo |
                                                        nir_var_mem_push_const | nir_var_mem_shared),
                                    NULL, get_align)) {
      nir_lower_alu_to_scalar(nir, NULL);
      nir_lower_pack(nir);
   }
   nir_lower_io(nir, (nir_variable_mode)(nir_var_shader_in | nir_var_shader_out), type_size, (nir_lower_io_options)0);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_global);
   nir_lower_explicit_io(nir, nir_var_mem_global, nir_address_format_64bit_global);

   /* lower ALU operations */
   nir_opt_idiv_const(nir, 32);
   nir_lower_idiv(nir, true);

   // TODO: implement logic64 in aco, it's more effective for sgprs
   nir_lower_int64(nir, (nir_lower_int64_options) (nir_lower_imul64 |
                                                   nir_lower_imul_high64 |
                                                   nir_lower_imul_2x32_64 |
                                                   nir_lower_divmod64 |
                                                   nir_lower_logic64 |
                                                   nir_lower_minmax64 |
                                                   nir_lower_iabs64 |
                                                   nir_lower_ineg64));

   /* optimize the lowered ALU operations */
   nir_copy_prop(nir);
   nir_opt_constant_folding(nir);
   nir_opt_algebraic(nir);
   nir_opt_algebraic_late(nir);
   nir_opt_constant_folding(nir);

   /* cleanup passes */
   nir_lower_load_const_to_scalar(nir);
   nir_opt_cse(nir);
   nir_to_lcssa(nir);
   nir_lower_phis_to_scalar(nir);
   nir_opt_dce(nir);
   nir_opt_shrink_load(nir);
   nir_move_options move_opts = (nir_move_options)(
      nir_move_const_undef | nir_move_load_ubo | nir_move_load_input | nir_move_comparisons);
   nir_opt_sink(nir, move_opts);
   nir_opt_move(nir, move_opts);

   nir_function_impl *func = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(func);

   if (options->dump_preoptir) {
      fprintf(stderr, "NIR shader before instruction selection:\n");
      nir_print_shader(nir, stderr);
   }
   ctx.divergent_vals = nir_divergence_analysis(nir);
   init_context(&ctx, func);

   ctx.block = ctx.program->create_and_insert_block();
   ctx.block->loop_nest_depth = 0;
   ctx.block->kind = block_kind_top_level;

   add_startpgm(&ctx);

   return ctx;
}

}
