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

#include "nir.h"
#include "nir_worklist.h"

/* This pass computes for each ssa definition if it is uniform.
 * That is, the variable has the same value for all invocations
 * of the group.
 *
 * This algorithm implements "The Simple Divergence Analysis" from
 * Diogo Sampaio, Rafael De Souza, Sylvain Collange, Fernando Magno Quintão Pereira.
 * Divergence Analysis.  ACM Transactions on Programming Languages and Systems (TOPLAS),
 * ACM, 2013, 35 (4), pp.13:1-13:36. <10.1145/2523815>. <hal-00909072v2>
 */


static bool
alu_src_is_divergent(bool *divergent, nir_alu_src src, unsigned num_input_components)
{
   /* If the alu src is swizzled and defined by a vec-instruction,
    * we can check if the originating value is non-divergent. */
   if (num_input_components == 1 &&
       src.src.ssa->num_components != 1 &&
       src.src.parent_instr->type == nir_instr_type_alu) {
      nir_alu_instr *parent = nir_instr_as_alu(src.src.parent_instr);

      switch(parent->op) {
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
         return divergent[parent->src[src.swizzle[0]].src.ssa->index];

      default:
         break;
      }
   }

   return divergent[src.src.ssa->index];
}

static bool
visit_alu(bool *divergent, nir_alu_instr *instr)
{
   if (divergent[instr->dest.dest.ssa.index])
      return false;

   unsigned num_src = nir_op_infos[instr->op].num_inputs;

   for (unsigned i = 0; i < num_src; i++) {
      if (alu_src_is_divergent(divergent, instr->src[i], nir_op_infos[instr->op].input_sizes[i])) {
         divergent[instr->dest.dest.ssa.index] = true;
         return true;
      }
   }

   divergent[instr->dest.dest.ssa.index] = false;
   return false;
}

static bool
visit_intrinsic(bool *divergent, nir_intrinsic_instr *instr)
{
   if (!nir_intrinsic_infos[instr->intrinsic].has_dest)
      return false;

   if (divergent[instr->dest.ssa.index])
      return false;

   bool is_divergent = false;
   switch (instr->intrinsic) {
   /* TODO: load_shared_var */
   /*       load_uniform etc.*/
   case nir_intrinsic_shader_clock:
   case nir_intrinsic_ballot:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_reduce:
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_load_work_group_id:
   case nir_intrinsic_load_num_work_groups:
   case nir_intrinsic_get_buffer_size:
      is_divergent = false;
      break;

   case nir_intrinsic_load_ubo:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_shared:
      for (unsigned i = 0; i < nir_intrinsic_infos[instr->intrinsic].num_srcs; i++) {
         if (divergent[instr->src[i].ssa->index]) {
            is_divergent = true;
            break;
         }
      }
      break;

   case nir_intrinsic_load_deref: {
      nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));
      switch (var->data.mode) {
      case nir_var_shared:
         is_divergent = divergent[instr->src[0].ssa->index];
         break;
      default:
         is_divergent = true;
         break;
      }
      break;
   }
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
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
   default:
      is_divergent = true;
      break;
   }

   divergent[instr->dest.ssa.index] = is_divergent;
   return is_divergent;
}

static bool
visit_tex(bool *divergent, nir_tex_instr *instr)
{
   if (divergent[instr->dest.ssa.index])
      return false;

   bool is_divergent = false;

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_coord:
         is_divergent |= divergent[instr->src[i].src.ssa->index];
         break;
      default:
         break;
      }
   }

   divergent[instr->dest.ssa.index] = is_divergent;
   return is_divergent;
}

static bool
visit_phi(bool *divergent, nir_phi_instr *instr)
{
   /* There are 3 types of phi instructions:
    * (1) gamma: represent the joining point of different paths
    *     created by an “if-then-else” branch.
    *     The resulting value is divergent iff the branch condition
    *     or any of the source values is divergent.
    *
    * (2) mu: which only exist at loop headers,
    *     merge initial and loop-carried values.
    *     The resulting value is divergent iff any source value
    *     is divergent or a divergent loop continue condition
    *     is associated with a different ssa-def.
    *
    * (3) eta: represent values that leave a loop.
    *     The resulting value is divergent iff any loop exit condition
    *     or source value is divergent.
    */

   if (divergent[instr->dest.ssa.index])
      return false;

   unsigned non_undef = 0;
   nir_foreach_phi_src(src, instr) {
      /* if any source value is divergent, the resulting value is divergent */
      if (divergent[src->src.ssa->index]) {
         divergent[instr->dest.ssa.index] = true;
         return true;
      }

      /* if all values but one are undef, the resulting value is uniform */
      if (src->src.ssa->parent_instr->type != nir_instr_type_ssa_undef)
         non_undef += 1;
   }

   if (non_undef <= 1) {
      assert(divergent[instr->dest.ssa.index] == false);
      return false;
   }

   nir_cf_node *prev = nir_cf_node_prev(&instr->instr.block->cf_node);

   if (!prev) {
      /* mu: if no predecessor node exists, the phi must be at a loop header */

      /* first, find the two unconditional ssa-defs (from incoming- and back-edge) */
      nir_loop *loop = nir_cf_node_as_loop(instr->instr.block->cf_node.parent);
      prev = nir_cf_node_prev(instr->instr.block->cf_node.parent);
      unsigned unconditional[2];
      unsigned idx = 0;

      nir_foreach_phi_src(src, instr) {
         if (src->pred == nir_loop_last_block(loop) ||
             src->pred == nir_cf_node_as_block(prev))
            unconditional[idx++] = src->src.ssa->index;
      }
      assert(idx == 2);

      /* check if the loop-carried values come from a different ssa-def
       * and the corresponding condition is divergent.
       */
      nir_foreach_phi_src(src, instr) {
         if (src->src.ssa->index == unconditional[0] ||
             src->src.ssa->index == unconditional[1])
            continue;

         nir_cf_node *current = src->pred->cf_node.parent;

         /* check recursively the conditions if any is divergent */
         while (current->type != nir_cf_node_loop) {
            if (current->type == nir_cf_node_if) {
               nir_if *if_node = nir_cf_node_as_if(current);
               if (divergent[if_node->condition.ssa->index]) {
                  divergent[instr->dest.ssa.index] = true;
                  return true;
               }
            }
            current = current->parent;
         }
      }

   } else if (prev->type == nir_cf_node_if) {
      /* gamma: check if the condition is divergent */
      nir_if *if_node = nir_cf_node_as_if(prev);
      if (divergent[if_node->condition.ssa->index]) {
         divergent[instr->dest.ssa.index] = true;
         return true;
      }

   } else {
      /* eta: check if any loop exit condition is divergent */
      assert(prev->type == nir_cf_node_loop);
      nir_foreach_phi_src(src, instr) {
         nir_cf_node *current = src->pred->cf_node.parent;
         assert(current->type == nir_cf_node_if);

         /* check recursively the conditions if any is divergent */
         while (current->type != nir_cf_node_loop) {
            if (current->type == nir_cf_node_if) {
               nir_if *if_node = nir_cf_node_as_if(current);
               if (divergent[if_node->condition.ssa->index]) {
                  divergent[instr->dest.ssa.index] = true;
                  return true;
               }
            }
            current = current->parent;
         }
      }
   }

   divergent[instr->dest.ssa.index] = false;
   return false;
}

static bool
visit_parallel_copy(bool *divergent, nir_parallel_copy_instr *instr)
{
   bool has_changed = false;

   nir_foreach_parallel_copy_entry(entry, instr) {
      if (divergent[entry->dest.ssa.index])
         continue;
      divergent[entry->dest.ssa.index] = divergent[entry->src.ssa->index];
      if (divergent[entry->dest.ssa.index])
         has_changed = true;
   }

   return has_changed;
}

static bool
visit_load_const(bool *divergent, nir_load_const_instr *instr)
{
   divergent[instr->def.index] = false;
   return false;
}

static bool
visit_ssa_undef(bool *divergent, nir_ssa_undef_instr *instr)
{
   divergent[instr->def.index] = false;
   return false;
}

static bool
visit_deref(bool *divergent, nir_deref_instr *instr)
{
   nir_foreach_use(src, &instr->dest.ssa)
   {
      if (src->parent_instr->type != nir_instr_type_tex) {
         divergent[instr->dest.ssa.index] = false;
         return false;
      }
   }

   bool before = divergent[instr->dest.ssa.index];
   divergent[instr->dest.ssa.index] = true;
   return !before;
}

bool*
nir_divergence_analysis(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   bool *t = rzalloc_array(shader, bool, impl->ssa_alloc);
   nir_block_worklist worklist;
   nir_block_worklist_init(&worklist, impl->num_blocks, NULL);
   nir_block_worklist_add_all(&worklist, impl);

   while (!nir_block_worklist_is_empty(&worklist)) {
      nir_block *block = nir_block_worklist_pop_head(&worklist);
      bool has_changed = false;

      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
            has_changed |= visit_alu(t, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_intrinsic:
            has_changed |= visit_intrinsic(t, nir_instr_as_intrinsic(instr));
            break;
         case nir_instr_type_tex:
            has_changed |= visit_tex(t, nir_instr_as_tex(instr));
            break;
         case nir_instr_type_phi:
            has_changed |= visit_phi(t, nir_instr_as_phi(instr));
            break;
         case nir_instr_type_parallel_copy:
            has_changed |= visit_parallel_copy(t, nir_instr_as_parallel_copy(instr));
            break;
         case nir_instr_type_load_const:
            has_changed |= visit_load_const(t, nir_instr_as_load_const(instr));
            break;
         case nir_instr_type_ssa_undef:
            has_changed |= visit_ssa_undef(t, nir_instr_as_ssa_undef(instr));
            break;
         case nir_instr_type_deref:
            has_changed |= visit_deref(t, nir_instr_as_deref(instr));
            break;
         case nir_instr_type_jump:
            break;
         case nir_instr_type_call:
            assert(false);
         default:
            unreachable("Invalid instruction type");
            break;
         }
      }

      if (has_changed) {
         // FIXME: this is quite inefficient!
         nir_block_worklist_add_all(&worklist, impl);
      }
   }

   return t;
}
