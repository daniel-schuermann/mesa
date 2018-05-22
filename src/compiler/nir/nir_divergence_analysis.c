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


static bool alu_src_is_divergent(bool *divergent, nir_alu_src src, unsigned num_input_components)
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
         case nir_op_vec4: {
            if (divergent[parent->src[src.swizzle[0]].src.ssa->index])
               return true;
            return false;
         }
         default:
            break;
      }
   }
   return divergent[src.src.ssa->index];
}

static void visit_alu(bool *divergent, nir_alu_instr *instr)
{
   unsigned num_src = nir_op_infos[instr->op].num_inputs;
   for (unsigned i = 0; i < num_src; i++) {
      if (alu_src_is_divergent(divergent, instr->src[i], nir_op_infos[instr->op].input_sizes[i])) {
         divergent[instr->dest.dest.ssa.index] = true;
         return;
      }
   }
   divergent[instr->dest.dest.ssa.index] = false;
}

static void visit_intrinsic(bool *divergent, nir_intrinsic_instr *instr)
{
   if (!nir_intrinsic_infos[instr->intrinsic].has_dest)
      return;

   bool is_divergent;
   switch (instr->intrinsic) {
      /* TODO: load_ubo (if index&buffer uniform) */
      /*       load_shared_var */
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
         is_divergent = false;
         break;
      default:
         is_divergent = true;
         break;
   }
   divergent[instr->dest.ssa.index] = is_divergent;
}

static void visit_tex(bool *divergent, nir_tex_instr *instr)
{
   /* TODO: */
   divergent[instr->dest.ssa.index] = true;
}


/** There are 3 types of phi instructions:
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
static void visit_phi(bool *divergent, nir_phi_instr *instr)
{

   /* if any source value is divergent, the resulting value is divergent */
   nir_foreach_phi_src(src, instr) {
      if (divergent[src->src.ssa->index]) {
         divergent[instr->dest.ssa.index] = true;
         return;
      }
   }

   nir_cf_node *prev = nir_cf_node_prev(&instr->instr.block->cf_node);

   /* mu: if no predecessor node exists, the phi must be at a loop header */
   if (!prev) {
      /* find the two unconditional ssa-defs (the incoming value and the back-edge) */
      unsigned unconditional[2];
      unsigned idx = 0;
      nir_foreach_phi_src(src, instr) {
         if (src->pred->cf_node.parent->type != nir_cf_node_if)
            unconditional[idx++] = src->src.ssa->index;
      }
      assert(idx == 2);

      /* check if the loop-carried values come from a different ssa-def
       * and the corresponding condition is divergent. */
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
                  return;
               }
            }
            current = current->parent;
         }
      }
   }

   /* gamma: check if the condition is divergent */
   else if (prev->type == nir_cf_node_if) {
      nir_if *if_node = nir_cf_node_as_if(prev);
      if (divergent[if_node->condition.ssa->index]) {
         divergent[instr->dest.ssa.index] = true;
         return;
      }
   }

   /* eta: check if any loop exit condition is divergent */
   else {
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
                  return;
               }
            }
            current = current->parent;
         }
      }
   }
   divergent[instr->dest.ssa.index] = false;
}

static void visit_parallel_copy(bool *divergent, nir_parallel_copy_instr *instr)
{
   nir_foreach_parallel_copy_entry(entry, instr) {
      divergent[entry->dest.ssa.index] = divergent[entry->src.ssa->index];
   }
}

static void visit_load_const(bool *divergent, nir_load_const_instr *instr)
{
   divergent[instr->def.index] = false;
}

static void visit_ssa_undef(bool *divergent, nir_ssa_undef_instr *instr)
{
   divergent[instr->def.index] = false;
}



bool* nir_divergence_analysis(nir_shader *shader)
{

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_block_worklist worklist;
   nir_block_worklist_init(&worklist, impl->num_blocks, NULL);
   /* we run this analysis twice as we do an optimistic approach
    * and assume all values to be uniform.
    * This algorithm converges after two passes. */
   nir_block_worklist_add_all(&worklist, impl);
   nir_block_worklist_add_all(&worklist, impl);

   bool *t = rzalloc_array(shader, bool, impl->ssa_alloc);

   while (!nir_block_worklist_is_empty(&worklist)) {

      nir_block *block = nir_block_worklist_pop_head(&worklist);

      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
            visit_alu(t, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_intrinsic:
            visit_intrinsic(t, nir_instr_as_intrinsic(instr));
            break;
         case nir_instr_type_tex:
            visit_tex(t, nir_instr_as_tex(instr));
            break;
         case nir_instr_type_phi:
            visit_phi(t, nir_instr_as_phi(instr));
            break;
         case nir_instr_type_parallel_copy:
            visit_parallel_copy(t, nir_instr_as_parallel_copy(instr));
            break;
         case nir_instr_type_load_const:
            visit_load_const(t, nir_instr_as_load_const(instr));
            break;
         case nir_instr_type_ssa_undef:
            visit_ssa_undef(t, nir_instr_as_ssa_undef(instr));
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
   }
   return t;
}
