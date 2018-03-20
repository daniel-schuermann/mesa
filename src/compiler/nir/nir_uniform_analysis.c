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

/* This pass computes for each ssa definition if it is uniform.
 * That is, the variable has the same value for all invocations
 * of the group.
 */


static void visit_alu(bool *uniform, nir_alu_instr *instr)
{
   unsigned num_src = nir_op_infos[instr->op].num_inputs;
   for (unsigned i = 0; i < num_src; i++) {
      if (!uniform[instr->src[i].src.ssa->index]) {
         uniform[instr->dest.dest.ssa.index] = false;
         return;
      }
   }
   uniform[instr->dest.dest.ssa.index] = true;
}

static void visit_intrinsic(bool *uniform, nir_intrinsic_instr *instr)
{
   if (!nir_intrinsic_infos[instr->intrinsic].has_dest)
      return;
   
   bool is_uniform;
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
         is_uniform = true;
      default:
         is_uniform = false;
   }
   uniform[instr->dest.ssa.index] = is_uniform;

}

static void visit_tex(bool *uniform, nir_tex_instr *instr)
{
   /* TODO: */
   uniform[instr->dest.ssa.index] = false;
}

static bool is_in_uniform_cf(bool *uniform, nir_block *block)
{
   nir_cf_node *parent = block->cf_node.parent;
   if (parent->type == nir_cf_node_if) {
      nir_if *if_node = nir_cf_node_as_if(parent);
      if (!uniform[if_node->condition.ssa->index])
         return false;
   }
   if (parent->type == nir_cf_node_loop) {
      nir_loop_info *info = nir_cf_node_as_loop(parent)->info;
      list_for_each_entry(nir_loop_terminator, terminator,
              &info->loop_terminator_list, loop_terminator_link) {
         if (!uniform[terminator->nif->condition.ssa->index])
            return false;
      }
   }
   return true;
}

static void visit_phi(bool *uniform, nir_phi_instr *instr)
{
   /* Phi instructions are only uniform if all
    * src instructions are uniform and are computed
    * in uniform control flow
    */
   nir_foreach_phi_src(src, instr) {
      if (!(uniform[src->src.ssa->index] && is_in_uniform_cf(uniform, src->pred))) {
         uniform[instr->dest.ssa.index] = false;
         return;
      }
   }
   uniform[instr->dest.ssa.index] = true;
}

static void visit_parallel_copy(bool *uniform, nir_parallel_copy_instr *instr)
{
   nir_foreach_parallel_copy_entry(entry, instr) {
      uniform[entry->dest.ssa.index] = uniform[entry->src.ssa->index];
   }
}

static void visit_load_const(bool *uniform, nir_load_const_instr *instr)
{
   uniform[instr->def.index] = true;
}

static void visit_ssa_undef(bool *uniform, nir_ssa_undef_instr *instr)
{
   /* better save than sorry... */
   uniform[instr->def.index] = false;
}



bool* nir_uniform_analysis(nir_shader *shader)
{
   
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   bool *t = rzalloc_array(shader, bool, impl->ssa_alloc);
   
   nir_foreach_block(block, impl) {
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