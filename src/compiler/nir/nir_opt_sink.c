/*
 * Copyright © 2018 Red Hat
 * Copyright © 2019 Valve Corporation
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
 *    Rob Clark (robdclark@gmail.com>
 *    Daniel Schürmann (daniel.schuermann@campus.tu-berlin.de)
 *    Rhys Perry (pendingchaos02@gmail.com)
 *
 */

#include "nir.h"


/*
 * A simple pass that moves some instructions into the least common
 * anscestor of consuming instructions.
 */

static nir_loop *
get_loop(nir_block *block)
{
   nir_cf_node *node = &block->cf_node;
   while (node) {
      if (node->type == nir_cf_node_loop)
         return (nir_loop*)node;
      node = node->parent;
   }
   return NULL;
}

static bool
inside_loop(nir_block *block, nir_loop *loop)
{
   if (!loop)
      return true;

   nir_cf_node *node = &block->cf_node;
   while (node) {
      if (node->type == nir_cf_node_loop && (nir_loop*)node == loop)
         return true;
      node = node->parent;
   }
   return false;
}

/* iterate a ssa def's use's and try to find a more optimal block to
 * move it to, using the dominance tree.  In short, if all of the uses
 * are contained in a single block, the load will be moved there,
 * otherwise it will be move to the least common ancestor block of all
 * the uses
 */
static nir_block *
get_preferred_block(nir_ssa_def *def, bool loop_aware)
{
   nir_block *lca = NULL;

   /* hmm, probably ignore if-uses: */
   if (!list_empty(&def->if_uses))
      return NULL;

   nir_foreach_use(use, def) {
      nir_instr *instr = use->parent_instr;
      nir_block *use_block = instr->block;

      /*
       * Kind of an ugly special-case, but phi instructions
       * need to appear first in the block, so by definition
       * we can't move a load_immed into a block where it is
       * consumed by a phi instruction.  We could conceivably
       * move it into a dominator block.
       */
      if (instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         nir_block *phi_lca = NULL;
         nir_foreach_phi_src(src, phi) {
            if (&src->src == use)
               phi_lca = nir_dominance_lca(phi_lca, src->pred);
         }
         use_block = phi_lca;
      }

      if (loop_aware) {
         nir_loop *use_loop = get_loop(use_block);
         if (!inside_loop(def->parent_instr->block, use_loop)) {
            use_block = nir_block_cf_tree_prev(nir_loop_first_block(use_loop));
            assert(nir_block_dominates(def->parent_instr->block, use_block));
         }
      }

      lca = nir_dominance_lca(lca, use_block);
   }

   return lca;
}

/* insert before first non-phi instruction: */
static void
insert_after_phi(nir_instr *instr, nir_block *block)
{
   nir_foreach_instr(instr2, block) {
      if (instr2->type == nir_instr_type_phi)
         continue;

      exec_node_insert_node_before(&instr2->node,
                                   &instr->node);

      return;
   }

   /* if haven't inserted it, push to tail (ie. empty block or possibly
    * a block only containing phi's?)
    */
   exec_list_push_tail(&block->instr_list, &instr->node);
}

static nir_ssa_def *
get_move_def(nir_instr *instr, bool sink_intrinsics) {
   if (instr->type == nir_instr_type_load_const) {
      return &nir_instr_as_load_const(instr)->def;
   } else if (instr->type == nir_instr_type_intrinsic && sink_intrinsics) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      return intrin->intrinsic == nir_intrinsic_load_interpolated_input ||
             intrin->intrinsic == nir_intrinsic_load_ubo ? &intrin->dest.ssa : NULL;
   } else if (instr->type == nir_instr_type_ssa_undef) {
      return &nir_instr_as_ssa_undef(instr)->def;
   } else {
      return NULL;
   }
}

bool
nir_opt_sink(nir_shader *shader, bool sink_intrinsics)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_metadata_require(function->impl,
                           nir_metadata_block_index | nir_metadata_dominance);

      nir_foreach_block_reverse(block, function->impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            nir_ssa_def *def = get_move_def(instr, sink_intrinsics);
            if (!def)
               continue;

            nir_block *use_block =
                  get_preferred_block(def, instr->type == nir_instr_type_intrinsic);

            if (!use_block)
               continue;

            if (use_block == instr->block)
               continue;

            exec_node_remove(&instr->node);

            insert_after_phi(instr, use_block);

            instr->block = use_block;

            progress = true;
         }
      }

      nir_metadata_preserve(function->impl,
                            nir_metadata_block_index | nir_metadata_dominance);
   }

   return progress;
}
