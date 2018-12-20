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

static nir_block *
find_lca(nir_ssa_def *def)
{
   nir_block *lca = NULL;
   nir_block *def_block = def->parent_instr->block;

   /* hmm, probably ignore if-uses: */
   if (!list_empty(&def->if_uses))
      return def_block;

   nir_foreach_use(use, def) {
      nir_instr *instr = use->parent_instr;
      nir_block *use_block = instr->block;
      if (use_block == def_block)
         return def_block;

      /*
       * Kind of an ugly special-case, but phi instructions
       * need to appear first in the block, so by definition
       * we can't move a load_immed into a block where it is
       * consumed by a phi instruction.  We could conceivably
       * move it into a dominator block.
       */
      if (instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         nir_foreach_phi_src(src, phi) {
            if (&src->src == use) {
               use_block = src->pred;
               break;
            }
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

static bool
nir_opt_sink_block(nir_block *block)
{
   bool progress = false;

   nir_foreach_instr_reverse_safe(instr, block) {
      nir_ssa_def *def;

      switch (instr->type) {
      case nir_instr_type_phi:
      case nir_instr_type_call:
      case nir_instr_type_jump:
      case nir_instr_type_parallel_copy:
      case nir_instr_type_alu:
         continue;
      case nir_instr_type_ssa_undef: {
         nir_ssa_undef_instr *undef = nir_instr_as_ssa_undef(instr);
         def = &undef->def;
         break;
      }
      case nir_instr_type_load_const: {
         nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
         def = &load_const->def;
         break;
      }
      case nir_instr_type_tex:
         //TODO
         continue;
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (!nir_intrinsic_infos[intrin->intrinsic].has_dest)
            continue;
         def = &intrin->dest.ssa;
         if (intrin->intrinsic == nir_intrinsic_load_interpolated_input ||
             intrin->intrinsic == nir_intrinsic_load_ubo)
            break;
         //TODO some can directly go on, some need beneficial test
         continue;
      }
      case nir_instr_type_deref:
         continue;
      default:
         nir_print_instr (instr, stderr);
         unreachable("");
      }

      nir_block *lca = find_lca(def);
      // TODO: check that lca is not in a loop when original block isn't
      if (lca == instr->block) {
         continue;
      }

      exec_node_remove(&instr->node);
      insert_after_phi(instr, lca);
      instr->block = lca;

      progress = true;
   }

   return progress;
}

static bool
nir_opt_sink_impl(nir_function_impl *impl)
{
   nir_metadata_require(impl,
                        nir_metadata_dominance |
                        nir_metadata_block_index | nir_metadata_loop_analysis);
   bool progress = false;

   nir_foreach_block_reverse(block, impl)
      progress |= nir_opt_sink_block(block);

   nir_metadata_preserve(impl,
                         nir_metadata_dominance |
                         nir_metadata_block_index | nir_metadata_loop_analysis);
   return progress;
}

bool
nir_opt_sink(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_sink_impl(function->impl);
   }

   return progress;
}

