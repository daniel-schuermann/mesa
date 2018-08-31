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

#include "aco_ir.h"
#include <vector>

/*
 * Implements the algorithms for computing the dominator tree from
 * "A Simple, Fast Dominance Algorithm" by Cooper, Harvey, and Kennedy.
 */

namespace aco {

inline
int update_idom(int pred_idx, int new_idom, std::vector<int>& doms)
{
   if (doms[pred_idx] != -1) {
      if (new_idom == -1) {
         return pred_idx;
      } else {
         while (pred_idx != new_idom) {
            while (pred_idx > new_idom)
               pred_idx = doms[pred_idx];
            while (new_idom > pred_idx)
               new_idom = doms[new_idom];
         }
      }
   }
   return new_idom;
}

std::pair<std::vector<int>, std::vector<int>> dominator_tree(Program* program)
{
   std::vector<int> logical_doms(program->blocks.size());
   std::vector<int> linear_doms(program->blocks.size());
   for (unsigned i = 0; i < program->blocks.size(); i++) {
      logical_doms[i] = -1;
      linear_doms[i] = -1;
   }
   logical_doms[0] = 0;
   linear_doms[0] = 0;

   bool changed = true;
   while (changed)
   {
      for (std::vector<std::unique_ptr<Block>>::iterator it = ++program->blocks.begin(); it != program->blocks.end(); ++it) {
         int new_logical_idom = -1;
         int new_linear_idom = -1;
         for (Block* pred : (*it)->logical_predecessors)
            new_logical_idom = update_idom(pred->index, new_logical_idom, logical_doms);
         for (Block* pred : (*it)->linear_predecessors)
            new_linear_idom = update_idom(pred->index, new_linear_idom, linear_doms);

         changed = logical_doms[(*it)->index] != new_logical_idom ||
                   linear_doms[(*it)->index] != new_linear_idom;
         logical_doms[(*it)->index] = new_logical_idom;
         linear_doms[(*it)->index] = new_linear_idom;
      }
   }

   return std::make_pair(logical_doms, linear_doms);
}

}
