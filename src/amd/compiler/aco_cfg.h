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

#ifndef ACO_CFG_H
#define ACO_CFG_H

typedef enum {
   aco_cf_node_block,
   aco_cf_node_if,
   aco_cf_node_loop,
} aco_cf_node_type;

typedef struct aco_cf_node {
   struct exec_node node;
   aco_cf_node_type type;
   struct aco_cf_node *parent;
} aco_cf_node;

typedef struct aco_block {
   aco_cf_node cf_node;

   struct exec_list instr_list;

   /** generic block index; */
   unsigned index;

   /*
    * Each block can only have up to 2 successors, so we put them in a simple
    * array - no need for anything more complicated.
    */
   struct aco_block *successors[2];
   struct aco_block *linear_successors[2];

   /* block predecessors in the CFG */
   struct exec_list predecessors;
   struct exec_list linear_predecessors;

} aco_block;

#endif /* ACO_CFG_H */

