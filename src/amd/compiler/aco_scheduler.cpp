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
#include <unordered_set>
#include <unordered_map>

namespace aco {

struct Node {
   std::unique_ptr<Instruction> instr;
   std::unordered_set<Node*> children;
   int succ_count = 0;
   int reg = 0;
   int est = 0;
   int parent_idx = 0;
   int latency = 0;
   Node* imm_dep = nullptr;

   Node(std::unique_ptr<Instruction> instr) : instr(std::move(instr)) {};
};

void schedule_block(Block* block)
{
   std::vector<Node> DG;
   DG.reserve(block->instructions.size());
   std::unordered_map<unsigned, Node*> map;
   Node* latest_scc_write = nullptr;

   unsigned j = 0;
   while (j < block->instructions.size()) {
      if (block->instructions[j]->opcode == aco_opcode::p_logical_start)
         break;
      j++;
   }
   if (j >= block->instructions.size())
      return;

   j++;

   while (block->instructions[j]->opcode != aco_opcode::p_logical_end)
   {
      DG.emplace_back(std::move(block->instructions[j]));
      Node* node = &DG.back();
      for (unsigned i = 0; i < node->instr->num_definitions; i++) {
         map.emplace(node->instr->getDefinition(i).tempId(), node);
         if (node->instr->getDefinition(i).isFixed() &&
             node->instr->getDefinition(i).physReg().reg == 253) {
            if (latest_scc_write) {
               auto test = node->children.emplace(latest_scc_write);
               if (test.second)
                  latest_scc_write->succ_count++;
            }

            latest_scc_write = node;
         }
      }

      for (unsigned i = 0; i < node->instr->num_operands; i++) {
         Operand op = node->instr->getOperand(i);
         if (!op.isTemp())
            continue;
         std::unordered_map<unsigned, Node*>::iterator it = map.find(op.tempId());
         if (it != map.end()) {
            auto test = node->children.emplace(it->second);
            if (test.second)
               it->second->succ_count++;
            /* that's a workaround to make the def of an scc temp the immediate predecessor */
            if (op.isFixed() && op.physReg().reg == 253) {
               node->imm_dep = latest_scc_write;
               auto test = node->children.emplace(latest_scc_write);
               if (test.second)
                  latest_scc_write->succ_count++;
               latest_scc_write = node;
            }
         }
      }
      j++;
   }

   std::unordered_set<Node*> ready;
   for (Node& node : DG)
   {
      /* calculate register requirement */
      node.reg = node.instr->isSALU() || node.instr->format == Format::SMEM ? 0 : 1;
      int k = 0;
      int i = -1;
      for (Node* child : node.children) {
         if (i == child->reg) {
            k++;
         } else if (child->reg > i) {
            i = child->reg;
            k = 0;
         }
         node.reg = i + k;
      }

      /* calculate EST */
      node.est = 1;
      for (Node* child : node.children) {
         int time = child->instr->format == Format::SMEM ? 15 : 1;
         time = child->instr->format == Format::MIMG ? 8 : time;
         time = child->instr->format == Format::MUBUF ? 8 : time;
         node.est = std::max(node.est, time + child->est);
      }
      node.est -= (uint32_t) node.instr->format & (uint32_t) Format::DPP ? -2 : 0;

      /* calculate latency */
      node.latency = node.instr->format == Format::SMEM ? 15 : 1;
      node.latency = node.instr->format == Format::MIMG ? 8 : node.latency;
      node.latency = node.instr->format == Format::MUBUF ? 8 : node.latency;
      node.latency = node.instr->isSALU() ? 0 : node.latency;
      /* check if ready */
      if (node.succ_count == 0)
         ready.insert(&node);
   }

   j--;

   while (!ready.empty()) {
      /* find candidate */
      std::unordered_set<Node*>::iterator it = ready.begin();
      std::unordered_set<Node*>::iterator select = it++;
      int lowest = (*select)->reg + (*select)->parent_idx - (*select)->est + (*select)->latency;
      while (it != ready.end()) {
         int candidate = (*it)->reg + ((*it)->parent_idx) - (*it)->est + (*it)->latency;
         if (candidate < lowest) {
            select = it;
            lowest = candidate;
         }
         ++it;
      }
      Node* next = *select;
      ready.erase(select);
      while (true) {
         /* add children to ready list */
         for (Node* child : next->children) {
            child->parent_idx = j;
            child->succ_count--;
            if (child->succ_count <= 0)
               ready.insert(child);
         }
         block->instructions[j--] = std::move(next->instr);

         if (next->imm_dep) {
            next = next->imm_dep;
            assert(ready.find(next) != ready.end());
            ready.erase(next);
         } else {
            break;
         }
      }
   }
   return;
}

void schedule_program(Program* program)
{
   for (auto&& block : program->blocks)
      schedule_block(block.get());
}

}
 
