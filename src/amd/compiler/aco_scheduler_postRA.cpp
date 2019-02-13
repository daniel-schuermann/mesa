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

#include <unordered_map>
#include <unordered_set>
#include <set>

#include "aco_ir.h"

namespace aco {

struct Node {
   int index;
   int priority;
   int nops = 0;
   std::unordered_set<Node*> successors;
   /* pairs of predecessor (node, num wait_states) */
   std::vector<std::pair<Node*,int>> predecessors;
   bool scheduled = false;

   Node(int i) : index(i) {}

   bool operator ==(const struct Node& other) const
   {
      return index == other.index;
   }

   bool operator <(const struct Node& other) const
   {
      return index < other.index;
   }
};

struct sched_ctx {
   unsigned current_index = 0;
   std::vector<Node> nodes;
   std::set<Node*> candidates;
   /* hashtable from PhysReg to Node */
   std::unordered_map<unsigned, Node*> def_table;

   /* here we can maintain information about the functional units */
   sched_ctx(unsigned num_instr)
   {
      nodes.reserve(num_instr);
   }
};

bool reads_exec_implicitly(Instruction* instr)
{
   if (instr->isSALU()) {
      switch (instr->opcode) {
         case aco_opcode::s_or_saveexec_b64:
         case aco_opcode::s_and_saveexec_b64:
         case aco_opcode::s_xor_saveexec_b64:
         case aco_opcode::s_andn2_saveexec_b64:
         case aco_opcode::s_orn2_saveexec_b64:
         case aco_opcode::s_nand_saveexec_b64:
         case aco_opcode::s_nor_saveexec_b64:
         case aco_opcode::s_xnor_saveexec_b64:
            return true;
         default:
            return false;
      }
   }

   if (instr->opcode == aco_opcode::v_readlane_b32 ||
       instr->opcode == aco_opcode::v_readfirstlane_b32 ||
       instr->opcode == aco_opcode::v_writelane_b32)
      return false;

   return true;
}

bool writes_exec_implicitly(Instruction* instr)
{
   if (instr->isSALU()) {
      switch (instr->opcode) {
         case aco_opcode::s_and_saveexec_b64:
         case aco_opcode::s_or_saveexec_b64:
         case aco_opcode::s_xor_saveexec_b64:
         case aco_opcode::s_andn2_saveexec_b64:
         case aco_opcode::s_orn2_saveexec_b64:
         case aco_opcode::s_nand_saveexec_b64:
         case aco_opcode::s_nor_saveexec_b64:
         case aco_opcode::s_xnor_saveexec_b64:
            return true;
         default:
            return false;
      }
   }

   // TODO v_cmpx_*
   return false;
}

unsigned detect_pipeline_hazard(Instruction* first, Instruction* second)
{
   return 0;
   /**
    * TODO: not sure if these can efficiently be modeled as RAW dependencies
    * we can check them when adding instructions to list of candidates.
    * (1) S_SETREG MODE.vskip / any vector op
    * (2) SALU writes M0 / GDS, S_SENDMSG or S_TTRACE_DATA
    * (3) VALU writes EXEC / VALU DPP op
    * (4) S_SETREG TRAPSTS / RFE, RFE_restore
    * (5) SALU writes M0 / LDS "add-TID" instruction, buffer_store_LDS_dword, scratch or global with LDS = 1, VINTERP or LDS_direct
    * (6) SALU writes M0 / S_MOVEREL
    */
}

unsigned detect_raw_hazard(Instruction* first, Instruction* second, int op_idx)
{
   /* VALU writes vgpr / VALU DPP reads vgpr */
   if (second->isDPP() && op_idx == 0 &&
       (first->isVALU() || first->format == Format::VINTRP))
         return 2;

   /* s_setreg / s_getreg */
   if (second->opcode == aco_opcode::s_getreg_b32)
   {
      if (first->opcode == aco_opcode::s_setreg_b32 ||
          first->opcode == aco_opcode::s_setreg_imm32_b32)
         return 2;
      if (first->opcode == aco_opcode::s_setvskip)
         return 2;
   }

   /* valu writes vcc / v_div_fmas */
   if (second->opcode == aco_opcode::v_div_fmas_f32 ||
       second->opcode == aco_opcode::v_div_fmas_f64)
   {
      // TODO: if first isVALU && writes vcc return 4
   }

   /* VALU writes sgpr / VMEM reads sgpr */
   if (first->isVALU() && second->isVMEM() &&
       second->operands[op_idx].getTemp().type() == RegType::sgpr)
      return 5;

   /* VALU writes sgpr/vcc / v_{read/write}lane uses sgpr/vcc */
   if (first->isVALU() && 
      (second->opcode == aco_opcode::v_readlane_b32 ||
       second->opcode == aco_opcode::v_writelane_b32))
      return 4;

   /* FIXME VEGA only: SALU write m0 / VINTERP */
   if (second->format == Format::VINTRP && first->isSALU()) {
      for (unsigned i = 0; i < first->num_definitions; i++)
         if (first->getDefinition(i).physReg() == m0)
            return 1;
   }

   return 0;

   // TODO: gds, s_sendmsg, s_ttrace_data, lds, s_moverel
}

unsigned detect_waw_hazard(Instruction* first, Instruction* second)
{
   if ((first->opcode == aco_opcode::s_setreg_b32 ||
        first->opcode == aco_opcode::s_setreg_imm32_b32) &&
       (second->opcode == aco_opcode::s_setreg_b32 ||
        second->opcode == aco_opcode::s_setreg_imm32_b32))
      return 2;
   else
      return 0;
}

unsigned detect_war_hazard(Instruction* first, Instruction* second)
{
   if (first->opcode == aco_opcode::buffer_store_dwordx3 ||
       first->opcode == aco_opcode::buffer_store_dwordx4) {
      if (second->isVMEM())
         return 0;
      for (unsigned i = 0; i < second->num_definitions; i++)
         for (unsigned j = 0; j < second->getDefinition(i).size(); j++)
            for (unsigned k = 0; k < first->getOperand(3).size(); k++)
               if (first->getOperand(3).physReg().reg + k == second->getDefinition(i).physReg().reg + j)
                  return 1;
   }
   if (first->opcode == aco_opcode::image_store &&
       first->getOperand(2).size() > 2) {
      if (second->isVMEM())
         return 0;
      for (unsigned i = 0; i < second->num_definitions; i++)
         for (unsigned j = 0; j < second->getDefinition(i).size(); j++)
            for (unsigned k = 0; k < first->getOperand(2).size(); k++)
               if (first->getOperand(2).physReg().reg + k == second->getDefinition(i).physReg().reg + j)
                  return 1;
    }

   return 0;
   // TODO: store instr > 64bit / write data vgpr
}

bool is_new_candidate(Node* node)
{
   for (std::pair<Node*,int> pre : node->predecessors)
   {
      if (!pre.first->scheduled)
         return false;
   }
   return true;
}

Node* select_candidate(sched_ctx& ctx)
{
   std::set<Node*>::iterator it = ctx.candidates.begin();
   // TODO: choose candidate based on priority
   Node* next = *it;
   ctx.candidates.erase(it);
   assert (!next->scheduled);
   next->scheduled = true;
   /* add successors to list of potential candidates */
   for (Node* n : next->successors)
   {
      if (is_new_candidate(n))
         ctx.candidates.insert(n);
   }

   /* check for unresolved hazards with predecessors */
   for (std::pair<Node*,int> pre : next->predecessors)
   {
      if (!pre.second)
         continue;
      int nops = pre.second - (ctx.current_index - pre.first->index);
      next->nops = std::max(next->nops, nops);
   }
   return next;
}

bool handle_read(Block* block, Instruction* instr, Node* node, int op_idx, unsigned reg, sched_ctx& ctx)
{
   bool is_candidate = true;
   std::unordered_map<unsigned, Node*>::iterator it = ctx.def_table.find(reg);
   if (it != ctx.def_table.end())
   {
      Node* predecessor = it->second;
      is_candidate = false;
      predecessor->successors.insert(node);
      int nops = detect_raw_hazard(block->instructions[predecessor->index].get(), instr, op_idx);
      node->predecessors.emplace_back(std::make_pair(predecessor, nops));
   }
   return is_candidate;
}

bool handle_write(Block *block, Instruction* instr, Node* node, unsigned reg, sched_ctx& ctx)
{
   bool is_candidate = true;
   std::unordered_map<unsigned, Node*>::iterator it = ctx.def_table.find(reg);
   if (it != ctx.def_table.end())
   {
      is_candidate = false;
      /* add all uses of previous write to predecessors */
      for (Node* use : it->second->successors)
      {
         if (use == node)
            continue;
         use->successors.insert(node);
         int nops = detect_war_hazard(block->instructions[use->index].get(), instr);
         node->predecessors.emplace_back(std::make_pair(use, nops));
      }
      /* add previous write as predecessor */
      it->second->successors.insert(node);
      int nops = detect_waw_hazard(block->instructions[it->second->index].get(), instr);
      node->predecessors.emplace_back(std::make_pair(it->second, nops));
   }
   ctx.def_table[reg] = node;
   return is_candidate;
}

void build_dag(Block* block, sched_ctx& ctx)
{
   // TODO: add / propagate priorities
   for (unsigned index = 0; index < block->instructions.size(); index++)
   {
      Instruction* instr = block->instructions[index].get();
      ctx.nodes.emplace_back(index);
      Node* node = &ctx.nodes.back();
      bool is_candidate = true;
      /* Read after Write */
      for (unsigned i = 0; i < instr->num_operands; i++)
      {
         if (instr->operands[i].isConstant()) continue;
         unsigned reg = instr->operands[i].physReg().reg;
         for (unsigned k = 0; k < instr->operands[i].size(); k++) {
            if (!handle_read(block, instr, node, i, reg + k, ctx))
               is_candidate = false;
         }

      }

      if (reads_exec_implicitly(instr)) {
         for (unsigned reg = exec.reg; reg <= exec.reg + 1; reg++) {
            if (!handle_read(block, instr, node, -1, reg, ctx))
               is_candidate = false;
         }
      }

      /* Write after Write/Read */
      for (unsigned i = 0; i < instr->num_definitions; i++)
      {
         unsigned reg = instr->definitions[i].physReg().reg;
         for (unsigned k = 0; k < instr->definitions[i].size(); k++)
         {
            if (!handle_write(block, instr, node, reg + k, ctx))
               is_candidate = false;
         }
      }
      if (writes_exec_implicitly(instr)) {
         for (unsigned reg = exec.reg; reg <= exec.reg + 1; reg++) {
            if (!handle_write(block, instr, node, reg, ctx))
               is_candidate = false;
         }
      }
      if (is_candidate)
         ctx.candidates.insert(node);
   }
}

void schedule(Program* program)
{
   for (auto&& block : program->blocks)
   {
      sched_ctx ctx(block->instructions.size());
      std::vector<aco_ptr<Instruction>> new_instructions;
      build_dag(block.get(), ctx);
      while (!ctx.candidates.empty())
      {
         Node* next_instr = select_candidate(ctx);
         if (next_instr->nops)
         {
            /* create NOP */
            SOPP_instruction* nop = create_instruction<SOPP_instruction>(aco_opcode::s_nop, Format::SOPP, 0, 0);
            nop->imm = next_instr->nops - 1;
            new_instructions.emplace_back(aco_ptr<Instruction>(nop));
         }
         ctx.current_index = new_instructions.size();
         new_instructions.emplace_back(std::move(block->instructions[next_instr->index]));
         next_instr->index = ctx.current_index;
      }
      block->instructions.swap(new_instructions);
   }
}

}

