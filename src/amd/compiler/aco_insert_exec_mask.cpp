/*
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
 */

#include <vector>

#include "aco_ir.h"
#include "aco_builder.h"

namespace aco {

namespace {

enum WQMState : uint8_t {
   Unspecified = 0,
   Exact = 1 << 0,
   WQM = 1 << 1 /* with control flow applied */
};

enum mask_type : uint8_t {
   mask_type_global = 1 << 0,
   mask_type_wqm = 1 << 1, /* 0 indicates exact */
   mask_type_loop = 1 << 2,
};

struct wqm_ctx {
   /* state for WQM propagation */
   std::set<unsigned> worklist;
   std::vector<uint16_t> defined_in;
   std::vector<bool> needs_wqm;
   std::vector<bool> branch_wqm; /* true if the branch condition in this block should be in wqm */
   wqm_ctx(Program* program) : defined_in(program->peekAllocationId(), 0xFFFF),
                               needs_wqm(program->peekAllocationId()),
                               branch_wqm(program->blocks.size())
   {
      for (unsigned i = 0; i < program->blocks.size(); i++)
         worklist.insert(i);
   }
};

struct loop_info {
   Block* loop_header;
   size_t num_exec_masks;
   bool has_divergent_break = false;
   bool has_divergent_continue = false;
   loop_info(Block* block, unsigned num) : loop_header(block), num_exec_masks(num) {}
};

struct block_info {
   std::vector<std::pair<Temp, uint8_t>> exec;
   std::vector<WQMState> instr_needs;
   /* more... */
};

struct exec_ctx {
   Program *program;
   std::vector<block_info> info;
   std::vector<loop_info> loop;
   exec_ctx(Program *program) : program(program), info(program->blocks.size()) {}
};

bool pred_by_exec_mask(aco_ptr<Instruction>& instr) {
   if (instr->format == Format::SMEM || instr->isSALU())
      return false;
   if (instr->format == Format::PSEUDO_BARRIER)
      return false;

   if (instr->format == Format::PSEUDO) {
      switch (instr->opcode) {
      case aco_opcode::p_create_vector:
         return instr->getDefinition(0).getTemp().type() == RegType::vgpr;
      case aco_opcode::p_extract_vector:
      case aco_opcode::p_split_vector:
         return instr->getOperand(0).getTemp().type() == RegType::vgpr;
      case aco_opcode::p_spill:
      case aco_opcode::p_reload:
         return false;
      default:
         break;
      }
   }

   if (instr->opcode == aco_opcode::v_readlane_b32 ||
       instr->opcode == aco_opcode::v_writelane_b32)
      return false;

   return true;
}

bool needs_exact(aco_ptr<Instruction>& instr) {
   if (instr->format == Format::MUBUF) {
      MUBUF_instruction *mubuf = static_cast<MUBUF_instruction *>(instr.get());
      return mubuf->disable_wqm;
   } else if (instr->format == Format::MIMG) {
      MIMG_instruction *mimg = static_cast<MIMG_instruction *>(instr.get());
      return mimg->disable_wqm;
   } else {
      return false;
   }
}

void set_needs_wqm(wqm_ctx &ctx, Block *block, Temp tmp)
{
   if (!ctx.needs_wqm[tmp.id()]) {
      ctx.needs_wqm[tmp.id()] = true;
      if (ctx.defined_in[tmp.id()] != 0xFFFF)
         ctx.worklist.insert(ctx.defined_in[tmp.id()]);
   }
}

void mark_block_wqm(wqm_ctx &ctx, Block *block)
{
   if (ctx.branch_wqm[block->index])
      return;
   ctx.branch_wqm[block->index] = true;

   aco_ptr<Instruction>& branch = *block->instructions.rbegin();
   if (branch->opcode != aco_opcode::p_branch) {
      assert(branch->operandCount() && branch->getOperand(0).isTemp());
      set_needs_wqm(ctx, block, branch->getOperand(0).getTemp());
   }

   for (Block *pred : block->logical_predecessors)
      mark_block_wqm(ctx, pred);
}

/* ensure the condition controlling the control flow for this phi is in WQM */
void mark_phi_wqm(wqm_ctx &ctx, Block *block)
{
   /* TODO: this sets more branch conditions to WQM than it needs to */
   for (Block *pred : block->logical_predecessors)
      mark_block_wqm(ctx, pred);
}

std::vector<WQMState> get_block_needs(wqm_ctx &ctx, Block* block)
{
   std::vector<WQMState> instr_needs(block->instructions.size());

   for (int i = block->instructions.size() - 1; i >= 0; --i)
   {
      aco_ptr<Instruction>& instr = block->instructions[i];

      WQMState needs = needs_exact(instr) ? Exact : Unspecified;
      bool propagate_wqm = instr->opcode == aco_opcode::p_wqm;

      bool pred_by_exec = pred_by_exec_mask(instr);
      for (unsigned j = 0; j < instr->definitionCount(); j++) {
         if (!instr->getDefinition(j).isTemp())
            continue;
         unsigned def = instr->getDefinition(j).tempId();
         ctx.defined_in[def] = block->index;
         if (needs == Unspecified && ctx.needs_wqm[def]) {
            needs = pred_by_exec ? WQM : Unspecified;
            propagate_wqm = true;
         }
      }

      if (propagate_wqm) {
         for (unsigned j = 0; j < instr->operandCount(); j++) {
            if (!instr->getOperand(j).isTemp())
               continue;
            set_needs_wqm(ctx, block, instr->getOperand(j).getTemp());
         }
      }

      if (needs == WQM && instr->opcode == aco_opcode::p_phi)
         mark_phi_wqm(ctx, block);

      instr_needs[i] = needs;
   }
   return instr_needs;
}

void calculate_wqm_needs(exec_ctx& exec_ctx)
{
   wqm_ctx ctx(exec_ctx.program);

   while (!ctx.worklist.empty()) {
      unsigned block_index = *std::prev(ctx.worklist.end());
      ctx.worklist.erase(std::prev(ctx.worklist.end()));

      Block *block = exec_ctx.program->blocks[block_index].get();
      exec_ctx.info[block->index].instr_needs = get_block_needs(ctx, block);
   }
}

void add_coupling_code(exec_ctx& ctx, std::unique_ptr<Block>& block)
{
   unsigned idx = block->index;
   Builder bld(ctx.program, block.get());
   bld.start = true;
   std::vector<Block*>& preds = block->linear_predecessors;

   if (idx == 0) {
      aco_ptr<Instruction>& startpgm = ctx.program->blocks[0]->instructions[0];
      assert(startpgm->opcode == aco_opcode::p_startpgm);
      Temp exec_mask = startpgm->getDefinition(startpgm->num_definitions - 1).getTemp();

      if (ctx.program->needs_wqm) {
         std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
         bld.it = ++it;
         bld.start = false;
         bld.use_iterator = true;
         exec_mask = bld.sop1(aco_opcode::s_wqm_b64, bld.def(s2, exec), bld.def(s1, scc), exec_mask);
         //assert(!ctx.program->needs_exact);
      }
      ctx.info[0].exec.emplace_back(exec_mask, mask_type_global);
      return;
   }

   /* loop entry block */
   if (block->kind & block_kind_loop_header) {
      assert(preds[0]->index == idx - 1);
      ctx.info[idx].exec = ctx.info[idx - 1].exec;
      unsigned num_exec_masks = ctx.info[idx].exec.size();
      ctx.loop.push_back({block.get(), num_exec_masks});

      bool has_divergent_break = false;
      bool has_divergent_continue = false;
      for (unsigned i = idx; ctx.program->blocks[i]->loop_nest_depth >= block->loop_nest_depth; i++) {
         Block* loop_block = ctx.program->blocks[i].get();
         if (loop_block->loop_nest_depth != block->loop_nest_depth)
            continue;
         else if (loop_block->kind & block_kind_uniform)
            continue;
         else if (loop_block->kind & block_kind_break)
            has_divergent_break = true;
         else if (loop_block->kind & block_kind_continue)
            has_divergent_continue = true;
         // TODO: extend for discards/wqm to skip phis
      }

      /* create ssa names for outer exec masks */
      for (unsigned i = 0; i < num_exec_masks; i++) {
         aco_ptr<Instruction> phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, preds.size(), 1)};
         /* uniform loop, re-use the live mask to restore afterwards */
         // TODO: probably not even necessary to create a phi if there is also no discard/wqm
         if (i == num_exec_masks - 1 && !(has_divergent_break || has_divergent_continue))
            phi->getDefinition(0) = bld.def(s2, exec);
         else
            phi->getDefinition(0) = bld.def(s2);
         phi->getOperand(0) = Operand(ctx.info[preds[0]->index].exec[i].first);
         ctx.info[idx].exec[i].first = bld.insert(std::move(phi));
      }

      /* create ssa name for loop active mask */
      if (has_divergent_break) {
         aco_ptr<Instruction> phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, preds.size(), 1)};

         if (!has_divergent_continue)
            phi->getDefinition(0) = bld.def(s2, exec);
         else
            phi->getDefinition(0) = bld.def(s2);
         phi->getOperand(0) = Operand(ctx.info[preds[0]->index].exec[num_exec_masks - 1].first);
         ctx.info[idx].exec.emplace_back(bld.insert(std::move(phi)), mask_type_loop);
      } else {
         ctx.info[idx].exec[num_exec_masks - 1].second |= mask_type_loop;
      }

      /* create a parallelcopy to move the active mask to exec */
      if (has_divergent_continue) {
         std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
         while ((*it)->opcode != aco_opcode::p_logical_start)
            ++it;
         bld.it = it;
         bld.use_iterator = true;
         ctx.info[idx].exec.emplace_back(bld.pseudo(aco_opcode::p_parallelcopy, bld.def(s2, exec),
                                                    ctx.info[idx].exec.back().first), 0);
      }

      return;
   }

   /* loop exit block */
   if (block->kind & block_kind_loop_exit) {
      Block* header = ctx.loop.back().loop_header;
      unsigned num_exec_masks = ctx.loop.back().num_exec_masks;

      for (Block* pred : preds)
         assert(ctx.info[pred->index].exec.size() >= num_exec_masks);

      /* fill the loop header phis */
      std::vector<Block*>& header_preds = header->linear_predecessors;
      for (unsigned k = 0; k < num_exec_masks; k++) {
         unsigned phi_pos = ctx.loop.back().has_divergent_break ? num_exec_masks - k :
                                                                  num_exec_masks - k - 1;
         aco_ptr<Instruction>& phi = header->instructions[phi_pos];
         assert(phi->opcode == aco_opcode::p_linear_phi);
         for (unsigned i = 1; i < phi->num_operands; i++)
            phi->getOperand(i) = Operand(ctx.info[header_preds[i]->index].exec[k].first);
      }
      if (ctx.loop.back().has_divergent_break) {
         unsigned k = num_exec_masks;
         aco_ptr<Instruction>& phi = header->instructions[0];
         for (unsigned i = 1; i < phi->num_operands; i++)
            phi->getOperand(i) = Operand(ctx.info[header_preds[i]->index].exec[k].first);
      }

      /* create the loop exit phis if not trivial */
      for (unsigned k = 0; k < num_exec_masks; k++) {
         Temp same = ctx.info[preds[0]->index].exec[k].first;
         uint8_t type = ctx.info[header_preds[0]->index].exec[k].second;
         bool trivial = true;

         for (unsigned i = 1; i < preds.size() && trivial; i++) {
            if (ctx.info[preds[i]->index].exec[k].first != same)
               trivial = false;
         }

         if (trivial) {
            ctx.info[idx].exec.emplace_back(same, type);
         } else {
            /* create phi for loop footer */
            aco_ptr<Instruction> phi{create_instruction<Instruction>(aco_opcode::p_linear_phi, Format::PSEUDO, preds.size(), 1)};
            phi->getDefinition(0) = bld.def(s2);
            for (unsigned i = 0; i < phi->num_operands; i++)
               phi->getOperand(i) = Operand(ctx.info[preds[i]->index].exec[k].first);
            ctx.info[idx].exec.emplace_back(bld.insert(std::move(phi)), type);
         }
      }

      /* create a parallelcopy to move the active mask to exec */
      std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
      while ((*it)->opcode != aco_opcode::p_logical_start)
         ++it;
      bld.it = it;
      bld.use_iterator = true;
      ctx.info[idx].exec.back().first = bld.pseudo(aco_opcode::p_parallelcopy, bld.def(s2, exec),
                                                            ctx.info[idx].exec.back().first);

      ctx.loop.pop_back();

      return;
   }

   ctx.info[idx].exec = ctx.info[preds[0]->index].exec;

   if (preds.size() == 1)
      return;

   assert(preds.size() == 2);
   assert(ctx.info[preds[0]->index].exec.size() == ctx.info[preds[1]->index].exec.size());

   /* create phis for diverged exec masks */
   for (unsigned i = 0; i < ctx.info[idx].exec.size(); i++) {
      if (ctx.info[preds[1]->index].exec[i].first == ctx.info[idx].exec[i].first)
         continue;
      Definition def = i == ctx.info[idx].exec.size() - 1 ? bld.def(s2, exec) : bld.def(s2);
      ctx.info[idx].exec[i].first = bld.pseudo(aco_opcode::p_linear_phi, def,
                                               ctx.info[idx].exec[i].first,
                                               ctx.info[preds[1]->index].exec[i].first);
   }

   if (block->kind & block_kind_merge) {
      assert(ctx.info[idx].exec.size() >= 2);
      ctx.info[idx].exec.pop_back();
      Temp restore = ctx.info[idx].exec.back().first;

      std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
      while ((*it)->opcode != aco_opcode::p_logical_start)
         ++it;
      bld.it = it;
      bld.use_iterator = true;

      ctx.info[idx].exec.back().first = bld.sop1(aco_opcode::s_mov_b64, bld.def(s2, exec),
                                                 restore);
   }

}

void handle_discard(exec_ctx& ctx, std::unique_ptr<Block>& block)
{


}

void add_branch_code(exec_ctx& ctx, std::unique_ptr<Block>& block)
{
   if (block->kind & block_kind_uniform)
      return;

   unsigned idx = block->index;
   Builder bld(ctx.program, block.get());

   if (block->kind & block_kind_branch) {
      // orig = s_and_saveexec_b64
      assert(block->linear_successors.size() == 2);
      assert(block->instructions.back()->opcode == aco_opcode::p_cbranch_z);
      Temp cond = block->instructions.back()->getOperand(0).getTemp();
      block->instructions.pop_back();
      Temp current_exec = ctx.info[idx].exec.back().first;

      Temp then_mask = bld.tmp(s2);
      Temp old_exec = bld.sop1(aco_opcode::s_and_saveexec_b64, bld.def(s2), bld.def(s1, scc),
                               bld.exec(Definition(then_mask)), cond, bld.exec(current_exec));

      ctx.info[idx].exec.back().first = old_exec;

      /* add next current exec to the stack */
      ctx.info[idx].exec.emplace_back(then_mask, 0);

      bld.branch(aco_opcode::p_cbranch_z, bld.exec(then_mask), block->linear_successors[1], block->linear_successors[0]);
      return;
   }

   if (block->kind & block_kind_invert) {
      // exec = s_andn2_b64 (original_exec, exec)
      assert(block->instructions.back()->opcode == aco_opcode::p_cbranch_nz);
      block->instructions.pop_back();
      Temp then_mask = ctx.info[idx].exec.back().first;
      ctx.info[idx].exec.pop_back();
      Temp orig_exec = ctx.info[idx].exec.back().first;
      Temp else_mask = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2, exec),
                                bld.def(s1, scc), orig_exec, bld.exec(then_mask));

      /* add next current exec to the stack */
      ctx.info[idx].exec.emplace_back(else_mask, 0);

      bld.branch(aco_opcode::p_cbranch_z, bld.exec(else_mask), block->linear_successors[1], block->linear_successors[0]);
      return;
   }

   if (block->kind & block_kind_break) {
      ctx.loop.back().has_divergent_break = true;
      // loop_mask = s_andn2_b64 (loop_mask, exec)
      assert(block->instructions.back()->opcode == aco_opcode::p_branch);
      block->instructions.pop_back();

      Temp current_exec = ctx.info[idx].exec.back().first;
      Temp cond;
      for (int exec_idx = ctx.info[idx].exec.size() - 2; exec_idx >= 0; exec_idx--) {
         cond = bld.tmp(s1);
         Temp exec_mask = ctx.info[idx].exec[exec_idx].first;
         exec_mask = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.scc(Definition(cond)),
                              exec_mask, bld.exec(current_exec));
         ctx.info[idx].exec[exec_idx].first = exec_mask;
         if (ctx.info[idx].exec[exec_idx].second & mask_type_loop)
            break;
      }

      /* check if the successor is the merge block, otherwise set exec to 0 */
      // TODO: this could be done better by directly branching to the merge block
      Block* succ = block->linear_successors[1]->linear_successors[0];
      if (!(succ->kind & block_kind_invert || succ->kind & block_kind_merge)) {
         ctx.info[idx].exec.back().first = bld.sop1(aco_opcode::s_mov_b64, bld.def(s2, exec), Operand(0u));
      }

      bld.branch(aco_opcode::p_cbranch_nz, bld.scc(cond), block->linear_successors[1], block->linear_successors[0]);
      return;
   }

   if (block->kind & block_kind_continue) {
      ctx.loop.back().has_divergent_continue = true;
      assert(block->instructions.back()->opcode == aco_opcode::p_branch);
      block->instructions.pop_back();

      Temp current_exec = ctx.info[idx].exec.back().first;
      Temp cond;
      for (int exec_idx = ctx.info[idx].exec.size() - 2; exec_idx >= 0; exec_idx--) {
         if (ctx.info[idx].exec[exec_idx].second & mask_type_loop)
            break;
         cond = bld.tmp(s1);
         Temp exec_mask = ctx.info[idx].exec[exec_idx].first;
         exec_mask = bld.sop2(aco_opcode::s_andn2_b64, bld.def(s2), bld.scc(Definition(cond)),
                              exec_mask, bld.exec(current_exec));
         ctx.info[idx].exec[exec_idx].first = exec_mask;
      }
      assert(cond != Temp());

      /* check if the successor is the merge block, otherwise set exec to 0 */
      // TODO: this could be done better by directly branching to the merge block
      Block* succ = block->linear_successors[1]->linear_successors[0];
      if (!(succ->kind & block_kind_invert || succ->kind & block_kind_merge)) {
         ctx.info[idx].exec.back().first = bld.sop1(aco_opcode::s_mov_b64, bld.def(s2, exec), Operand(0u));
      }

      bld.branch(aco_opcode::p_cbranch_nz, bld.scc(cond), block->linear_successors[1], block->linear_successors[0]);
      return;
   }

}

void process_block(exec_ctx& ctx, std::unique_ptr<Block>& block)
{
   add_coupling_code(ctx, block);

   assert(block->index != ctx.program->blocks.size() - 1 ||
          ctx.info[block->index].exec.size() <= 2);

   if (block->kind & block_kind_uses_discard)
      handle_discard(ctx, block);

   add_branch_code(ctx, block);
}

} /* end namespace */


void insert_exec_mask(Program *program)
{
   exec_ctx ctx(program);

   if (program->needs_wqm && program->needs_exact)
      calculate_wqm_needs(ctx);

   for (std::unique_ptr<Block>& block : program->blocks)
      process_block(ctx, block);

}

}

