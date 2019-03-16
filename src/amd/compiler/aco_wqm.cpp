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
 * Authors:
 *    Rhys Perry (pendingchaos02@gmail.com)
 *
 */

#include <unordered_map>
#include <vector>

#include "aco_ir.h"


namespace aco {

enum WQMState : uint8_t {
   Unspecified = 0,
   Exact = 1 << 0,
   WQM = 1 << 1 /* with control flow applied */
};

struct block_info
{
   std::vector<WQMState> instr_needs;
   bool wqm; /* true if the branch condition in this block should be in wqm */
};

struct wqm_ctx
{
   Program *program;
   std::vector<block_info> blocks;
   Temp exact_mask;

   /* state for WQM propagation */
   std::vector<uint16_t> defined_in;
   std::set<unsigned> propagate_worklist;
   std::vector<bool> needs_wqm;

   /* process_block() state */
   Temp wqm_exec_mask;
};

bool pred_by_exec_mask(aco_ptr<Instruction>& instr) {
   if (instr->format == Format::SMEM || instr->isSALU())
      return false;
   if (instr->format == Format::PSEUDO_BARRIER)
      return false;

   if (instr->format == Format::PSEUDO) {
      switch (instr->opcode) {
      case aco_opcode::p_extract_vector:
      case aco_opcode::p_create_vector:
      case aco_opcode::p_split_vector:
         return instr->getDefinition(0).getTemp().type() == RegType::vgpr;
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

bool directly_reads_exec(Instruction* instr)
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
            break;
      }
   }

   if (instr->opcode == aco_opcode::p_discard_if)
      return true;

   bool isVOPC = ((uint16_t) instr->format & (uint16_t) Format::VOPC) == (uint16_t) Format::VOPC;
   if (!isVOPC && instr->isVALU()) {
      switch (instr->opcode) {
      case aco_opcode::v_cndmask_b32:
      case aco_opcode::v_writelane_b32:
      case aco_opcode::v_addc_co_u32:
      case aco_opcode::v_subb_co_u32:
      case aco_opcode::v_subbrev_co_u32:
         break;
      default:
         return false;
      }
   } else if (instr->format == Format::DS || instr->format == Format::EXP) {
      return false;
   }

   for (unsigned i = 0; i < instr->operandCount(); i++) {
      Operand &op = instr->getOperand(i);
      if (op.isFixed() && op.physReg().reg + op.size() >= exec.reg && op.physReg().reg <= exec.reg + 1)
         return true;
   }
   
   return false;
}

bool writes_exec(Instruction* instr)
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
            break;
      }
   }

   if (instr->opcode >= aco_opcode::v_cmpx_class_f16 &&
       instr->opcode <= aco_opcode::v_cmpx_u_f64)
      return true;

   bool isVOPC = ((uint16_t) instr->format & (uint16_t) Format::VOPC) == (uint16_t) Format::VOPC;
   if (!isVOPC && instr->isVALU()) {
      switch (instr->opcode) {
      case aco_opcode::v_add_co_u32:
      case aco_opcode::v_sub_co_u32:
      case aco_opcode::v_subrev_co_u32:
      case aco_opcode::v_readfirstlane_b32:
      case aco_opcode::v_readlane_b32:
      case aco_opcode::v_addc_co_u32:
      case aco_opcode::v_subb_co_u32:
      case aco_opcode::v_subbrev_co_u32:
         break;
      default:
         return false;
      }
   } else if (instr->format == Format::DS || instr->format == Format::EXP ||
              instr->format == Format::FLAT || instr->format == Format::GLOBAL ||
              instr->format == Format::SCRATCH) {
      return false;
   }

   for (unsigned i = 0; i < instr->definitionCount(); i++) {
      Definition &def = instr->getDefinition(i);
      if (def.isFixed() && def.physReg().reg + def.size() >= exec.reg && def.physReg().reg <= exec.reg + 1)
         return true;
   }
   
   return false;
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
         ctx.propagate_worklist.insert(ctx.defined_in[tmp.id()]);
   }
}

void mark_block_wqm(wqm_ctx &ctx, Block *block)
{
   if (ctx.blocks[block->index].wqm)
      return;
   ctx.blocks[block->index].wqm = true;

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

void get_block_needs(wqm_ctx &ctx, block_info *info, Block* block)
{
   info->instr_needs.resize(block->instructions.size());

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
         if (pred_by_exec && needs == Unspecified && ctx.needs_wqm[def]) {
            needs = WQM;
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

      info->instr_needs[i] = needs;
   }
}

WQMState transition_state(wqm_ctx &ctx, std::vector<aco_ptr<Instruction>>& instrs, WQMState from, WQMState to)
{
   if (from == to)
      return to;

   if (to == WQM) {
      /* Unspecified/Exact -> WQM */
      assert(ctx.wqm_exec_mask.id());
      aco_ptr<Instruction> mov{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1)};
      mov->getOperand(0) = Operand(ctx.wqm_exec_mask);
      mov->getDefinition(0) = Definition(exec, s2);
      instrs.emplace_back(std::move(mov));
      ctx.wqm_exec_mask = Temp(0, s1);
      return WQM;
   } else if (to == Exact) {
      /* Unspecified/WQM -> Exact */
      ctx.wqm_exec_mask = {ctx.program->allocateId(), s2};
      aco_ptr<Instruction> instr{create_instruction<SOP2_instruction>(aco_opcode::s_and_saveexec_b64, Format::SOP1, 1, 3)};
      instr->getOperand(0) = Operand(ctx.exact_mask);
      instr->getDefinition(0) = Definition(ctx.wqm_exec_mask);
      instr->getDefinition(1) = Definition(ctx.program->allocateId(), scc, b);
      instr->getDefinition(2) = Definition(exec, s2);
      instrs.emplace_back(std::move(instr));
      return Exact;
   } else {
      assert(to == Unspecified);
      return from;
   }
}

void process_block(wqm_ctx &ctx, block_info *info, Block *block)
{
   std::vector<aco_ptr<Instruction>> instructions;
   instructions.swap(block->instructions);
   block->instructions.reserve(instructions.size());

   WQMState state = WQM;

   unsigned i = 0;
   if (block->index == 0 && instructions.size() && instructions[0]->opcode == aco_opcode::p_startpgm) {
      block->instructions.emplace_back(std::move(instructions[0]));
      i = 1;
   }

   if (block->index == 0) {
      ctx.exact_mask = {ctx.program->allocateId(), s2};
      aco_ptr<Instruction> mov{create_instruction<SOP1_instruction>(aco_opcode::s_mov_b64, Format::SOP1, 1, 1)};
      mov->getOperand(0) = Operand(exec, s2);
      mov->getDefinition(0) = Definition(ctx.exact_mask);
      block->instructions.emplace_back(std::move(mov));

      aco_ptr<Instruction> wqm{create_instruction<SOP1_instruction>(aco_opcode::s_wqm_b64, Format::SOP1, 1, 2)};
      wqm->getOperand(0) = Operand(exec, s2);
      wqm->getDefinition(0) = Definition(exec, s2);
      wqm->getDefinition(1) = Definition(ctx.program->allocateId(), scc, b);
      block->instructions.emplace_back(std::move(wqm));
   }

   for (; i < instructions.size(); i++) {
      aco_ptr<Instruction> instr = std::move(instructions[i]);

      WQMState needs = info->instr_needs[i];

      if (instr->format == Format::PSEUDO_BRANCH || directly_reads_exec(instr.get())) {
         assert(needs == Unspecified || needs == WQM);
         needs = WQM;
      }

      state = transition_state(ctx, block->instructions, state, needs);

      /* transitioning from Exact to WQM could overwrite it */
      if (writes_exec(instr.get()))
         state = WQM;

      block->instructions.emplace_back(std::move(instr));
   }
}

void lower_wqm(Program *program, live& live_vars,
               const struct radv_nir_compiler_options *options)
{
   if (program->needs_wqm && !program->needs_exact) {
      Block *block = program->blocks[0].get();
      aco_ptr<Instruction> wqm{create_instruction<SOP1_instruction>(aco_opcode::s_wqm_b64, Format::SOP1, 1, 2)};
      wqm->getOperand(0) = Operand(exec, s2);
      wqm->getDefinition(0) = Definition(exec, s2);
      wqm->getDefinition(1) = Definition(scc, b);
      if (block->instructions.size() && block->instructions[0]->opcode == aco_opcode::p_startpgm) {
         block->instructions.emplace(std::next(block->instructions.begin()), std::move(wqm));
         live_vars.register_demand[0].emplace(live_vars.register_demand[0].begin(), live_vars.register_demand[0][0]);
      } else {
         block->instructions.emplace(block->instructions.begin(), std::move(wqm));
         live_vars.register_demand[0].emplace(live_vars.register_demand[0].begin(), std::pair<uint16_t, uint16_t>(0, 0));
      }
      return;
   } else if (!program->needs_wqm) {
      return;
   }

   wqm_ctx ctx;
   ctx.program = program;
   ctx.blocks.resize(program->blocks.size());
   ctx.defined_in.resize(program->peekAllocationId());
   memset(ctx.defined_in.data(), 0xFF, ctx.defined_in.size() * 2);
   ctx.needs_wqm.resize(program->peekAllocationId());

   for (unsigned i = 0; i < program->blocks.size(); ++i)
      ctx.propagate_worklist.insert(i);

   while (!ctx.propagate_worklist.empty()) {
      unsigned block_index = *std::prev(ctx.propagate_worklist.end());
      ctx.propagate_worklist.erase(std::prev(ctx.propagate_worklist.end()));

      Block *block = program->blocks[block_index].get();
      block_info *info = &ctx.blocks[block_index];
      get_block_needs(ctx, info, block);
   }

   for (std::vector<std::unique_ptr<Block>>::iterator it = program->blocks.begin(); it != program->blocks.end(); ++it) {
      Block *block = it->get();
      block_info *info = &ctx.blocks[block->index];
      process_block(ctx, info, block);
   }

   live_vars = aco::live_var_analysis<true>(program, options);
}
}
