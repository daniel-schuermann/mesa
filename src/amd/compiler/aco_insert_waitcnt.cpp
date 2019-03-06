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

#include "aco_ir.h"

namespace aco {

/**
 * The general idea of this pass is a worklist algorithm:
 * The CFG is traversed in reverse postorder (forward).
 * Per BB one wait_ctx is maintained.
 * The in-context is the joined out-contexts of the predecessors.
 * The context contains a hashmap: vgpr -> wait_entry
 * consisting of the information about the cnt values to be waited for.
 * Note: After merge-nodes, it might occur that for the same register
 *       multiple cnt values are to be waited for.
 * 
 * The values are updated according to the encountered instructions:
 * - additional events increment the counter of waits of the same type
 * - or erase vgprs with counters higher than to be waited for.
 */

// TODO: do a more clever insertion of wait_cnt (lgkm_cnt) when there is a load followed by a use of a previous load

enum wait_type : uint8_t {
   done = 0,
   exp_position = 1,
   exp_parameter = 2,
   exp_color = 3,
   exp_depth = 4,
   exp_type = 7,
   vm_type = 1 << 4,
   lgkm_type = 1 << 5,
};

/* TODO: replace this by architecture dependant hardware limits */
uint16_t max_vm_cnt = 0xFF;
uint16_t max_exp_cnt = 0xFF;
uint16_t max_lgkm_cnt = 0xFF;

struct wait_entry {
   uint16_t type; /* use wait_type notion */
   uint16_t vm_cnt;
   uint16_t exp_cnt;
   uint16_t lgkm_cnt;
   wait_entry(wait_type t, uint8_t vm, uint8_t exp, uint8_t lgkm)
           : type(t), vm_cnt(vm), exp_cnt(exp), lgkm_cnt(lgkm) {}

   bool operator==(const wait_entry& rhs) const
   {
      return type == rhs.type &&
             vm_cnt == rhs.vm_cnt &&
             exp_cnt == rhs.exp_cnt &&
             lgkm_cnt == rhs.lgkm_cnt;
   }
};

struct wait_ctx {
   uint16_t vm_cnt = 0;
   uint16_t exp_cnt = 0;
   uint16_t lgkm_cnt = 0;

   std::unordered_map<uint8_t,wait_entry> vgpr_map;
   std::unordered_map<uint8_t,wait_entry> sgpr_map;

   void join(wait_ctx* other)
   {
      exp_cnt = std::max(exp_cnt, other->exp_cnt);
      vm_cnt = std::max(vm_cnt, other->vm_cnt);
      lgkm_cnt = std::max(lgkm_cnt, other->lgkm_cnt);

      for (std::pair<uint8_t,wait_entry> entry : other->vgpr_map)
      {
         std::unordered_map<uint8_t,wait_entry>::iterator it = vgpr_map.find(entry.first);
         if (it != vgpr_map.end())
         {
            /* update entry */
            it->second.type = it->second.type | entry.second.type;
            it->second.exp_cnt = std::min(it->second.exp_cnt, entry.second.exp_cnt);
            it->second.vm_cnt = std::min(it->second.vm_cnt, entry.second.vm_cnt);
            it->second.lgkm_cnt = std::min(it->second.lgkm_cnt, entry.second.vm_cnt);
         } else {
            vgpr_map.insert(entry);
         }
      }
      for (std::pair<uint8_t,wait_entry> entry : other->sgpr_map)
      {
         std::unordered_map<uint8_t,wait_entry>::iterator it = sgpr_map.find(entry.first);
         if (it != sgpr_map.end())
         {
            /* update entry */
            it->second.type = it->second.type | entry.second.type;
            it->second.exp_cnt = std::min(it->second.exp_cnt, entry.second.exp_cnt);
            it->second.vm_cnt = std::min(it->second.vm_cnt, entry.second.vm_cnt);
            it->second.lgkm_cnt = std::min(it->second.lgkm_cnt, entry.second.vm_cnt);
         } else {
            sgpr_map.insert(entry);
         }
      }
   }

   bool operator==(const wait_ctx& rhs) const
   {
      return sgpr_map == rhs.sgpr_map &&
             vgpr_map == rhs.vgpr_map;
   }
};

uint16_t create_waitcnt_imm(uint16_t vm, uint16_t exp, uint16_t lgkm)
{
   return ((vm & 0x30) << 10) | ((lgkm & 0xF) << 8) | ((exp & 0x7) << 4) | (vm & 0xF);
}

SOPP_instruction* create_waitcnt(uint16_t imm)
{
   SOPP_instruction* waitcnt = create_instruction<SOPP_instruction>(aco_opcode::s_waitcnt, Format::SOPP, 0, 0);
   waitcnt->imm = imm;
   return waitcnt;
}

void reset_counters(wait_ctx& ctx, uint16_t types)
{
   /* update vgpr/sgpr maps */
   for (int i = 0; i < 2; i++) {
      bool sgpr_map = i == 1;
      if (sgpr_map && !(types & lgkm_type)) // the sgpr map only contains lgkm counters
         continue;
      std::unordered_map<uint8_t,wait_entry>& map = i ? ctx.sgpr_map : ctx.vgpr_map;
      std::unordered_map<uint8_t,wait_entry>::iterator it = map.begin();
      while (it != map.end())
      {
         if ((it->second.type & types) == it->second.type) {
            it = map.erase(it);
            continue;
         }

         if ((types & exp_type) && (it->second.type & exp_type)) {
            it->second.exp_cnt = max_exp_cnt;
            it->second.type &= ~exp_type;
         }
         if ((types & vm_type) && (it->second.type & vm_type)) {
            it->second.vm_cnt = max_vm_cnt;
            it->second.type &= ~vm_type;
         }
         if ((types & lgkm_type) && (it->second.type & lgkm_type)) {
            it->second.lgkm_cnt = max_lgkm_cnt;
            it->second.type &= ~lgkm_type;
         }
         it++;
      }
   }

   /* reset counters */
   if (types & exp_type)
      ctx.exp_cnt = 0;
   if (types & vm_type)
      ctx.vm_cnt = 0;
   if (types & lgkm_type)
      ctx.lgkm_cnt = 0;
}

bool writes_exec(Instruction* instr, wait_ctx& ctx)
{
   switch (instr->opcode)
   {
      case aco_opcode::s_and_saveexec_b64:
      case aco_opcode::s_or_saveexec_b64:
      case aco_opcode::s_xor_saveexec_b64:
      case aco_opcode::s_andn2_saveexec_b64:
      case aco_opcode::s_orn2_saveexec_b64:
      case aco_opcode::s_nand_saveexec_b64:
      case aco_opcode::s_nor_saveexec_b64:
      case aco_opcode::s_xnor_saveexec_b64:
      case aco_opcode::s_andn1_saveexec_b64:
      case aco_opcode::s_orn1_saveexec_b64:
      case aco_opcode::s_andn1_wrexec_b64:
      case aco_opcode::s_andn2_wrexec_b64:
         return true;
      default:
         if (instr->opcode >= aco_opcode::v_cmpx_class_f16 &&
             instr->opcode <= aco_opcode::v_cmpx_u_f64)
            return true;
   }

   switch (instr->format)
   {
      default:
         return false;
      case Format::SOP1: 
      case Format::SOP2:
      case Format::SOPK:
      case Format::VOP3:
         break;
   }

   /* check if the dst writes exec */
   for (unsigned i = 0; i < instr->definitionCount(); i++)
   {
      if ((instr->getDefinition(i).regClass() == RegClass::s2 &&
           instr->getDefinition(i).physReg().reg == 126 /* EXEC */) ||
          (instr->getDefinition(i).regClass() == RegClass::s1 &&
          (instr->getDefinition(i).physReg().reg == 126 /* EXEC_LO */ ||
           instr->getDefinition(i).physReg().reg == 127 /* EXEC_HI */ )))
         return true;
   }
   return false;
}

uint16_t writes_vgpr(Instruction* instr, wait_ctx& ctx)
{
   bool writes_vgpr = false;
   uint16_t new_exp_cnt = max_exp_cnt;
   uint16_t new_vm_cnt = max_vm_cnt;
   uint16_t new_lgkm_cnt = max_lgkm_cnt;

   for (unsigned i = 0; i < instr->definitionCount(); i++)
   {
      if (instr->getDefinition(i).getTemp().type() != RegType::vgpr)
         continue;

      /* check consecutively written vgprs */
      for (unsigned j = 0; j < instr->getDefinition(i).getTemp().size(); j++)
      {
         uint8_t reg = (uint8_t) instr->getDefinition(i).physReg().reg + j;

         std::unordered_map<uint8_t,wait_entry>::iterator it;
         it = ctx.vgpr_map.find(reg);
         if (it == ctx.vgpr_map.end())
            continue;

         /* Vector Memory reads and writes return in the order they were issued */
         if (instr->isVMEM() && it->second.type == vm_type) {
            it->second.vm_cnt = max_vm_cnt;
            it->second.type = it->second.type &= ~vm_type;
            if (it->second.type == done)
               it = ctx.vgpr_map.erase(it);
            continue;
         }
         writes_vgpr = true;
         wait_entry entry = it->second;

         /* remove all vgprs with higher counter from map */
         it = ctx.vgpr_map.begin();
         while (it != ctx.vgpr_map.end())
         {
            if (entry.type & it->second.type) {
               if ((entry.type & exp_type) && entry.exp_cnt <= it->second.exp_cnt)
               {
                  it->second.exp_cnt = max_exp_cnt;
                  it->second.type = it->second.type &= ~exp_type;
               }
               if ((entry.type & vm_type) && entry.vm_cnt <= it->second.vm_cnt)
               {
                  it->second.vm_cnt = max_vm_cnt;
                  it->second.type = it->second.type &= ~vm_type;
               }
               if ((entry.type & lgkm_type) && entry.lgkm_cnt <= it->second.lgkm_cnt)
               {
                  it->second.lgkm_cnt = max_lgkm_cnt;
                  it->second.type = it->second.type &= ~lgkm_type;
               }
               if (it->second.type == done)
                  it = ctx.vgpr_map.erase(it);
               else
                  it++;
            } else {
               it++;
            }
         }
         new_exp_cnt = std::min(new_exp_cnt, entry.exp_cnt);
         new_vm_cnt = std::min(new_vm_cnt, entry.vm_cnt);
         new_lgkm_cnt = std::min(new_lgkm_cnt, entry.lgkm_cnt);
      }
   }
   if (writes_vgpr)
   {
      /* reset counters */
      ctx.exp_cnt = std::min(ctx.exp_cnt, new_exp_cnt);
      ctx.vm_cnt = std::min(ctx.vm_cnt, new_vm_cnt);
      ctx.lgkm_cnt = std::min(ctx.lgkm_cnt, new_lgkm_cnt);
      return create_waitcnt_imm(new_vm_cnt, new_exp_cnt, new_lgkm_cnt);
   } else {
      return -1;
   }
}

uint16_t writes_sgpr(Instruction* instr, wait_ctx& ctx)
{
   bool needs_waitcnt = false;
   uint16_t new_lgkm_cnt = max_lgkm_cnt;
   for (unsigned i = 0; i < instr->definitionCount(); i++)
   {
      if (instr->getDefinition(i).getTemp().type() != RegType::sgpr)
         continue;

      /* check consecutively written sgprs */
      for (unsigned j = 0; j < instr->getDefinition(i).getTemp().size(); j++)
      {
         uint8_t reg = (uint8_t) instr->getDefinition(i).physReg().reg + j;

         std::unordered_map<uint8_t,wait_entry>::iterator it;
         it = ctx.sgpr_map.find(reg);
         if (it == ctx.sgpr_map.end())
            continue;

         needs_waitcnt = true;
         wait_entry entry = it->second;

         /* remove all vgprs with higher counter from map */
         it = ctx.sgpr_map.begin();
         while (it != ctx.sgpr_map.end())
         {
            if (entry.type & it->second.type) {
               if ((entry.type & lgkm_type) && entry.lgkm_cnt <= it->second.lgkm_cnt)
               {
                  it->second.lgkm_cnt = max_lgkm_cnt;
                  it->second.type = it->second.type &= ~lgkm_type;
               }
               if (it->second.type == done)
                  it = ctx.sgpr_map.erase(it);
               else
                  it++;
            } else {
               it++;
            }
         }
         new_lgkm_cnt = std::min(new_lgkm_cnt, entry.lgkm_cnt);
      }
   }
   if (needs_waitcnt) {
      /* reset counter */
      ctx.lgkm_cnt = std::min(ctx.lgkm_cnt, new_lgkm_cnt);
      return create_waitcnt_imm(max_vm_cnt, max_exp_cnt, new_lgkm_cnt);
   } else {
      return -1;
   }
}

uint16_t uses_gpr(Instruction* instr, wait_ctx& ctx)
{
   bool needs_waitcnt = false;
   uint16_t new_lgkm_cnt = max_lgkm_cnt;
   uint16_t new_vm_cnt = max_vm_cnt;
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (instr->getOperand(i).isConstant() || instr->getOperand(i).isUndefined())
         continue;

      if (instr->getOperand(i).getTemp().type() == RegType::sgpr) {
         /* check consecutively read sgprs */
         for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
         {
            uint8_t reg = (uint8_t) instr->getOperand(i).physReg().reg + j;

            std::unordered_map<uint8_t,wait_entry>::iterator it;
            it = ctx.sgpr_map.find(reg);
            if (it == ctx.sgpr_map.end())
               continue;

            needs_waitcnt = true;
            wait_entry entry = it->second;

            /* remove all sgprs with higher counter from map */
            it = ctx.sgpr_map.begin();
            while (it != ctx.sgpr_map.end())
            {
               if (entry.type & it->second.type) {
                  if ((entry.type & lgkm_type) && entry.lgkm_cnt <= it->second.lgkm_cnt)
                  {
                     it->second.lgkm_cnt = max_lgkm_cnt;
                     it->second.type = it->second.type &= ~lgkm_type;
                  }
                  if (it->second.type == done)
                     it = ctx.sgpr_map.erase(it);
                  else
                     it++;
               } else {
                  it++;
               }
            }
            new_lgkm_cnt = std::min(new_lgkm_cnt, entry.lgkm_cnt);
         }
      } else {
         /* check consecutively read vgprs */
         for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
         {
            uint8_t reg = (uint8_t) instr->getOperand(i).physReg().reg + j;

            std::unordered_map<uint8_t,wait_entry>::iterator it;
            it = ctx.vgpr_map.find(reg);
            if (it == ctx.vgpr_map.end())
               continue;

            needs_waitcnt = true;
            wait_entry entry = it->second;

            /* remove all vgprs with higher counter from map */
            it = ctx.vgpr_map.begin();
            while (it != ctx.vgpr_map.end())
            {
               if (entry.type & it->second.type) {
                  if ((entry.type & vm_type) && entry.vm_cnt <= it->second.vm_cnt)
                  {
                     it->second.vm_cnt = max_vm_cnt;
                     it->second.type = it->second.type &= ~vm_type;
                  }
                  if ((entry.type & lgkm_type) && entry.lgkm_cnt <= it->second.lgkm_cnt)
                  {
                     it->second.lgkm_cnt = max_lgkm_cnt;
                     it->second.type = it->second.type &= ~lgkm_type;
                  }
                  if (it->second.type == done)
                     it = ctx.vgpr_map.erase(it);
                  else
                     it++;
               } else {
                  it++;
               }
            }
            new_vm_cnt = std::min(new_vm_cnt, entry.vm_cnt);
            new_lgkm_cnt = std::min(new_lgkm_cnt, entry.lgkm_cnt);
         }
      }
   }
   if (needs_waitcnt)
   {
      /* reset counter */
      ctx.vm_cnt = std::min(ctx.vm_cnt, new_vm_cnt);
      ctx.lgkm_cnt = std::min(ctx.lgkm_cnt, new_lgkm_cnt);
      return create_waitcnt_imm(new_vm_cnt, max_exp_cnt, new_lgkm_cnt);
   } else {
      return -1;
   }
}

// TODO: introduce more fine-grained counters to differenciate
// the types of memory operations we want to wait for
uint16_t emit_memory_barrier(Instruction* instr, wait_ctx& ctx) {
   uint16_t vm_cnt, lgkm_cnt;
   switch (instr->opcode) {
      case aco_opcode::p_memory_barrier_all:
         vm_cnt = ctx.vm_cnt ? 0 : -1;
         lgkm_cnt = ctx.lgkm_cnt ? 0 : -1;
         if (ctx.vm_cnt || ctx.lgkm_cnt) {
            reset_counters(ctx, vm_type | lgkm_type);
            return create_waitcnt_imm(vm_cnt, -1, lgkm_cnt);
         }
         break;
      case aco_opcode::p_memory_barrier_atomic:
      case aco_opcode::p_memory_barrier_buffer:
      case aco_opcode::p_memory_barrier_image:
         if (ctx.vm_cnt) {
            reset_counters(ctx, vm_type);
            return create_waitcnt_imm(0, -1, -1);
         }
         break;
      case aco_opcode::p_memory_barrier_shared:
         if (ctx.lgkm_cnt) {
            reset_counters(ctx, lgkm_type);
            return create_waitcnt_imm(-1, -1, 0);
         }
         break;
      default:
         unreachable("emit_memory_barrier() should only be called with PSEUDO_BARRIER instructions.");
   }
   return -1;
}

Instruction* kill(Instruction* instr, wait_ctx& ctx)
{
   if (instr->format == Format::PSEUDO)
      return nullptr;

   uint16_t imm = 0xFFFF;
   if (ctx.exp_cnt && writes_exec(instr, ctx))
   {
      reset_counters(ctx, exp_type);
      imm = create_waitcnt_imm(-1, 0, -1);
   }
   if (ctx.exp_cnt || ctx.vm_cnt || ctx.lgkm_cnt)
   {
      if (instr->format == Format::PSEUDO_BARRIER) {
         imm &= emit_memory_barrier(instr, ctx);
      } else {
         imm &= uses_gpr(instr, ctx);
         imm &= writes_vgpr(instr, ctx);
         imm &= writes_sgpr(instr, ctx);
      }
   }
   if (imm != 0xFFFF)
      return create_waitcnt(imm);
   else
      return nullptr;
}

bool gen(Instruction* instr, wait_ctx& ctx)
{
   switch(instr->format) {
   case Format::EXP: {
      Export_instruction* exp_instr = static_cast<Export_instruction*>(instr);
      wait_type t;
      if (exp_instr->dest <= 7) {
         t = wait_type::exp_color;
      } else if (exp_instr->dest == 8) {
         t = wait_type::exp_depth;
      } else if (exp_instr->dest == 9) { /* null */
         t = wait_type::exp_color;
      } else if (exp_instr->dest <= 15) {
         t = wait_type::exp_position;
      } else {
         t = wait_type::exp_parameter;
      }
      /* increase counter for all entries of same wait_type */
      for (auto& e : ctx.vgpr_map)
      {
         if ((e.second.type & t) == t)
            e.second.exp_cnt++;
      }

      /* insert new entries for exported vgprs */
      unsigned idx = 0;
      for (unsigned i = 0; i < exp_instr->num_operands; i++)
      {
         if (exp_instr->enabled_mask & (1 << i)) {
            auto it = ctx.vgpr_map.emplace(exp_instr->getOperand(idx++).physReg().reg,
                                           wait_entry(t, max_vm_cnt, 0, max_lgkm_cnt)).first;
            it->second.exp_cnt = 0;

         }
      }
      ctx.exp_cnt++;
      return true;
   }
   case Format::SMEM: {
      ctx.lgkm_cnt++;
      if (instr->num_definitions) {
         for (unsigned i = 0; i < instr->getDefinition(0).size(); i++)
         {
            ctx.sgpr_map.emplace(instr->getDefinition(0).physReg().reg + i,
            wait_entry(lgkm_type, max_vm_cnt, max_exp_cnt, 0));
         }
         return true;
      }
      break;
   }
   case Format::DS: {
      // TODO: check if reads and writes are in-order
      /* the counter is also used as check for membars, thus we need it also on writes */
      ctx.lgkm_cnt++;
      if (instr->num_definitions) {
         for (unsigned i = 0; i < instr->getDefinition(0).size(); i++)
         {
            ctx.vgpr_map.emplace(instr->getDefinition(0).physReg().reg + i,
            wait_entry(lgkm_type, max_vm_cnt, max_exp_cnt, 0));
         }
         return true;
      }
      break;
   }
   case Format::MUBUF:
   case Format::MIMG: {
      /* increase counter for all entries of same wait_type */
      for (auto& e : ctx.vgpr_map)
      {
         if (e.second.type == vm_type)
            e.second.vm_cnt++;
      }
      ctx.vm_cnt++;
      if (instr->num_definitions) {
         for (unsigned i = 0; i < instr->getDefinition(0).size(); i++)
         {
            ctx.vgpr_map.emplace(instr->getDefinition(0).physReg().reg + i,
            wait_entry(vm_type, 0, max_exp_cnt, max_lgkm_cnt));
         }
      } else if (instr->num_operands == 4) {
         for (unsigned i = 0; i < instr->getOperand(3).size(); i++)
         {
            ctx.vgpr_map.emplace(instr->getOperand(3).physReg().reg + i,
            wait_entry(vm_type, 0, max_exp_cnt, max_lgkm_cnt));
         }
      }
      return true;
   }
   default:
      return false;
   }
   return false;
}

bool handle_block(Block* block, wait_ctx& ctx)
{
   bool has_gen = false;
   std::vector<aco_ptr<Instruction>> new_instructions;
   for(auto& instr : block->instructions)
   {
      Instruction* wait_instr;
      if ((wait_instr = kill(instr.get(), ctx)))
         new_instructions.emplace_back(aco_ptr<Instruction>(wait_instr));

      has_gen = gen(instr.get(), ctx) | has_gen;
      new_instructions.emplace_back(std::move(instr));
   }

   /* check if this block is at the end of a loop */
   for (Block* succ : block->linear_successors) {
      /* eliminate any remaining counters */
      if (succ->index <= block->index && (ctx.vm_cnt || ctx.exp_cnt || ctx.lgkm_cnt)) {
         uint16_t imm = create_waitcnt_imm(ctx.vm_cnt ? 0 : -1, ctx.exp_cnt ? 0 : -1, ctx.lgkm_cnt ? 0 : -1);
         auto it = std::prev(new_instructions.end());
         new_instructions.insert(it, aco_ptr<Instruction>{create_waitcnt(imm)});
         ctx = wait_ctx();
         break;
      }
   }
   block->instructions.swap(new_instructions);
   return has_gen;
}

void insert_wait_states(Program* program)
{
   wait_ctx out_ctx[program->blocks.size()]; /* per BB ctx */
   for (unsigned i = 0; i < program->blocks.size(); i++)
      out_ctx[i] = wait_ctx();

   for (unsigned i = 0; i < program->blocks.size(); i++)
   {
      Block* current = program->blocks[i].get();
      wait_ctx& in = out_ctx[current->index];
      for (Block* b : current->linear_predecessors)
         in.join(&out_ctx[b->index]);

      handle_block(current, in);
   }
}

}

