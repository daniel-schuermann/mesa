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
uint8_t max_vm_cnt = 63; /* Vega */
uint8_t max_exp_cnt = 7;
uint8_t max_lgkm_cnt = 15;

struct wait_entry {
   uint8_t type; /* use wait_type notion */
   uint8_t vm_cnt;
   uint8_t exp_cnt;
   uint8_t lgkm_cnt;
   wait_entry(wait_type t, uint8_t vm, uint8_t exp, uint8_t lgkm)
           : type(t), vm_cnt(vm), lgkm_cnt(lgkm) {}

   bool operator==(const wait_entry& rhs) const
   {
      return type == rhs.type &&
             vm_cnt == rhs.vm_cnt &&
             exp_cnt == rhs.exp_cnt &&
             lgkm_cnt == rhs.lgkm_cnt;
   }
};

struct wait_ctx {
   uint8_t vm_cnt = 0;
   uint8_t exp_cnt = 0;
   uint8_t lgkm_cnt = 0;

   std::unordered_map<uint8_t,wait_entry> vgpr_map;

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
   }

   bool operator==(const wait_ctx& rhs) const
   {
      return vm_cnt == rhs.vm_cnt &&
             exp_cnt == rhs.exp_cnt &&
             lgkm_cnt == rhs.lgkm_cnt &&
             vgpr_map == rhs.vgpr_map;
   }
};

uint16_t create_waitcnt_imm(uint8_t vm, uint8_t exp, uint8_t lgkm)
{
   return (vm & 48) << 14 | (lgkm & 15) << 8 | (exp & 7) << 4 | (vm & 15);
}

SOPP_instruction* create_waitcnt(uint16_t imm)
{
   SOPP_instruction* waitcnt = create_instruction<SOPP_instruction>(aco_opcode::s_waitcnt, Format::SOPP, 0, 0);
   waitcnt->imm = imm;
   return waitcnt;
}

void reset_exp_cnt(wait_ctx& ctx)
{
   std::unordered_map<uint8_t,wait_entry>::iterator it = ctx.vgpr_map.begin();
   while (it != ctx.vgpr_map.end())
   {
      if (it->second.type < exp_type) {
         it = ctx.vgpr_map.erase(it);
      } else if (it->second.type & exp_type) {
         it->second.exp_cnt = max_exp_cnt;
         it->second.type &= ~exp_type;
         it++;
      } else {
         it++;
      }
   }
   /* reset counter */
   ctx.exp_cnt = 0;
   return;
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
      case Format::VOP3B:
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
   uint8_t new_exp_cnt = max_exp_cnt;
   uint8_t new_vm_cnt = max_vm_cnt;
   uint8_t new_lgkm_cnt = max_lgkm_cnt;

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


Instruction* kill(Instruction* instr, wait_ctx& ctx)
{
   uint16_t imm = -1;
   if (ctx.exp_cnt && writes_exec(instr, ctx))
   {
      reset_exp_cnt(ctx);
      imm = create_waitcnt_imm(-1, 0, -1);
   }
   if (ctx.exp_cnt || ctx.vm_cnt || ctx.lgkm_cnt)
   {
      imm &= writes_vgpr(instr, ctx);
   }
   if (imm != -1)
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
      for (std::pair<uint8_t,wait_entry> e : ctx.vgpr_map)
      {
         if ((e.second.type & t) == t)
            e.second.exp_cnt++;
      }

      /* insert new entries for exported vgprs */
      for (unsigned i = 0; i < exp_instr->num_operands; i++)
      {
         ctx.vgpr_map.emplace(exp_instr->getOperand(i).physReg().reg,
                 wait_entry(t, max_vm_cnt, 0, max_lgkm_cnt));
      }
      ctx.exp_cnt++;
      return true;
   }
   // TODO: cases which generate vm_cnt and lgkm_cnt
   default:
      return false;
   }
   return false;
}

bool handle_block(Block* block, wait_ctx& ctx)
{
   bool has_gen = false;
   std::vector<std::unique_ptr<Instruction>> new_instructions;
   for(auto& instr : block->instructions)
   {
      Instruction* wait_instr;
      if ((wait_instr = kill(instr.get(), ctx)))
         new_instructions.emplace_back(std::unique_ptr<Instruction>(wait_instr));

      has_gen = gen(instr.get(), ctx) || has_gen;
      new_instructions.emplace_back(std::move(instr));
   }
   block->instructions = std::move(new_instructions);
   return has_gen;
}

void insert_wait_states(Program* program)
{
   wait_ctx out_ctx[program->blocks.size()]; /* per BB ctx */
   unsigned i = 0; /* i represents the worklist range [i,num_BBs) */

   while (i < program->blocks.size())
   {
      Block* current = program->blocks[i].get();
      wait_ctx in;
      for (Block* b : current->linear_predecessors)
      {
         in.join(&out_ctx[b->index]);
      }

      handle_block(current, in);

      if (out_ctx[current->index] == in) /* if old_out == new_out */
      {
         i++;
      } else {
         /* we re-iterate loops if the out_ctx has changed */
         for (Block* b : current->linear_successors)
            i = std::min(i, b->index);
         out_ctx[current->index] = in;
      }
   }
}

}