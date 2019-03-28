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
 *    Bas Nieuwenhuizen (bas@basnieuwenhuizen.nl)
 *
 */

#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <functional>
#include <bitset>

#include "aco_ir.h"
#include "sid.h"


namespace aco {
namespace {

struct ra_ctx {
   Program* program;
   std::unordered_map<unsigned, std::pair<PhysReg, RegClass>> assignments;
   unsigned max_used_sgpr = 0;
   unsigned max_used_vgpr = 0;

   ra_ctx(Program* program) : program(program) {}
};


/* forward declarations */
PhysReg get_reg(ra_ctx& ctx,
                std::array<uint32_t, 512>& reg_file,
                RegClass rc,
                std::vector<std::pair<Operand, Definition>>& parallelcopies,
                aco_ptr<Instruction>& instr);

std::pair<PhysReg, bool> get_reg_impl(ra_ctx& ctx,
                                      std::array<uint32_t, 512>& reg_file,
                                      std::vector<std::pair<Operand, Definition>>& parallelcopies,
                                      uint32_t lb, uint32_t ub,
                                      uint32_t size, uint32_t stride,
                                      uint32_t num_moves,
                                      bool is_sgpr);

bool get_reg_for_copies(ra_ctx& ctx,
                      std::array<uint32_t, 512>& reg_file,
                      std::vector<std::pair<Operand, Definition>>& parallelcopies,
                      std::set<unsigned> vars,
                      uint32_t lb, uint32_t ub,
                      uint32_t remaining_moves,
                      bool is_sgpr)
{
   for (unsigned id : vars) {
      std::pair<PhysReg, RegClass> var = ctx.assignments[id];
      uint32_t stride = 1;
      if (is_sgpr) {
         if (sizeOf(var.second) == 2)
            stride = 2;
         if (sizeOf(var.second) > 3)
            stride = 4;
      }
      unsigned num_moves = 0;
      std::pair<PhysReg, bool> res = get_reg_impl(ctx, reg_file, parallelcopies, lb, ub, sizeOf(var.second), stride, num_moves, is_sgpr);
      while (!res.second && remaining_moves > 0 && num_moves < sizeOf(var.second)) {
         remaining_moves--;
         num_moves++;
         res = get_reg_impl(ctx, reg_file, parallelcopies, lb, ub, sizeOf(var.second), stride, num_moves, is_sgpr);
      }
      if (!res.second)
         return false;

      /* mark the area as blocked */
      for (unsigned i = res.first.reg; i < res.first.reg + sizeOf(var.second); i++)
         reg_file[i] = 0xFFFF;

      /* create parallelcopy pair (without definition id) */
      Temp tmp = Temp(id, var.second);
      Operand pc_op = Operand(tmp);
      pc_op.setFixed(var.first);
      Definition pc_def = Definition(res.first, pc_op.regClass());
      parallelcopies.emplace_back(pc_op, pc_def);
   }
   return true;
}


void adjust_max_used_regs(ra_ctx& ctx, RegClass rc, unsigned reg)
{
   unsigned max_addressible_sgpr = ctx.program->chip_class >= VI ? 102 : 104;
   unsigned size = sizeOf(rc);
   if (typeOf(rc) == vgpr) {
      unsigned hi = reg - 256 + size - 1;
      ctx.max_used_vgpr = std::max(ctx.max_used_vgpr, hi);
   } else if (reg + sizeOf(rc) <= max_addressible_sgpr) {
      unsigned hi = reg + size - 1;
      ctx.max_used_sgpr = std::max(ctx.max_used_sgpr, std::min(hi, max_addressible_sgpr));
   }
}


std::pair<PhysReg, bool> get_reg_impl(ra_ctx& ctx,
                                      std::array<uint32_t, 512>& reg_file,
                                      std::vector<std::pair<Operand, Definition>>& parallelcopies,
                                      uint32_t lb, uint32_t ub,
                                      uint32_t size, uint32_t stride,
                                      uint32_t num_moves,
                                      bool is_sgpr)
{
   assert(num_moves <= size); // FIXME: extend this algorithm
   unsigned reg_lo = lb;
   unsigned reg_hi = lb + size - 1;
   /* trivial case: without moves */
   if (num_moves == 0) {
      bool found = false;
      while (!found && reg_lo + size <= ub) {
         if (reg_file[reg_lo] != 0) {
            reg_lo += stride;
            continue;
         }
         reg_hi = reg_lo + size - 1;
         found = true;
         for (unsigned reg = reg_lo + 1; reg <= reg_hi; reg++) {
            if (reg_file[reg] != 0) {
               found = false;
               break;
            }
         }
         if (found) {
            adjust_max_used_regs(ctx, is_sgpr ? s1 : v1, reg_hi);
            return std::make_pair(PhysReg{reg_lo}, true);
         }
         while (reg_lo <= reg_hi)
            reg_lo += stride;
      }
      return std::make_pair(PhysReg{0}, false);
   }

   /* we use a sliding window to find potential positions */
   for (reg_lo = lb, reg_hi = lb + size - 1; reg_hi < ub; reg_lo += stride, reg_hi += stride) {
      /* first check the edges: this is what we have to fix to allow for num_moves > size */
      if (reg_lo > lb + 1 && reg_file[reg_lo] != 0 && reg_file[reg_lo] == reg_file[reg_lo - 1])
         continue;
      if (reg_hi < ub - 1 && reg_file[reg_hi] != 0 && reg_file[reg_hi] == reg_file[reg_hi + 1])
         continue;

      /* second, check that we have at most k=num_moves elements in the window
       * and no element is larger than the currently processed one */
      unsigned k = 0;
      std::set<unsigned> vars;
      bool stop = false;
      for (unsigned j = reg_lo; j <= reg_hi; j++) {
         if (reg_file[j] == 0)
            continue;
         k++;
         /* 0xFFFF signals that this area is already blocked! */
         if (reg_file[j] == 0xFFFF || k > num_moves) {
            stop = true;
            break;
         }
         if (sizeOf(ctx.assignments[reg_file[j]].second) >= size) {
            stop = true;
            break;
         }
         /* we cannot split live ranges of linear vgprs */
         if (ctx.assignments[reg_file[j]].second & (1 << 6)) {
            stop = true;
            break;
         }
         vars.emplace(reg_file[j]);
      }
      if (stop)
         continue;

      /* now, we have a list of vars, we want to move away from the current slot */
      /* copy the current register file */
      std::array<uint32_t, 512> register_file = reg_file;
      /* mark the area as blocked: [reg_lo, reg_hi] */
      for (unsigned j = reg_lo; j <= reg_hi; j++)
         register_file[j] = 0xFFFF;

      std::vector<std::pair<Operand, Definition>> parallelcopy;

      bool success = get_reg_for_copies(ctx, register_file, parallelcopy, vars, lb, ub, num_moves - k, is_sgpr);

      if (success) {
         /* if everything worked out: insert parallelcopies, release [reg_lo,reg_hi], copy back reg_file */
         parallelcopies.insert(parallelcopies.end(), parallelcopy.begin(), parallelcopy.end());
         reg_file = register_file;
         adjust_max_used_regs(ctx, is_sgpr ? s1 : v1, reg_hi);
         return std::make_pair(PhysReg{reg_lo}, true);
      }
   }

   return std::make_pair(PhysReg{0}, false);
}


std::pair<PhysReg, bool> get_reg_helper(ra_ctx& ctx,
                                        std::array<uint32_t, 512>& reg_file,
                                        std::vector<std::pair<Operand, Definition>>& pc,
                                        aco_ptr<Instruction>& instr,
                                        RegClass rc,
                                        uint32_t lb, uint32_t ub,
                                        uint32_t size, uint32_t stride,
                                        uint32_t num_moves)
{
   if (num_moves == 0)
      return get_reg_impl(ctx, reg_file, pc, lb, ub, size, stride, 0, typeOf(rc) == sgpr);

   std::pair<PhysReg, bool> res = get_reg_impl(ctx, reg_file, pc, lb, ub, size, stride, num_moves, typeOf(rc) == sgpr);
   if (!res.second)
      return res;

   /* we set the definition regs == 0. the actual caller is responsible for correct setting */
   for (unsigned i = 0; i < size; i++)
      reg_file[res.first.reg + i] = 0;

   /* allocate id's and rename operands: this is done transparently here */
   for (std::pair<Operand, Definition>& copy : pc) {
      /* the definitions with id are not from this function and already handled */
      if (!copy.second.isTemp()) {
         copy.second.setTemp(Temp(ctx.program->allocateId(), copy.second.regClass()));
         ctx.assignments[copy.second.tempId()] = {copy.second.physReg(), copy.second.regClass()};
         for (unsigned i = copy.second.physReg().reg; i < copy.second.physReg().reg + copy.second.size(); i++)
            reg_file[i] = copy.second.tempId();
         /* check if we moved an operand */
         for (unsigned i = 0; i < instr->num_operands; i++) {
            if (!instr->getOperand(i).isTemp())
               continue;
            if (instr->getOperand(i).tempId() == copy.first.tempId()) {
               instr->getOperand(i).setTemp(copy.second.getTemp());
               instr->getOperand(i).setFixed(copy.second.physReg());
            }
         }
      }
   }

   /* it might happen that something was moved to the position of an killed operand */
   // TODO: it would be better to try to find another register than to insert more moves */
   for (unsigned i = 0; i < instr->num_operands; i++) {
      Operand op = instr->getOperand(i);
      if (!op.isTemp() || !op.isFixed() || op.getTemp().type() != typeOf(rc) || !op.isFirstKill())
         continue;

      for (unsigned j = 0; j < op.size(); j++) {
         /* if that is the case, we have to find another position for this operand */
         if (reg_file[op.physReg().reg + j] != 0) {
            Definition def = Definition(ctx.program->allocateId(), op.regClass());

            /* re-enable other killed operands */
            for (unsigned k = 0; k < instr->num_operands; k++) {
               Operand& op_ = instr->getOperand(k);
               if (!op_.isTemp() || !op_.isFirstKill() || !op_.isFixed() || op_.getTemp() == op.getTemp())
                  continue;

               assert(op_.isFixed());
               for (unsigned r = op_.physReg().reg; r < op_.physReg().reg + op_.size(); r++) {
                  if (!reg_file[r])
                     reg_file[r] = 0xFFFF;
               }
            }

            /* prevent potential infinite recursion */
            for (unsigned k = 0; k < instr->num_operands; k++) {
               if (instr->getOperand(k).tempId() == op.tempId())
                  instr->getOperand(k).setFixed((PhysReg){(unsigned)-1});
            }

            PhysReg reg = get_reg(ctx, reg_file, op.regClass(), pc, instr);

            /* disable the killed operands */
            for (unsigned k = 0; k < instr->num_operands; k++) {
               Operand& op = instr->getOperand(k);
               if (!op.isTemp() || !op.isFirstKill() || !op.isFixed())
                  continue;
               for (unsigned r = op.physReg().reg; r < op.physReg().reg + op.size(); r++) {
                  if (reg_file[r] == 0xFFFF)
                     reg_file[r] = 0;
               }
            }

            def.setFixed(reg);
            pc.emplace_back(op, def);
            /* update operands */
            for (unsigned k = 0; k < instr->num_operands; k++) {
               if (instr->getOperand(k).tempId() != op.tempId())
                  continue;
               instr->getOperand(k).setTemp(def.getTemp());
               instr->getOperand(k).setFixed(reg);
            }
            break;
         }
      }
   }

   return res;
}

std::pair<PhysReg, bool> get_reg_vec(ra_ctx& ctx,
                                     std::array<uint32_t, 512>& reg_file,
                                     std::vector<std::pair<Operand, Definition>>& pc,
                                     RegClass rc)
{
   uint32_t size = sizeOf(rc);
   uint32_t stride = 1;
   uint32_t lb, ub;
   if (typeOf(rc) == vgpr) {
      lb = 256;
      ub = 256 + ctx.program->max_vgpr;
   } else {
      lb = 0;
      ub = ctx.program->max_sgpr;
      if (size == 2)
         stride = 2;
      else if (size >= 4)
         stride = 4;
   }
   return get_reg_impl(ctx, reg_file, pc, lb, ub, size, stride, 0, typeOf(rc) == sgpr);
}


bool get_reg_specified(ra_ctx& ctx,
                       std::array<uint32_t, 512>& reg_file,
                       RegClass rc,
                       std::vector<std::pair<Operand, Definition>>& parallelcopies,
                       aco_ptr<Instruction>& instr,
                       PhysReg reg,
                       uint32_t num_moves)
{
   uint32_t size = sizeOf(rc);
   assert(num_moves <= size);
   uint32_t stride = 1;
   uint32_t lb, ub;
   bool is_sgpr = false;

   if (typeOf(rc) == vgpr) {
      lb = 256;
      ub = 256 + ctx.program->max_vgpr;
   } else {
      is_sgpr = true;
      if (size == 2)
         stride = 2;
      else if (size >= 4)
         stride = 4;
      if (reg.reg % stride != 0)
         return false;
      lb = 0;
      ub = ctx.program->max_sgpr;
   }

   uint32_t reg_lo = reg.reg;
   uint32_t reg_hi = reg.reg + (size - 1);

   if (reg_lo < lb || reg_hi >= ub || reg_lo > reg_hi)
      return false;

   if (num_moves == 0) {
      for (unsigned i = reg_lo; i <= reg_hi; i++) {
         if (reg_file[i] != 0)
            return false;
      }
      adjust_max_used_regs(ctx, rc, reg_lo);
      return true;
   }

   /* first check the edges: this is what we have to fix to allow for num_moves > size */
   if (reg_lo > lb + 1 && reg_file[reg_lo] != 0 && reg_file[reg_lo] == reg_file[reg_lo - 1])
      return false;
   if (reg_hi < ub - 1 && reg_file[reg_hi] != 0 && reg_file[reg_hi] == reg_file[reg_hi + 1])
      return false;

   /* second, check that we have at most k=num_moves elements in the window
    * and no element is larger than the currently processed one */
   unsigned k = 0;
   std::set<unsigned> vars;
   for (unsigned i = reg_lo; i <= reg_hi; i++) {
      if (reg_file[i] == 0)
         continue;
      k++;

      assert(reg_file[i] != 0xFFFF);

      if (k > num_moves ||
          sizeOf(ctx.assignments[reg_file[i]].second) >= size ||
          ctx.assignments[reg_file[i]].second & (1 << 6))
         return false;

      vars.emplace(reg_file[i]);
   }

   /* now, we have a list of vars, we want to move away from the current slot */
   /* copy the current register file */
   std::array<uint32_t, 512> register_file = reg_file;
   /* mark the area as blocked: [reg_lo, reg_hi] */
   for (unsigned j = reg_lo; j <= reg_hi; j++)
      register_file[j] = 0xFFFF;

   std::vector<std::pair<Operand, Definition>> parallelcopy;
   bool success = get_reg_for_copies(ctx, register_file, parallelcopy, vars, lb, ub, num_moves - k, is_sgpr);

   if (success) {
      for (unsigned i = reg_lo; i < reg_lo + size; i++)
         register_file[i] = 0;

      /* check that no variable was moved to a killed operand's register */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         Operand op = instr->getOperand(i);
         if (!op.isTemp() || op.getTemp().type() != typeOf(rc) || !op.isFirstKill())
            continue;

         for (unsigned j = 0; j < op.size(); j++) {
            if (register_file[op.physReg().reg + j] != 0)
               return false;
         }
      }

      /* if everything worked out: insert parallelcopies, release [reg_lo,reg_hi], copy back reg_file */
      reg_file = register_file;

      adjust_max_used_regs(ctx, rc, reg_lo);
      parallelcopies.insert(parallelcopies.end(), parallelcopy.begin(), parallelcopy.end());
      /* allocate id's and rename operands: this is done transparently here */
      for (std::pair<Operand, Definition>& copy : parallelcopies) {
         /* the definitions with id are not from this function and already handled */
         if (!copy.second.isTemp()) {
            copy.second.setTemp(Temp(ctx.program->allocateId(), copy.second.regClass()));
            ctx.assignments[copy.second.tempId()] = {copy.second.physReg(), copy.second.regClass()};
            for (unsigned i = copy.second.physReg().reg; i < copy.second.physReg().reg + copy.second.size(); i++)
               reg_file[i] = copy.second.tempId();
            /* check if we moved an operand */
            for (unsigned i = 0; i < instr->num_operands; i++) {
               if (!instr->getOperand(i).isTemp())
                  continue;
               if (instr->getOperand(i).tempId() == copy.first.tempId()) {
                  instr->getOperand(i).setTemp(copy.second.getTemp());
                  instr->getOperand(i).setFixed(copy.second.physReg());
               }
            }
         }
      }

      return true;
   } else {
      return false;
   }
}


PhysReg get_reg(ra_ctx& ctx,
                std::array<uint32_t, 512>& reg_file,
                RegClass rc,
                std::vector<std::pair<Operand, Definition>>& parallelcopies,
                aco_ptr<Instruction>& instr)
{
   uint32_t size = sizeOf(rc);
   uint32_t stride = 1;
   uint32_t lb, ub;
   if (typeOf(rc) == vgpr) {
      lb = 256;
      ub = 256 + ctx.program->max_vgpr;
   } else {
      lb = 0;
      ub = ctx.program->max_sgpr;
      if (size == 2)
         stride = 2;
      else if (size >= 4)
         stride = 4;
   }

   for (unsigned k = 0; k <= size; k++) {
      std::pair<PhysReg, bool> res = get_reg_helper(ctx, reg_file, parallelcopies, instr, rc, lb, ub, size, stride, k);
      if (res.second)
         return res.first;
   }

   unreachable("did not find a register");
}

} /* end namespace */


void register_allocation(Program *program, std::vector<std::set<Temp>> live_out_per_block)
{
   ra_ctx ctx(program);

   std::vector<std::unordered_map<unsigned, Temp>> renames(program->blocks.size());
   std::map<unsigned, Temp> orig_names;

   struct phi_info {
      Instruction* phi;
      unsigned block_idx;
      std::set<Instruction*> uses;
   };

   bool filled[program->blocks.size()];
   bool sealed[program->blocks.size()];
   memset(filled, 0, sizeof filled);
   memset(sealed, 0, sizeof sealed);
   std::vector<std::vector<Instruction*>> incomplete_phis(program->blocks.size());
   std::map<unsigned, phi_info> phi_map;
   std::map<unsigned, unsigned> affinities;
   std::function<Temp(Temp,Block*)> read_variable;
   std::function<Temp(Temp,Block*)> handle_live_in;
   std::function<Temp(std::map<unsigned, phi_info>::iterator)> try_remove_trivial_phi;

   read_variable = [&](Temp val, Block* block) -> Temp {
      std::unordered_map<unsigned, Temp>::iterator it = renames[block->index].find(val.id());
      assert(it != renames[block->index].end());
      return it->second;
   };

   handle_live_in = [&](Temp val, Block *block) -> Temp {
      std::vector<Block*>& preds = val.is_linear() ? block->linear_predecessors : block->logical_predecessors;
      if (preds.size() == 0 && block->index != 0) {
         renames[block->index][val.id()] = val;
         return val;
      }
      assert(preds.size() > 0);

      Temp new_val;
      if (!sealed[block->index]) {
         /* consider rename from already processed predecessor */
         Temp tmp = read_variable(val, preds[0]);

         /* if the block is not sealed yet, we create an incomplete phi (which might later get removed again) */
         new_val = Temp{program->allocateId(), val.regClass()};
         aco_opcode opcode = val.is_linear() ? aco_opcode::p_linear_phi : aco_opcode::p_phi;
         aco_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
         phi->getDefinition(0) = Definition(new_val);
         for (unsigned i = 0; i < preds.size(); i++)
            phi->getOperand(i) = Operand(val);
         affinities[new_val.id()] = tmp.id();

         phi_map.emplace(new_val.id(), phi_info{phi.get(), block->index});
         incomplete_phis[block->index].emplace_back(phi.get());
         block->instructions.insert(block->instructions.begin(), std::move(phi));

      } else if (preds.size() == 1) {
         /* if the block has only one predecessor, just look there for the name */
         new_val = read_variable(val, preds[0]);
      } else {
         /* there are multiple predecessors and the block is sealed */
         Temp ops[preds.size()];

         /* we start assuming that the name is the same from all predecessors */
         renames[block->index][val.id()] = val;
         bool needs_phi = false;

         /* get the rename from each predecessor and check if they are the same */
         for (unsigned i = 0; i < preds.size(); i++) {
            ops[i] = read_variable(val, preds[i]);
            if (i == 0)
               new_val = ops[i];
            else
               needs_phi |= !(new_val == ops[i]);
         }

         if (needs_phi) {
            /* the variable has been renamed differently in the predecessors: we need to insert a phi */
            aco_opcode opcode = val.is_linear() ? aco_opcode::p_linear_phi : aco_opcode::p_phi;
            aco_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
            new_val = Temp{program->allocateId(), val.regClass()};
            phi->getDefinition(0) = Definition(new_val);
            for (unsigned i = 0; i < preds.size(); i++) {
               phi->getOperand(i) = Operand(ops[i]);
               phi->getOperand(i).setFixed(ctx.assignments[ops[i].id()].first);
               affinities[new_val.id()] = ops[i].id();
            }
            phi_map.emplace(new_val.id(), phi_info{phi.get(), block->index});
            block->instructions.insert(block->instructions.begin(), std::move(phi));
         }
      }

      renames[block->index][val.id()] = new_val;
      renames[block->index][new_val.id()] = new_val;
      orig_names[new_val.id()] = val;
      return new_val;
   };

   try_remove_trivial_phi = [&] (std::map<unsigned, phi_info>::iterator info) -> Temp {
      assert(info->second.block_idx != 0);
      Instruction* instr = info->second.phi;
      Temp same = Temp();
      Definition def = instr->getDefinition(0);
      /* a phi node is trivial iff all operands are the same or the definition of the phi */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         Temp op = instr->getOperand(i).getTemp();
         if (op == same || op == def.getTemp())
            continue;
         if (!(same == Temp()) || !(instr->getOperand(i).physReg() == def.physReg())) {
            /* phi is not trivial */
            return def.getTemp();
         }
         same = op;
      }
      assert(!(same == Temp() || same == def.getTemp()));

      /* reroute all uses to same and remove phi */
      std::vector<std::map<unsigned, phi_info>::iterator> phi_users;
      std::map<unsigned, phi_info>::iterator same_phi_info = phi_map.find(same.id());
      for (Instruction* instr : info->second.uses) {
         for (unsigned i = 0; i < instr->num_operands; i++) {
            if (instr->getOperand(i).isTemp() && instr->getOperand(i).tempId() == def.tempId()) {
               instr->getOperand(i).setTemp(same);
               if (same_phi_info != phi_map.end())
                  same_phi_info->second.uses.emplace(instr);
            }
         }
         /* recursively try to remove trivial phis */
         if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi) {
            std::map<unsigned, phi_info>::iterator it = phi_map.find(instr->getDefinition(0).tempId());
            if (it != phi_map.end() && it != info)
               phi_users.emplace_back(it);
         }
      }

      auto it = orig_names.find(same.id());
      unsigned orig_var = it != orig_names.end() ? it->second.id() : same.id();
      for (unsigned i = 0; i < program->blocks.size(); i++) {
         auto it = renames[i].find(orig_var);
         if (it != renames[i].end() && it->second == def.getTemp())
            renames[i][orig_var] = same;
      }

      unsigned block_idx = info->second.block_idx;
      instr->num_definitions = 0; /* this indicates that the phi can be removed */ // FIXME: this might cause memory leaks depending on how we implement the unique_ptr custom deleter
      phi_map.erase(info);
      for (auto it : phi_users)
         try_remove_trivial_phi(it);

      /* due to the removal of other phis, the name might have changed once again! */
      return renames[block_idx][orig_var];
   };

   std::map<unsigned, Instruction*> vectors;
   std::vector<std::vector<Temp>> phi_ressources;
   std::map<unsigned, unsigned> temp_to_phi_ressources;

   for (std::vector<std::unique_ptr<Block>>::reverse_iterator it = program->blocks.rbegin(); it != program->blocks.rend(); it++) {
      std::unique_ptr<Block>& block = *it;

      /* first, compute the death points of all live vars within the block */
      std::set<Temp>& live = live_out_per_block[block->index];

      std::vector<aco_ptr<Instruction>>::reverse_iterator rit;
      for (rit = block->instructions.rbegin(); rit != block->instructions.rend(); ++rit) {
         aco_ptr<Instruction>& instr = *rit;
         if (instr->opcode != aco_opcode::p_linear_phi && instr->opcode != aco_opcode::p_phi) {
            for (unsigned i = 0; i < instr->num_operands; i++) {
               if (instr->getOperand(i).isTemp())
                  live.emplace(instr->getOperand(i).getTemp());
            }
            if (instr->opcode == aco_opcode::p_create_vector) {
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == instr->getDefinition(0).getTemp().type())
                     vectors[instr->getOperand(i).tempId()] = instr.get();
               }
            }
         } else {
            /* collect information about affinity-related temporaries */
            std::vector<Temp> affinity_related;
            /* affinity_related[0] is the last seen affinity-related temp */
            affinity_related.emplace_back(instr->getDefinition(0).getTemp());
            affinity_related.emplace_back(instr->getDefinition(0).getTemp());
            for (unsigned i = 0; i < instr->num_operands; i++) {
               if (instr->getOperand(i).isTemp() && instr->getOperand(i).regClass() == instr->getDefinition(0).regClass()) {
                  affinity_related.emplace_back(instr->getOperand(i).getTemp());
                  temp_to_phi_ressources[instr->getOperand(i).tempId()] = phi_ressources.size();
               }
            }
            phi_ressources.emplace_back(std::move(affinity_related));
         }

         for (unsigned i = 0; i < instr->num_definitions; i++) {
            /* erase from live */
            if (instr->getDefinition(i).isTemp()) {
               live.erase(instr->getDefinition(i).getTemp());
               std::map<unsigned, unsigned>::iterator it = temp_to_phi_ressources.find(instr->getDefinition(i).tempId());
               if (it != temp_to_phi_ressources.end() && instr->getDefinition(i).regClass() == phi_ressources[it->second][0].regClass())
                  phi_ressources[it->second][0] = instr->getDefinition(i).getTemp();
            }
         }
      }
   }
   /* create affinities */
   for (std::vector<Temp>& vec : phi_ressources) {
      assert(vec.size() > 1);
      for (unsigned i = 1; i < vec.size(); i++)
         if (vec[i].id() != vec[0].id())
            affinities[vec[i].id()] = vec[0].id();
   }

   for (std::unique_ptr<Block>& block : program->blocks) {
      std::set<Temp>& live = live_out_per_block[block->index];
      /* initialize register file */
      assert(block->index != 0 || live.empty());
      std::array<uint32_t, 512> register_file = {0};

      for (Temp t : live) {
         Temp renamed = handle_live_in(t, block.get());
         if (ctx.assignments.find(renamed.id()) != ctx.assignments.end()) {
            for (unsigned i = 0; i < t.size(); i++)
               register_file[ctx.assignments[renamed.id()].first.reg + i] = renamed.id();
         }
      }

      std::vector<aco_ptr<Instruction>> instructions;
      std::vector<aco_ptr<Instruction>>::iterator it;

      /* this is a slight adjustment from the paper as we already have phi nodes:
       * We consider them incomplete phis and only handle the definition.
       * First, we look up the affinities.  */
      for (it = block->instructions.begin(); it != block->instructions.end(); ++it) {
         aco_ptr<Instruction>& phi = *it;
         if (phi->opcode != aco_opcode::p_phi && phi->opcode != aco_opcode::p_linear_phi)
            break;
         Definition& definition = phi->getDefinition(0);
         assert(!definition.isFixed());
         if (affinities.find(definition.tempId()) != affinities.end() &&
             ctx.assignments.find(affinities[definition.tempId()]) != ctx.assignments.end()) {
            assert(ctx.assignments[affinities[definition.tempId()]].second == phi->getDefinition(0).regClass());
            PhysReg reg = ctx.assignments[affinities[definition.tempId()]].first;
            bool reg_free = true;
            for (unsigned i = reg.reg; i < reg.reg + definition.size(); i++) {
               if (register_file[i] != 0) {
                  reg_free = false;
                  break;
               }
            }
            if (reg_free) {
               definition.setFixed(reg);
               for (unsigned i = 0; i < definition.size(); i++)
                  register_file[definition.physReg().reg + i] = definition.tempId();
            }
         }
      }

      it = block->instructions.begin();
      /* Second, we find registers for phis without affinity or where the register was blocked */
      for (;it != block->instructions.end(); ++it) {
         aco_ptr<Instruction>& phi = *it;
         if (phi->opcode != aco_opcode::p_phi && phi->opcode != aco_opcode::p_linear_phi)
            break;

         Definition& definition = phi->getDefinition(0);
         assert(definition.isTemp());
         renames[block->index][definition.tempId()] = definition.getTemp();

         if (!definition.isFixed()) {
            std::vector<std::pair<Operand, Definition>> parallelcopy;
            definition.setFixed(get_reg(ctx, register_file, definition.regClass(), parallelcopy, phi));
            assert(parallelcopy.empty());
            for (unsigned i = 0; i < definition.size(); i++)
               register_file[definition.physReg().reg + i] = definition.tempId();
         }
         ctx.assignments[definition.tempId()] = {definition.physReg(), definition.regClass()};
         live.emplace(definition.getTemp());

         /* update phi affinities */
         for (unsigned i = 0; i < phi->num_operands; i++) {
            if (phi->getOperand(i).isTemp() && phi->getOperand(i).regClass() == phi->getDefinition(0).regClass())
               affinities[phi->getOperand(i).tempId()] = definition.tempId();
         }

         instructions.emplace_back(std::move(*it));
      }

      /* Handle all other instructions of the block */
      for (; it != block->instructions.end(); ++it) {
         aco_ptr<Instruction>& instr = *it;
         std::vector<std::pair<Operand, Definition>> parallelcopy;

         assert(instr->opcode != aco_opcode::p_phi && instr->opcode != aco_opcode::p_linear_phi);

         /* handle operands */
         for (unsigned i = 0; i < instr->num_operands; ++i) {
            auto& operand = instr->getOperand(i);
            if (!operand.isTemp())
               continue;

            /* rename operands */
            operand.setTemp(read_variable(operand.getTemp(), block.get()));

            /* check if the operand is fixed */
            if (operand.isFixed()) {
               adjust_max_used_regs(ctx, operand.regClass(), operand.physReg().reg);

               if (operand.physReg() == ctx.assignments[operand.tempId()].first) {
                  /* we are fine: the operand is already assigned the correct reg */

               } else {
                  /* check if target reg is blocked, and move away the blocking var */
                  if (register_file[operand.physReg().reg]) {
                     uint32_t blocking_id = register_file[operand.physReg().reg];
                     RegClass rc = operand.physReg() == scc ? RegClass::s1 : ctx.assignments[blocking_id].second;
                     Operand pc_op = Operand(Temp{blocking_id, rc});
                     pc_op.setFixed(operand.physReg());
                     Definition pc_def = Definition(Temp{program->allocateId(), pc_op.regClass()});
                     /* find free reg */
                     PhysReg reg = get_reg(ctx, register_file, pc_op.regClass(), parallelcopy, instr);
                     pc_def.setFixed(reg);
                     ctx.assignments[pc_def.tempId()] = {reg, pc_def.regClass()};
                     for (unsigned i = 0; i < operand.size(); i++) {
                        register_file[pc_op.physReg().reg + i] = 0;
                        register_file[pc_def.physReg().reg + i] = pc_def.tempId();
                     }
                     parallelcopy.emplace_back(pc_op, pc_def);

                     /* handle renames of previous operands */
                     for (unsigned j = 0; j < i; j++) {
                        Operand& op = instr->getOperand(j);
                        if (op.isTemp() && op.tempId() == blocking_id) {
                           op = Operand(pc_def.getTemp());
                           op.setFixed(reg);
                        }
                     }
                  }
                  /* move operand to fixed reg and create parallelcopy pair */
                  Operand pc_op = operand;
                  Temp tmp = Temp{program->allocateId(), operand.physReg() == scc ? RegClass::b : operand.regClass()};
                  Definition pc_def = Definition(tmp);
                  pc_def.setFixed(operand.physReg());
                  pc_op.setFixed(ctx.assignments[operand.tempId()].first);
                  operand.setTemp(tmp);
                  ctx.assignments[tmp.id()] = {pc_def.physReg(), pc_def.regClass()};
                  operand.setFixed(pc_def.physReg());
                  for (unsigned i = 0; i < operand.size(); i++) {
                     register_file[pc_op.physReg().reg + i] = 0;
                     register_file[pc_def.physReg().reg + i] = tmp.id();
                  }
                  parallelcopy.emplace_back(pc_op, pc_def);
               }
            } else {
               assert(ctx.assignments.find(operand.tempId()) != ctx.assignments.end());
               operand.setFixed(ctx.assignments[operand.tempId()].first);
            }
            std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.getTemp().id());
            if (phi != phi_map.end())
               phi->second.uses.emplace(instr.get());

         }
         /* remove dead vars from register file */
         for (unsigned i = 0; i < instr->num_operands; i++)
            if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
               for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  register_file[instr->getOperand(i).physReg().reg + j] = 0;

         /* handle definitions which must have the same register as an operand */
         if (instr->opcode == aco_opcode::v_interp_p2_f32 ||
             instr->opcode == aco_opcode::v_mac_f32)
            instr->getDefinition(0).setFixed(instr->getOperand(2).physReg());
         else if (instr->opcode == aco_opcode::s_addk_i32 ||
                  instr->opcode == aco_opcode::s_mulk_i32)
            instr->getDefinition(0).setFixed(instr->getOperand(0).physReg());
         else if (instr->opcode == aco_opcode::p_wqm)
            instr->getDefinition(0).setFixed(instr->getOperand(0).physReg());
         else if ((instr->format == Format::MUBUF ||
                   instr->format == Format::MIMG) &&
                  instr->num_definitions == 1 &&
                  instr->num_operands == 4)
            instr->getDefinition(0).setFixed(instr->getOperand(3).physReg());

         /* handle definitions */
         for (unsigned i = 0; i < instr->num_definitions; ++i) {
            auto& definition = instr->getDefinition(i);
            if (definition.isFixed()) {
               adjust_max_used_regs(ctx, definition.regClass(), definition.physReg().reg);

               /* check if target dst is blocked */
               if (register_file[definition.physReg().reg] != 0) {
                  /* create parallelcopy pair to move blocking var */
                  Temp tmp = {register_file[definition.physReg().reg], ctx.assignments[register_file[definition.physReg().reg]].second};
                  Operand pc_op = Operand(tmp);
                  pc_op.setFixed(ctx.assignments[register_file[definition.physReg().reg]].first);
                  RegClass rc = definition.physReg() == scc ? RegClass::s1 : pc_op.regClass();
                  tmp = Temp{program->allocateId(), rc};
                  Definition pc_def = Definition(tmp);

                  /* re-enable the killed operands, so that we don't move the blocking var there */
                  for (unsigned i = 0; i < instr->num_operands; i++)
                     if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                        for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                           register_file[instr->getOperand(i).physReg().reg + j] = 0xFFFF;
                  /* find a new register for the blocking variable */
                  PhysReg reg = get_reg(ctx, register_file, rc, parallelcopy, instr);
                  /* once again, disable killed operands */
                  for (unsigned i = 0; i < instr->num_operands; i++) {
                     if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                        for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                           register_file[instr->getOperand(i).physReg().reg + j] = 0;
                  }
                  for (unsigned k = 0; k < i; k++) {
                     if (instr->getDefinition(k).isTemp() && !instr->getDefinition(k).isKill())
                        for (unsigned j = 0; j < instr->getDefinition(k).size(); j++)
                           register_file[instr->getDefinition(k).physReg().reg + j] = instr->getDefinition(k).tempId();
                  }
                  pc_def.setFixed(reg);

                  /* finish assignment of parallelcopy */
                  ctx.assignments[pc_def.tempId()] = {reg, pc_def.regClass()};
                  parallelcopy.emplace_back(pc_op, pc_def);

                  /* add changes to reg_file */
                  for (unsigned i = 0; i < pc_op.size(); i++) {
                     register_file[pc_op.physReg().reg + i] = 0x0;
                     register_file[pc_def.physReg().reg + i] = pc_def.tempId();
                  }
               }
            } else if (definition.isTemp()) {
               /* find free reg */
               if (definition.hasHint() && register_file[definition.physReg().reg] == 0)
                  definition.setFixed(definition.physReg());
               else if (instr->opcode == aco_opcode::p_split_vector) {
                  PhysReg reg = PhysReg{instr->getOperand(0).physReg().reg + i};
                  if (!get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg, 0))
                     reg = get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr);
                  definition.setFixed(reg);
               } else if (instr->opcode == aco_opcode::p_extract_vector) {
                  PhysReg reg;
                  if (instr->getOperand(0).isKill()) {
                     reg = instr->getOperand(0).physReg();
                     reg.reg += definition.size() * instr->getOperand(1).constantValue();
                     assert(register_file[reg.reg] == 0);
                  } else {
                     reg = get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr);
                  }
                  definition.setFixed(reg);
               } else if (instr->opcode == aco_opcode::p_create_vector) {
                  unsigned max_moves = 0;
                  for (unsigned i = 0; i < instr->num_operands; i++)
                     if (instr->getOperand(i).isTemp() && instr->getOperand(i).isKill())
                        max_moves += instr->getOperand(i).size();
                  for (unsigned num_moves = 0; num_moves <= max_moves; num_moves++) {
                     unsigned k = 0;
                     for (unsigned i = 0; i < instr->num_operands; i++) {
                        if (!(instr->getOperand(i).isTemp() && instr->getOperand(i).isKill())) {
                           k += instr->getOperand(i).size();
                           continue;
                        }
                        PhysReg reg = instr->getOperand(i).physReg();
                        reg.reg -= k;
                        if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg, num_moves)) {
                           definition.setFixed(reg);
                           break;
                        } else {
                           k += instr->getOperand(i).size();
                        }
                     }
                     if (definition.isFixed())
                        break;
                  }
                  if (!definition.isFixed())
                     definition.setFixed(get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr));

               } else if (affinities.find(definition.tempId()) != affinities.end() &&
                          ctx.assignments.find(affinities[definition.tempId()]) != ctx.assignments.end()) {
                  PhysReg reg = ctx.assignments[affinities[definition.tempId()]].first;
                  if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg, 0))
                     definition.setFixed(reg);
                  else
                     definition.setFixed(get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr));

               } else if (vectors.find(definition.tempId()) != vectors.end()) {
                  Instruction* vec = vectors[definition.tempId()];
                  unsigned offset = 0;
                  for (unsigned i = 0; i < vec->num_operands; i++) {
                     if (vec->getOperand(i).isTemp() && vec->getOperand(i).tempId() == definition.tempId())
                        break;
                     else
                        offset += vec->getOperand(i).size();
                  }
                  unsigned k = 0;
                  for (unsigned i = 0; i < vec->num_operands; i++) {
                     if (vec->getOperand(i).isTemp() &&
                         vec->getOperand(i).tempId() != definition.tempId() &&
                         vec->getOperand(i).getTemp().type() == definition.getTemp().type() &&
                         ctx.assignments.find(vec->getOperand(i).tempId()) != ctx.assignments.end()) {
                        PhysReg reg = ctx.assignments[vec->getOperand(i).tempId()].first;
                        reg.reg = reg.reg - k + offset;
                        if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg, 0)) {
                           definition.setFixed(reg);
                           break;
                        }
                     }
                     k += vec->getOperand(i).size();
                  }
                  if (!definition.isFixed()) {
                     std::pair<PhysReg, bool> res = get_reg_vec(ctx, register_file, parallelcopy, vec->getDefinition(0).regClass());
                     PhysReg reg = res.first;
                     if (res.second) {
                        reg.reg += offset;
                     } else {
                        reg = get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr);
                     }
                     definition.setFixed(reg);
                  }
               } else
                  definition.setFixed(get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr));

               assert(definition.isFixed() && ((definition.getTemp().type() == vgpr && definition.physReg().reg >= 256) ||
                                               (definition.getTemp().type() != vgpr && definition.physReg().reg < 256)));
            } else {
               continue;
            }

            ctx.assignments[definition.tempId()] = {definition.physReg(), definition.regClass()};
            for (unsigned i = 0; i < definition.size(); i++)
               register_file[definition.physReg().reg + i] = definition.tempId();
            /* set live if it has a kill point */
            if (!definition.isKill()) {
               live.emplace(definition.getTemp());
            }
            /* add to renames table */
            renames[block->index][definition.tempId()] = definition.getTemp();
         }

         /* kill definitions */
         for (unsigned i = 0; i < instr->num_definitions; i++)
             if (instr->getDefinition(i).isTemp() && instr->getDefinition(i).isKill())
                for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                   register_file[instr->getDefinition(i).physReg().reg + j] = 0;

         /* emit parallelcopy */
         if (!parallelcopy.empty()) {
            aco_ptr<Instruction> pc;
            pc.reset(create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, parallelcopy.size(), parallelcopy.size()));
            for (unsigned i = 0; i < parallelcopy.size(); i++) {
               pc->getOperand(i) = parallelcopy[i].first;
               pc->getDefinition(i) = parallelcopy[i].second;

               /* it might happen that the operand is already renamed. we have to restore the original name. */
               std::map<unsigned, Temp>::iterator it = orig_names.find(pc->getOperand(i).tempId());
               if (it != orig_names.end())
                  pc->getOperand(i).setTemp(it->second);
               unsigned orig_id = pc->getOperand(i).tempId();
               orig_names[pc->getDefinition(i).tempId()] = pc->getOperand(i).getTemp();

               pc->getOperand(i).setTemp(read_variable(pc->getOperand(i).getTemp(), block.get()));
               renames[block->index][orig_id] = pc->getDefinition(i).getTemp();
               renames[block->index][pc->getDefinition(i).tempId()] = pc->getDefinition(i).getTemp();
               std::map<unsigned, phi_info>::iterator phi = phi_map.find(pc->getOperand(i).tempId());
               if (phi != phi_map.end())
                  phi->second.uses.emplace(instr.get());
            }
            instructions.emplace_back(std::move(pc));
         }

         /* some instructions need VOP3 encoding if operand/definition is not assigned to VCC */
         bool instr_needs_vop3 = !instr->isVOP3() &&
                                 ((instr->format == Format::VOPC && !(instr->getDefinition(0).physReg() == vcc)) ||
                                  (instr->opcode == aco_opcode::v_cndmask_b32 && !(instr->getOperand(2).physReg() == vcc)) ||
                                  ((instr->opcode == aco_opcode::v_add_co_u32 ||
                                    instr->opcode == aco_opcode::v_addc_co_u32 ||
                                    instr->opcode == aco_opcode::v_sub_co_u32 ||
                                    instr->opcode == aco_opcode::v_subb_co_u32 ||
                                    instr->opcode == aco_opcode::v_subrev_co_u32 ||
                                    instr->opcode == aco_opcode::v_subbrev_co_u32) &&
                                   !(instr->getDefinition(1).physReg() == vcc)) ||
                                  ((instr->opcode == aco_opcode::v_addc_co_u32 ||
                                    instr->opcode == aco_opcode::v_subb_co_u32 ||
                                    instr->opcode == aco_opcode::v_subbrev_co_u32) &&
                                   !(instr->getOperand(2).physReg() == vcc)));
         if (instr_needs_vop3) {

            /* if the first operand is a literal, we have to move it to a reg */
            if (instr->num_operands && instr->getOperand(0).isLiteral()) {
               bool can_sgpr = true;
               /* check, if we have to move to vgpr */
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).getTemp().type() == sgpr) {
                     can_sgpr = false;
                     break;
                  }
               }
               aco_ptr<Instruction> mov;
               if (can_sgpr)
                  mov.reset(create_instruction<SOP1_instruction>(aco_opcode::s_mov_b32, Format::SOP1, 1, 1));
               else
                  mov.reset(create_instruction<VOP1_instruction>(aco_opcode::v_mov_b32, Format::VOP1, 1, 1));
               mov->getOperand(0) = instr->getOperand(0);
               Temp tmp = {program->allocateId(), can_sgpr ? s1 : v1};
               mov->getDefinition(0) = Definition(tmp);
               /* disable definitions and re-enable operands */
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                     register_file[instr->getDefinition(i).physReg().reg + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill()) {
                     for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                        register_file[instr->getOperand(i).physReg().reg + j] = 0xFFFF;
                  }
               }
               mov->getDefinition(0).setFixed(get_reg(ctx, register_file, tmp.regClass(), parallelcopy, mov));
               instr->getOperand(0) = Operand(tmp);
               instr->getOperand(0).setFixed(mov->getDefinition(0).physReg());
               instructions.emplace_back(std::move(mov));
               /* re-enable live vars */
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                     for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                        register_file[instr->getOperand(i).physReg().reg + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  if (instr->getDefinition(i).isTemp() && !instr->getDefinition(i).isKill())
                     for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                        register_file[instr->getDefinition(i).physReg().reg + j] = instr->getDefinition(i).tempId();
               }
            }

            /* change the instruction to VOP3 to enable an arbitrary register pair as dst */
            aco_ptr<Instruction> tmp = std::move(instr);
            Format format = (Format) ((int) tmp->format | (int) Format::VOP3A);
            instr.reset(create_instruction<VOP3A_instruction>(tmp->opcode, format, tmp->num_operands, tmp->num_definitions));
            for (unsigned i = 0; i < instr->num_operands; i++) {
               Operand& operand = tmp->getOperand(i);
               instr->getOperand(i) = operand;
               /* keep phi_map up to date */
               if (operand.isTemp()) {
                  std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.tempId());
                  if (phi != phi_map.end()) {
                     phi->second.uses.erase(tmp.get());
                     phi->second.uses.emplace(instr.get());
                  }
               }
            }
            for (unsigned i = 0; i < instr->num_definitions; i++)
               instr->getDefinition(i) = tmp->getDefinition(i);
         }

         instructions.emplace_back(std::move(*it));
      } /* end for Instr */

      block->instructions = std::move(instructions);

      filled[block->index] = true;
      for (Block* succ : block->linear_successors) {
         /* seal block if all predecessors are filled */
         bool all_filled = true;
         for (Block* pred : succ->linear_predecessors) {
            if (!filled[pred->index]) {
               all_filled = false;
               break;
            }
         }
         if (all_filled) {
            /* finish incomplete phis and check if they became trivial */
            for (Instruction* phi : incomplete_phis[succ->index]) {
               std::vector<Block*> preds = phi->getDefinition(0).getTemp().is_linear() ? succ->linear_predecessors : succ->logical_predecessors;
               for (unsigned i = 0; i < phi->num_operands; i++) {
                  phi->getOperand(i).setTemp(read_variable(phi->getOperand(i).getTemp(), preds[i]));
                  phi->getOperand(i).setFixed(ctx.assignments[phi->getOperand(i).tempId()].first);
               }
               try_remove_trivial_phi(phi_map.find(phi->getDefinition(0).tempId()));
            }
            /* complete the original phi nodes, but no need to check triviality */
            for (aco_ptr<Instruction>& instr : succ->instructions) {
               if (instr->opcode != aco_opcode::p_phi && instr->opcode != aco_opcode::p_linear_phi)
                  break;
               std::vector<Block*> preds = instr->opcode == aco_opcode::p_phi ? succ->logical_predecessors : succ->linear_predecessors;

               for (unsigned i = 0; i < instr->num_operands; i++) {
                  auto& operand = instr->getOperand(i);
                  if (!operand.isTemp())
                     continue;
                  operand.setTemp(read_variable(operand.getTemp(), preds[i]));
                  operand.setFixed(ctx.assignments[operand.tempId()].first);
                  std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.getTemp().id());
                  if (phi != phi_map.end())
                     phi->second.uses.emplace(instr.get());
               }
            }
            sealed[succ->index] = true;
         }
      }
   } /* end for BB */

   /* remove trivial phis */
   for (std::unique_ptr<Block>& block : program->blocks) {
      std::vector<aco_ptr<Instruction>>::iterator it = block->instructions.begin();
      for (; it != block->instructions.end();) {
         if ((*it)->opcode != aco_opcode::p_phi && (*it)->opcode != aco_opcode::p_linear_phi)
            break;
         if (!(*it)->num_definitions)
            it = block->instructions.erase(it);
         else
            ++it;
      }
   }

   program->config->num_vgprs = ctx.max_used_vgpr + 1;
   program->config->num_sgprs = ctx.max_used_sgpr + 1 + 2;
}

}
