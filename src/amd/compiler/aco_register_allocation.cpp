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
   std::bitset<512> war_hint;
   Program* program;
   std::unordered_map<unsigned, std::pair<PhysReg, RegClass>> assignments;
   std::map<unsigned, Temp> orig_names;
   unsigned max_used_sgpr = 0;
   unsigned max_used_vgpr = 0;

   ra_ctx(Program* program) : program(program) {}
};


/* helper function for debugging */
#if 0
void print_regs(ra_ctx& ctx, bool vgprs, std::array<uint32_t, 512>& reg_file)
{
   unsigned max = vgprs ? ctx.program->max_vgpr : ctx.program->max_sgpr;
   unsigned lb = vgprs ? 256 : 0;
   unsigned ub = lb + max;
   char reg_char = vgprs ? 'v' : 's';

   /* print markers */
   printf("       ");
   for (unsigned i = lb; i < ub; i += 3) {
      printf("%.2u ", i - lb);
   }
   printf("\n");

   /* print usage */
   printf("%cgprs: ", reg_char);
   unsigned free_regs = 0;
   unsigned prev = 0;
   bool char_select = false;
   for (unsigned i = lb; i < ub; i++) {
      if (reg_file[i] == 0xFFFF) {
         printf("~");
      } else if (reg_file[i]) {
         if (reg_file[i] != prev) {
            prev = reg_file[i];
            char_select = !char_select;
         }
         printf(char_select ? "#" : "@");
      } else {
         free_regs++;
         printf(".");
      }
   }
   printf("\n");

   printf("%u/%u used, %u/%u free\n", max - free_regs, max, free_regs, max);

   /* print assignments */
   prev = 0;
   unsigned size = 0;
   for (unsigned i = lb; i < ub; i++) {
      if (reg_file[i] != prev) {
         if (prev && size > 1)
            printf("-%d]\n", i - 1 - lb);
         else if (prev)
            printf("]\n");
         prev = reg_file[i];
         if (prev && prev != 0xFFFF) {
            if (ctx.orig_names.count(reg_file[i]) && ctx.orig_names[reg_file[i]].id() != reg_file[i])
               printf("%%%u (was %%%d) = %c[%d", reg_file[i], ctx.orig_names[reg_file[i]].id(), reg_char, i - lb);
            else
               printf("%%%u = %c[%d", reg_file[i], reg_char, i - lb);
         }
         size = 1;
      } else {
         size++;
      }
   }
   if (prev && size > 1)
      printf("-%d]\n", ub - lb - 1);
   else if (prev)
      printf("]\n");
}
#endif


void adjust_max_used_regs(ra_ctx& ctx, RegClass rc, unsigned reg)
{
   unsigned max_addressible_sgpr = ctx.program->sgpr_limit;
   unsigned size = rc.size();
   if (rc.type() == vgpr) {
      assert(reg >= 256);
      unsigned hi = reg - 256 + size - 1;
      ctx.max_used_vgpr = std::max(ctx.max_used_vgpr, hi);
   } else if (reg + rc.size() <= max_addressible_sgpr) {
      unsigned hi = reg + size - 1;
      ctx.max_used_sgpr = std::max(ctx.max_used_sgpr, std::min(hi, max_addressible_sgpr));
   }
}


void update_renames(ra_ctx& ctx, std::array<uint32_t, 512>& reg_file,
                    std::vector<std::pair<Operand, Definition>>& parallelcopies,
                    aco_ptr<Instruction>& instr)
{
   /* allocate id's and rename operands: this is done transparently here */
   for (std::pair<Operand, Definition>& copy : parallelcopies) {
      /* the definitions with id are not from this function and already handled */
      if (copy.second.isTemp())
         continue;

      // FIXME: if a definition got moved, change the target location and remove the parallelcopy
      copy.second.setTemp(Temp(ctx.program->allocateId(), copy.second.regClass()));
      ctx.assignments[copy.second.tempId()] = {copy.second.physReg(), copy.second.regClass()};
      for (unsigned i = copy.second.physReg().reg; i < copy.second.physReg() + copy.second.size(); i++)
         reg_file[i] = copy.second.tempId();
      /* check if we moved an operand */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         if (!instr->getOperand(i).isTemp())
            continue;
         if (instr->getOperand(i).tempId() == copy.first.tempId()) {
            bool omit_renaming = instr->opcode == aco_opcode::p_create_vector && !instr->getOperand(i).isKill();
            for (std::pair<Operand, Definition>& pc : parallelcopies) {
               PhysReg def_reg = pc.second.physReg();
               omit_renaming &= def_reg > copy.first.physReg() ?
                                (copy.first.physReg() + copy.first.size() <= def_reg.reg) :
                                (def_reg + pc.second.size() <= copy.first.physReg().reg);
            }
            if (omit_renaming)
               continue;
            instr->getOperand(i).setTemp(copy.second.getTemp());
            instr->getOperand(i).setFixed(copy.second.physReg());
         }
      }
   }
}

std::pair<PhysReg, bool> get_reg_simple(ra_ctx& ctx,
                                        std::array<uint32_t, 512>& reg_file,
                                        uint32_t lb, uint32_t ub,
                                        uint32_t size, uint32_t stride,
                                        RegClass rc)
{
   /* best fit algorithm: find the smallest gap to fit in the variable */
   if (stride == 1) {
      unsigned best_pos = 0xFFFF;
      unsigned gap_size = 0xFFFF;
      unsigned next_pos = 0xFFFF;

      for (unsigned current_reg = lb; current_reg < ub; current_reg++) {
         if (reg_file[current_reg] != 0 || ctx.war_hint[current_reg]) {
            if (next_pos == 0xFFFF)
               continue;

            /* check if the variable fits */
            if (next_pos + size > current_reg) {
               next_pos = 0xFFFF;
               continue;
            }

            /* check if the tested gap is smaller */
            if (current_reg - next_pos < gap_size) {
               best_pos = next_pos;
               gap_size = current_reg - next_pos;
            }
            next_pos = 0xFFFF;
            continue;
         }

         if (next_pos == 0xFFFF)
            next_pos = current_reg;
      }

      /* final check */
      if (next_pos != 0xFFFF &&
          next_pos + size <= ub &&
          ub - next_pos < gap_size) {
         best_pos = next_pos;
         gap_size = ub - next_pos;
      }
      if (best_pos != 0xFFFF) {
         adjust_max_used_regs(ctx, rc, best_pos);
         return {PhysReg{best_pos}, true};
      }
      return {{}, false};
   }

   bool found = false;
   unsigned reg_lo = lb;
   unsigned reg_hi = lb + size - 1;
   while (!found && reg_lo + size <= ub) {
      if (reg_file[reg_lo] != 0) {
         reg_lo += stride;
         continue;
      }
      reg_hi = reg_lo + size - 1;
      found = true;
      for (unsigned reg = reg_lo + 1; found && reg <= reg_hi; reg++) {
         if (reg_file[reg] != 0)
            found = false;
      }
      if (found) {
         adjust_max_used_regs(ctx, rc, reg_lo);
         return {PhysReg{reg_lo}, true};
      }

      reg_lo += stride;
   }

   return {{}, false};
}

bool get_regs_for_copies(ra_ctx& ctx,
                         std::array<uint32_t, 512>& reg_file,
                         std::vector<std::pair<Operand, Definition>>& parallelcopies,
                         std::set<std::pair<unsigned, unsigned>> vars,
                         uint32_t lb, uint32_t ub,
                         aco_ptr<Instruction>& instr,
                         uint32_t def_reg_lo,
                         uint32_t def_reg_hi)
{

   /* variables are sorted from small sized to large */
   for (std::set<std::pair<unsigned, unsigned>>::reverse_iterator it = vars.rbegin(); it != vars.rend(); ++it) {
      unsigned id = it->second;
      std::pair<PhysReg, RegClass> var = ctx.assignments[id];
      uint32_t size = it->first;
      uint32_t stride = 1;
      if (var.second.type() == sgpr) {
         if (size == 2)
            stride = 2;
         if (size > 3)
            stride = 4;
      }

      /* check if this is a dead operand, then we can re-use the space from the definition */
      bool is_dead_operand = false;
      for (unsigned i = 0; !is_phi(instr) && !is_dead_operand && i < instr->num_operands; i++) {
         if (instr->getOperand(i).isTemp() && instr->getOperand(i).isKill() && instr->getOperand(i).tempId() == id)
            is_dead_operand = true;
      }

      std::pair<PhysReg, bool> res;
      if (is_dead_operand) {
         if (instr->opcode == aco_opcode::p_create_vector) {
            for (unsigned i = 0, offset = 0; i < instr->num_operands; offset += instr->getOperand(i).size(), i++) {
               if (instr->getOperand(i).isTemp() && instr->getOperand(i).tempId() == id) {
                  for (unsigned j = 0; j < size; j++)
                     assert(reg_file[def_reg_lo + offset + j] == 0);
                  res = {PhysReg{def_reg_lo + offset}, true};
                  break;
               }
            }
         } else {
            res = get_reg_simple(ctx, reg_file, def_reg_lo, def_reg_hi + 1, size, stride, var.second);
         }
      } else {
         res = get_reg_simple(ctx, reg_file, lb, def_reg_lo, size, stride, var.second);
         if (!res.second) {
            unsigned lb = (def_reg_hi + stride) & ~(stride - 1);
            res = get_reg_simple(ctx, reg_file, lb, ub, size, stride, var.second);
         }
      }

      if (res.second) {
         /* mark the area as blocked */
         for (unsigned i = res.first.reg; i < res.first + size; i++)
            reg_file[i] = 0xFFFFFFFF;
         /* create parallelcopy pair (without definition id) */
         Temp tmp = Temp(id, var.second);
         Operand pc_op = Operand(tmp);
         pc_op.setFixed(var.first);
         Definition pc_def = Definition(res.first, pc_op.regClass());
         parallelcopies.emplace_back(pc_op, pc_def);
         continue;
      }

      unsigned best_pos = lb;
      unsigned num_moves = 0xFF;
      unsigned num_vars = 0;

      /* we use a sliding window to find potential positions */
      unsigned reg_lo = lb;
      unsigned reg_hi = lb + size - 1;
      for (reg_lo = lb, reg_hi = lb + size - 1; reg_hi < ub; reg_lo += stride, reg_hi += stride) {
         if (!is_dead_operand && ((reg_lo >= def_reg_lo && reg_lo <= def_reg_hi) ||
                                  (reg_hi >= def_reg_lo && reg_hi <= def_reg_hi)))
            continue;

         /* second, check that we have at most k=num_moves elements in the window
          * and no element is larger than the currently processed one */
         unsigned k = 0;
         unsigned n = 0;
         unsigned last_var = 0;
         bool found = true;
         for (unsigned j = reg_lo; found && j <= reg_hi; j++) {
            if (reg_file[j] == 0 || reg_file[j] == last_var)
               continue;

            /* 0xFFFF signals that this area is already blocked! */
            if (reg_file[j] == 0xFFFFFFFF || k > num_moves) {
               found = false;
               break;
            }
            /* we cannot split live ranges of linear vgprs */
            if (ctx.assignments[reg_file[j]].second & (1 << 6)) {
               found = false;
               break;
            }
            bool is_kill = false;
            for (unsigned i = 0; !is_kill && i < instr->num_operands; i++) {
               if (instr->getOperand(i).isTemp() && instr->getOperand(i).isKill() && instr->getOperand(i).tempId() == reg_file[j])
                  is_kill = true;
            }
            if (!is_kill && ctx.assignments[reg_file[j]].second.size() >= size) {
               found = false;
               break;
            }

            k += ctx.assignments[reg_file[j]].second.size();
            last_var = reg_file[j];
            n++;
            if (k > num_moves || (k == num_moves && n <= num_vars)) {
               found = false;
               break;
            }
         }

         if (found) {
            best_pos = reg_lo;
            num_moves = k;
            num_vars = n;
         }
      }

      assert(num_moves != 0xFF);
      reg_lo = best_pos;
      reg_hi = best_pos + size - 1;

      /* collect variables and block reg file */
      std::set<std::pair<unsigned, unsigned>> new_vars;
      for (unsigned j = reg_lo; j <= reg_hi; j++) {
         if (reg_file[j] != 0) {
            unsigned size = ctx.assignments[reg_file[j]].second.size();
            unsigned id = reg_file[j];
            new_vars.emplace(size, id);
            for (unsigned k = 0; k < size; k++)
               reg_file[ctx.assignments[id].first + k] = 0;
         }
      }

      /* mark the area as blocked */
      for (unsigned i = reg_lo; i <= reg_hi; i++)
         reg_file[i] = 0xFFFFFFFF;

      get_regs_for_copies(ctx, reg_file, parallelcopies, new_vars, lb, ub, instr, def_reg_lo, def_reg_hi);
      adjust_max_used_regs(ctx, var.second, reg_lo);

      /* create parallelcopy pair (without definition id) */
      Temp tmp = Temp(id, var.second);
      Operand pc_op = Operand(tmp);
      pc_op.setFixed(var.first);
      Definition pc_def = Definition(PhysReg{reg_lo}, pc_op.regClass());
      parallelcopies.emplace_back(pc_op, pc_def);
   }

   return true;
}


std::pair<PhysReg, bool> get_reg_impl(ra_ctx& ctx,
                                      std::array<uint32_t, 512>& reg_file,
                                      std::vector<std::pair<Operand, Definition>>& parallelcopies,
                                      uint32_t lb, uint32_t ub,
                                      uint32_t size, uint32_t stride,
                                      RegClass rc,
                                      aco_ptr<Instruction>& instr)
{
   unsigned regs_free = 0;
   /* check how many free regs we have */
   for (unsigned j = lb; j < ub; j++) {
      if (reg_file[j] == 0)
         regs_free++;
   }

   /* mark and count killed operands */
   unsigned killed_ops = 0;
   for (unsigned j = 0; !is_phi(instr) && j < instr->num_operands; j++) {
      if (instr->getOperand(j).isTemp() &&
          instr->getOperand(j).isFirstKill() &&
          instr->getOperand(j).getTemp().type() == rc.type()) {
         assert(instr->getOperand(j).isFixed());
         assert(reg_file[instr->getOperand(j).physReg().reg] == 0);
         for (unsigned k = 0; k < instr->getOperand(j).size(); k++)
            reg_file[instr->getOperand(j).physReg() + k] = 0xFFFFFFFF;
         killed_ops += instr->getOperand(j).getTemp().size();
      }
   }

   assert(regs_free >= size);
   /* we might have to move dead operands to dst in order to make space */
   unsigned op_moves = 0;

   if (size > (regs_free - killed_ops))
      op_moves = size - (regs_free - killed_ops);

   /* find the best position to place the definition */
   unsigned best_pos = lb;
   unsigned num_moves = 0xFF;
   unsigned num_vars = 0;

   /* we use a sliding window to check potential positions */
   unsigned reg_lo = lb;
   unsigned reg_hi = lb + size - 1;
   for (reg_lo = lb, reg_hi = lb + size - 1; reg_hi < ub; reg_lo += stride, reg_hi += stride) {
      /* first check the edges: this is what we have to fix to allow for num_moves > size */
      if (reg_lo > lb && reg_file[reg_lo] != 0 && reg_file[reg_lo] == reg_file[reg_lo - 1])
         continue;
      if (reg_hi < ub - 1 && reg_file[reg_hi] != 0 && reg_file[reg_hi] == reg_file[reg_hi + 1])
         continue;

      /* second, check that we have at most k=num_moves elements in the window
       * and no element is larger than the currently processed one */
      unsigned k = op_moves;
      unsigned n = 0;
      unsigned remaining_op_moves = op_moves;
      unsigned last_var = 0;
      bool found = true;
      bool aligned = rc == RegClass::v4 && reg_lo % 4 == 0;
      for (unsigned j = reg_lo; found && j <= reg_hi; j++) {
         if (reg_file[j] == 0 || reg_file[j] == last_var)
            continue;

         /* dead operands effectively reduce the number of estimated moves */
         if (remaining_op_moves && reg_file[j] == 0xFFFFFFFF) {
            k--;
            remaining_op_moves--;
            continue;
         }

         if (ctx.assignments[reg_file[j]].second.size() >= size) {
            found = false;
            break;
         }


         /* we cannot split live ranges of linear vgprs */
         if (ctx.assignments[reg_file[j]].second & (1 << 6)) {
            found = false;
            break;
         }

         k += ctx.assignments[reg_file[j]].second.size();
         n++;
         last_var = reg_file[j];
      }

      if (!found || k > num_moves)
         continue;
      if (k == num_moves && n < num_vars)
         continue;
      if (!aligned && k == num_moves && n == num_vars)
         continue;

      if (found) {
         best_pos = reg_lo;
         num_moves = k;
         num_vars = n;
      }
   }

   if (num_moves == 0xFF) {
      /* remove killed operands from reg_file once again */
      for (unsigned i = 0; !is_phi(instr) && i < instr->num_operands; i++) {
         if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill()) {
            for (unsigned k = 0; k < instr->getOperand(i).getTemp().size(); k++)
               reg_file[instr->getOperand(i).physReg() + k] = 0;
         }
      }
      return {{}, false};
   }

   /* now, we figured the placement for our definition */
   std::set<std::pair<unsigned, unsigned>> vars;
   for (unsigned j = best_pos; j < best_pos + size; j++) {
      if (reg_file[j] != 0xFFFFFFFF && reg_file[j] != 0)
         vars.emplace(ctx.assignments[reg_file[j]].second.size(), reg_file[j]);
      reg_file[j] = 0;
   }

   if (instr->opcode == aco_opcode::p_create_vector) {
      /* move killed operands which aren't yet at the correct position */
      for (unsigned i = 0, offset = 0; i < instr->num_operands; offset += instr->getOperand(i).size(), i++) {
         if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill() &&
             instr->getOperand(i).getTemp().type() == rc.type()) {

            if (instr->getOperand(i).physReg() != best_pos + offset) {
               vars.emplace(instr->getOperand(i).size(), instr->getOperand(i).tempId());
               for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  reg_file[instr->getOperand(i).physReg() + j] = 0;
            } else {
               for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  reg_file[instr->getOperand(i).physReg() + j] = instr->getOperand(i).tempId();
            }
         }
      }
   } else {
      /* re-enable the killed operands */
      for (unsigned j = 0; !is_phi(instr) && j < instr->num_operands; j++) {
         if (instr->getOperand(j).isTemp() && instr->getOperand(j).isFirstKill()) {
            for (unsigned k = 0; k < instr->getOperand(j).getTemp().size(); k++)
               reg_file[instr->getOperand(j).physReg() + k] = instr->getOperand(j).tempId();
         }
      }
   }

   get_regs_for_copies(ctx, reg_file, parallelcopies, vars, lb, ub, instr, best_pos, best_pos + size - 1);

   /* we set the definition regs == 0. the actual caller is responsible for correct setting */
   for (unsigned i = 0; i < size; i++)
      reg_file[best_pos + i] = 0;

   update_renames(ctx, reg_file, parallelcopies, instr);

   /* remove killed operands from reg_file once again */
   for (unsigned i = 0; !is_phi(instr) && i < instr->num_operands; i++) {
      if (!instr->getOperand(i).isTemp() || !instr->getOperand(i).isFixed())
         continue;
      assert(!instr->getOperand(i).isUndefined());
      if (instr->getOperand(i).isFirstKill()) {
         for (unsigned j = 0; j < instr->getOperand(i).getTemp().size(); j++)
            reg_file[instr->getOperand(i).physReg() + j] = 0;
      }
   }

   adjust_max_used_regs(ctx, rc, best_pos);
   return {PhysReg{best_pos}, true};
}

PhysReg get_reg(ra_ctx& ctx,
                std::array<uint32_t, 512>& reg_file,
                RegClass rc,
                std::vector<std::pair<Operand, Definition>>& parallelcopies,
                aco_ptr<Instruction>& instr)
{
   uint32_t size = rc.size();
   uint32_t stride = 1;
   uint32_t lb, ub;
   if (rc.type() == vgpr) {
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

   std::pair<PhysReg, bool> res = {{}, false};
   /* try to find space without live-range splits */
   if (rc.type() == vgpr && (size == 4 || size == 8))
      res = get_reg_simple(ctx, reg_file, lb, ub, size, 4, rc);
   if (!res.second)
      res = get_reg_simple(ctx, reg_file, lb, ub, size, stride, rc);
   if (res.second)
      return res.first;

   /* try to find space with live-range splits */
   res = get_reg_impl(ctx, reg_file, parallelcopies, lb, ub, size, stride, rc, instr);

   if (res.second)
      return res.first;

   unsigned regs_free = 0;
   for (unsigned i = lb; i < ub; i++) {
      if (!reg_file[i])
         regs_free++;
   }

   /* We should only fail here because keeping under the limit would require
    * too many moves. */
   assert(regs_free >= size);

   /* try using more registers */
   uint16_t max_addressible_sgpr = ctx.program->sgpr_limit;
   if (rc.type() == vgpr && ctx.program->max_vgpr < 256) {
      update_vgpr_sgpr_demand(ctx.program, ctx.program->max_vgpr + 1, ctx.program->max_sgpr);
      return get_reg(ctx, reg_file, rc, parallelcopies, instr);
   } else if (rc.type() == sgpr && ctx.program->max_sgpr < max_addressible_sgpr) {
      update_vgpr_sgpr_demand(ctx.program, ctx.program->max_vgpr, ctx.program->max_sgpr + 1);
      return get_reg(ctx, reg_file, rc, parallelcopies, instr);
   }

   //FIXME: if nothing helps, shift-rotate the registers to make space

   unreachable("did not find a register");
}


std::pair<PhysReg, bool> get_reg_vec(ra_ctx& ctx,
                                     std::array<uint32_t, 512>& reg_file,
                                     RegClass rc)
{
   uint32_t size = rc.size();
   uint32_t stride = 1;
   uint32_t lb, ub;
   if (rc.type() == vgpr) {
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
   return get_reg_simple(ctx, reg_file, lb, ub, size, stride, rc);
}


PhysReg get_reg_create_vector(ra_ctx& ctx,
                              std::array<uint32_t, 512>& reg_file,
                              RegClass rc,
                              std::vector<std::pair<Operand, Definition>>& parallelcopies,
                              aco_ptr<Instruction>& instr)
{
   /* create_vector instructions have different costs w.r.t. register coalescing */
   uint32_t size = rc.size();
   uint32_t stride = 1;
   uint32_t lb, ub;
   if (rc.type() == vgpr) {
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

   unsigned best_pos = -1;
   unsigned num_moves = 0xFF;

   /* test for each operand which definition placement causes the least shuffle instructions */
   for (unsigned i = 0, offset = 0; i < instr->num_operands; offset += instr->getOperand(i).size(), i++) {
      // TODO: think about, if we can alias live operands on the same register
      if (!instr->getOperand(i).isTemp() || !instr->getOperand(i).isKill() || instr->getOperand(i).getTemp().type() != rc.type())
         continue;

      unsigned reg_lo = instr->getOperand(i).physReg() - offset;
      unsigned reg_hi = reg_lo + size - 1;
      unsigned k = 0;

      /* no need to check multiple times */
      if (reg_lo == best_pos)
         continue;

      /* check borders */
      // TODO: this can be improved */
      if (reg_lo < lb || reg_hi >= ub || reg_lo % stride != 0)
         continue;
      if (reg_lo > lb && reg_file[reg_lo] != 0 && reg_file[reg_lo] == reg_file[reg_lo - 1])
         continue;
      if (reg_hi < ub - 1 && reg_file[reg_hi] != 0 && reg_file[reg_hi] == reg_file[reg_hi + 1])
         continue;

      /* count variables to be moved */
      for (unsigned j = reg_lo; j <= reg_hi; j++) {
         if (reg_file[j] != 0)
            k++;
      }

      /* count operands in wrong positions */
      for (unsigned j = 0, offset = 0; j < instr->num_operands; offset += instr->getOperand(j).size(), j++) {
         if (j == i ||
             !instr->getOperand(j).isTemp() ||
             instr->getOperand(j).getTemp().type() != rc.type())
            continue;
         if (instr->getOperand(j).physReg() != reg_lo + offset)
            k += instr->getOperand(j).size();
      }
      bool aligned = rc == RegClass::v4 && reg_lo % 4 == 0;
      if (k > num_moves || (!aligned && k == num_moves))
         continue;

      best_pos = reg_lo;
      num_moves = k;
   }

   if (num_moves >= size)
      return get_reg(ctx, reg_file, rc, parallelcopies, instr);

   /* collect variables to be moved */
   std::set<std::pair<unsigned, unsigned>> vars;
   for (unsigned i = best_pos; i < best_pos + size; i++) {
      if (reg_file[i] != 0)
         vars.emplace(ctx.assignments[reg_file[i]].second.size(), reg_file[i]);
      reg_file[i] = 0;
   }

   /* move killed operands which aren't yet at the correct position */
   for (unsigned i = 0, offset = 0; i < instr->num_operands; offset += instr->getOperand(i).size(), i++) {
      if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill() && instr->getOperand(i).getTemp().type() == rc.type()) {
         if (instr->getOperand(i).physReg() != best_pos + offset) {
            vars.emplace(instr->getOperand(i).size(), instr->getOperand(i).tempId());
         } else {
            for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
               reg_file[instr->getOperand(i).physReg() + j] = instr->getOperand(i).tempId();
         }
      }
   }

   get_regs_for_copies(ctx, reg_file, parallelcopies, vars, lb, ub, instr, best_pos, best_pos + size - 1);

   update_renames(ctx, reg_file, parallelcopies, instr);
   adjust_max_used_regs(ctx, rc, best_pos);
   return PhysReg{best_pos};
}

bool get_reg_specified(ra_ctx& ctx,
                       std::array<uint32_t, 512>& reg_file,
                       RegClass rc,
                       std::vector<std::pair<Operand, Definition>>& parallelcopies,
                       aco_ptr<Instruction>& instr,
                       PhysReg reg)
{
   uint32_t size = rc.size();
   uint32_t stride = 1;
   uint32_t lb, ub;

   if (rc.type() == vgpr) {
      lb = 256;
      ub = 256 + ctx.program->max_vgpr;
   } else {
      if (size == 2)
         stride = 2;
      else if (size >= 4)
         stride = 4;
      if (reg % stride != 0)
         return false;
      lb = 0;
      ub = ctx.program->max_sgpr;
   }

   uint32_t reg_lo = reg.reg;
   uint32_t reg_hi = reg + (size - 1);

   if (reg_lo < lb || reg_hi >= ub || reg_lo > reg_hi)
      return false;

   for (unsigned i = reg_lo; i <= reg_hi; i++) {
      if (reg_file[i] != 0)
         return false;
   }
   adjust_max_used_regs(ctx, rc, reg_lo);
   return true;
}

void handle_pseudo(ra_ctx& ctx,
                   const std::array<uint32_t, 512>& reg_file,
                   Instruction* instr)
{
   if (instr->format != Format::PSEUDO)
      return;

   /* all instructions which use handle_operands() need this information */
   switch (instr->opcode) {
   case aco_opcode::p_extract_vector:
   case aco_opcode::p_create_vector:
   case aco_opcode::p_split_vector:
   case aco_opcode::p_parallelcopy:
      break;
   default:
      return;
   }

   Pseudo_instruction *pi = (Pseudo_instruction *)instr;
   if (reg_file[scc.reg]) {
      pi->tmp_in_scc = true;

      int reg = ctx.max_used_sgpr;
      for (; reg >= 0 && reg_file[reg]; reg--)
         ;
      if (reg < 0) {
         reg = ctx.max_used_sgpr + 1;
         for (; reg < ctx.program->max_sgpr && reg_file[reg]; reg++)
            ;
         assert(reg < ctx.program->max_sgpr);
      }

      adjust_max_used_regs(ctx, s1, reg);
      pi->scratch_sgpr = PhysReg{(unsigned)reg};
   } else {
      pi->tmp_in_scc = false;
   }
}

bool operand_can_use_reg(aco_ptr<Instruction>& instr, unsigned idx, PhysReg reg)
{
   switch (instr->format) {
   case Format::SMEM:
      return reg != scc &&
             reg != exec &&
             (reg != m0 || idx == 1 || idx == 3) && /* offset can be m0 */
             (reg != vcc || (instr->num_definitions == 0 && idx == 2)); /* sdata can be vcc */
   default:
      // TODO: there are more instructions with restrictions on registers
      return true;
   }
}

} /* end namespace */


void register_allocation(Program *program, std::vector<std::set<Temp>> live_out_per_block)
{
   ra_ctx ctx(program);

   std::vector<std::unordered_map<unsigned, Temp>> renames(program->blocks.size());

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
   std::function<Temp(Temp,unsigned)> read_variable;
   std::function<Temp(Temp,Block*)> handle_live_in;
   std::function<Temp(std::map<unsigned, phi_info>::iterator)> try_remove_trivial_phi;

   read_variable = [&](Temp val, unsigned block_idx) -> Temp {
      std::unordered_map<unsigned, Temp>::iterator it = renames[block_idx].find(val.id());
      assert(it != renames[block_idx].end());
      return it->second;
   };

   handle_live_in = [&](Temp val, Block *block) -> Temp {
      std::vector<unsigned>& preds = val.is_linear() ? block->linear_preds : block->logical_preds;
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
         aco_ptr<Instruction> phi{create_instruction<Pseudo_instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
         phi->getDefinition(0) = Definition(new_val);
         for (unsigned i = 0; i < preds.size(); i++)
            phi->getOperand(i) = Operand(val);
         if (tmp.regClass() == new_val.regClass())
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
            aco_ptr<Instruction> phi{create_instruction<Pseudo_instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
            new_val = Temp{program->allocateId(), val.regClass()};
            phi->getDefinition(0) = Definition(new_val);
            for (unsigned i = 0; i < preds.size(); i++) {
               phi->getOperand(i) = Operand(ops[i]);
               phi->getOperand(i).setFixed(ctx.assignments[ops[i].id()].first);
               if (ops[i].regClass() == new_val.regClass())
                  affinities[new_val.id()] = ops[i].id();
            }
            phi_map.emplace(new_val.id(), phi_info{phi.get(), block->index});
            block->instructions.insert(block->instructions.begin(), std::move(phi));
         }
      }

      renames[block->index][val.id()] = new_val;
      renames[block->index][new_val.id()] = new_val;
      ctx.orig_names[new_val.id()] = val;
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
         if (is_phi(instr)) {
            std::map<unsigned, phi_info>::iterator it = phi_map.find(instr->getDefinition(0).tempId());
            if (it != phi_map.end() && it != info)
               phi_users.emplace_back(it);
         }
      }

      auto it = ctx.orig_names.find(same.id());
      unsigned orig_var = it != ctx.orig_names.end() ? it->second.id() : same.id();
      for (unsigned i = 0; i < program->blocks.size(); i++) {
         auto it = renames[i].find(orig_var);
         if (it != renames[i].end() && it->second == def.getTemp())
            renames[i][orig_var] = same;
      }

      unsigned block_idx = info->second.block_idx;
      instr->num_definitions = 0; /* this indicates that the phi can be removed */
      phi_map.erase(info);
      for (auto it : phi_users)
         try_remove_trivial_phi(it);

      /* due to the removal of other phis, the name might have changed once again! */
      return renames[block_idx][orig_var];
   };

   std::map<unsigned, Instruction*> vectors;
   std::vector<std::vector<Temp>> phi_ressources;
   std::map<unsigned, unsigned> temp_to_phi_ressources;

   for (std::vector<Block>::reverse_iterator it = program->blocks.rbegin(); it != program->blocks.rend(); it++) {
      Block& block = *it;

      /* first, compute the death points of all live vars within the block */
      std::set<Temp>& live = live_out_per_block[block.index];

      std::vector<aco_ptr<Instruction>>::reverse_iterator rit;
      for (rit = block.instructions.rbegin(); rit != block.instructions.rend(); ++rit) {
         aco_ptr<Instruction>& instr = *rit;
         if (!is_phi(instr)) {
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
         } else if (!instr->getDefinition(0).isKill() && !instr->getDefinition(0).isFixed()) {
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

   std::vector<std::bitset<128>> sgpr_live_out(program->blocks.size());

   for (Block& block : program->blocks) {
      std::set<Temp>& live = live_out_per_block[block.index];
      /* initialize register file */
      assert(block.index != 0 || live.empty());
      std::array<uint32_t, 512> register_file = {0};
      ctx.war_hint.reset();

      for (Temp t : live) {
         Temp renamed = handle_live_in(t, &block);
         if (ctx.assignments.find(renamed.id()) != ctx.assignments.end()) {
            for (unsigned i = 0; i < t.size(); i++)
               register_file[ctx.assignments[renamed.id()].first + i] = renamed.id();
         }
      }

      std::vector<aco_ptr<Instruction>> instructions;
      std::vector<aco_ptr<Instruction>>::iterator it;

      /* this is a slight adjustment from the paper as we already have phi nodes:
       * We consider them incomplete phis and only handle the definition. */

      /* handle fixed phi definitions */
      for (it = block.instructions.begin(); it != block.instructions.end(); ++it) {
         aco_ptr<Instruction>& phi = *it;
         if (!is_phi(phi))
            break;
         Definition& definition = phi->getDefinition(0);
         if (definition.isKill() || !definition.isFixed())
            continue;

         assert(definition.physReg() == exec);
         for (unsigned i = 0; i < definition.size(); i++) {
            assert(register_file[definition.physReg() + i] == 0);
            register_file[definition.physReg() + i] = definition.tempId();
         }
      }

      /* look up the affinities */
      for (it = block.instructions.begin(); it != block.instructions.end(); ++it) {
         aco_ptr<Instruction>& phi = *it;
         if (!is_phi(phi))
            break;
         Definition& definition = phi->getDefinition(0);
         if (definition.isKill() || definition.isFixed())
             continue;

         if (affinities.find(definition.tempId()) != affinities.end() &&
             ctx.assignments.find(affinities[definition.tempId()]) != ctx.assignments.end()) {
            assert(ctx.assignments[affinities[definition.tempId()]].second == definition.regClass());
            PhysReg reg = ctx.assignments[affinities[definition.tempId()]].first;
            bool try_use_special_reg = reg == scc || reg == exec;
            if (try_use_special_reg) {
               for (unsigned i = 0; try_use_special_reg && i < phi->num_operands; i++) {
                  if (!phi->getOperand(i).isTemp() ||
                      ctx.assignments.find(phi->getOperand(i).tempId()) == ctx.assignments.end() ||
                      !(ctx.assignments[phi->getOperand(i).tempId()].first == reg)) {
                     try_use_special_reg = false;
                  }
               }
               if (!try_use_special_reg)
                  continue;
            }
            bool reg_free = true;
            for (unsigned i = reg.reg; reg_free && i < reg + definition.size(); i++) {
               if (register_file[i] != 0)
                  reg_free = false;
            }
            /* only assign if register is still free */
            if (reg_free) {
               definition.setFixed(reg);
               for (unsigned i = 0; i < definition.size(); i++)
                  register_file[definition.physReg() + i] = definition.tempId();
            }
         }
      }

      /* find registers for phis without affinity or where the register was blocked */
      for (it = block.instructions.begin();it != block.instructions.end(); ++it) {
         aco_ptr<Instruction>& phi = *it;
         if (!is_phi(phi))
            break;

         Definition& definition = phi->getDefinition(0);
         if (definition.isKill())
            continue;

         renames[block.index][definition.tempId()] = definition.getTemp();

         if (!definition.isFixed()) {
            std::vector<std::pair<Operand, Definition>> parallelcopy;
            /* try to find a register that is used by at least one operand */
            for (unsigned i = 0; i < phi->num_operands; i++) {
               if (!phi->getOperand(i).isTemp() ||
                   ctx.assignments.find(phi->getOperand(i).tempId()) == ctx.assignments.end())
                  continue;
               PhysReg reg = ctx.assignments[phi->getOperand(i).tempId()].first;
               /* we tried this already on the previous loop */
               if (reg == scc || reg == exec)
                  continue;
               if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, phi, reg)) {
                  definition.setFixed(reg);
                  break;
               }
            }
            if (!definition.isFixed())
               definition.setFixed(get_reg(ctx, register_file, definition.regClass(), parallelcopy, phi));

            /* process parallelcopy */
            for (std::pair<Operand, Definition> pc : parallelcopy) {
               /* rename */
               std::map<unsigned, Temp>::iterator orig_it = ctx.orig_names.find(pc.first.tempId());
               Temp orig = pc.first.getTemp();
               if (orig_it != ctx.orig_names.end())
                  orig = orig_it->second;
               else
                  ctx.orig_names[pc.second.tempId()] = orig;
               renames[block.index][orig.id()] = pc.second.getTemp();
               renames[block.index][pc.second.tempId()] = pc.second.getTemp();

               /* see if it's a copy from a previous phi */
               //TODO: prefer moving some previous phis over live-ins
               Instruction *prev_phi = NULL;
               for (auto it2 = block.instructions.begin(); !prev_phi && (it2 != it); ++it2) {
                  if (*it2 && (*it2)->getDefinition(0).tempId() == pc.first.tempId())
                     prev_phi = it2->get();
               }
               if (prev_phi) {
                  /* if so, just update that phi */
                  prev_phi->getDefinition(0) = pc.second;
                  continue;
               }

               /* otherwise, this is a live-in and we need to create a new phi
                * to move it in this block's predecessors */
               aco_opcode opcode = pc.first.getTemp().is_linear() ? aco_opcode::p_linear_phi : aco_opcode::p_phi;
               std::vector<unsigned>& preds = pc.first.getTemp().is_linear() ? block.linear_preds : block.logical_preds;
               aco_ptr<Instruction> new_phi{create_instruction<Pseudo_instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
               new_phi->getDefinition(0) = pc.second;
               for (unsigned i = 0; i < preds.size(); i++)
                  new_phi->getOperand(i) = Operand(pc.first);
               instructions.emplace_back(std::move(new_phi));
            }

            for (unsigned i = 0; i < definition.size(); i++)
               register_file[definition.physReg() + i] = definition.tempId();
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
      for (; it != block.instructions.end(); ++it) {
         aco_ptr<Instruction>& instr = *it;
         std::vector<std::pair<Operand, Definition>> parallelcopy;

         assert(!is_phi(instr));

         /* handle operands */
         for (unsigned i = 0; i < instr->num_operands; ++i) {
            auto& operand = instr->getOperand(i);
            if (!operand.isTemp())
               continue;

            /* rename operands */
            operand.setTemp(read_variable(operand.getTemp(), block.index));

            /* check if the operand is fixed */
            if (operand.isFixed()) {

               if (operand.physReg() == ctx.assignments[operand.tempId()].first) {
                  /* we are fine: the operand is already assigned the correct reg */

               } else {
                  /* check if target reg is blocked, and move away the blocking var */
                  if (register_file[operand.physReg().reg]) {
                     uint32_t blocking_id = register_file[operand.physReg().reg];
                     RegClass rc = ctx.assignments[blocking_id].second;
                     Operand pc_op = Operand(Temp{blocking_id, rc});
                     pc_op.setFixed(operand.physReg());
                     Definition pc_def = Definition(Temp{program->allocateId(), pc_op.regClass()});
                     /* find free reg */
                     PhysReg reg = get_reg(ctx, register_file, pc_op.regClass(), parallelcopy, instr);
                     pc_def.setFixed(reg);
                     ctx.assignments[pc_def.tempId()] = {reg, pc_def.regClass()};
                     for (unsigned i = 0; i < operand.size(); i++) {
                        register_file[pc_op.physReg() + i] = 0;
                        register_file[pc_def.physReg() + i] = pc_def.tempId();
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
                  Temp tmp = Temp{program->allocateId(), operand.regClass()};
                  Definition pc_def = Definition(tmp);
                  pc_def.setFixed(operand.physReg());
                  pc_op.setFixed(ctx.assignments[operand.tempId()].first);
                  operand.setTemp(tmp);
                  ctx.assignments[tmp.id()] = {pc_def.physReg(), pc_def.regClass()};
                  operand.setFixed(pc_def.physReg());
                  for (unsigned i = 0; i < operand.size(); i++) {
                     register_file[pc_op.physReg() + i] = 0;
                     register_file[pc_def.physReg() + i] = tmp.id();
                  }
                  parallelcopy.emplace_back(pc_op, pc_def);
               }
            } else {
               assert(ctx.assignments.find(operand.tempId()) != ctx.assignments.end());
               PhysReg reg = ctx.assignments[operand.tempId()].first;

               if (operand_can_use_reg(instr, i, reg)) {
                  operand.setFixed(ctx.assignments[operand.tempId()].first);
               } else {
                  Operand pc_op = operand;
                  pc_op.setFixed(reg);
                  PhysReg new_reg = get_reg(ctx, register_file, operand.regClass(), parallelcopy, instr);
                  Definition pc_def = Definition(program->allocateId(), new_reg, pc_op.regClass());
                  ctx.assignments[pc_def.tempId()] = {reg, pc_def.regClass()};
                  for (unsigned i = 0; i < operand.size(); i++) {
                        register_file[pc_op.physReg() + i] = 0;
                        register_file[pc_def.physReg() + i] = pc_def.tempId();
                  }
                  parallelcopy.emplace_back(pc_op, pc_def);
                  operand.setFixed(new_reg);
               }

               if (instr->format == Format::EXP)
                  ctx.war_hint.set(operand.physReg().reg);
            }
            std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.getTemp().id());
            if (phi != phi_map.end())
               phi->second.uses.emplace(instr.get());

         }
         /* remove dead vars from register file */
         for (unsigned i = 0; i < instr->num_operands; i++)
            if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
               for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                  register_file[instr->getOperand(i).physReg() + j] = 0;

         /* try to optimize v_mad_f32 -> v_mac_f32 */
         if (instr->opcode == aco_opcode::v_mad_f32 &&
             instr->getOperand(2).isTemp() &&
             instr->getOperand(2).isKill() &&
             instr->getOperand(2).getTemp().type() == vgpr &&
             instr->getOperand(1).isTemp() &&
             instr->getOperand(1).getTemp().type() == vgpr) { /* TODO: swap src0 and src1 in this case */
            VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr.get());
            bool can_use_mac = !(vop3->abs[0] || vop3->abs[1] || vop3->abs[2] ||
                                 vop3->opsel[0] || vop3->opsel[1] || vop3->opsel[2] ||
                                 vop3->neg[0] || vop3->neg[1] || vop3->neg[2] ||
                                 vop3->clamp || vop3->omod);
            if (can_use_mac) {
               instr->format = Format::VOP2;
               instr->opcode = aco_opcode::v_mac_f32;
            }
         }

         /* handle definitions which must have the same register as an operand */
         if (instr->opcode == aco_opcode::v_interp_p2_f32 ||
             instr->opcode == aco_opcode::v_mac_f32 ||
             instr->opcode == aco_opcode::v_writelane_b32) {
            if (!instr->getOperand(2).isUndefined())
               instr->getDefinition(0).setFixed(instr->getOperand(2).physReg());
         } else if (instr->opcode == aco_opcode::s_addk_i32 ||
                  instr->opcode == aco_opcode::s_mulk_i32 ||
                  instr->opcode == aco_opcode::p_wqm) {
            if (!instr->getOperand(0).isUndefined())
               instr->getDefinition(0).setFixed(instr->getOperand(0).physReg());
         } else if ((instr->format == Format::MUBUF ||
                   instr->format == Format::MIMG) &&
                  instr->num_definitions == 1 &&
                  instr->num_operands == 4) {
            if (!instr->getOperand(3).isUndefined())
               instr->getDefinition(0).setFixed(instr->getOperand(3).physReg());
         }

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
                  RegClass rc = pc_op.regClass();
                  tmp = Temp{program->allocateId(), rc};
                  Definition pc_def = Definition(tmp);

                  /* re-enable the killed operands, so that we don't move the blocking var there */
                  for (unsigned i = 0; i < instr->num_operands; i++)
                     if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                        for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                           register_file[instr->getOperand(i).physReg() + j] = 0xFFFF;
                  /* find a new register for the blocking variable */
                  PhysReg reg = get_reg(ctx, register_file, rc, parallelcopy, instr);
                  /* once again, disable killed operands */
                  for (unsigned i = 0; i < instr->num_operands; i++) {
                     if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                        for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                           register_file[instr->getOperand(i).physReg() + j] = 0;
                  }
                  for (unsigned k = 0; k < i; k++) {
                     if (instr->getDefinition(k).isTemp() && !instr->getDefinition(k).isKill())
                        for (unsigned j = 0; j < instr->getDefinition(k).size(); j++)
                           register_file[instr->getDefinition(k).physReg() + j] = instr->getDefinition(k).tempId();
                  }
                  pc_def.setFixed(reg);

                  /* finish assignment of parallelcopy */
                  ctx.assignments[pc_def.tempId()] = {reg, pc_def.regClass()};
                  parallelcopy.emplace_back(pc_op, pc_def);

                  /* add changes to reg_file */
                  for (unsigned i = 0; i < pc_op.size(); i++) {
                     register_file[pc_op.physReg() + i] = 0x0;
                     register_file[pc_def.physReg() + i] = pc_def.tempId();
                  }
               }
            } else if (definition.isTemp()) {
               /* find free reg */
               if (definition.hasHint() && register_file[definition.physReg().reg] == 0)
                  definition.setFixed(definition.physReg());
               else if (instr->opcode == aco_opcode::p_split_vector) {
                  PhysReg reg = PhysReg{instr->getOperand(0).physReg() + i * definition.size()};
                  if (!get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg))
                     reg = get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr);
                  definition.setFixed(reg);
               } else if (instr->opcode == aco_opcode::p_extract_vector) {
                  PhysReg reg;
                  if (instr->getOperand(0).isKill() &&
                      instr->getOperand(0).getTemp().type() == definition.getTemp().type()) {
                     reg = instr->getOperand(0).physReg();
                     reg.reg += definition.size() * instr->getOperand(1).constantValue();
                     assert(register_file[reg.reg] == 0);
                  } else {
                     reg = get_reg(ctx, register_file, definition.regClass(), parallelcopy, instr);
                  }
                  definition.setFixed(reg);
               } else if (instr->opcode == aco_opcode::p_create_vector) {
                  PhysReg reg = get_reg_create_vector(ctx, register_file, definition.regClass(),
                                                      parallelcopy, instr);
                  definition.setFixed(reg);
               } else if (affinities.find(definition.tempId()) != affinities.end() &&
                          ctx.assignments.find(affinities[definition.tempId()]) != ctx.assignments.end()) {
                  PhysReg reg = ctx.assignments[affinities[definition.tempId()]].first;
                  if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg))
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
                        reg.reg = reg - k + offset;
                        if (get_reg_specified(ctx, register_file, definition.regClass(), parallelcopy, instr, reg)) {
                           definition.setFixed(reg);
                           break;
                        }
                     }
                     k += vec->getOperand(i).size();
                  }
                  if (!definition.isFixed()) {
                     std::pair<PhysReg, bool> res = get_reg_vec(ctx, register_file, vec->getDefinition(0).regClass());
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

               assert(definition.isFixed() && ((definition.getTemp().type() == vgpr && definition.physReg() >= 256) ||
                                               (definition.getTemp().type() != vgpr && definition.physReg() < 256)));
            } else {
               continue;
            }

            ctx.assignments[definition.tempId()] = {definition.physReg(), definition.regClass()};
            for (unsigned i = 0; i < definition.size(); i++)
               register_file[definition.physReg() + i] = definition.tempId();
            /* set live if it has a kill point */
            if (!definition.isKill()) {
               live.emplace(definition.getTemp());
            }
            /* add to renames table */
            renames[block.index][definition.tempId()] = definition.getTemp();
         }

         handle_pseudo(ctx, register_file, instr.get());

         /* kill definitions */
         for (unsigned i = 0; i < instr->num_definitions; i++)
             if (instr->getDefinition(i).isTemp() && instr->getDefinition(i).isKill())
                for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                   register_file[instr->getDefinition(i).physReg() + j] = 0;

         /* emit parallelcopy */
         if (!parallelcopy.empty()) {
            aco_ptr<Pseudo_instruction> pc;
            pc.reset(create_instruction<Pseudo_instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, parallelcopy.size(), parallelcopy.size()));
            bool temp_in_scc = register_file[scc.reg];
            bool sgpr_operands_alias_defs = false;
            uint64_t sgpr_operands[4] = {0, 0, 0, 0};
            for (unsigned i = 0; i < parallelcopy.size(); i++) {
               if (temp_in_scc && parallelcopy[i].first.isTemp() && parallelcopy[i].first.getTemp().type() == sgpr) {
                  if (!sgpr_operands_alias_defs) {
                     unsigned reg = parallelcopy[i].first.physReg().reg;
                     unsigned size = parallelcopy[i].first.getTemp().size();
                     sgpr_operands[reg / 64u] |= ((1u << size) - 1) << (reg % 64u);

                     reg = parallelcopy[i].second.physReg().reg;
                     size = parallelcopy[i].second.getTemp().size();
                     if (sgpr_operands[reg / 64u] & ((1u << size) - 1) << (reg % 64u))
                        sgpr_operands_alias_defs = true;
                  }
               }

               pc->getOperand(i) = parallelcopy[i].first;
               pc->getDefinition(i) = parallelcopy[i].second;

               /* it might happen that the operand is already renamed. we have to restore the original name. */
               std::map<unsigned, Temp>::iterator it = ctx.orig_names.find(pc->getOperand(i).tempId());
               if (it != ctx.orig_names.end())
                  pc->getOperand(i).setTemp(it->second);
               unsigned orig_id = pc->getOperand(i).tempId();
               ctx.orig_names[pc->getDefinition(i).tempId()] = pc->getOperand(i).getTemp();

               pc->getOperand(i).setTemp(read_variable(pc->getOperand(i).getTemp(), block.index));
               renames[block.index][orig_id] = pc->getDefinition(i).getTemp();
               renames[block.index][pc->getDefinition(i).tempId()] = pc->getDefinition(i).getTemp();
               std::map<unsigned, phi_info>::iterator phi = phi_map.find(pc->getOperand(i).tempId());
               if (phi != phi_map.end())
                  phi->second.uses.emplace(pc.get());
            }

            if (temp_in_scc && sgpr_operands_alias_defs) {
               /* disable definitions and re-enable operands */
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  if (instr->getDefinition(i).isTemp() && !instr->getDefinition(i).isKill())
                     for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                        register_file[instr->getDefinition(i).physReg() + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill()) {
                     for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                        register_file[instr->getOperand(i).physReg() + j] = 0xFFFF;
                  }
               }

               handle_pseudo(ctx, register_file, pc.get());

               /* re-enable live vars */
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill())
                     for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                        register_file[instr->getOperand(i).physReg() + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  if (instr->getDefinition(i).isTemp() && !instr->getDefinition(i).isKill())
                     for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                        register_file[instr->getDefinition(i).physReg() + j] = instr->getDefinition(i).tempId();
               }
            } else {
               pc->tmp_in_scc = false;
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
                     register_file[instr->getDefinition(i).physReg() + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_operands; i++) {
                  if (instr->getOperand(i).isTemp() && instr->getOperand(i).isFirstKill()) {
                     for (unsigned j = 0; j < instr->getOperand(i).size(); j++)
                        register_file[instr->getOperand(i).physReg() + j] = 0xFFFF;
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
                        register_file[instr->getOperand(i).physReg() + j] = 0x0;
               }
               for (unsigned i = 0; i < instr->num_definitions; i++) {
                  if (instr->getDefinition(i).isTemp() && !instr->getDefinition(i).isKill())
                     for (unsigned j = 0; j < instr->getDefinition(i).size(); j++)
                        register_file[instr->getDefinition(i).physReg() + j] = instr->getDefinition(i).tempId();
               }
            }

            /* change the instruction to VOP3 to enable an arbitrary register pair as dst */
            aco_ptr<Instruction> tmp = std::move(instr);
            Format format = asVOP3(tmp->format);
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

      block.instructions = std::move(instructions);

      filled[block.index] = true;
      for (unsigned succ_idx : block.linear_succs) {
         Block& succ = program->blocks[succ_idx];
         /* seal block if all predecessors are filled */
         bool all_filled = true;
         for (unsigned pred_idx : succ.linear_preds) {
            if (!filled[pred_idx]) {
               all_filled = false;
               break;
            }
         }
         if (all_filled) {
            /* finish incomplete phis and check if they became trivial */
            for (Instruction* phi : incomplete_phis[succ_idx]) {
               std::vector<unsigned> preds = phi->getDefinition(0).getTemp().is_linear() ? succ.linear_preds : succ.logical_preds;
               for (unsigned i = 0; i < phi->num_operands; i++) {
                  phi->getOperand(i).setTemp(read_variable(phi->getOperand(i).getTemp(), preds[i]));
                  phi->getOperand(i).setFixed(ctx.assignments[phi->getOperand(i).tempId()].first);
               }
               try_remove_trivial_phi(phi_map.find(phi->getDefinition(0).tempId()));
            }
            /* complete the original phi nodes, but no need to check triviality */
            for (aco_ptr<Instruction>& instr : succ.instructions) {
               if (!is_phi(instr))
                  break;
               std::vector<unsigned> preds = instr->opcode == aco_opcode::p_phi ? succ.logical_preds : succ.linear_preds;

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
            sealed[succ_idx] = true;
         }
      }

      /* fill in sgpr_live_out */
      for (unsigned i = 0; i < ctx.max_used_sgpr; i++) {
         if (register_file[i])
            sgpr_live_out[block.index].set(i);
      }
   } /* end for BB */

   /* find scc spill registers which may be needed for parallelcopies created by phis */
   for (Block& block : program->blocks) {
      if (block.linear_succs.size() != 1)
         continue;
      Block& succ = program->blocks[block.linear_succs[0]];
      unsigned pred_index = 0;
      for (; pred_index < succ.linear_preds.size() &&
             succ.linear_preds[pred_index] != block.index; pred_index++)
         ;
      assert(pred_index < succ.linear_preds.size());

      std::bitset<128> regs = sgpr_live_out[block.index];
      if (!regs[scc.reg]) {
         /* early exit */
         block.scc_live_out = false;
         continue;
      }

      bool has_phi = false;

      /* remove phi operands and add phi definitions */
      for (aco_ptr<Instruction>& instr : succ.instructions) {
         if (!is_phi(instr))
            break;
         if (instr->opcode == aco_opcode::p_linear_phi) {
            has_phi = true;

            Definition& def = instr->getDefinition(0);
            assert(def.getTemp().type() == sgpr);
            for (unsigned i = 0; i < def.size(); i++)
               regs[def.physReg() + i] = 1;
            if (instr->getOperand(pred_index).isTemp()) {
               Operand& op = instr->getOperand(pred_index);
               assert(op.isFixed());
               for (unsigned i = 0; i < op.size(); i++)
                  regs[op.physReg() + i] = 0;
            }
         }
      }

      if (!has_phi || !regs[scc.reg]) {
         block.scc_live_out = false;
         continue;
      }
      block.scc_live_out = true;

      /* choose a register */
      unsigned reg = 0;
      for (; reg < ctx.program->max_sgpr && regs[reg]; reg++)
         ;
      assert(reg < ctx.program->max_sgpr);
      adjust_max_used_regs(ctx, s1, reg);
      block.scratch_sgpr = PhysReg{reg};
   }

   /* remove trivial phis */
   for (Block& block : program->blocks) {
      std::vector<aco_ptr<Instruction>>::iterator it = block.instructions.begin();
      for (; it != block.instructions.end();) {
         if (!is_phi(*it))
            break;
         if (!(*it)->num_definitions)
            it = block.instructions.erase(it);
         else
            ++it;
      }
   }

   /* num_gpr = rnd_up(max_used_gpr + 1) */
   program->config->num_vgprs = (ctx.max_used_vgpr + 1 + 3) & ~3;
   if (program->family == CHIP_TONGA) {
      assert(ctx.max_used_sgpr <= 93);
      ctx.max_used_sgpr = 93; /* workaround hardware bug */
   }
   program->config->num_sgprs = (ctx.max_used_sgpr + 1 + 2 + 7) & ~7; /* + 2 sgprs for vcc */
}

}
