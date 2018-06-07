#include "aco_ir.h"

#include <algorithm>
#include <set>
#include <vector>
#include <unordered_map>

#include "sid.h"

namespace aco {
// TODO most of the functions here do not support multiple basic blocks yet.

/* Insert copies of all live temps before an instruction that uses any of
 * the temps at a fixed register. That way we avoid collisions where multiple
 * temps are assigned the same fixed register. This does not properly keep
 * the SSA property, so this needs fix_ssa afterwards.
 */
void insert_copies(Program * program)
{
   for (auto&& block : program->blocks) {
      std::vector<std::unique_ptr<Instruction>> instructions;
      std::unordered_map<unsigned, Temp> live;
      for (auto it = block->instructions.rbegin(); it != block->instructions.rend(); ++it) {
         Instruction *insn = it->get();
         bool needMove = false;
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               auto it2 = live.find(definition.tempId());
               if (it2 != live.end())
                  live.erase(it2);
               if (definition.isFixed())
                  needMove = true;
            }
         }

         for (unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               live[operand.tempId()] = operand.getTemp();
               if (operand.isFixed())
                  needMove = true;
            }
         }

         instructions.push_back(std::move(*it));

         if (needMove && !live.empty()) {
            std::unique_ptr<Instruction> move{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, live.size(), live.size())};
            int idx = 0;
            for (auto e : live) {
               //std::cerr << "Move " << e.second.id() << "\n";
               move->getOperand(idx) = Operand{e.second};
               move->getDefinition(idx) = Definition{e.second};
               ++idx;
            }
            instructions.push_back(std::move(move));
         }
      }
      std::reverse(instructions.begin(), instructions.end());
      block->instructions = std::move(instructions);
   }
}

/* Splits temps with multiple defs into multiple temps according to
 * SSa constraints.
 */
void fix_ssa(Program *program)
{
   for (auto&& block : program->blocks) {
      std::unordered_map<unsigned, unsigned> renames;
      for(auto&& insn : block->instructions) {
         for (unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               assert(renames.find(operand.tempId()) != renames.end());
               operand.setTemp(Temp{renames[operand.tempId()], operand.regClass()});
            }
         }
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            unsigned id = definition.tempId();
            if (renames.find(id) != renames.end()) {
               id = program->allocateId();
               renames[definition.tempId()] = id;
            } else {
               renames[definition.tempId()] = definition.tempId();
            }
            definition.setTemp(Temp{id, definition.regClass()});
         }
      }
   }
}

static unsigned reg_count(RegClass reg_class)
{
   switch (reg_class) {
      case b:
      case s1:
      case v1:
         return 1;
      case s2:
      case v2:
         return 2;
      case s3:
      case v3:
         return 3;
      case s4:
      case v4:
         return 4;
      case s8:
         return 8;
      case s16:
         return 16;
      default:
         abort();
   }
}

bool can_allocate_register(const std::set<std::pair<unsigned, unsigned>> *interfere,
                           const std::unordered_map<unsigned, std::pair<PhysReg, unsigned>> *assignments,
                           unsigned id, unsigned reg, unsigned count)
{
   for (auto it = interfere->lower_bound({id, 0}); it != interfere->end() && it->first == id; ++it) {
      unsigned other_id = it->second;
      auto other_it = assignments->find(other_id);
      if (other_it == assignments->end())
         continue;

      unsigned other_count = other_it->second.second;
      unsigned other_reg = other_it->second.first.reg;
      if (other_reg + other_count <= reg || reg + count <= other_reg)
         continue;

      return false;
   }
   return true;
}

void register_allocation(Program *program)
{
   unsigned num_accessed_sgpr = 0;
   unsigned num_accessed_vgpr = 0;
   insert_copies(program);
   fix_ssa(program);

   std::unordered_map<unsigned, std::pair<PhysReg, unsigned>> assignments;
   std::unordered_map<unsigned, unsigned> temp_assignments;
   /* First assign fixed regs. */
   for(auto&& block : program->blocks) {
      for (auto&& insn : block->instructions) {
         for(unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp() && operand.isFixed()) {
               assignments[operand.tempId()] = {operand.physReg(), reg_count(operand.regClass())};
            }
         }

         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp() && definition.isFixed()) {
               assignments[definition.tempId()] = {definition.physReg(), reg_count(definition.regClass())};
            }
         }
      }
   }

   std::set<void*> kills;
   std::set<std::pair<unsigned, unsigned>> interfere;
   std::vector<Definition*> simplicial_order;

   /* Determine operands & defs which are the last use of a temp. */
   /* Also create an interference graph. */
   for(auto b_it = program->blocks.rbegin(); b_it != program->blocks.rend(); ++b_it) {
      std::set<unsigned> live;
      for(auto i_it = (*b_it)->instructions.rbegin(); i_it != (*b_it)->instructions.rend(); ++i_it) {
         auto& insn = *i_it;
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               simplicial_order.push_back(&definition);

               auto it = live.find(definition.tempId());
               if (it != live.end())
                  live.erase(it);
               else
                  kills.insert(&definition);
               for(auto e : live) {
                  if (e != definition.tempId()) {
                     interfere.insert({definition.tempId(), e});
                     interfere.insert({e, definition.tempId()});
                  }
               }
            }
         }

         if (insn->opcode == aco_opcode::v_interp_p1_f32 ||
             insn->opcode == aco_opcode::v_interp_p2_f32) {
            auto& definition = insn->getDefinition(0);
            auto& operand = insn->getOperand(0);
            interfere.insert({definition.tempId(), operand.tempId()});
            interfere.insert({operand.tempId(), definition.tempId()});
         }

         if (insn->opcode == aco_opcode::v_interp_p2_f32 ||
             insn->opcode == aco_opcode::v_mac_f32)
            temp_assignments[insn->getDefinition(0).tempId()] = insn->getOperand(2).tempId();

         for(unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               auto it = live.find(operand.tempId());
               if (it == live.end()) {
                  live.insert(operand.tempId());
                  kills.insert(&operand);
               }
            }
         }
      }
   }

   /* Finish the assignment */
   for(auto b_it = program->blocks.rbegin(); b_it != program->blocks.rend(); ++b_it) {
      std::set<unsigned> live;
      for(auto i_it = (*b_it)->instructions.begin(); i_it != (*b_it)->instructions.end(); ++i_it) {
         auto& insn = *i_it;
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               auto id = definition.tempId();
               if (assignments.find(id) != assignments.end())
                  continue;
               if (temp_assignments.find(id) != temp_assignments.end()){
                  assignments[id] = assignments[temp_assignments.find(id)->second];
                  continue;
               }

               unsigned count = reg_count(definition.regClass());
               unsigned alignment = 1;
               unsigned start = 0;
               unsigned end = 256;
               unsigned alloc_reg = 0;
               bool allocated = false;

               if (definition.getTemp().type() == RegType::vgpr) {
                  start = 256;
                  end = 512;
               } else {
                  if (!((count - 1) & count))
                     alignment = count;
               }

               if (insn->opcode == aco_opcode::p_parallelcopy) {
                  unsigned preferred_reg = assignments.find(insn->getOperand(i).tempId())->second.first.reg;
                  if (can_allocate_register(&interfere, &assignments, id, preferred_reg, count)) {
                     alloc_reg = preferred_reg;
                     allocated = true;
                  }
               }

               if (!allocated) {
                  /* try hint first if available */
                  if (definition.hasHint() && can_allocate_register(&interfere, &assignments, id, definition.physReg().reg, count)) {
                     alloc_reg = definition.physReg().reg;

                  } else {
                     for(unsigned reg = start; reg + count <= end; reg += alignment) {
                        if (can_allocate_register(&interfere, &assignments, id, reg, count)) {
                           alloc_reg = reg;
                           break;
                        }
                     }
                  }
               }

               assignments[id] = {PhysReg{alloc_reg}, count};
               if (definition.getTemp().type() == RegType::vgpr)
                  num_accessed_vgpr = std::max(num_accessed_vgpr, alloc_reg - 256 + definition.getTemp().size());
               else if (alloc_reg <= 102)
                  num_accessed_sgpr = std::max(num_accessed_sgpr, alloc_reg + definition.getTemp().size());
            }
         }
      }
   }

   program->config->num_vgprs = num_accessed_vgpr;
   program->config->num_sgprs = num_accessed_sgpr + 2;
   program->config->spi_ps_input_addr = S_0286CC_PERSP_CENTER_ENA(1);
   program->config->spi_ps_input_ena = S_0286CC_PERSP_CENTER_ENA(1);

   for(auto&& block : program->blocks) {
      for (auto&& insn : block->instructions) {
         for(unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               auto it = assignments.find(operand.tempId());
               assert(it != assignments.end());
               operand.setFixed(it->second.first);
            }
         }

         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               auto it = assignments.find(definition.tempId());
               assert(it != assignments.end());
               definition.setFixed(it->second.first);
            }
         }
      }
   }
}


}
