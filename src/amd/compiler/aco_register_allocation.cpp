#include "aco_ir.h"

#include <algorithm>
#include <set>
#include <vector>
#include <unordered_map>

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
               assert (it2 != live.end());
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

         if (needMove) {
            std::unique_ptr<Instruction> move{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, live.size(), live.size())};
            int idx = 0;
            for (auto e : live) {
               insn->getOperand(idx) = Operand{e.second};
               insn->getDefinition(idx) = Definition{e.second};
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
         for (int i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            operand.setTemp(Temp{renames[operand.tempId()], operand.regClass()});
         }
         for (int i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            unsigned id = definition.tempId();
            if (renames.find(id) != renames.end()) {
               id = program->allocateId();
               renames[definition.tempId()] = id;
            }
            definition.setTemp(Temp{id, definition.regClass()});
         }
      }
   }
}

void register_allocation(Program *program)
{
   insert_copies(program);
   fix_ssa(program);

   std::unordered_map<unsigned, PhysReg> assignments;
   /* First assign fixed regs. */
   for(auto&& block : program->blocks) {
      for (auto&& insn : block->instructions) {
         for(unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp() && operand.isFixed()) {
               assignments[operand.tempId()] = operand.physReg();
            }
         }

         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp() && definition.isFixed()) {
               assignments[definition.tempId()] = definition.physReg();
            }
         }
      }
   }

   std::set<void*> kills;
   std::set<std::pair<unsigned, unsigned>> interfere;

   /* Determine operands & defs which are the last use of a temp. */
   /* Also create an interference graph. */
   for(auto b_it = program->blocks.rbegin(); b_it != program->blocks.rend(); ++b_it) {
      std::set<unsigned> live;
      for(auto i_it = (*b_it)->instructions.rbegin(); i_it != (*b_it)->instructions.rend(); ++i_it) {
         auto& insn = *i_it;
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
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
   for (auto&& block : program->blocks) {
      for(auto&& insn : block->instructions) {

      }
   }
}


}
