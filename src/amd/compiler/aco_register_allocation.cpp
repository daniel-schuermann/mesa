#include "aco_ir.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <functional>

#include "sid.h"

namespace aco {

/* Insert copies of all live temps before an instruction that uses any of
 * the temps at a fixed register. That way we avoid collisions where multiple
 * temps are assigned the same fixed register. This does not properly keep
 * the SSA property, so this needs fix_ssa afterwards.
 */
void insert_copies(Program *program, std::vector<std::set<Temp>> live_out_per_block)
{

   for (auto&& block : program->blocks) {
      std::vector<std::unique_ptr<Instruction>> instructions;
      std::set<Temp> live = live_out_per_block[block.get()->index];

      for (auto it = block->instructions.rbegin(); it != block->instructions.rend(); ++it) {
         Instruction *insn = it->get();
         bool need_sgpr_move = false;
         bool need_vgpr_move = false;
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               auto temp = live.find(definition.getTemp());
               if (temp != live.end())
                  live.erase(temp);
               if (definition.isFixed())
                  (definition.getTemp().type() == vgpr ? need_vgpr_move : need_sgpr_move) = true;
            }
         }

         for (unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               assert(operand.tempId() != 0);
               live.insert(operand.getTemp());
               if (operand.isFixed())
                  (operand.getTemp().type() == vgpr ? need_vgpr_move : need_sgpr_move) = true;
            }
         }

         instructions.push_back(std::move(*it));

         if ((need_vgpr_move || need_sgpr_move) && !live.empty()) {
            unsigned count = 0;
            for (auto temp : live) {
               if ((need_sgpr_move && temp.type() != vgpr) ||
                   (need_vgpr_move && temp.type() == vgpr))
                  ++count;
            }

            std::unique_ptr<Instruction> move{create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, count, count)};
            int idx = 0;
            for (auto temp : live) {
               if ((need_sgpr_move && temp.type() == vgpr) ||
                   (need_vgpr_move && temp.type() != vgpr))
                  continue;
               move->getOperand(idx) = Operand{temp};
               move->getDefinition(idx) = Definition{temp};
               ++idx;
            }
            instructions.push_back(std::move(move));
         }
      }
      std::reverse(instructions.begin(), instructions.end());
      block->instructions = std::move(instructions);
   }
}


/**
 * SSA Reconstruction Pass
 * from Simple and Efficient Construction of Static Single Assignment Form
 * by M. Braun, S. Buchwald, S. Hack, R. Leißa, C. Mallon, and A. Zwinkau
 */
void fix_ssa(Program *program)
{
   struct phi_info {
      Instruction* phi;
      unsigned block_idx;
      std::set<Instruction*> uses;

   };
   bool filled[program->blocks.size()];
   bool sealed[program->blocks.size()];
   memset(filled, 0, sizeof filled);
   memset(sealed, 0, sizeof sealed);
   std::vector<std::unordered_map<unsigned, Temp>> renames(program->blocks.size());
   std::vector<std::vector<std::unique_ptr<Instruction>>> phis(program->blocks.size());
   std::vector<std::vector<std::unique_ptr<Instruction>>> incomplete_phis(program->blocks.size());
   std::map<unsigned, phi_info> phi_map;
   std::function<Temp(Temp,Block*)> read_variable;
   std::function<Temp(Temp,Block*)> read_variable_recursive;
   std::function<Temp(std::map<unsigned, phi_info>::iterator)> try_remove_trivial_phi;


   read_variable = [&](Temp val, Block* block) -> Temp {
      std::unordered_map<unsigned, Temp>::iterator it = renames[block->index].find(val.id());

      /* check if the variable got a name in the current block */
      if (it != renames[block->index].end()) {
         return it->second;
      /* if not, look up the predecessor blocks */
      } else {
         return read_variable_recursive(val, block);
      }
   };

   read_variable_recursive = [&](Temp val, Block* block) -> Temp {
      bool is_logical = val.type() == vgpr;
      std::vector<Block*>& preds = is_logical ? block->logical_predecessors : block->linear_predecessors;
      assert(preds.size() > 0);

      Temp new_val;
      if (!sealed[block->index]) {
         /* if the block is not sealed yet, we create an incomplete phi (which might later get removed again) */
         new_val = Temp{program->allocateId(), val.regClass()};
         aco_opcode opcode = is_logical ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
         std::unique_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
         phi->getDefinition(0) = Definition(new_val);
         for (unsigned i = 0; i < preds.size(); i++)
            phi->getOperand(i) = Operand(val);

         phi_map.emplace(new_val.id(), phi_info{phi.get(), block->index});
         incomplete_phis[block->index].emplace_back(std::move(phi));

      } else if (preds.size() == 1) {
         /* if the block has only one predecessor, just look there for the name */
         new_val = read_variable(val, preds[0]);
      } else {
         /* if there are more predecessors, we create a phi just in case */
         new_val = Temp{program->allocateId(), val.regClass()};
         renames[block->index][val.id()] = new_val;
         aco_opcode opcode = is_logical ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
         std::unique_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};

         phi->getDefinition(0) = Definition(new_val);
         phi_map.emplace(new_val.id(), phi_info{phi.get(), block->index});
         Instruction* instr = phi.get();

         /* we look up the name in all predecessors */
         for (unsigned i = 0; i < preds.size(); i++) {
            Temp op_temp = read_variable(val, preds[i]);
            instr->getOperand(i).setTemp(op_temp);
            if (!(op_temp == new_val) && phi_map.find(op_temp.id()) != phi_map.end())
               phi_map[op_temp.id()].uses.emplace(instr);
         }
         /* we check if the phi is trivial (in which case we return the original value) */
         new_val = try_remove_trivial_phi(phi_map.find(new_val.id()));
         // FIXME: that is quite inefficient, better keep temporaries because most phis are trivial
         phis[block->index].push_back(std::move(phi));
      }
      renames[block->index][val.id()] = new_val;
      return new_val;
   };

   try_remove_trivial_phi = [&] (std::map<unsigned, phi_info>::iterator info) -> Temp {
      assert(info->second.block_idx != 0);
      Instruction* instr = info->second.phi;
      Temp same = Temp();
      Temp def = instr->getDefinition(0).getTemp();
      /* a phi node is trivial iff all operands are the same or the definition of the phi */
      for (unsigned i = 0; i < instr->num_operands; i++) {
         Temp op = instr->getOperand(i).getTemp();
         if (op == same || op == def)
            continue;
         if (!(same == Temp())) {
            /* phi is not trivial */
            return def;
         }
         same = op;
      }
      assert(!(same == Temp()));

      /* reroute all uses to same and remove phi */
      std::vector<std::map<unsigned, phi_info>::iterator> phi_users;
      for (Instruction* instr : info->second.uses) {
         for (unsigned i = 0; i < instr->num_operands; i++) {
            if (instr->getOperand(i).isTemp() && instr->getOperand(i).tempId() == def.id())
               instr->getOperand(i).setTemp(same);
         }
         /* recursively try to remove trivial phis */
         if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi) {
            std::map<unsigned, phi_info>::iterator it = phi_map.find(instr->getDefinition(0).tempId());
            if (it != phi_map.end())
               phi_users.emplace_back(it);
         }
      }
      renames[info->second.block_idx][same.id()] = same;
      instr->num_definitions = 0; /* this indicates that the phi can be removed */
      phi_map.erase(info);
      for (auto it : phi_users)
         try_remove_trivial_phi(it);

      return same;
   };

   for (std::unique_ptr<Block>& block : program->blocks) {
      for (std::unique_ptr<Instruction>& instr : block->instructions) {
         /* this is a slight adjustment from the paper as we already have phi nodes */
         if (instr->opcode == aco_opcode::p_phi || instr->opcode == aco_opcode::p_linear_phi) {
            Definition def = instr->getDefinition(0);
            renames[block->index][def.tempId()] = def.getTemp();
            continue;
         }
         /* rename operands */
         for (unsigned i = 0; i < instr->num_operands; ++i) {
            auto& operand = instr->getOperand(i);
            if (!operand.isTemp())
               continue;
            operand.setTemp(read_variable(operand.getTemp(), block.get()));
            std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.getTemp().id());
            if (phi != phi_map.end())
               phi->second.uses.emplace(instr.get());
         }
         /* if an id is already defined, get a new id */
         for (unsigned i = 0; i < instr->num_definitions; ++i) {
            auto& definition = instr->getDefinition(i);
            if (!definition.isTemp())
               continue;
            if (renames[block->index].find(definition.tempId()) != renames[block->index].end()) {
               Temp new_temp = Temp{program->allocateId(), definition.regClass()};
               renames[block->index][definition.tempId()] = new_temp;
               definition.setTemp(new_temp);
            } else {
               renames[block->index][definition.tempId()] = definition.getTemp();
            }
         }
      }
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
            /* finish incomplete phis and check if they become trivial */
            for (std::unique_ptr<Instruction>& phi : incomplete_phis[succ->index]) {
               std::vector<Block*> preds = phi->getDefinition(0).getTemp().type() == vgpr ? succ->logical_predecessors : succ->linear_predecessors;
               for (unsigned i = 0; i < phi->num_operands; i++)
                  phi->getOperand(i).setTemp(read_variable(phi->getOperand(i).getTemp(), preds[i]));
               try_remove_trivial_phi(phi_map.find(phi->getDefinition(0).tempId()));
            }
            /* complete the original phi nodes, but no need to check triviality */
            for (std::unique_ptr<Instruction>& instr : succ->instructions) {
               if (instr->opcode != aco_opcode::p_phi && instr->opcode != aco_opcode::p_linear_phi)
                  break;
               std::vector<Block*> preds = instr->opcode == aco_opcode::p_phi ? succ->logical_predecessors : succ->linear_predecessors;

               for (unsigned i = 0; i < instr->num_operands; i++) {
                  auto& operand = instr->getOperand(i);
                  if (!operand.isTemp())
                     continue;
                  operand.setTemp(read_variable(operand.getTemp(), preds[i]));
                  std::map<unsigned, phi_info>::iterator phi = phi_map.find(operand.getTemp().id());
                  if (phi != phi_map.end())
                     phi->second.uses.emplace(instr.get());
               }
            }
            /* merge incomplete phis */
            phis[succ->index].insert(phis[succ->index].end(),
                                     std::make_move_iterator(incomplete_phis[succ->index].begin()),
                                     std::make_move_iterator(incomplete_phis[succ->index].end()));
            sealed[succ->index] = true;
         }
      }
   }

   /* merge phis with normal instructions */
   for (std::unique_ptr<Block>& block : program->blocks) {
      std::vector<std::unique_ptr<Instruction>> tmp;
      std::vector<std::unique_ptr<Instruction>>::iterator it = phis[block->index].begin();
      while (it != phis[block->index].end()) {
         if ((*it)->num_definitions != 0)
            tmp.emplace_back(std::move(*it));
         ++it;
      }
      block->instructions.insert(block->instructions.begin(),
                                 std::make_move_iterator(tmp.begin()),
                                 std::make_move_iterator(tmp.end()));
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
      case v6:
         return 6;
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
   std::vector<std::set<Temp>> live_out_per_block = live_temps_at_end_of_block(program);
   insert_copies(program, live_out_per_block);
   fix_ssa(program);

   std::unordered_map<unsigned, std::pair<PhysReg, unsigned>> assignments;
   std::unordered_map<unsigned, unsigned> temp_assignments;
   std::unordered_map<unsigned, unsigned> preferences;
   /* First assign fixed regs. */
   for(auto&& block : program->blocks) {
      for (std::vector<std::unique_ptr<Instruction>>::reverse_iterator it = block->instructions.rbegin(); it != block->instructions.rend(); ++it) {
         std::unique_ptr<Instruction>& insn = *it;
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

         /* also add some affinities */
         if (insn->opcode == aco_opcode::p_phi || insn->opcode == aco_opcode::p_linear_phi) {
            unsigned preferred = insn->getDefinition(0).tempId();
            unsigned op_idx = insn->num_operands;
            std::vector<Block*>& preds = insn->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
            for (unsigned i = 0; i < insn->num_operands; i++) {
               if (preds[i]->index < block->index && insn->getOperand(i).isTemp()) {
                  preferred = insn->getOperand(i).tempId();
                  op_idx = i;
                  break;
               }
            }
            for (unsigned i = 0; i < insn->num_operands; i++) {
               if (insn->getOperand(i).isTemp() && i != op_idx)
                  preferences.emplace(insn->getOperand(i).tempId(), preferred);
            }
            if (op_idx < insn->num_operands)
               preferences.emplace(insn->getDefinition(0).tempId(), preferred);
         }
      }
   }

   std::set<void*> kills;
   std::set<std::pair<unsigned, unsigned>> interfere;
   std::vector<Definition*> simplicial_order;

   /* Determine operands & defs which are the last use of a temp. */
   /* Also create an interference graph. */
   live_out_per_block = live_temps_at_end_of_block(program);
   for(auto b_it = program->blocks.rbegin(); b_it != program->blocks.rend(); ++b_it) {
      std::set<Temp> live = live_out_per_block[(*b_it)->index];
      for(auto i_it = (*b_it)->instructions.rbegin(); i_it != (*b_it)->instructions.rend(); ++i_it) {
         auto& insn = *i_it;

         // Add definitions that are not used to live
         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               auto it = live.find(definition.getTemp());
               if (it == live.end()) {
                  live.insert(definition.getTemp());
               }
            }
         }

         for (unsigned i = 0; i < insn->definitionCount(); ++i) {
            auto& definition = insn->getDefinition(i);
            if (definition.isTemp()) {
               simplicial_order.push_back(&definition);

               auto it = live.find(definition.getTemp());
               if (it != live.end())
                  live.erase(it);
               else
                  kills.insert(&definition);
               for(auto temp : live) {
                  if (temp.id() != definition.tempId()) {
                     interfere.insert({definition.tempId(), temp.id()});
                     interfere.insert({temp.id(), definition.tempId()});
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

         if (insn->opcode == aco_opcode::p_phi || insn->opcode == aco_opcode::p_linear_phi)
            continue;

         for(unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (operand.isTemp()) {
               auto it = live.find(operand.getTemp());
               if (it == live.end()) {
                  live.insert(operand.getTemp());
                  kills.insert(&operand);
               }
            }
         }
      }
   }

   /* Finish the assignment */
   for(auto b_it = program->blocks.begin(); b_it != program->blocks.end(); ++b_it) {
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
               unsigned end = 102;
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
                  assert(assignments.find(insn->getOperand(i).tempId()) != assignments.end());
                  unsigned preferred_reg = assignments.find(insn->getOperand(i).tempId())->second.first.reg;
                  if (can_allocate_register(&interfere, &assignments, id, preferred_reg, count)) {
                     alloc_reg = preferred_reg;
                     allocated = true;
                  }
               } else if (insn->opcode == aco_opcode::p_split_vector && count == 1) {
                  assert(assignments.find(insn->getOperand(0).tempId()) != assignments.end());
                  unsigned preferred_reg = assignments.find(insn->getOperand(0).tempId())->second.first.reg;
                  preferred_reg += i;
                  if (can_allocate_register(&interfere, &assignments, id, preferred_reg, count)) {
                     alloc_reg = preferred_reg;
                     allocated = true;
                  }
               } else if (preferences.find(id) != preferences.end()) {
                  if (assignments.find(preferences[id]) != assignments.end()) {
                     unsigned preferred_reg = assignments.find(preferences[id])->second.first.reg;
                     if (can_allocate_register(&interfere, &assignments, id, preferred_reg, count)) {
                        alloc_reg = preferred_reg;
                        allocated = true;
                     }
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
                           allocated = true;
                           break;
                        }
                     }
                     assert(allocated && "Couldn't find free register!");
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
