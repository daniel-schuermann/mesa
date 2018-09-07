#include "aco_ir.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>

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
            if (operand.tempId() == 0) {
            aco_print_instr(insn, stderr);
               assert(operand.tempId() != 0);}
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


bool fix_ssa_block(Block* block, std::vector<std::map<unsigned, unsigned>>& renames_per_block, Program* program, std::vector<std::set<Temp>>& live_out_per_block)
{
   std::vector<std::unique_ptr<Instruction>> new_instructions;
   bool reprocess = false;

   /* compute the live-in renames which is the intersection of the predecessor's renames */
   std::map<unsigned, unsigned> renames;
   if (block->logical_predecessors.size()) {
      std::vector<Block*> preds = block->logical_predecessors;
      for (Temp temp : live_out_per_block[preds[0]->index]) {
         if (temp.type() != vgpr)
            continue;
         if (renames_per_block[preds[0]->index].find(temp.id()) == renames_per_block[preds[0]->index].end())
            continue;
         unsigned idx = renames_per_block[preds[0]->index][temp.id()];
         bool has_different_def = false;
         for (unsigned i = 1; i < preds.size(); i++) {
            if (renames_per_block[preds[i]->index].find(temp.id()) == renames_per_block[preds[i]->index].end() ||
                renames_per_block[preds[i]->index][temp.id()] != idx) {
               has_different_def = true;
               break;
            }
         }
         if (has_different_def)
            continue;
         renames.insert({temp.id(), idx});
      }
   }
   if (block->linear_predecessors.size()) {
      std::vector<Block*> preds = block->linear_predecessors;
      for (Temp temp : live_out_per_block[preds[0]->index]) {
         if (temp.type() == vgpr)
            continue;
         if (renames_per_block[preds[0]->index].find(temp.id()) == renames_per_block[preds[0]->index].end())
            continue;
         unsigned idx = renames_per_block[preds[0]->index][temp.id()];
         bool has_different_def = false;
         for (unsigned i = 1; i < preds.size(); i++) {
            if (renames_per_block[preds[i]->index].find(temp.id()) == renames_per_block[preds[i]->index].end() ||
                renames_per_block[preds[i]->index][temp.id()] != idx) {
               has_different_def = true;
               break;
            }
         }
         if (has_different_def)
            continue;
         renames.insert({temp.id(), idx});
      }
   }

   /* rename operands of each instruction */
   for (auto&& insn : block->instructions) {
      /* phi operand renames are found in the corresponding predecessor rename sets */
      if (insn->opcode == aco_opcode::p_phi || insn->opcode == aco_opcode::p_linear_phi) {
         std::vector<Block*> preds = insn->opcode == aco_opcode::p_phi ? block->logical_predecessors : block->linear_predecessors;
         for (unsigned i = 0; i < preds.size(); ++i) {
            auto& operand = insn->getOperand(i);
            std::map<unsigned, unsigned> phi_renames = renames_per_block[preds[i]->index];
            if (operand.isTemp()) {
               if (phi_renames.find(operand.tempId()) != phi_renames.end())
                  operand.setTemp(Temp{phi_renames[operand.tempId()], operand.regClass()});
               else /* if we didn't find the renamed id, we have not yet processed the predecessor block */
                  reprocess = true;
            }
         }
      } else {
         for (unsigned i = 0; i < insn->operandCount(); ++i) {
            auto& operand = insn->getOperand(i);
            if (!operand.isTemp())
               continue;
            if (renames.find(operand.tempId()) != renames.end()) {
               operand.setTemp(Temp{renames[operand.tempId()], operand.regClass()});
               continue;
            }

            /* if we didn't find the operand in the renames set, check if it has been processed in all predecessors */
            Temp temp = operand.getTemp();
            std::vector<Block*> preds = temp.type() == vgpr ? block->logical_predecessors : block->linear_predecessors;
            unsigned operand_ids[preds.size()];
            for (unsigned i = 0; i < preds.size(); i++) {
               std::map<unsigned, unsigned> pred_renames = renames_per_block[preds[i]->index];
               if (pred_renames.find(temp.id()) != pred_renames.end())
                  operand_ids[i] = pred_renames[temp.id()];
               else
                  reprocess = true;
            }
            if (reprocess) {
               /* we don't know all operand renames yet */
               renames[temp.id()] = temp.id();
               continue;
            }
            /* the operand has been renamed differently in the predecessors. we have to insert a phi-node. */
            aco_opcode opcode = temp.type() == vgpr ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
            std::unique_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
            for (unsigned i = 0; i < preds.size(); i++)
               phi->getOperand(i) = Operand({operand_ids[i], temp.regClass()});
            phi->getDefinition(0) = Definition{Temp(program->allocateId(), temp.regClass())};
            operand.setTemp(phi->getDefinition(0).getTemp());
            renames[temp.id()] = phi->getDefinition(0).tempId();
            renames[phi->getDefinition(0).tempId()] = phi->getDefinition(0).tempId();
            new_instructions.emplace_back(std::move(phi));
         }
      }
      /* rename re-definitions of the same id */
      for (unsigned i = 0; i < insn->definitionCount(); ++i) {
         auto& definition = insn->getDefinition(i);
         unsigned id = definition.tempId();
         if (renames.find(id) != renames.end()) {
            id = program->allocateId();
            renames[definition.tempId()] = id;
            definition.setTemp(Temp{id, definition.regClass()});
         }
        renames[id] = id;
      }
   }

   /* check if there are live-outs of this block, which have not yet been renamed */
   std::set<Temp>& live_out = live_out_per_block[block->index];
   for (Temp temp : live_out) {
      if (renames.find(temp.id()) != renames.end()) {
         /* if we have a new name for this live_out, we add it to live_outs */
         if (renames[temp.id()] != temp.id())
            live_out.insert({renames[temp.id()], temp.regClass()});
         continue;
      }

      /* the live-out has not yet been renamed. check, if we already processed all predecessors */
      std::vector<Block*>& preds = temp.type() == vgpr ? block->logical_predecessors : block->linear_predecessors;
      unsigned operand_ids[preds.size()];
      for (unsigned i = 0; i < preds.size(); i++) {
         std::map<unsigned, unsigned> pred_renames = renames_per_block[preds[i]->index];
         if (pred_renames.find(temp.id()) != pred_renames.end())
            operand_ids[i] = pred_renames[temp.id()];
         else
            reprocess = true;
      }
      if (reprocess) {
         /* at least one predecessor has not yet been processed, thus we have to come back */
         renames[temp.id()] = temp.id();
         continue;
      }
      /* the live-out has been renamed differently in the predecessors. we have to insert a phi-node. */
      aco_opcode opcode = temp.type() == vgpr ? aco_opcode::p_phi : aco_opcode::p_linear_phi;
      std::unique_ptr<Instruction> phi{create_instruction<Instruction>(opcode, Format::PSEUDO, preds.size(), 1)};
      for (unsigned i = 0; i < preds.size(); i++)
         phi->getOperand(i) = Operand({operand_ids[i], temp.regClass()});
      phi->getDefinition(0) = Definition{Temp(program->allocateId(), temp.regClass())};
      renames[temp.id()] = phi->getDefinition(0).tempId();
      renames[phi->getDefinition(0).tempId()] = phi->getDefinition(0).tempId();
      live_out_per_block[block->index].insert(phi->getDefinition(0).getTemp());
      new_instructions.emplace_back(std::move(phi));
   }

   renames_per_block[block->index].insert(renames.begin(), renames.end());
   if (new_instructions.size()) {
      new_instructions.insert(new_instructions.end(),
                              std::make_move_iterator(block->instructions.begin()),
                              std::make_move_iterator(block->instructions.end()));
      block->instructions.swap(new_instructions);
   }
   return reprocess;
}


/* Splits temps with multiple defs into multiple temps according to
 * SSa constraints.
 */
void fix_ssa(Program *program, std::vector<std::set<Temp>> live_out_per_block)
{
   std::vector<std::map<unsigned, unsigned>> renames_per_block(program->blocks.size());
   bool reprocess = false;
   std::deque<Block*> worklist;
   for (auto& block : program->blocks)
      reprocess |= fix_ssa_block(block.get(), renames_per_block, program, live_out_per_block);

   if (reprocess) {
      reprocess = false;
      for (auto& block : program->blocks)
         reprocess |= fix_ssa_block(block.get(), renames_per_block, program, live_out_per_block);
   }

   assert(!reprocess && "Fix ssa: this algorithm should have terminated in two interations.");
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
   fix_ssa(program, live_out_per_block);

   std::unordered_map<unsigned, std::pair<PhysReg, unsigned>> assignments;
   std::unordered_map<unsigned, unsigned> temp_assignments;
   std::unordered_map<unsigned, unsigned> preferences;
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

         if (insn->opcode == aco_opcode::p_phi || insn->opcode == aco_opcode::p_linear_phi) {
            for (unsigned i = 0; i < insn->num_operands; i++) {
               if (insn->getOperand(i).isTemp()) {
                  for (unsigned j = i + 1; j < insn->num_operands; j++)
                     preferences.emplace(insn->getOperand(j).tempId(), insn->getOperand(i).tempId());
                  preferences.emplace(insn->getDefinition(0).tempId(), insn->getOperand(i).tempId());
                  break;
               }
            }
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
               }
               if (preferences.find(id) != preferences.end()) {
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
