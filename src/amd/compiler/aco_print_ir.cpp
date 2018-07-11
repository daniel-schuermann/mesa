#include "aco_ir.h"

namespace aco {

static
void aco_print_reg_class(const RegClass rc, FILE *output)
{
   switch (rc) {
      case b: fprintf(output, "  b: "); return;
      case s1: fprintf(output, " s1: "); return;
      case s2: fprintf(output, " s2: "); return;
      case s3: fprintf(output, " s3: "); return;
      case s4: fprintf(output, " s4: "); return;
      case s8: fprintf(output, " s8: "); return;
      case s16: fprintf(output, "s16: "); return;
      case v1: fprintf(output, " v1: "); return;
      case v2: fprintf(output, " v2: "); return;
      case v3: fprintf(output, " v3: "); return;
      case v4: fprintf(output, " v4: "); return;
   }
}

void aco_print_physReg(unsigned reg, unsigned size, FILE *output)
{
   if (reg == 124) {
      fprintf(output, ":m0");
   } else if (reg == 106) {
      fprintf(output, ":vcc");
   } else {
      bool is_vgpr = reg / 256;
      reg = reg % 256;
      fprintf(output, ":%c[%d", is_vgpr ? 'v' : 's', reg);
      if (size > 1)
         fprintf(output, "-%d]", reg + size -1);
      else
         fprintf(output, "]");
   }
}

static
void aco_print_operand(const Operand *operand, FILE *output)
{
   if (operand->isConstant()) {
      fprintf(output, "0x%x", operand->constantValue());
      return;
   }

   fprintf(output, "%%%d", operand->tempId());

   if (operand->isFixed())
      aco_print_physReg(operand->physReg().reg, operand->size(), output);
}

static
void aco_print_definition(const Definition *definition, FILE *output)
{
   aco_print_reg_class(definition->regClass(), output);
   fprintf(output, "%%%d", definition->tempId());

   if (definition->isFixed())
      aco_print_physReg(definition->physReg().reg, definition->size(), output);
}

void aco_print_instr(struct Instruction *instr, FILE *output)
{
   if (instr->definitionCount()) {
      for (unsigned i = 0; i < instr->definitionCount(); ++i) {
         aco_print_definition(&instr->getDefinition(i), output);
         if (i + 1 != instr->definitionCount())
            fprintf(output, ", ");
      }
      fprintf(output, " = ");
   }
   fprintf(output, "%s", opcode_infos[(int)instr->opcode].name);
   if (instr->operandCount()) {
      bool abs[instr->num_operands];
      bool neg[instr->num_operands];
      if ((int)instr->format & (int)Format::VOP3A) {
         VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr);
         for (unsigned i = 0; i < instr->operandCount(); ++i) {
            abs[i] = vop3->abs[i];
            neg[i] = vop3->neg[i];
         }
      } else {
         for (unsigned i = 0; i < instr->operandCount(); ++i) {
            abs[i] = false;
            neg[i] = false;
         }
      }
      for (unsigned i = 0; i < instr->operandCount(); ++i) {
         if (i)
            fprintf(output, ", ");
         else
            fprintf(output, " ");

         if (neg[i])
            fprintf(output, "-");
         if (abs[i])
            fprintf(output, "|");
         aco_print_operand(&instr->getOperand(i), output);
         if (abs[i])
            fprintf(output, "|");
       }
   }
   if (instr->opcode == aco_opcode::s_waitcnt) {
      SOPP_instruction* waitcnt = static_cast<SOPP_instruction*>(instr);
      uint16_t imm = waitcnt->imm;
      if ((imm & 0xF) < 0xF) fprintf(output, " vmcnt(%d)", imm & 0xF);
      if (((imm >> 4) & 0x7) < 0x7) fprintf(output, " expcnt(%d)", (imm >> 4) & 0x7);
      if (((imm >> 7) & 0x1F) < 0x1F) fprintf(output, " lgkmcnt(%d)", (imm >> 7) & 0x1F);
   }
   if (instr->format == Format::PSEUDO_BRANCH) {
      Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(instr);
      fprintf(output, " BB%d", branch->targets[0]->index);
      if (branch->targets[1])
         fprintf(output, ", BB%d", branch->targets[1]->index);
   }
}

void aco_print_block(const struct Block* block, FILE *output)
{
   fprintf(output, "BB%d\n", block->index);
   fprintf(output, "/* logical preds: ");
   for (auto const& pred : block->logical_predecessors)
      fprintf(output, "BB%d, ", pred->index);
   fprintf(output, "/ linear preds: ");
   for (auto const& pred : block->linear_predecessors)
      fprintf(output, "BB%d, ", pred->index);
   fprintf(output, " */\n");
   for (auto const& instr : block->instructions) {
      fprintf(output, "\t");
      aco_print_instr(instr.get(), output);
      fprintf(output, "\n");
   }
}

void aco_print_program(Program *program, FILE *output)
{
   for (auto const& block : program->blocks)
      aco_print_block(block.get(), output);

   fprintf(output, "\n");
}

}
