#include "aco_ir.h"

namespace aco {

static
void aco_print_operand(const Operand *operand, FILE *output)
{
   if (operand->isConstant()) {
      fprintf(output, "%x", operand->constantValue());
      return;
   }

   fprintf(output, "%%%d", operand->tempId());

   if (operand->isFixed()) {
      bool is_vgpr = operand->physReg().reg / 256;
      int reg = operand->physReg().reg  % 256;
      fprintf(output, ":%c[%d]", is_vgpr ? 'v' : 's', reg);
   }
}

static
void aco_print_definition(const Definition *definition, FILE *output)
{
   fprintf(output, "%%%d", definition->tempId());

   if (definition->isFixed()) {
      bool is_vgpr = definition->physReg().reg / 256;
      int reg = definition->physReg().reg  % 256;
      fprintf(output, ":%c[%d]", is_vgpr ? 'v' : 's', reg);
   }
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
      for (unsigned i = 0; i < instr->operandCount(); ++i) {
         if (i)
            fprintf(output, ", ");
         else
            fprintf(output, " ");

         aco_print_operand(&instr->getOperand(i), output);
       }
   }
   if (instr->opcode == aco_opcode::s_waitcnt) {
      SOPP_instruction* waitcnt = static_cast<SOPP_instruction*>(instr);
      uint16_t imm = waitcnt->imm;
      if ((imm & 0xF) < 0xF) fprintf(output, " vmcnt(%d)", imm & 0xF);
      if (((imm >> 4) & 0x7) < 0x7) fprintf(output, " expcnt(%d)", (imm >> 4) & 0x7);
      if (((imm >> 7) & 0x1F) < 0x1F) fprintf(output, " lgkmcnt(%d)", (imm >> 7) & 0x1F);
   }
}

void aco_print_program(Program *program, FILE *output)
{
   int BB = 0;
   for (auto const& block : program->blocks) {
      fprintf(output, "BB%d\n", BB);
      for (auto const& instr : block->instructions) {
         fprintf(output, "\t");
         aco_print_instr(instr.get(), output);
         fprintf(output, "\n");
      }
      ++BB;
   }
   fprintf(output, "\n");
}

}
