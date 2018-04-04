#include <fstream>

#include "aco_ir.h"

namespace aco {

struct asm_context {
   // TODO: keep track of branch instructions referring blocks
   // and, when emitting the block, correct the offset in instr
};

void emit_instruction(asm_context ctx, std::ofstream& out, Instruction* instr)
{
   char *word = nullptr;
   std::streamsize count = 0;
   switch (instr->format())
   {
   // FIXME: casting is broken.
   // ideally, we don't want to check the number of operands or definitions
   // but only bitwise_or them, if a field can be def or op
   case Format::SOP2: {
      uint32_t encoding = (0b10 << 30);
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 23;
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 16 : 0;
      encoding |= instr->operandCount() == 2 ? instr->getOperand(1).physReg().reg << 8 : 0;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::SOPK: {
      uint32_t encoding = (0b1011 << 28);
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 23;
      encoding |=
         instr->definitionCount() && instr->getDefinition(0).regClass() != RegClass::b ?
         instr->getDefinition(0).physReg().reg << 16 :
         instr->operandCount() && instr->getOperand(0).regClass() != RegClass::b ?
         instr->getOperand(0).physReg().reg << 16 : 0;
      encoding |= static_cast<SOPK<0,0>*>(instr)->imm;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::SOP1: {
      uint32_t encoding = (0b101111101 << 23);
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 16 : 0;
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 8;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::SOPP: {
      uint32_t encoding = (0b101111111 << 23);
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 16;
      encoding |= static_cast<SOPP<0,0>*>(instr)->imm;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::VOP2: {
      uint32_t encoding = 0;
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 25;
      encoding |= instr->getDefinition(0).physReg().reg << 17;
      encoding |= instr->getOperand(1).physReg().reg << 9;
      encoding |= instr->getOperand(0).physReg().reg;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::VOPC: {
      uint32_t encoding = 0b0111110 << 25;
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 17;
      encoding |= instr->getOperand(1).physReg().reg << 9;
      encoding |= instr->getOperand(0).physReg().reg;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::VINTRP: {
      InterpInstruction<0,0>* interp = static_cast<InterpInstruction<0,0>*>(instr);
      uint32_t encoding = 0b110010 << 26;
      encoding |= instr->getDefinition(0).physReg().reg << 18;
      encoding |= opcode_infos[(int)instr->opcode()].opcode << 16;
      encoding |= interp->attribute_ << 10;
      encoding |= interp->component_ << 8;
      encoding |= instr->getOperand(0).physReg().reg;
      word = reinterpret_cast<char*>(&encoding);
      count = 4;
      break;
   }
   case Format::EXP: {
      ExportInstruction* exp = static_cast<ExportInstruction*>(instr);
      uint64_t encoding = 0b110001 << 26;
      encoding |= (uint64_t) exp->getOperand(0).physReg().reg << 32;
      encoding |= (uint64_t) exp->getOperand(1).physReg().reg << 40;
      encoding |= (uint64_t) exp->getOperand(2).physReg().reg << 48;
      encoding |= (uint64_t) exp->getOperand(3).physReg().reg << 56;
      encoding |= exp->validMask_ ? 0b1 << 12 : 0;
      encoding |= exp->done_ ? 0b1 << 11 : 0;
      encoding |= exp->compressed_ ? 0b1 << 10 : 0;
      encoding |= exp->dest_ << 4;
      encoding |= exp->enabledMask_;
      word = reinterpret_cast<char*>(&encoding);
      count = 8;
      break;
   }
   default:
      unreachable("unimplemented instruction format");
   }
   out.write(word, count);
   /* append literal dword */
   if (instr->operandCount() && instr->getOperand(0).physReg().reg == 255)
   {
      uint32_t literal = instr->getOperand(0).constantValue();
      out.write(reinterpret_cast<char*>(&literal), 4);
   }
      
}

void emit_block(asm_context ctx, std::ofstream& out, Block* block)
{
   // TODO: emit offsets on previous branches to this block
   //std::iostream::pos_type current = out.tellp();
   
   for (auto const& instr : block->instructions)
      emit_instruction(ctx, out, instr.get());
}

void emit_elf_header(asm_context ctx, std::ofstream& out, Program* program)
{
   // TODO
}

void emit_program(std::ofstream& out, Program* program)
{
   // TODO: initialize context
   asm_context ctx;
   emit_elf_header(ctx, out, program);
   for (auto const& block : program->blocks)
      emit_block(ctx, out, block.get());
   // footer?
}

}