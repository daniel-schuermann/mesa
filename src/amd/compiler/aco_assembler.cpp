#include <vector>

#include "aco_ir.h"

namespace aco {

struct asm_context {
   // TODO: keep track of branch instructions referring blocks
   // and, when emitting the block, correct the offset in instr
};

void emit_instruction(asm_context ctx, std::vector<uint32_t>& out, Instruction* instr)
{
   switch (instr->format)
   {
   case Format::SOP2: {
      uint32_t encoding = (0b10 << 30);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 23;
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 16 : 0;
      encoding |= instr->operandCount() == 2 ? instr->getOperand(1).physReg().reg << 8 : 0;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      out.push_back(encoding);
      break;
   }
   case Format::SOPK: {
      uint32_t encoding = (0b1011 << 28);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 23;
      encoding |=
         instr->definitionCount() && instr->getDefinition(0).regClass() != RegClass::b ?
         instr->getDefinition(0).physReg().reg << 16 :
         instr->operandCount() && instr->getOperand(0).regClass() != RegClass::b ?
         instr->getOperand(0).physReg().reg << 16 : 0;
      encoding |= static_cast<SOPK_instruction*>(instr)->imm;
      out.push_back(encoding);
      break;
   }
   case Format::SOP1: {
      uint32_t encoding = (0b101111101 << 23);
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 16 : 0;
      encoding |= opcode_infos[(int)instr->opcode].opcode << 8;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      out.push_back(encoding);
      break;
   }
   case Format::SOPP: {
      uint32_t encoding = (0b101111111 << 23);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 16;
      encoding |= static_cast<SOPP_instruction*>(instr)->imm;
      out.push_back(encoding);
      break;
   }
   case Format::SMEM: {
      SMEM_instruction* smem = static_cast<SMEM_instruction*>(instr);
      uint32_t encoding = (0b110000 << 26);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 18;
      encoding |= instr->getOperand(1).isConstant() ? 1 << 17 : 0;
      encoding |= smem->glc ? 1 << 16 : 0;
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 6 : 0;
      encoding |= instr->getOperand(0).physReg().reg >> 1;
      out.push_back(encoding);
      encoding = instr->getOperand(1).isConstant() ? instr->getOperand(1).constantValue() : instr->getOperand(1).physReg().reg;
      out.push_back(encoding);
      return;
   }
   case Format::VOP2: {
      uint32_t encoding = 0;
      encoding |= opcode_infos[(int)instr->opcode].opcode << 25;
      encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 17;
      encoding |= (0xFF & instr->getOperand(1).physReg().reg) << 9;
      encoding |= instr->getOperand(0).physReg().reg;
      out.push_back(encoding);
      break;
   }
   case Format::VOP1: {
      uint32_t encoding = (0b0111111 << 25);
      encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 17;
      encoding |= opcode_infos[(int)instr->opcode].opcode << 9;
      encoding |= instr->getOperand(0).physReg().reg;
      out.push_back(encoding);
      break;
   }
   case Format::VOPC: {
      uint32_t encoding = (0b0111110 << 25);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 17;
      encoding |= (0xFF & instr->getOperand(1).physReg().reg) << 9;
      encoding |= instr->getOperand(0).physReg().reg;
      out.push_back(encoding);
      break;
   }
   case Format::VINTRP: {
      Interp_instruction* interp = static_cast<Interp_instruction*>(instr);
      uint32_t encoding = (0b110101 << 26);
      encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 18;
      encoding |= opcode_infos[(int)instr->opcode].opcode << 16;
      encoding |= interp->attribute << 10;
      encoding |= interp->component << 8;
      encoding |= (0xFF & instr->getOperand(0).physReg().reg);
      out.push_back(encoding);
      break;
   }
   case Format::MIMG: {
      MIMG_instruction* mimg = static_cast<MIMG_instruction*>(instr);
      uint32_t encoding = (0b111100 << 26);
      encoding |= mimg->slc ? 1 << 25 : 0;
      encoding |= opcode_infos[(int)instr->opcode].opcode << 18;
      encoding |= mimg->lwe ? 1 << 17 : 0;
      encoding |= mimg->tfe ? 1 << 16 : 0;
      encoding |= mimg->r128 ? 1 << 16 : 0;
      encoding |= mimg->da ? 1 << 14 : 0;
      encoding |= mimg->glc ? 1 << 13 : 0;
      encoding |= mimg->unrm ? 1 << 12 : 0;
      encoding |= (0xF & mimg->dmask) << 8;
      out.push_back(encoding);
      encoding = (0xFF & instr->getOperand(0).physReg().reg);
      encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 8;
      encoding |= (0x1F & (instr->getOperand(1).physReg().reg >> 2)) << 16;
      encoding |= instr->num_operands > 2 ? (0x1F & (instr->getOperand(2).physReg().reg >> 2)) << 21 : 0;
      // TODO VEGA: D16
      out.push_back(encoding);
      break;
   }
   case Format::EXP: {
      Export_instruction* exp = static_cast<Export_instruction*>(instr);
      uint32_t encoding = (0b110001 << 26);
      encoding |= exp->valid_mask ? 0b1 << 12 : 0;
      encoding |= exp->done ? 0b1 << 11 : 0;
      encoding |= exp->compressed ? 0b1 << 10 : 0;
      encoding |= exp->dest << 4;
      encoding |= exp->enabled_mask;
      out.push_back(encoding);
      encoding = 0xFF & exp->getOperand(0).physReg().reg;
      encoding |= (0xFF & exp->getOperand(1).physReg().reg) << 8;
      encoding |= (0xFF & exp->getOperand(2).physReg().reg) << 16;
      encoding |= (0xFF & exp->getOperand(3).physReg().reg) << 24;
      out.push_back(encoding);
      break;
   }
   case Format::PSEUDO:
      break;
   default:
      if ((uint16_t) instr->format & (uint16_t) Format::VOP3A) {
         VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr);

         uint32_t opcode;
         if ((uint16_t) instr->format & (uint16_t) Format::VOP2)
            opcode = opcode_infos[(int)instr->opcode].opcode + 0x100;
         else if ((uint16_t) instr->format & (uint16_t) Format::VOP1)
            opcode = opcode_infos[(int)instr->opcode].opcode + 0x140;
         else if ((uint16_t) instr->format & (uint16_t) Format::VOPC)
            opcode = opcode_infos[(int)instr->opcode].opcode + 0x0;
         else if ((uint16_t) instr->format & (uint16_t) Format::VINTRP)
            opcode = opcode_infos[(int)instr->opcode].opcode + 0x270;
         else
            opcode = opcode_infos[(int)instr->opcode].opcode;

         // TODO: clmp, op_sel
         uint32_t encoding = (0b110100 << 26);
         encoding |= opcode << 16;
         for (unsigned i = 0; i < 3; i++)
            encoding |= vop3->abs[i] << (8+i);
         encoding |= (0xFF & instr->getDefinition(0).physReg().reg);
         out.push_back(encoding);
         // TODO: omod
         encoding = 0;
         for (unsigned i = 0; i < instr->operandCount(); i++)
            encoding |= instr->getOperand(i).physReg().reg << (i * 9);
         for (unsigned i = 0; i < 3; i++)
            encoding |= vop3->neg[i] << (29+i);
         out.push_back(encoding);
         return;
      } else {
         unreachable("unimplemented instruction format");
      }
   }

   /* append literal dword */
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (instr->getOperand(i).isConstant() && instr->getOperand(i).physReg().reg == 255)
      {
         uint32_t literal = instr->getOperand(i).constantValue();
         out.push_back(literal);
         break;
      }
   }
}

void emit_block(asm_context ctx, std::vector<uint32_t>& out, Block* block)
{
   // TODO: emit offsets on previous branches to this block

   for (auto const& instr : block->instructions)
      emit_instruction(ctx, out, instr.get());
}

void emit_elf_header(asm_context ctx, std::vector<uint32_t>& out, Program* program)
{
   // TODO
}

std::vector<uint32_t> emit_program(Program* program)
{
   // TODO: initialize context
   asm_context ctx;
   std::vector<uint32_t> out;
   emit_elf_header(ctx, out, program);
   for (auto const& block : program->blocks)
      emit_block(ctx, out, block.get());
   // footer?
   return out;
}

}
