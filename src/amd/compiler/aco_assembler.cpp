#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "aco_ir.h"

namespace aco {

struct asm_context {
   enum chip_class chip_class;
   std::map<int, SOPP_instruction*> branches;
   // TODO: keep track of branch instructions referring blocks
   // and, when emitting the block, correct the offset in instr
};

void emit_instruction(asm_context& ctx, std::vector<uint32_t>& out, Instruction* instr)
{
   switch (instr->format)
   {
   case Format::SOP2: {
      uint32_t encoding = (0b10 << 30);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 23;
      encoding |= instr->definitionCount() ? instr->getDefinition(0).physReg().reg << 16 : 0;
      encoding |= instr->operandCount() >= 2 ? instr->getOperand(1).physReg().reg << 8 : 0;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      out.push_back(encoding);
      break;
   }
   case Format::SOPK: {
      uint32_t encoding = (0b1011 << 28);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 23;
      encoding |=
         instr->definitionCount() && !(instr->getDefinition(0).physReg() == scc) ?
         instr->getDefinition(0).physReg().reg << 16 :
         instr->operandCount() && !(instr->getOperand(0).physReg() == scc) ?
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
   case Format::SOPC: {
      uint32_t encoding = (0b101111110 << 23);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 16;
      encoding |= instr->operandCount() == 2 ? instr->getOperand(1).physReg().reg << 8 : 0;
      encoding |= instr->operandCount() ? instr->getOperand(0).physReg().reg : 0;
      out.push_back(encoding);
      break;
   }
   case Format::SOPP: {
      SOPP_instruction* sopp = static_cast<SOPP_instruction*>(instr);
      uint32_t encoding = (0b101111111 << 23);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 16;
      encoding |= (uint16_t) sopp->imm;
      if (sopp->block)
         ctx.branches.insert({out.size(), sopp});
      out.push_back(encoding);
      break;
   }
   case Format::SMEM: {
      SMEM_instruction* smem = static_cast<SMEM_instruction*>(instr);
      uint32_t encoding = (0b110000 << 26);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 18;
      if (instr->num_operands >= 2)
         encoding |= instr->getOperand(1).isConstant() ? 1 << 17 : 0;
      bool soe = instr->num_operands >= (instr->num_definitions ? 3 : 4);
      assert(!soe || ctx.chip_class >= GFX9);
      encoding |= soe ? 1 << 14 : 0;
      encoding |= smem->glc ? 1 << 16 : 0;
      if (instr->num_definitions || instr->num_operands >= 3)
         encoding |= (instr->num_definitions ? instr->getDefinition(0).physReg().reg : instr->getOperand(2).physReg().reg) << 6;
      if (instr->num_operands >= 1)
         encoding |= instr->getOperand(0).physReg().reg >> 1;
      out.push_back(encoding);
      encoding = 0;
      if (instr->num_operands >= 2)
         encoding |= instr->getOperand(1).isConstant() ? instr->getOperand(1).constantValue() : instr->getOperand(1).physReg().reg;
      encoding |= soe ? instr->getOperand(instr->num_operands - 1).physReg().reg << 25 : 0;
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
   case Format::DS: {
      DS_instruction* ds = static_cast<DS_instruction*>(instr);
      uint32_t encoding = (0b110110 << 26);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 17;
      encoding |= (ds->gds ? 1 : 0) << 16;
      encoding |= ((0xFF & ds->offset1) << 8);
      encoding |= (0xFFFF & ds->offset0);
      out.push_back(encoding);
      encoding = 0;
      unsigned reg = instr->num_definitions ? instr->getDefinition(0).physReg().reg : 0;
      encoding |= (0xFF & reg) << 24;
      reg = instr->num_operands >= 3 && !(instr->getOperand(2).physReg() == m0)  ? instr->getOperand(2).physReg().reg : 0;
      encoding |= (0xFF & reg) << 16;
      reg = instr->num_operands >= 2 && !(instr->getOperand(1).physReg() == m0) ? instr->getOperand(1).physReg().reg : 0;
      encoding |= (0xFF & reg) << 8;
      encoding |= (0xFF & instr->getOperand(0).physReg().reg);
      out.push_back(encoding);
      break;
   }
   case Format::MUBUF: {
      MUBUF_instruction* mubuf = static_cast<MUBUF_instruction*>(instr);
      uint32_t encoding = (0b111000 << 26);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 18;
      encoding |= (mubuf->slc ? 1 : 0) << 17;
      encoding |= (mubuf->lds ? 1 : 0) << 16;
      encoding |= (mubuf->glc ? 1 : 0) << 14;
      encoding |= (mubuf->idxen ? 1 : 0) << 13;
      encoding |= (mubuf->offen ? 1 : 0) << 12;
      encoding |= 0x0FFF & mubuf->offset;
      out.push_back(encoding);
      encoding = 0;
      encoding |= instr->getOperand(2).physReg().reg << 24;
      encoding |= (mubuf->tfe ? 1 : 0) << 23;
      encoding |= (instr->getOperand(1).physReg().reg >> 2) << 16;
      unsigned reg = instr->num_operands > 3 ? instr->getOperand(3).physReg().reg : instr->getDefinition(0).physReg().reg;
      encoding |= (0xFF & reg) << 8;
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
      encoding |= mimg->r128 ? 1 << 15 : 0;
      encoding |= mimg->da ? 1 << 14 : 0;
      encoding |= mimg->glc ? 1 << 13 : 0;
      encoding |= mimg->unrm ? 1 << 12 : 0;
      encoding |= (0xF & mimg->dmask) << 8;
      out.push_back(encoding);
      encoding = (0xFF & instr->getOperand(0).physReg().reg); /* VADDR */
      if (instr->num_definitions) {
         encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 8; /* VDATA */
      } else if (instr->num_operands == 4) {
         encoding |= (0xFF & instr->getOperand(3).physReg().reg) << 8; /* VDATA */
      }
      encoding |= (0x1F & (instr->getOperand(1).physReg().reg >> 2)) << 16; /* T# (resource) */
      if (instr->num_operands > 2)
         encoding |= (0x1F & (instr->getOperand(2).physReg().reg >> 2)) << 21; /* sampler */
      // TODO VEGA: D16
      out.push_back(encoding);
      break;
   }
   case Format::FLAT:
   case Format::SCRATCH:
   case Format::GLOBAL: {
      FLAT_instruction *flat = static_cast<FLAT_instruction*>(instr);
      uint32_t encoding = (0b110111 << 26);
      encoding |= opcode_infos[(int)instr->opcode].opcode << 18;
      encoding |= flat->offset & 0x1fff;
      if (instr->format == Format::SCRATCH)
         encoding |= 1 << 14;
      else if (instr->format == Format::GLOBAL)
         encoding |= 2 << 14;
      encoding |= flat->lds ? 1 << 13 : 0;
      encoding |= flat->glc ? 1 << 13 : 0;
      encoding |= flat->slc ? 1 << 13 : 0;
      out.push_back(encoding);
      encoding = (0xFF & instr->getOperand(0).physReg().reg);
      if (instr->num_definitions)
         encoding |= (0xFF & instr->getDefinition(0).physReg().reg) << 24;
      else
         encoding |= (0xFF & instr->getOperand(2).physReg().reg) << 8;
      if (!instr->getOperand(1).isUndefined()) {
         assert(instr->getOperand(1).physReg().reg != 0x7f);
         assert(instr->format != Format::FLAT);
         encoding |= instr->getOperand(1).physReg().reg << 16;
      } else if (instr->format != Format::FLAT) {
         encoding |= 0x7F << 16;
      }
      encoding |= flat->nv ? 1 << 23 : 0;
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
   case Format::PSEUDO_BARRIER:
      return;
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

         // TODO: op_sel
         uint32_t encoding = (0b110100 << 26);
         encoding |= opcode << 16;
         encoding |= (vop3->clamp ? 1 : 0) << 15;
         for (unsigned i = 0; i < 3; i++)
            encoding |= vop3->abs[i] << (8+i);
         if (instr->num_definitions == 2)
            encoding |= instr->getDefinition(1).physReg().reg << 8;
         encoding |= (0xFF & instr->getDefinition(0).physReg().reg);
         out.push_back(encoding);
         encoding = 0;
         for (unsigned i = 0; i < instr->operandCount(); i++)
            encoding |= instr->getOperand(i).physReg().reg << (i * 9);
         encoding |= vop3->omod << 27;
         for (unsigned i = 0; i < 3; i++)
            encoding |= vop3->neg[i] << (29+i);
         out.push_back(encoding);
         return;

      } else if (instr->isDPP()){
         /* first emit the instruction without the DPP operand */
         Operand dpp_op = instr->getOperand(0);
         instr->getOperand(0) = Operand(PhysReg{250}, v1);
         instr->format = (Format) ((uint32_t) instr->format & ~(1 << 14));
         emit_instruction(ctx, out, instr);
         DPP_instruction* dpp = static_cast<DPP_instruction*>(instr);
         uint32_t encoding = (0xF & dpp->row_mask) << 28;
         encoding |= (0xF & dpp->bank_mask) << 24;
         encoding |= dpp->abs[1] << 23;
         encoding |= dpp->neg[1] << 22;
         encoding |= dpp->abs[0] << 21;
         encoding |= dpp->neg[0] << 20;
         encoding |= dpp->bound_ctrl << 19;
         encoding |= dpp->dpp_ctrl << 8;
         encoding |= (0xFF) & dpp_op.physReg().reg;
         out.push_back(encoding);
         return;
      } else {
         unreachable("unimplemented instruction format");
      }
   }

   /* append literal dword */
   for (unsigned i = 0; i < instr->num_operands; i++)
   {
      if (instr->getOperand(i).isLiteral())
      {
         uint32_t literal = instr->getOperand(i).constantValue();
         out.push_back(literal);
         break;
      }
   }
}

void emit_block(asm_context& ctx, std::vector<uint32_t>& out, Block* block)
{
   for (auto const& instr : block->instructions) {
#if 0
      int start_idx = out.size();
      std::cerr << "Encoding:\t" << std::endl;
      aco_print_instr(&*instr, stderr);
      std::cerr << std::endl;
#endif
      emit_instruction(ctx, out, instr.get());
#if 0
      for (int i = start_idx; i < out.size(); i++)
         std::cerr << "encoding: " << "0x" << std::setfill('0') << std::setw(8) << std::hex << out[i] << std::endl;
#endif
   }
}

void fix_exports(asm_context& ctx, std::vector<uint32_t>& out, Program* program)
{
   // TODO
   for (std::vector<std::unique_ptr<Block>>::reverse_iterator block_it = program->blocks.rbegin(); block_it != program->blocks.rend(); ++block_it)
   {
      Block* block = block_it->get();
      std::vector<aco_ptr<Instruction>>::reverse_iterator it = block->instructions.rbegin();
      bool endBlock = false;
      bool exported = false;
      while ( it != block->instructions.rend())
      {
         if ((*it)->format == Format::EXP && endBlock) {
            Export_instruction* exp = static_cast<Export_instruction*>((*it).get());
            exp->done = true;
            exp->valid_mask = true;
            exported = true;
            break;
         } else if ((*it)->num_definitions && (*it)->getDefinition(0).physReg() == exec)
            break;
         else if ((*it)->opcode == aco_opcode::s_endpgm) {
            if (endBlock)
               break;
            endBlock = true;
         }
         ++it;
      }
      if (!endBlock || exported)
         continue;
      /* we didn't find an Export instruction and have to insert a null export */
      aco_ptr<Export_instruction> exp{create_instruction<Export_instruction>(aco_opcode::exp, Format::EXP, 4, 0)};
      for (unsigned i = 0; i < 4; i++)
         exp->getOperand(i) = Operand();
      exp->enabled_mask = 0;
      exp->compressed = false;
      exp->done = true;
      exp->valid_mask = true;
      exp->dest = 9; /* NULL */
      /* insert the null export 1 instruction before endpgm */
      block->instructions.insert(block->instructions.end() - 1, std::move(exp));
   }
}

void fix_branches(asm_context& ctx, std::vector<uint32_t>& out)
{
   for (std::pair<int, SOPP_instruction*> branch : ctx.branches)
   {
      int offset = (int)branch.second->block->offset - branch.first - 1;
      out[branch.first] |= (uint16_t) offset;
   }
}

std::vector<uint32_t> emit_program(Program* program)
{
   asm_context ctx;
   ctx.chip_class = program->chip_class;
   std::vector<uint32_t> out;
   if (program->stage == MESA_SHADER_FRAGMENT)
      fix_exports(ctx, out, program);
   for (auto const& block : program->blocks)
   {
      block->offset = out.size();
      emit_block(ctx, out, block.get());
   }
   fix_branches(ctx, out);

   return out;
}

}
