#include "aco_ir.h"

#include "common/sid.h"

namespace aco {

static const char *reduce_ops[] = {
   [iadd32] = "iadd32",
   [iadd64] = "iadd64",
   [imul32] = "imul32",
   [imul64] = "imul64",
   [fadd32] = "fadd32",
   [fadd64] = "fadd64",
   [fmul32] = "fmul32",
   [fmul64] = "fmul64",
   [imin32] = "imin32",
   [imin64] = "imin64",
   [imax32] = "imax32",
   [imax64] = "imax64",
   [umin32] = "umin32",
   [umin64] = "umin64",
   [umax32] = "umax32",
   [umax64] = "umax64",
   [fmin32] = "fmin32",
   [fmin64] = "fmin64",
   [fmax32] = "fmax32",
   [fmax64] = "fmax64",
   [iand32] = "iand32",
   [iand64] = "iand64",
   [ior32] = "ior32",
   [ior64] = "ior64",
   [ixor32] = "ixor32",
   [ixor64] = "ixor64",
};

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
      case v6: fprintf(output, " v6: "); return;
      case v7: fprintf(output, " v7: "); return;
      case v1_linear: fprintf(output, " v1: "); return;
      case v2_linear: fprintf(output, " v2: "); return;
   }
}

void aco_print_physReg(unsigned reg, unsigned size, FILE *output)
{
   if (reg == 124) {
      fprintf(output, ":m0");
   } else if (reg == 106) {
      fprintf(output, ":vcc");
   } else if (reg == 253) {
      fprintf(output, ":scc");
   } else if (reg == 126) {
      fprintf(output, ":exec");
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
   if (operand->isUndefined()) {
      fprintf(output, "undef");
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

static
void aco_print_instr_format_specific(struct Instruction *instr, FILE *output)
{
   switch (instr->format) {
   case Format::SOPK: {
      SOPK_instruction* sopk = static_cast<SOPK_instruction*>(instr);
      fprintf(output, " imm:%d", sopk->imm & 0x8000 ? (sopk->imm - 65536) : sopk->imm);
      break;
   }
   case Format::SOPP: {
      SOPP_instruction* sopp = static_cast<SOPP_instruction*>(instr);
      uint16_t imm = sopp->imm;
      switch (instr->opcode) {
      case aco_opcode::s_waitcnt: {
         if ((imm & 0xF) < 0xF) fprintf(output, " vmcnt(%d)", imm & 0xF);
         if (((imm >> 4) & 0x7) < 0x7) fprintf(output, " expcnt(%d)", (imm >> 4) & 0x7);
         if (((imm >> 8) & 0xF) < 0xF) fprintf(output, " lgkmcnt(%d)", (imm >> 8) & 0xF);
         break;
      }
      case aco_opcode::s_endpgm:
      case aco_opcode::s_endpgm_saved:
      case aco_opcode::s_endpgm_ordered_ps_done:
      case aco_opcode::s_wakeup:
      case aco_opcode::s_barrier:
      case aco_opcode::s_icache_inv:
      case aco_opcode::s_ttracedata:
      case aco_opcode::s_set_grp_idx_off: {
         break;
      }
      default: {
         if (imm)
            fprintf(output, " imm:%u", imm);
         break;
      }
      }
      if (sopp->block)
         fprintf(output, " block:BB%d", sopp->block->index);
      break;
   }
   case Format::SMEM: {
      SMEM_instruction* smem = static_cast<SMEM_instruction*>(instr);
      if (smem->glc)
         fprintf(output, " glc");
      if (smem->nv)
         fprintf(output, " nv");
      break;
   }
   case Format::VINTRP: {
      Interp_instruction* vintrp = static_cast<Interp_instruction*>(instr);
      fprintf(output, " attr%d.%c", vintrp->attribute, "xyzw"[vintrp->component]);
      break;
   }
   case Format::DS: {
      DS_instruction* ds = static_cast<DS_instruction*>(instr);
      if (ds->offset0)
         fprintf(output, " offset0:%u", ds->offset0);
      if (ds->offset1)
         fprintf(output, " offset1:%u", ds->offset1);
      if (ds->gds)
         fprintf(output, " gds");
      break;
   }
   case Format::MUBUF: {
      MUBUF_instruction* mubuf = static_cast<MUBUF_instruction*>(instr);
      if (mubuf->offset)
         fprintf(output, " offset:%u", mubuf->offset);
      if (mubuf->offen)
         fprintf(output, " offen");
      if (mubuf->idxen)
         fprintf(output, " idxen");
      if (mubuf->glc)
         fprintf(output, " glc");
      if (mubuf->slc)
         fprintf(output, " slc");
      if (mubuf->tfe)
         fprintf(output, " tfe");
      if (mubuf->lds)
         fprintf(output, " lds");
      if (mubuf->disable_wqm)
         fprintf(output, " disable_wqm");
      break;
   }
   case Format::MIMG: {
      MIMG_instruction* mimg = static_cast<MIMG_instruction*>(instr);
      unsigned identity_dmask = instr->num_definitions ?
                                (1 << instr->getDefinition(0).size()) - 1 :
                                0xf;
      if ((mimg->dmask & identity_dmask) != identity_dmask)
         fprintf(output, " dmask:%s%s%s%s",
                 mimg->dmask & 0x1 ? "x" : "",
                 mimg->dmask & 0x2 ? "y" : "",
                 mimg->dmask & 0x4 ? "z" : "",
                 mimg->dmask & 0x8 ? "w" : "");
      if (mimg->unrm)
         fprintf(output, " unrm");
      if (mimg->glc)
         fprintf(output, " glc");
      if (mimg->slc)
         fprintf(output, " slc");
      if (mimg->tfe)
         fprintf(output, " tfe");
      if (mimg->da)
         fprintf(output, " da");
      if (mimg->lwe)
         fprintf(output, " lwe");
      if (mimg->r128 || mimg->a16)
         fprintf(output, " r128/a16");
      if (mimg->d16)
         fprintf(output, " d16");
      if (mimg->disable_wqm)
         fprintf(output, " disable_wqm");
      break;
   }
   case Format::EXP: {
      Export_instruction* exp = static_cast<Export_instruction*>(instr);
      unsigned identity_mask = exp->compressed ? 0x5 : 0xf;
      if ((exp->enabled_mask & identity_mask) != identity_mask)
         fprintf(output, " en:%c%c%c%c",
                 exp->enabled_mask & 0x1 ? 'r' : '*',
                 exp->enabled_mask & 0x2 ? 'g' : '*',
                 exp->enabled_mask & 0x4 ? 'b' : '*',
                 exp->enabled_mask & 0x8 ? 'a' : '*');
      if (exp->compressed)
         fprintf(output, " compr");
      if (exp->done)
         fprintf(output, " done");
      if (exp->valid_mask)
         fprintf(output, " vm");

      if (exp->dest <= V_008DFC_SQ_EXP_MRT + 7)
         fprintf(output, " mrt%d", exp->dest - V_008DFC_SQ_EXP_MRT);
      else if (exp->dest == V_008DFC_SQ_EXP_MRTZ)
         fprintf(output, " mrtz");
      else if (exp->dest == V_008DFC_SQ_EXP_NULL)
         fprintf(output, " null");
      else if (exp->dest >= V_008DFC_SQ_EXP_POS && exp->dest <= V_008DFC_SQ_EXP_POS + 3)
         fprintf(output, " pos%d", exp->dest - V_008DFC_SQ_EXP_POS);
      else if (exp->dest >= V_008DFC_SQ_EXP_PARAM && exp->dest <= V_008DFC_SQ_EXP_PARAM + 31)
         fprintf(output, " param%d", exp->dest - V_008DFC_SQ_EXP_PARAM);
      break;
   }
   case Format::PSEUDO_BRANCH: {
      Pseudo_branch_instruction* branch = static_cast<Pseudo_branch_instruction*>(instr);
      fprintf(output, " BB%d", branch->targets[0]->index);
      if (branch->targets[1])
         fprintf(output, ", BB%d", branch->targets[1]->index);
      break;
   }
   case Format::PSEUDO_REDUCTION: {
      Pseudo_reduction_instruction* reduce = static_cast<Pseudo_reduction_instruction*>(instr);
      fprintf(output, " op:%s", reduce_ops[reduce->reduce_op]);
      if (reduce->cluster_size)
         fprintf(output, " cluster_size:%u", reduce->cluster_size);
      break;
   }
   case Format::FLAT:
   case Format::GLOBAL:
   case Format::SCRATCH: {
      FLAT_instruction* flat = static_cast<FLAT_instruction*>(instr);
      if (flat->offset)
         fprintf(output, " offset:%u", flat->offset);
      if (flat->glc)
         fprintf(output, " glc");
      if (flat->slc)
         fprintf(output, " slc");
      if (flat->lds)
         fprintf(output, " lds");
      if (flat->nv)
         fprintf(output, " nv");
      break;
   }
   case Format::MTBUF: {
      fprintf(output, " (printing unimplemented)");
      break;
   }
   default: {
      break;
   }
   }
   if (instr->isVOP3()) {
      VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr);
      switch (vop3->omod) {
      case 1:
         fprintf(output, " *2");
         break;
      case 2:
         fprintf(output, " *4");
         break;
      case 3:
         fprintf(output, " *0.5");
         break;
      }
      if (vop3->clamp)
         fprintf(output, " clamp");
   } else if (instr->isDPP()) {
      DPP_instruction* dpp = static_cast<DPP_instruction*>(instr);
      if (dpp->dpp_ctrl <= 0xff) {
         fprintf(output, " quad_perm:[%d,%d,%d,%d]",
                 dpp->dpp_ctrl & 0x3, (dpp->dpp_ctrl >> 2) & 0x3,
                 (dpp->dpp_ctrl >> 4) & 0x3, (dpp->dpp_ctrl >> 6) & 0x3);
      } else if (dpp->dpp_ctrl >= 0x101 && dpp->dpp_ctrl <= 0x10f) {
         fprintf(output, " row_shl:%d", dpp->dpp_ctrl & 0xf);
      } else if (dpp->dpp_ctrl >= 0x111 && dpp->dpp_ctrl <= 0x11f) {
         fprintf(output, " row_shr:%d", dpp->dpp_ctrl & 0xf);
      } else if (dpp->dpp_ctrl >= 0x121 && dpp->dpp_ctrl <= 0x12f) {
         fprintf(output, " row_ror:%d", dpp->dpp_ctrl & 0xf);
      } else if (dpp->dpp_ctrl == 0x130) {
         fprintf(output, " wave_shl:1");
      } else if (dpp->dpp_ctrl == 0x134) {
         fprintf(output, " wave_rol:1");
      } else if (dpp->dpp_ctrl == 0x138) {
         fprintf(output, " wave_shr:1");
      } else if (dpp->dpp_ctrl == 0x13c) {
         fprintf(output, " wave_ror:1");
      } else if (dpp->dpp_ctrl == 0x140) {
         fprintf(output, " row_mirror");
      } else if (dpp->dpp_ctrl == 0x141) {
         fprintf(output, " row_half_mirror");
      } else if (dpp->dpp_ctrl == 0x142) {
         fprintf(output, " row_bcast:15");
      } else if (dpp->dpp_ctrl == 0x143) {
         fprintf(output, " row_bcast:31");
      } else {
         fprintf(output, " dpp_ctrl:0x%.3x", dpp->dpp_ctrl);
      }
      if (dpp->row_mask != 0xf)
         fprintf(output, " row_mask:0x%.1x", dpp->row_mask);
      if (dpp->bank_mask != 0xf)
         fprintf(output, " bank_mask:0x%.1x", dpp->bank_mask);
      if (dpp->bound_ctrl)
         fprintf(output, " bound_ctrl:1");
   } else if ((int)instr->format & (int)Format::SDWA) {
      fprintf(output, " (printing unimplemented)");
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
      bool abs[instr->num_operands];
      bool neg[instr->num_operands];
      if ((int)instr->format & (int)Format::VOP3A) {
         VOP3A_instruction* vop3 = static_cast<VOP3A_instruction*>(instr);
         for (unsigned i = 0; i < instr->operandCount(); ++i) {
            abs[i] = vop3->abs[i];
            neg[i] = vop3->neg[i];
         }
      } else if (instr->isDPP()) {
         DPP_instruction* dpp = static_cast<DPP_instruction*>(instr);
         assert(instr->operandCount() <= 2);
         for (unsigned i = 0; i < instr->operandCount(); ++i) {
            abs[i] = dpp->abs[i];
            neg[i] = dpp->neg[i];
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
   aco_print_instr_format_specific(instr, output);
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
