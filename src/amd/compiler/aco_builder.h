/*
 * Copyright © 2018 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Daniel Schürmann (daniel.schuermann@campus.tu-berlin.de)
 *
 */

#include "aco_ir.h"

namespace aco {

struct dpp {
   unsigned dpp_ctrl : 9;
   bool bound_ctrl : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   unsigned bank_mask : 4;
   unsigned row_mask : 4;
};

class Builder {
public:
   Builder(Program* program, Block* current_block)
   : P(program), block(current_block)
   {}
   /**
   * Examples for Builder::instruction_factories()
   * @return 
   */
   Instruction* p_startpgm(unsigned size)
   {
      Instruction* instr = create_instruction<Instruction>(aco_opcode::p_startpgm, Format::PSEUDO, 0, size);
      insertInstruction(instr);
      return instr;
   }

   Instruction* p_parallelcopy(unsigned size)
   {
      Instruction* instr = create_instruction<Instruction>(aco_opcode::p_parallelcopy, Format::PSEUDO, size, size);
      insertInstruction(instr);
      return instr;
   }

   Instruction* p_phi(unsigned size)
   {
      Instruction* instr = create_instruction<Instruction>(aco_opcode::p_phi, Format::PSEUDO, size, 1);
      insertInstruction(instr);
      return instr;
   }

// here comes the rest of the definitions
#include "aco_builder_instr_defs.h"

#if 0
   // DPP example:
   DPP<VOP1,1,1>*
   v_mov_b32(Operand src0, dpp ctrl)
   {
      DPP<VOP1,1,1>* instr = new DPP<VOP1,1,1>(aco_opcode::v_mov_b32,
           ctrl.dpp_ctrl, ctrl.bound_ctrl,
           ctrl.src0_neg, ctrl.src0_abs,
           ctrl.src1_neg, ctrl.src1_abs,
           ctrl.bank_mask, ctrl.row_mask);
      instr->getDefinition(0) = Definition(P->allocateId(), RegClass::v1);
      instr->getOperand(0) = src0;
      insertInstruction(instr);
      return instr;
   }
#endif

private:
   Program* P;
   Block* block;
   void insertInstruction(Instruction* instr) {
      block->instructions.push_back(std::unique_ptr<Instruction>(instr));
   }
};
}
