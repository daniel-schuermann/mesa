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

#include <memory>
#include "aco_opcodes.h"
#include "aco_IR.cpp"

namespace aco {
class Builder {
public:
   Builder(Program program) : P(program) {}
   /**
   * Examples for Builder::instruction_factories()
   * @return 
   */
   PseudoInstruction* p_startpgm(unsigned size)
   {
      PseudoInstruction* instr = new PseudoInstruction(aco_opcode::p_startpgm, 0, size);
      insertInstruction(instr);
      return instr;
   }

   PseudoInstruction* p_parallelcopy(unsigned size)
   {
      PseudoInstruction* instr = new PseudoInstruction(aco_opcode::p_parallelcopy, size, size);
      insertInstruction(instr);
      return instr;
   }

   PseudoInstruction* p_phi(unsigned size)
   {
      PseudoInstruction* instr = new PseudoInstruction(aco_opcode::p_phi, size, 1);
      insertInstruction(instr);
      return instr;
   }

// here comes the rest of the definitions
#include "aco_builder_instr_defs.h"

private:
   void insertInstruction(Instruction* instr) {
      currentBlock->instructions.push_back(std::unique_ptr<Instruction>(instr));
   }
   Program& P;
   Block* currentBlock;
};
}