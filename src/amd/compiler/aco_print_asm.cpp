
#include <iomanip>
#include "aco_ir.h"
#include "llvm-c/Disassembler.h"
#include "ac_llvm_util.h"

#include <llvm/ADT/StringRef.h>

namespace aco {

void print_asm(Program *program, std::vector<uint32_t>& binary, enum radeon_family family, std::ostream& out)
{
   std::vector<bool> referenced_blocks(program->blocks.size());
   referenced_blocks[0] = true;
   for (Block& block : program->blocks) {
      for (unsigned succ : block.linear_succs)
         referenced_blocks[succ] = true;
   }

   std::vector<std::tuple<uint64_t, llvm::StringRef, uint8_t>> symbols;
   std::vector<std::array<char,16>> block_names;
   block_names.reserve(program->blocks.size());
   for (Block& block : program->blocks) {
      if (!referenced_blocks[block.index])
         continue;
      std::array<char, 16> name;
      sprintf(name.data(), "BB%u", block.index);
      block_names.push_back(name);
      symbols.emplace_back(block.offset * 4, llvm::StringRef(block_names[block_names.size() - 1].data()), 0);
   }

   LLVMDisasmContextRef disasm = LLVMCreateDisasmCPU("amdgcn-mesa-mesa3d",
                                                     ac_get_llvm_processor_name(family),
                                                     &symbols, 0, NULL, NULL);

   char outline[1024];
   size_t pos = 0;
   bool invalid = false;
   unsigned next_block = 0;
   while (pos < binary.size()) {
      while (next_block < program->blocks.size() && pos == program->blocks[next_block].offset) {
         if (referenced_blocks[next_block])
            out << "BB" << std::dec << next_block << ":" << std::endl;
         next_block++;
      }

      size_t l = LLVMDisasmInstruction(disasm, (uint8_t *) &binary[pos],
                                       (binary.size() - pos) * sizeof(uint32_t), pos * 4,
                                       outline, sizeof(outline));

      size_t new_pos;
      const int align_width = 60;
      if (!l) {
         out << std::left << std::setw(align_width) << std::setfill(' ') << "(invalid instruction)";
         new_pos = pos + 1;
         invalid = true;
      } else {
         out << std::left << std::setw(align_width) << std::setfill(' ') << outline;
         assert(l % 4 == 0);
         new_pos = pos + l / 4;
      }

      out << " ;";
      for (; pos < new_pos; pos++)
         out << " " << std::setfill('0') << std::setw(8) << std::hex << binary[pos];
      out << std::endl;
   }
   assert(next_block == program->blocks.size());

   LLVMDisasmDispose(disasm);

   if (invalid) {
      /* Invalid instructions usually lead to GPU hangs, which can make
       * getting the actual invalid instruction hard. Abort here so that we
       * can find the problem.
       */
      abort();
   }
}

}
