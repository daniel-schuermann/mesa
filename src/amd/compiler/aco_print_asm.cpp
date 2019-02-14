#include <vector>
#include <ostream>
#include <iomanip>
#include "aco_ir.h"
#include "llvm-c/Disassembler.h"
#include "common/ac_llvm_util.h"

namespace aco {

void print_asm(std::vector<uint32_t>& binary, enum radeon_family family, std::ostream& out)
{
   LLVMDisasmContextRef disasm = LLVMCreateDisasmCPU("amdgcn-mesa-mesa3d",
                                                     ac_get_llvm_processor_name(family),
                                                     NULL, 0, NULL, NULL);

   char outline[1024];
   size_t pos = 0;
   bool invalid = false;
   while (pos < binary.size()) {
      size_t l = LLVMDisasmInstruction(disasm, (uint8_t *) &binary[pos],
                                       (binary.size() - pos) * sizeof(uint32_t), 0,
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
