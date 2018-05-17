#include <vector>
#include <ostream>
#include "aco_ir.h"

namespace aco {

void print_asm(std::vector<uint32_t>& binary, char* llvm_mc, std::ostream& out)
{
   char path[] = "/tmp/fileXXXXXX";
   char line[2048], tmp[128];
   FILE *p;
   int fd;

   /* Dump the binary into a temporary file. */
   fd = mkstemp(path);
   if (fd < 0)
      return;

   for (uint32_t w : binary)
   {
      sprintf(tmp, "0x%02x 0x%02x 0x%02x 0x%02x\n", (w >> 0 ) & 0xFF, (w >> 8 ) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF);
      if (write(fd, tmp, 20) == -1)
         goto fail;
   }

   sprintf(tmp, "%s --arch=amdgcn -mcpu=gfx906 -disassemble %s", llvm_mc, path);
   p = popen(tmp, "r");
   if (p) {
      while (fgets(line, sizeof(line), p))
         out << line;
      pclose(p);
   }

fail:
   close(fd);
   unlink(path);

}

}
