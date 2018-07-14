/*
 * Copyright © 2018 Google
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
 */

#include "aco_interface.h"
#include "aco_ir.h"

#include "../vulkan/radv_shader.h"

#include <iostream>
void aco_compile_shader(struct nir_shader *shader, struct ac_shader_config* config,
                        struct ac_shader_binary* binary, struct radv_shader_variant_info *info,
                        struct radv_nir_compiler_options *options)
{
   if (shader->info.stage != MESA_SHADER_FRAGMENT)
      return;

   struct ac_shader_config local_config = *config;
   struct ac_shader_binary local_binary = *binary;
   struct radv_shader_variant_info local_info = *info;
   struct radv_nir_compiler_options local_options = *options;

   if (getenv("ACO_DRY_RUN")) {
      config = &local_config;
      binary = &local_binary;
      info = &local_info;
      options = &local_options;
   }


   struct radv_shader_info pre_info = info->info;

   memset(info, 0, sizeof(*info));
   memset(config, 0, sizeof(*config));

   /* This is actually form a pass done by radv and w ewant to keep it, just need to clear all the
    * nir_to_llvm stuff. */
   info->info = pre_info;

   /* Instruction Selection */
   auto program = aco::select_program(shader, config, info, options);
   std::cerr << "After Instruction Selection:\n";
   aco_print_program(program.get(), stderr);
   aco::validate(program.get(), stderr);

   /* Optimization: doesn't work yet with control flow */
   aco::optimize(program.get());
   std::cerr << "After Opt:\n";
   aco_print_program(program.get(), stderr);
   aco::validate(program.get(), stderr);

   /* Register Allocation */
   aco::register_allocation(program.get());
   std::cerr << "After RA:\n";
   aco_print_program(program.get(), stderr);

   aco::eliminate_phis(program.get());
   /* Lower to HW Instructions */
   aco::lower_to_hw_instr(program.get());
   //std::cerr << "After Eliminate Pseudo Instr:\n";
   //aco_print_program(program.get(), stderr);

   /* Scheduling / Insert NOPs */
   aco::schedule(program.get());
   //std::cerr << "After PostRA Schedule:\n";
   //aco_print_program(program.get(), stderr);

   /* Insert Waitcnt */
   aco::insert_wait_states(program.get());
   std::cerr << "After Insert-Waitcnt:\n";
   aco_print_program(program.get(), stderr);

   /* Assembly */
   std::vector<uint32_t> code = aco::emit_program(program.get());
   std::cerr << "After Assembly:\n";
   std::cerr << "Num VGPRs: " << program->config->num_vgprs << "\n";
   std::cerr << "Num SGPRs: " << program->config->num_sgprs << "\n";
   char llvm_mc[] = "/usr/bin/llvm-mc-7";
   aco::print_asm(code, llvm_mc, std::cerr);
   //std::cerr << binary->disasm_string;
   uint32_t* bin = (uint32_t*) malloc(code.size() * sizeof(uint32_t));
   for (unsigned i = 0; i < code.size(); i++)
      bin[i] = code[i];

   binary->code = (unsigned char*) bin;
   binary->code_size = code.size() * sizeof(uint32_t);

}
