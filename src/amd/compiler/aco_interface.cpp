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
   if (shader->info.stage != MESA_SHADER_FRAGMENT && shader->info.stage != MESA_SHADER_COMPUTE)
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

   /* Instruction Selection */
   auto program = aco::select_program(shader, config, info, options);
   if (options->dump_preoptir) {
      std::cerr << "After Instruction Selection:\n";
      aco_print_program(program.get(), stderr);
   }
   aco::validate(program.get(), stderr);

   /* Boolean phi lowering */
   aco::lower_bool_phis(program.get());
   //std::cerr << "After Boolean Phi Lowering:\n";
   //aco_print_program(program.get(), stderr);

   aco::dominator_tree(program.get());

   /* Optimization */
   aco::value_numbering(program.get());
   aco::optimize(program.get());
   aco::validate(program.get(), stderr);

   aco::setup_reduce_temp(program.get());
   aco::insert_exec_mask(program.get());
   aco::validate(program.get(), stderr);

   aco::live live_vars = aco::live_var_analysis<true>(program.get(), options);
   aco::spill(program.get(), live_vars, options);

   //std::cerr << "Before Schedule:\n";
   //aco_print_program(program.get(), stderr);
   aco::schedule_program(program.get(), live_vars);

   /* Register Allocation */
   aco::register_allocation(program.get(), live_vars.live_out);
   if (options->dump_shader) {
      std::cerr << "After RA:\n";
      aco_print_program(program.get(), stderr);
   }

   if (aco::validate_ra(program.get(), options, stderr)) {
      std::cerr << "Program after RA validation failure:\n";
      aco_print_program(program.get(), stderr);
      abort();
   }

   aco::ssa_elimination(program.get());
   /* Lower to HW Instructions */
   aco::lower_to_hw_instr(program.get());
   //std::cerr << "After Eliminate Pseudo Instr:\n";
   //aco_print_program(program.get(), stderr);

   /* Insert Waitcnt */
   aco::insert_wait_states(program.get());
   aco::insert_NOPs(program.get());

   //std::cerr << "After Insert-Waitcnt:\n";
   //aco_print_program(program.get(), stderr);

   /* Assembly */
   std::vector<uint32_t> code = aco::emit_program(program.get());

   if (options->dump_shader) {
      std::cerr << "After Assembly:\n";
      //std::cerr << "Num VGPRs: " << program->config->num_vgprs << "\n";
      //std::cerr << "Num SGPRs: " << program->config->num_sgprs << "\n";
      aco::print_asm(program.get(), code, options->family, std::cerr);
   }
   //std::cerr << binary->disasm_string;
   uint32_t* bin = (uint32_t*) malloc(code.size() * sizeof(uint32_t));
   for (unsigned i = 0; i < code.size(); i++)
      bin[i] = code[i];

   binary->code = (unsigned char*) bin;
   binary->code_size = code.size() * sizeof(uint32_t);
   binary->disasm_string = (char*) malloc(1);
   binary->disasm_string[0] = '\0';
   binary->llvm_ir_string = (char*) malloc(1);
   binary->llvm_ir_string[0] = '\0';
}
