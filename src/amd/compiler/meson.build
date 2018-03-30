# Copyright © 2018 Valve Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

aco_depends = files('aco_opcodes.py')

aco_opcodes_h = custom_target(
  'aco_opcodes.h',
  input : 'aco_opcodes_h.py',
  output : 'aco_opcodes.h',
  command : [prog_python2, '@INPUT@'],
  capture : true,
  depend_files : aco_depends,
)

aco_opcodes_c = custom_target(
  'aco_opcodes.cpp',
  input : 'aco_opcodes_cpp.py',
  output : 'aco_opcodes.cpp',
  command : [prog_python2, '@INPUT@'],
  capture : true,
  depend_files : aco_depends,
)

aco_builder_instr_defs = custom_target(
  'aco_builder_instr_defs.h',
  input : 'aco_builder_instr_defs.py',
  output : 'aco_builder_instr_defs.h',
  command : [prog_python2, '@INPUT@'],
  capture : true,
  depend_files : aco_depends,
)

# Headers-only dependency
idep_aco_headers = declare_dependency(
  sources : [aco_opcodes_h],
  include_directories : include_directories('.'),
)

libaco_files = files(
  'aco_interface.cpp',
  'aco_interface.h',
  'aco_ir.h',
  'aco_builder.cpp',
)

libaco = static_library(
  'aco',
  [libaco_files, aco_opcodes_c, aco_opcodes_h, aco_builder_instr_defs],
  include_directories : [
   inc_common, inc_compiler, inc_mesa, inc_mapi, inc_amd
  ],
  dependencies : [
    dep_thread, dep_elf, dep_libdrm_amdgpu, dep_valgrind,
    idep_nir_headers,
  ],
  c_args : [c_vis_args],
  cpp_args : [cpp_vis_args],
  build_by_default : true,
)

# Also link with aco
idep_aco = declare_dependency(
  dependencies : idep_aco_headers,
  link_with : libaco,
)
