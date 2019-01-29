
template = """\
/* 
 * Copyright (c) 2018 Valve Corporation
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
 *    Daniel Schuermann (daniel.schuermann@campus.tu-berlin.de)
 */

#include <stdbool.h>
#include "aco_ir.h"

const opcode_info opcode_infos[static_cast<int>(aco_opcode::num_opcodes)] = {
% for name, opcode in sorted(opcodes.items()):
{
   .name = "${name}",
   .opcode = ${opcode.opcode},
   .can_use_input_modifiers = ${opcode.input_mod},
   .can_use_output_modifiers = ${opcode.output_mod},
   .format = aco::Format::${str(opcode.format.name)}
},
% endfor
};

const unsigned VOPC_to_GFX6[256] = {
% for code in VOPC_GFX6:
    ${code},
% endfor
};

"""

from aco_opcodes import opcodes, VOPC_GFX6
from mako.template import Template

print(Template(template).render(opcodes=opcodes, VOPC_GFX6=VOPC_GFX6))
