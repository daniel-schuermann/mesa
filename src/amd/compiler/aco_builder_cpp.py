
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

#ifndef _ACO_BUILDER_
#define _ACO_BUILDER_

#include "aco_IR.cpp"
#include "aco_builder.h"
#include "aco_opcodes.h"
namespace aco {

/* TODO: - insert instructions into CFG */
class Builder {
public:
% for name in SOP2:
<%
   Operands = ['src0']
   if name != 's_rfe_restore_b64':
      Operands.append('src1')
   if opcodes[name].num_inputs == 3:
      Operands.append('scc')
%>
static Instruction
${name}(\\
   % for op in Operands:
Operand ${op}${', ' if op != Operands[-1] else ')'}\\
   % endfor

{
   std::vector<Operand> operands = {\\
   % for op in Operands:
${op}${', ' if op != Operands[-1] else ''}\\
   % endfor
};
   std::vector<Definition> defs = {\\
   % for type in opcodes[name].output_type:
Definition(RegClass::${type})${', ' if type != opcodes[name].output_type[-1] else ''}\\
   % endfor
};
   return SOP2<${opcodes[name].num_inputs},${opcodes[name].num_outputs}>(\
aco_opcode::${name}, operands, defs);
}
% endfor
};
}
#endif /* _ACO_BUILDER_H */
"""

import aco_opcodes
from aco_opcodes import opcodes
from mako.template import Template

print Template(template).render(opcodes=opcodes, SOP2=aco_opcodes.SOP2)