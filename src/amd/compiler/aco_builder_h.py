
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

% for name in SOP2:
aco_instr *
${name}(aco_builder *builder, aco_src src0\
${', aco_src src1' if name != 's_rfe_restore_b64' else ''}\
${', aco_src scc' if opcodes[name].num_inputs == 3 else ''}\
);

% endfor
#endif /* _ACO_BUILDER_H */
"""

import aco_opcodes
from aco_opcodes import opcodes
from mako.template import Template

print Template(template).render(opcodes=opcodes, SOP2=aco_opcodes.SOP2)