#
# Copyright (c) 2018 Valve Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# Authors:
#    Daniel Schuermann (daniel.schuermann@campus.tu-berlin.de)


# Class that represents all the information we have about the opcode
# NOTE: this must be kept in sync with aco_op_info

from enum import Enum

class Format(Enum):
   PSEUDO = 0
   SOP1 = 1
   SOP2 = 2
   SOPK = 3
   SOPP = 4
   SOPC = 5
   SMEM = 6
   DS = 8
   MTBUF = 9
   MUBUF = 10
   MIMG = 11
   EXP = 12
   FLAT = 13
   GLOBAL = 14
   SCRATCH = 15
   PSEUDO_BRANCH = 16
   PSEUDO_BARRIER = 17
   PSEUDO_REDUCTION = 18
   VOP1 = 1 << 8
   VOP2 = 1 << 9
   VOPC = 1 << 10
   VOP3A = 1 << 11
   VOP3B = 1 << 11
   VOP3P = 1 << 12
   VINTRP = 1 << 13
   DPP = 1 << 14
   SDWA = 1 << 15

   def get_builder_fields(self):
      if self == Format.SOPK:
         return [('uint16_t', 'imm', None)]
      elif self == Format.SOPP:
         return [('Block *', 'block', 'NULL'),
                 ('uint32_t', 'imm', '0')]
      elif self == Format.SMEM:
         return [('bool', 'can_reorder', 'true'),
                 ('bool', 'glc', 'false'),
                 ('bool', 'nv', 'false')]
      elif self == Format.DS:
         return [('int16_t', 'offset0', '0'),
                 ('int8_t', 'offset1', '0'),
                 ('bool', 'gds', 'false')]
      elif self == Format.MUBUF:
         return [('unsigned', 'offset', None),
                 ('bool', 'offen', None),
                 ('bool', 'idxen', 'false'),
                 ('bool', 'disable_wqm', 'false'),
                 ('bool', 'glc', 'false'),
                 ('bool', 'slc', 'false'),
                 ('bool', 'tfe', 'false'),
                 ('bool', 'lds', 'false')]
      elif self == Format.MIMG:
         return [('unsigned', 'dmask', '0xF'),
                 ('bool', 'da', 'false'),
                 ('bool', 'unrm', 'true'),
                 ('bool', 'disable_wqm', 'false'),
                 ('bool', 'glc', 'false'),
                 ('bool', 'slc', 'false'),
                 ('bool', 'tfe', 'false'),
                 ('bool', 'lwe', 'false'),
                 ('bool', 'r128_a16', 'false', 'r128'),
                 ('bool', 'd16', 'false')]
         return [('unsigned', 'attribute', None),
                 ('unsigned', 'component', None)]
      elif self == Format.EXP:
         return [('unsigned', 'enabled_mask', None),
                 ('unsigned', 'dest', None),
                 ('bool', 'compr', 'false', 'compressed'),
                 ('bool', 'done', 'false'),
                 ('bool', 'vm', 'false', 'valid_mask'),
                 ('bool', 'waitcnt_ignore', 'false')]
      elif self == Format.PSEUDO_BRANCH:
         return [('uint32_t', 'target0', '0', 'target[0]'),
                 ('uint32_t', 'target1', '0', 'target[1]')]
      elif self == Format.PSEUDO_REDUCTION:
         return [('ReduceOp', 'op', None, 'reduce_op'),
                 ('unsigned', 'cluster_size', '0')]
      elif self == Format.VINTRP:
         return [('unsigned', 'attribute', None),
                 ('unsigned', 'component', None)]
      elif self == Format.DPP:
         return [('uint16_t', 'dpp_ctrl', None),
                 ('uint8_t', 'row_mask', '0xF'),
                 ('uint8_t', 'bank_mask', '0xF'),
                 ('bool', 'bound_ctrl', 'false')]
      elif self in [Format.FLAT, Format.GLOBAL, Format.SCRATCH]:
         return [('uint16_t', 'offset', 0),
                 ('bool', 'glc', 'false'),
                 ('bool', 'slc', 'false'),
                 ('bool', 'lds', 'false'),
                 ('bool', 'nv', 'false')]
      else:
         return []

   def get_builder_field_names(self):
      return [f[1] for f in self.get_builder_fields()]

   def get_builder_field_dests(self):
      return [(f[3] if len(f) >= 4 else f[1]) for f in self.get_builder_fields()]

   def get_builder_field_decls(self):
      return [('%s %s=%s' % (f[0], f[1], f[2]) if f[2] != None else '%s %s' % (f[0], f[1])) for f in self.get_builder_fields()]


class Opcode(object):
   """Class that represents all the information we have about the opcode
   NOTE: this must be kept in sync with aco_op_info
   """
   def __init__(self, name, opcode_gfx9, format, input_mod, output_mod):
      """Parameters:

      - name is the name of the opcode (prepend nir_op_ for the enum name)
      - all types are strings that get nir_type_ prepended to them
      - input_types is a list of types
      - algebraic_properties is a space-seperated string, where nir_op_is_ is
        prepended before each entry
      - const_expr is an expression or series of statements that computes the
        constant value of the opcode given the constant values of its inputs.
      """
      assert isinstance(name, str)
      assert isinstance(opcode_gfx9, int)
      assert isinstance(format, Format)
      assert isinstance(input_mod, bool)
      assert isinstance(output_mod, bool)

      self.name = name
      self.opcode_gfx9 = opcode_gfx9
      self.input_mod = "1" if input_mod else "0"
      self.output_mod = "1" if output_mod else "0"
      self.format = format


# global dictionary of opcodes
opcodes = {}

# VOPC to GFX6 opcode translation map
VOPC_GFX6 = [0] * 256

def opcode(name, opcode_gfx9 = -1, format = Format.PSEUDO, input_mod = False, output_mod = False):
   assert name not in opcodes
   opcodes[name] = Opcode(name, opcode_gfx9, format, input_mod, output_mod)

opcode("exp", 0, format = Format.EXP)
opcode("p_parallelcopy")
opcode("p_startpgm")
opcode("p_phi")
opcode("p_linear_phi")
opcode("p_as_uniform")

opcode("p_create_vector")
opcode("p_extract_vector")
opcode("p_split_vector")

# start/end the parts where we can use exec based instructions
# implicitly
opcode("p_logical_start")
opcode("p_logical_end")

# e.g. subgroupMin() in SPIR-V
opcode("p_reduce", format=Format.PSEUDO_REDUCTION)
# e.g. subgroupInclusiveMin()
opcode("p_inclusive_scan", format=Format.PSEUDO_REDUCTION)
# e.g. subgroupExclusiveMin()
opcode("p_exclusive_scan", format=Format.PSEUDO_REDUCTION)

opcode("p_branch", format=Format.PSEUDO_BRANCH)
opcode("p_cbranch", format=Format.PSEUDO_BRANCH)
opcode("p_cbranch_z", format=Format.PSEUDO_BRANCH)
opcode("p_cbranch_nz", format=Format.PSEUDO_BRANCH)

opcode("p_memory_barrier_all", format=Format.PSEUDO_BARRIER)
opcode("p_memory_barrier_atomic", format=Format.PSEUDO_BARRIER)
opcode("p_memory_barrier_buffer", format=Format.PSEUDO_BARRIER)
opcode("p_memory_barrier_image", format=Format.PSEUDO_BARRIER)
opcode("p_memory_barrier_shared", format=Format.PSEUDO_BARRIER)

opcode("p_spill")
opcode("p_reload")

# start/end linear vgprs
opcode("p_start_linear_vgpr")
opcode("p_end_linear_vgpr")

opcode("p_wqm")
opcode("p_discard_if")
opcode("p_load_helper")
opcode("p_demote_to_helper")
opcode("p_is_helper")

opcode("p_fs_buffer_store_smem", format=Format.SMEM)


# SOP2 instructions: 2 scalar inputs, 1 scalar output (+optional scc)
SOP2 = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x00, 0x00, 0x00, 0x00, 0x00, "s_add_u32"),
   (0x01, 0x01, 0x01, 0x01, 0x01, "s_sub_u32"),
   (0x02, 0x02, 0x02, 0x02, 0x02, "s_add_i32"),
   (0x03, 0x03, 0x03, 0x03, 0x03, "s_sub_i32"),
   (0x04, 0x04, 0x04, 0x04, 0x04, "s_addc_u32"),
   (0x05, 0x05, 0x05, 0x05, 0x05, "s_subb_u32"),
   (0x06, 0x06, 0x06, 0x06, 0x06, "s_min_i32"),
   (0x07, 0x07, 0x07, 0x07, 0x07, "s_min_u32"),
   (0x08, 0x08, 0x08, 0x08, 0x08, "s_max_i32"),
   (0x09, 0x09, 0x09, 0x09, 0x09, "s_max_u32"),
   (0x0a, 0x0a, 0x0a, 0x0a, 0x0a, "s_cselect_b32"),
   (0x0b, 0x0b, 0x0b, 0x0b, 0x0b, "s_cselect_b64"),
   (0x0e, 0x0e, 0x0c, 0x0c, 0x0e, "s_and_b32"),
   (0x0f, 0x0f, 0x0d, 0x0d, 0x0f, "s_and_b64"),
   (0x10, 0x10, 0x0e, 0x0e, 0x10, "s_or_b32"),
   (0x11, 0x11, 0x0f, 0x0f, 0x11, "s_or_b64"),
   (0x12, 0x12, 0x10, 0x10, 0x12, "s_xor_b32"),
   (0x13, 0x13, 0x11, 0x11, 0x13, "s_xor_b64"),
   (0x14, 0x14, 0x12, 0x12, 0x14, "s_andn2_b32"),
   (0x15, 0x15, 0x13, 0x13, 0x15, "s_andn2_b64"),
   (0x16, 0x16, 0x14, 0x14, 0x16, "s_orn2_b32"),
   (0x17, 0x17, 0x15, 0x15, 0x17, "s_orn2_b64"),
   (0x18, 0x18, 0x16, 0x16, 0x18, "s_nand_b32"),
   (0x19, 0x19, 0x17, 0x17, 0x19, "s_nand_b64"),
   (0x1a, 0x1a, 0x18, 0x18, 0x1a, "s_nor_b32"),
   (0x1b, 0x1b, 0x19, 0x19, 0x1b, "s_nor_b64"),
   (0x1c, 0x1c, 0x1a, 0x1a, 0x1c, "s_xnor_b32"),
   (0x1d, 0x1d, 0x1b, 0x1b, 0x1d, "s_xnor_b64"),
   (0x1e, 0x1e, 0x1c, 0x1c, 0x1e, "s_lshl_b32"),
   (0x1f, 0x1f, 0x1d, 0x1d, 0x1f, "s_lshl_b64"),
   (0x20, 0x20, 0x1e, 0x1e, 0x20, "s_lshr_b32"),
   (0x21, 0x21, 0x1f, 0x1f, 0x21, "s_lshr_b64"),
   (0x22, 0x22, 0x20, 0x20, 0x22, "s_ashr_i32"),
   (0x23, 0x23, 0x21, 0x21, 0x23, "s_ashr_i64"),
   (0x24, 0x24, 0x22, 0x22, 0x24, "s_bfm_b32"),
   (0x25, 0x25, 0x23, 0x23, 0x25, "s_bfm_b64"),
   (0x26, 0x26, 0x24, 0x24, 0x26, "s_mul_i32"),
   (0x27, 0x27, 0x25, 0x25, 0x27, "s_bfe_u32"),
   (0x28, 0x28, 0x26, 0x26, 0x28, "s_bfe_i32"),
   (0x29, 0x29, 0x27, 0x27, 0x29, "s_bfe_u64"),
   (0x2a, 0x2a, 0x28, 0x28, 0x2a, "s_bfe_i64"),
   (0x2b, 0x2b, 0x29, 0x29,   -1, "s_cbranch_g_fork"),
   (0x2c, 0x2c, 0x2a, 0x2a, 0x2c, "s_absdiff_i32"),
   (  -1,   -1, 0x2b, 0x2b,   -1, "s_rfe_restore_b64"),
   (  -1,   -1,   -1, 0x2e, 0x2e, "s_lshl1_add_u32"),
   (  -1,   -1,   -1, 0x2f, 0x2f, "s_lshl2_add_u32"),
   (  -1,   -1,   -1, 0x30, 0x30, "s_lshl3_add_u32"),
   (  -1,   -1,   -1, 0x31, 0x31, "s_lshl4_add_u32"),
   (  -1,   -1,   -1, 0x32, 0x32, "s_pack_ll_b32_b16"),
   (  -1,   -1,   -1, 0x33, 0x33, "s_pack_lh_b32_b16"),
   (  -1,   -1,   -1, 0x34, 0x34, "s_pack_hh_b32_b16"),
   (  -1,   -1,   -1, 0x2c, 0x35, "s_mul_hi_u32"),
   (  -1,   -1,   -1, 0x2c, 0x35, "s_mul_hi_i32"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SOP2:
    opcode(name, gfx9, Format.SOP2)


# SOPK instructions: 0 input (+ imm), 1 output + optional scc
SOPK = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x00, 0x00, 0x00, 0x00, 0x00, "s_movk_i32"),
   (  -1,   -1,   -1,   -1, 0x01, "s_version"), # GFX10+
   (0x02, 0x02, 0x01, 0x01, 0x02, "s_cmovk_i32"), # GFX8_GFX9
   (0x03, 0x03, 0x02, 0x02, 0x03, "s_cmpk_eq_i32"),
   (0x04, 0x04, 0x03, 0x03, 0x04, "s_cmpk_lg_i32"),
   (0x05, 0x05, 0x04, 0x04, 0x05, "s_cmpk_gt_i32"),
   (0x06, 0x06, 0x05, 0x05, 0x06, "s_cmpk_ge_i32"),
   (0x07, 0x07, 0x06, 0x06, 0x07, "s_cmpk_lt_i32"),
   (0x08, 0x08, 0x07, 0x07, 0x08, "s_cmpk_le_i32"),
   (0x09, 0x09, 0x08, 0x08, 0x09, "s_cmpk_eq_u32"),
   (0x0a, 0x0a, 0x09, 0x09, 0x0a, "s_cmpk_lg_u32"),
   (0x0b, 0x0b, 0x0a, 0x0a, 0x0b, "s_cmpk_gt_u32"),
   (0x0c, 0x0c, 0x0b, 0x0b, 0x0c, "s_cmpk_ge_u32"),
   (0x0d, 0x0d, 0x0c, 0x0c, 0x0d, "s_cmpk_lt_u32"),
   (0x0e, 0x0e, 0x0d, 0x0d, 0x0e, "s_cmpk_le_u32"),
   (0x0f, 0x0f, 0x0e, 0x0e, 0x0f, "s_addk_i32"),
   (0x10, 0x10, 0x0f, 0x0f, 0x10, "s_mulk_i32"),
   (0x11, 0x11, 0x10, 0x10,   -1, "s_cbranch_i_fork"),
   (0x12, 0x12, 0x11, 0x11, 0x12, "s_getreg_b32"),
   (0x13, 0x13, 0x12, 0x12, 0x13, "s_setreg_b32"),
   (0x15, 0x15, 0x14, 0x14, 0x15, "s_setreg_imm32_b32"), # requires 32bit literal
   (  -1,   -1, 0x15, 0x15, 0x16, "s_call_b64"),
   (  -1,   -1,   -1,   -1, 0x17, "s_waitcnt_vscnt"),
   (  -1,   -1,   -1,   -1, 0x18, "s_waitcnt_vmcnt"),
   (  -1,   -1,   -1,   -1, 0x19, "s_waitcnt_expcnt"),
   (  -1,   -1,   -1,   -1, 0x1a, "s_waitcnt_lgkmcnt"),
   (  -1,   -1,   -1,   -1, 0x1b, "s_subvector_loop_begin"),
   (  -1,   -1,   -1,   -1, 0x1c, "s_subvector_loop_end"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SOPK:
   opcode(name, gfx9, Format.SOPK)


# SOP1 instructions: 1 input, 1 output (+optional SCC)
SOP1 = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x03, 0x03, 0x00, 0x00, 0x03, "s_mov_b32"),
   (0x04, 0x04, 0x01, 0x01, 0x04, "s_mov_b64"),
   (0x05, 0x05, 0x02, 0x02, 0x05, "s_cmov_b32"),
   (0x06, 0x06, 0x03, 0x03, 0x06, "s_cmov_b64"),
   (0x07, 0x07, 0x04, 0x04, 0x07, "s_not_b32"),
   (0x08, 0x08, 0x05, 0x05, 0x08, "s_not_b64"),
   (0x09, 0x09, 0x06, 0x06, 0x09, "s_wqm_b32"),
   (0x0a, 0x0a, 0x07, 0x07, 0x0a, "s_wqm_b64"),
   (0x0b, 0x0b, 0x08, 0x08, 0x0b, "s_brev_b32"),
   (0x0c, 0x0c, 0x09, 0x09, 0x0c, "s_brev_b64"),
   (0x0d, 0x0d, 0x0a, 0x0a, 0x0d, "s_bcnt0_i32_b32"),
   (0x0e, 0x0e, 0x0b, 0x0b, 0x0e, "s_bcnt0_i32_b64"),
   (0x0f, 0x0f, 0x0c, 0x0c, 0x0f, "s_bcnt1_i32_b32"),
   (0x10, 0x10, 0x0d, 0x0d, 0x10, "s_bcnt1_i32_b64"),
   (0x11, 0x11, 0x0e, 0x0e, 0x11, "s_ff0_i32_b32"),
   (0x12, 0x12, 0x0f, 0x0f, 0x12, "s_ff0_i32_b64"),
   (0x13, 0x13, 0x10, 0x10, 0x13, "s_ff1_i32_b32"),
   (0x14, 0x14, 0x11, 0x11, 0x14, "s_ff1_i32_b64"),
   (0x15, 0x15, 0x12, 0x12, 0x15, "s_flbit_i32_b32"),
   (0x16, 0x16, 0x13, 0x13, 0x16, "s_flbit_i32_b64"),
   (0x17, 0x17, 0x14, 0x14, 0x17, "s_flbit_i32"),
   (0x18, 0x18, 0x15, 0x15, 0x18, "s_flbit_i32_i64"),
   (0x19, 0x19, 0x16, 0x16, 0x19, "s_sext_i32_i8"),
   (0x1a, 0x1a, 0x17, 0x17, 0x1a, "s_sext_i32_i16"),
   (0x1b, 0x1b, 0x18, 0x18, 0x1b, "s_bitset0_b32"),
   (0x1c, 0x1c, 0x19, 0x19, 0x1c, "s_bitset0_b64"),
   (0x1d, 0x1d, 0x1a, 0x1a, 0x1d, "s_bitset1_b32"),
   (0x1e, 0x1e, 0x1b, 0x1b, 0x1e, "s_bitset1_b64"),
   (0x1f, 0x1f, 0x1c, 0x1c, 0x1f, "s_getpc_b64"),
   (0x20, 0x20, 0x1d, 0x1d, 0x20, "s_setpc_b64"),
   (0x21, 0x21, 0x1e, 0x1e, 0x21, "s_swappc_b64"),
   (0x22, 0x22, 0x1f, 0x1f, 0x22, "s_rfe_b64"),
   (0x24, 0x24, 0x20, 0x20, 0x24, "s_and_saveexec_b64"),
   (0x25, 0x25, 0x21, 0x21, 0x25, "s_or_saveexec_b64"),
   (0x26, 0x26, 0x22, 0x22, 0x26, "s_xor_saveexec_b64"),
   (0x27, 0x27, 0x23, 0x23, 0x27, "s_andn2_saveexec_b64"),
   (0x28, 0x28, 0x24, 0x24, 0x28, "s_orn2_saveexec_b64"),
   (0x29, 0x29, 0x25, 0x25, 0x29, "s_nand_saveexec_b64"),
   (0x2a, 0x2a, 0x26, 0x26, 0x2a, "s_nor_saveexec_b64"),
   (0x2b, 0x2b, 0x27, 0x27, 0x2b, "s_xnor_saveexec_b64"),
   (0x2c, 0x2c, 0x28, 0x28, 0x2c, "s_quadmask_b32"),
   (0x2d, 0x2d, 0x29, 0x29, 0x2d, "s_quadmask_b64"),
   (0x2e, 0x2e, 0x2a, 0x2a, 0x2e, "s_movrels_b32"),
   (0x2f, 0x2f, 0x2b, 0x2b, 0x2f, "s_movrels_b64"),
   (0x30, 0x30, 0x2c, 0x2c, 0x30, "s_movreld_b32"),
   (0x31, 0x31, 0x2d, 0x2d, 0x31, "s_movreld_b64"),
   (0x32, 0x32, 0x2e, 0x2e,   -1, "s_cbranch_join"),
   (0x34, 0x34, 0x30, 0x30, 0x34, "s_abs_i32"),
   (0x35, 0x35,   -1,   -1, 0x35, "s_mov_fed_b32"),
   (  -1,   -1, 0x32, 0x32,   -1, "s_set_gpr_idx_idx"),
   (  -1,   -1,   -1, 0x33, 0x37, "s_andn1_saveexec_b64"),
   (  -1,   -1,   -1, 0x34, 0x38, "s_orn1_saveexec_b64"),
   (  -1,   -1,   -1, 0x35, 0x39, "s_andn1_wrexec_b64"),
   (  -1,   -1,   -1, 0x36, 0x3a, "s_andn2_wrexec_b64"),
   (  -1,   -1,   -1, 0x37, 0x3b, "s_bitreplicate_b64_b32"),
   (  -1,   -1,   -1,   -1, 0x3c, "s_and_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x3d, "s_or_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x3e, "s_xor_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x3f, "s_andn2_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x40, "s_orn2_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x41, "s_nand_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x42, "s_nor_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x43, "s_xnor_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x44, "s_andn1_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x45, "s_orn1_saveexec_b32"),
   (  -1,   -1,   -1,   -1, 0x46, "s_andn1_wrexec_b32"),
   (  -1,   -1,   -1,   -1, 0x47, "s_andn2_wrexec_b32"),
   (  -1,   -1,   -1,   -1, 0x49, "s_movrelsd_2_b32"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SOP1:
   opcode(name, gfx9, Format.SOP1)


# SOPC instructions: 2 inputs and 0 outputs (+SCC)
SOPC = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x00, 0x00, 0x00, 0x00, 0x00, "s_cmp_eq_i32"),
   (0x01, 0x01, 0x01, 0x01, 0x01, "s_cmp_lg_i32"),
   (0x02, 0x02, 0x02, 0x02, 0x02, "s_cmp_gt_i32"),
   (0x03, 0x03, 0x03, 0x03, 0x03, "s_cmp_ge_i32"),
   (0x04, 0x04, 0x04, 0x04, 0x04, "s_cmp_lt_i32"),
   (0x05, 0x05, 0x05, 0x05, 0x05, "s_cmp_le_i32"),
   (0x06, 0x06, 0x06, 0x06, 0x06, "s_cmp_eq_u32"),
   (0x07, 0x07, 0x07, 0x07, 0x07, "s_cmp_lg_u32"),
   (0x08, 0x08, 0x08, 0x08, 0x08, "s_cmp_gt_u32"),
   (0x09, 0x09, 0x09, 0x09, 0x09, "s_cmp_ge_u32"),
   (0x0a, 0x0a, 0x0a, 0x0a, 0x0a, "s_cmp_lt_u32"),
   (0x0b, 0x0b, 0x0b, 0x0b, 0x0b, "s_cmp_le_u32"),
   (0x0c, 0x0c, 0x0c, 0x0c, 0x0c, "s_bitcmp0_b32"),
   (0x0d, 0x0d, 0x0d, 0x0d, 0x0d, "s_bitcmp1_b32"),
   (0x0e, 0x0e, 0x0e, 0x0e, 0x0e, "s_bitcmp0_b64"),
   (0x0f, 0x0f, 0x0f, 0x0f, 0x0f, "s_bitcmp1_b64"),
   (0x10, 0x10, 0x10, 0x10,   -1, "s_setvskip"),
   (  -1,   -1, 0x11, 0x11,   -1, "s_set_gpr_idx_on"),
   (  -1,   -1, 0x12, 0x12, 0x12, "s_cmp_eq_u64"),
   (  -1,   -1, 0x13, 0x13, 0x13, "s_cmp_lg_u64"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SOPC:
   opcode(name, gfx9, Format.SOPC)


# SOPP instructions: 0 inputs (+optional scc/vcc), 0 outputs
SOPP = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x00, 0x00, 0x00, 0x00, 0x00, "s_nop"),
   (0x01, 0x01, 0x01, 0x01, 0x01, "s_endpgm"),
   (0x02, 0x02, 0x02, 0x02, 0x02, "s_branch"),
   (  -1,   -1, 0x03, 0x03, 0x03, "s_wakeup"),
   (0x04, 0x04, 0x04, 0x04, 0x04, "s_cbranch_scc0"),
   (0x05, 0x05, 0x05, 0x05, 0x05, "s_cbranch_scc1"),
   (0x06, 0x06, 0x06, 0x06, 0x06, "s_cbranch_vccz"),
   (0x07, 0x07, 0x07, 0x07, 0x07, "s_cbranch_vccnz"),
   (0x08, 0x08, 0x08, 0x08, 0x08, "s_cbranch_execz"),
   (0x09, 0x09, 0x09, 0x09, 0x09, "s_cbranch_execnz"),
   (0x0a, 0x0a, 0x0a, 0x0a, 0x0a, "s_barrier"),
   (  -1, 0x0b, 0x0b, 0x0b, 0x0b, "s_setkill"),
   (0x0c, 0x0c, 0x0c, 0x0c, 0x0c, "s_waitcnt"),
   (0x0d, 0x0d, 0x0d, 0x0d, 0x0d, "s_sethalt"),
   (0x0e, 0x0e, 0x0e, 0x0e, 0x0e, "s_sleep"),
   (0x0f, 0x0f, 0x0f, 0x0f, 0x0f, "s_setprio"),
   (0x10, 0x10, 0x10, 0x10, 0x10, "s_sendmsg"),
   (0x11, 0x11, 0x11, 0x11, 0x11, "s_sendmsghalt"),
   (0x12, 0x12, 0x12, 0x12, 0x12, "s_trap"),
   (0x13, 0x13, 0x13, 0x13, 0x13, "s_icache_inv"),
   (0x14, 0x14, 0x14, 0x14, 0x14, "s_incperflevel"),
   (0x15, 0x15, 0x15, 0x15, 0x15, "s_decperflevel"),
   (0x16, 0x16, 0x16, 0x16, 0x16, "s_ttracedata"),
   (  -1, 0x17, 0x17, 0x17, 0x17, "s_cbranch_cdbgsys"),
   (  -1, 0x18, 0x18, 0x18, 0x18, "s_cbranch_cdbguser"),
   (  -1, 0x19, 0x19, 0x19, 0x19, "s_cbranch_cdbgsys_or_user"),
   (  -1, 0x1a, 0x1a, 0x1a, 0x1a, "s_cbranch_cdbgsys_and_user"),
   (  -1,   -1, 0x1b, 0x1b, 0x1b, "s_endpgm_saved"),
   (  -1,   -1, 0x1c, 0x1c,   -1, "s_set_gpr_idx_off"),
   (  -1,   -1, 0x1d, 0x1d,   -1, "s_set_gpr_idx_mode"),
   (  -1,   -1,   -1, 0x1e, 0x1e, "s_endpgm_ordered_ps_done"),
   (  -1,   -1,   -1,   -1, 0x1f, "s_code_end"),
   (  -1,   -1,   -1,   -1, 0x20, "s_inst_prefetch"),
   (  -1,   -1,   -1,   -1, 0x21, "s_clause"),
   (  -1,   -1,   -1,   -1, 0x22, "s_wait_idle"),
   (  -1,   -1,   -1,   -1, 0x23, "s_waitcnt_depctr"),
   (  -1,   -1,   -1,   -1, 0x24, "s_round_mode"),
   (  -1,   -1,   -1,   -1, 0x25, "s_denorm_mode"),
   (  -1,   -1,   -1,   -1, 0x26, "s_ttracedata_imm"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SOPP:
   opcode(name, gfx9, Format.SOPP)


# SMEM instructions: sbase input (2 sgpr), potentially 2 offset inputs, 1 sdata input/output
SMEM = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name
   (0x00, 0x00, 0x00, 0x00, 0x00, "s_load_dword"),
   (0x01, 0x01, 0x01, 0x01, 0x01, "s_load_dwordx2"),
   (0x02, 0x02, 0x02, 0x02, 0x02, "s_load_dwordx4"),
   (0x03, 0x03, 0x03, 0x03, 0x03, "s_load_dwordx8"),
   (0x04, 0x04, 0x04, 0x04, 0x04, "s_load_dwordx16"),
   (  -1,   -1,   -1, 0x05, 0x05, "s_scratch_load_dword"),
   (  -1,   -1,   -1, 0x06, 0x06, "s_scratch_load_dwordx2"),
   (  -1,   -1,   -1, 0x07, 0x07, "s_scratch_load_dwordx4"),
   (0x08, 0x08, 0x08, 0x08, 0x08, "s_buffer_load_dword"),
   (0x09, 0x09, 0x09, 0x09, 0x09, "s_buffer_load_dwordx2"),
   (0x0a, 0x0a, 0x0a, 0x0a, 0x0a, "s_buffer_load_dwordx4"),
   (0x0b, 0x0b, 0x0b, 0x0b, 0x0b, "s_buffer_load_dwordx8"),
   (0x0c, 0x0c, 0x0c, 0x0c, 0x0c, "s_buffer_load_dwordx16"),
   (  -1,   -1, 0x10, 0x10, 0x10, "s_store_dword"),
   (  -1,   -1, 0x11, 0x11, 0x11, "s_store_dwordx2"),
   (  -1,   -1, 0x12, 0x12, 0x12, "s_store_dwordx4"),
   (  -1,   -1,   -1, 0x15, 0x15, "s_scratch_store_dword"),
   (  -1,   -1,   -1, 0x16, 0x16, "s_scratch_store_dwordx2"),
   (  -1,   -1,   -1, 0x17, 0x17, "s_scratch_store_dwordx4"),
   (  -1,   -1, 0x18, 0x18, 0x18, "s_buffer_store_dword"),
   (  -1,   -1, 0x19, 0x19, 0x19, "s_buffer_store_dwordx2"),
   (  -1,   -1, 0x1a, 0x1a, 0x1a, "s_buffer_store_dwordx4"),
   (  -1,   -1, 0x1f, 0x1f, 0x1f, "s_gl1_inv"),
   (0x1f, 0x1f, 0x20, 0x20, 0x20, "s_dcache_inv"),
   (  -1,   -1, 0x21, 0x21, 0x21, "s_dcache_wb"),
   (  -1, 0x1d, 0x22, 0x22,   -1, "s_dcache_inv_vol"),
   (  -1,   -1, 0x23, 0x23, 0x23, "s_dcache_wb_vol"),
   (0x1e, 0x1e, 0x24, 0x24, 0x24, "s_memtime"),
   (  -1,   -1, 0x25, 0x25, 0x25, "s_memrealtime"),
   (  -1,   -1, 0x26, 0x26, 0x26, "s_atc_probe"),
   (  -1,   -1, 0x27, 0x27, 0x27, "s_atc_probe_buffer"),
   (  -1,   -1,   -1, 0x28, 0x28, "s_dcache_discard"),
   (  -1,   -1,   -1, 0x29, 0x29, "s_dcache_discard_x2"),
   (  -1,   -1,   -1,   -1, 0x2a, "s_get_waveid_in_workgroup"),
   (  -1,   -1,   -1, 0x40, 0x40, "s_buffer_atomic_swap"),
   (  -1,   -1,   -1, 0x41, 0x41, "s_buffer_atomic_cmpswap"),
   (  -1,   -1,   -1, 0x42, 0x42, "s_buffer_atomic_add"),
   (  -1,   -1,   -1, 0x43, 0x43, "s_buffer_atomic_sub"),
   (  -1,   -1,   -1, 0x44, 0x44, "s_buffer_atomic_smin"),
   (  -1,   -1,   -1, 0x45, 0x45, "s_buffer_atomic_umin"),
   (  -1,   -1,   -1, 0x46, 0x46, "s_buffer_atomic_smax"),
   (  -1,   -1,   -1, 0x47, 0x47, "s_buffer_atomic_umax"),
   (  -1,   -1,   -1, 0x48, 0x48, "s_buffer_atomic_and"),
   (  -1,   -1,   -1, 0x49, 0x49, "s_buffer_atomic_or"),
   (  -1,   -1,   -1, 0x4a, 0x4a, "s_buffer_atomic_xor"),
   (  -1,   -1,   -1, 0x4b, 0x4b, "s_buffer_atomic_inc"),
   (  -1,   -1,   -1, 0x4c, 0x4c, "s_buffer_atomic_dec"),
   (  -1,   -1,   -1, 0x60, 0x60, "s_buffer_atomic_swap_x2"),
   (  -1,   -1,   -1, 0x61, 0x61, "s_buffer_atomic_cmpswap_x2"),
   (  -1,   -1,   -1, 0x62, 0x62, "s_buffer_atomic_add_x2"),
   (  -1,   -1,   -1, 0x63, 0x63, "s_buffer_atomic_sub_x2"),
   (  -1,   -1,   -1, 0x64, 0x64, "s_buffer_atomic_smin_x2"),
   (  -1,   -1,   -1, 0x65, 0x65, "s_buffer_atomic_umin_x2"),
   (  -1,   -1,   -1, 0x66, 0x66, "s_buffer_atomic_smax_x2"),
   (  -1,   -1,   -1, 0x67, 0x67, "s_buffer_atomic_umax_x2"),
   (  -1,   -1,   -1, 0x68, 0x68, "s_buffer_atomic_and_x2"),
   (  -1,   -1,   -1, 0x69, 0x69, "s_buffer_atomic_or_x2"),
   (  -1,   -1,   -1, 0x6a, 0x6a, "s_buffer_atomic_xor_x2"),
   (  -1,   -1,   -1, 0x6b, 0x6b, "s_buffer_atomic_inc_x2"),
   (  -1,   -1,   -1, 0x6c, 0x6c, "s_buffer_atomic_dec_x2"),
   (  -1,   -1,   -1, 0x80, 0x80, "s_atomic_swap"),
   (  -1,   -1,   -1, 0x81, 0x81, "s_atomic_cmpswap"),
   (  -1,   -1,   -1, 0x82, 0x82, "s_atomic_add"),
   (  -1,   -1,   -1, 0x83, 0x83, "s_atomic_sub"),
   (  -1,   -1,   -1, 0x84, 0x84, "s_atomic_smin"),
   (  -1,   -1,   -1, 0x85, 0x85, "s_atomic_umin"),
   (  -1,   -1,   -1, 0x86, 0x86, "s_atomic_smax"),
   (  -1,   -1,   -1, 0x87, 0x87, "s_atomic_umax"),
   (  -1,   -1,   -1, 0x88, 0x88, "s_atomic_and"),
   (  -1,   -1,   -1, 0x89, 0x89, "s_atomic_or"),
   (  -1,   -1,   -1, 0x8a, 0x8a, "s_atomic_xor"),
   (  -1,   -1,   -1, 0x8b, 0x8b, "s_atomic_inc"),
   (  -1,   -1,   -1, 0x8c, 0x8c, "s_atomic_dec"),
   (  -1,   -1,   -1, 0xa0, 0xa0, "s_atomic_swap_x2"),
   (  -1,   -1,   -1, 0xa1, 0xa1, "s_atomic_cmpswap_x2"),
   (  -1,   -1,   -1, 0xa2, 0xa2, "s_atomic_add_x2"),
   (  -1,   -1,   -1, 0xa3, 0xa3, "s_atomic_sub_x2"),
   (  -1,   -1,   -1, 0xa4, 0xa4, "s_atomic_smin_x2"),
   (  -1,   -1,   -1, 0xa5, 0xa5, "s_atomic_umin_x2"),
   (  -1,   -1,   -1, 0xa6, 0xa6, "s_atomic_smax_x2"),
   (  -1,   -1,   -1, 0xa7, 0xa7, "s_atomic_umax_x2"),
   (  -1,   -1,   -1, 0xa8, 0xa8, "s_atomic_and_x2"),
   (  -1,   -1,   -1, 0xa9, 0xa9, "s_atomic_or_x2"),
   (  -1,   -1,   -1, 0xaa, 0xaa, "s_atomic_xor_x2"),
   (  -1,   -1,   -1, 0xab, 0xab, "s_atomic_inc_x2"),
   (  -1,   -1,   -1, 0xac, 0xac, "s_atomic_dec_x2"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in SMEM:
   opcode(name, gfx9, Format.SMEM)


# VOP2 instructions: 2 inputs, 1 output (+ optional vcc)
# TODO: misses some GFX6_7 opcodes which were shifted to VOP3 in GFX8
VOP2 = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name, input/output modifiers
   (0x00, 0x00, 0x00, 0x00, 0x01, "v_cndmask_b32", False),
   (0x03, 0x03, 0x01, 0x01, 0x03, "v_add_f32", True),
   (0x04, 0x04, 0x02, 0x02, 0x04, "v_sub_f32", True),
   (0x05, 0x05, 0x03, 0x03, 0x05, "v_subrev_f32", True),
   (0x06, 0x06,   -1,   -1, 0x06, "v_mac_legacy_f32", True),
   (0x07, 0x07, 0x04, 0x04, 0x07, "v_mul_legacy_f32", True),
   (0x08, 0x08, 0x05, 0x05, 0x08, "v_mul_f32", True),
   (0x09, 0x09, 0x06, 0x06, 0x09, "v_mul_i32_i24", False),
   (0x0a, 0x0a, 0x07, 0x07, 0x0a, "v_mul_hi_i32_i24", False),
   (0x0b, 0x0b, 0x08, 0x08, 0x0b, "v_mul_u32_u24", False),
   (0x0c, 0x0c, 0x09, 0x09, 0x0c, "v_mul_hi_u32_u24", False),
   (0x0d, 0x0d,   -1,   -1,   -1, "v_min_legacy_f32", True),
   (0x0e, 0x0e,   -1,   -1,   -1, "v_max_legacy_f32", True),
   (0x0f, 0x0f, 0x0a, 0x0a, 0x0f, "v_min_f32", True),
   (0x10, 0x10, 0x0b, 0x0b, 0x10, "v_max_f32", True),
   (0x11, 0x11, 0x0c, 0x0c, 0x11, "v_min_i32", False),
   (0x12, 0x12, 0x0d, 0x0d, 0x12, "v_max_i32", False),
   (0x13, 0x13, 0x0e, 0x0e, 0x13, "v_min_u32", False),
   (0x14, 0x14, 0x0f, 0x0f, 0x14, "v_max_u32", False),
   (0x15, 0x15,   -1,   -1, 0x15, "v_lshr_b32", False),
   (0x16, 0x16, 0x10, 0x10, 0x16, "v_lshrrev_b32", False),
   (0x17, 0x17,   -1,   -1, 0x17, "v_ashr_i32", False),
   (0x18, 0x18, 0x11, 0x11, 0x18, "v_ashrrev_i32", False),
   (0x19, 0x19,   -1,   -1, 0x19, "v_lshl_b32", False),
   (0x1a, 0x1a, 0x12, 0x12, 0x1a, "v_lshlrev_b32", False),
   (0x1b, 0x1b, 0x13, 0x13, 0x1b, "v_and_b32", False),
   (0x1c, 0x1c, 0x14, 0x14, 0x1c, "v_or_b32", False),
   (0x1d, 0x1d, 0x15, 0x15, 0x1d, "v_xor_b32", False),
   (  -1,   -1,   -1,   -1, 0x1e, "v_xnor_b32", False),
   (0x1f, 0x1f, 0x16, 0x16, 0x1f, "v_mac_f32", True),
   (0x20, 0x20, 0x17, 0x17, 0x20, "v_madmk_f32", False),
   (0x21, 0x21, 0x18, 0x18, 0x21, "v_madak_f32", False),
   (0x25, 0x25, 0x19, 0x19, 0x25, "v_add_co_u32", False),
   (0x26, 0x26, 0x1a, 0x1a, 0x26, "v_sub_co_u32", False),
   (0x27, 0x27, 0x1b, 0x1b, 0x27, "v_subrev_co_u32", False),
   (0x28, 0x28, 0x1c, 0x1c, 0x28, "v_addc_co_u32", False),
   (0x29, 0x29, 0x1d, 0x1d, 0x29, "v_subb_co_u32", False),
   (0x2a, 0x2a, 0x1e, 0x1e, 0x2a, "v_subbrev_co_u32", False),
   (  -1,   -1,   -1,   -1, 0x2b, "v_fmac_f32", True),
   (  -1,   -1,   -1,   -1, 0x2c, "v_fmamk_f32", True),
   (  -1,   -1,   -1,   -1, 0x2d, "v_fmaak_f32", True),
   (  -1,   -1, 0x1f, 0x1f, 0x32, "v_add_f16", True),
   (  -1,   -1, 0x20, 0x20, 0x33, "v_sub_f16", True),
   (  -1,   -1, 0x21, 0x21, 0x34, "v_subrev_f16", True),
   (  -1,   -1, 0x22, 0x22, 0x35, "v_mul_f16", True),
   (  -1,   -1, 0x23, 0x23,   -1, "v_mac_f16", True),
   (  -1,   -1, 0x24, 0x24,   -1, "v_madmk_f16", False),
   (  -1,   -1, 0x25, 0x25,   -1, "v_madak_f16", False),
   (  -1,   -1, 0x26, 0x26,   -1, "v_add_u16", False),
   (  -1,   -1, 0x27, 0x27,   -1, "v_sub_u16", False),
   (  -1,   -1, 0x28, 0x28,   -1, "v_subrev_u16", False),
   (  -1,   -1, 0x29, 0x29,   -1, "v_mul_lo_u16", False),
   (  -1,   -1, 0x2a, 0x2a,   -1, "v_lshlrev_b16", False),
   (  -1,   -1, 0x2b, 0x2b,   -1, "v_lshrrev_b16", False),
   (  -1,   -1, 0x2c, 0x2c,   -1, "v_ashrrev_b16", False),
   (  -1,   -1, 0x2d, 0x2d, 0x39, "v_max_f16", True),
   (  -1,   -1, 0x2e, 0x2e, 0x3a, "v_min_f16", True),
   (  -1,   -1, 0x2f, 0x2f,   -1, "v_max_u16", False),
   (  -1,   -1, 0x30, 0x30,   -1, "v_max_i16", False),
   (  -1,   -1, 0x31, 0x31,   -1, "v_min_u16", False),
   (  -1,   -1, 0x32, 0x32,   -1, "v_min_i16", False),
   (  -1,   -1, 0x33, 0x33, 0x3b, "v_ldexp_f16", False),
   (  -1,   -1, 0x34, 0x34,   -1, "v_add_u32", False),
   (  -1,   -1, 0x35, 0x35,   -1, "v_sub_u32", False),
   (  -1,   -1, 0x36, 0x36,   -1, "v_subrev_u32", False),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name, modifiers) in VOP2:
   opcode(name, gfx9, Format.VOP2, modifiers, modifiers)


# VOP1 instructions: instructions with 1 input and 1 output
VOP1 = {
  # GFX6, GFX7, GFX8, GFX9, GFX10, name, input_modifiers, output_modifiers
   (0x00, 0x00, 0x00, 0x00, 0x00, "v_nop", False, False),
   (0x01, 0x01, 0x01, 0x01, 0x01, "v_mov_b32", False, False),
   (0x02, 0x02, 0x02, 0x02, 0x02, "v_readfirstlane_b32", False, False),
   (0x03, 0x03, 0x03, 0x03, 0x03, "v_cvt_i32_f64", True, False),
   (0x04, 0x04, 0x04, 0x04, 0x04, "v_cvt_f64_i32", False, True),
   (0x05, 0x05, 0x05, 0x05, 0x05, "v_cvt_f32_i32", False, True),
   (0x06, 0x06, 0x06, 0x06, 0x06, "v_cvt_f32_u32", False, True),
   (0x07, 0x07, 0x07, 0x07, 0x07, "v_cvt_u32_f32", True, False),
   (0x08, 0x08, 0x08, 0x08, 0x08, "v_cvt_i32_f32", True, False),
   (0x09, 0x09,   -1,   -1, 0x09, "v_mov_fed_b32", True, False), # LLVM mentions it for GFX8_9
   (0x0a, 0x0a, 0x0a, 0x0a, 0x0a, "v_cvt_f16_f32", True, True),
   (0x0b, 0x0b, 0x0b, 0x0b, 0x0b, "v_cvt_f32_f16", True, True),
   (0x0c, 0x0c, 0x0c, 0x0c, 0x0c, "v_cvt_rpi_i32_f32", True, False),
   (0x0d, 0x0d, 0x0d, 0x0d, 0x0d, "v_cvt_flr_i32_f32", True, False),
   (0x0e, 0x0e, 0x0e, 0x0e, 0x0e, "v_cvt_off_f32_i4", False, True),
   (0x0f, 0x0f, 0x0f, 0x0f, 0x0f, "v_cvt_f32_f64", True, True),
   (0x10, 0x10, 0x10, 0x10, 0x10, "v_cvt_f64_f32", True, True),
   (0x11, 0x11, 0x11, 0x11, 0x11, "v_cvt_f32_ubyte0", False, True),
   (0x12, 0x12, 0x12, 0x12, 0x12, "v_cvt_f32_ubyte1", False, True),
   (0x13, 0x13, 0x13, 0x13, 0x13, "v_cvt_f32_ubyte2", False, True),
   (0x14, 0x14, 0x14, 0x14, 0x14, "v_cvt_f32_ubyte3", False, True),
   (0x15, 0x15, 0x15, 0x15, 0x15, "v_cvt_u32_f64", True, False),
   (0x16, 0x16, 0x16, 0x16, 0x16, "v_cvt_f64_u32", False, True),
   (  -1, 0x17, 0x17, 0x17, 0x17, "v_trunc_f64", True, True),
   (  -1, 0x18, 0x18, 0x18, 0x18, "v_ceil_f64", True, True),
   (  -1, 0x19, 0x19, 0x19, 0x19, "v_rndne_f64", True, True),
   (  -1, 0x1a, 0x1a, 0x1a, 0x1a, "v_floor_f64", True, True),
   (  -1,   -1,   -1,   -1, 0x1b, "v_pipeflush", False, False),
   (0x20, 0x20, 0x1b, 0x1b, 0x20, "v_fract_f32", True, True),
   (0x21, 0x21, 0x1c, 0x1c, 0x21, "v_trunc_f32", True, True),
   (0x22, 0x22, 0x1d, 0x1d, 0x22, "v_ceil_f32", True, True),
   (0x23, 0x23, 0x1e, 0x1e, 0x23, "v_rndne_f32", True, True),
   (0x24, 0x24, 0x1f, 0x1f, 0x24, "v_floor_f32", True, True),
   (0x25, 0x25, 0x20, 0x20, 0x25, "v_exp_f32", True, True),
   (0x26, 0x26,   -1,   -1,   -1, "v_log_clamp_f32", True, True),
   (0x27, 0x27, 0x21, 0x21, 0x27, "v_log_f32", True, True),
   (0x28, 0x28,   -1,   -1,   -1, "v_rcp_clamp_f32", True, True),
   (0x29, 0x29,   -1,   -1,   -1, "v_rcp_legacy_f32", True, True),
   (0x2a, 0x2a, 0x22, 0x22, 0x2a, "v_rcp_f32", True, True),
   (0x2b, 0x2b, 0x23, 0x23, 0x2b, "v_rcp_iflag_f32", True, True),
   (0x2c, 0x2c,   -1,   -1,   -1, "v_rsq_clamp_f32", True, True),
   (0x2d, 0x2d,   -1,   -1,   -1, "v_rsq_legacy_f32", True, True),
   (0x2e, 0x2e, 0x24, 0x24, 0x2e, "v_rsq_f32", True, True),
   (0x2f, 0x2f, 0x25, 0x25, 0x2f, "v_rcp_f64", True, True),
   (0x30, 0x30,   -1,   -1,   -1, "v_rcp_clamp_f64", True, True),
   (0x31, 0x31, 0x26, 0x26, 0x31, "v_rsq_f64", True, True),
   (0x32, 0x32,   -1,   -1,   -1, "v_rsq_clamp_f64", True, True),
   (0x33, 0x33, 0x27, 0x27, 0x33, "v_sqrt_f32", True, True),
   (0x34, 0x34, 0x28, 0x28, 0x34, "v_sqrt_f64", True, True),
   (0x35, 0x35, 0x29, 0x29, 0x35, "v_sin_f32", True, True),
   (0x36, 0x36, 0x2a, 0x2a, 0x36, "v_cos_f32", True, True),
   (0x37, 0x37, 0x2b, 0x2b, 0x37, "v_not_b32", False, False),
   (0x38, 0x38, 0x2c, 0x2c, 0x38, "v_bfrev_b32", False, False),
   (0x39, 0x39, 0x2d, 0x2d, 0x39, "v_ffbh_u32", False, False),
   (0x3a, 0x3a, 0x2e, 0x2e, 0x3a, "v_ffbl_b32", False, False),
   (0x3b, 0x3b, 0x2f, 0x2f, 0x3b, "v_ffbh_i32", False, False),
   (0x3c, 0x3c, 0x30, 0x30, 0x3c, "v_frexp_exp_i32_f64", True, False),
   (0x3d, 0x3d, 0x31, 0x31, 0x3d, "v_frexp_mant_f64", True, False),
   (0x3e, 0x3e, 0x32, 0x32, 0x3e, "v_fract_f64", True, True),
   (0x3f, 0x3f, 0x33, 0x33, 0x3f, "v_frexp_exp_i32_f32", True, False),
   (0x40, 0x40, 0x34, 0x34, 0x40, "v_frexp_mant_f32", True, False),
   (0x41, 0x41, 0x35, 0x35, 0x41, "v_clrexcp", False, False),
   (0x42, 0x42, 0x36,   -1, 0x42, "v_movreld_b32", False, False),
   (0x43, 0x43, 0x37,   -1, 0x43, "v_movrels_b32", False, False),
   (0x44, 0x44, 0x38,   -1, 0x44, "v_movrelsd_b32", False, False),
   (  -1,   -1,   -1,   -1, 0x48, "v_movrelsd_2_b32", False, False),
   (  -1,   -1,   -1, 0x37,   -1, "v_screen_partition_4se_b32", False, False),
   (  -1,   -1, 0x39, 0x39, 0x50, "v_cvt_f16_u16", False, True),
   (  -1,   -1, 0x3a, 0x3a, 0x51, "v_cvt_f16_i16", False, True),
   (  -1,   -1, 0x3b, 0x3b, 0x52, "v_cvt_u16_f16", True, False),
   (  -1,   -1, 0x3c, 0x3c, 0x53, "v_cvt_i16_f16", True, False),
   (  -1,   -1, 0x3d, 0x3d, 0x54, "v_rcp_f16", True, True),
   (  -1,   -1, 0x3e, 0x3e, 0x55, "v_sqrt_f16", True, True),
   (  -1,   -1, 0x3f, 0x3f, 0x56, "v_rsq_f16", True, True),
   (  -1,   -1, 0x40, 0x40, 0x57, "v_log_f16", True, True),
   (  -1,   -1, 0x41, 0x41, 0x58, "v_exp_f16", True, True),
   (  -1,   -1, 0x42, 0x42, 0x59, "v_frexp_mant_f16", True, False),
   (  -1,   -1, 0x43, 0x43, 0x5a, "v_frexp_exp_i16_f16", True, False),
   (  -1,   -1, 0x44, 0x44, 0x5b, "v_floor_f16", True, True),
   (  -1,   -1, 0x45, 0x45, 0x5c, "v_ceil_f16", True, True),
   (  -1,   -1, 0x46, 0x46, 0x5d, "v_trunc_f16", True, True),
   (  -1,   -1, 0x47, 0x47, 0x5e, "v_rndne_f16", True, True),
   (  -1,   -1, 0x48, 0x48, 0x5f, "v_fract_f16", True, True),
   (  -1,   -1, 0x49, 0x49, 0x60, "v_sin_f16", True, True),
   (  -1,   -1, 0x4a, 0x4a, 0x61, "v_cos_f16", True, True),
   (  -1, 0x46, 0x4b, 0x4b,   -1, "v_exp_legacy_f32", True, True),
   (  -1, 0x45, 0x4c, 0x4c,   -1, "v_log_legacy_f32", True, True),
   (  -1,   -1,   -1, 0x4f, 0x62, "v_sat_pk_u8_i16", False, False),
   (  -1,   -1,   -1, 0x4d, 0x63, "v_cvt_norm_i16_f16", True, False),
   (  -1,   -1,   -1, 0x4e, 0x64, "v_cvt_norm_u16_f16", True, False),
   (  -1,   -1,   -1, 0x51, 0x65, "v_swap_b32", False, False),
   (  -1,   -1,   -1,   -1, 0x68, "v_swaprel_b32", False, False),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name, in_mod, out_mod) in VOP1:
   opcode(name, gfx9, Format.VOP1, in_mod, out_mod)


# VOPC instructions:

VOPC_CLASS = {
   (0x88, 0x88, 0x10, 0x10, 0x88, "v_cmp_class_f32"),
   (  -1,   -1, 0x14, 0x14, 0x8f, "v_cmp_class_f16"),
   (0x98, 0x98, 0x11, 0x11, 0x98, "v_cmpx_class_f32"),
   (  -1,   -1, 0x15, 0x15, 0x9f, "v_cmpx_class_f16"),
   (0xa8, 0xa8, 0x12, 0x12, 0xa8, "v_cmp_class_f64"),
   (0xb8, 0xb8, 0x13, 0x13, 0xb8, "v_cmpx_class_f64"),
}
for (gfx6, gfx7, gfx8, gfx9, gfx10, name) in VOPC_CLASS:
    opcode(name, gfx9, Format.VOPC, True, False)

COMPF = ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"]

for i in range(8):
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0x20+i, 0x20+i, 0xc8+i, "v_cmp_"+COMPF[i]+"_f16")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0x30+i, 0x30+i, 0xd8+i, "v_cmpx_"+COMPF[i]+"_f16")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0x28+i, 0x28+i, 0xe8+i, "v_cmp_"+COMPF[i+8]+"_f16")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0x38+i, 0x38+i, 0xf8+i, "v_cmpx_"+COMPF[i+8]+"_f16")
   opcode(name, gfx9, Format.VOPC, True, False)

for i in range(16):
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x00+i, 0x00+i, 0x40+i, 0x40+i, 0x00+i, "v_cmp_"+COMPF[i]+"_f32")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x10+i, 0x10+i, 0x50+i, 0x50+i, 0x10+i, "v_cmpx_"+COMPF[i]+"_f32")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x20+i, 0x20+i, 0x60+i, 0x60+i, 0x20+i, "v_cmp_"+COMPF[i]+"_f64")
   opcode(name, gfx9, Format.VOPC, True, False)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x30+i, 0x30+i, 0x70+i, 0x70+i, 0x30+i, "v_cmpx_"+COMPF[i]+"_f64")
   opcode(name, gfx9, Format.VOPC, True, False)
   # GFX_6_7
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x40+i, 0x40+i, -1, -1, -1, "v_cmps_"+COMPF[i]+"_f32")
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x50+i, 0x50+i, -1, -1, -1, "v_cmpsx_"+COMPF[i]+"_f32")
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x60+i, 0x60+i, -1, -1, -1, "v_cmps_"+COMPF[i]+"_f64")
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x70+i, 0x70+i, -1, -1, -1, "v_cmpsx_"+COMPF[i]+"_f64")

COMPI = ["f", "lt", "eq", "le", "gt", "lg", "ge", "tru"]

# GFX_8_9
for i in [0,7]: # only 0 and 7
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xa0+i, 0xa0+i, -1, "v_cmp_"+COMPI[i]+"_i16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xb0+i, 0xb0+i, -1, "v_cmpx_"+COMPI[i]+"_i16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xa8+i, 0xa8+i, -1, "v_cmp_"+COMPI[i]+"_u16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xb8+i, 0xb8+i, -1, "v_cmpx_"+COMPI[i]+"_u16")
   opcode(name, gfx9, Format.VOPC)

for i in range(1, 7): # [1..6]
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xa0+i, 0xa0+i, 0x88+i, "v_cmp_"+COMPI[i]+"_i16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xb0+i, 0xb0+i, 0x98+i, "v_cmpx_"+COMPI[i]+"_i16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xa8+i, 0xa8+i, 0xa8+i, "v_cmp_"+COMPI[i]+"_u16")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, 0xb8+i, 0xb8+i, 0xb8+i, "v_cmpx_"+COMPI[i]+"_u16")
   opcode(name, gfx9, Format.VOPC)

for i in range(8):
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x80+i, 0x80+i, 0xc0+i, 0xc0+i, 0x80+i, "v_cmp_"+COMPI[i]+"_i32")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0x90+i, 0x90+i, 0xd0+i, 0xd0+i, 0x90+i, "v_cmpx_"+COMPI[i]+"_i32")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xa0+i, 0xa0+i, 0xe0+i, 0xe0+i, 0xa0+i, "v_cmp_"+COMPI[i]+"_i64")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xb0+i, 0xb0+i, 0xf0+i, 0xf0+i, 0xb0+i, "v_cmpx_"+COMPI[i]+"_i64")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xc0+i, 0xc0+i, 0xc8+i, 0xc8+i, 0xc0+i, "v_cmp_"+COMPI[i]+"_u32")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xd0+i, 0xd0+i, 0xd8+i, 0xd8+i, 0xd0+i, "v_cmpx_"+COMPI[i]+"_u32")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xe0+i, 0xe0+i, 0xe8+i, 0xe8+i, 0xe0+i, "v_cmp_"+COMPI[i]+"_u64")
   opcode(name, gfx9, Format.VOPC)
   (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (0xf0+i, 0xf0+i, 0xf8+i, 0xf8+i, 0xf0+i, "v_cmpx_"+COMPI[i]+"_u64")
   opcode(name, gfx9, Format.VOPC)


# VOPP instructions: packed 16bit instructions - 1 or 2 inputs and 1 output
VOPP = {
   (0x00, "v_pk_mad_i16"),
   (0x01, "v_pk_mul_lo_u16"),
   (0x02, "v_pk_add_i16"),
   (0x03, "v_pk_sub_i16"),
   (0x04, "v_pk_lshlrev_b16"),
   (0x05, "v_pk_lshrrev_b16"),
   (0x06, "v_pk_ashrrev_i16"),
   (0x00, "v_pk_max_i16"),
   (0x00, "v_pk_min_i16"),
   (0x00, "v_pk_mad_u16"),
   (0x00, "v_pk_add_u16"),
   (0x00, "v_pk_sub_u16"),
   (0x00, "v_pk_max_u16"),
   (0x00, "v_pk_min_u16"),
   (0x00, "v_pk_fma_f16"),
   (0x00, "v_pk_add_f16"),
   (0x00, "v_pk_mul_f16"),
   (0x00, "v_pk_min_f16"),
   (0x00, "v_pk_max_f16"),
   (0x00, "v_pk_fma_mix_f32"), # v_pk_mad_mix_f32 in VEGA ISA
   (0x00, "v_pk_fma_mixlo_f16"),
   (0x00, "v_pk_fma_mixhi_f16"),
}
# (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (-1, -1, -1, code, code, name)
for (code, name) in VOPP:
   opcode(name, code, Format.VOP3P)


# VINTERP instructions: 

VINTRP = {
   (0x00, "v_interp_p1_f32"),
   (0x01, "v_interp_p2_f32"),
   (0x02, "v_interp_mov_f32"),
}
# (gfx6, gfx7, gfx8, gfx9, gfx10, name) = (code, code, code, code, code, name)
for (code, name) in VINTRP:
   opcode(name, code, Format.VINTRP)

# VOP3 instructions: 3 inputs, 1 output
# VOP3b instructions: have a unique scalar output, e.g. VOP2 with vcc out

VOP3b = [
   "v_div_scale_f32",
   "v_div_scale_f64",
   "v_mad_u64_u32",
   "v_mad_i64_i32"
]
#TODO opcode("v_mad_u64_u32", 3, [1,1,2], 2, [2,2], [0, 1], 0, 1, 0, 0, [0, 0, 0])
opcode("v_mad_u64_u32", 488, Format.VOP3B)


VOP3a_32 = {
   (448, "v_mad_legacy_f32", True, True),
   (449, "v_mad_f32", True, True),
   (450, "v_mad_i32_i24", False, False),
   (451, "v_mad_u32_u24", False, False),
   (452, "v_cubeid_f32", True, True),
   (453, "v_cubesc_f32", True, True),
   (454, "v_cubetc_f32", True, True),
   (455, "v_cubema_f32", True, True),
   (456, "v_bfe_u32", False, False),
   (457, "v_bfe_i32", False, False),
   (458, "v_bfi_b32", False, False),
   (459, "v_fma_f32", True, True),
   (460, "v_fma_f64", True, True),
   (461, "v_lerp_u8", False, False),
   (462, "v_alignbit_b32", False, False),
   (463, "v_alignbyte_b32", False, False),
   (464, "v_min3_f32", True, True),
   (465, "v_min3_i32", False, False),
   (466, "v_min3_u32", False, False),
   (467, "v_max3_f32", True, True),
   (468, "v_max3_i32", False, False),
   (469, "v_max3_u32", False, False),
   (470, "v_med3_f32", True, True),
   (471, "v_med3_i32", False, False),
   (472, "v_med3_u32", False, False),
   (473, "v_sad_u8", False, False),
   (474, "v_sad_hi_u8", False, False),
   (475, "v_sad_u16", False, False),
   (476, "v_sad_u32", False, False),
   (477, "v_cvt_pk_u8_f32", True, False),
   (478, "v_div_fixup_f32", True, True),
   (479, "v_div_fixup_f64", True, True),
   (484, "v_msad_u8", False, False),
   (490, "v_mad_legacy_f16", True, True),
   (491, "v_mad_legacy_u16", False, False),
   (492, "v_mad_legacy_i16", False, False),
   (493, "v_perm_b32", False, False),
   (494, "v_fma_legacy_f16", True, True),
   (495, "v_div_fixup_legacy_f16", True, True),
   (497, "v_mad_u32_u16", False, False),
   (498, "v_mad_i32_i16", False, False),
   (499, "v_xad_u32", False, False),
   (500, "v_min3_f16", True, True),
   (501, "v_min3_i16", False, False),
   (502, "v_min3_u16", False, False),
   (503, "v_max3_f16", True, True),
   (504, "v_max3_i16", False, False),
   (505, "v_max3_u16", False, False),
   (506, "v_med3_f16", True, True),
   (507, "v_med3_i16", False, False),
   (508, "v_med3_u16", False, False),
   (509, "v_lshl_add_u32", False, False),
   (510, "v_add_lshl_u32", False, False),
   (511, "v_add3_u32", False, False),
   (512, "v_lshl_or_b32", False, False),
   (513, "v_and_or_b32", False, False),
   (514, "v_or3_b32", False, False),
   (515, "v_mad_f16", True, True),
   (516, "v_mad_u16", False, False),
   (517, "v_mad_i16", False, False),
   (518, "v_fma_f16", True, True),
   (519, "v_div_fixup_f16", True, True),
}
for code, name, in_mod, out_mod in VOP3a_32:
   opcode(name, code, Format.VOP3A, in_mod, out_mod)


#two parameters
VOP3a_32_2 = {
   (496, "v_cvt_pkaccum_u8_f32"),
   (645, "v_mul_lo_u32"),
   (646, "v_mul_hi_u32"),
   (647, "v_mul_hi_i32"),
   (648, "v_ldexp_f32"),
   (650, "v_writelane_b32"),
   (652, "v_mbcnt_lo_u32_b32"),
   (653, "v_mbcnt_hi_u32_b32"),
   (659, "v_bfm_b32"),
   (660, "v_cvt_pknorm_i16_f32"),
   (661, "v_cvt_pknorm_u16_f32"),
   (662, "v_cvt_pkrtz_f16_f32"),
   (663, "v_cvt_pk_u16_u32"),
   (664, "v_cvt_pk_i16_i32"),
   (665, "v_cvt_pknorm_i16_f16"),
   (666, "v_cvt_pknorm_u16_f16"),
   (668, "v_add_i32"),
   (669, "v_sub_i32"),
   (670, "v_add_i16"),
   (671, "v_sub_i16"),
   (672, "v_pack_b32_f16"),
}
for code, name in VOP3a_32_2:
   opcode(name, code, Format.VOP3A)

VOP3a_64_2 = [
   (640, "v_add_f64"),
   (641, "v_mul_f64"),
   (642, "v_min_f64"),
   (643, "v_max_f64"),
   (644, "v_ldexp_f64"),
   (655, "v_lshlrev_b64"),
   (656, "v_lshrrev_b64"), #64bit
   (657, "v_ashrrev_i64"), #64bit
   (658, "v_trig_preop_f64"), #64bit
]
for code, name in VOP3a_64_2:
   opcode(name, code, Format.VOP3A)
   
VOP3a_SPECIAL = [
   "v_bcnt_u32_b32", # one input
   "v_readlane_b32", # 2 inputs, returns sgpr
   "v_div_fmas_f32", #uses vcc
   "v_div_fmas_f64", #3 inputs, takes vcc
   "v_qsad_pk_u16_u8", #(2,1,2)gpr's
   "v_mqsad_pk_u16_u8", #(2,1,2)gpr's
   "v_mqsad_u32_u8", #(2,1,4)gpr's

   "v_interp_p1ll_f16", #parameters?
   "v_interp_p1lv_f16", #parameters?
   "v_interp_p2_legacy_f16",
   "v_interp_p2_f16"
]
opcode("v_bcnt_u32_b32", 651, Format.VOP3A)
opcode("v_readlane_b32", 649, Format.VOP3A)
opcode("v_div_fmas_f32", 482, Format.VOP3A)
opcode("v_div_fmas_f64", 483, Format.VOP3A)
opcode("v_qsad_pk_u16_u8", 485, Format.VOP3A)
opcode("v_mqsad_pk_u16_u8", 486, Format.VOP3A)
opcode("v_mqsad_u32_u8", 487, Format.VOP3A)


# DS instructions: 3 inputs (1 addr, 2 data), 1 output

DS = [
   (0, "ds_add_u32"),
   (1, "ds_sub_u32"),
   (2, "ds_rsub_u32"),
   (3, "ds_inc_u32"),
   (4, "ds_dec_u32"),
   (5, "ds_min_i32"),
   (6, "ds_max_i32"),
   (7, "ds_min_u32"),
   (8, "ds_max_u32"),
   (9, "ds_and_b32"),
   (10, "ds_or_b32"),
   (11, "ds_xor_b32"),
   (12, "ds_mskor_b32"),
   (13, "ds_write_b32"),
   (14, "ds_write2_b32"),
   (15, "ds_write2st64_b32"),
   (16, "ds_cmpst_b32"),
   (17, "ds_cmpst_f32"),
   (18, "ds_min_f32"),
   (19, "ds_max_f32"),
   (20, "ds_nop"),
   (21, "ds_add_f32"),
   (29, "ds_write_addtid_b32"),
   (30, "ds_write_b8"),
   (31, "ds_write_b16"),
   (32, "ds_add_rtn_u32"),
   (33, "ds_sub_rtn_u32"),
   (34, "ds_rsub_rtn_u32"),
   (35, "ds_inc_rtn_u32"),
   (36, "ds_dec_rtn_u32"),
   (37, "ds_min_rtn_i32"),
   (38, "ds_max_rtn_i32"),
   (39, "ds_min_rtn_u32"),
   (40, "ds_max_rtn_u32"),
   (41, "ds_and_rtn_b32"),
   (42, "ds_or_rtn_b32"),
   (43, "ds_xor_rtn_b32"),
   (44, "ds_mskor_rtn_b32"),
   (45, "ds_wrxchg_rtn_b32"),
   (46, "ds_wrxchg2_rtn_b32"),
   (47, "ds_wrxchg2st64_rtn_b32"),
   (48, "ds_cmpst_rtn_b32"),
   (49, "ds_cmpst_rtn_f32"),
   (50, "ds_min_rtn_f32"),
   (51, "ds_max_rtn_f32"),
   (52, "ds_wrap_rtn_b32"),
   (53, "ds_add_rtn_f32"),
   (54, "ds_read_b32"),
   (55, "ds_read2_b32"),
   (56, "ds_read2st64_b32"),
   (57, "ds_read_i8"),
   (58, "ds_read_u8"),
   (59, "ds_read_i16"),
   (60, "ds_read_u16"),
   (61, "ds_swizzle_b32"), #data1 & offset, no addr/data2
   (62, "ds_permute_b32"),
   (63, "ds_bpermute_b32"),
   (64, "ds_add_u64"),
   (65, "ds_sub_u64"),
   (66, "ds_rsub_u64"),
   (67, "ds_inc_u64"),
   (68, "ds_dec_u64"),
   (69, "ds_min_i64"),
   (70, "ds_max_i64"),
   (71, "ds_min_u64"),
   (72, "ds_max_u64"),
   (73, "ds_and_b64"),
   (74, "ds_or_b64"),
   (75, "ds_xor_b64"),
   (76, "ds_mskor_b64"),
   (77, "ds_write_b64"),
   (78, "ds_write2_b64"),
   (79, "ds_write2st64_b64"),
   (80, "ds_cmpst_b64"),
   (81, "ds_cmpst_f64"),
   (82, "ds_min_f64"),
   (83, "ds_max_f64"),
   (84, "ds_write_b8_d16_hi"),
   (85, "ds_write_b16_d16_hi"),
   (86, "ds_read_u8_d16"),
   (87, "ds_read_u8_d16_hi"),
   (88, "ds_read_i8_d16"),
   (89, "ds_read_i8_d16_hi"),
   (90, "ds_read_u16_d16"),
   (91, "ds_read_u16_d16_hi"),
   (96, "ds_add_rtn_u64"),
   (97, "ds_sub_rtn_u64"),
   (98, "ds_rsub_rtn_u64"),
   (99, "ds_inc_rtn_u64"),
   (100, "ds_dec_rtn_u64"),
   (101, "ds_min_rtn_i64"),
   (102, "ds_max_rtn_i64"),
   (103, "ds_min_rtn_u64"),
   (104, "ds_max_rtn_u64"),
   (105, "ds_and_rtn_b64"),
   (106, "ds_or_rtn_b64"),
   (107, "ds_xor_rtn_b64"),
   (108, "ds_mskor_rtn_b64"),
   (109, "ds_wrxchg_rtn_b64"),
   (110, "ds_wrxchg2_rtn_b64"),
   (111, "ds_wrxchg2st64_rtn_b64"),
   (112, "ds_cmpst_rtn_b64"),
   (113, "ds_cmpst_rtn_f64"),
   (114, "ds_min_rtn_f64"),
   (115, "ds_max_rtn_f64"),
   (118, "ds_read_b64"),
   (119, "ds_read2_b64"),
   (120, "ds_read2st64_b64"),
   (126, "ds_condxchg32_rtn_b64"),
   (128, "ds_add_src2_u32"),
   (129, "ds_sub_src2_u32"),
   (130, "ds_rsub_src2_u32"),
   (131, "ds_inc_src2_u32"),
   (132, "ds_dec_src2_u32"),
   (133, "ds_min_src2_i32"),
   (134, "ds_max_src2_i32"),
   (135, "ds_min_src2_u32"),
   (136, "ds_max_src2_u32"),
   (137, "ds_and_src2_b32"),
   (138, "ds_or_src2_b32"),
   (139, "ds_xor_src2_b32"),
   (141, "ds_write_src2_b32"),
   (146, "ds_min_src2_f32"),
   (147, "ds_max_src2_f32"),
   (149, "ds_add_src2_f32"),
   (152, "ds_gws_sema_release_all"),
   (153, "ds_gws_init"),
   (154, "ds_gws_sema_v"),
   (155, "ds_gws_sema_br"),
   (156, "ds_gws_sema_p"),
   (157, "ds_gws_barrier"),
   (182, "ds_read_addtid_b32"),
   (189, "ds_consume"),
   (190, "ds_append"),
   (191, "ds_ordered_count"),
   (192, "ds_add_src2_u64"),
   (193, "ds_sub_src2_u64"),
   (194, "ds_rsub_src2_u64"),
   (195, "ds_inc_src2_u64"),
   (196, "ds_dec_src2_u64"),
   (197, "ds_min_src2_i64"),
   (198, "ds_max_src2_i64"),
   (199, "ds_min_src2_u64"),
   (200, "ds_max_src2_u64"),
   (201, "ds_and_src2_b64"),
   (202, "ds_or_src2_b64"),
   (203, "ds_xor_src2_b64"),
   (205, "ds_write_src2_b64"),
   (210, "ds_min_src2_f64"),
   (211, "ds_max_src2_f64"),
   (222, "ds_write_b96"),
   (223, "ds_write_b128"),
   (254, "ds_read_b96"),
   (255, "ds_read_b128"),
]
for (code, name) in DS:
    opcode(name, code, Format.DS)


# MUBUF instructions:

MUBUF = {
   (0, "buffer_load_format_x"),
   (1, "buffer_load_format_xy"),
   (2, "buffer_load_format_xyz"),
   (3, "buffer_load_format_xyzw"),
   (4, "buffer_store_format_x"),
   (5, "buffer_store_format_xy"),
   (6, "buffer_store_format_xyz"),
   (7, "buffer_store_format_xyzw"),
   (8, "buffer_load_format_d16_x"),
   (9, "buffer_load_format_d16_xy"),
   (10, "buffer_load_format_d16_xyz"),
   (11, "buffer_load_format_d16_xyzw"),
   (12, "buffer_store_format_d16_x"),
   (13, "buffer_store_format_d16_xy"),
   (14, "buffer_store_format_d16_xyz"),
   (15, "buffer_store_format_d16_xyzw"),
   (16, "buffer_load_ubyte"),
   (17, "buffer_load_sbyte"),
   (18, "buffer_load_ushort"),
   (19, "buffer_load_sshort"),
   (20, "buffer_load_dword"),
   (21, "buffer_load_dwordx2"),
   (22, "buffer_load_dwordx3"),
   (23, "buffer_load_dwordx4"),
   (24, "buffer_store_byte"),
   (25, "buffer_store_byte_d16_hi"),
   (26, "buffer_store_short"),
   (27, "buffer_store_short_d16_hi"),
   (28, "buffer_store_dword"),
   (29, "buffer_store_dwordx2"),
   (30, "buffer_store_dwordx3"),
   (31, "buffer_store_dwordx4"),
   (32, "buffer_load_ubyte_d16"),
   (33, "buffer_load_ubyte_d16_hi"),
   (34, "buffer_load_sbyte_d16"),
   (35, "buffer_load_sbyte_d16_hi"),
   (36, "buffer_load_short_d16"),
   (37, "buffer_load_short_d16_hi"),
   (38, "buffer_load_format_d16_hi_x"),
   (39, "buffer_store_format_d16_hi_x"),
   (61, "buffer_store_lds_dword"),
   (62, "buffer_wbinvl1"),
   (63, "buffer_wbinvl1_vol"),
   (64, "buffer_atomic_swap"),
   (65, "buffer_atomic_cmpswap"),
   (66, "buffer_atomic_add"),
   (67, "buffer_atomic_sub"),
   (68, "buffer_atomic_smin"),
   (69, "buffer_atomic_umin"),
   (70, "buffer_atomic_smax"),
   (71, "buffer_atomic_umax"),
   (72, "buffer_atomic_and"),
   (73, "buffer_atomic_or"),
   (74, "buffer_atomic_xor"),
   (75, "buffer_atomic_inc"),
   (76, "buffer_atomic_dec"),
   (96, "buffer_atomic_swap_x2"),
   (97, "buffer_atomic_cmpswap_x2"),
   (98, "buffer_atomic_add_x2"),
   (99, "buffer_atomic_sub_x2"),
   (100, "buffer_atomic_smin_x2"),
   (101, "buffer_atomic_umin_x2"),
   (102, "buffer_atomic_smax_x2"),
   (103, "buffer_atomic_umax_x2"),
   (104, "buffer_atomic_and_x2"),
   (105, "buffer_atomic_or_x2"),
   (106, "buffer_atomic_xor_x2"),
   (107, "buffer_atomic_inc_x2"),
   (108, "buffer_atomic_dec_x2"),
}
for (code, name) in MUBUF:
    opcode(name, code, Format.MUBUF)


MIMG = [
   (0, "image_load"),
   (1, "image_load_mip"),
   (2, "image_load_pck"),
   (3, "image_load_pck_sgn"),
   (4, "image_load_mip_pck"),
   (5, "image_load_mip_pck_sgn"),
   (8, "image_store"),
   (9, "image_store_mip"),
   (10, "image_store_pck"),
   (11, "image_store_mip_pck"),
   (14, "image_get_resinfo"),
   (16, "image_atomic_swap"),
   (17, "image_atomic_cmpswap"),
   (18, "image_atomic_add"),
   (19, "image_atomic_sub"),
   (20, "image_atomic_smin"),
   (21, "image_atomic_umin"),
   (22, "image_atomic_smax"),
   (23, "image_atomic_umax"),
   (24, "image_atomic_and"),
   (25, "image_atomic_or"),
   (26, "image_atomic_xor"),
   (27, "image_atomic_inc"),
   (28, "image_atomic_dec"),
   (32, "image_sample"),
   (33, "image_sample_cl"),
   (34, "image_sample_d"),
   (35, "image_sample_d_cl"),
   (36, "image_sample_l"),
   (37, "image_sample_b"),
   (38, "image_sample_b_cl"),
   (39, "image_sample_lz"),
   (40, "image_sample_c"),
   (41, "image_sample_c_cl"),
   (42, "image_sample_c_d"),
   (43, "image_sample_c_d_cl"),
   (44, "image_sample_c_l"),
   (45, "image_sample_c_b"),
   (46, "image_sample_c_b_cl"),
   (47, "image_sample_c_lz"),
   (48, "image_sample_o"),
   (49, "image_sample_cl_o"),
   (50, "image_sample_d_o"),
   (51, "image_sample_d_cl_o"),
   (52, "image_sample_l_o"),
   (53, "image_sample_b_o"),
   (54, "image_sample_b_cl_o"),
   (55, "image_sample_lz_o"),
   (56, "image_sample_c_o"),
   (57, "image_sample_c_cl_o"),
   (58, "image_sample_c_d_o"),
   (59, "image_sample_c_d_cl_o"),
   (60, "image_sample_c_l_o"),
   (61, "image_sample_c_b_o"),
   (62, "image_sample_c_b_cl_o"),
   (63, "image_sample_c_lz_o"),
   (64, "image_gather4"),
   (65, "image_gather4_cl"),
   (66, "image_gather4h"),
   (68, "image_gather4_l"),
   (69, "image_gather4_b"),
   (70, "image_gather4_b_cl"),
   (71, "image_gather4_lz"),
   (72, "image_gather4_c"),
   (73, "image_gather4_c_cl"),
   (74, "image_gather4h_pck"),
   (75, "image_gather8h_pck"),
   (76, "image_gather4_c_l"),
   (77, "image_gather4_c_b"),
   (78, "image_gather4_c_b_cl"),
   (79, "image_gather4_c_lz"),
   (80, "image_gather4_o"),
   (81, "image_gather4_cl_o"),
   (84, "image_gather4_l_o"),
   (85, "image_gather4_b_o"),
   (86, "image_gather4_b_cl_o"),
   (87, "image_gather4_lz_o"),
   (88, "image_gather4_c_o"),
   (89, "image_gather4_c_cl_o"),
   (92, "image_gather4_c_l_o"),
   (93, "image_gather4_c_b_o"),
   (94, "image_gather4_c_b_cl_o"),
   (95, "image_gather4_c_lz_o"),
   (96, "image_get_lod"),
   (104, "image_sample_cd"),
   (105, "image_sample_cd_cl"),
   (106, "image_sample_c_cd"),
   (107, "image_sample_c_cd_cl"),
   (108, "image_sample_cd_o"),
   (109, "image_sample_cd_cl_o"),
   (110, "image_sample_c_cd_o"),
   (111, "image_sample_c_cd_cl_o"),
]
for code, name in MIMG:
    opcode(name, code, Format.MIMG)

NOT_DPP = [
   "v_madmk_f32",
   "v_madak_f32",
   "v_madmk_f16",
   "v_madak_f16",
   "v_readfirstlane_b32",
   "v_cvt_i32_f64",
   "v_cvt_f64_i32",
   "v_cvt_f32_f64",
   "v_cvt_f64_f32",
   "v_cvt_u32_f64",
   "v_cvt_f64_u32",
   "v_trunc_f64",
   "v_ceil_f64",
   "v_rndne_f64",
   "v_floor_f64",
   "v_rcp_f64",
   "v_rsq_f64",
   "v_sqrt_f64",
   "v_frexp_exp_i32_f64",
   "v_frexp_mant_f64",
   "v_fract_f64",
   "v_clrexcp",
   "v_swap_b32"
]

GLOBAL = [
   (16, "global_load_ubyte"),
   (17, "global_load_sbyte"),
   (18, "global_load_ushort"),
   (19, "global_load_sshort"),
   (20, "global_load_dword"),
   (21, "global_load_dwordx2"),
   (22, "global_load_dwordx3"),
   (23, "global_load_dwordx4"),
   (24, "global_store_byte"),
   (25, "global_store_byte_d16_hi"),
   (26, "global_store_short"),
   (27, "global_store_short_d16_hi"),
   (28, "global_store_dword"),
   (29, "global_store_dwordx2"),
   (30, "global_store_dwordx3"),
   (31, "global_store_dwordx4"),
   (32, "global_load_ubyte_d16"),
   (33, "global_load_ubyte_d16_hi"),
   (34, "global_load_sbyte_d16"),
   (35, "global_load_sbyte_d16_hi"),
   (36, "global_load_short_d16"),
   (37, "global_load_short_d16_hi"),
   (64, "global_atomic_swap"),
   (65, "global_atomic_cmpswap"),
   (66, "global_atomic_add"),
   (67, "global_atomic_sub"),
   (68, "global_atomic_smin"),
   (69, "global_atomic_umin"),
   (70, "global_atomic_smax"),
   (71, "global_atomic_umax"),
   (72, "global_atomic_and"),
   (73, "global_atomic_or"),
   (74, "global_atomic_xor"),
   (75, "global_atomic_inc"),
   (76, "global_atomic_dec"),
   (96, "global_atomic_swap_x2"),
   (97, "global_atomic_cmpswap_x2"),
   (98, "global_atomic_add_x2"),
   (99, "global_atomic_sub_x2"),
   (100, "global_atomic_smin_x2"),
   (101, "global_atomic_umin_x2"),
   (102, "global_atomic_smax_x2"),
   (103, "global_atomic_umax_x2"),
   (104, "global_atomic_and_x2"),
   (105, "global_atomic_or_x2"),
   (106, "global_atomic_xor_x2"),
   (107, "global_atomic_inc_x2"),
   (108, "global_atomic_dec_x2"),
]
for code, name in GLOBAL:
    opcode(name, code, Format.GLOBAL)

FLAT = [
   (16, "flat_load_ubyte"),
   (17, "flat_load_sbyte"),
   (18, "flat_load_ushort"),
   (19, "flat_load_sshort"),
   (20, "flat_load_dword"),
   (21, "flat_load_dwordx2"),
   (22, "flat_load_dwordx3"),
   (23, "flat_load_dwordx4"),
   (24, "flat_store_byte"),
   (25, "flat_store_byte_d16_hi"),
   (26, "flat_store_short"),
   (27, "flat_store_short_d16_hi"),
   (28, "flat_store_dword"),
   (29, "flat_store_dwordx2"),
   (30, "flat_store_dwordx3"),
   (31, "flat_store_dwordx4"),
   (32, "flat_load_ubyte_d16"),
   (33, "flat_load_ubyte_d16_hi"),
   (34, "flat_load_sbyte_d16"),
   (35, "flat_load_sbyte_d16_hi"),
   (36, "flat_load_short_d16"),
   (37, "flat_load_short_d16_hi"),
   (64, "flat_atomic_swap"),
   (65, "flat_atomic_cmpswap"),
   (66, "flat_atomic_add"),
   (67, "flat_atomic_sub"),
   (68, "flat_atomic_smin"),
   (69, "flat_atomic_umin"),
   (70, "flat_atomic_smax"),
   (71, "flat_atomic_umax"),
   (72, "flat_atomic_and"),
   (73, "flat_atomic_or"),
   (74, "flat_atomic_xor"),
   (75, "flat_atomic_inc"),
   (76, "flat_atomic_dec"),
   (96, "flat_atomic_swap_x2"),
   (97, "flat_atomic_cmpswap_x2"),
   (98, "flat_atomic_add_x2"),
   (99, "flat_atomic_sub_x2"),
   (100, "flat_atomic_smin_x2"),
   (101, "flat_atomic_umin_x2"),
   (102, "flat_atomic_smax_x2"),
   (103, "flat_atomic_umax_x2"),
   (104, "flat_atomic_and_x2"),
   (105, "flat_atomic_or_x2"),
   (106, "flat_atomic_xor_x2"),
   (107, "flat_atomic_inc_x2"),
   (108, "flat_atomic_dec_x2"),
]
for code, name in FLAT:
    opcode(name, code, Format.FLAT)
