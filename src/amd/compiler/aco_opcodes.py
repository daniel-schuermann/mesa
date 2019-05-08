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

# helper variables
VCC = "VCC"
SCC = "SCC"
b = "b"
s1 = "s1"
s2 = "s2"
s3 = "s3"
s4 = "s4"
s8 = "s8"
s16 = "s16"
v1 = "v1"
v2 = "v2"
v3 = "v3"
v4 = "v4"

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
                 ('bool', 'vm', 'false', 'valid_mask')]
      elif self == Format.PSEUDO_BRANCH:
         return [('Block*', 'target0', None, 'targets[0]'),
                 ('Block*', 'target1', 'NULL', 'targets[1]')]
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
      else:
         return []

   def get_builder_field_names(self):
      return [f[1] for f in self.get_builder_fields()]

   def get_builder_field_dests(self):
      return [(f[3] if len(f) >= 4 else f[1]) for f in self.get_builder_fields()]

   def get_builder_field_decls(self):
      return [('%s %s=%s' % (f[0], f[1], f[2]) if f[2] else '%s %s' % (f[0], f[1])) for f in self.get_builder_fields()]


class Opcode(object):
   """Class that represents all the information we have about the opcode
   NOTE: this must be kept in sync with aco_op_info
   """
   def __init__(self, name, code, format, input_mod, output_mod):
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
      assert isinstance(code, int)
      assert isinstance(format, Format)
      assert isinstance(input_mod, bool)
      assert isinstance(output_mod, bool)

      self.name = name
      self.opcode = code
      self.input_mod = "true" if input_mod else "false"
      self.output_mod = "true" if output_mod else "false"
      self.format = format


# global dictionary of opcodes
opcodes = {}

# VOPC to GFX6 opcode translation map
VOPC_GFX6 = [0] * 256

def opcode(name, code = 0, format = Format.PSEUDO, input_mod = False, output_mod = False):
   assert name not in opcodes
   opcodes[name] = Opcode(name, code, format, input_mod, output_mod)

opcode("exp", format = Format.EXP)
opcode("p_parallelcopy")
opcode("p_startpgm")
opcode("p_phi")
opcode("p_linear_phi")
opcode("p_discard_if")

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

# start/end linear vgprs ?
opcode("p_start_linear_vgpr")
opcode("p_end_linear_vgpr")

opcode("p_wqm")
opcode("p_is_helper")

opcode("p_as_uniform")

opcode("p_fs_buffer_store_smem", format=Format.SMEM)

# SOP2 instructions: 2 scalar inputs, 1 scalar output (+optional scc)
SOP2_SCC = {
   (0, "s_add_u32"), 
   (1, "s_sub_u32"),
   (2, "s_add_i32"),
   (3, "s_sub_i32"),
   (6, "s_min_i32"),
   (7, "s_min_u32"),
   (8, "s_max_i32"),
   (9, "s_max_u32"),
   (12, "s_and_b32"),
   (14, "s_or_b32"),
   (16, "s_xor_b32"),
   (18, "s_andn2_b32"),
   (20, "s_orn2_b32"),
   (22, "s_nand_b32"),
   (24, "s_nor_b32"),
   (26, "s_xnor_b32"),
   (28, "s_lshl_b32"),
   (30, "s_lshr_b32"),
   (32, "s_ashr_i32"),
   (37, "s_bfe_u32"),
   (38, "s_bfe_i32"),
   (42, "s_absdiff_i32"),
   (46, "s_lshl1_add_u32"),
   (47, "s_lshl2_add_u32"),
   (48, "s_lshl3_add_u32"),
   (49, "s_lshl4_add_u32"),
}
for code, name in SOP2_SCC:
   opcode(name, code, Format.SOP2)

SOP2_NOSCC = {
   (34, "s_bfm_b32"),
   (36, "s_mul_i32"),
   (44, "s_mul_hi_u32"),
   (45, "s_mul_hi_i32"),
   (50, "s_pack_ll_b32_b16"),
   (51, "s_pack_lh_b32_b16"),
   (52, "s_pack_hh_b32_b16"),
}
for code, name in SOP2_NOSCC:
   opcode(name, code, Format.SOP2)

SOP2_64 = {
   (13, "s_and_b64"),
   (15, "s_or_b64"),
   (17, "s_xor_b64"),
   (19, "s_andn2_b64"),
   (21, "s_orn2_b64"),
   (23, "s_nand_b64"),
   (25, "s_nor_b64"),
   (27, "s_xnor_b64"),
   (29, "s_lshl_b64"),
   (31, "s_lshr_b64"),
   (33, "s_ashr_i64"),
   (39, "s_bfe_u64"),
   (40, "s_bfe_i64"),
}
for code, name in SOP2_64:
   opcode(name, code, Format.SOP2)

SOP2_SPECIAL = {
   (4, "s_addc_u32"),
   (5, "s_subb_u32"),
   (10, "s_cselect_b32"),
   (11, "s_cselect_b64"),
   (35, "s_bfm_b64"),
   (41, "s_cbranch_g_fork"),
   (43, "s_rfe_restore_b64"),
}
opcode("s_addc_u32", 4, Format.SOP2)
opcode("s_subb_u32", 5, Format.SOP2)
opcode("s_cselect_b32", 10, Format.SOP2)
opcode("s_cselect_b64", 11, Format.SOP2)
opcode("s_bfm_b64", 35, Format.SOP2)
opcode("s_cbranch_g_fork", 41, Format.SOP2)
opcode("s_rfe_restore_b64", 43, Format.SOP2)


# SOPK instructions: 0 input (+ imm), 1 output + optional scc
SOPK_SCC = {
   (2, "s_cmpk_eq_i32"),
   (3, "s_cmpk_lg_i32"),
   (4, "s_cmpk_gt_i32"),
   (5, "s_cmpk_ge_i32"),
   (6, "s_cmpk_lt_i32"),
   (7, "s_cmpk_le_i32"),
   (8, "s_cmpk_eq_u32"),
   (9, "s_cmpk_lg_u32"),
   (10, "s_cmpk_gt_u32"),
   (11, "s_cmpk_ge_u32"),
   (12, "s_cmpk_lt_u32"),
   (13, "s_cmpk_le_u32"),
}
for code, name in SOPK_SCC:
   opcode(name, code, Format.SOPK)

SOPK_SPECIAL = {
   (0, "s_movk_i32"),
   (1, "s_cmovk_i32"),
   (14, "s_addk_i32"),
   (15, "s_mulk_i32"),
   (16, "s_cbranch_i_fork"),
   (17, "s_getreg_b32"),
   (18, "s_setreg_b32"),
   (20, "s_setreg_imm32_b32"), # requires 32bit literal
   (21, "s_call_b64"),
}
opcode("s_movk_i32", 0, Format.SOPK)
opcode("s_cmovk_i32", 1, Format.SOPK)
opcode("s_addk_i32", 14, Format.SOPK)
opcode("s_mulk_i32", 15, Format.SOPK)
opcode("s_cbranch_i_fork", 16, Format.SOPK)
opcode("s_getreg_b32", 17, Format.SOPK)
opcode("s_setreg_b32", 18, Format.SOPK)
opcode("s_setreg_imm32_b32", 20, Format.SOPK)
opcode("s_call_b64", 21, Format.SOPK)


# SOP1 instructions: 1 input, 1 output (+optional SCC)
SOP1_NOSCC = {
   (0, "s_mov_b32"),
   (8, "s_brev_b32"),
   (14, "s_ff0_i32_b32"),
   (15, "s_ff0_i32_b64"),
   (16, "s_ff1_i32_b32"),
   (17, "s_ff1_i32_b64"),
   (18, "s_flbit_i32_b32"),
   (19, "s_flbit_i32_b64"),
   (20, "s_flbit_i32"),
   (21, "s_flbit_i32_i64"),
   (22, "s_sext_i32_i8"),
   (23, "s_sext_i32_i16"),
   (24, "s_bitset0_b32"),
   (26, "s_bitset1_b32"),
   (42, "s_movrels_b32"),
   (44, "s_movreld_b32"),
}
for code, name in SOP1_NOSCC:
   opcode(name, code, Format.SOP1)

SOP1_NOSCC_64 = {
   (1, "s_mov_b64"),
   (9, "s_brev_b64"),
   (25, "s_bitset0_b64"),
   (27, "s_bitset1_b64"),
   (30, "s_swappc_b64"),
   (43, "s_movrels_b64"),
   (45, "s_movreld_b64")
}
for code, name in SOP1_NOSCC_64:
   opcode(name, code, Format.SOP1)

SOP1_SCC = {
   (4, "s_not_b32"),
   (6, "s_wqm_b32"),
   (10, "s_bcnt0_i32_b32"),
   (12, "s_bcnt1_i32_b32"),
   (40, "s_quadmask_b32"),
   (48, "s_abs_i32")
}
for code, name in SOP1_SCC:
   opcode(name, code, Format.SOP1)

SOP1_SCC_64 = {
   (5, "s_not_b64"),
   (7, "s_wqm_b64"),
   (32, "s_and_saveexec_b64"),
   (33, "s_or_saveexec_b64"),
   (34, "s_xor_saveexec_b64"),
   (35, "s_andn2_saveexec_b64"),
   (36, "s_orn2_saveexec_b64"),
   (37, "s_nand_saveexec_b64"),
   (38, "s_nor_saveexec_b64"),
   (39, "s_xnor_saveexec_b64"),
   (41, "s_quadmask_b64"),
   (51, "s_andn1_saveexec_b64"),
   (52, "s_orn1_saveexec_b64"),
   (53, "s_andn1_wrexec_b64"),
   (54, "s_andn2_wrexec_b64"),
}
for code, name in SOP1_SCC_64:
   opcode(name, code, Format.SOP1)

SOP1_SPECIAL = [
   "s_cmov_b32",
   "s_cmov_b64",
   "s_bcnt0_i32_b64",
   "s_bcnt1_i32_b64",
   "s_getpc_b64",
   "s_setpc_b64",
   "s_rfe_b64",
   "s_cbranch_join",
   "s_set_gpr_idx_idx",
   "s_bitreplicate_b64_b32"
]
opcode("s_cmov_b32", 2, Format.SOP1)
opcode("s_cmov_b64", 3, Format.SOP1)
opcode("s_bcnt0_i32_b64", 11, Format.SOP1)
opcode("s_bcnt1_i32_b64", 13, Format.SOP1)
opcode("s_getpc_b64", 28, Format.SOP1)
opcode("s_setpc_b64", 29, Format.SOP1)
opcode("s_rfe_b64", 31, Format.SOP1)
opcode("s_cbranch_join", 46, Format.SOP1)
opcode("s_set_gpr_idx_idx", 50, Format.SOP1)
opcode("s_bitreplicate_b64_b32", 55, Format.SOP1)


# SOPC instructions: 2 inputs and 0 outputs (+SCC)
SOPC_SCC = {
   (0, "s_cmp_eq_i32"),
   (1, "s_cmp_lg_i32"),
   (2, "s_cmp_gt_i32"),
   (3, "s_cmp_ge_i32"),
   (4, "s_cmp_lt_i32"),
   (5, "s_cmp_le_i32"),
   (6, "s_cmp_eq_u32"),
   (7, "s_cmp_lg_u32"),
   (8, "s_cmp_gt_u32"),
   (9, "s_cmp_ge_u32"),
   (10, "s_cmp_lt_u32"),
   (11, "s_cmp_le_u32"),
   (12, "s_bitcmp0_b32"),
   (13, "s_bitcmp1_b32"),
   (14, "s_bitcmp0_b64"),
   (15, "s_bitcmp1_b64"),
   (18, "s_cmp_eq_u64"),
   (19, "s_cmp_lg_u64")
}
for code, name in SOPC_SCC:
   opcode(name, code, Format.SOPC)

SOPC_SPECIAL = [
   "s_setvskip",
   "s_set_gpr_idx_on"
]
opcode("s_setvskip", 2, Format.SOPC)
opcode("s_set_gpr_idx_on", 1, Format.SOPC)


# SOPP instructions: 0 inputs (+optional scc/vcc) , 0 outputs
SOPP_SPECIAL = {
   (0, "s_nop"),
   (1, "s_endpgm"),
   (2, "s_branch"),
   (3, "s_wakeup"),
   (4, "s_cbranch_scc0"),
   (5, "s_cbranch_scc1"),
   (6, "s_cbranch_vccz"),
   (7, "s_cbranch_vccnz"),
   (8, "s_cbranch_execz"),
   (9, "s_cbranch_execnz"),
   (10, "s_barrier"),
   (11, "s_setkill"),
   (12, "s_waitcnt"),
   (13, "s_sethalt"),
   (14, "s_sleep"),
   (15, "s_setprio"),
   (16, "s_sendmsg"),
   (17, "s_sendmsghalt"),
   (18, "s_trap"),
   (19, "s_icache_inv"),
   (20, "s_incperflevel"),
   (21, "s_decperflevel"),
   (22, "s_ttracedata"),
   (23, "s_cbranch_cdbgsys"),
   (24, "s_cbranch_cdbguser"),
   (25, "s_cbranch_cdbgsys_or_user"),
   (26, "s_cbranch_cdbgsys_and_user"),
   (27, "s_endpgm_saved"),
   (28, "s_set_grp_idx_off"),
   (29, "s_set_grp_idx_mode"),
   (30, "s_endpgm_ordered_ps_done")
}
for code, name in SOPP_SPECIAL:
   opcode(name, code, Format.SOPP)


# SMEM instructions: sbase input (2 sgpr), potentially 2 offset inputs, 1 sdata input/output
SMEM_LOAD = [
   (0, "s_load_dword", s1),
   (1, "s_load_dwordx2", s2),
   (2, "s_load_dwordx4", s4),
   (3, "s_load_dwordx8", s8),
   (4, "s_load_dwordx16", s16),
   (5, "s_scratch_load_dword", s1),
   (6, "s_scratch_load_dwordx2", s2),
   (7, "s_scratch_load_dwordx4", s4),
   (8, "s_buffer_load_dword", s1),
   (9, "s_buffer_load_dwordx2", s2),
   (10, "s_buffer_load_dwordx4", s4),
   (11, "s_buffer_load_dwordx8", s8),
   (12, "s_buffer_load_dwordx16", s16)
]
for (code, name, size) in SMEM_LOAD:
   opcode(name, code, Format.SMEM)

SMEM_STORE = [
   (16, "s_store_dword", 1),
   (17, "s_store_dwordx2", 2),
   (18, "s_store_dwordx4", 4),
   (21, "s_scratch_store_dword", 1),
   (22, "s_scratch_store_dwordx2", 2),
   (23, "s_scratch_store_dwordx4", 4),
   (24, "s_buffer_store_dword", 1),
   (25, "s_buffer_store_dwordx2", 2),
   (26, "s_buffer_store_dwordx4", 4)
]
for (code, name, size) in SMEM_STORE:
   opcode(name, code, Format.SMEM)

SMEM_ATOMIC = [
   (64, "s_buffer_atomic_swap"),
   (66, "s_buffer_atomic_add"),
   (67, "s_buffer_atomic_sub"),
   (68, "s_buffer_atomic_smin"),
   (69, "s_buffer_atomic_umin"),
   (70, "s_buffer_atomic_smax"),
   (71, "s_buffer_atomic_umax"),
   (72, "s_buffer_atomic_and"),
   (73, "s_buffer_atomic_or"),
   (74, "s_buffer_atomic_xor"),
   (75, "s_buffer_atomic_inc"),
   (76, "s_buffer_atomic_dec"),
   (128, "s_atomic_swap"),
   (130, "s_atomic_add"),
   (131, "s_atomic_sub"),
   (132, "s_atomic_smin"),
   (133, "s_atomic_umin"),
   (134, "s_atomic_smax"),
   (135, "s_atomic_umax"),
   (136, "s_atomic_and"),
   (137, "s_atomic_or"),
   (138, "s_atomic_xor"),
   (139, "s_atomic_inc"),
   (140, "s_atomic_dec"),
]
for code, name in SMEM_ATOMIC:
   opcode(name, code, Format.SMEM)

SMEM_ATOMIC_64 = [
   (96, "s_buffer_atomic_swap_x2"),
   (98, "s_buffer_atomic_add_x2"),
   (99, "s_buffer_atomic_sub_x2"),
   (100, "s_buffer_atomic_smin_x2"),
   (101, "s_buffer_atomic_umin_x2"),
   (102, "s_buffer_atomic_smax_x2"),
   (103, "s_buffer_atomic_umax_x2"),
   (104, "s_buffer_atomic_and_x2"),
   (105, "s_buffer_atomic_or_x2"),
   (106, "s_buffer_atomic_xor_x2"),
   (107, "s_buffer_atomic_inc_x2"),
   (108, "s_buffer_atomic_dec_x2"),
   (160, "s_atomic_swap_x2"),
   (162, "s_atomic_add_x2"),
   (163, "s_atomic_sub_x2"),
   (164, "s_atomic_smin_x2"),
   (165, "s_atomic_umin_x2"),
   (166, "s_atomic_smax_x2"),
   (167, "s_atomic_umax_x2"),
   (168, "s_atomic_and_x2"),
   (169, "s_atomic_or_x2"),
   (170, "s_atomic_xor_x2"),
   (171, "s_atomic_inc_x2"),
   (172, "s_atomic_dec_x2"),
]
for code, name in SMEM_ATOMIC_64:
   opcode(name, code, Format.SMEM)

SMEM_DCACHE = [
   (32, "s_dcache_inv"),
   (33, "s_dcache_wb"),
   (34, "s_dcache_inv_vol"),
   (35, "s_dcache_wb_vol")
]
for code, name in SMEM_DCACHE:
   opcode(name, code, Format.SMEM)

SMEM_SPECIAL = [
   (36, "s_memtime"),
   (37, "s_memrealtime"),
   (38, "s_atc_probe"),
   (39, "s_atc_probe_buffer"),
   (40, "s_dcache_discard"),
   (41, "s_dcache_discard_x2"),
   (65, "s_buffer_atomic_cmpswap"),
   (97, "s_buffer_atomic_cmpswap_x2"),
   (129, "s_atomic_cmpswap"),
   (161, "s_atomic_cmpswap_x2"),
]
for code, name in SMEM_SPECIAL:
   opcode(name, code, Format.SMEM)
#opcode("s_memtime", 0, [s2])
#opcode("s_memrealtime", 0, [s2])
#opcode("s_atc_probe", 3, [])
#opcode("s_atc_probe_buffer", 3, [])
#opcode("s_dcache_discard", 3, [])
#opcode("s_dcache_discard_x2", 3, [])
#opcode("s_buffer_atomic_cmpswap", 4, [s1], kills_input = [0, 0, 0, 1])
#opcode("s_buffer_atomic_cmpswap_x2", 4, [s2], kills_input = [0, 0, 0, 1])
#opcode("s_atomic_cmpswap", 4, [s1], kills_input = [0, 0, 0, 1])
#opcode("s_atomic_cmpswap_x2", 4, [s2], kills_input = [0, 0, 0, 1])


# VOP2 instructions: 2 inputs, 1 output (+ optional vcc)
VOP2_NOVCC = {
   (1, "v_add_f32", True),
   (2, "v_sub_f32", True),
   (3, "v_subrev_f32", True),
   (4, "v_mul_legacy_f32", True),
   (5, "v_mul_f32", True),
   (6, "v_mul_i32_i24", False),
   (7, "v_mul_hi_i32_i24", False),
   (8, "v_mul_u32_u24", False),
   (9, "v_mul_hi_u32_u24", False),
   (10, "v_min_f32", True),
   (11, "v_max_f32", True),
   (12, "v_min_i32", False),
   (13, "v_max_i32", False),
   (14, "v_min_u32", False),
   (15, "v_max_u32", False),
   (16, "v_lshrrev_b32", False),
   (17, "v_ashrrev_i32", False),
   (18, "v_lshlrev_b32", False),
   (19, "v_and_b32", False),
   (20, "v_or_b32", False),
   (21, "v_xor_b32", False),
   (31, "v_add_f16", True),
   (32, "v_sub_f16", True),
   (33, "v_subrev_f16", True),
   (34, "v_mul_f16", True),
   (38, "v_add_u16", False),
   (39, "v_sub_u16", False),
   (40, "v_subrev_u16", False),
   (41, "v_mul_lo_u16", False),
   (42, "v_lshlrev_b16", False),
   (43, "v_lshrrev_b16", False),
   (44, "v_ashrrev_b16", False),
   (45, "v_max_f16", True),
   (46, "v_min_f16", True),
   (47, "v_max_u16", False),
   (48, "v_max_i16", False),
   (49, "v_min_u16", False),
   (50, "v_min_i16", False),
   (51, "v_ldexp_f16", False),
   (52, "v_add_u32", False),
   (53, "v_sub_u32", False),
   (54, "v_subrev_u32", False)
}
for code, name, modifiers in VOP2_NOVCC:
   opcode(name, code, Format.VOP2, modifiers, modifiers)

VOP2_LITERAL = {
   (23, "v_madmk_f32"),
   (24, "v_madak_f32"),
   (36, "v_madmk_f16"),
   (37, "v_madak_f16")
}
for code, name in VOP2_LITERAL:
   opcode(name, code, Format.VOP2)

VOP2_VCCOUT = [
   (25, "v_add_co_u32"),
   (26, "v_sub_co_u32"),
   (27, "v_subrev_co_u32")
]
for code, name in VOP2_VCCOUT:
   opcode(name, code, Format.VOP2)

VOP2_VCCINOUT = [
   (28, "v_addc_co_u32"),
   (29, "v_subb_co_u32"),
   (30, "v_subbrev_co_u32")
]
for code, name in VOP2_VCCINOUT:
   opcode(name, code, Format.VOP2)
   
VOP2_SPECIAL = [
   "v_cndmask_b32",
   "v_mac_f32",
   "v_mac_f16"
]
opcode("v_cndmask_b32", 0, Format.VOP2)
opcode("v_mac_f32", 22, Format.VOP2, True, True)
opcode("v_mac_f16", 35, Format.VOP2, True, True)


# VOP1 instructions: instructions with 1 input and 1 output
VOP1_32 = {
   (1, "v_mov_b32", False, False),
   (2, "v_readfirstlane_b32", False, False),
   (3, "v_cvt_i32_f64", True, False),
   (5, "v_cvt_f32_i32", False, True),
   (6, "v_cvt_f32_u32", False, True),
   (7, "v_cvt_u32_f32", True, False),
   (8, "v_cvt_i32_f32", True, False),
   (10, "v_cvt_f16_f32", True, True),
   (11, "v_cvt_f32_f16", True, True),
   (12, "v_cvt_rpi_i32_f32", True, False),
   (13, "v_cvt_flr_i32_f32", True, False),
   (14, "v_cvt_off_f32_i4", False, True),
   (15, "v_cvt_f32_f64", True, True),
   (17, "v_cvt_f32_ubyte0", False, True),
   (18, "v_cvt_f32_ubyte1", False, True),
   (19, "v_cvt_f32_ubyte2", False, True),
   (20, "v_cvt_f32_ubyte3", False, True),
   (21, "v_cvt_u32_f64", True, False),
   (27, "v_fract_f32", True, True),
   (28, "v_trunc_f32", True, True),
   (29, "v_ceil_f32", True, True),
   (30, "v_rndne_f32", True, True),
   (31, "v_floor_f32", True, True),
   (32, "v_exp_f32", True, True),
   (33, "v_log_f32", True, True),
   (34, "v_rcp_f32", True, True),
   (35, "v_rcp_iflag_f32", True, True),
   (36, "v_rsq_f32", True, True),
   (39, "v_sqrt_f32", True, True),
   (41, "v_sin_f32", True, True),
   (42, "v_cos_f32", True, True),
   (43, "v_not_b32", False, False),
   (44, "v_bfrev_b32", False, False),
   (45, "v_ffbh_u32", False, False),
   (46, "v_ffbl_b32", False, False),
   (47, "v_ffbh_i32", False, False),
   (48, "v_frexp_exp_i32_f64", True, False),
   (51, "v_frexp_exp_i32_f32", True, False),
   (52, "v_frexp_mant_f32", True, False),
   (55, "v_screen_partition_4se_b32", False, False),
   (57, "v_cvt_f16_u16", False, True),
   (58, "v_cvt_f16_i16", False, True),
   (59, "v_cvt_u16_f16", True, False),
   (60, "v_cvt_i16_f16", True, False),
   (61, "v_rcp_f16", True, True),
   (62, "v_sqrt_f16", True, True),
   (63, "v_rsq_f16", True, True),
   (64, "v_log_f16", True, True),
   (65, "v_exp_f16", True, True),
   (66, "v_frexp_mant_f16", True, False),
   (67, "v_frexp_exp_i16_f16", True, False),
   (68, "v_floor_f16", True, True),
   (69, "v_ceil_f16", True, True),
   (70, "v_trunc_f16", True, True),
   (71, "v_rndne_f16", True, True),
   (72, "v_fract_f16", True, True),
   (73, "v_sin_f16", True, True),
   (74, "v_cos_f16", True, True),
   (75, "v_exp_legacy_f32", True, True),
   (76, "v_log_legacy_f32", True, True),
   (77, "v_cvt_norm_i16_f16", True, False),
   (78, "v_cvt_norm_u16_f16", True, False),
   (79, "v_sat_pk_u8_i16", False, False)
}
for code, name, in_mod, out_mod in VOP1_32:
   opcode(name, code, Format.VOP1, in_mod, out_mod)

VOP1_64 = [
   (4, "v_cvt_f64_i32", False, True),
   (16, "v_cvt_f64_f32", True, True),
   (22, "v_cvt_f64_u32", False, True),
   (23, "v_trunc_f64", True, True),
   (24, "v_ceil_f64", True, True),
   (25, "v_rndne_f64", True, True),
   (26, "v_floor_f64", True, True),
   (37, "v_rcp_f64", True, True),
   (38, "v_rsq_f64", True, True),
   (40, "v_sqrt_f64", True, True),
   (49, "v_frexp_mant_f64", True, False),
   (50, "v_fract_f64", True, True)
]
for code, name, in_mod, out_mod in VOP1_64:
   opcode(name, code, Format.VOP1, in_mod, out_mod)

VOP1_SPECIAL = [
   "v_nop",
   "v_clrexcp",
   "v_swap_b32"
]
opcode("v_nop", 0, Format.VOP1)
opcode("v_clrexcp", 53, Format.VOP1)
opcode("v_swap_b32", 81, Format.VOP1)


# VOPC instructions:

VOPC_CLASS = {
   (16, "v_cmp_class_f32"),
   (17, "v_cmpx_class_f32"),
   (18, "v_cmp_class_f64"),
   (19, "v_cmpx_class_f64"),
   (20, "v_cmp_class_f16"),
   (21, "v_cmpx_class_f16"),
}
for code, name in VOPC_CLASS:
    opcode(name, code, Format.VOPC, True, False)

PREFIX = ["v_cmp_", "v_cmpx_"]

COMPF = ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"]
SUFFIX_F = ["_f16", "_f32", "_f64"]

code = 32
for post in SUFFIX_F:
   for pre in PREFIX:
      for comp in COMPF:
         opcode(pre+comp+post, code, Format.VOPC, True, False)
         VOPC_GFX6[code] = max(0, code - 0x40)
         code = code + 1
assert(code == 128)

COMPI = ["f", "lt", "eq", "le", "gt", "lg", "ge", "tru"]
SIGNED = ["_i", "_u"]
BITSIZE = ["16", "32", "64"]

code = 160
for bits in BITSIZE:
   for pre in PREFIX:
      for s in SIGNED:
         for cmp in COMPI:
            opcode(pre+cmp+s+bits, code, Format.VOPC)
            if (bits != "16"):
                VOPC_GFX6[code] = code - (0x40 if s == "_i" else 0x8)
            code = code + 1
assert(code == 256)


# VOPP instructions: packed 16bit instructions - 1 or 2 inputs and 1 output

VOPP_2 = [
   "v_pk_mul_lo_u16",
   "v_pk_add_i16",
   "v_pk_sub_i16",
   "v_pk_lshlrev_b16",
   "v_pk_lshrrev_b16",
   "v_pk_ashrrev_i16",
   "v_pk_max_i16",
   "v_pk_min_i16",
   "v_pk_add_u16",
   "v_pk_sub_u16",
   "v_pk_max_u16",
   "v_pk_min_u16",
   "v_pk_add_f16",
   "v_pk_mul_f16",
   "v_pk_min_f16",
   "v_pk_max_f16"
]
#for name in VOPP_2:
#   opcode(name, 2, [v1])

VOPP_3 = [
   "v_pk_mad_i16",
   "v_pk_mad_u16",
   "v_pk_fma_f16",
   "v_pk_mad_mix_f32",
   "v_pk_mad_mixlo_f16",
   "v_pk_mad_mixhi_f16"
]
#for name in VOPP_3:
#   opcode(name, 3, [v1])

VOPP = VOPP_2 + VOPP_3


# VINTERP instructions: 

VINTRP = [
   "v_interp_p1_f32",
   "v_interp_p2_f32",
   "v_interp_mov_f32"
]
opcode("v_interp_p1_f32", 0, Format.VINTRP)
opcode("v_interp_p2_f32", 1, Format.VINTRP)
opcode("v_interp_mov_f32", 2, Format.VINTRP)

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
