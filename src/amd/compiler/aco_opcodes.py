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

class Opcode(object):
   """Class that represents all the information we have about the opcode
   NOTE: this must be kept in sync with aco_op_info
   """
   def __init__(self, name, num_inputs, output_type, code,
                read_reg, write_reg, kills_input):
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
      assert isinstance(num_inputs, int)
      assert isinstance(output_type, list)
      assert isinstance(kills_input, list)
      assert isinstance(code, int)

      num_outputs = len(output_type)
      assert 0 <= num_inputs <= 4
      assert 0 <= num_outputs <= 2
      self.name = name
      self.num_inputs = num_inputs
      self.num_outputs = num_outputs
      self.opcode = code
      self.output_type = output_type
      self.read_reg = read_reg
      self.write_reg = write_reg
      self.kills_input = kills_input

# global dictionary of opcodes
opcodes = {}

def opcode(name, num_inputs, output_type, code = 0,
           read_reg = 0, write_reg = 0, kills_input = []):
   assert name not in opcodes
   if not kills_input:
      kills_input = [0] * num_inputs
   opcodes[name] = Opcode(name, num_inputs, output_type, code,
                          read_reg, write_reg, kills_input)

opcode("exp", 0, [])
opcode("p_parallelcopy", 0, [])
opcode("p_startpgm", 0, [])
opcode("p_phi", 0, [])

opcode("p_create_vector", 0, [])
opcode("p_extract_vector", 0, [])

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
   opcode(name, 2, [s1, b], code, write_reg = SCC)

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
   opcode(name, 2, [s1], code)

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
   opcode(name, 2, [s2,b], code, write_reg = SCC)

SOP2_SPECIAL = {
   (4, "s_addc_u32"),
   (5, "s_subb_u32"),
   (10, "s_cselect_b32"),
   (11, "s_cselect_b64"),
   (35, "s_bfm_b64"),
   (41, "s_cbranch_g_fork"),
   (43, "s_rfe_restore_b64"),
}
opcode("s_addc_u32", 3, [s1,b], 4, read_reg = SCC, write_reg = SCC)
opcode("s_subb_u32", 3, [s1,b], 5, read_reg = SCC, write_reg = SCC)
opcode("s_cselect_b32", 3, [s1], 10, read_reg = SCC)
opcode("s_cselect_b64", 3, [s2], 11, read_reg = SCC)
opcode("s_bfm_b64", 2, [s2], 35)
opcode("s_cbranch_g_fork", 2, [], 41)
opcode("s_rfe_restore_b64", 1, [], 43)

SOP2 = dict(SOP2_SCC).values() + dict(SOP2_NOSCC).values() + dict(SOP2_64).values() + dict(SOP2_SPECIAL).values()


# SOPK instructions: 0 input (+ imm), 1 output + optional scc
SOPK_SCC = [
   "s_cmpk_eq_i32",
   "s_cmpk_lg_i32",
   "s_cmpk_gt_i32",
   "s_cmpk_ge_i32",
   "s_cmpk_lt_i32",
   "s_cmpk_le_i32",
   "s_cmpk_eq_u32",
   "s_cmpk_lg_u32",
   "s_cmpk_gt_u32",
   "s_cmpk_ge_u32",
   "s_cmpk_lt_u32",
   "s_cmpk_le_u32",
]
for name in SOPK_SCC:
   opcode(name, 1, [b], write_reg = SCC)

SOPK_SPECIAL = [
   "s_movk_i32",
   "s_cmovk_i32",
   "s_addk_i32",
   "s_mulk_i32",
   "s_cbranch_i_fork",
   "s_getreg_b32",
   "s_setreg_b32",
   "s_setreg_imm32_b32", # requires 32bit literal
   "s_call_b64"
]
opcode("s_movk_i32", 0, [s1])
opcode("s_cmovk_i32", 1, [s1], read_reg = SCC)
opcode("s_addk_i32", 1, [s1,b], write_reg = SCC, kills_input = [1])
opcode("s_mulk_i32", 1, [s1], kills_input = [1])
opcode("s_cbranch_i_fork", 1, [])
opcode("s_getreg_b32", 0, [s1])
opcode("s_setreg_b32", 1, [])
opcode("s_setreg_imm32_b32", 0, [])
opcode("s_call_b64", 0, [s2])

SOPK = SOPK_SCC + SOPK_SPECIAL


# SOP1 instructions: 1 input, 1 output (+optional SCC)
SOP1_NOSCC = [
   "s_mov_b32",
   "s_brev_b32",
   "s_ff0_i32_b32",
   "s_ff0_i32_b64",
   "s_ff1_i32_b32",
   "s_ff1_i32_b64",
   "s_flbit_i32_b32",
   "s_flbit_i32_b64",
   "s_flbit_i32",
   "s_flbit_i32_i64"
   "s_sext_i32_i8",
   "s_sext_i32_i16",
   "s_bitset0_b32",
   "s_bitset1_b32",
   "s_movrels_b32",
   "s_movreld_b32",
]
for name in SOP1_NOSCC:
   opcode(name, 1, [s1])

SOP1_NOSCC_64 = [
   "s_mov_b64",
   "s_brev_b64",
   "s_bitset0_b64",
   "s_bitset1_b64",
   "s_swappc_b64",
   "s_movrels_b64",
   "s_movreld_b64"
]
for name in SOP1_NOSCC_64:
   opcode(name, 1, [s2])

SOP1_SCC = [
   "s_not_b32",
   "s_wqm_b32",
   "s_bcnt0_i32_b32",
   "s_bcnt1_i32_b32",
   "s_quadmask_b32",
   "s_abs_i32"
]
for name in SOP1_SCC:
   opcode(name, 1, [s1,b], write_reg = SCC)

SOP1_SCC_64 = [
   "s_not_b64",
   "s_wqm_b64",
   "s_and_saveexec_b64",
   "s_or_saveexec_b64",
   "s_xor_saveexec_b64",
   "s_andn2_saveexec_b64",
   "s_orn2_saveexec_b64",
   "s_nand_saveexec_b64",
   "s_nor_saveexec_b64",
   "s_xnor_saveexec_b64",
   "s_quadmask_b64",
   "s_andn1_saveexec_b64",
   "s_orn1_saveexec_b64",
   "s_andn1_wrexec_b64",
   "s_andn2_wrexec_b64",
]
for name in SOP1_SCC_64:
   opcode(name, 1, [s2,b], write_reg = SCC + ", EXEC" if 'exec' in name else "")

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
opcode("s_cmov_b32", 2, [s1], read_reg = SCC)
opcode("s_cmov_b64", 2, [s2], read_reg = SCC)
opcode("s_bcnt0_i32_b64", 1, [s1,b], write_reg = SCC)
opcode("s_bcnt1_i32_b64", 1, [s1,b], write_reg = SCC)
opcode("s_getpc_b64", 0, [s2])
opcode("s_setpc_b64", 1, [])
opcode("s_rfe_b64", 1, [])
opcode("s_cbranch_join", 1, [])
opcode("s_set_gpr_idx_idx", 1, [])
opcode("s_bitreplicate_b64_b32", 1, [s2])

SOP1 = SOP1_NOSCC + SOP1_NOSCC_64 + SOP1_SCC + SOP1_SCC_64 + SOP1_SPECIAL


# SOPC instructions: 2 inputs and 0 outputs (+SCC)
SOPC_SCC = [
   "s_cmp_eq_i32",
   "s_cmp_lg_i32",
   "s_cmp_gt_i32",
   "s_cmp_ge_i32",
   "s_cmp_lt_i32",
   "s_cmp_le_i32",
   "s_cmp_eq_u32",
   "s_cmp_lg_u32",
   "s_cmp_gt_u32",
   "s_cmp_ge_u32",
   "s_cmp_lt_u32",
   "s_cmp_le_u32",
   "s_bitcmp0_b32",
   "s_bitcmp1_b32",
   "s_bitcmp0_b64",
   "s_bitcmp1_b64",
   "s_cmp_eq_u64",
   "s_cmp_lg_u64"
]
for name in SOPC_SCC:
   opcode(name, 2, [b], write_reg = SCC)

SOPC_SPECIAL = [
   "s_setvskip",
   "s_set_gpr_idx_on"
]
opcode("s_setvskip", 2, [])
opcode("s_set_gpr_idx_on", 1, [])

SOPC = SOPC_SCC + SOPC_SPECIAL


# SOPP instructions: 0 inputs (+optional scc/vcc) , 0 outputs
SOPP_SCC = [
   "s_cbranch_scc0",
   "s_cbranch_scc1"
]
for name in SOPP_SCC:
   opcode(name, 1, [], read_reg = SCC)

SOPP_VCC = [
   "s_cbranch_vccz",
   "s_cbranch_vccnz"
]
for name in SOPP_VCC:
   opcode(name, 1, [], read_reg = VCC)

SOPP_SPECIAL = [
   "s_nop",
   "s_endpgm",
   "s_branch",
   "s_wakeup",
   "s_cbranch_execz",
   "s_cbranch_execnz",
   "s_barrier",
   "s_setkill",
   "s_waitcnt",
   "s_sethalt",
   "s_sleep",
   "s_setprio",
   "s_sendmsg",
   "s_sendmsghalt",
   "s_trap",
   "s_icache_inv",
   "s_incperflevel",
   "s_decperflevel",
   "s_ttracedata",
   "s_cbranch_cdbgsys",
   "s_cbranch_cdbguser",
   "s_cbranch_cdbgsys_or_user",
   "s_cbranch_cdbgsys_and_user",
   "s_endpgm_saved",
   "s_set_grp_idx_off",
   "s_set_grp_idx_mode",
   "s_endpgm_ordered_ps_done"
]
for name in SOPP_SPECIAL:
   opcode(name, 0, [])

SOPP = SOPP_SCC + SOPP_VCC + SOPP_SPECIAL


# SMEM instructions: sbase input (2 sgpr), potentially 2 offset inputs, 1 sdata input/output
SMEM_LOAD = [
   ("s_load_dword", s1),
   ("s_load_dwordx2", s2),
   ("s_load_dwordx4", s4),
   ("s_load_dwordx8", s8),
   ("s_load_dwordx16", s16),
   ("s_scratch_load_dword", s1),
   ("s_scratch_load_dwordx2", s2),
   ("s_scratch_load_dwordx4", s4),
   ("s_buffer_load_dword", s1),
   ("s_buffer_load_dwordx2", s2),
   ("s_buffer_load_dwordx4", s4),
   ("s_buffer_load_dwordx8", s8),
   ("s_buffer_load_dwordx16", s16)
]
for (name, size) in SMEM_LOAD:
   opcode(name, 3, [size])

SMEM_STORE = [
   ("s_store_dword", 1),
   ("s_store_dwordx2", 2),
   ("s_store_dwordx4", 4),
   ("s_scratch_store_dword", 1),
   ("s_scratch_store_dwordx2", 2),
   ("s_scratch_store_dwordx4", 4),
   ("s_buffer_store_dword", 1),
   ("s_buffer_store_dwordx2", 2),
   ("s_buffer_store_dwordx4", 4)
]
for (name, size) in SMEM_STORE:
   opcode(name, 4, [])

SMEM_ATOMIC = [
   "s_buffer_atomic_swap",
   "s_buffer_atomic_add",
   "s_buffer_atomic_sub",
   "s_buffer_atomic_smin",
   "s_buffer_atomic_umin",
   "s_buffer_atomic_smax",
   "s_buffer_atomic_umax",
   "s_buffer_atomic_and",
   "s_buffer_atomic_or",
   "s_buffer_atomic_xor",
   "s_buffer_atomic_inc",
   "s_buffer_atomic_dec",
   "s_atomic_swap",
   "s_atomic_add",
   "s_atomic_sub",
   "s_atomic_smin",
   "s_atomic_umin",
   "s_atomic_smax",
   "s_atomic_umax",
   "s_atomic_and",
   "s_atomic_or",
   "s_atomic_xor",
   "s_atomic_inc",
   "s_atomic_dec",
]
for name in SMEM_ATOMIC:
   opcode(name, 4, [s1], kills_input = [0, 0, 0, 1])

SMEM_ATOMIC_64 = [
   "s_buffer_atomic_swap_x2",
   "s_buffer_atomic_add_x2",
   "s_buffer_atomic_sub_x2",
   "s_buffer_atomic_smin_x2",
   "s_buffer_atomic_umin_x2",
   "s_buffer_atomic_smax_x2",
   "s_buffer_atomic_umax_x2",
   "s_buffer_atomic_and_x2",
   "s_buffer_atomic_or_x2",
   "s_buffer_atomic_xor_x2",
   "s_buffer_atomic_inc_x2",
   "s_buffer_atomic_dec_x2",
   "s_atomic_swap_x2",
   "s_atomic_add_x2",
   "s_atomic_sub_x2",
   "s_atomic_smin_x2",
   "s_atomic_umin_x2",
   "s_atomic_smax_x2",
   "s_atomic_umax_x2",
   "s_atomic_and_x2",
   "s_atomic_or_x2",
   "s_atomic_xor_x2",
   "s_atomic_inc_x2",
   "s_atomic_dec_x2",
]
for name in SMEM_ATOMIC_64:
   opcode(name, 4, [s2], kills_input = [0, 0, 0, 1])

SMEM_DCACHE = [
   "s_dcache_inv",
   "s_dcache_wb",
   "s_dcache_inv_vol",
   "s_dcache_wb_vol"
]
for name in SMEM_DCACHE:
   opcode(name, 0, [])

SMEM_SPECIAL = [
   "s_memtime",
   "s_memrealtime",
   "s_atc_probe",
   "s_atc_probe_buffer",
   "s_dcache_discard",
   "s_dcache_discard_x2",
   "s_buffer_atomic_cmpswap",
   "s_buffer_atomic_cmpswap_x2",
   "s_atomic_cmpswap",
   "s_atomic_cmpswap_x2",
]
opcode("s_memtime", 0, [s2])
opcode("s_memrealtime", 0, [s2])
opcode("s_atc_probe", 3, [])
opcode("s_atc_probe_buffer", 3, [])
opcode("s_dcache_discard", 3, [])
opcode("s_dcache_discard_x2", 3, [])
opcode("s_buffer_atomic_cmpswap", 4, [s1], kills_input = [0, 0, 0, 1])
opcode("s_buffer_atomic_cmpswap_x2", 4, [s2], kills_input = [0, 0, 0, 1])
opcode("s_atomic_cmpswap", 4, [s1], kills_input = [0, 0, 0, 1])
opcode("s_atomic_cmpswap_x2", 4, [s2], kills_input = [0, 0, 0, 1])

SMEM = SMEM_LOAD + SMEM_STORE + SMEM_ATOMIC + SMEM_ATOMIC_64 + SMEM_SPECIAL


# VOP2 instructions: 2 inputs, 1 output (+ optional vcc)
VOP2_NOVCC = [
   "v_add_f32",
   "v_sub_f32",
   "v_subrev_f32",
   "v_mul_legacy_f32",
   "v_mul_f32",
   "v_mul_i32_i24",
   "v_mul_hi_i32_i24",
   "v_mul_u32_u24",
   "v_mul_hi_u32_u24",
   "v_min_f32",
   "v_max_f32",
   "v_min_i32",
   "v_max_i32",
   "v_min_u32",
   "v_max_u32",
   "v_lshrrev_b32",
   "v_ashrrev_i32",
   "v_lshlrev_b32",
   "v_and_b32",
   "v_or_b32",
   "v_xor_b32",
   "v_add_f16",
   "v_sub_f16",
   "v_subrev_f16",
   "v_mul_f16",
   "v_add_u16",
   "v_sub_u16",
   "v_subrev_u16",
   "v_mul_lo_u16",
   "v_lshlrev_b16",
   "v_lshrrev_b16",
   "v_ashrrev_b16",
   "v_max_f16",
   "v_min_f16",
   "v_max_u16",
   "v_min_u16",
   "v_max_i16",
   "v_min_i16",
   "v_ldexp_f16",
   "v_add_u32",
   "v_sub_u32",
   "v_subrev_u32"
]
for name in VOP2_NOVCC:
   opcode(name, 2, [v1])

VOP2_LITERAL = [
   "v_madmk_f32",
   "v_madak_f32",
   "v_madmk_f16",
   "v_madak_f16"
]
for name in VOP2_LITERAL:
   opcode(name, 3, [v1])

VOP2_VCCOUT = [
   "v_add_co_u32",
   "v_sub_co_u32",
   "v_subrev_co_u32"
]
for name in VOP2_VCCOUT:
   opcode(name, 2, [v1,s2], write_reg = VCC)

VOP2_VCCINOUT = [
   "v_addc_co_u32",
   "v_subb_co_u32",
   "v_subbrev_co_u32"
]
for name in VOP2_VCCINOUT:
   opcode(name, 3, [v1,s2], read_reg = VCC, write_reg = VCC, kills_input = [0, 0, 1])
   
VOP2_SPECIAL = [
   "v_cndmask_b32",
   "v_mac_f16"
]
opcode("v_cndmask_b32", 3, [v1], read_reg = VCC)
opcode("v_mac_f16", 3, [v1], kills_input = [0, 0, 1])

VOP2 = VOP2_NOVCC + VOP2_LITERAL + VOP2_VCCOUT + VOP2_VCCINOUT + VOP2_SPECIAL


# VOP1 instructions: instructions with 1 input and 1 output
VOP1_32 = [
   "v_mov_b32",
   "v_readfirstlane_b32",
   "v_cvt_i32_f64",
   "v_cvt_f32_i32",
   "v_cvt_f32_u32",
   "v_cvt_u32_f32",
   "v_cvt_i32_f32",
   "v_cvt_f16_f32",
   "v_cvt_f32_f16",
   "v_cvt_rpi_i32_f32",
   "v_cvt_flr_i32_f32",
   "v_cvt_off_f32_i4",
   "v_cvt_f32_f64",
   "v_cvt_f32_ubyte0",
   "v_cvt_f32_ubyte1",
   "v_cvt_f32_ubyte2",
   "v_cvt_f32_ubyte3",
   "v_cvt_u32_f64",
   "v_fract_f32",
   "v_trunc_f32",
   "v_ceil_f32",
   "v_rndne_f32",
   "v_floor_f32",
   "v_exp_f32",
   "v_log_f32",
   "v_rcp_f32",
   "v_rcp_iflag_f32",
   "v_rsq_f32",
   "v_sqrt_f32",
   "v_sin_f32",
   "v_cos_f32",
   "v_not_b32",
   "v_bfrev_b32",
   "v_ffbh_u32",
   "v_ffbl_b32",
   "v_ffbh_i32",
   "v_frexp_exp_i32_f64",
   "v_frexp_exp_i32_f32",
   "v_frexp_mant_f32",
   "v_screen_partition_4se_b32",
   "v_cvt_f16_u16",
   "v_cvt_f16_i16",
   "v_cvt_u16_f16",
   "v_cvt_i16_f16",
   "v_rcp_f16",
   "v_sqrt_f16",
   "v_rsq_f16",
   "v_log_f16",
   "v_exp_f16",
   "v_frexp_mant_f16",
   "v_frexp_exp_i16_f16",
   "v_floor_f16",
   "v_ceil_f16",
   "v_trunc_f16",
   "v_rndne_f16",
   "v_fract_f16",
   "v_sin_f16",
   "v_cos_f16",
   "v_exp_legacy_f32",
   "v_log_legacy_f32",
   "v_cvt_norm_i16_f16",
   "v_cvt_norm_u16_f16",
   "v_sat_pk_u8_i16"
]
for name in VOP1_32:
   opcode(name, 1, [v1])

VOP1_64 = [
   "v_cvt_f64_i32",
   "v_cvt_f64_f32",
   "v_cvt_f64_u32",
   "v_trunc_f64",
   "v_ceil_f64",
   "v_rndne_f64",
   "v_floor_f64",
   "v_rcp_f64",
   "v_rsq_f64",
   "v_sqrt_f64",
   "v_frexp_mant_f64",
   "v_fract_f64"
]
for name in VOP1_64:
   opcode(name, 1, [v2])

VOP1_SPECIAL = [
   "v_nop",
   "v_clrexcp",
   "v_swap_b32"
]
opcode("v_nop", 0, [])
opcode("v_clrexcp", 0, [])
opcode("v_swap_b32", 2, [v1,v1], kills_input = [1,1])

VOP1 = VOP1_32 + VOP1_64 + VOP1_SPECIAL


# VOPC instructions:

COMPF = ["class", "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"]
COMPI = ["f", "lt", "eq", "le", "gt", "lg", "ge", "tru"]

PREFIX = ["v_cmp_", "v_cmpx_"]
SUFFIX_F32 = ["_f16", "_f32"]
SUFFIX_U32 = ["_i16", "_u16", "_i32", "_u32"]
SUFFIX_U64 = ["_i64", "_u64"]

VOPC_32 = [pre+mid+post for pre in PREFIX for mid in COMPF for post in SUFFIX_F32]+[pre+mid+post for pre in PREFIX for mid in COMPI for post in SUFFIX_U32]
for name in VOPC_32:
   opcode(name, 2, [s2], write_reg = VCC + ", EXEC" if 'x' in name else "")

VOPC_64 = [pre+mid+"_f64" for pre in PREFIX for mid in COMPF]+[pre+mid+post for pre in PREFIX for mid in COMPI for post in SUFFIX_U64]
for name in VOPC_64:
   opcode(name, 2, [s2], write_reg = VCC + ", EXEC" if 'x' in name else "")

VOPC = VOPC_32 + VOPC_64


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
for name in VOPP_2:
   opcode(name, 2, [v1])

VOPP_3 = [
   "v_pk_mad_i16",
   "v_pk_mad_u16",
   "v_pk_fma_f16",
   "v_pk_mad_mix_f32",
   "v_pk_mad_mixlo_f16",
   "v_pk_mad_mixhi_f16"
]
for name in VOPP_3:
   opcode(name, 3, [v1])

VOPP = VOPP_2 + VOPP_3


# VINTERP instructions: 

VINTERP = [
   "v_interp_p1_f32",
   "v_interp_p2_f32",
   "v_interp_mov_f32"
]
opcode("v_interp_p1_f32", 1, [v1])
opcode("v_interp_p2_f32", 1, [v1])
opcode("v_interp_mov_f32", 1, [v1])

# VOP3 instructions: 3 inputs, 1 output
# VOP3b instructions: have a unique scalar output, e.g. VOP2 with vcc out

VOP3b = [
   "v_div_scale_f32",
   "v_div_scale_f64",
   "v_mad_u64_u32",
   "v_mad_i64_i32"
]
#TODO opcode("v_mad_u64_u32", 3, [1,1,2], 2, [2,2], [0, 1], 0, 1, 0, 0, [0, 0, 0])


VOP3a_32 = [
   "v_mad_legacy_f32",
   "v_mad_f32",
   "v_mad_i32_i24",
   "v_mad_u32_u24",
   "v_cubeid_f32",
   "v_cubesc_f32",
   "v_cubetc_f32",
   "v_cubema_f32",
   "v_bfe_u32",
   "v_bfe_i32",
   "v_bfi_b32",
   "v_fma_f32",
   "v_lerp_u8",
   "v_alignbit_b32",
   "v_alignbyte_b32",
   "v_min3_f32",
   "v_min3_i32",
   "v_min3_u32",
   "v_max3_f32",
   "v_max3_i32",
   "v_max3_u32",
   "v_med3_f32",
   "v_med3_i32",
   "v_med3_u32",
   "v_sad_u8",
   "v_sad_hi_u8",
   "v_sad_u16",
   "v_sad_u32",
   "v_cvt_pk_u8_f32",
   "v_div_fixup_f32",
   "v_msad_u8",
   "v_mad_legacy_f16",
   "v_mad_legacy_u16",
   "v_mad_legacy_i16",
   "v_perm_b32",
   "v_fma_legacy_f16",
   "v_div_fixup_legacy_f16",
   "v_mad_u32_u16",
   "v_mad_i32_i16",
   "v_xad_u32",
   "v_min3_f16",
   "v_min3_i16",
   "v_min3_u16",
   "v_max3_f16",
   "v_max3_i16",
   "v_max3_u16",
   "v_med3_f16",
   "v_med3_i16",
   "v_med3_u16",
   "v_lshl_add_u32",
   "v_add_lshl_u32",
   "v_add3_u32",
   "v_lshl_or_b32",
   "v_and_or_b32",
   "v_or3_b32",
   "v_mad_f16",
   "v_mad_u16",
   "v_mad_i16",
   "v_fma_f16",
   "v_div_fixup_f16"
]
for name in VOP3a_32:
   opcode(name, 3, [v1])

VOP3a_64 = [
   "v_fma_f64",
   "v_div_fixup_f64"
]
for name in VOP3a_64:
   opcode(name, 3, [v2])

#two parameters
VOP3a_32_2 = [
   "v_cvt_pkaccum_u8_f32",
   "v_mul_lo_u32",
   "v_mul_hi_u32",
   "v_mul_hi_i32",
   "v_ldexp_f32",
   "v_writelane_b32",
   "v_mbcnt_lo_u32_b32",
   "v_mbcnt_hi_u32_b32",
   "v_bfm_b32",
   "v_cvt_pknorm_i16_f32",
   "v_cvt_pknorm_u16_f32",
   "v_cvt_pkrtz_f16_f32",
   "v_cvt_pk_u16_u32",
   "v_cvt_pk_i16_i32",
   "v_cvt_pknorm_i16_f16",
   "v_cvt_pknorm_u16_f16",
   "v_add_i32",
   "v_sub_i32",
   "v_add_i16",
   "v_sub_i16",
   "v_pack_b32_f16"
]
for name in VOP3a_32_2:
   opcode(name, 2, [v1])

VOP3a_64_2 = [
   "v_add_f64",
   "v_mul_f64",
   "v_min_f64",
   "v_max_f64",
   "v_ldexp_f64",
   "v_lshlrev_b64",
   "v_lshrrev_b64", #64bit
   "v_ashrrev_i64", #64bit
   "v_trig_preop_f64" #64bit
]
for name in VOP3a_64_2:
   opcode(name, 2, [v2])
   
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
opcode("v_bcnt_u32_b32", 1, [v1])
opcode("v_readlane_b32", 2, [s1])
opcode("v_div_fmas_f32", 4, [v1], read_reg = VCC)
opcode("v_div_fmas_f64", 4, [v2], read_reg = VCC)
opcode("v_qsad_pk_u16_u8", 3, [v1])
opcode("v_mqsad_pk_u16_u8", 3, [v1])
opcode("v_mqsad_u32_u8", 3, [v2])
# TODO

VOP3a = VOP3a_32 + VOP3a_64 + VOP3a_32_2 + VOP3a_64_2 + VOP3a_SPECIAL


# DS instructions: 3 inputs (1 addr, 2 data), 1 output
#TODO
DS_1 = [
   "ds_add_u32",
   "ds_sub_u32",
   "ds_rsub_u32",
   "ds_inc_u32",
   "ds_dec_u32",
   "ds_min_i32",
   "ds_max_i32",
   "ds_min_u32",
   "ds_max_u32",
   "ds_and_b32",
   "ds_or_b32",
   "ds_xor_b32",
   "ds_add_f32",
   "ds_add_rtn_u32",
   "ds_sub_rtn_u32",
   "ds_rsub_rtn_u32",
   "ds_inc_rtn_u32",
   "ds_dec_rtn_u32",
   "ds_min_rtn_i32",
   "ds_max_rtn_i32",
   "ds_min_rtn_u32",
   "ds_max_rtn_u32",
   "ds_and_rtn_b32",
   "ds_or_rtn_b32",
   "ds_xor_rtn_b32",
   "ds_wrxchg_rtn_b32",
   "ds_add_rtn_f32",
   "ds_permute_b32",
   "ds_bpermute_b32",

]

DS_2 = [
   "ds_mskor_b32",
   "ds_cmpst_b32",
   "ds_cmpst_f32",
   "ds_min_f32",
   "ds_max_f32",
   "ds_mskor_rtn_b32",
   "ds_wrxchg2_rtn_b32",
   "ds_wrxchg2st64_rtn_b32",
   "ds_cmpst_rtn_b32",
   "ds_cmpst_rtn_f32",
   "ds_min_rtn_f32",
   "ds_max_rtn_f32",
   "ds_wrap_rtn_b32",

   
]

DS_64_1 = [
   "ds_add_u64",
   "ds_sub_u64",
   "ds_rsub_u64",
   "ds_inc_u64",
   "ds_dec_u64",
   "ds_min_i64",
   "ds_max_i64",
   "ds_min_u64",
   "ds_max_u64",
   "ds_and_b64",
   "ds_or_b64",
   "ds_xor_b64",
   "ds_add_rtn_u64",
   "ds_sub_rtn_u64",
   "ds_rsub_rtn_u64",
   "ds_inc_rtn_u64",
   "ds_dec_rtn_u64",
   "ds_min_rtn_i64",
   "ds_max_rtn_i64",
   "ds_min_rtn_u64",
   "ds_max_rtn_u64",
   "ds_and_rtn_b64",
   "ds_or_rtn_b64",
   "ds_xor_rtn_b64",
   "ds_wrxchg_rtn_b64",

]

DS_64_2 = [
   "ds_mskor_b64",
   "ds_cmpst_b64",
   "ds_cmpst_f64",
   "ds_min_f64",
   "ds_max_f64",
   "ds_mskor_rtn_b64",
   "ds_wrxchg2_rtn_b64",
   "ds_wrxchg2st64_rtn_b64",
   "ds_cmpst_rtn_b64",
   "ds_cmpst_rtn_f64",
   "ds_min_rtn_f64",
   "ds_max_rtn_f64",
]

DS_WRITE1 = [
   "ds_write_b32",
   "ds_write_addtid_b32",
   "ds_write_b8",
   "ds_write_b16",
   "ds_write_b8_d16_hi",
   "ds_write_b16_d16_hi",
   "ds_write_b64",
   "ds_write_b96",
   "ds_write_b128"
]

DS_WRITE2 = [
   "ds_write2_b32",
   "ds_write2st64_b32",
   "ds_write2_b64",
   "ds_write2st64_b64"
]

DS_VARIOUS = [
   "ds_nop",
   "ds_read_b32",
   "ds_read2_b32",
   "ds_read2st64_b32",
   "ds_read_i8",
   "ds_read_u8",
   "ds_read_i16",
   "ds_read_u16",
   "ds_swizzle_b32", #data1 & offset, no addr/data2
   "ds_read_u8_d16",
   "ds_read_u8_d16_hi",
   "ds_read_i8_d16",
   "ds_read_i8_d16_hi",
   "ds_read_u16_d16",
   "ds_read_u16_d16_hi",
   "ds_read_b64",
   "ds_read2_b64",
   "ds_read2st64_b64",
   "ds_condxchg32_rtn_b64",
   "ds_read_addtid_b32",
   "ds_consume",
   "ds_append",
   "ds_ordered_count",
   "ds_read_b96",
   "ds_read_b128"
]

DS_0 = [ # 0 input, 0 output
   "ds_add_src2_u32",
   "ds_sub_src2_u32",
   "ds_rsub_src2_u32",
   "ds_inc_src2_u32",
   "ds_dec_src2_u32",
   "ds_min_src2_i32",
   "ds_max_src2_i32",
   "ds_min_src2_u32",
   "ds_max_src2_u32",
   "ds_and_src2_b32",
   "ds_or_src2_b32",
   "ds_xor_src2_b32",
   "ds_write_src2_b32",
   "ds_min_src2_f32",
   "ds_max_src2_f32",
   "ds_add_src2_f32"
   "ds_add_src2_u64",
   "ds_sub_src2_u64",
   "ds_rsub_src2_u64",
   "ds_inc_src2_u64",
   "ds_dec_src2_u64",
   "ds_min_src2_i64",
   "ds_max_src2_i64",
   "ds_min_src2_u64",
   "ds_max_src2_u64",
   "ds_and_src2_b64",
   "ds_or_src2_b64",
   "ds_xor_src2_b64",
   "ds_write_src2_b64",
   "ds_min_src2_f64",
   "ds_max_src2_f64",
]

DS_GWS = [
   "ds_gws_sema_release_all",
   "ds_gws_init",
   "ds_gws_sema_v",
   "ds_gws_sema_br",
   "ds_gws_sema_p",
   "ds_gws_barrier"
]


# MUBUF instructions: TODO

MUBUF = [
   "buffer_load_format_x",
   "buffer_load_format_xy",
   "buffer_load_format_xyz",
   "buffer_load_format_xyzw",
   "buffer_store_format_x",
   "buffer_store_format_xy",
   "buffer_store_format_xyz",
   "buffer_store_format_xyzw",
   "buffer_load_format_d16_x",
   "buffer_load_format_d16_xy",
   "buffer_load_format_d16_xyz",
   "buffer_load_format_d16_xyzw",
   "buffer_store_format_d16_x",
   "buffer_store_format_d16_xy",
   "buffer_store_format_d16_xyz",
   "buffer_store_format_d16_xyzw",
   "buffer_load_ubyte",
   "buffer_load_sbyte",
   "buffer_load_ushort",
   "buffer_load_sshort",
   "buffer_load_dword",
   "buffer_load_dwordx2",
   "buffer_load_dwordx3",
   "buffer_load_dwordx4",
   "buffer_store_byte",
   "buffer_store_byte_d16_hi",
   "buffer_store_short",
   "buffer_store_short_d16_hi",
   "buffer_store_dword",
   "buffer_store_dwordx2",
   "buffer_store_dwordx3",
   "buffer_store_dwordx4",
   "buffer_load_ubyte_d16",
   "buffer_load_ubyte_d16_hi",
   "buffer_load_sbyte_d16",
   "buffer_load_sbyte_d16_hi",
   "buffer_load_short_d16",
   "buffer_load_short_d16_hi",
   "buffer_load_format_d16_hi_x",
   "buffer_store_format_d16_hi_x",
   "buffer_store_lds_dword",
   "buffer_wbinvl1",
   "buffer_wbinvl1_vol"
]

MUBUF_ATOMIC = [
   "buffer_atomic_swap",
   "buffer_atomic_cmpswap",
   "buffer_atomic_add",
   "buffer_atomic_sub",
   "buffer_atomic_smin",
   "buffer_atomic_umin",
   "buffer_atomic_smax",
   "buffer_atomic_umax",
   "buffer_atomic_and",
   "buffer_atomic_or",
   "buffer_atomic_xor",
   "buffer_atomic_inc",
   "buffer_atomic_dec",
   "buffer_atomic_swap_x2",
   "buffer_atomic_cmpswap_x2",
   "buffer_atomic_add_x2",
   "buffer_atomic_sub_x2",
   "buffer_atomic_smin_x2",
   "buffer_atomic_umin_x2",
   "buffer_atomic_smax_x2",
   "buffer_atomic_umax_x2",
   "buffer_atomic_and_x2",
   "buffer_atomic_or_x2",
   "buffer_atomic_xor_x2",
   "buffer_atomic_inc_x2",
   "buffer_atomic_dec_x2"
]

MIMG = [
   "image_load",
   "image_load_mip",
   "image_load_pck",
   "image_load_pck_sgn",
   "image_load_mip_pck",
   "image_load_mip_pck_sgn",
   "image_store",
   "image_store_mip",
   "image_store_pck",
   "image_store_mip_pck",
   "image_get_resinfo",
   "image_atomic_swap",
   "image_atomic_cmpswap",
   "image_atomic_add",
   "image_atomic_sub",
   "image_atomic_smin",
   "image_atomic_umin",
   "image_atomic_smax",
   "image_atomic_umax",
   "image_atomic_and",
   "image_atomic_or",
   "image_atomic_xor",
   "image_atomic_inc",
   "image_atomic_dec",
   "image_sample",
   "image_sample_cl",
   "image_sample_d",
   "image_sample_d_cl",
   "image_sample_l",
   "image_sample_b",
   "image_sample_b_cl",
   "image_sample_lz",
   "image_sample_c",
   "image_sample_c_cl",
   "image_sample_c_d",
   "image_sample_c_d_cl",
   "image_sample_c_l",
   "image_sample_c_b",
   "image_sample_c_b_cl",
   "image_sample_c_lz",
   "image_sample_o",
   "image_sample_cl_o",
   "image_sample_d_o",
   "image_sample_d_cl_o",
   "image_sample_l_o",
   "image_sample_b_o",
   "image_sample_b_cl_o",
   "image_sample_lz_o",
   "image_sample_c_o",
   "image_sample_c_cl_o",
   "image_sample_c_d_o",
   "image_sample_c_d_cl_o",
   "image_sample_c_l_o",
   "image_sample_c_b_o",
   "image_sample_c_b_cl_o"
   "image_sample_c_lz_o",
   "image_gather4",
   "image_gather4_cl",
   "image_gather4h",
   "image_gather4_l",
   "image_gather4_b",
   "image_gather4_b_cl",
   "image_gather4_lz",
   "image_gather4_c",
   "image_gather4_c_cl",
   "image_gather4h_pck",
   "image_gather8h_pck",
   "image_gather4_c_l",
   "image_gather4_c_b",
   "image_gather4_c_b_cl",
   "image_gather4_c_lz",
   "image_gather4_o",
   "image_gather4_cl_o",
   "image_gather4_l_o",
   "image_gather4_b_o",
   "image_gather4_b_cl_o",
   "image_gather4_lz_o",
   "image_gather4_c_o",
   "image_gather4_c_cl_o",
   "image_gather4_c_l_o",
   "image_gather4_c_b_o",
   "image_gather4_c_b_cl_o",
   "image_gather4_c_lz_o",
   "image_get_lod",
   "image_sample_cd",
   "image_sample_cd_cl",
   "image_sample_c_cd",
   "image_sample_c_cd_cl",
   "image_sample_cd_o",
   "image_sample_cd_cl_o",
   "image_sample_c_cd_o",
   "image_sample_c_cd_cl_o"
]

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

VOP2_DPP = [ x for x in VOP2 if x not in NOT_DPP ]
VOP1_DPP = [ x for x in VOP1 if x not in NOT_DPP ]
VOPC_DPP = [ x for x in VOPC if x not in VOPC_64 ]