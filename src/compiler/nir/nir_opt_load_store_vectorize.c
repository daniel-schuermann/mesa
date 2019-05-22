/*
 * Copyright © 2019 Valve Corporation
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

/**
 * Although it's called a load/store "vectorization" pass, this also combines
 * intersecting and identical loads/stores. It currently supports derefs, ubo,
 * ssbo and push constant loads/stores.
 *
 * This doesn't handle copy_deref intrinsics and assumes that
 * nir_lower_alu_to_scalar() has been called and that the IR is free from ALU
 * modifiers.
 *
 * After vectorization, the backend may want to call nir_lower_alu_to_scalar()
 * and nir_lower_pack(). Also this creates cast instructions taking derefs as a
 * source and some parts of NIR may not be able to handle that well.
 *
 * There are a few situations where this doesn't vectorize as well as it could:
 * - It won't turn four consecutive vec3 loads into 3 vec4 loads.
 * - If it can't move the first store to the second, it doesn't try moving the
 *   second to the first.
 * - If it can't move the second load to the first, it doesn't try moving the
 *   first to the second.
 * - It doesn't do global vectorization.
 * Handling these cases probably wouldn't provide much benefit though.
*/

#include "nir.h"
#include "nir_deref.h"
#include "nir_builder.h"
#include "nir_worklist.h"

struct intrinsic_info {
   nir_variable_mode mode; /* 0 if the mode is obtained from the deref. */
   nir_intrinsic_op op;
   bool is_atomic;
   /* Indices into nir_intrinsic::src[] or -1 if not applicable. */
   int resource_src; /* resource (e.g. from vulkan_resource_index) */
   int base_src; /* offset which it loads/stores from */
   int deref_src; /* deref which is loads/stores from */
   int value_src; /* the data it is storing */
};

static const struct intrinsic_info *
get_info(nir_intrinsic_op op) {
   switch (op) {
#define INFO(mode, op, atomic, res, base, deref, val) \
case nir_intrinsic_##op: {\
   static const struct intrinsic_info op##_info = {mode, nir_intrinsic_##op, atomic, res, base, deref, val};\
   return &op##_info;\
}
#define LOAD(mode, op, res, base, deref, val) INFO(mode, load_##op, false, res, base, deref, -1)
#define STORE(mode, op, res, base, deref, val) INFO(mode, store_##op, false, res, base, deref, val)
#define ATOMIC(mode, type, op, res, base, deref, val) INFO(mode, type##_atomic_##op, true, res, base, deref, val)
   LOAD(nir_var_mem_push_const, push_constant, -1, 0, -1, -1)
   LOAD(nir_var_mem_ubo, ubo, 0, 1, -1, -1)
   LOAD(nir_var_mem_ssbo, ssbo, 0, 1, -1, -1)
   STORE(nir_var_mem_ssbo, ssbo, 1, 2, -1, 0)
   LOAD(0, deref, -1, -1, 0, -1)
   STORE(0, deref, -1, -1, 0, 1)
   ATOMIC(nir_var_mem_ssbo, ssbo, add, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, imin, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, umin, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, imax, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, umax, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, and, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, or, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, xor, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, exchange, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, comp_swap, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, fadd, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, fmin, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, fmax, 0, 1, -1, 2)
   ATOMIC(nir_var_mem_ssbo, ssbo, fcomp_swap, 0, 1, -1, 2)
   ATOMIC(0, deref, add, -1, -1, 0, 1)
   ATOMIC(0, deref, imin, -1, -1, 0, 1)
   ATOMIC(0, deref, umin, -1, -1, 0, 1)
   ATOMIC(0, deref, imax, -1, -1, 0, 1)
   ATOMIC(0, deref, umax, -1, -1, 0, 1)
   ATOMIC(0, deref, and, -1, -1, 0, 1)
   ATOMIC(0, deref, or, -1, -1, 0, 1)
   ATOMIC(0, deref, xor, -1, -1, 0, 1)
   ATOMIC(0, deref, exchange, -1, -1, 0, 1)
   ATOMIC(0, deref, comp_swap, -1, -1, 0, 1)
   ATOMIC(0, deref, fadd, -1, -1, 0, 1)
   ATOMIC(0, deref, fmin, -1, -1, 0, 1)
   ATOMIC(0, deref, fmax, -1, -1, 0, 1)
   ATOMIC(0, deref, fcomp_swap, -1, -1, 0, 1)
   default:
      break;
#undef ATOMIC
#undef STORE
#undef LOAD
#undef INFO
   }
   return NULL;
}

/*
 * Information used to compare memory operations.
 * It canonically represents an offset as:
 * `offset_base + offset_defs[0]*offset_defs_mul[0] + offset_defs[1]*offset_defs_mul[1] + ...`
 * "offset_defs" is sorted in ascenting order by the ssa definition's index.
 * "resource" or "var" may be NULL.
 */
struct entry_key {
   nir_ssa_def *resource;
   nir_variable *var;
   uint32_t offset_base;
   unsigned offset_def_count;
   nir_ssa_def **offset_defs;
   uint32_t *offset_defs_mul;
};

/* Information on a single memory operation. */
struct entry {
   struct list_head head;
   bool behind_barrier;
   bool can_move;

   struct entry_key key;

   nir_instr *instr;
   nir_intrinsic_instr *intrin;
   const struct intrinsic_info *info;
   enum gl_access_qualifier access;

   nir_ssa_def *store_value;

   nir_deref_instr *deref;
};

struct block_state {
   struct list_head entries[nir_num_variable_modes];
};

struct vectorize_ctx {
   nir_variable_mode modes;
   int (*type_size)(nir_variable_mode, const struct glsl_type *);
   int (*align)(nir_variable_mode, bool, unsigned, unsigned);
   struct block_state *block_states;
};

static unsigned
get_bit_size(struct entry *entry)
{
   return entry->store_value ? entry->store_value->bit_size : entry->intrin->dest.ssa.bit_size;
}

/* If "def" is from an alu instruction with the opcode "op" and one of it's
 * sources is a constant, update "def" to be the non-constant source, fill "c"
 * with the constant and return true. */
static bool
parse_alu(nir_ssa_def **def, nir_op op, uint32_t *c)
{
   nir_src src = nir_src_for_ssa(*def);
   nir_alu_instr *alu = nir_src_as_alu_instr(src);
   if (!alu || alu->op != op || alu->dest.dest.ssa.num_components != 1)
      return false;

   if (alu->src[0].swizzle[0] != 0 || alu->src[1].swizzle[0] != 0)
      return false;

   for (unsigned i = op == nir_op_ishl ? 1 : 0; i < 2; i++) {
      nir_const_value *cv = nir_src_as_const_value(alu->src[i].src);
      if (cv) {
         *c = cv->u32;
         *def = alu->src[!i].src.ssa;
         return true;
      }
   }

   return false;
}

/* Parses an offset expression such as "a * 16 + 4" and "(a * 16 + 4) * 64 + 32". */
static void
parse_offset(nir_ssa_def **base, uint32_t *base_mul, uint32_t *offset)
{
   nir_const_value *const_offset = nir_src_as_const_value(nir_src_for_ssa(*base));
   if (const_offset) {
      *base = NULL;
      *offset += const_offset->u32;
      return;
   }

   uint32_t mul = 1;
   uint32_t add = 0;
   bool progress = false;
   do {
      uint32_t mul2 = 1, add2 = 0;

      progress = parse_alu(base, nir_op_imul, &mul2);
      mul *= mul2;

      mul2 = 0;
      progress |= parse_alu(base, nir_op_ishl, &mul2);
      mul *= 1 << mul2;

      progress |= parse_alu(base, nir_op_iadd, &add2);
      add += add2 * mul;
   } while (progress);

   *base_mul = mul;
   *offset = add;
}

static unsigned
type_scalar_size_bytes(const struct glsl_type *type)
{
   assert(glsl_type_is_vector_or_scalar(type) ||
          glsl_type_is_matrix(type));
   return glsl_type_is_boolean(type) ? 4u : glsl_get_bit_size(type) / 8u;
}

static int
array_stride(struct vectorize_ctx *ctx, nir_variable_mode mode, const struct glsl_type *type)
{
   unsigned explicit_stride = glsl_get_explicit_stride(type);
   if (explicit_stride || (mode & (nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_global))) {
      if ((glsl_type_is_matrix(type) &&
           glsl_matrix_type_is_row_major(type)) ||
          (glsl_type_is_vector(type) && explicit_stride == 0))
         return type_scalar_size_bytes(type);
      return explicit_stride;
   }

   return ctx->type_size(mode, glsl_get_array_element(type));
}

static int
struct_field_offset(struct vectorize_ctx *ctx, nir_variable_mode mode, const struct glsl_type *type, unsigned field)
{
   int explicit_offset = glsl_get_struct_field_offset(type, field);
   if (explicit_offset >= 0 || (mode & (nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_global))) {
      /* taken from nir_lower_explicit_io() */
      return explicit_offset;
   }

   unsigned offset = 0;
   for (unsigned i = 0; i < field; i++)
      offset += ctx->type_size(mode, glsl_get_struct_field(type, i));
   return offset;
}

static struct entry_key
create_entry_key_from_deref(void *mem_ctx, struct vectorize_ctx *ctx, nir_deref_path *path)
{
   unsigned path_len = 0;
   for (; path->path[path_len]; path_len++) ;

   nir_ssa_def *offset_defs_stack[32];
   uint32_t offset_defs_mul_stack[32];
   nir_ssa_def **offset_defs = offset_defs_stack;
   uint32_t *offset_defs_mul = offset_defs_mul_stack;
   if (path_len > 32) {
      offset_defs = malloc(path_len * sizeof(nir_ssa_def *));
      offset_defs_mul = malloc(path_len * sizeof(uint32_t));
   }
   unsigned offset_def_count = 0;

   struct entry_key key;
   key.resource = NULL;
   key.var = NULL;
   key.offset_base = 0;

   for (unsigned i = 0; i < path_len; i++) {
      nir_deref_instr *parent = i ? path->path[i - 1] : NULL;
      nir_deref_instr *deref = path->path[i];

      switch (deref->deref_type) {
      case nir_deref_type_var: {
         assert(!parent);
         key.var = deref->var;
         break;
      }
      case nir_deref_type_array: {
         assert(parent);
         nir_ssa_def *index = deref->arr.index.ssa;
         uint32_t stride = array_stride(ctx, deref->mode, parent->type);

         nir_ssa_def *base = index;
         uint32_t offset = 0, base_mul = 1;
         parse_offset(&base, &base_mul, &offset);

         key.offset_base += offset * stride;
         if (base) {
            for (unsigned j = 0; j <= offset_def_count; j++) {
               if (j == offset_def_count || base->index > offset_defs[j]->index) {
                  /* insert before j */
                  memmove(offset_defs + j + 1, offset_defs + j, (offset_def_count - j) * sizeof(nir_ssa_def *));
                  memmove(offset_defs_mul + j + 1, offset_defs_mul + j, (offset_def_count - j) * sizeof(uint32_t));
                  offset_defs[j] = base;
                  offset_defs_mul[j] = base_mul * stride;
                  offset_def_count++;
                  break;
               } else if (base->index == offset_defs[j]->index) {
                  /* merge with offset_def at i */
                  offset_defs_mul[j] += base_mul * stride;
                  break;
               }
            }
         }
         break;
      }
      case nir_deref_type_struct: {
         assert(parent);
         int offset = struct_field_offset(ctx, deref->mode, parent->type, deref->strct.index);
         key.offset_base += offset;
         break;
      }
      case nir_deref_type_cast: {
         if (!parent)
            key.resource = deref->parent.ssa;
         break;
      }
      default:
         unreachable("Unhandled deref type");
      }
   }

   key.offset_def_count = offset_def_count;
   key.offset_defs = ralloc_array(mem_ctx, nir_ssa_def *, offset_def_count);
   key.offset_defs_mul = ralloc_array(mem_ctx, uint32_t, offset_def_count);
   memcpy(key.offset_defs, offset_defs, offset_def_count * sizeof(nir_ssa_def *));
   memcpy(key.offset_defs_mul, offset_defs_mul, offset_def_count * sizeof(uint32_t));

   if (offset_defs != offset_defs_stack)
      free(offset_defs);
   if (offset_defs_mul != offset_defs_mul_stack)
      free(offset_defs_mul);

   return key;
}

static struct entry_key
create_entry_key_from_offset(void *mem_ctx, uint32_t offset, nir_ssa_def *base, uint32_t base_mul)
{
   struct entry_key key;
   key.resource = NULL;
   key.var = NULL;
   key.offset_base = offset;
   if (base) {
      key.offset_def_count = 1;
      key.offset_defs = ralloc_array(mem_ctx, nir_ssa_def *, 1);
      key.offset_defs_mul = ralloc_array(mem_ctx, uint32_t, 1);
      key.offset_defs[0] = base;
      key.offset_defs_mul[0] = base_mul;
   } else {
      key.offset_def_count = 0;
      key.offset_defs = NULL;
      key.offset_defs_mul = NULL;
   }
   return key;
}

static nir_variable_mode
get_variable_mode(struct entry *entry)
{
   if (entry->info->mode)
      return entry->info->mode;
   assert(entry->deref);
   return entry->deref->mode;
}

static struct entry *
create_entry(struct vectorize_ctx *ctx, const struct intrinsic_info *info, nir_intrinsic_instr *intrin)
{
   struct entry *entry = rzalloc(ctx->block_states, struct entry);
   entry->can_move = true;
   entry->intrin = intrin;
   entry->instr = &intrin->instr;
   entry->info = info;
   if (entry->info->value_src >= 0)
      entry->store_value = intrin->src[entry->info->value_src].ssa;

   if (entry->info->deref_src >= 0) {
      entry->deref = nir_src_as_deref(intrin->src[entry->info->deref_src]);
      nir_deref_path path;
      nir_deref_path_init(&path, entry->deref, NULL);
      entry->key = create_entry_key_from_deref(entry, ctx, &path);
      nir_deref_path_finish(&path);
   } else {
      nir_ssa_def *base = entry->info->base_src >= 0 ? intrin->src[entry->info->base_src].ssa : NULL;
      uint32_t offset = 0, base_mul = 1;
      if (base)
         parse_offset(&base, &base_mul, &offset);
      if (nir_intrinsic_infos[intrin->intrinsic].index_map[NIR_INTRINSIC_BASE])
         offset += nir_intrinsic_base(intrin);
      entry->key = create_entry_key_from_offset(entry, offset, base, base_mul);
   }

   if (entry->info->resource_src >= 0)
      entry->key.resource = intrin->src[entry->info->resource_src].ssa;

   if (nir_intrinsic_infos[intrin->intrinsic].index_map[NIR_INTRINSIC_ACCESS])
      entry->access = nir_intrinsic_access(intrin);
   else if (entry->key.var)
      entry->access = entry->key.var->data.image.access;

   uint32_t restrict_modes = nir_var_shader_in | nir_var_shader_out;
   restrict_modes |= nir_var_shader_temp | nir_var_function_temp;
   restrict_modes |= nir_var_uniform | nir_var_mem_push_const;
   restrict_modes |= nir_var_system_value | nir_var_mem_shared;
   if (get_variable_mode(entry) & restrict_modes)
      entry->access |= ACCESS_RESTRICT;

   return entry;
}

static bool
is_entry_aligned(struct vectorize_ctx *ctx, struct entry *entry, unsigned needed_alignment)
{
   nir_intrinsic_op intrinsic_op = entry->intrin->intrinsic;

   if (nir_intrinsic_infos[intrinsic_op].index_map[NIR_INTRINSIC_ALIGN_MUL] &&
       nir_intrinsic_infos[intrinsic_op].index_map[NIR_INTRINSIC_ALIGN_OFFSET]) {
      if (nir_intrinsic_align(entry->intrin) % needed_alignment == 0)
         return true;
   }

   for (unsigned i = 0; i < entry->key.offset_def_count; i++) {
      if (entry->key.offset_defs_mul[i] % needed_alignment != 0)
         return false;
   }
   return entry->key.offset_base % needed_alignment == 0;
}

struct schedule_ssa_def_state {
   nir_instr_worklist *list;
   nir_instr *use_instr;
};

static bool
is_instr_before(nir_instr *def, nir_instr *use)
{
   if (def->block == use->block)
      return def->index < use->index;

   return nir_block_dominates(def->block, use->block);
}

static bool
add_src_to_worklist(nir_src *src, void *void_state)
{
   struct schedule_ssa_def_state *state = void_state;
   nir_instr *instr = src->ssa->parent_instr;
   assert(instr != state->use_instr);

   if (!is_instr_before(instr, state->use_instr))
      nir_instr_worklist_push_tail(state->list, instr);

   return true;
}

static bool
can_move(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_deref:
   case nir_instr_type_load_const:
   case nir_instr_type_ssa_undef:
   case nir_instr_type_parallel_copy:
      return true;
   default:
      return false;
   }
}

/* Ensure "def" is defined before "use_instr".
 * Assumes "def" doesn't depend on "use_instr". */
static bool
schedule_ssa_def(nir_ssa_def *def, nir_instr *use_instr)
{
   nir_instr *def_instr = def->parent_instr;

   unsigned index = 0;
   nir_foreach_instr(instr, use_instr->block)
      instr->index = index++;

   if (is_instr_before(def_instr, use_instr))
      return true;

   nir_instr_worklist *list = nir_instr_worklist_create();
   nir_instr_worklist_push_tail(list, def_instr);

   nir_cursor cursor = nir_before_instr(use_instr);
   nir_foreach_instr_in_worklist(instr, list) {
      if (!can_move(instr)) {
         nir_instr_worklist_destroy(list);
         return false;
      }

      nir_instr_remove(instr);
      nir_instr_insert(cursor, instr);
      cursor = nir_before_instr(instr);

      /* not correct but should work fine */
      instr->index = nir_instr_next(instr)->index;

      struct schedule_ssa_def_state state;
      state.list = list;
      state.use_instr = instr;
      nir_foreach_src(instr, &add_src_to_worklist, &state);
   }

   nir_instr_worklist_destroy(list);

   return true;
}

static nir_deref_instr *
strip_deref_casts(nir_deref_instr *deref)
{
   while (deref->deref_type == nir_deref_type_cast)
      deref = nir_src_as_deref(deref->parent);
   return deref;
}

static nir_deref_instr *
cast_deref(nir_builder *b, unsigned num_components, unsigned bit_size, nir_src src)
{
   enum glsl_base_type types[] = {
      GLSL_TYPE_UINT8, GLSL_TYPE_UINT16, GLSL_TYPE_UINT, GLSL_TYPE_UINT64};
   const struct glsl_type *type = glsl_vector_type(types[ffs(bit_size / 8u) - 1u], num_components);

   nir_deref_instr *deref = strip_deref_casts(nir_instr_as_deref(src.ssa->parent_instr));
   return nir_build_deref_cast(b, &deref->dest.ssa, deref->mode, type, 0);
}

/* Reinterpret "def" as a vector with "size"-sized elements and extract an
 * element at bit "offset". */
static nir_ssa_def *
extract_vector(nir_builder *b, nir_ssa_def *def, unsigned size, unsigned offset)
{
   assert(offset % size == 0);

   if (size > def->bit_size) {
      nir_ssa_def *low = extract_vector(b, def, size / 2u, offset);
      nir_ssa_def *high = extract_vector(b, def, size / 2u, offset + size / 2u);
      if (size == 64) {
         return nir_pack_64_2x32(b, nir_vec2(b, low, high));
      } else if (size == 32) {
         return nir_pack_32_2x16(b, nir_vec2(b, low, high));
      } else {
         low = nir_u2u(b, low, size);
         high = nir_u2u(b, high, size);
         high = nir_ishl(b, high, nir_imm_int(b, size / 2u));
         return nir_ior(b, low, high);
      }
   } else if (size == def->bit_size) {
      return nir_channel(b, def, offset / size);
   } else {
      unsigned index = offset % def->bit_size / size;
      def = nir_channel(b, def, offset / def->bit_size);
      def = nir_unpack_bits(b, def, size);
      return nir_channel(b, def, index);
   }
}

/* Reinterpret "def" as a vector with "low_size"-sized elements and extract
 * "count" elements at bit "start". */
static nir_ssa_def *
extract_subvector(nir_builder *b, nir_ssa_def *def, unsigned start, unsigned count, unsigned size)
{
   assert(start % size == 0);
   assert(count <= 16);
   assert(count <= NIR_MAX_VEC_COMPONENTS);
   assert(start + count * size <= def->bit_size * def->num_components);

   nir_ssa_def *res[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < count; i++) {
      unsigned offset = start + i * size;
      assert(offset % size == 0);
      res[i] = extract_vector(b, def, size, offset);
   }
   if (count == 1)
      return res[0];
   return nir_vec(b, res, count);
}

/* Return true if the write mask "write_mask" of a store with "old_bit_size"
 * bits per element can be represented for a store with "new_bit_size" bits per
 * element. */
static bool
writemask_representable(unsigned write_mask, unsigned old_bit_size, unsigned new_bit_size)
{
   while (write_mask) {
      int start, count;
      u_bit_scan_consecutive_range(&write_mask, &start, &count);
      start *= old_bit_size;
      count *= old_bit_size;
      if (start % new_bit_size != 0)
         return false;
      if (count % new_bit_size != 0)
         return false;
   }
   return true;
}

/* Return true if "new_bit_size" is a usable bit size for a vectorized store of
 * "low" and "high". */
static bool
new_bitsize_acceptable(struct vectorize_ctx *ctx, unsigned new_bit_size,
                       struct entry *low, struct entry *high,
                       unsigned low_size, unsigned high_size,
                       unsigned size)
{
   if (size % new_bit_size != 0)
      return false;
   unsigned new_num_components = size / new_bit_size;
   if (new_num_components > NIR_MAX_VEC_COMPONENTS)
      return false;

   int alignment = ctx->align(get_variable_mode(low), low->store_value, new_bit_size, new_num_components);
   if (alignment < 0 || !is_entry_aligned(ctx, low, alignment))
      return false;

   if (low->store_value) {
      if (low_size % new_bit_size != 0)
         return false;
      if (high_size % new_bit_size != 0)
         return false;

      unsigned write_mask = nir_intrinsic_write_mask(low->intrin);
      if (!writemask_representable(write_mask, low_size, new_bit_size))
         return false;

      write_mask = nir_intrinsic_write_mask(high->intrin);
      if (!writemask_representable(write_mask, high_size, new_bit_size))
         return false;
   }

   return true;
}

/* Updates a write mask, "write_mask", so that it can be used with a
 * "new_bit_size"-bit store instead of a "old_bit_size"-bit store. */
static uint32_t
update_writemask(unsigned write_mask, unsigned old_bit_size, unsigned new_bit_size)
{
   uint32_t res = 0;
   while (write_mask) {
      int start, count;
      u_bit_scan_consecutive_range(&write_mask, &start, &count);
      start = start * old_bit_size / new_bit_size;
      count = count * old_bit_size / new_bit_size;
      res |= ((1 << count) - 1) << start;
   }
   return res;
}

static void
vectorize_loads(nir_builder *b,
                struct entry *low, struct entry *high,
                struct entry *first, struct entry *second,
                unsigned new_bit_size, unsigned new_num_components,
                unsigned high_start)
{
   unsigned low_bit_size = get_bit_size(low);
   unsigned high_bit_size = get_bit_size(high);

   b->cursor = nir_after_instr(first->instr);

   first->intrin->dest.ssa.num_components = new_num_components;
   first->intrin->dest.ssa.bit_size = new_bit_size;

   nir_ssa_def *low_def = nir_imov(b, extract_subvector(b, &first->intrin->dest.ssa, 0, low->intrin->num_components, low_bit_size));
   nir_ssa_def *high_def = nir_imov(b, extract_subvector(b, &first->intrin->dest.ssa, high_start, high->intrin->num_components, high_bit_size));
   if (first == low) {
      nir_ssa_def_rewrite_uses_after(&low->intrin->dest.ssa, nir_src_for_ssa(low_def), high_def->parent_instr);
      nir_ssa_def_rewrite_uses(&high->intrin->dest.ssa, nir_src_for_ssa(high_def));
   } else {
      nir_ssa_def_rewrite_uses(&low->intrin->dest.ssa, nir_src_for_ssa(low_def));
      nir_ssa_def_rewrite_uses_after(&high->intrin->dest.ssa, nir_src_for_ssa(high_def), high_def->parent_instr);
   }

   first->intrin->num_components = new_num_components;

   const struct intrinsic_info *info = first->info;

   if (first != low && nir_intrinsic_infos[first->intrin->intrinsic].index_map[NIR_INTRINSIC_BASE])
      nir_intrinsic_set_base(first->intrin, nir_intrinsic_base(low->intrin));

   if (first != low && info->base_src >= 0)
      nir_instr_rewrite_src(first->instr, &first->intrin->src[info->base_src], low->intrin->src[info->base_src]);

   if (info->deref_src >= 0) {
      b->cursor = nir_before_instr(first->instr);
      first->deref = cast_deref(b, new_num_components, new_bit_size, low->intrin->src[info->deref_src]);
      nir_instr_rewrite_src(first->instr, &first->intrin->src[info->deref_src], nir_src_for_ssa(&first->deref->dest.ssa));
   }

   if (first != low && nir_intrinsic_infos[second->intrin->intrinsic].index_map[NIR_INTRINSIC_ALIGN_MUL]) {
      nir_intrinsic_set_align(first->intrin,
                              nir_intrinsic_align_mul(low->intrin),
                              nir_intrinsic_align_offset(low->intrin));
   }

   first->key = low->key;

   nir_instr_remove(second->instr);
}

static void
vectorize_stores(nir_builder *b,
                 struct entry *low, struct entry *high,
                 struct entry *first, struct entry *second,
                 unsigned new_bit_size, unsigned new_num_components,
                 unsigned high_start)
{
   unsigned low_size = low->intrin->num_components * get_bit_size(low);

   b->cursor = nir_before_instr(second->instr);

   assert(low_size % new_bit_size == 0);
   uint32_t low_write_mask = update_writemask(nir_intrinsic_write_mask(low->intrin), get_bit_size(low), new_bit_size);
   uint32_t high_write_mask = update_writemask(nir_intrinsic_write_mask(high->intrin), get_bit_size(high), new_bit_size) << (high_start / new_bit_size);
   uint32_t write_mask = low_write_mask | high_write_mask;

   uint32_t write_mask2 = write_mask;
   nir_ssa_def *data_channels[NIR_MAX_VEC_COMPONENTS];
   while (write_mask2) {
      int i = u_bit_scan(&write_mask2);
      bool set_low = low_write_mask & (1 << i);
      bool set_high = high_write_mask & (1 << i);
      if (set_low && (!set_high || low == second)) {
         data_channels[i] = extract_vector(b, low->store_value, new_bit_size, i * new_bit_size);
      } else if (set_high) {
         assert(!set_low || high == second);
         data_channels[i] = extract_vector(b, high->store_value, new_bit_size, i * new_bit_size - high_start);
      }
   }
   nir_ssa_def *data = nir_vec(b, data_channels, new_num_components);

   nir_intrinsic_set_write_mask(second->intrin, write_mask);
   second->intrin->num_components = data->num_components;
   second->store_value = data;

   const struct intrinsic_info *info = second->info;
   assert(info->value_src >= 0);
   nir_instr_rewrite_src(second->instr, &second->intrin->src[info->value_src], nir_src_for_ssa(data));
   if (second != low && info->base_src >= 0)
      nir_instr_rewrite_src(second->instr, &second->intrin->src[info->base_src], low->intrin->src[info->base_src]);
   if (info->deref_src >= 0) {
      b->cursor = nir_before_instr(second->instr);
      second->deref = cast_deref(b, new_num_components, new_bit_size, low->intrin->src[info->deref_src]);
      nir_instr_rewrite_src(second->instr, &second->intrin->src[info->deref_src], nir_src_for_ssa(&second->deref->dest.ssa));
   }

   if (second != low && nir_intrinsic_infos[second->intrin->intrinsic].index_map[NIR_INTRINSIC_ALIGN_MUL]) {
      nir_intrinsic_set_align(second->intrin,
                              nir_intrinsic_align_mul(low->intrin),
                              nir_intrinsic_align_offset(low->intrin));
   }
   if (second != low && nir_intrinsic_infos[second->intrin->intrinsic].index_map[NIR_INTRINSIC_BASE])
      nir_intrinsic_set_base(second->intrin, nir_intrinsic_base(low->intrin));

   second->key = low->key;

   list_del(&first->head);
   nir_instr_remove(first->instr);
}

static int64_t
compare_entry_keys(struct entry_key *a, struct entry_key *b)
{
   if (a->var != b->var || a->resource != b->resource || a->offset_def_count != b->offset_def_count ||
       memcmp(a->offset_defs, b->offset_defs, a->offset_def_count * sizeof(nir_ssa_def *)) ||
       memcmp(a->offset_defs_mul, b->offset_defs_mul, a->offset_def_count * sizeof(uint32_t)))
      return INT64_MAX;

   return (int64_t)b->offset_base - (int64_t)a->offset_base;
}

static bool
try_vectorize(nir_function_impl *impl, struct vectorize_ctx *ctx, struct entry *first, struct entry *second)
{
   /* don't execute speculatively */
   if (first->instr->block->cf_node.parent != second->instr->block->cf_node.parent)
      return false;

   int64_t diff = compare_entry_keys(&first->key, &second->key);
   if (diff == INT64_MAX)
      return false;
   struct entry *low = diff > 0 ? first : second;
   struct entry *high = diff > 0 ? second : first;
   diff = llabs(diff);
   if (diff > get_bit_size(low) / 8u * low->intrin->num_components)
      return false;

   unsigned low_bit_size = get_bit_size(low);
   unsigned high_bit_size = get_bit_size(high);
   unsigned low_size = low->intrin->num_components * low_bit_size;
   unsigned high_size = high->intrin->num_components * high_bit_size;
   unsigned new_size = MAX2(diff * 8u + high_size, low_size);

   unsigned new_bit_size = 0;
   if (new_bitsize_acceptable(ctx, low_bit_size, low, high, low_size, high_size, new_size)) {
      new_bit_size = low_bit_size;
   } else if (low_bit_size != high_bit_size &&
            new_bitsize_acceptable(ctx, high_bit_size, low, high, low_size, high_size, new_size)) {
      new_bit_size = high_bit_size;
   } else {
      new_bit_size = 64;
      for (; new_bit_size >= 8; new_bit_size /= 2) {
         if (new_bit_size == low_bit_size || new_bit_size == high_bit_size)
            continue;
         if (new_bitsize_acceptable(ctx, new_bit_size, low, high, low_size, high_size, new_size))
            break;
      }
      if (new_bit_size < 8)
         return false;
   }
   unsigned new_num_components = new_size / new_bit_size;

   nir_builder b;
   nir_builder_init(&b, impl);

   if (first != low && low->info->base_src >= 0) {
      if (!schedule_ssa_def(low->intrin->src[low->info->base_src].ssa, first->instr))
         return false;
   }

   if (first != low && low->info->deref_src >= 0) {
      if (!schedule_ssa_def(low->intrin->src[low->info->deref_src].ssa, first->instr))
         return false;
   }

   if (first->store_value)
      vectorize_stores(&b, low, high, first, second, new_bit_size, new_num_components, diff * 8u);
   else
      vectorize_loads(&b, low, high, first, second, new_bit_size, new_num_components, diff * 8u);

   return true;
}

static bool
handle_barrier(struct block_state *state, nir_instr *instr)
{
   unsigned modes = 0;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      switch (intrin->intrinsic) {
      case nir_intrinsic_barrier:
      case nir_intrinsic_group_memory_barrier:
      case nir_intrinsic_memory_barrier:
      /* prevent speculative loads/stores */
      case nir_intrinsic_discard_if:
      case nir_intrinsic_discard:
         modes = nir_var_all;
         break;
      case nir_intrinsic_memory_barrier_buffer:
         modes = nir_var_mem_ssbo;
         break;
      case nir_intrinsic_memory_barrier_shared:
         modes = nir_var_mem_shared;
         break;
      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_call) {
      modes = nir_var_all;
   } else {
      return false;
   }

   while (modes) {
      unsigned mode = u_bit_scan(&modes);
      list_for_each_entry_safe(struct entry, entry, &state->entries[mode], head)
         entry->behind_barrier = true;
   }

   return true;
}

/* Returns true if it can prove that "a" and "b" point to different resources. */
static bool
resources_different(nir_ssa_def *a, nir_ssa_def *b)
{
   if (!a || !b)
      return false;

   nir_const_value *acv = nir_src_as_const_value(nir_src_for_ssa(a));
   nir_const_value *bcv = nir_src_as_const_value(nir_src_for_ssa(b));
   if (acv && bcv)
      return acv->u32 != bcv->u32;
   else if (acv || bcv)
      return false;

   if (a->parent_instr->type == nir_instr_type_intrinsic &&
       b->parent_instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *aintrin = nir_instr_as_intrinsic(a->parent_instr);
      nir_intrinsic_instr *bintrin = nir_instr_as_intrinsic(b->parent_instr);
      if (aintrin->intrinsic == nir_intrinsic_vulkan_resource_index &&
          bintrin->intrinsic == nir_intrinsic_vulkan_resource_index) {
         return nir_intrinsic_desc_set(aintrin) != nir_intrinsic_desc_set(bintrin) ||
                nir_intrinsic_binding(aintrin) != nir_intrinsic_binding(aintrin) ||
                resources_different(aintrin->src[0].ssa, bintrin->src[0].ssa);
      }
   }

   return false;
}

static bool
may_alias(struct entry *a, struct entry *b)
{
   assert(get_variable_mode(a) == get_variable_mode(b));

   if ((a->key.var != b->key.var || resources_different(a->key.resource, b->key.resource)) &&
       (a->access & ACCESS_RESTRICT) && (b->access & ACCESS_RESTRICT))
      return false;
   if (a->key.var != b->key.var || a->key.resource != b->key.resource)
      return true;

   int64_t diff = compare_entry_keys(&a->key, &b->key);
   if (diff != INT64_MAX) {
      /* with atomics, intrin->num_components can be 0 */
      if (diff < 0)
         return llabs(diff) < MAX2(b->intrin->num_components, 1u) * (get_bit_size(b) / 8u);
      else
         return diff < MAX2(a->intrin->num_components, 1u) * (get_bit_size(a) / 8u);
   }

   return true;
}

static bool
can_vectorize(struct entry *first, struct entry *second)
{
   return first->info == second->info && first->access == second->access &&
          !(first->access & ACCESS_VOLATILE) && !first->info->is_atomic;
}

static void
handle_dependency(struct entry *first, struct entry *second, bool *before, bool *after)
{
   *before = *after = false;

   if (!first->store_value && !second->store_value)
      return;

   if (!may_alias(first, second))
      return;

   if (first->store_value && second->store_value)
      *after = true;
   else
      *before = *after = true;
}

static bool
process_block(nir_function_impl *impl, struct vectorize_ctx *ctx, nir_block *block)
{
   bool progress = false;

   struct block_state *state = &ctx->block_states[block->index];

   for (unsigned i = 0; i < nir_num_variable_modes; i++)
      list_inithead(&state->entries[i]);

   nir_foreach_instr_safe(instr, block) {
      if (handle_barrier(state, instr))
         continue;

      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      const struct intrinsic_info *info = get_info(intrin->intrinsic); 
      if (!info)
         continue;

      nir_variable_mode mode = info->mode;
      if (!mode)
         mode = nir_src_as_deref(intrin->src[info->deref_src])->mode;
      if (!(mode & ctx->modes))
         continue;

      struct entry *second = create_entry(ctx, info, intrin);
      if (!second)
         continue;

      bool done = false;
      unsigned mode_index = ffs(mode) - 1;
      bool barrier = false;
      list_for_each_entry_safe_rev(struct entry, first, &state->entries[mode_index], head) {
         if (first->behind_barrier)
            barrier = true;
         bool before = true, after = true;
         if (!barrier)
            handle_dependency(first, second, &before, &after);

         barrier = barrier || before;

         if (!done && !barrier && can_vectorize(first, second) && (first->store_value ? first->can_move : true))
            done = try_vectorize(impl, ctx, first, second);

         barrier = barrier || after;
         if (barrier)
            first->can_move = false;
      }
      if (!done || second->store_value)
         list_addtail(&second->head, &state->entries[mode_index]);
      progress |= done;
   }

   return progress;
}

bool
nir_opt_load_store_vectorize(nir_shader *shader, nir_variable_mode modes,
                             int (*type_size)(nir_variable_mode, const struct glsl_type *),
                             int (*align)(nir_variable_mode, bool, unsigned, unsigned))
{
   bool progress = false;

   struct vectorize_ctx *ctx = ralloc(NULL, struct vectorize_ctx);
   ctx->modes = modes;
   ctx->type_size = type_size;
   ctx->align = align;

   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_metadata_require(function->impl,
                              nir_metadata_block_index | nir_metadata_dominance);

         ctx->block_states = ralloc_array(ctx, struct block_state, function->impl->num_blocks);
         nir_foreach_block(block, function->impl)
            progress |= process_block(function->impl, ctx, block);
         ralloc_free(ctx->block_states);

         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index | nir_metadata_dominance);
      }
   }

   ralloc_free(ctx);
   return progress;
}
