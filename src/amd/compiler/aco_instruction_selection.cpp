#include <algorithm>
#include <unordered_map>

#include "aco_ir.h"
#include "aco_builder.h"
#include "aco_interface.h"
#include "nir/nir.h"

namespace aco {
namespace {
struct isel_context {
   Program *program;
   Block *block;
   bool *uniform_vals;
   std::unique_ptr<bool[]> use_vgpr;
   std::unordered_map<unsigned, unsigned> allocated;

   Temp barycentric_coords;
   Temp prim_mask;
};

static void visit_cf_list(struct isel_context *ctx,
                          struct exec_list *list);

RegClass get_ssa_reg_class(struct isel_context *ctx, nir_ssa_def *def)
{
   if (def->bit_size != 32)
      abort();
   unsigned v = def->num_components;
   if (ctx->use_vgpr[def->index])
      v |= 1 << 5;
   return (RegClass)v;
}

Temp get_ssa_temp(struct isel_context *ctx, nir_ssa_def *def)
{
   RegClass rc = get_ssa_reg_class(ctx, def);
   auto it = ctx->allocated.find(def->index);
   if (it != ctx->allocated.end())
      return Temp{it->second, rc};
   uint32_t id = ctx->program->allocateId();
   ctx->allocated.insert({def->index, id});
   return Temp{id, rc};
}

void visit_alu_instr(isel_context *ctx, nir_alu_instr *instr)
{
   if (!instr->dest.dest.is_ssa)
      abort();
   switch(instr->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      std::unique_ptr<PseudoInstruction> vec{new PseudoInstruction(aco_opcode::p_create_vector, instr->dest.dest.ssa.num_components, 1)};
      for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; ++i) {
         if (instr->src[i].swizzle[0])
            abort();
          vec->getOperand(i) = Operand{get_ssa_temp(ctx, instr->src[i].src.ssa)};
      }
      vec->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.dest.ssa));
      ctx->block->instructions.emplace_back(std::move(vec));
      break;
   }
   default:
      fprintf(stderr, "Unknown NIR instr type: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
   }
}

void visit_load_const(isel_context *ctx, nir_load_const_instr *instr)
{
   if (instr->def.bit_size != 32)
      abort();
   if (instr->def.num_components != 1)
      abort();
   if (ctx->use_vgpr[instr->def.index]) {
      std::unique_ptr<VOP1<1, 1>> mov{new VOP1<1,1>(aco_opcode::v_mov_b32)};
      mov->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->def));
      mov->getOperand(0) = Operand{instr->value.u32[0]};
      ctx->block->instructions.emplace_back(std::move(mov));
   } else {
      std::unique_ptr<SOP1<1, 1>> mov{new SOP1<1,1>(aco_opcode::s_mov_b32)};
      mov->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->def));
      mov->getOperand(0) = Operand{instr->value.u32[0]};
      ctx->block->instructions.emplace_back(std::move(mov));
   }
}

void visit_store_output(isel_context *ctx, nir_intrinsic_instr *instr)
{
   std::unique_ptr<ExportInstruction> exp{new ExportInstruction(0xf, 0, false, true, true)};

   Temp src = get_ssa_temp(ctx, instr->src[0].ssa);
   for (unsigned i = 0; i < 4; ++i) {
      Temp tmp{ctx->program->allocateId(), v1};
      std::unique_ptr<FixedInstruction<2, 1>> extract{new FixedInstruction<2,1>(aco_opcode::p_extract_vector, Format::PSEUDO)};

      extract->getOperand(0) = Operand(src);
      extract->getOperand(1) = Operand(i);
      extract->getDefinition(0) = Definition(tmp);

      ctx->block->instructions.emplace_back(std::move(extract));

      exp->getOperand(i) = Operand(tmp);
   }

   ctx->block->instructions.emplace_back(std::move(exp));
}

void visit_load_interpolated_input(isel_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr);
   unsigned component = nir_intrinsic_component(instr);
   Temp coord1{ctx->program->allocateId(), v1};
   Temp coord2{ctx->program->allocateId(), v1};

   std::unique_ptr<FixedInstruction<2, 1>> coord1_extract{new FixedInstruction<2,1>(aco_opcode::p_extract_vector, Format::PSEUDO)};
   coord1_extract->getOperand(0) = Operand{get_ssa_temp(ctx, instr->src[0].ssa)};
   coord1_extract->getOperand(1) = Operand((uint32_t)0);

   coord1_extract->getDefinition(0) = Definition{coord1};
   std::unique_ptr<FixedInstruction<2, 1>> coord2_extract{new FixedInstruction<2,1>(aco_opcode::p_extract_vector, Format::PSEUDO)};
   coord2_extract->getOperand(0) = Operand{get_ssa_temp(ctx, instr->src[0].ssa)};
   coord2_extract->getOperand(1) = Operand((uint32_t)1);

   coord2_extract->getDefinition(0) = Definition{coord2};

   ctx->block->instructions.emplace_back(std::move(coord1_extract));
   ctx->block->instructions.emplace_back(std::move(coord2_extract));

   Temp tmp{ctx->program->allocateId(), v1};
   std::unique_ptr<InterpInstruction<2, 1>> p1{new InterpInstruction<2, 1>(aco_opcode::v_interp_p1_f32, base, component)};
   p1->getOperand(0) = Operand(coord1);
   p1->getOperand(1) = Operand(ctx->prim_mask);
   p1->getOperand(1).setFixed(m0);
   p1->getDefinition(0) = Definition(tmp);
   std::unique_ptr<InterpInstruction<3, 1>> p2{new InterpInstruction<3, 1>(aco_opcode::v_interp_p2_f32, base, component)};
   p2->getOperand(0) = Operand(coord2);
   p2->getOperand(1) = Operand(ctx->prim_mask);
   p2->getOperand(1).setFixed(m0);
   p2->getOperand(2) = Operand(tmp);
   p2->getDefinition(0) = Definition(get_ssa_temp(ctx, &instr->dest.ssa));

   ctx->block->instructions.emplace_back(std::move(p1));
   ctx->block->instructions.emplace_back(std::move(p2));
}

void visit_intrinsic(isel_context *ctx, nir_intrinsic_instr *instr)
{
   switch(instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
      ctx->allocated[instr->dest.ssa.index] = ctx->barycentric_coords.id();
      break;
   case nir_intrinsic_load_interpolated_input:
      visit_load_interpolated_input(ctx, instr);
      break;
   case nir_intrinsic_store_output:
      visit_store_output(ctx, instr);
      break;
   default:
      break;
   }
}

void visit_block(isel_context *ctx, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
         visit_alu_instr(ctx, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_load_const:
         visit_load_const(ctx, nir_instr_as_load_const(instr));
         break;
      case nir_instr_type_intrinsic:
         visit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
         break;
      default:
         fprintf(stderr, "Unknown NIR instr type: ");
         nir_print_instr(instr, stderr);
         fprintf(stderr, "\n");
         //abort();
      }
   }
}

static void visit_if(isel_context *ctx, nir_if *if_stmt)
{
   /* Disabled for now till I find time to untangle the assumptions. */
#if 0
   Temp cond = get_ssa_temp(ctx, if_stmt->condition.ssa);

   Block* aco_then = ctx->program->createAndInsertBlock();
   aco_then->logical_predecessors.push_back(ctx->block);
   ctx->block->logical_successors.push_back(aco_then);

   if (cond.type() == RegType::scc) { /* uniform condition */
      Block* aco_else = ctx->program->createAndInsertBlock();
      aco_else->logical_predecessors.push_back(ctx->block);
      ctx->block->logical_successors.push_back(aco_else);

      /* branch to the new block if condition is false */
      Builder(ctx->program, ctx->block).s_cbranch_scc0(Operand(cond), aco_else);

      /* emit then block */
      ctx->block = aco_then;
      visit_cf_list(ctx, &if_stmt->then_list);

      if (exec_list_is_empty(&if_stmt->else_list)) {
         /* if there is no else-list, take the
          * created aco_else block to continue */
         ctx->block = aco_else;
      } else {
         Block* aco_cont = ctx->program->createAndInsertBlock();
         aco_else->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_else);

         /* at the end of the then block, jump to the cont block */
         Builder(ctx->program, ctx->block).s_branch(aco_cont);

         /* emit else block */
         ctx->block = aco_else;
         visit_cf_list(ctx, &if_stmt->else_list);

         ctx->block = aco_cont;
      }
      aco_then->logical_successors.push_back(ctx->block);
      ctx->block->logical_predecessors.push_back(aco_then);

   } else { /* non-uniform condition */

      if (exec_list_is_empty(&if_stmt->else_list)) {
         Block* aco_cont = ctx->program->createAndInsertBlock();
         ctx->block->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(ctx->block);
         aco_then->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_then);

         /* without else-list, directly branch on condition */
         Builder(ctx->program, ctx->block).s_cbranch_vccz(Operand(cond), aco_cont);

         ctx->block = aco_then;
         Builder B(ctx->program, aco_then);
         /* set the exec mask inside then-block */
         Instruction* orig_exec = B.s_and_saveexec_b64(Operand(cond));

         /* emit then block */
         visit_cf_list(ctx, &if_stmt->then_list);

         /* restore exec mask */
         Instruction* restore = B.s_mov_b64(orig_exec->asOperand(0));
         restore->getDefinition(0).setFixed(PhysReg{126});
         ctx->block = aco_cont;

      } else {
         Block* aco_T = ctx->program->createAndInsertBlock();
         Block* aco_else = ctx->program->createAndInsertBlock();
         Block* aco_cont = ctx->program->createAndInsertBlock();
         ctx->block->logical_successors.push_back(aco_else);
         aco_else->logical_predecessors.push_back(ctx->block);
         aco_then->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_then);
         aco_else->logical_successors.push_back(aco_cont);
         aco_cont->logical_predecessors.push_back(aco_else);
         Builder B(ctx->program, ctx->block);
         /* with else-list, first set exec mask */
         Instruction* orig_exec = B.s_and_saveexec_b64(Operand(cond));

         /* branch on exec mask to T block */
         B.s_cbranch_execz(aco_T);

         /* emit then block */
         ctx->block = aco_then;
         visit_cf_list(ctx, &if_stmt->then_list);

         ctx->block = aco_T;
         B = Builder(ctx->program, aco_T);
         /* negate exec mask */
         Instruction* else_exec = B.s_xor_b64(Operand(PhysReg{126}, s2), orig_exec->asOperand(0));
         else_exec->getDefinition(0).setFixed(PhysReg{126});

         /* branch on exec mask to cont block */
         B.s_cbranch_execz(aco_cont);

         /* emit else block */
         ctx->block = aco_else;
         visit_cf_list(ctx, &if_stmt->else_list);

         ctx->block = aco_cont;
         /* restore original exec mask */
         Instruction* restore = Builder(ctx->program, aco_cont).s_mov_b64(orig_exec->asOperand(0));
         restore->getDefinition(0).setFixed(PhysReg{126});
      }
   }
#endif
}

static void visit_cf_list(isel_context *ctx,
                          struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         visit_block(ctx, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         visit_if(ctx, nir_cf_node_as_if(node));
         break;
      default:
         unreachable("unimplemented cf list type");
      }
   }
}


std::unique_ptr<bool[]> init_reg_type(nir_function_impl *impl)
{
   std::unique_ptr<bool[]> use_vgpr{new bool[impl->ssa_alloc]};
   for(unsigned i = 0; i < impl->ssa_alloc; ++i)
      use_vgpr[i] = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch(instr->type) {
         case nir_instr_type_alu: {
            nir_alu_instr *alu_instr = nir_instr_as_alu(instr);
            bool emit_vgpr = false;
            switch(alu_instr->op) {
               case nir_op_fmul:
               case nir_op_fadd:
               case nir_op_fsub:
               case nir_op_fneg:
               case nir_op_fabs:
                  emit_vgpr = true;
                  break;
               default:
                  for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                     if (use_vgpr[alu_instr->src[i].src.ssa->index])
                        emit_vgpr = true;
                  }
                  break;
            }
            use_vgpr[alu_instr->dest.dest.ssa.index] = emit_vgpr;
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
            bool emit_vgpr = false;
            switch(intrinsic->intrinsic) {
               case nir_intrinsic_load_input:
               case nir_intrinsic_load_vertex_id:
               case nir_intrinsic_load_vertex_id_zero_base:
               case nir_intrinsic_load_barycentric_pixel:
               case nir_intrinsic_load_interpolated_input:
                  emit_vgpr = true;
                  break;
               default:
                  for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs; i++) {
                     if (use_vgpr[intrinsic->src[i].ssa->index])
                        emit_vgpr = true;
                  }
                  break;
            }
            if (nir_intrinsic_infos[intrinsic->intrinsic].has_dest) {
               use_vgpr[intrinsic->dest.ssa.index] = emit_vgpr;
            }
            break;
         }
         }
      }
   }
   return use_vgpr;
}

int
type_size(const struct glsl_type *type)
{
        return glsl_count_attribute_slots(type, false);
}

void add_startpgm(struct isel_context *ctx)
{
   std::unique_ptr<PseudoInstruction> startpgm{new PseudoInstruction(aco_opcode::p_startpgm, 0, 2)};

   ctx->barycentric_coords = Temp{ctx->program->allocateId(), v2};
   ctx->prim_mask = Temp{ctx->program->allocateId(), s1};

   startpgm->getDefinition(0) = Definition{ctx->barycentric_coords};
   startpgm->getDefinition(0).setFixed(fixed_vgpr(0));

   startpgm->getDefinition(1) = Definition{ctx->prim_mask};
   startpgm->getDefinition(1).setFixed(fixed_sgpr(0));

   ctx->block->instructions.push_back(std::move(startpgm));
}

}

std::unique_ptr<Program> select_program(struct nir_shader *nir)
{
   std::unique_ptr<Program> program{new Program};
   isel_context ctx;
   ctx.program = program.get();

   nir_lower_io(nir, (nir_variable_mode)(nir_var_shader_in | nir_var_shader_out), type_size, (nir_lower_io_options)0);
   struct nir_function *func = (struct nir_function *)exec_list_get_head(&nir->functions);
   nir_index_ssa_defs(func->impl);
   ctx.uniform_vals = nir_uniform_analysis(nir);
   ctx.use_vgpr = init_reg_type(func->impl);

   nir_print_shader(nir, stderr);

   ctx.program->blocks.push_back(std::unique_ptr<Block>{new Block});
   ctx.block = ctx.program->blocks.back().get();

   add_startpgm(&ctx);

   visit_cf_list(&ctx, &func->impl->body);

   ctx.block->instructions.push_back(std::unique_ptr<SOPP<0, 0>>(new SOPP<0, 0>(aco_opcode::s_endpgm)));

   return program;
}
}
