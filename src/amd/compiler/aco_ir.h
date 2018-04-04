/*
 * Copyright © 2018 Valve Corporation
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
 *    Daniel Schürmann (daniel.schuermann@campus.tu-berlin.de)
 *
 */

#ifndef ACO_IR_H
#define ACO_IR_H

#include <ostream>

#include "nir/nir.h"
#include "aco_opcodes.h"

typedef enum {
   b = 0,
   s1 = 1,
   s2 = 2,
   s3 = 3,
   s4 = 4,
   s8 = 8,
   s16 = 16,
   v1 = s1 | (1 << 5),
   v2 = s2 | (1 << 5),
   v3 = s3 | (1 << 5),
   v4 = s4 | (1 << 5),
} RegClass;

typedef struct {
   const char *name;
   unsigned num_inputs;
   unsigned num_outputs;
   RegClass output_type[2];
   bool kills_input[4];
   /* TODO: everything that depends on the instruction rather than the format */
   // like sideeffects on spr's

} opcode_info;

extern const opcode_info opcode_infos[static_cast<int>(aco_opcode::num_opcodes)];

#ifdef __cplusplus
#include <cstdint>
#include <vector>
#include <bitset>
#include <deque>
#include <memory>

namespace aco {



enum RegType {
   scc,
   sgpr,
   vgpr,
};

/**
 * Temp Class
 * Each temporary virtual register has a
 * register class (i.e. size and type)
 * and SSA id.
 */
class Temp {
public:
   Temp() = default;
   constexpr Temp(uint32_t id, RegClass cls) noexcept
      : id_(id), reg_class(cls) {}

   uint32_t id() const noexcept
   {
      return id_;
   }

   unsigned size() const noexcept
   {
      return (unsigned) reg_class & 0x1F;
   }

   RegType type()
   {
      if (reg_class == RegClass::b) return RegType::scc;
      if (reg_class <= RegClass::s16) return RegType::sgpr;
      else return RegType::vgpr;
   }

   RegClass regClass() const noexcept
   {
      return reg_class;
   }

private:
   uint32_t id_;
   RegClass reg_class;
};

struct PhysReg
{
   unsigned reg;
};

static inline PhysReg fixed_vgpr(unsigned idx)
{
   return PhysReg{idx + 256};
}

static inline PhysReg fixed_sgpr(unsigned idx)
{
   return PhysReg{idx};
}

static constexpr PhysReg m0{124};

/**
 * Operand Class
 * Initially, each Operand refers to either
 * a temporary virtual register
 * or to a constant value
 * Temporary registers get mapped to physical register during RA
 * Constant values are inlined into the instruction sequence.
 */
class Operand final
{
public:
   Operand() = default;
   explicit Operand(Temp r) noexcept
   {
      data_.temp = r;
      control_[0] = 1; /* isTemp */
   };
   explicit Operand(uint32_t v) noexcept
   {
      data_.i = v;
      control_[2] = 1; /* isConst */
   };
   explicit Operand(float v) noexcept
   {
      data_.f = v;
      control_[2] = 1; /* isConst */
   };
   explicit Operand(PhysReg reg, RegClass type) noexcept
   {
      data_.temp = Temp(0, type);
      setFixed(reg);
   }

   bool isTemp() const noexcept
   {
      return control_[0];
   }

   Temp getTemp() const noexcept
   {
      return data_.temp;
   }

   uint32_t tempId() const noexcept
   {
      return data_.temp.id();
   }

   RegClass regClass() const noexcept
   {
      return data_.temp.regClass();
   }

   unsigned size() const noexcept
   {
      return data_.temp.size();
   }

   bool isFixed() const noexcept
   {
      return control_[1];
   }

   PhysReg physReg() const noexcept
   {
      return reg_;
   }

   void setFixed(PhysReg reg) noexcept
   {
      control_[1] = 1;
      reg_ = reg;
   }

   bool isConstant() const noexcept
   {
      return control_[2];
   }

   uint32_t constantValue() const noexcept
   {
      return data_.i;
   }

   void setKill(bool) noexcept
   {
      control_[3] = 1;
   }

   bool kill() const noexcept
   {
      return control_[3];
   }
   
   std::string to_string()
   {
      if (isConstant())
         return std::to_string(data_.i);

      std::string s = "%" + std::to_string(data_.temp.id());

      if (isFixed())
         return s + ":" +  (data_.temp.type() == vgpr ? "v[" : "s[") + std::to_string(reg_.reg) + "]";
      return s;
   }

private:
   union {
      uint32_t i;
      float f;
      Temp temp;
   } data_;
   PhysReg reg_;
   std::bitset<8> control_;
};

/**
 * Definition Class
 * Definitions are the results of Instructions
 * and refer to temporary virtual registers
 * which are later mapped to physical registers
 */
class Definition final
{
public:
   Definition() = default;
   Definition(uint32_t index, RegClass type) noexcept
      : temp(index, type) {}
   Definition(Temp tmp) noexcept
      : temp(tmp) {}
   Definition(PhysReg reg, RegClass type) noexcept
      : temp(Temp(0, type))
   {
      setFixed(reg);
   }

   Temp getTemp() noexcept
   {
      return temp;
   }

   uint32_t tempId() const noexcept
   {
      return temp.id();
   }

   RegClass regClass() const noexcept
   {
      return temp.regClass();
   }

   unsigned size() const noexcept
   {
      return temp.size();
   }

   bool isFixed() const noexcept
   {
      return control_[0];
   }

   PhysReg physReg() const noexcept
   {
      return reg_;
   }

   void setFixed(PhysReg reg) noexcept
   {
      control_[0] = 1;
      reg_ = reg;
   }

   bool mustReuseInput() const noexcept
   {
      return control_[1];
   }

   void setReuseInput(bool v) noexcept
   {
      control_[1] = v;
   }

   std::string to_string()
   {
      std::string s = "%" + std::to_string(temp.id());

      if (isFixed())
         return s + ":" +  (temp.type() == vgpr ? "v[" : "s[") + std::to_string(reg_.reg) + "]";
      return s;
   }
private:
   Temp temp;
   std::bitset<8> control_;
   PhysReg reg_;
};

/**
 * Basic Instruction class
 * which derives into FixedInstruction
 * or PseudoInstruction
 * @param opcode
 */
class Instruction
{
public:
   Instruction(aco_opcode opcode) noexcept : opcode_(opcode) {}
   virtual ~Instruction() noexcept {}

   aco_opcode opcode() const noexcept { return opcode_; }
   
   /* return a value defined by this instruction as new operand */
   Operand asOperand(uint32_t index = 0)
   {
      return Operand(getDefinition(index).getTemp());
   }
 
   virtual std::size_t operandCount() const noexcept { return 0; }
   virtual Operand& getOperand(std::size_t index) noexcept { __builtin_unreachable(); }
 
   virtual std::size_t definitionCount() const noexcept { return 0; }
   virtual Definition& getDefinition(std::size_t index) noexcept { __builtin_unreachable(); }

   virtual std::string to_string()
   {
      std::string s;
      if (definitionCount()) {
         for (unsigned i = 0; i < definitionCount(); i++) {
            if (i)
               s.append(", ");
            s.append(getDefinition(i).to_string());
         }
         s.append(" = ");
      }
      s.append(std::string(opcode_infos[(int)opcode_].name));
      for (unsigned i = 0; i < operandCount(); i++)
         s.append(" " + getOperand(i).to_string());
      return s;
   }
private:
   aco_opcode opcode_;
};

/**
 * FixedInstruction class
 * number of Operands and Definitions is known at instantiation.
 * All FixedInstructions are actual machine instructions.
 * @param opcode
 */
template <std::size_t num_src, std::size_t num_dst>
class FixedInstruction : public Instruction
{
  public:
   FixedInstruction(aco_opcode opcode) noexcept : Instruction{opcode} {}
 
   std::size_t operandCount() const noexcept final override { return num_src; }
   Operand& getOperand(std::size_t index) noexcept final override { return operands_[index]; }
   std::size_t definitionCount() const noexcept final override { return num_dst; }
   Definition& getDefinition(std::size_t index) noexcept final override { return defs_[index]; }
 
  private:
   Definition defs_[num_dst];
   Operand operands_[num_src];
};

/**
 * SOP2 Instruction class
 * see aco_builder.cpp for how this class is used
 * @param opcode
 * @param operands
 * @param defs
 */
template <std::size_t num_src, std::size_t num_dst>
class SOP2 final : public FixedInstruction<num_src, num_dst>
{
public:
   SOP2(aco_opcode opcode)
      : FixedInstruction<num_src,num_dst>(opcode)
   {}
};

template <std::size_t num_src, std::size_t num_dst>
class SOPK final : public FixedInstruction<num_src, num_dst>
{
public:
   SOPK(aco_opcode opcode, unsigned imm)
      : FixedInstruction<num_src,num_dst>{opcode},
        imm(imm)
   {}
   unsigned imm;
};

template <std::size_t num_src, std::size_t num_dst>
class SOP1 final : public FixedInstruction<num_src, num_dst>
{
public:
   SOP1(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
};

template <std::size_t num_src, std::size_t num_dst>
class SOPC final : public FixedInstruction<num_src, num_dst>
{
public:
   SOPC(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
};

// Forward declaration
struct Block;

template <std::size_t num_src, std::size_t num_dst>
class SOPP final : public FixedInstruction<num_src, num_dst>
{
public:
   SOPP(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
   SOPP(aco_opcode opcode, Block* block)
      : FixedInstruction<num_src, num_dst>{opcode}, block_(block)
   {}
   SOPP(aco_opcode opcode, unsigned imm)
      : FixedInstruction<num_src, num_dst>{opcode}, immediate(imm)
   {}
private:
   unsigned immediate;
   Block* block_;
};

template <std::size_t num_src, std::size_t num_dst>
class SMEM final : public FixedInstruction<num_src, num_dst>
{
public:
   SMEM(aco_opcode opcode, bool glc, bool imm)
      : FixedInstruction<num_src, num_dst>{opcode},
        glc(glc),
        imm(imm)
   {}
private:
   bool glc;
   bool imm;
};

/**
 * Mixin Class for Sub-DWord-Addressing:
 * Must only be used with VOP1, VOP2 or VOPC template parameter.
 */
template <template<std::size_t, std::size_t>class T, std::size_t num_src, std::size_t num_dst>
class SDWA final : public T<num_src, num_dst>
{
public:
   SDWA(aco_opcode opcode, unsigned dst_sel, unsigned dst_u, bool clamp,
           unsigned src0_sel, bool src0_sext, bool src0_neg, bool src0_abs,
           unsigned src1_sel, bool src1_sext, bool src1_neg, bool src1_abs)
   : T<num_src, num_dst>{opcode}, dst_sel(dst_sel), dst_u(dst_u), clamp(clamp),
     src0_sel(src0_sel), src0_neg(src0_neg), src0_abs(src0_abs),
     src1_sel(src1_sel), src1_neg(src1_neg), src1_abs(src1_abs)
   {}
private: // TODO use Bitset
   unsigned dst_sel : 3;
   unsigned dst_u : 2;
   bool clamp : 1;
   unsigned src0_sel : 3;
   bool src0_sext : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   unsigned src1_sel : 3;
   bool src1_sext : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   
};

/**
 * Mixin Class for Data Parallel Primitives:
 * Must only be used with VOP1, VOP2 or VOPC template parameter.
 */
template <template<std::size_t, std::size_t>class T, std::size_t num_src, std::size_t num_dst>
class DPP final : public T<num_src, num_dst>
{
public:
   DPP(aco_opcode opcode, unsigned dpp_ctrl, bool bound_ctrl,
           bool src0_neg, bool src0_abs, bool src1_neg, bool src1_abs,
           unsigned bank_mask, unsigned row_mask)
   : T<num_src, num_dst>{opcode},
     dpp_ctrl(dpp_ctrl), bound_ctrl(bound_ctrl),
     src0_neg(src0_neg), src0_abs(src0_abs), src1_neg(src1_neg), src1_abs(src1_abs),
     bank_mask(bank_mask), row_mask(row_mask)
   {}
private: // TODO: use Bitset
   unsigned dpp_ctrl : 9;
   bool bound_ctrl : 1;
   bool src0_neg : 1;
   bool src0_abs : 1;
   bool src1_neg : 1;
   bool src1_abs : 1;
   unsigned bank_mask : 4;
   unsigned row_mask : 4;
   
};

template <std::size_t num_src, std::size_t num_dst>
class VOP2 : public FixedInstruction<num_src, num_dst>
{
public:
   VOP2(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
};

template <std::size_t num_src, std::size_t num_dst>
class VOP1 : public FixedInstruction<num_src, num_dst>
{
public:
   VOP1(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
};

template <std::size_t num_src, std::size_t num_dst>
class VOPC : public FixedInstruction<num_src, num_dst>
{
public:
   VOPC(aco_opcode opcode)
      : FixedInstruction<num_src, num_dst>{opcode}
   {}
};

/**
 * Export Instruction class
 * @param enabledMask
 * @param dest
 * @param compressed
 * @param done
 * @param validMask
 */
class ExportInstruction final : public FixedInstruction<4, 0>
{
  public:
    ExportInstruction(unsigned enabledMask, unsigned dest, bool compressed, bool done, bool validMask) noexcept
      : FixedInstruction{aco_opcode::exp},
        enabledMask_{enabledMask},
        dest_{dest},
        compressed_{compressed},
        done_{done},
        validMask_{validMask}
    {}
 
  private:
    unsigned enabledMask_;
    unsigned dest_;
    bool compressed_;
    bool done_;
    bool validMask_;
};


template <std::size_t num_src, std::size_t num_dst>
class InterpInstruction final : public FixedInstruction<num_src, num_dst>
{
   public:
      InterpInstruction(aco_opcode opcode, unsigned attribute, unsigned component) noexcept
        : FixedInstruction<num_src, num_dst>{opcode}, attribute_{attribute}, component_{component} {}

   private:
      unsigned attribute_;
      unsigned component_;
};

/**
 * PseudoInstruction Class
 * @param opcode
 * @param num_src
 * @param num_dst
 */
class PseudoInstruction final : public Instruction
{
public:
   PseudoInstruction(aco_opcode opcode, std::size_t num_src, std::size_t num_dst) :
      Instruction{opcode}, defs_(num_dst), operands_(num_src) {}

   std::size_t operandCount() const noexcept final override { return operands_.size(); }
   Operand& getOperand(std::size_t index) noexcept final override { return operands_[index]; }
   std::size_t definitionCount() const noexcept final override { return defs_.size(); }
   Definition& getDefinition(std::size_t index) noexcept final override { return defs_[index]; }

private:
    std::vector<Definition> defs_;
    std::vector<Operand> operands_;
};

/* CFG */
struct Block {

   std::deque<std::unique_ptr<Instruction>> instructions;
   std::deque<Block*> logical_predecessors;
   Block* linear_predecessor;
   std::deque<Block*> logical_successors;
   Block* linear_successor;
};


class Program final {
public:
   std::vector<std::unique_ptr<Block>> blocks;

   uint32_t allocateId()
   {
      return allocationID++;
   }

   uint32_t peekAllocationId()
   {
      return allocationID;
   }

   void setAllocationId(uint32_t id)
   {
      allocationID = id;
   }

   Block* createAndInsertBlock()
   {
      Block* b = new Block
      {
         std::deque<std::unique_ptr<Instruction>>(),
         std::deque<Block*>(),
         blocks.back().get(), /* linear predecessor */
         std::deque<Block*>(),
         nullptr
      };
      blocks.back().get()->linear_successor = b;
      blocks.push_back(std::unique_ptr<Block>(b));
      return b;
   }

   void print(std::ostream& out)
   {
      int BB = 0;
      for (auto const& block : blocks)
      {
         out << "BB" << BB << ":" << std::endl;
         for (auto const& instr : block->instructions)
         {
            out << "\t" << instr->to_string() << std::endl;
         }
      }
   }

private:
   uint32_t allocationID = 0;
};

std::unique_ptr<Program> select_program(struct nir_shader *nir);
}
#endif /* __cplusplus */
#endif /* ACO_IR_H */

