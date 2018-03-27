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

#include <cstdint>
#include <vector>
#include "aco_opcodes.h"

namespace aco {

enum class RegClass {
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
};

typedef struct {
   const char *name;
   unsigned num_inputs;
   unsigned num_outputs;
   RegClass output_type[2];
   bool kills_input[4];
   /* TODO: everything that depends on the instruction rather than the format */
   // like sideeffects on spr's

} opcode_info;
extern const opcode_info opcode_infos[num_opcodes];

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
   Temp();
   constexpr Temp(std::uint32_t id, RegClass cls) noexcept : id_(id), reg_class(cls) {};
   //constexpr Temp(std::uint32_t id, std::uint32_t control) noexcept;
 
   std::uint32_t id() const noexcept
   {
      return id_;
   }
   unsigned size()
   {
      return (unsigned) reg_class & 0x1F;
   }
   RegType type()
   {
      if (reg_class == RegClass::b) return RegType::scc;
      if (reg_class <= RegClass::s16) return RegType::sgpr;
      else return RegType::vgpr;
   }
   
   
private:
   std::uint32_t id_;
   //std::uint32_t control_;
   RegClass reg_class;
   friend class Definition;
   friend class Operand;
};

struct PhysReg
{
   unsigned reg;
};


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
   explicit Operand(Temp r) noexcept : temp(r) {};
   Operand(Temp r, PhysReg reg) noexcept : temp(r) {};
   explicit Operand(std::uint32_t v) noexcept { data_.i = v; };
   explicit Operand(float v) noexcept { data_.f = v; };
 
   bool isTemp() const noexcept;
   Temp getTemp() const noexcept;
 
   std::uint32_t tempId() const noexcept;
   void setTempId(std::uint32_t) noexcept;
   RegClass regClass() const noexcept;
   unsigned size() const noexcept;
 
   bool isFixed() const noexcept;
   PhysReg physReg() const noexcept;
   void setFixed(PhysReg reg) noexcept;
 
   bool isConstant() const noexcept;
   std::uint32_t constantValue() const noexcept
   {
      return data_.i;
   }
 
   void setKill(bool) noexcept;
   bool kill() const noexcept;
 
private:
   Temp temp;
   union {
      std::uint32_t i;
      float f;
   } data_;
   std::uint32_t control_;
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
   Definition(RegClass type) noexcept;
   explicit Definition(Temp r) noexcept;
   explicit Definition(Temp t, PhysReg reg) noexcept;
   explicit Definition(PhysReg reg) noexcept;
 
   bool isTemp() const noexcept;
   Temp getTemp() const noexcept;
 
   std::uint32_t tempId() const noexcept;
   void setTempId(std::uint32_t) noexcept;
   RegClass regClass() const noexcept;
   unsigned size() const noexcept;
 
   bool isFixed() const noexcept;
   PhysReg physReg() const noexcept;
   void setFixed(PhysReg reg) noexcept;
 
private:
   std::uint32_t tempId_;
   std::uint32_t control_;
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

   aco_opcode opcode() const noexcept;
   
   /* return a value defined by this instruction as new operand */
   Operand asOperand(uint32_t index = 0)
   {
      return Operand(getDefinition(index).getTemp());
   }
 
   virtual std::size_t operandCount() const noexcept { return 0; }
   virtual Operand& getOperand(std::size_t index) noexcept { __builtin_unreachable(); }
 
   virtual std::size_t definitionCount() const noexcept { return 0; }
   virtual Definition& getDefinition(std::size_t index) noexcept { __builtin_unreachable(); }
 
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
   FixedInstruction(aco_opcode opcode, Operand* operands) noexcept : Instruction{opcode}, operands_(operands) // this doesn't work
   { }
   FixedInstruction(aco_opcode opcode, Operand* operands, Definition* defs) noexcept :
   Instruction{opcode}, operands_(operands), defs_(defs) {}
 
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
 * see aco_builder*cpp for how this class is used
 * @param opcode
 * @param operands
 * @param defs
 */
template <std::size_t num_src, std::size_t num_dst>
class SOP2 final : public FixedInstruction<num_src, num_dst>
{
public:
   SOP2(aco_opcode opcode, std::vector<Operand> &operands, std::vector<Definition> &defs) : FixedInstruction<num_src,num_dst>(opcode) {};
};

template <std::size_t num_src, std::size_t num_dst>
class SOPK final : public FixedInstruction<num_src, num_dst>
{
public:
   SOPK(aco_opcode opcode, unsigned imm) : FixedInstruction<1,num_dst>{opcode} {}
   SOPK(aco_opcode opcode, Operand src, unsigned imm) : FixedInstruction<2,1>{opcode} {}
};

template <std::size_t num_src, std::size_t num_dst>
class SOP1 final : public FixedInstruction<num_src, num_dst>
{
public:
   SOP1(aco_opcode opcode, Operand ssrc0) : FixedInstruction<1,1>{opcode} {}
};

template <std::size_t num_src, std::size_t num_dst>
class SOPP final : public FixedInstruction<num_src, num_dst>
{
public:
   SOPP(aco_opcode opcode) : FixedInstruction<0,0>{opcode} {}
   SOPP(aco_opcode opcode, Operand src) : FixedInstruction<1,0>{opcode} {}
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
private:
    std::vector<Definition*> defs_;
    std::vector<Operand*> operands_;
};

/**
 * Examples for Builder::instruction_factories()
 * @return 
 */
Instruction s_endpgm()
{
   return SOPP<0,0>(aco_opcode::s_endpgm);
}

Instruction p_parallelcopy(unsigned size)
{
   return PseudoInstruction(aco_opcode::p_parallelcopy, size, size);
}
}
#endif /* ACO_IR_H */

