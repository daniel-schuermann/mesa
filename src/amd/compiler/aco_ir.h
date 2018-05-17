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
#include "common/ac_binary.h"
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
   unsigned opcode;
} opcode_info;

extern const opcode_info opcode_infos[static_cast<int>(aco_opcode::num_opcodes)];

#ifdef __cplusplus
#include <cstdint>
#include <vector>
#include <bitset>
#include <deque>
#include <memory>

namespace aco {

/**
 * Representation of the instruction's microcode encoding format
 * Note: Some Vector ALU Formats can be combined, such that:
 * - VOP2* | VOP3A represents a VOP2 instruction in VOP3A encoding
 * - VOP2* | DPP represents a VOP2 instruction with data parallel primitive.
 * - VOP2* | SDWA represents a VOP2 instruction with sub-dword addressing.
 * 
 * (*) The same is applicable for VOP1 and VOPC instructions.
 */
enum class Format : std::uint16_t {
   /* Pseudo Instruction Format */
   PSEUDO = 0,
   /* Scalar ALU & Control Formats */
   SOP1 = 1,
   SOP2 = 2,
   SOPK = 3,
   SOPP = 4,
   SOPC = 5,
   /* Scalar Memory Format */
   SMEM = 6,
   /* Vector Parameter Interpolation Format */
   VINTRP = 7,
   /* LDS/GDS Format */
   DS = 8,
   /* Vector Memory Buffer Formats */
   MTBUF = 9,
   MUBUF = 10,
   /* Vector Memory Image Format */
   MIMG = 11,
   /* Export Format */
   EXP = 12,
   /* Flat Formats */
   FLAT = 13,
   GLOBAL = 14,
   SCRATCH = 15,
   /* Vector ALU Formats */
   VOP1 = 16,
   VOP2 = 17,
   VOPC = 18,
   VOP3B = 19,
   VOP3P = 20,
   VOP3A = 1 << 8,
   DPP = 1 << 9,
   SDWA = 1 << 10,
};

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
      if (v <= 64)
         setFixed(PhysReg{128 + v});
      else if (v >= 0xFFFFFFFFFFFFFFF0) /* [-16 .. -1] */
         setFixed(PhysReg{192 - v});
      else if (v == 0x3f000000) /* 0.5 */
         setFixed(PhysReg{240});
      else if (v == 0xbf000000) /* -0.5 */
         setFixed(PhysReg{241});
      else if (v == 0x3f800000) /* 1.0 */
         setFixed(PhysReg{242});
      else if (v == 0xbf800000) /* -1.0 */
         setFixed(PhysReg{243});
      else if (v == 0x40000000) /* 2.0 */
         setFixed(PhysReg{244});
      else if (v == 0xc0000000) /* -2.0 */
         setFixed(PhysReg{245});
      else if (v == 0x40800000) /* 4.0 */
         setFixed(PhysReg{246});
      else if (v == 0xc0800000) /* -4.0 */
         setFixed(PhysReg{247});
//      else if (v == 1/2PI ???)
//         setFixed(PhysReg{248});
      else /* Literal Constant */
         setFixed(PhysReg{255});
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

   void setTemp(Temp t) {
      assert(control_[0]);
      data_.temp = t;
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

      if (isFixed()) {
         int reg = reg_.reg - (data_.temp.type() == vgpr ? 256 : 0);
         return s + ":" +  (data_.temp.type() == vgpr ? "v[" : "s[") + std::to_string(reg) + "]";
      }
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

   bool isTemp() const noexcept
   {
      return tempId() > 0;
   }

   Temp getTemp() noexcept
   {
      return temp;
   }

   uint32_t tempId() const noexcept
   {
      return temp.id();
   }

   void setTemp(Temp t) {
      temp = t;
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

      if (isFixed()) {
         int reg = reg_.reg - (temp.type() == vgpr ? 256 : 0);
         return s + ":" +  (temp.type() == vgpr ? "v[" : "s[") + std::to_string(reg) + "]";
      }
      return s;
   }
private:
   Temp temp;
   std::bitset<8> control_;
   PhysReg reg_;
};

class Block;

struct Instruction {
   aco_opcode opcode;
   Format format;

   Operand *operands;
   uint32_t num_operands;

   Definition *definitions;
   uint32_t num_definitions;

   // TODO remove
   uint32_t definitionCount() const { return num_definitions; }
   uint32_t operandCount() const { return num_operands; }
   Operand& getOperand(int index) { return operands[index]; }
   Definition& getDefinition(int index) { return definitions[index]; }

   std::string to_string()
   {
      std::string s = opcode_infos[(int)opcode].name;
      if (definitionCount()) {
         s = " = " + s;
         for (int i = definitionCount() - 1; i >= 0; --i) {
            if (i < (int) definitionCount() - 1)
               s = ", " + s;
            s = getDefinition(i).to_string() + s;
         }
      }
      if (operandCount()) {
         for (int i = 0; i < (int) operandCount(); ++i) {
            if (i)
               s += ',';
            s += " " + getOperand(i).to_string();
         }
      }
      return s;
   }

   bool isVALU()
   {
      return ((uint16_t) format & (uint16_t) Format::VOP1) == (uint16_t) Format::VOP1
          || ((uint16_t) format & (uint16_t) Format::VOP2) == (uint16_t) Format::VOP2
          || ((uint16_t) format & (uint16_t) Format::VOPC) == (uint16_t) Format::VOPC
          || ((uint16_t) format & (uint16_t) Format::VOP3A) == (uint16_t) Format::VOP3A
          || ((uint16_t) format & (uint16_t) Format::VOP3B) == (uint16_t) Format::VOP3B
          || ((uint16_t) format & (uint16_t) Format::VOP3P) == (uint16_t) Format::VOP3P;
   }
   bool isSALU()
   {
      return ((uint16_t) format & (uint16_t) Format::SOP1) == (uint16_t) Format::SOP1
          || ((uint16_t) format & (uint16_t) Format::SOP2) == (uint16_t) Format::SOP1
          || ((uint16_t) format & (uint16_t) Format::SOPC) == (uint16_t) Format::SOP1
          || ((uint16_t) format & (uint16_t) Format::SOPK) == (uint16_t) Format::SOP1
          || ((uint16_t) format & (uint16_t) Format::SOPP) == (uint16_t) Format::SOPP;
   }
   bool isVMEM()
   {
      return ((uint16_t) format & (uint16_t) Format::MTBUF) == (uint16_t) Format::MTBUF
          || ((uint16_t) format & (uint16_t) Format::MUBUF) == (uint16_t) Format::MUBUF
          || ((uint16_t) format & (uint16_t) Format::MIMG) == (uint16_t) Format::MIMG;
   }
   bool isDPP()
   {
      return (uint16_t) format & (uint16_t) Format::DPP;
   }
};

struct SOPK_instruction : public Instruction {
   uint16_t imm;
};

struct SOPP_instruction : public Instruction {
   uint32_t imm;
   Block *block;
};

struct SOP1_instruction : public Instruction {
};

struct VOP1_instruction : public Instruction {
};

struct VOP2_instruction : public Instruction {
};

struct VOPC_instruction : public Instruction {
};

struct VOP3A_instruction : public Instruction {
   bool abs[3];
   bool opsel[3];
   bool clamp;
   unsigned omod;
   bool neg[3];
};

struct Interp_instruction : public Instruction {
   unsigned attribute;
   unsigned component;
};

struct Export_instruction : public Instruction {
   unsigned enabled_mask;
   unsigned dest;
   bool compressed;
   bool done;
   bool valid_mask;

};

template<typename T>
T* create_instruction(aco_opcode opcode, Format format, uint32_t num_operands, uint32_t num_definitions)
{
   std::size_t size = sizeof(T) + num_operands * sizeof(Operand) + num_definitions * sizeof(Definition);
   char *data = (char*)calloc(1, size);
   auto inst = (T*)data;

   inst->opcode = opcode;
   inst->format = format;
   inst->num_operands = num_operands;
   inst->num_definitions = num_definitions;

   inst->operands = (Operand*)(data + sizeof(T));
   inst->definitions = (Definition*)(data + sizeof(T) + num_operands * sizeof(Operand));

   return inst;
}

/* CFG */
struct Block {
   unsigned index;
   std::vector<std::unique_ptr<Instruction>> instructions;
   std::vector<Block*> logical_predecessors;
   std::vector<Block*> linear_predecessors;
   std::vector<Block*> logical_successors;
   std::vector<Block*> linear_successors;
};


class Program final {
public:
   std::vector<std::unique_ptr<Block>> blocks;
   ac_shader_config* config;

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
         (unsigned) blocks.size(),
         std::vector<std::unique_ptr<Instruction>>(),
         std::vector<Block*>(2),
         std::vector<Block*>(2),
         std::vector<Block*>(2),
         std::vector<Block*>(2),
      };
      b->linear_predecessors.push_back(blocks.back().get());
      blocks.back().get()->linear_successors.push_back(b);
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
      out << "\n";
   }

private:
   uint32_t allocationID = 1;
};

std::unique_ptr<Program> select_program(struct nir_shader *nir, ac_shader_config* config);
void register_allocation(Program *program);
void eliminate_pseudo_instr(Program* program);
void schedule(Program* program);
void insert_wait_states(Program* program);
std::vector<uint32_t> emit_program(Program* program);
void print_asm(std::vector<uint32_t>& binary, char* llvm_mc, std::ostream& out);
}
#endif /* __cplusplus */
#endif /* ACO_IR_H */

