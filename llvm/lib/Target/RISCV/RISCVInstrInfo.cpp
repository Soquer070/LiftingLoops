//===-- RISCVInstrInfo.cpp - RISCV Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "RISCVGenInstrInfo.inc"

using namespace llvm;

RISCVInstrInfo::RISCVInstrInfo(RISCVSubtarget &STI)
    : RISCVGenInstrInfo(RISCV::ADJCALLSTACKDOWN, RISCV::ADJCALLSTACKUP),
      STI(STI) {}

unsigned RISCVInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                             int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case RISCV::LB:
  case RISCV::LBU:
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::LW:
  case RISCV::FLW:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLD:
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

unsigned RISCVInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::FSW:
  case RISCV::SD:
  case RISCV::FSD:
    break;
  }

  if (MI.getOperand(0).isFI() && MI.getOperand(1).isImm() &&
      MI.getOperand(1).getImm() == 0) {
    FrameIndex = MI.getOperand(0).getIndex();
    return MI.getOperand(2).getReg();
  }

  return 0;
}

void RISCVInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, unsigned DstReg,
                                 unsigned SrcReg, bool KillSrc) const {
  if (RISCV::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
    return;
  }

  // FPR->FPR
  unsigned Opc;
  if (RISCV::FPR32RegClass.contains(DstReg, SrcReg) ||
      RISCV::FPR64RegClass.contains(DstReg, SrcReg)) {
    if (RISCV::FPR64RegClass.contains(DstReg, SrcReg))
      Opc = RISCV::FSGNJ_D;
    else
      Opc = RISCV::FSGNJ_S;

    BuildMI(MBB, MBBI, DL, get(Opc), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // GPR->FPR
  if (RISCV::FPR32RegClass.contains(DstReg) &&
      RISCV::GPRRegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_W_X), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (RISCV::FPR64RegClass.contains(DstReg) &&
      RISCV::GPRRegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_D_X), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // FPR->GPR
  if (RISCV::GPRRegClass.contains(DstReg) &&
      RISCV::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_X_W), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (RISCV::GPRRegClass.contains(DstReg) &&
      RISCV::FPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_X_D), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // EPIVR->EPIVR, EPIVR2->EPIVR2, EPIVR4->EPIVR4, EPIVR8->EPIVR8
  if (RISCV::EPIVRRegClass.contains(DstReg, SrcReg) ||
      RISCV::EPIVR2RegClass.contains(DstReg, SrcReg) ||
      RISCV::EPIVR4RegClass.contains(DstReg, SrcReg) ||
      RISCV::EPIVR8RegClass.contains(DstReg, SrcReg)) {
    // readvl t0
    // readvtype t1
    // vsetvli x0, x0, ELEN, VLMUL
    // vmv.v.v dest, src
    // vsetvl x0, t0, t1
    RegScavenger RS;

    RS.enterBasicBlock(MBB);
    RS.forward(*MBBI);

    BitVector Available = RS.getRegsAvailable(&RISCV::GPRRegClass);

    int OldVTypeReg = -1;
    int OldVLReg = -1;
    SmallVector<Register, 2> RegistersToSpill;

    OldVTypeReg = Available.find_first();
    if (OldVTypeReg == -1) {
      // Use t0.
      OldVTypeReg = RISCV::X5;
      RegistersToSpill.push_back(OldVTypeReg);
    } else {
      OldVLReg = Available.find_next(OldVTypeReg);
    }

    if (OldVLReg == -1) {
      // Use t1.
      OldVLReg = RISCV::X6;
      RegistersToSpill.push_back(OldVLReg);
    }

    MachineFunction *MF = MBB.getParent();
    const TargetRegisterInfo &RI = *MF->getSubtarget().getRegisterInfo();

    // Emergency spill.
    if (!RegistersToSpill.empty()) {
      assert(RegistersToSpill.size() <= 2 && "Too many registers!");
      // Grow stack.
      BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), RISCV::X2)
          .addReg(RISCV::X2)
          .addImm(-(RegistersToSpill.size() *
                    RI.getSpillSize(RISCV::GPRRegClass)));
      int Idx = 0;
      for (auto &R : RegistersToSpill) {
        unsigned Opcode =
            RI.getRegSizeInBits(RISCV::GPRRegClass) == 32 ? RISCV::SW : RISCV::SD;
        BuildMI(MBB, MBBI, DL, get(Opcode))
          .addReg(R)
          .addReg(RISCV::X2)
          .addImm(Idx * RI.getSpillSize(RISCV::GPRRegClass));
        Idx++;
      }
    }

    unsigned VLMul;
    if (RISCV::EPIVRRegClass.contains(DstReg, SrcReg))
      VLMul = 0;
    else {
      if (RISCV::EPIVR2RegClass.contains(DstReg, SrcReg))
        VLMul = 1;
      else if (RISCV::EPIVR4RegClass.contains(DstReg, SrcReg))
        VLMul = 2;
      else {
        assert(RISCV::EPIVR8RegClass.contains(DstReg, SrcReg));
        VLMul = 3;
      }
      SrcReg = RI.getSubReg(SrcReg, RISCV::epivreven);
      DstReg = RI.getSubReg(DstReg, RISCV::epivreven);
      assert(SrcReg && DstReg && "Subregister does not exist");
    }

    BuildMI(MBB, MBBI, DL, get(RISCV::PseudoReadVTYPE), OldVTypeReg);
    BuildMI(MBB, MBBI, DL, get(RISCV::PseudoReadVL), OldVLReg);
    // Note: VL and VTYPE are alive here.
    BuildMI(MBB, MBBI, DL, get(RISCV::VSETVLI), RISCV::X0)
        .addReg(RISCV::X0)
        // FIXME - Hardcoded to SEW=64
        .addImm(3)
        .addImm(VLMul);
    BuildMI(MBB, MBBI, DL, get(RISCV::VMV_V_V), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    BuildMI(MBB, MBBI, DL, get(RISCV::VSETVL), RISCV::X0)
        .addReg(OldVLReg, RegState::Kill)
        .addReg(OldVTypeReg, RegState::Kill);

    // Emergency reload.
    if (!RegistersToSpill.empty()) {
      assert(RegistersToSpill.size() <= 2 && "Too many registers!");
      int Idx = 0;
      for (auto &R : RegistersToSpill) {
        unsigned Opcode =
            RI.getRegSizeInBits(RISCV::GPRRegClass) == 32 ? RISCV::LW : RISCV::LD;
        BuildMI(MBB, MBBI, DL, get(Opcode), R)
          .addReg(RISCV::X2)
          .addImm(Idx * RI.getSpillSize(RISCV::GPRRegClass));
        Idx++;
      }
      // Shrink stack.
      BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), RISCV::X2)
          .addReg(RISCV::X2)
          .addImm((RegistersToSpill.size() *
                    RI.getSpillSize(RISCV::GPRRegClass)));
    }
    return;
  }

  llvm_unreachable("Impossible reg-to-reg copy");
}

void RISCVInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned SrcReg, bool IsKill, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  RISCVMachineFunctionInfo *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();

  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  unsigned Opcode;

  if (RISCV::GPRRegClass.hasSubClassEq(RC))
    Opcode = TRI->getRegSizeInBits(RISCV::GPRRegClass) == 32 ?
             RISCV::SW : RISCV::SD;
  else if (RISCV::FPR32RegClass.hasSubClassEq(RC))
    Opcode = RISCV::FSW;
  else if (RISCV::FPR64RegClass.hasSubClassEq(RC))
    Opcode = RISCV::FSD;
  else if (RISCV::EPIVRRegClass.hasSubClassEq(RC)) {
    RVFI->setHasSpilledEPIVR();
    FrameInfo.setStackID(FI, TargetStackID::EPIVector);
    BuildMI(MBB, I, DL, get(RISCV::PseudoVSPILL))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI);
    return;
  } else
    llvm_unreachable("Can't store this register to stack slot");

  BuildMI(MBB, I, DL, get(Opcode))
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FI)
      .addImm(0);
}

void RISCVInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          unsigned DstReg, int FI,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  RISCVMachineFunctionInfo *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();

  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  unsigned Opcode;

  if (RISCV::GPRRegClass.hasSubClassEq(RC))
    Opcode = TRI->getRegSizeInBits(RISCV::GPRRegClass) == 32 ?
             RISCV::LW : RISCV::LD;
  else if (RISCV::FPR32RegClass.hasSubClassEq(RC))
    Opcode = RISCV::FLW;
  else if (RISCV::FPR64RegClass.hasSubClassEq(RC))
    Opcode = RISCV::FLD;
  else if (RISCV::EPIVRRegClass.hasSubClassEq(RC)) {
    RVFI->setHasSpilledEPIVR();
    FrameInfo.setStackID(FI, TargetStackID::EPIVector);
    BuildMI(MBB, I, DL, get(RISCV::PseudoVRELOAD), DstReg)
        .addFrameIndex(FI);
    return;
  } else
    llvm_unreachable("Can't load this register from stack slot");

  BuildMI(MBB, I, DL, get(Opcode), DstReg).addFrameIndex(FI).addImm(0);
}

void RISCVInstrInfo::movImm(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            const DebugLoc &DL, Register DstReg, uint64_t Val,
                            MachineInstr::MIFlag Flag) const {
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  bool IsRV64 = MF->getSubtarget<RISCVSubtarget>().is64Bit();
  Register SrcReg = RISCV::X0;
  Register Result = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  unsigned Num = 0;

  if (!IsRV64 && !isInt<32>(Val))
    report_fatal_error("Should only materialize 32-bit constants for RV32");

  RISCVMatInt::InstSeq Seq;
  RISCVMatInt::generateInstSeq(Val, IsRV64, Seq);
  assert(Seq.size() > 0);

  for (RISCVMatInt::Inst &Inst : Seq) {
    // Write the final result to DstReg if it's the last instruction in the Seq.
    // Otherwise, write the result to the temp register.
    if (++Num == Seq.size())
      Result = DstReg;

    if (Inst.Opc == RISCV::LUI) {
      BuildMI(MBB, MBBI, DL, get(RISCV::LUI), Result)
          .addImm(Inst.Imm)
          .setMIFlag(Flag);
    } else {
      BuildMI(MBB, MBBI, DL, get(Inst.Opc), Result)
          .addReg(SrcReg, RegState::Kill)
          .addImm(Inst.Imm)
          .setMIFlag(Flag);
    }
    // Only the first instruction has X0 as its source.
    SrcReg = Result;
  }
}

// The contents of values added to Cond are not examined outside of
// RISCVInstrInfo, giving us flexibility in what to push to it. For RISCV, we
// push BranchOpcode, Reg1, Reg2.
static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  Cond.push_back(MachineOperand::CreateImm(LastInst.getOpcode()));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

static unsigned getOppositeBranchOpcode(int Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case RISCV::BEQ:
    return RISCV::BNE;
  case RISCV::BNE:
    return RISCV::BEQ;
  case RISCV::BLT:
    return RISCV::BGE;
  case RISCV::BGE:
    return RISCV::BLT;
  case RISCV::BLTU:
    return RISCV::BGEU;
  case RISCV::BGEU:
    return RISCV::BLTU;
  }
}

bool RISCVInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // We can't handle blocks that end in an indirect branch.
  if (I->getDesc().isIndirectBranch())
    return true;

  // We can't handle blocks with more than 2 terminators.
  if (NumTerminators > 2)
    return true;

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = I->getOperand(0).getMBB();
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = I->getOperand(0).getMBB();
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

unsigned RISCVInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                      int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isUnconditionalBranch() &&
      !I->getDesc().isConditionalBranch())
    return 0;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 2;
}

// Inserts a branch into the end of the specific MachineBasicBlock, returning
// the number of instructions inserted.
unsigned RISCVInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "RISCV branch conditions have two components!");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  unsigned Opc = Cond[0].getImm();
  MachineInstr &CondMI =
      *BuildMI(&MBB, DL, get(Opc)).add(Cond[1]).add(Cond[2]).addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

unsigned RISCVInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                              MachineBasicBlock &DestBB,
                                              const DebugLoc &DL,
                                              int64_t BrOffset,
                                              RegScavenger *RS) const {
  assert(RS && "RegScavenger required for long branching");
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const auto &TM = static_cast<const RISCVTargetMachine &>(MF->getTarget());

  if (TM.isPositionIndependent())
    report_fatal_error("Unable to insert indirect branch");

  if (!isInt<32>(BrOffset))
    report_fatal_error(
        "Branch offsets outside of the signed 32-bit range not supported");

  // FIXME: A virtual register must be used initially, as the register
  // scavenger won't work with empty blocks (SIInstrInfo::insertIndirectBranch
  // uses the same workaround).
  Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  auto II = MBB.end();

  MachineInstr &LuiMI = *BuildMI(MBB, II, DL, get(RISCV::LUI), ScratchReg)
                             .addMBB(&DestBB, RISCVII::MO_HI);
  BuildMI(MBB, II, DL, get(RISCV::PseudoBRIND))
      .addReg(ScratchReg, RegState::Kill)
      .addMBB(&DestBB, RISCVII::MO_LO);

  RS->enterBasicBlockEnd(MBB);
  unsigned Scav = RS->scavengeRegisterBackwards(RISCV::GPRRegClass,
                                                LuiMI.getIterator(), false, 0);
  MRI.replaceRegWith(ScratchReg, Scav);
  MRI.clearVirtRegs();
  RS->setRegUsed(Scav);
  return 8;
}

bool RISCVInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() == 3) && "Invalid branch condition!");
  Cond[0].setImm(getOppositeBranchOpcode(Cond[0].getImm()));
  return false;
}

MachineBasicBlock *
RISCVInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  int NumOp = MI.getNumExplicitOperands();
  return MI.getOperand(NumOp - 1).getMBB();
}

bool RISCVInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                           int64_t BrOffset) const {
  // Ideally we could determine the supported branch offset from the
  // RISCVII::FormMask, but this can't be used for Pseudo instructions like
  // PseudoBR.
  switch (BranchOp) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case RISCV::BEQ:
  case RISCV::BNE:
  case RISCV::BLT:
  case RISCV::BGE:
  case RISCV::BLTU:
  case RISCV::BGEU:
    return isIntN(13, BrOffset);
  case RISCV::JAL:
  case RISCV::PseudoBR:
    return isIntN(21, BrOffset);
  }
}

unsigned RISCVInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();

  switch (Opcode) {
  default: { return get(Opcode).getSize(); }
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::DBG_VALUE:
    return 0;
  case RISCV::PseudoCALLReg:
  case RISCV::PseudoCALL:
  case RISCV::PseudoTAIL:
  case RISCV::PseudoLLA:
  case RISCV::PseudoLA:
  case RISCV::PseudoLA_TLS_IE:
  case RISCV::PseudoLA_TLS_GD:
    return 8;
  case TargetOpcode::INLINEASM:
  case TargetOpcode::INLINEASM_BR: {
    const MachineFunction &MF = *MI.getParent()->getParent();
    const auto &TM = static_cast<const RISCVTargetMachine &>(MF.getTarget());
    return getInlineAsmLength(MI.getOperand(0).getSymbolName(),
                              *TM.getMCAsmInfo());
  }
  }
}

std::pair<unsigned, unsigned>
RISCVInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  // No bitmap values yet
  return {TF, 0};
}

ArrayRef<std::pair<unsigned, const char *>>
RISCVInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace RISCVII;

  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_LO, "riscv-abs-lo"},
      {MO_HI, "riscv-abs-hi"},
      {MO_GOT_HI, "riscv-got-hi"},
      {MO_PCREL_HI, "riscv-pcrel-hi"},
      {MO_PCREL_LO, "riscv-pcrel-lo"},
      {MO_CALL, "riscv-call"},
      {MO_PLT, "riscv-plt"},
      {MO_TPREL_HI, "riscv-tprel-hi"},
      {MO_TPREL_ADD, "riscv-tprel-add"},
      {MO_TPREL_LO, "riscv-tprel-lo"},
      {MO_TLS_GOT_HI, "riscv-tls-got-hi"},
      {MO_TLS_GD_HI, "riscv-tls-gd-hi"},
  };

  return makeArrayRef(TargetFlags);
}

bool RISCVInstrInfo::isAsCheapAsAMove(const MachineInstr &MI) const {
  const unsigned Opcode = MI.getOpcode();
  switch(Opcode) {
    default:
      break;
    case RISCV::ADDI:
    case RISCV::ORI:
    case RISCV::XORI:
      return (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == RISCV::X0);
  }
  return MI.isAsCheapAsAMove();
}

bool RISCVInstrInfo::verifyInstruction(const MachineInstr &MI,
                                       StringRef &ErrInfo) const {
  const MCInstrInfo *MCII = STI.getInstrInfo();
  MCInstrDesc const &Desc = MCII->get(MI.getOpcode());

  for (auto &OI : enumerate(Desc.operands())) {
    unsigned OpType = OI.value().OperandType;
    if (OpType >= RISCVOp::OPERAND_FIRST_RISCV_IMM &&
        OpType <= RISCVOp::OPERAND_LAST_RISCV_IMM) {
      const MachineOperand &MO = MI.getOperand(OI.index());
      if (MO.isImm()) {
        int64_t Imm = MO.getImm();
        bool Ok;
        switch (OpType) {
        default:
          llvm_unreachable("Unexpected operand type");
        case RISCVOp::OPERAND_UIMM4:
          Ok = isUInt<4>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM5:
          Ok = isUInt<5>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM12:
          Ok = isUInt<12>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM12:
          Ok = isInt<12>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM13_LSB0:
          Ok = isShiftedInt<12, 1>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM20:
          Ok = isUInt<20>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM21_LSB0:
          Ok = isShiftedInt<20, 1>(Imm);
          break;
        case RISCVOp::OPERAND_UIMMLOG2XLEN:
          if (STI.getTargetTriple().isArch64Bit())
            Ok = isUInt<6>(Imm);
          else
            Ok = isUInt<5>(Imm);
          break;
        }
        if (!Ok) {
          ErrInfo = "Invalid immediate";
          return false;
        }
      }
    }
  }

  return true;
}

// Return true if get the base operand, byte offset of an instruction and the
// memory width. Width is the size of memory that is being loaded/stored.
bool RISCVInstrInfo::getMemOperandWithOffsetWidth(
    const MachineInstr &LdSt, const MachineOperand *&BaseReg, int64_t &Offset,
    unsigned &Width, const TargetRegisterInfo *TRI) const {
  assert(LdSt.mayLoadOrStore() && "Expected a memory operation.");

  // Here we assume the standard RISC-V ISA, which uses a base+offset
  // addressing mode. You'll need to relax these conditions to support custom
  // load/stores instructions.
  if (LdSt.getNumExplicitOperands() != 3)
    return false;
  if (!LdSt.getOperand(1).isReg() || !LdSt.getOperand(2).isImm())
    return false;

  if (!LdSt.hasOneMemOperand())
    return false;

  Width = (*LdSt.memoperands_begin())->getSize();
  BaseReg = &LdSt.getOperand(1);
  Offset = LdSt.getOperand(2).getImm();
  return true;
}

bool RISCVInstrInfo::areMemAccessesTriviallyDisjoint(
    const MachineInstr &MIa, const MachineInstr &MIb) const {
  assert(MIa.mayLoadOrStore() && "MIa must be a load or store.");
  assert(MIb.mayLoadOrStore() && "MIb must be a load or store.");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects() ||
      MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  // Retrieve the base register, offset from the base register and width. Width
  // is the size of memory that is being loaded/stored (e.g. 1, 2, 4).  If
  // base registers are identical, and the offset of a lower memory access +
  // the width doesn't overlap the offset of a higher memory access,
  // then the memory accesses are different.
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const MachineOperand *BaseOpA = nullptr, *BaseOpB = nullptr;
  int64_t OffsetA = 0, OffsetB = 0;
  unsigned int WidthA = 0, WidthB = 0;
  if (getMemOperandWithOffsetWidth(MIa, BaseOpA, OffsetA, WidthA, TRI) &&
      getMemOperandWithOffsetWidth(MIb, BaseOpB, OffsetB, WidthB, TRI)) {
    if (BaseOpA->isIdenticalTo(*BaseOpB)) {
      int LowOffset = std::min(OffsetA, OffsetB);
      int HighOffset = std::max(OffsetA, OffsetB);
      int LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
      if (LowOffset + LowWidth <= HighOffset)
        return true;
    }
  }
  return false;
}
