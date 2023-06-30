//===- InstCombineVPOps.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitVP* functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"

#define DEBUG_TYPE "instcombine"

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool> OptimizeVectorGEP("optimize-vector-gep",
                                       cl::desc("Optimize vector GEP"),
                                       cl::init(true));

static cl::opt<bool> ExpandVectorGEPToVP("expand-vector-gep-to-vp",
                                         cl::desc("Expand vector GEP to VP"),
                                         cl::init(true));

Value *createIntExtOrTruncVP(VectorBuilder &VB, Value *From, Type *ToTy,
                             bool IsSigned) {
  Type *FromTy = From->getType();
  assert(FromTy->isIntOrIntVectorTy() && ToTy->isIntOrIntVectorTy() &&
         "Invalid integer cast");
  if (FromTy == ToTy)
    return From;

  unsigned SrcBits = FromTy->getScalarSizeInBits();
  unsigned DstBits = ToTy->getScalarSizeInBits();
  assert(SrcBits != DstBits);
  Instruction::CastOps Opcode =
      SrcBits > DstBits ? Instruction::Trunc
                        : (IsSigned ? Instruction::SExt : Instruction::ZExt);

  return VB.createVectorInstruction(Opcode, ToTy, {From});
}

Value *InstCombinerImpl::tryToOptimizeGEP(GetElementPtrInst &GEP) {
  if (!OptimizeVectorGEP)
    return nullptr;

  // Pointer operand must be a scalar.
  Value *PtrOp = GEP.getPointerOperand();
  Type *PtrOpTy = GEP.getPointerOperandType();
  if (PtrOpTy->isVectorTy()) {
    if (!isSplatValue(PtrOp)) {
      // Special case: the base pointer is the result of a GEP.
      if (auto *BaseFromGEP = dyn_cast<GetElementPtrInst>(PtrOp)) {
        // Use the base pointer of this GEP as the one for the optimized GEP; as
        // indices, sum together the ones from this GEP and the ones from the
        // to-be-optimized one.

        // We only try to optimize GEPs with just one index operand.
        if (GEP.getNumIndices() > 1 || BaseFromGEP->getNumIndices() > 1)
          return nullptr;
        // Ensure the source element type is the same for both GEPS.
        if (GEP.getSourceElementType() != BaseFromGEP->getSourceElementType())
          return nullptr;

        ElementCount EC = cast<VectorType>(PtrOpTy)->getElementCount();
        Value *Idx = GEP.getOperand(GEP.getPointerOperandIndex() + 1);
        if (!Idx->getType()->isVectorTy())
          Idx = Builder.CreateVectorSplat(EC, Idx);

        Value *BaseGEPIdx =
            BaseFromGEP->getOperand(BaseFromGEP->getPointerOperandIndex() + 1);
        if (!BaseGEPIdx->getType()->isVectorTy())
          BaseGEPIdx = Builder.CreateVectorSplat(EC, BaseGEPIdx);

        Type *IdxTy = Idx->getType();
        Type *BaseGEPIdxTy = BaseGEPIdx->getType();
        if (IdxTy != BaseGEPIdxTy) {
          unsigned int IdxElemSize = IdxTy->getScalarSizeInBits();
          unsigned int BaseGEPIdxElemSize = BaseGEPIdxTy->getScalarSizeInBits();
          assert(IdxElemSize != BaseGEPIdxElemSize);
          if (IdxElemSize > BaseGEPIdxElemSize)
            BaseGEPIdx = Builder.CreateSExt(BaseGEPIdx, IdxTy);
          else
            Idx = Builder.CreateSExt(Idx, BaseGEPIdxTy);
        }
        assert(Idx->getType() == BaseGEPIdx->getType() &&
               "Indices types shouldn't differ!");

        Value *NewBasePtr = BaseFromGEP->getPointerOperand();
        Value *NewIdx = Builder.CreateAdd(BaseGEPIdx, Idx);

        return Builder.CreateGEP(GEP.getSourceElementType(), NewBasePtr, NewIdx,
                                 "gep.opt", GEP.isInBounds());
      }

      // In all other cases, we bail out.
      return nullptr;
    }

    PtrOp = getSplatValue(PtrOp);
  }
  assert(PtrOp && !PtrOpTy->isVectorTy() &&
         "Invalid ptr operand for getelementptr instruction");

  // We only try to optimize GEPs with just one index vector operand.
  if (GEP.getNumIndices() > 1)
    return nullptr;
  Value *Idx = GEP.getOperand(GEP.getPointerOperandIndex() + 1);
  Type *IdxTy = Idx->getType();
  if (!IdxTy->isVectorTy())
    return nullptr;

  auto IndexIsA = [] (Value *V, unsigned int Opcode) -> bool {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;
    auto *CI = dyn_cast<CallInst>(I);
    if (I->getOpcode() == Opcode ||
        (CI && CI->getCalledFunction()->getIntrinsicID() ==
                   VPIntrinsic::getForOpcode(Opcode))) {
      return true;
    }
    return false;
  };

  if (IndexIsA(Idx, Instruction::SExt))
    Idx = cast<Instruction>(Idx)->getOperand(0);
  Value *NewBasePtr = nullptr;
  Value *NewIdx = nullptr;
  if (IndexIsA(Idx, Instruction::Add) ||
             IndexIsA(Idx, Instruction::Sub)) {
    auto *I = cast<Instruction>(Idx);
    Value *LHS = I->getOperand(0);
    Value *RHS = I->getOperand(1);

    Value *Offset = getSplatValue(LHS);
    NewIdx = RHS;
    if (!Offset) {
      Offset = getSplatValue(RHS);
      NewIdx = LHS;
    }
    if (!Offset)
      return nullptr;

    NewBasePtr = Builder.CreateGEP(GEP.getSourceElementType(), PtrOp, Offset,
                                   "base.ptr", GEP.isInBounds());
  } else {
    return nullptr;
  }

  return Builder.CreateGEP(GEP.getSourceElementType(), NewBasePtr, NewIdx,
                           "gep.opt", GEP.isInBounds());
}

Value *InstCombinerImpl::emitGEPOffsetVP(GetElementPtrInst &GEP,
                                         VectorBuilder &VB) {
  // Get the right integer type needed to represent a pointer.
  Type *IntPtrTy = Builder.getIntPtrTy(DL);

  // Build a mask for high order bits.
  unsigned IntPtrWidth = IntPtrTy->getIntegerBitWidth();
  uint64_t PtrSizeMask =
      std::numeric_limits<uint64_t>::max() >> (64 - IntPtrWidth);

  auto *IntIdxTy = VectorType::get(IntPtrTy, cast<VectorType>(GEP.getType()));
  Value *Result = nullptr;
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end(); I != E;
       ++I, ++GTI) {
    Value *Op = *I;
    uint64_t Size = DL.getTypeAllocSize(GTI.getIndexedType()) & PtrSizeMask;
    Value *Offset = nullptr;
    if (auto *OpC = dyn_cast<Constant>(Op)) {
      if (OpC->isZeroValue())
        continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        uint64_t OpValue = OpC->getUniqueInteger().getZExtValue();
        Size = DL.getStructLayout(STy)->getElementOffset(OpValue);
        if (!Size)
          continue;

        Offset = ConstantInt::get(IntIdxTy, Size);
      }
    }
    if (!Offset) {
      // Splat the index if needed.
      if (!Op->getType()->isVectorTy())
        Op = Builder.CreateVectorSplat(IntIdxTy->getElementCount(), Op);
      // Convert to correct type.
      if (Op->getType() != IntIdxTy)
        Op = createIntExtOrTruncVP(VB, Op, IntIdxTy, /*IsSigned*/ true);
      if (Size != 1)
        // We'll let instcombine(mul) convert this to a shl if possible.
        Op = VB.createVectorInstruction(Instruction::Mul, IntIdxTy,
                                        {Op, ConstantInt::get(IntIdxTy, Size)});

      Offset = Op;
    }

    if (Result)
      Result = VB.createVectorInstruction(Instruction::Add, IntIdxTy,
                                          {Result, Offset});
    else
      Result = Offset;
  }

  return Result ? Result : ConstantInt::get(IntIdxTy, 0);
}

Instruction *InstCombinerImpl::visitVPInst(VPIntrinsic *VPI) {
  switch (VPI->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::vp_icmp:
    return visitVPICmp(VPI);
  case Intrinsic::vp_mul:
    return visitVPMul(VPI);
  case Intrinsic::vp_select:
    return visitVPSelect(VPI);
  }
  return nullptr;
}

Instruction *
InstCombinerImpl::visitVPGatherScatterOnlyGEP(GetElementPtrInst &GEP) {
  if (!OptimizeVectorGEP && !ExpandVectorGEPToVP)
    return nullptr;
  if (!GEP.getType()->isVectorTy())
    return nullptr;

  // Check if the GEP is a viable candidate:
  // - Check that all uses of this GEP are the pointer operand of either a
  // vp.gather or vp.scatter intrinsic.
  // - Check that all users have the same mask and the same VL.
  Value *Mask = nullptr;
  Value *VL = nullptr;
  for (auto &GEPUse : GEP.uses()) {
    User *GEPUser = GEPUse.getUser();
    auto *VPI = dyn_cast<VPIntrinsic>(GEPUser);
    if (!VPI)
      return nullptr;
    if (VPI->getIntrinsicID() != Intrinsic::vp_gather &&
        VPI->getIntrinsicID() != Intrinsic::vp_scatter)
      return nullptr;
    // The GEP must be the pointer operand of the gather/scatter.
    if (GEPUse != VPI->getMemoryPointerParam())
      return nullptr;
    // If either the mask or VL do not dominate the GEP, for now ignore this
    // case. FIXME: It looks like it should be possible to insert the VP "gep"
    // right after the definiton of the mask or the VL.
    if (!DT.dominates(VPI->getMaskParam(), &GEP) ||
        !DT.dominates(VPI->getVectorLengthParam(), &GEP))
      return nullptr;
    if (!Mask && !VL) {
      Mask = VPI->getMaskParam();
      VL = VPI->getVectorLengthParam();
    } else {
      if (VPI->getMaskParam() != Mask || VPI->getVectorLengthParam() != VL)
        return nullptr;
    }
  }

  if (Value *OptGEP = tryToOptimizeGEP(GEP))
    return replaceInstUsesWith(GEP, OptGEP);

  if (!ExpandVectorGEPToVP)
    return nullptr;

  // Replace the GEP with VP instrinsics:
  // - vp.ptrtoint
  // - vp.add to calculate new vector of pointers
  // - vp.inttoptr
  // N.B.: indices are correctly scaled to represent the number of bytes when
  // calculating the offset in VPEmitGEPOffset().
  VectorBuilder VB(Builder);
  VB.setMask(ConstantInt::getAllOnesValue(Mask->getType()));
  VB.setEVL(VL);
  auto *GEPType = cast<VectorType>(GEP.getType());
  Value *PtrOp = GEP.getPointerOperand();
  Value *Offset = emitGEPOffsetVP(GEP, VB);
  auto *PtrSizeVecTy = VectorType::get(Builder.getIntPtrTy(DL), GEPType);
  if (!PtrOp->getType()->isVectorTy())
    PtrOp = Builder.CreateVectorSplat(GEPType->getElementCount(), PtrOp);
  if (!Offset->getType()->isVectorTy())
    Offset = Builder.CreateVectorSplat(GEPType->getElementCount(), Offset);
  if (Offset->getType() != PtrSizeVecTy)
    Offset = createIntExtOrTruncVP(VB, Offset, PtrSizeVecTy, /*IsSigned*/ true);

  Value *PtrToInt =
      VB.createVectorInstruction(Instruction::PtrToInt, PtrSizeVecTy, {PtrOp});
  Value *Add = VB.createVectorInstruction(Instruction::Add, PtrSizeVecTy,
                                          {PtrToInt, Offset});
  Value *IntToPtr =
      VB.createVectorInstruction(Instruction::IntToPtr, GEPType, {Add});

  return replaceInstUsesWith(GEP, IntToPtr);
}

Instruction *InstCombinerImpl::visitVPICmp(VPIntrinsic *VPICmp) {
  auto *VPCmp = cast<VPCmpIntrinsic>(VPICmp);
  if (VPCmp->getPredicate() == ICmpInst::ICMP_NE) {
    if (match(VPCmp->getOperand(1), m_Zero())) {
      // If first operand is just a zext of a mask, this then ICmp is just a
      // truncation which results in that same mask.
      // In order to be conservative, we require that both mask and vl are the
      // same for this icmp and the zext.
      auto *Op0 = dyn_cast<VPIntrinsic>(VPCmp->getOperand(0));
      if (Op0 && Op0->getIntrinsicID() == Intrinsic::vp_zext &&
          Op0->getOperand(0)->getType()->getScalarSizeInBits() == 1) {
        // Here we have a expand/truncate sequence over a mask vector.
        if (Op0->getMaskParam() == VPCmp->getMaskParam() &&
            Op0->getVectorLengthParam() == VPCmp->getVectorLengthParam())
          return replaceInstUsesWith(*VPICmp, Op0->getOperand(0));
      }
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitVPMul(VPIntrinsic *VPMul) {
  auto *RetTy = cast<ScalableVectorType>(VPMul->getType());
  Value *Op0 = VPMul->getOperand(0), *Op1 = VPMul->getOperand(1);
  if (isa<Constant>(Op0) && !isa<Constant>(Op1)) {
    // FIXME: we should have canonicalised this earlier.
    std::swap(Op0, Op1);
  }

  if (auto *CI = dyn_cast_or_null<ConstantInt>(getSplatValue(Op1))) {
    if (auto *Log2CI = ConstantExpr::getExactLogBase2(CI)) {
      auto *NewOp1 =
          Builder.CreateVectorSplat(RetTy->getElementCount(), Log2CI);
      VectorBuilder VB(Builder);
      VB.setMask(VPMul->getMaskParam());
      VB.setEVL(VPMul->getVectorLengthParam());
      Value *Shl =
          VB.createVectorInstruction(Instruction::Shl, RetTy, {Op0, NewOp1});
      return replaceInstUsesWith(*VPMul, Shl);
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitVPSelect(VPIntrinsic *VPSelect) {
  Type *RetType = VPSelect->getType();
  Value *CondVal = VPSelect->getArgOperand(0);
  Value *TrueVal = VPSelect->getArgOperand(1);
  Value *FalseVal = VPSelect->getArgOperand(2);
  Value *VL = VPSelect->getVectorLengthParam();

  // If true and false values are the same, no need for the select.
  if (TrueVal == FalseVal)
    return replaceInstUsesWith(*VPSelect, TrueVal);

  // No need for a select when the cond is an allzeros or allones vector.
  if (match(CondVal, m_One()))
    return replaceInstUsesWith(*VPSelect, TrueVal);
  if (match(CondVal, m_Zero()))
    return replaceInstUsesWith(*VPSelect, FalseVal);

  // Merge two selects with the same condition value.
  if (auto *PrevVPSelect = dyn_cast<VPIntrinsic>(FalseVal))
    if (PrevVPSelect->getIntrinsicID() == Intrinsic::vp_select &&
        PrevVPSelect->getArgOperand(0) == CondVal &&
        PrevVPSelect->getVectorLengthParam() == VL)
      return replaceOperand(*VPSelect, 2, PrevVPSelect->getArgOperand(2));

  if (RetType->isIntOrIntVectorTy(1) &&
      CondVal->getType() == TrueVal->getType()) {
    VectorBuilder VB(Builder);
    VB.setMask(ConstantInt::getAllOnesValue(CondVal->getType()));
    VB.setEVL(VL);

    // If TrueVal is an allones vector, transform the select in an or.
    if (match(TrueVal, m_One())) {
      Value *Or = VB.createVectorInstruction(Instruction::Or, RetType,
                                             {CondVal, FalseVal});
      return replaceInstUsesWith(*VPSelect, Or);
    }

    // If FalseVal is an allzeros vector, transform the select in an and.
    if (match(FalseVal, m_Zero())) {
      Value *And = VB.createVectorInstruction(Instruction::And, RetType,
                                              {CondVal, TrueVal});
      return replaceInstUsesWith(*VPSelect, And);
    }
  }

  return nullptr;
}
