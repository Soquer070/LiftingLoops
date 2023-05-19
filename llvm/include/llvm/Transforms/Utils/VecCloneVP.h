//===---------- VecCloneVP.h - Class definition -*- C++ //-*---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// ===--------------------------------------------------------------------=== //
///
/// \file
/// This file defines the VecCloneVP pass class.
///
// ===--------------------------------------------------------------------=== //

#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <map>

#ifndef LLVM_TRANSFORMS_VPO_VECCLONE_H
#define LLVM_TRANSFORMS_VPO_VECCLONE_H

namespace llvm {

class ModulePass;

/// \brief Contains the names of the declared vector function variants
typedef std::vector<VFInfo> DeclaredVariants;

/// \brief Contains a mapping of a function to its vector function variants
typedef std::map<Function *, DeclaredVariants> FunctionVariants;

struct VecCloneVPPass : public PassInfoMixin<VecCloneVPPass> {
  MDNode* MemoryAccessGroup = nullptr;
public:
  /// \brief Get all functions marked for vectorization in module and their
  /// list of variants.
  void getFunctionsToVectorize(Module &M, FunctionVariants &FuncVars);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  // Glue for old PM
  bool runImpl(Module &M, Function &F, const VFInfo &Variant,
               std::function<AssumptionCache &(Function &F)> GetAC);

private:
  /// \brief Returns a floating point or integer constant depending on Ty.
  template <typename T>
  Constant *getConstantValue(Type *Ty, LLVMContext &Context, T Val);

  /// \brief Make a copy of the function if it requires a vector variant.
  Function *cloneFunction(Module &M, Function &F, const VFInfo &V);

  /// \brief Update the users of vector and linear parameters. Vector
  /// parameters must be now be indexed to reference the appropriate
  /// element and for linear parameters the stride will be added.
  void updateParameterUsers(Function *Clone, const VFInfo &Variant,
                            BasicBlock &EntryBlock, PHINode *Phi,
                            const DataLayout &DL, Value *Mask, Value *VL);

  /// \brief Performs a translation of a -> &a[i] for widened alloca
  /// instructions within the loop body of a simd function.
  void updateAllocaUsers(Function *Clone, PHINode *Phi,
                         DenseMap<AllocaInst *, Instruction *> &AllocaMap);

  /// \brief Widen alloca instructions. Vector parameters will have a vector
  /// alloca of size VF and and linear/uniform parameters will have an array
  /// alloca of size VF.
  void widenAllocaInstructions(Function *Clone,
                               DenseMap<AllocaInst *, Instruction *> &AllocaMap,
                               BasicBlock &EntryBlock, const VFInfo &Variant,
                               const DataLayout &DL, Value *Mask, Value *VL);

  /// \brief Remove any incompatible parameter attributes as a result of
  /// widening vector parameters.
  void removeIncompatibleAttributes(Function *Clone);

  /// \brief Add access group metadata to memory accessing instructions.
  void addAccessGroup(Function *Clone);

  /// \brief Check to see if the function is simple enough that a loop does
  /// not need to be inserted into the function.
  bool isSimpleFunction(Function *Clone, BasicBlock &EntryBlock);

  /// \brief Inserts the if/else split and mask condition for masked SIMD
  /// functions.
  void insertSplitForMaskedVariant(Function *Clone,
                                   const DataLayout &DL,
                                   BasicBlock *LoopBlock,
                                   BasicBlock *LoopExitBlock,
                                   AllocaInst *MaskAlloca, PHINode *Phi);

  /// \brief Adds metadata to the conditional branch of the simd loop latch to
  /// prevent loop unrolling and to force vectorization at VF.
  void addLoopMetadata(BasicBlock *Latch, ElementCount VF);
};

class VecCloneVP : public ModulePass {

  bool runOnModule(Module &M) override;

public:
  static char ID;
  VecCloneVP();
  void print(raw_ostream &OS, const Module * = nullptr) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  VecCloneVPPass Impl;

}; // end pass class

ModulePass *createVecCloneVPPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_VPO_VECCLONE_H
