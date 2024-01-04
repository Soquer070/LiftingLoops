#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace ll{

struct LiftingLoopsPass : public mlir::PassWrapper<LiftingLoopsPass, mlir::OperationPass<>> {
  //void printBlock(mlir::Block &block);
  //void printRegion(mlir::Region &region);
  //void printOperation(mlir::Operation *op); 
  //void contructStructure(mlir::Operation *op); 
  //void Inductioning(const mlir::CFGLoopInfo &loopInfo); 
  void runOnOperation() override; 
};

} // namespace ll

