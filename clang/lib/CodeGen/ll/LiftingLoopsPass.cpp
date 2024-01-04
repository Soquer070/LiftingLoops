#include "LiftingLoopsPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "lifting-loops"

namespace ll{

  void LiftingLoopsPass::runOnOperation() {
    mlir::Operation *op = getOperation();
    assert(op->getNumRegions() == 1);
    mlir::Region &region = op->getRegion(0);
    
    llvm::dbgs() << " Dumping original blocks!\n";
    for (auto &b : region.getBlocks())
      b.dump();
    llvm::dbgs() << " \n";
	
    // Initialize variables for loop information
    mlir::BlockArgument indVar;
    mlir::Value lb, ub, step;

    // Get dominance information and construct loop information
    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    llvm::DominatorTreeBase<mlir::Block, false> &domTree = domInfo.getDomTree(&region);
    mlir::CFGLoopInfo loopInfo(domTree);
    assert(!loopInfo.getTopLevelLoopsVector().empty());
    
    loopInfo.print(llvm::dbgs());
    llvm::dbgs() << " \n";
    
    // Extract loop properties
    for (auto *it : loopInfo.getTopLevelLoopsVector()){
      for (auto *b : it->getBlocks()){
        if (loopInfo.isLoopHeader(b)){
          // Extract loop induction variable and bounds
          for (auto &ba : b->getArguments()){
            if (ba.isUsedOutsideOfBlock(b)){
              indVar = ba;
              // Extract lower bound from the terminator of the predecessor block
              for (mlir::Block *predBlock : ba.getOwner()->getPredecessors()) {
                if (loopInfo.getLoopFor(predBlock) == nullptr){
                  if (mlir::Operation *terminator = predBlock->getTerminator()) {
                    lb = terminator->getOperand(ba.getArgNumber());
                  }
                }
              }
              // Extract upper bound from the loop predicate compare
              for (auto *op : ba.getUsers()){
                for (mlir::NamedAttribute arg : op->getAttrs()){
                  if (arg.getName() == "predicate" && arg.getValue().isa<mlir::IntegerAttr>()) {
                    auto *opOperand = &op->getOpOperand(1);
                    ub = opOperand->get();
                    if (arg.getValue().cast<mlir::IntegerAttr>().getInt() == 3){ //llvm::dbgs() << "SLeeeee!";
                      mlir::OpBuilder builder(op);
                      mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(op->getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
                      ub = builder.create<mlir::LLVM::AddOp>(op->getLoc(), one.getType(), opOperand->get(), one);
                    }
                  }
                }
              }
            }
          }
        }
        // Extract step value from the loop latch block
        if (it->isLoopLatch(b)){
          for (auto &op : b->getOperations()){
            if (op.getOpOperands().size() == 2){
              for (auto opo : op.getOperands()){
                if (opo != indVar){
                  step = llvm::dyn_cast<mlir::Value>(opo);
                }
              }
            }
          }
        }
      }
      
      llvm::dbgs() << "IndVar: " << indVar << "\n";
      llvm::dbgs() << "UpperBound: " << ub << "\n";
      llvm::dbgs() << "LowerBound: " << lb << "\n";
      llvm::dbgs() << "Step: " << step << "\n";

    }
  }

} // namespace ll
