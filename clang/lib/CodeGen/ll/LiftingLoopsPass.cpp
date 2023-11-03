#include "mlir/Pass/Pass.h"
#include "LiftingLoopsPass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"



#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"

//Added
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Region.h"

#define DEBUG_TYPE "lifting-loops"

namespace ll{
void LiftingLoopsPass::printBlock(mlir::Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    llvm::dbgs()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    for (mlir::Operation &op : block.getOperations())
      printOperation(&op);
  }

  void LiftingLoopsPass::printRegion(mlir::Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::dbgs() << "Region with " << region.getBlocks().size() << " blocks:\n";
    for (mlir::Block &block : region.getBlocks())
    block.dump();
      //printBlock(block);
  }

  void LiftingLoopsPass::printOperation(mlir::Operation *op) {
    // Print the operation itself and some of its properties
    llvm::dbgs() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      llvm::dbgs() << op->getAttrs().size() << " attributes:\n";
      //for (mlir::NamedAttribute attr : op->getAttrs())
        //llvm::dbgs() << "visiting attr: '" << attr.getName() << "' with value: " << attr.getValue() << "\n";
    }

    // Recurse into each of the regions attached to the operation.
    llvm::dbgs() << " " << op->getNumRegions() << " nested regions:\n";
    //auto indent = pushIndent();
    for (mlir::Region &region : op->getRegions())
      printRegion(region);
      //llvm::dbgs() << " i just wawnt to check things...- '" << "'\n";//<< region->dump() 
  }

  void LiftingLoopsPass::runOnOperation() {
    llvm::dbgs() << "////////////////////////////\n" <<
                    "///////              ///////\n" <<
                    "/////// OnOperation  ///////\n" <<
                    "///////              ///////\n" <<
                    "////////////////////////////\n";
    // Get the current operation being operated on.
    mlir::Operation *op = getOperation();
    printOperation(op);
    //op->dump();
  }
} // namespace ll

  
