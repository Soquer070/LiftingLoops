#include "LiftingLoopsPass.h"
#include "mlir-c/IR.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include "llvm/Analysis/LoopInfo.h"

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"


#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"


#define DEBUG_TYPE "lifting-loops"

namespace ll{

  
struct MyPattern : public mlir::RewritePattern {
  public:
  MyPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::scf::ForOp::getOperationName(), /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

  void LiftingLoopsPass::runOnOperation() {
    mlir::Operation *op = getOperation();
    op->dump();
    assert(op->getNumRegions() == 1);
    mlir::Region &region = op->getRegion(0);
    
    llvm::dbgs() << " Dumping original blocks!\n";
    for (mlir::Block &b : region.getBlocks())
      //b.dump();
    llvm::dbgs() << " \n";
	
    mlir::BlockArgument indVar;
    mlir::Value lb, ub, step;
    mlir::Block *header;

    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    llvm::DominatorTreeBase<mlir::Block, false> &domTree = domInfo.getDomTree(&region);
    mlir::CFGLoopInfo loopInfo(domTree);
    assert(!loopInfo.getTopLevelLoopsVector().empty());
    
mlir::LoopLikeOpInterface loop;
    loopInfo.print(llvm::dbgs());
    llvm::dbgs() << " \n";
    
    for (auto *it : loopInfo.getTopLevelLoopsVector()){
      for (auto *b : it->getBlocks()){
        if (loopInfo.isLoopHeader(b)){
          header = b;
          for (auto &ba : b->getArguments()){
            if (ba.isUsedOutsideOfBlock(b)){
              indVar = ba;
              for (mlir::Block *predBlock : ba.getOwner()->getPredecessors()) {
                if (loopInfo.getLoopFor(predBlock) == nullptr){
                  if (mlir::Operation *terminator = predBlock->getTerminator()) {
                    lb = terminator->getOperand(ba.getArgNumber());
                  }
                }
              }
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



      mlir::Region &region = op->getRegion(0);
      mlir::Block &entryBlock = region.getBlocks().front();
      mlir::OpBuilder builder(op->getContext());

auto ip = builder.saveInsertionPoint();
builder.setInsertionPointAfter(getOperation());

      auto loc = header->begin()->getLoc();

      llvm::dbgs() << "preeeee:\n";
      //op->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
      getOperation()->getRegion(0).getOps().begin()->getLoc()->dump();
      getOperation()->getRegion(0).getOps().begin()->dump();

      auto forOp = builder.create<mlir::scf::ForOp>(getOperation()->getRegion(0).getOps().begin()->getLoc(), lb, ub, step);
      forOp->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
      builder.restoreInsertionPoint(ip);


      
    }
/*auto bodyBuilder = mlir::OpBuilder::atBlockEnd(forOp.getBody());
auto iv = forOp.getInductionVar();
auto ivType = iv.getType();
auto ptr = bodyBuilder.create<mlir::LLVM::IntToPtrOp>(entryBlock.front().getLoc(), ivType, iv);
auto storeOp = bodyBuilder.create<mlir::LLVM::StoreOp>(entryBlock.front().getLoc(), iv, ptr);

llvm::dbgs() << "pppp\n";
storeOp->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());

for (auto it : llvm::zip(entryBlock.front().getResults(), forOp.getResults())) {
std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
}

  llvm::dbgs() << "eeeend\n";
for (auto &b : region.getBlocks()){
for (auto &oper : b.getOperations()) {
  //oper.remove();
}
}
//entryBlock.erase();
*/

    
      llvm::dbgs() << "eeeend\n";

  }

} // namespace ll
