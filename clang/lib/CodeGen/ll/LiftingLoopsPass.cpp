#include "LiftingLoopsPass.h"
#include "mlir-c/IR.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#include "llvm/Analysis/LoopInfo.h"

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"


#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"


#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


//Holy dirty
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Operator.h"


#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <vector>

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"


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
    //op->dump();
    assert(op->getNumRegions() == 1);
    mlir::Region &region = op->getRegion(0);
    //llvm::dbgs() << " Dumping original blocks!\n";
    //for (mlir::Block &b : region.getBlocks())
      //b.dump();
    //llvm::dbgs() << " \n";
	
    mlir::BlockArgument indVar;
    mlir::Value lb, ub, step;

    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    llvm::DominatorTreeBase<mlir::Block, false> &domTree = domInfo.getDomTree(&region);
    mlir::CFGLoopInfo loopInfo(domTree);
    assert(!loopInfo.getTopLevelLoopsVector().empty());
    
    loopInfo.print(llvm::dbgs());
    llvm::dbgs() << " \n";
    mlir::Operation *terminator;
    mlir::Block *exitBlock;
    
    for (auto *loop : loopInfo.getTopLevelLoopsVector()){
      llvm::SmallVector<mlir::Block *> exitBlocks;
      loop->getExitBlocks(exitBlocks);
      exitBlock = exitBlocks[0];
      for (auto *b : loop->getBlocks()){
        if (loopInfo.isLoopHeader(b)){
          //header = b;
          for (auto &ba : b->getArguments()){
            if (ba.isUsedOutsideOfBlock(b)){
              indVar = ba;
              for (mlir::Block *predBlock : ba.getOwner()->getPredecessors()) {
                if (loopInfo.getLoopFor(predBlock) == nullptr){
                  terminator = predBlock->getTerminator();
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
        if (loop->isLoopLatch(b)){
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

      mlir::OpBuilder builder(op->getContext());
      auto ip = builder.saveInsertionPoint();

      llvm::dbgs() << "print op\n";
      getOperation()->getRegion(0).getParentOp()->dump();
      llvm::dbgs() << "print op\n";


      builder.setInsertionPoint(terminator);
      //getOperation()->getRegion(0).getOps().begin()->dump();
      auto loc = getOperation()->getRegion(0).getOps().begin()->getLoc();
      auto forOp = builder.create<mlir::scf::ForOp>(loc, lb, ub, step);
      builder.setInsertionPointToStart(&forOp->getRegion(0).front());
      mlir::scf::ExecuteRegionOp executeRegionOp = builder.create<mlir::scf::ExecuteRegionOp>(loc, mlir::TypeRange{});

      auto &region= executeRegionOp->getRegion(0);
      mlir::IRMapping mapper;
      for (auto *b : loop->getBlocks()){
        if (!(loopInfo.isLoopHeader(b) || loop->isLoopLatch(b)) ){
          auto &newBlock = region.emplaceBlock();
          mapper.map(b,&newBlock);
        }
      }

      //llvm::dbgs() << "Esto es indvAr\n";
      //forOp.getInductionVar().dump();
      mapper.map(indVar,forOp.getInductionVar());
      for (auto *b : loop->getBlocks()){
        if (!loopInfo.isLoopHeader(b) && !loop->isLoopLatch(b)){
          auto *newBlock = mapper.lookup(b);
          builder.setInsertionPointToEnd(newBlock);
          for (auto &ope : b->getOperations()){
            ope.dump();
            if (&ope != b->getTerminator()){ 
              mlir::Operation *newOpe = ope.clone(mapper);
              newBlock->push_back(newOpe); //sin esto!
              builder.setInsertionPointToEnd(newBlock);
              //newOpe->moveBefore(&loopYield);//esto no va.... pero me deja un bloque vacio....
            }
            else {
              builder.create<mlir::scf::YieldOp>(loc);
            }
          }
        }
      }
      builder.setInsertionPoint(terminator);
      auto newTerminator = builder.create<mlir::cf::BranchOp>(loc, exitBlock);
      builder.setInsertionPoint(newTerminator);
      terminator->erase();
      std::vector<mlir::Block*> listOfBlocks;
      for (auto *b : loop->getBlocks()){
        // if (!loopInfo.isLoopHeader(b) && !loop->isLoopLatch(b))
        {
          listOfBlocks.push_back(b);
          
        }
      }
      for (auto *b : listOfBlocks){
        std::vector<mlir::Operation*> rev_ops;
        for(auto it = b->rbegin(); it != b->rend(); it++){
          auto &op = *it;
          rev_ops.push_back(&op);
        }
        for (auto *op : rev_ops){
          op->dump();
          op->dropAllUses();
          op->erase();
        }
      }
      for (auto *b : listOfBlocks){
        //b->dump();
        //b->walk([] (mlir::Operation *op){op->dump(); op->erase();});
        b->dump();
        b->dropAllUses();
        b->erase();
      }
      forOp->moveBefore(newTerminator);
      builder.restoreInsertionPoint(ip);
    }
    llvm::dbgs() << "\nend!\n";
    op = getOperation();
     op->dump();
    llvm::dbgs() << "\nend!\n";
  }

} // namespace ll
