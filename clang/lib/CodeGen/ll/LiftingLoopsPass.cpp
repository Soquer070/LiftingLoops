#include "mlir/IR/Block.h"
#include "LiftingLoopsPass.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <ostream>
#include <utility>

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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Analysis/CFGLoopInfo.h"


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
  }

  void LiftingLoopsPass::printRegion(mlir::Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::dbgs() << "Region with " << region.getBlocks().size() << " blocks:\n";
    for (mlir::Block &block : region.getBlocks()){
      block.dump();
      llvm::dbgs() << "\n";
      printBlock(block);
      llvm::dbgs() << "\n";
    }
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

  void LiftingLoopsPass::contructStructure(mlir::Operation *op) {
    //op->dump();


    for (mlir::Region &region : op->getRegions()){
        llvm::dbgs() << "\nRegion argument num: " << region.getArguments().size();
      for (auto rArgs : region.getArguments()){
        llvm::dbgs() << "\nRegion argument num: " << rArgs.getArgNumber();
        llvm::dbgs() << "\nRegion argument owner block: " << rArgs.getOwner();
        //rArgs.isUsedOutsideOfBlock(Block *block)
      }
      //const unsigned long n= region.getBlocks().size();
      //uint i= 0;
      uint n= region.getBlocks().size();
      std::vector<mlir::Block> blocs = std::vector<mlir::Block> (n);// = region.getBlocks();
      std::vector<bool> vst = std::vector<bool> (n, false);
      //mlir::Block *CurrEntry = &blocs.front();
      
      //llvm::iplist<mlir::Block> CurrEntry();
      //  CurrEntry().addNodeToList(&block);
      std::vector<mlir::Block *> DFS;
      llvm::SmallPtrSet<mlir::Block *, 4> VisitedMap;
      llvm::DenseMap<mlir::Block *, unsigned int> dfsNumber;

      auto &first = region.getBlocks().front();

      llvm::dbgs() << "This block is parent: " << first.isEntryBlock() << "\n";
      //DFS.push_back(&first);
      //VisitedMap.insert(&first);
      //DFS.push_back(CurrEntry);
      //VisitedMap[CurrEntry] = true;
      
      std::vector<mlir::Block *> workList;
      workList.push_back(&first);
      while (!workList.empty()) {
        auto *ThisBB = workList.back();
        workList.pop_back();
        if(VisitedMap.contains(ThisBB))
          continue;
        VisitedMap.insert(ThisBB);
        dfsNumber.insert(std::make_pair(ThisBB, DFS.size()));
        DFS.push_back(ThisBB);
        for (auto *SI : ThisBB->getSuccessors()) { //fix me:: recorre a la inversa
          workList.push_back(SI);
        }
      }

      /*for (auto &b : region.getBlocks()) {
        assert(VisitedMap.contains(&b)&&"Caca, no visitado...");
        for(auto *s : b.getSuccessors()){
          s->dump();
        }
      }*/
          
      /*for (auto *b : DFS) {
        llvm::dbgs() << " Blocking::\n";
        b->dump();
      }*/
    }
  }

  void LiftingLoopsPass::runOnOperation() {
    /*
    llvm::dbgs() << "////////////////////////////\n" <<
                    "///////              ///////\n" <<
                    "/////// OnOperation  ///////\n" <<
                    "///////              ///////\n" <<
                    "////////////////////////////\n";
    */
    // Get the current operation being operated on.
    mlir::Operation *op = getOperation();
    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    
    mlir::Region &region = op->getRegion(0);//getFunctionBody();
    llvm::DominatorTreeBase<mlir::Block, false> &domTree = domInfo.getDomTree(&region);
    mlir::CFGLoopInfo loopInfo(domTree);
    
    if (loopInfo.getTopLevelLoops().empty())
      llvm::errs() << "no loops!!!!!\n";
    else{

      /*for (auto rArgs : region.getArguments()){
        llvm::dbgs() << "\nRegion argument num: " << rArgs.getArgNumber();
        llvm::dbgs() << "\nRegion argument owner block: " << rArgs.getOwner();
        //rArgs.isUsedOutsideOfBlock(Block *block)
        rArgs.dump();
      }llvm::dbgs() << "\n";
*/
      loopInfo.print(llvm::errs());

      // step 1: print for all blocks, the header
/*
      for (auto &b : region.getBlocks()){
        if (loopInfo.isLoopHeader(&b)){
          llvm::dbgs() << "Found header" << "\n";
          b.print(llvm::dbgs()); llvm::dbgs() << "\n";


          mlir::Block *up = b.getUniquePredecessor();
          if (up != NULL)
            llvm::dbgs() << "In unique predecessor\n";
          else
            llvm::dbgs() << "NO unique pred\n";
          
          std::vector<mlir::BlockArgument> bas;
          llvm::dbgs() << "\nNumArgs:" << b.getNumArguments() << "\n";
          for (auto &ba : b.getArguments()){
            ba.dump();
            bas.push_back(ba);
          }

          llvm::dbgs() << "\nOpSize:" << b.getOperations().size() << "\n";
          for (auto &op : b.getOperations()){
            op.dump();
            for (auto & at : op.getAttrs()){
              op.getAttr(at.getName()).dump();
            }
            auto opops = op.getOpOperands();
            for (auto &op : opops){
              llvm::dbgs() << "This is opop num: " << op.getOperandNumber() << "\n";
              //op.getNextOperandUsingThisValue().
            }
            //opops.take_while(PredicateT Pred);
          }

        }
        //else if ()
      }
*/

      // step 2: print latch nodes!
/*
      auto loop = loopInfo.getTopLevelLoops();
      for (auto *it : loopInfo.getTopLevelLoops()){
        for (auto *b : it->getBlocks()){
          if (it->isLoopLatch(b)){

              llvm::dbgs() << "Found loop latch" << "\n";
              b->print(llvm::dbgs()); llvm::dbgs() << "\n";


              mlir::Block *up = b->getUniquePredecessor();
              if (up != NULL)
                llvm::dbgs() << "In unique predecessor\n";
              else
                llvm::dbgs() << "NO unique pred\n";
              
              std::vector<mlir::BlockArgument> bas;
              llvm::dbgs() << "\nNumArgs:" << b->getNumArguments() << "\n";
              for (auto &ba : b->getArguments()){
                ba.dump();
                bas.push_back(ba);
              }

              llvm::dbgs() << "\nOpSize:" << b->getOperations().size() << "\n";
              for (auto &op : b->getOperations()){
                op.dump();
                for (auto & at : op.getAttrs()){
                  op.getAttr(at.getName()).dump();
                }
                auto opops = op.getOpOperands();
                for (auto &op : opops){
                  llvm::dbgs() << "This is opop num: " << op.getOperandNumber() << "\n";
                  //op.getNextOperandUsingThisValue().
                }
                //opops.take_while(PredicateT Pred);
              }
          }
        }
      }
*/

        llvm::dbgs() << "\nDumping loop! \n";
      for (auto *it : loopInfo.getTopLevelLoops()){
        //llvm::dbgs() << "\n inIterator geting blocks= " << it->getNumBlocks();
        for (auto *b : it->getBlocks()){
          std::vector<mlir::BlockArgument> bas; //Block arguments to keep track
          llvm::dbgs() << "\nBlock ";
          
          if (loopInfo.isLoopHeader(b)){
            llvm::dbgs() << " HEADER \n";
            //llvm::dbgs() << "\nNumArgs:" << b->getNumArguments() << "\n";
            for (auto &ba : b->getArguments()){
              if (ba.getDefiningOp() != NULL) {
                llvm::dbgs() << "Found somethig...\n";
                ba.getDefiningOp()->dump();
                llvm::dbgs() << "\n";
              }
              ba.dump();
              if (ba.isUsedOutsideOfBlock(b)){ 
                bas.push_back(ba);
              }
              else {
                llvm::dbgs() << "   ..... unused arg LOL\n";
              }
            }llvm::dbgs() << "\n";
          }
          else if (it->isLoopLatch(b))
            llvm::dbgs() << " LATCH ";
          else{
            llvm::dbgs() << " randomMofos ";
            for (auto arg : b->getArguments()){
              for (auto ba : bas){
                if (ba == arg){
                  llvm::dbgs() << "wut?\n;";
                  ba.dump();
                }
              }
            }
            llvm::dbgs() << "\n";
          }
          llvm::dbgs() << "\n";
          b->dump();
        }
      }
    }
  }
} // namespace ll
//mlir::CFGLoopInfo > latch -> termintaror -> 
// haura de ser cond > para cap al bloc basic, pista para la var de ind

// exemple amb dos bucles seguits
// exemple amb dos bucles anidats -> 