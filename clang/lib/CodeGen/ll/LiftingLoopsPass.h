#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace ll{

/*const int MAX_CANT_NODES = 1000;
std::vector<int> graph[MAX_CANT_NODES]; // Lista de adjacencia
bool vst[MAX_CANT_NODES]; // Vector de nodos visitados
*/

struct LiftingLoopsPass : public mlir::PassWrapper<LiftingLoopsPass, mlir::OperationPass<>> {
  void printBlock(mlir::Block &block);
  void printRegion(mlir::Region &region);
  void printOperation(mlir::Operation *op); 
  void contructStructure(mlir::Operation *op); 
  void Inductioning(const mlir::CFGLoopInfo &loopInfo); 
  void runOnOperation() override; 
};

} // namespace ll

