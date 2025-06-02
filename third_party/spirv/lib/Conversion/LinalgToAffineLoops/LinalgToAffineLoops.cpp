#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "spirv/include/Conversion/LinalgToAffineLoops/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace mlir::triton::spirv {
#define GEN_PASS_DEF_LINALGTOAFFINELOOPS
#include "spirv/include/Conversion/LinalgToAffineLoops/Passes.h.inc"
} // namespace mlir::triton::spirv

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::spirv;

namespace {



struct LinalgToAffineLoops
    : public mlir::triton::spirv::impl::LinalgToAffineLoopsBase<LinalgToAffineLoops> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.dump();
  }
};

} // namespace

namespace mlir::triton::spirv {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToAffineLoopsPass() {
  return std::make_unique<LinalgToAffineLoops>();
}

} // namespace mlir::triton::spirv
