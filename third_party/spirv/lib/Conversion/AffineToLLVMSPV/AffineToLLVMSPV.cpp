#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-to-llvmspv"

namespace mlir::triton::spirv {
#define GEN_PASS_DEF_AFFINETOLLVMSPV
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h.inc"
} // namespace mlir::triton::spirv

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::spirv;

namespace {

struct AffineToLLVMSPV
    : public mlir::triton::spirv::impl::AffineToLLVMSPVBase<AffineToLLVMSPV> {

  void runOnOperation() override {
    getOperation().dump();
  }
};

} // namespace

namespace mlir::triton::spirv {

std::unique_ptr<OperationPass<ModuleOp>> createAffineToLLVMSPVPass() {
  return std::make_unique<AffineToLLVMSPV>();
}

} // namespace mlir::triton::spirv
