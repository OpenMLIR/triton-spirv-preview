#include "spirv/include/Conversion/TritonToLinalg/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace mlir::triton::spirv {
#define GEN_PASS_DEF_TRITONTOLINALG
#include "spirv/include/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton::spirv

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::spirv;

namespace {

struct TritonToLinalg
    : public mlir::triton::spirv::impl::TritonToLinalgBase<TritonToLinalg> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.dump();
  }
};

} // namespace

namespace mlir::triton::spirv {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalg>();
}

} // namespace mlir::triton::spirv
