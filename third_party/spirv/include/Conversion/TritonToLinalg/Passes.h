#ifndef TRITON_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

namespace triton::spirv {

#define GEN_PASS_DECL
#include "spirv/include/Conversion/TritonToLinalg/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass();

#define GEN_PASS_REGISTRATION
#include "spirv/include/Conversion/TritonToLinalg/Passes.h.inc"

} // namespace triton::spirv
} // namespace mlir

#endif
