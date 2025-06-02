#ifndef AFFINE_TO_LLVMSPV_CONVERSION_PASSES_H
#define AFFINE_TO_LLVMSPV_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

namespace triton::spirv {

#define GEN_PASS_DECL
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAffineToLLVMSPVPass();

#define GEN_PASS_REGISTRATION
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h.inc"

} // namespace triton::spirv
} // namespace mlir

#endif
