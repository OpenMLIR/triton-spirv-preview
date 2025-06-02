#ifndef AFFINE_TO_LLVMSPV_CONVERSION_PASSES
#define AFFINE_TO_LLVMSPV_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AffineToLLVMSPV : Pass<"affine-to-llvmspv", "mlir::ModuleOp"> {
  let summary = "Convert affine to LLVMSPV dialect";
  let constructor = "mlir::triton::spirv::createAffineToLLVMSPVPass()";
}

#endif
