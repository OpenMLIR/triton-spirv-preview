#ifndef LINALG_TO_AFFINE_LOOPS_CONVERSION_PASSES
#define LINALG_TO_AFFINE_LOOPS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToAffineLoops : Pass<"linalg-to-affine-loops", "mlir::ModuleOp"> {
  let summary = "Convert linalg to affine dialect";
  let constructor = "mlir::triton::spirv::createLinalgToAffineLoopsPass()";
}

#endif
