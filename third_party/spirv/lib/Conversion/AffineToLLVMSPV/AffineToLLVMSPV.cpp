#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable,
                                              StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType, bool isMemNone,
                                              bool isConvergent) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (!func) {
    OpBuilder b(symbolTable->getRegion(0));
    func = b.create<LLVM::LLVMFuncOp>(
        symbolTable->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes));
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    func.setNoUnwind(true);
    func.setWillReturn(true);

    if (isMemNone) {
      // no externally observable effects
      constexpr auto noModRef = mlir::LLVM::ModRefInfo::NoModRef;
      auto memAttr = b.getAttr<LLVM::MemoryEffectsAttr>(
          /*other=*/noModRef,
          /*argMem=*/noModRef, /*inaccessibleMem=*/noModRef);
      func.setMemoryEffectsAttr(memAttr);
    }

    func.setConvergent(isConvergent);
  }
  return func;
}

static LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           LLVM::LLVMFuncOp func,
                                           ValueRange args) {
  auto call = rewriter.create<LLVM::CallOp>(loc, func, args);
  call.setCConv(func.getCConv());
  call.setConvergentAttr(func.getConvergentAttr());
  call.setNoUnwindAttr(func.getNoUnwindAttr());
  call.setWillReturnAttr(func.getWillReturnAttr());
  call.setMemoryEffectsAttr(func.getMemoryEffectsAttr());
  return call;
}

//===----------------------------------------------------------------------===//
// SPIR-V Builtins
//===----------------------------------------------------------------------===//

/// Replace `gpu.*` with an `llvm.call` to the corresponding SPIR-V builtin with
/// a constant argument for the `dimension` attribute. Return type will depend
/// on index width option:
/// ```
/// // %thread_id_y = gpu.thread_id y
/// %c1 = llvm.mlir.constant(1: i32) : i32
/// %0 = llvm.call spir_funccc @_Z12get_local_idj(%c1) : (i32) -> i64
/// ```
struct LaunchConfigConversion : ConvertToLLVMPattern {
  LaunchConfigConversion(StringRef funcName, StringRef rootOpName,
                         MLIRContext *context,
                         const LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertToLLVMPattern(rootOpName, context, typeConverter, benefit),
        funcName(funcName) {}

  virtual gpu::Dimension getDimension(Operation *op) const = 0;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expecting module");
    Type dimTy = rewriter.getI32Type();
    Type indexTy = getTypeConverter()->getIndexType();
    LLVM::LLVMFuncOp func = lookupOrCreateSPIRVFn(moduleOp, funcName, dimTy,
                                                  indexTy, /*isMemNone=*/true,
                                                  /*isConvergent=*/false);

    Location loc = op->getLoc();
    gpu::Dimension dim = getDimension(op);
    Value dimVal = rewriter.create<LLVM::ConstantOp>(loc, dimTy,
                                                     static_cast<int64_t>(dim));
    rewriter.replaceOp(op, createSPIRVBuiltinCall(loc, rewriter, func, dimVal));
    return success();
  }

  StringRef funcName;
};

template <typename SourceOp>
struct LaunchConfigOpConversion final : LaunchConfigConversion {
  static StringRef getFuncName();

  explicit LaunchConfigOpConversion(const LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit = 1)
      : LaunchConfigConversion(getFuncName(), SourceOp::getOperationName(),
                               &typeConverter.getContext(), typeConverter,
                               benefit) {}

  gpu::Dimension getDimension(Operation *op) const final {
    return cast<SourceOp>(op).getDimension();
  }
};

template <> StringRef LaunchConfigOpConversion<gpu::GlobalIdOp>::getFuncName() {
  return "_Z13get_global_idj";
}

struct AffineToLLVMSPV
    : public mlir::triton::spirv::impl::AffineToLLVMSPVBase<AffineToLLVMSPV> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    LLVMConversionTarget target(*context);
    LLVMTypeConverter converter(context);
    target.addIllegalOp<gpu::GlobalIdOp>();
    patterns.add<LaunchConfigOpConversion<gpu::GlobalIdOp>>(converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spirv {

std::unique_ptr<OperationPass<ModuleOp>> createAffineToLLVMSPVPass() {
  return std::make_unique<AffineToLLVMSPV>();
}

} // namespace mlir::triton::spirv
