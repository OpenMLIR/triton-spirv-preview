get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(spirv-opt spirv-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(spirv-opt)
target_link_libraries(spirv-opt PRIVATE
  TritonTransforms
  AffineToLLVMSPV
  LinalgToAffineLoops
  TritonToLinalg
  ${dialect_libs}
  ${conversion_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(spirv-opt)

add_llvm_executable(standalone-translate EmitHLSCpp.cpp)
llvm_update_compile_flags(standalone-translate)
target_link_libraries(standalone-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
)
mlir_check_link_libraries(standalone-translate)
