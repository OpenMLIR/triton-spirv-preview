add_triton_library(AffineToLLVMSPV
  AffineToLLVMSPV.cpp

  DEPENDS
  AffineToLLVMSPVConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
)
