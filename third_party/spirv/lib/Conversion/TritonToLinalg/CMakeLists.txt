add_triton_library(TritonToLinalg
  TritonToLinalg.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  TritonSharedAnalysis
)
