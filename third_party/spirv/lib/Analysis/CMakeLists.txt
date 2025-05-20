add_triton_library(TritonSharedAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
