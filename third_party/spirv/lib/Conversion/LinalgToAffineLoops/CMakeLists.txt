add_triton_library(LinalgToAffineLoops
  LinalgToAffineLoops.cpp

  DEPENDS
  LinalgToAffineLoopsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
)
