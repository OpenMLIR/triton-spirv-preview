include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonSPIRV ${CMAKE_CURRENT_SOURCE_DIR}/triton_spirv.cc)
  target_link_libraries(TritonSPIRV PUBLIC PRIVATE Python3::Module pybind11::headers)
endif()

option(BUILD_SPIRV_OPT "build spirv-opt to debug" OFF)
if(BUILD_SPIRV_OPT)
  add_subdirectory(tool)
endif()
