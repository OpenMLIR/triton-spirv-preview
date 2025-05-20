#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "spirv/include/Conversion/TritonToLinalg/Passes.h"

namespace py = pybind11;

void init_triton_spirv_passes_lair(py::module &&m) {
  m.def("triton_to_linalg", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::spirv::createTritonToLinalgPass());
  });

}

void init_triton_spirv(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_spirv_passes_lair(passes.def_submodule("lair"));
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {});
}
