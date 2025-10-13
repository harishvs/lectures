#include <torch/extension.h>

// Forward declaration
torch::Tensor square_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_cuda", &square_cuda, "Square function (CUDA)");
}