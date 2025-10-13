#include <torch/extension.h>
#include <cuda_runtime.h>

__global__
void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index] * input[index];
    }
}

torch::Tensor square_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads_per_block = 1024;
    const int blocks_per_grid = (input.numel() + threads_per_block - 1) / threads_per_block;
    
    square_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    return output;
}