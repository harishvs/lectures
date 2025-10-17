// a. Write a kernel that has each thread produce one output matrix row. 
// Fill in the execution configuration parameters for the design.
// Kernel: Each thread computes one row of output_matrix
// output_matrix = input_matrix1 × input_matrix2, where input_matrix1 is total_rows×K, input_matrix2 is K×total_cols, output_matrix is total_rows×total_cols
__global__ void matMulRow(float *input_matrix1, float *input_matrix2, float *output_matrix, int total_rows, int K, int total_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < total_rows) {
        // Compute entire row
        for (int col = 0; col < total_cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += input_matrix1[row * K + k] * input_matrix2[k * total_cols + col];
            }
            output_matrix[row * total_cols + col] = sum;
        }
    }
}

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // Matrix dimensions
    int total_rows = 512;
    int K = 256;
    int total_cols = 512;
    
    size_t size1 = total_rows * K * sizeof(float);
    size_t size2 = K * total_cols * sizeof(float);
    size_t size_out = total_rows * total_cols * sizeof(float);
    
    // Allocate host memory
    float *h_input1 = (float*)malloc(size1);
    float *h_input2 = (float*)malloc(size2);
    float *h_output = (float*)malloc(size_out);
    
    // Initialize input matrices
    for (int i = 0; i < total_rows * K; i++) h_input1[i] = 1.0f;
    for (int i = 0; i < K * total_cols; i++) h_input2[i] = 1.0f;
    
    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size1);
    cudaMalloc(&d_input2, size2);
    cudaMalloc(&d_output, size_out);
    
    // Copy to device
    cudaMemcpy(d_input1, h_input1, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, size2, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (total_rows + threadsPerBlock - 1) / threadsPerBlock;
    matMulRow<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, total_rows, K, total_cols);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_output, d_output, size_out, cudaMemcpyDeviceToHost);
    
    // Verify result (first element should be K * 1.0 = 256.0)
    printf("Result[0][0] = %.2f (expected %.2f)\n", h_output[0], (float)K);
    
    // Cleanup
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    free(h_input1);
    free(h_input2);
    free(h_output);
    
    return 0;
}
