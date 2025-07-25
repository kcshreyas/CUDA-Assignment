#include <stdio.h>
#include <cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 1024  // Must be power of 2

// CUDA kernel for parallel reduction
__global__ void reduce_sum(int *input, int *result) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    shared_data[tid] = input[idx];
    __syncthreads();

    // Perform tree-based reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

int main() {
    int h_input[N];
    int h_result = 0;

    // Initialize input array with values 1 to N
    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;
    }

    int *d_input, *d_result;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block of THREADS_PER_BLOCK threads
    reduce_sum<<<1, THREADS_PER_BLOCK>>>(d_input, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("CUDA Sum: %d\n", h_result);
    printf("Expected Sum: %d\n", (N * (N + 1)) / 2);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
