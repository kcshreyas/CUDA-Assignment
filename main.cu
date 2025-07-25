#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void reduce_sum(float* input, float* output) {
    // Allocate shared memory dynamically
    extern __shared__ float sdata[];

    // Thread
    unsigned int tid = threadIdx.x;

    // Load input into shared memory
    sdata[tid] = input[tid];
    __syncthreads();

    // Do reduction in shared memory
    // Halve stride each iteration
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = sdata[0];
    }
}

int main() {
    float *h_input, *h_output; // host variables
    float *d_input, *d_output; // device variables

    size_t size = N * sizeof(float);
    h_input = (float*) malloc(size);
    h_output = (float*) malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    reduce_sum<<<1, N, N * sizeof(float)>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum = %f\n", h_output[0]);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
