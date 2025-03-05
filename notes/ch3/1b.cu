#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "../../common.h"

__host__ __device__ float matmul_col(float* M, float* N, int row, int col, int width) {
    float pVal = 0.0;
    for (int k = 0; k < width; ++k) {
        pVal += M[row * width + k] * N[k * width + col];
    }
    return pVal;
}

__global__
void gpu(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width) {
        for (int row = 0; row < width; ++row) {
            P[row * width + col] = matmul_col(M, N, row, col, width);
        }
    }
}

__host__
void cpu(float* M, float* N, float* P, int width) {
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < width; ++row) {
            P[row * width + col] = matmul_col(M, N, row, col, width);
        }
    }
}


int main() {
    int width = 1024;
    float *M, *N, *P;
    M = static_cast<float*>(malloc(width * width * sizeof(float)));
    N = static_cast<float*>(malloc(width * width * sizeof(float)));
    P = static_cast<float*>(malloc(width * width * sizeof(float)));

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu(M, N, P, width);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    printf("CPU execution time: %f seconds\n", duration_cpu.count());

    auto start_gpu = std::chrono::high_resolution_clock::now();
    float *M_d, *N_d, *P_d;
    size_t size = width * width * sizeof(float);

    cudaMalloc(reinterpret_cast<void**>(&M_d), size);
    cudaMalloc(reinterpret_cast<void**>(&N_d), size);
    cudaMalloc(reinterpret_cast<void**>(&P_d), size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 512;
    unsigned int numBlocks = (width + numThreads - 1) / numThreads;

    auto start_gpu_kernel = std::chrono::high_resolution_clock::now();
    gpu<<<numBlocks, numThreads>>>(M_d, N_d, P_d, width);
    auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu_kernel = end_gpu_kernel - start_gpu_kernel;
    printf("GPU kernel execution time: %f seconds\n", duration_gpu_kernel.count());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    free(M);
    free(N);
    free(P);

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
    printf("GPU execution time: %f seconds\n", duration_gpu.count());

    return 0;
}