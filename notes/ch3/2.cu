#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "../../common.h"

__host__ __device__
float matmul_vec(float* M, float* V, int row, int col, int width) {
    float pVal = 0.0f;
    for (int k = 0; k < width; ++k) {
        pVal += M[row * width + k] * V[k];
    }
    return pVal;
}

__global__
void gpu(float* M, float* V, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width) {
        P[row * width + col] = matmul_vec(M, V, row, col, width);
    }
}

__host__
void cpu(float* M, float* V, float* P, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            P[row * width + col] = matmul_vec(M, V, row, col, width);
        }
    }
}


int main() {
    int width = 1024;
    float *M, *V, *P;
    M = static_cast<float*>(malloc(width * width * sizeof(float)));
    V = static_cast<float*>(malloc(width * sizeof(float)));
    P = static_cast<float*>(malloc(width * width * sizeof(float)));

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu(M, V, P, width);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    printf("CPU execution time: %f seconds\n", duration_cpu.count());

    auto start_gpu = std::chrono::high_resolution_clock::now();
    float *M_d, *V_d, *P_d;
    size_t size_MP = width * width * sizeof(float);
    size_t size_V = width * sizeof(float);

    cudaMalloc(reinterpret_cast<void**>(&M_d), size_MP);
    cudaMalloc(reinterpret_cast<void**>(&V_d), size_V);
    cudaMalloc(reinterpret_cast<void**>(&P_d), size_MP);

    cudaMemcpy(M_d, M, size_MP, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, size_V, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 512;
    unsigned int numBlocks = (width + numThreads - 1) / numThreads;

    auto start_gpu_kernel = std::chrono::high_resolution_clock::now();
    gpu<<<numBlocks, numThreads>>>(M_d, V_d, P_d, width);
    auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu_kernel = end_gpu_kernel - start_gpu_kernel;
    printf("GPU kernel execution time: %f seconds\n", duration_gpu_kernel.count());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(P, P_d, size_MP, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(V_d);
    cudaFree(P_d);

    free(M);
    free(V);
    free(P);

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
    printf("GPU execution time: %f seconds\n", duration_gpu.count());

    return 0;
}