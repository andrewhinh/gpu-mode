#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "../../common.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__
void matmul_tile_coarse(float* A, float* B, float* C, unsigned int width) {
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    float sum[COARSE_FACTOR];
    for (unsigned int i = 0; i < COARSE_FACTOR; ++i) {
        sum[i] = 0.0f;
    }

    for (unsigned int tile = 0; tile < width / TILE_WIDTH; ++tile) {
        A_s[threadIdx.y][threadIdx.x] = A[row * width + tile * TILE_WIDTH + threadIdx.x];

        for (unsigned int coarse = 0; coarse < COARSE_FACTOR; ++coarse) {
            unsigned int col = colStart + coarse * TILE_WIDTH;
            
            B_s[threadIdx.x][threadIdx.y] = B[(tile * TILE_WIDTH + threadIdx.y) * width + col];
            __syncthreads();

            for (unsigned int k = 0; k < TILE_WIDTH; ++k) {
                sum[coarse] += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
            }
        }
        __syncthreads();

        for (unsigned int coarse = 0; coarse < COARSE_FACTOR; ++coarse) {
            unsigned int col = colStart + coarse * TILE_WIDTH;
            C[row * width + col] = sum[coarse];
        }
    }
}


int main() {
    unsigned int width = 1024;
    float *A, *B, *C;
    A = static_cast<float*>(malloc(width * width * sizeof(float)));
    B = static_cast<float*>(malloc(width * width * sizeof(float)));
    C = static_cast<float*>(malloc(width * width * sizeof(float)));

    float *A_d, *B_d, *C_d;
    size_t size = width * width * sizeof(float);

    cudaMalloc(reinterpret_cast<void**>(&A_d), size);
    cudaMalloc(reinterpret_cast<void**>(&B_d), size);
    cudaMalloc(reinterpret_cast<void**>(&C_d), size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    dim3 numThreads(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((width + numThreads.x - 1) / numThreads.x / COARSE_FACTOR, (width + numThreads.y - 1) / numThreads.y);

    auto start_gpu_kernel = std::chrono::high_resolution_clock::now();
    matmul_tile_coarse<<<numBlocks, numThreads>>>(A_d, B_d, C_d, width);
    auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu_kernel = end_gpu_kernel - start_gpu_kernel;
    printf("GPU kernel execution time: %f seconds\n", duration_gpu_kernel.count());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);

    return 0;
}