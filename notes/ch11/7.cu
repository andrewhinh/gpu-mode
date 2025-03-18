#include "../../common.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 32

__global__ void exclusive_scan(float *X, float *Y, int n) {
  __shared__ float temp[BLOCK_DIM];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  temp[tx] = (i < n) ? X[i] : 0;
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = (tx + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      temp[index] += temp[index - stride];
    }
    __syncthreads();
  }

  float value = (tx > 0) ? temp[tx - 1] : 0;
  __syncthreads();
  temp[tx] = value;
  __syncthreads();

  if (i < n) {
    Y[i] = temp[tx];
  }
}

int main() {
  unsigned int width = 1024;
  float *A, *B;
  A = static_cast<float *>(malloc(width * width * sizeof(float)));
  B = static_cast<float *>(malloc(width * width * sizeof(float)));

  float *A_d, *B_d;
  size_t size = width * width * sizeof(float);

  cudaMalloc(reinterpret_cast<void **>(&A_d), size);
  cudaMalloc(reinterpret_cast<void **>(&B_d), size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  dim3 numThreads(32, 32);
  dim3 numBlocks((width + numThreads.x - 1) / numThreads.x,
                 (width + numThreads.y - 1) / numThreads.y);

  exclusive_scan<<<numBlocks, numThreads>>>(A_d, B_d, width);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(B, B_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);

  free(A);
  free(B);

  return 0;
}