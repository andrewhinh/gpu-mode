#include "../../common.h"
#include <cuda_runtime.h>

__global__ void sumReductionKernel(float *input, float *output) {
  unsigned int i = threadIdx.x;

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (i > stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (i == 0) {
    output[0] = input[0];
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

  sumReductionKernel<<<numBlocks, numThreads>>>(A_d, B_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(B, B_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);

  free(A);
  free(B);

  return 0;
}