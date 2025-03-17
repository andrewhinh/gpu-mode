#include "../../common.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

__global__ void sumReductionKernel(int *input, int *output) {
  __shared__ int input_s[BLOCK_DIM];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  int max_val = input[i];
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
    max_val = max(max_val, input[i + tile * BLOCK_DIM]);
  }

  input_s[t] = max_val;
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] = max(input_s[t], input_s[t + stride]);
    }
  }

  if (t == 0) {
    atomicMax(output, input_s[0]);
  }
}

int main() {
  unsigned int width = 1024;
  int *A, *B;
  A = static_cast<int *>(malloc(width * width * sizeof(int)));
  B = static_cast<int *>(malloc(width * width * sizeof(int)));

  int *A_d, *B_d;
  size_t size = width * width * sizeof(int);

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