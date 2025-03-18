#include "../../common.h"
#include <cuda_runtime.h>

#define BLOCK_DIM 32

__global__ void koggeStoneScan(float *input, float *output, unsigned int N) {
  __shared__ float buffer[2][BLOCK_DIM];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    buffer[0][threadIdx.x] = input[i];
  } else {
    buffer[0][threadIdx.x] = 0.0f;
  }

  int inBuf = 0;
  int outBuf = 1;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    buffer[outBuf][threadIdx.x] = buffer[inBuf][threadIdx.x];
    if (threadIdx.x >= stride)
      buffer[outBuf][threadIdx.x] += buffer[inBuf][threadIdx.x - stride];

    __syncthreads();

    int temp = inBuf;
    inBuf = outBuf;
    outBuf = temp;
  }

  if (i < N) {
    output[i] = buffer[inBuf][threadIdx.x];
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

  koggeStoneScan<<<numBlocks, numThreads>>>(A_d, B_d, width);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(B, B_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);

  free(A);
  free(B);

  return 0;
}