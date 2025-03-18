#include "../../common.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define SECTION_SIZE BLOCK_SIZE

__global__ void koggeStoneScan1(float *input, float *S, float *output,
                                unsigned int N) {
  __shared__ float XY[SECTION_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    XY[threadIdx.x] = input[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    float temp = 0.0f;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x - stride];
    }
    __syncthreads();

    if (threadIdx.x >= stride) {
      XY[threadIdx.x] += temp;
    }
    __syncthreads();
  }

  if (i < N) {
    output[i] = XY[threadIdx.x];
  }

  if (threadIdx.x == blockDim.x - 1) {
    S[blockIdx.x] = XY[blockDim.x - 1];
  }
}

__global__ void koggeStoneScan2(float *S, unsigned int numBlocks) {
  __shared__ float XY[SECTION_SIZE];

  if (threadIdx.x < numBlocks) {
    XY[threadIdx.x] = S[threadIdx.x];
  } else {
    XY[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    float temp = 0.0f;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x - stride];
    }
    __syncthreads();

    if (threadIdx.x >= stride) {
      XY[threadIdx.x] += temp;
    }
    __syncthreads();
  }

  if (threadIdx.x < numBlocks) {
    S[threadIdx.x] = XY[threadIdx.x];
  }
}

__global__ void koggeStoneScan3(float *S, float *Y, unsigned int N) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N && blockIdx.x > 0) {
    Y[i] += S[blockIdx.x - 1];
  }
}

int main() {
  unsigned int N = 1024 * 1024;
  unsigned int bytes = N * sizeof(float);

  float *h_input = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);

  for (unsigned int i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(rand() % 10);
  }

  float *d_input, *d_output, *d_blockSums;
  cudaMalloc((void **)&d_input, bytes);
  cudaMalloc((void **)&d_output, bytes);

  unsigned int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaMalloc((void **)&d_blockSums, numBlocks * sizeof(float));

  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

  koggeStoneScan1<<<numBlocks, BLOCK_SIZE>>>(d_input, d_blockSums, d_output, N);
  koggeStoneScan2<<<1, BLOCK_SIZE>>>(d_blockSums, numBlocks);
  koggeStoneScan3<<<numBlocks, BLOCK_SIZE>>>(d_blockSums, d_output, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_blockSums);

  return 0;
}