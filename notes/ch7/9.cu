#include <cuda_runtime.h>
#include "../../common.h"

#define TILE_WIDTH 16
#define FILTER_SIZE 3
#define RADIUS (FILTER_SIZE / 2)

__constant__ float F_d[2*RADIUS+1][2*RADIUS+1][2*RADIUS+1];

__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P,
    int width, int height, int depth) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outZ = blockIdx.z;
  
  if (outCol >= width || outRow >= height || outZ >= depth)
    return;
  
  float Pvalue = 0.0f;
  
  for (int fZ = 0; fZ < FILTER_SIZE; fZ++) {
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
      for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
        int inZ = outZ - RADIUS + fZ;
        int inRow = outRow - RADIUS + fRow;
        int inCol = outCol - RADIUS + fCol;
        
        if (inZ >= 0 && inZ < depth &&
            inRow >= 0 && inRow < height && 
            inCol >= 0 && inCol < width) {
          
          int inputIdx = inZ*height*width + inRow*width + inCol;
          
          Pvalue += F_d[fZ][fRow][fCol] * N[inputIdx];
        }
      }
    }
  }
  
  int outputIdx = outZ*height*width + outRow*width + outCol;
  P[outputIdx] = Pvalue;
}

int main() {
  unsigned int channels = 3;
  unsigned int dim = 64;
  
  size_t volume_size = channels * dim * dim * sizeof(float);
  float *N = static_cast<float*>(malloc(volume_size));
  float *P = static_cast<float*>(malloc(volume_size));

  float *F = static_cast<float*>(malloc(FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
  
  for (unsigned int i = 0; i < channels * dim * dim; i++) {
    N[i] = 1.0f;
  }
  
  for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; i++) {
    F[i] = 1.0f / (FILTER_SIZE * FILTER_SIZE * FILTER_SIZE);
  }

  float *N_d = nullptr, *F_d = nullptr, *P_d = nullptr;

  cudaMemcpyToSymbol(F_d, F, FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float));
  
  cudaMalloc(reinterpret_cast<void**>(&N_d), volume_size);
  cudaMalloc(reinterpret_cast<void**>(&P_d), volume_size);

  cudaMemcpy(N_d, N, volume_size, cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize(
      (dim + blockSize.x - 1) / blockSize.x,
      (dim + blockSize.y - 1) / blockSize.y,
      channels
  );

  convolution_3D_basic_kernel<<<gridSize, blockSize>>>(N_d, F_d, P_d, dim, dim, channels);

  cudaFree(N_d);
  cudaFree(F_d);
  cudaFree(P_d);

  free(N);
  free(F);
  free(P);

  return 0;
}