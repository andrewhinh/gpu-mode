#include <cuda_runtime.h>
#include "../../common.h"

#define TILE_WIDTH 16

__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P,
    int r, int width, int height, int depth) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outZ = blockIdx.z;
  
  if (outCol >= width || outRow >= height || outZ >= depth)
    return;
  
  float Pvalue = 0.0f;
  
  for (int fZ = 0; fZ < 2*r+1; fZ++) {
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
      for (int fCol = 0; fCol < 2*r+1; fCol++) {
        int inZ = outZ - r + fZ;
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        
        if (inZ >= 0 && inZ < depth &&
            inRow >= 0 && inRow < height && 
            inCol >= 0 && inCol < width) {
          
          int filterIdx = fZ*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol;
          int inputIdx = inZ*height*width + inRow*width + inCol;
          
          Pvalue += F[filterIdx] * N[inputIdx];
        }
      }
    }
  }
  
  int outputIdx = outZ*height*width + outRow*width + outCol;
  P[outputIdx] = Pvalue;
}

int main() {
  unsigned int channels = 3;
  unsigned int filter_size = 3;
  unsigned int r = filter_size / 2;
  unsigned int dim = 64;
  
  size_t volume_size = channels * dim * dim * sizeof(float);
  float *N = static_cast<float*>(malloc(volume_size));
  float *P = static_cast<float*>(malloc(volume_size));

  size_t filter_cube_size = (2*r+1) * (2*r+1) * (2*r+1) * sizeof(float);
  float *F = static_cast<float*>(malloc(filter_cube_size));
  

  for (unsigned int i = 0; i < channels * dim * dim; i++) {
    N[i] = 1.0f;
  }
  
  for (unsigned int i = 0; i < (2*r+1) * (2*r+1) * (2*r+1); i++) {
    F[i] = 1.0f / ((2*r+1) * (2*r+1) * (2*r+1));
  }

  float *N_d = nullptr, *F_d = nullptr, *P_d = nullptr;
  
  cudaMalloc(reinterpret_cast<void**>(&N_d), volume_size);
  cudaMalloc(reinterpret_cast<void**>(&F_d), filter_cube_size);
  cudaMalloc(reinterpret_cast<void**>(&P_d), volume_size);

  cudaMemcpy(N_d, N, volume_size, cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, filter_cube_size, cudaMemcpyHostToDevice);


  dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize(
      (dim + blockSize.x - 1) / blockSize.x,
      (dim + blockSize.y - 1) / blockSize.y,
      channels
  );

  convolution_3D_basic_kernel<<<gridSize, blockSize>>>(N_d, F_d, P_d, r, dim, dim, channels);

  cudaFree(N_d);
  cudaFree(F_d);
  cudaFree(P_d);

  free(N);
  free(F);
  free(P);

  return 0;
}