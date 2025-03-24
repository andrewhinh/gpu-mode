#include "../common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define TILE_WIDTH 16

__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P, int r,
                                            int width, int height, int depth) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outZ = blockIdx.z;

  if (outCol >= width || outRow >= height || outZ >= depth)
    return;

  float Pvalue = 0.0f;

  for (int fZ = 0; fZ < 2 * r + 1; fZ++) {
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
        int inZ = outZ - r + fZ;
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;

        if (inZ >= 0 && inZ < depth && inRow >= 0 && inRow < height &&
            inCol >= 0 && inCol < width) {

          int filterIdx =
              fZ * (2 * r + 1) * (2 * r + 1) + fRow * (2 * r + 1) + fCol;
          int inputIdx = inZ * height * width + inRow * width + inCol;

          Pvalue += F[filterIdx] * N[inputIdx];
        }
      }
    }
  }

  int outputIdx = outZ * height * width + outRow * width + outCol;
  P[outputIdx] = Pvalue;
}

// CPU implementation of 3D convolution
void convolution_3D_cpu(float *N, float *F, float *P, int r, int width, int height, int depth) {
  for (int outZ = 0; outZ < depth; outZ++) {
    for (int outRow = 0; outRow < height; outRow++) {
      for (int outCol = 0; outCol < width; outCol++) {
        float Pvalue = 0.0f;
        
        for (int fZ = 0; fZ < 2 * r + 1; fZ++) {
          for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
              int inZ = outZ - r + fZ;
              int inRow = outRow - r + fRow;
              int inCol = outCol - r + fCol;
              
              if (inZ >= 0 && inZ < depth && inRow >= 0 && inRow < height &&
                  inCol >= 0 && inCol < width) {
                
                int filterIdx = fZ * (2 * r + 1) * (2 * r + 1) + fRow * (2 * r + 1) + fCol;
                int inputIdx = inZ * height * width + inRow * width + inCol;
                
                Pvalue += F[filterIdx] * N[inputIdx];
              }
            }
          }
        }
        
        int outputIdx = outZ * height * width + outRow * width + outCol;
        P[outputIdx] = Pvalue;
      }
    }
  }
}

int main() {
  unsigned int channels = 3;
  unsigned int filter_size = 3;
  unsigned int r = filter_size / 2;
  unsigned int dim = 64;

  size_t volume_size = channels * dim * dim * sizeof(float);
  float *N = static_cast<float *>(malloc(volume_size));
  float *P = static_cast<float *>(malloc(volume_size));
  float *P_cpu = static_cast<float *>(malloc(volume_size));

  size_t filter_cube_size =
      (2 * r + 1) * (2 * r + 1) * (2 * r + 1) * sizeof(float);
  float *F = static_cast<float *>(malloc(filter_cube_size));

  // Initialize input and filter with some values
  for (unsigned int i = 0; i < channels * dim * dim; i++) {
    N[i] = 1.0f;
  }

  for (unsigned int i = 0; i < (2 * r + 1) * (2 * r + 1) * (2 * r + 1); i++) {
    F[i] = 1.0f / ((2 * r + 1) * (2 * r + 1) * (2 * r + 1));
  }

  // CPU timing with CUDA events
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  
  cudaEventRecord(start_cpu);
  convolution_3D_cpu(N, F, P_cpu, r, dim, dim, channels);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);
  
  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);

  // GPU timing with CUDA events
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  
  cudaEventRecord(start_gpu);
  
  float *N_d = nullptr, *F_d = nullptr, *P_d = nullptr;

  cudaMalloc(reinterpret_cast<void **>(&N_d), volume_size);
  cudaMalloc(reinterpret_cast<void **>(&F_d), filter_cube_size);
  cudaMalloc(reinterpret_cast<void **>(&P_d), volume_size);

  cudaMemcpy(N_d, N, volume_size, cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, filter_cube_size, cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize((dim + blockSize.x - 1) / blockSize.x,
                (dim + blockSize.y - 1) / blockSize.y, channels);

  cudaEventRecord(start_kernel);
  convolution_3D_basic_kernel<<<gridSize, blockSize>>>(N_d, F_d, P_d, r, dim,
                                                       dim, channels);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy the result back to host
  cudaMemcpy(P, P_d, volume_size, cudaMemcpyDeviceToHost);

  cudaFree(N_d);
  cudaFree(F_d);
  cudaFree(P_d);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);
  
  // Verify results - calculate max error and mean error
  float max_error = 0.0f;
  float mean_error = 0.0f;
  int error_count = 0;
  
  for (unsigned int i = 0; i < channels * dim * dim; i++) {
    float error = fabs(P[i] - P_cpu[i]);
    if (error > max_error) {
      max_error = error;
    }
    if (error > 1e-3) { // Count significant errors
      error_count++;
    }
    mean_error += error;
  }
  mean_error /= (channels * dim * dim);
  
  // Report verification results
  printf("Verification results:\n");
  printf("  Max error: %e\n", max_error);
  printf("  Mean error: %e\n", mean_error);
  printf("  Error count (>1e-3): %d out of %d elements\n", error_count, channels * dim * dim);
  
  // Determine if verification passed using a reasonable threshold
  bool verification_passed = (max_error < 1e-2) && (error_count < channels * dim * dim / 1000);
  printf("%s\n", verification_passed ? "Verification PASSED!" : "Verification FAILED!");
  
  // Calculate speedup
  printf("Speedup (CPU vs GPU kernel): %.2fx\n", cpu_time / kernel_time);
  printf("Speedup (CPU vs GPU total): %.2fx\n", cpu_time / gpu_time);
  
  // Cleanup events
  cudaEventDestroy(start_cpu);
  cudaEventDestroy(stop_cpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);

  // Free memory
  free(N);
  free(F);
  free(P);
  free(P_cpu);

  return 0;
}