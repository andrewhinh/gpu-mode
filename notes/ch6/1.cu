#include "../common.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matmul_tile_coarse(float *A, float *B, float *C,
                                   unsigned int width) {
  __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

  // Initialize sums to zero
  float sum[COARSE_FACTOR] = {0.0f};

  // Boundary check
  if (row < width) {
    // Loop over tiles
    for (unsigned int tile = 0; tile < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
      // Boundary check for A
      if (tile * TILE_WIDTH + threadIdx.x < width) {
        A_s[threadIdx.y][threadIdx.x] = A[row * width + tile * TILE_WIDTH + threadIdx.x];
      } else {
        A_s[threadIdx.y][threadIdx.x] = 0.0f;
      }

      // Loop over elements in the coarse factor direction
      for (unsigned int coarse = 0; coarse < COARSE_FACTOR; ++coarse) {
        unsigned int col = colStart + coarse * blockDim.x;
        
        // Boundary check for B
        if (col < width && tile * TILE_WIDTH + threadIdx.y < width) {
          B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * width + col];
        } else {
          B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute matrix multiplication for this tile
        if (row < width && col < width) {
          for (unsigned int k = 0; k < TILE_WIDTH; ++k) {
            if (tile * TILE_WIDTH + k < width) {
              sum[coarse] += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
            }
          }
        }
        
        __syncthreads();
      }
    }

    // Write results to global memory
    for (unsigned int coarse = 0; coarse < COARSE_FACTOR; ++coarse) {
      unsigned int col = colStart + coarse * blockDim.x;
      if (row < width && col < width) {
        C[row * width + col] = sum[coarse];
      }
    }
  }
}

// CPU implementation of matrix multiplication
__host__ void matmul_cpu(float *A, float *B, float *C, unsigned int width) {
  for (unsigned int row = 0; row < width; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      float sum = 0.0f;
      for (unsigned int k = 0; k < width; ++k) {
        sum += A[row * width + k] * B[k * width + col];
      }
      C[row * width + col] = sum;
    }
  }
}

int main() {
  unsigned int width = 1024;
  float *A, *B, *C;
  A = static_cast<float *>(malloc(width * width * sizeof(float)));
  B = static_cast<float *>(malloc(width * width * sizeof(float)));
  C = static_cast<float *>(malloc(width * width * sizeof(float)));

  // Initialize matrices with small random values to prevent overflow
  srand(42); // Use fixed seed for reproducibility
  for (unsigned int i = 0; i < width * width; ++i) {
    A[i] = static_cast<float>(rand() % 10) / 10.0f;
    B[i] = static_cast<float>(rand() % 10) / 10.0f;
  }

  // CPU timing with CUDA events
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  
  cudaEventRecord(start_cpu);
  matmul_cpu(A, B, C, width);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);
  
  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);
  
  // Save CPU results for verification
  float *C_cpu = static_cast<float *>(malloc(width * width * sizeof(float)));
  memcpy(C_cpu, C, width * width * sizeof(float));
  
  // Reset C for GPU computation
  memset(C, 0, width * width * sizeof(float));
  
  // GPU timing with CUDA events
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  
  cudaEventRecord(start_gpu);
  
  float *A_d, *B_d, *C_d;
  size_t size = width * width * sizeof(float);

  cudaMalloc(reinterpret_cast<void **>(&A_d), size);
  cudaMalloc(reinterpret_cast<void **>(&B_d), size);
  cudaMalloc(reinterpret_cast<void **>(&C_d), size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
  cudaMemset(C_d, 0, size); // Initialize C_d to zeros

  dim3 numThreads(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks((width + numThreads.x - 1) / numThreads.x / COARSE_FACTOR,
                 (width + numThreads.y - 1) / numThreads.y);

  cudaEventRecord(start_kernel);
  matmul_tile_coarse<<<numBlocks, numThreads>>>(A_d, B_d, C_d, width);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);
  
  // Verify results - calculate max error and mean error
  float max_error = 0.0f;
  float mean_error = 0.0f;
  int error_count = 0;
  
  for (unsigned int i = 0; i < width * width; ++i) {
    float error = fabs(C[i] - C_cpu[i]);
    if (error > max_error) {
      max_error = error;
    }
    if (error > 1e-3) { // Count significant errors
      error_count++;
    }
    mean_error += error;
  }
  mean_error /= (width * width);
  
  // Report verification results
  printf("Verification results:\n");
  printf("  Max error: %e\n", max_error);
  printf("  Mean error: %e\n", mean_error);
  printf("  Error count (>1e-3): %d out of %d elements\n", error_count, width * width);
  
  // Determine if verification passed using a reasonable threshold
  bool verification_passed = (max_error < 1e-2) && (error_count < width * width / 1000);
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

  free(A);
  free(B);
  free(C);
  free(C_cpu);

  return 0;
}