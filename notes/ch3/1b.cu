#include "../common.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ float matmul_col(float *M, float *N, int row, int col,
                                     int width) {
  float pVal = 0.0;
  for (int k = 0; k < width; ++k) {
    pVal += M[row * width + k] * N[k * width + col];
  }
  return pVal;
}

__global__ void gpu(float *M, float *N, float *P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < width) {
    for (int row = 0; row < width; ++row) {
      P[row * width + col] = matmul_col(M, N, row, col, width);
    }
  }
}

__host__ void cpu(float *M, float *N, float *P, int width) {
  for (int col = 0; col < width; ++col) {
    for (int row = 0; row < width; ++row) {
      P[row * width + col] = matmul_col(M, N, row, col, width);
    }
  }
}

int main() {
  int width = 1024;
  float *M, *N, *P;
  M = static_cast<float *>(malloc(width * width * sizeof(float)));
  N = static_cast<float *>(malloc(width * width * sizeof(float)));
  P = static_cast<float *>(malloc(width * width * sizeof(float)));

  // Initialize matrices with small random values to prevent overflow
  srand(42); // Use fixed seed for reproducibility
  for (int i = 0; i < width * width; ++i) {
    M[i] = static_cast<float>(rand() % 10) / 10.0f;
    N[i] = static_cast<float>(rand() % 10) / 10.0f;
  }

  // CPU timing with CUDA events
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  
  cudaEventRecord(start_cpu);
  cpu(M, N, P, width);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);
  
  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);
  
  // Save CPU results for verification
  float *P_cpu = static_cast<float *>(malloc(width * width * sizeof(float)));
  memcpy(P_cpu, P, width * width * sizeof(float));
  
  // Reset P for GPU computation
  memset(P, 0, width * width * sizeof(float));
  
  // GPU timing with CUDA events
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  
  cudaEventRecord(start_gpu);
  float *M_d, *N_d, *P_d;
  size_t size = width * width * sizeof(float);

  cudaMalloc(reinterpret_cast<void **>(&M_d), size);
  cudaMalloc(reinterpret_cast<void **>(&N_d), size);
  cudaMalloc(reinterpret_cast<void **>(&P_d), size);

  cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

  const unsigned int numThreads = 512;
  unsigned int numBlocks = (width + numThreads - 1) / numThreads;

  cudaEventRecord(start_kernel);
  gpu<<<numBlocks, numThreads>>>(M_d, N_d, P_d, width);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU execution time: %f milliseconds\n", gpu_time);
  
  // Verify results - calculate max error and mean error
  float max_error = 0.0f;
  float mean_error = 0.0f;
  int error_count = 0;
  
  for (int i = 0; i < width * width; ++i) {
    float error = fabs(P[i] - P_cpu[i]);
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

  // Free memory
  free(M);
  free(N);
  free(P);
  free(P_cpu);

  return 0;
}