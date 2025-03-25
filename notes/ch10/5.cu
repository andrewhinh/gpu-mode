#include "../common.h"
#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

__global__ void sumReductionKernel(int *input, int *output, unsigned int N) {
  __shared__ int input_s[BLOCK_DIM];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  int sum = 0;
  if (i < N) {
    sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
      unsigned int index = i + tile * BLOCK_DIM;
      if (index < N) {
        sum += input[index];
      }
    }
  }

  input_s[t] = sum;
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

// CPU implementation of sum reduction
int sumReduction_cpu(int *input, unsigned int n) {
  int sum = 0;
  for (unsigned int i = 0; i < n; i++) {
    sum += input[i];
  }
  return sum;
}

int main() {
  unsigned int size = 1024 * 1024;
  int *A;
  int *result_gpu;
  int result_cpu;
  
  // Allocate and initialize host memory
  A = static_cast<int *>(malloc(size * sizeof(int)));
  result_gpu = static_cast<int *>(malloc(sizeof(int)));
  
  // Initialize input data with random values
  for (unsigned int i = 0; i < size; i++) {
    A[i] = rand() % 100;  // Small values to avoid integer overflow
  }
  *result_gpu = 0;
  
  // CPU timing with CUDA events
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  
  cudaEventRecord(start_cpu);
  result_cpu = sumReduction_cpu(A, size);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);
  
  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);
  printf("CPU sum result: %d\n", result_cpu);

  // GPU timing with CUDA events
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  
  cudaEventRecord(start_gpu);
  
  // Allocate device memory
  int *A_d, *result_d;
  cudaMalloc(reinterpret_cast<void **>(&A_d), size * sizeof(int));
  cudaMalloc(reinterpret_cast<void **>(&result_d), sizeof(int));
  
  // Copy input to device
  cudaMemcpy(A_d, A, size * sizeof(int), cudaMemcpyHostToDevice);
  
  // Initialize result to 0
  int init_val = 0;
  cudaMemcpy(result_d, &init_val, sizeof(int), cudaMemcpyHostToDevice);
  
  // Launch kernel
  int blockSize = BLOCK_DIM;
  int numBlocks = (size + blockSize * COARSE_FACTOR * 2 - 1) / (blockSize * COARSE_FACTOR * 2);
  
  cudaEventRecord(start_kernel);
  sumReductionKernel<<<numBlocks, blockSize>>>(A_d, result_d, size);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);
  
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  
  // Copy result back to host
  cudaMemcpy(result_gpu, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(A_d);
  cudaFree(result_d);
  
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);
  printf("GPU sum result: %d\n", *result_gpu);
  
  // Verify results
  int absolute_error = abs(*result_gpu - result_cpu);
  float relative_error = (float)absolute_error / ((result_cpu != 0) ? (float)abs(result_cpu) : 1.0f);
  
  printf("Verification results:\n");
  printf("  Absolute error: %d\n", absolute_error);
  printf("  Relative error: %e\n", relative_error);
  
  // Determine if verification passed using a reasonable threshold
  bool verification_passed = relative_error < 1e-3;
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
  free(A);
  free(result_gpu);
  
  return 0;
}