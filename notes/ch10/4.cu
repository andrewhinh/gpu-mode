#include "../common.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

__global__ void maxReductionKernel(int *input, int *output, int n) {
  __shared__ int sdata[BLOCK_DIM];

  // Each thread loads multiple elements and finds their max
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * COARSE_FACTOR * 2) + tid;

  int max_val = INT_MIN;
  for (int offset = 0; offset < COARSE_FACTOR * 2; offset++) {
    int idx = i + offset * blockDim.x;
    if (idx < n) {
      max_val = max(max_val, input[idx]);
    }
  }

  sdata[tid] = max_val;
  __syncthreads();

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  // Update the global max
  if (tid == 0) {
    atomicMax(output, sdata[0]);
  }
}

// CPU implementation of max reduction
int maxReduction_cpu(int *input, int n) {
  int max_val = INT_MIN;
  for (int i = 0; i < n; i++) {
    max_val = std::max(max_val, input[i]);
  }
  return max_val;
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
    A[i] = rand() % 10000;
  }
  *result_gpu = INT_MIN;

  // CPU timing with CUDA events
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);

  cudaEventRecord(start_cpu);
  result_cpu = maxReduction_cpu(A, size);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);

  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);
  printf("CPU max result: %d\n", result_cpu);

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

  // Initialize result to minimum integer
  int init_val = INT_MIN;
  cudaMemcpy(result_d, &init_val, sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  int blockSize = BLOCK_DIM;
  int numBlocks = (size + blockSize * COARSE_FACTOR * 2 - 1) /
                  (blockSize * COARSE_FACTOR * 2);

  cudaEventRecord(start_kernel);
  maxReductionKernel<<<numBlocks, blockSize>>>(A_d, result_d, size);
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
  printf("GPU max result: %d\n", *result_gpu);

  // Verify results
  int error = *result_gpu - result_cpu;

  printf("Verification results:\n");
  printf("  Error: %d\n", error);

  // Determine if verification passed
  bool verification_passed = (error == 0);
  printf("%s\n",
         verification_passed ? "Verification PASSED!" : "Verification FAILED!");

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