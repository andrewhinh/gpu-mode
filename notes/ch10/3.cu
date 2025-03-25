#include "../common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void sumReductionKernel(float *input, float *output, int n) {
  __shared__ float sdata[1024];
  
  // Load data into shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  sdata[tid] = (i < n) ? input[i] : 0;
  __syncthreads();
  
  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result for this block to global memory
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

// CPU implementation of sum reduction
float sumReduction_cpu(float *input, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += input[i];
  }
  return sum;
}

int main() {
  unsigned int size = 1024 * 1024;
  float *A;
  float *result_gpu;
  float result_cpu;
  
  // Allocate and initialize host memory
  A = static_cast<float *>(malloc(size * sizeof(float)));
  result_gpu = static_cast<float *>(malloc(sizeof(float)));
  
  // Initialize input data with random values between 0 and 1
  for (unsigned int i = 0; i < size; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  *result_gpu = 0.0f;
  
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
  printf("CPU sum result: %f\n", result_cpu);

  // GPU timing with CUDA events
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  
  cudaEventRecord(start_gpu);
  
  // Allocate device memory
  float *A_d, *result_d;
  cudaMalloc(reinterpret_cast<void **>(&A_d), size * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&result_d), sizeof(float));
  
  // Copy input to device
  cudaMemcpy(A_d, A, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(result_d, 0, sizeof(float));
  
  // Launch kernel
  int blockSize = 1024;
  int numBlocks = (size + blockSize - 1) / blockSize;
  
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
  cudaMemcpy(result_gpu, result_d, sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(A_d);
  cudaFree(result_d);
  
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);
  printf("GPU sum result: %f\n", *result_gpu);
  
  // Verify results
  float absolute_error = fabs(*result_gpu - result_cpu);
  float relative_error = absolute_error / (fabs(result_cpu) > 1e-6 ? fabs(result_cpu) : 1.0f);
  
  printf("Verification results:\n");
  printf("  Absolute error: %e\n", absolute_error);
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