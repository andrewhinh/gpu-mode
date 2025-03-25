#include "../common.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_DIM 32

// This kernel performs a block-level exclusive scan
__global__ void exclusive_scan(float *X, float *Y, int n) {
  __shared__ float temp[BLOCK_DIM];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  temp[tx] = (i < n) ? X[i] : 0;
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = (tx + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
      temp[index] += temp[index - stride];
    }
    __syncthreads();
  }

  float value = (tx > 0) ? temp[tx - 1] : 0;
  __syncthreads();
  temp[tx] = value;
  __syncthreads();

  if (i < n) {
    Y[i] = temp[tx];
  }
}

// CPU implementation of block-level exclusive scan
void exclusive_scan_cpu(float *input, float *output, int n, int blockDim) {
  // Process each block independently
  int numBlocks = (n + blockDim - 1) / blockDim;

  for (int b = 0; b < numBlocks; b++) {
    int blockStart = b * blockDim;
    int blockEnd = std::min(blockStart + blockDim, n);

    // Step 1: Copy input values to temp array
    float temp[BLOCK_DIM];
    for (int i = 0; i < blockDim; i++) {
      int globalIdx = blockStart + i;
      temp[i] = (globalIdx < n) ? input[globalIdx] : 0;
    }

    // Step 2: Perform up-sweep (reduction) phase
    for (int stride = 1; stride < blockDim; stride *= 2) {
      for (int j = (stride * 2 - 1); j < blockDim; j += stride * 2) {
        if (j < blockDim) {
          temp[j] += temp[j - stride];
        }
      }
    }

    // Step 3: Set last element to 0 (exclusive scan) and perform down-sweep
    // Store values in output directly
    for (int i = 0; i < blockDim; i++) {
      int globalIdx = blockStart + i;
      if (globalIdx < n) {
        if (i == 0) {
          output[globalIdx] = 0;
        } else {
          output[globalIdx] = temp[i - 1];
        }
      }
    }
  }
}

int main() {
  unsigned int N = 1024;
  float *h_input, *h_output_cpu, *h_output_gpu;

  // Allocate and initialize host memory
  h_input = (float *)malloc(N * sizeof(float));
  h_output_cpu = (float *)malloc(N * sizeof(float));
  h_output_gpu = (float *)malloc(N * sizeof(float));

  // Initialize input data with small random values
  for (unsigned int i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(rand() % 10);
  }

  // CPU timing
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);

  cudaEventRecord(start_cpu);
  exclusive_scan_cpu(h_input, h_output_cpu, N, BLOCK_DIM);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);

  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);

  // GPU timing
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);

  cudaEventRecord(start_gpu);

  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(float));
  cudaMalloc((void **)&d_output, N * sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = BLOCK_DIM;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaEventRecord(start_kernel);
  exclusive_scan<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);

  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy result back to host
  cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);

  // Verify results
  float max_error = 0.0f;
  float total_error = 0.0f;
  int significant_errors = 0;

  for (unsigned int i = 0; i < N; i++) {
    float error = fabs(h_output_gpu[i] - h_output_cpu[i]);
    max_error = std::max(max_error, error);
    total_error += error;

    if (error > 1e-5) {
      significant_errors++;

      // Print some of the errors for debugging (limit to first 5)
      if (significant_errors <= 5) {
        printf("Error at index %u: GPU = %f, CPU = %f, diff = %f\n", i,
               h_output_gpu[i], h_output_cpu[i], error);
      }
    }
  }

  printf("Verification results:\n");
  printf("  Max error: %e\n", max_error);
  printf("  Mean error: %e\n", total_error / N);
  printf("  Significant errors: %d\n", significant_errors);

  // Determine if verification passed
  bool verification_passed = (significant_errors == 0);
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
  free(h_input);
  free(h_output_cpu);
  free(h_output_gpu);

  return 0;
}