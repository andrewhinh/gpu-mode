#include "../common.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_DIM 32

// In the current implementation, this does a block-level scan only
// Each block scans its own segment independently
__global__ void koggeStoneScan(float *input, float *output, unsigned int N) {
  __shared__ float buffer[2][BLOCK_DIM];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    buffer[0][threadIdx.x] = input[i];
  } else {
    buffer[0][threadIdx.x] = 0.0f;
  }

  int inBuf = 0;
  int outBuf = 1;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    buffer[outBuf][threadIdx.x] = buffer[inBuf][threadIdx.x];
    if (threadIdx.x >= stride)
      buffer[outBuf][threadIdx.x] += buffer[inBuf][threadIdx.x - stride];

    __syncthreads();

    int temp = inBuf;
    inBuf = outBuf;
    outBuf = temp;
  }

  if (i < N) {
    output[i] = buffer[inBuf][threadIdx.x];
  }
}

// CPU implementation of the block-level scan (to match GPU behavior)
void koggeStoneScan_cpu(float *input, float *output, unsigned int N,
                        unsigned int blockDim) {
  // In the current implementation, the scan is done independently for each
  // block
  unsigned int numBlocks = (N + blockDim - 1) / blockDim;

  for (unsigned int b = 0; b < numBlocks; b++) {
    unsigned int blockStart = b * blockDim;
    unsigned int blockEnd = std::min(blockStart + blockDim, N);
    unsigned int blockSize = blockEnd - blockStart;

    // Create a temporary buffer to exactly match the GPU implementation
    float buffer[2][BLOCK_DIM];
    int inBuf = 0;
    int outBuf = 1;

    // Initialize the buffer
    for (unsigned int i = 0; i < blockDim; i++) {
      unsigned int globalIdx = blockStart + i;
      buffer[inBuf][i] = (globalIdx < N) ? input[globalIdx] : 0.0f;
    }

    // Perform scan operation within this block - exactly as the GPU does
    for (unsigned int stride = 1; stride < blockDim; stride *= 2) {
      // Copy values from input buffer to output buffer
      for (unsigned int i = 0; i < blockDim; i++) {
        buffer[outBuf][i] = buffer[inBuf][i];
        if (i >= stride) {
          buffer[outBuf][i] += buffer[inBuf][i - stride];
        }
      }

      // Swap buffers
      int temp = inBuf;
      inBuf = outBuf;
      outBuf = temp;
    }

    // Copy results back to output
    for (unsigned int i = 0; i < blockSize; i++) {
      output[blockStart + i] = buffer[inBuf][i];
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

  // Initialize input data with small values to prevent overflow
  for (unsigned int i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(rand() % 10);
  }

  // CPU timing
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);

  cudaEventRecord(start_cpu);
  koggeStoneScan_cpu(h_input, h_output_cpu, N, BLOCK_DIM);
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
  koggeStoneScan<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
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