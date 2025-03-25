#include "../common.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define SECTION_SIZE BLOCK_SIZE

// First kernel: Performs block-level scan and stores block sums
__global__ void koggeStoneScan1(float *input, float *S, float *output,
                                unsigned int N) {
  __shared__ float XY[SECTION_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    XY[threadIdx.x] = input[i];
  } else {
    XY[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    float temp = 0.0f;
    if (threadIdx.x >= stride) {
      temp = XY[threadIdx.x - stride];
    }
    __syncthreads();

    if (threadIdx.x >= stride) {
      XY[threadIdx.x] += temp;
    }
    __syncthreads();
  }

  if (i < N) {
    output[i] = XY[threadIdx.x];
  }

  if (threadIdx.x == blockDim.x - 1) {
    S[blockIdx.x] = XY[blockDim.x - 1];
  }
}

// Second kernel: Scan the block sums
__global__ void koggeStoneScan2(float *S, unsigned int numBlocks) {
  __shared__ float XY[SECTION_SIZE];

  // Process SECTION_SIZE elements at a time
  for (unsigned int blockStart = 0; blockStart < numBlocks;
       blockStart += SECTION_SIZE) {
    // Reset shared memory
    XY[threadIdx.x] = 0.0f;

    // Load block sums into shared memory
    if (blockStart + threadIdx.x < numBlocks) {
      XY[threadIdx.x] = S[blockStart + threadIdx.x];
    }
    __syncthreads();

    // Perform scan on this section
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      float temp = 0.0f;
      if (threadIdx.x >= stride) {
        temp = XY[threadIdx.x - stride];
      }
      __syncthreads();

      if (threadIdx.x >= stride) {
        XY[threadIdx.x] += temp;
      }
      __syncthreads();
    }

    // Write back to global memory
    if (blockStart + threadIdx.x < numBlocks) {
      // Add previous section's last value (if not the first section)
      if (blockStart > 0 && threadIdx.x == 0) {
        // Get the last value from the previous section
        float previousSum = S[blockStart - 1];
        for (unsigned int i = 0; i < SECTION_SIZE && blockStart + i < numBlocks;
             i++) {
          XY[i] += previousSum;
        }
      }
      __syncthreads();

      S[blockStart + threadIdx.x] = XY[threadIdx.x];
    }
    __syncthreads();
  }
}

// Third kernel: Add the scanned block sums to each block's elements
__global__ void koggeStoneScan3(float *S, float *Y, unsigned int N) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N && blockIdx.x > 0) {
    Y[i] += S[blockIdx.x - 1];
  }
}

// CPU implementation for verification
void hierarchicalScan_cpu(float *input, float *output, unsigned int N,
                          unsigned int blockSize) {
  // Simple inclusive scan implementation that doesn't try to mimic the GPU
  // algorithm This will produce the mathematically correct result to verify
  // against

  if (N == 0)
    return;

  // First element is copied as is
  output[0] = input[0];

  // Each subsequent element is the sum of the previous result and current input
  for (unsigned int i = 1; i < N; i++) {
    output[i] = output[i - 1] + input[i];
  }
}

int main() {
  unsigned int N = 1024 * 1024;
  unsigned int bytes = N * sizeof(float);

  // Allocate and initialize host memory
  float *h_input = (float *)malloc(bytes);
  float *h_output_gpu = (float *)malloc(bytes);
  float *h_output_cpu = (float *)malloc(bytes);
  float *h_blockSums = NULL;
  float *h_blockSums_after = NULL;

  // Initialize input with small values to prevent overflow
  for (unsigned int i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(rand() % 10);
  }

  // CPU timing
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);

  cudaEventRecord(start_cpu);
  hierarchicalScan_cpu(h_input, h_output_cpu, N, BLOCK_SIZE);
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
  float *d_input, *d_output, *d_blockSums;
  cudaMalloc((void **)&d_input, bytes);
  cudaMalloc((void **)&d_output, bytes);

  unsigned int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaMalloc((void **)&d_blockSums, numBlocks * sizeof(float));

  // Debug: Allocate host memory to examine block sums
  h_blockSums = (float *)malloc(numBlocks * sizeof(float));
  h_blockSums_after = (float *)malloc(numBlocks * sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

  // Launch kernels
  cudaEventRecord(start_kernel);

  // First kernel: Block-level scan
  koggeStoneScan1<<<numBlocks, BLOCK_SIZE>>>(d_input, d_blockSums, d_output, N);

  // Debug: Copy intermediate results to check them
  cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Second kernel: Scan block sums
  koggeStoneScan2<<<1, BLOCK_SIZE>>>(d_blockSums, numBlocks);

  // Debug: Copy block sums after second kernel
  cudaMemcpy(h_blockSums_after, d_blockSums, numBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Third kernel: Add scanned block sums to each block
  koggeStoneScan3<<<numBlocks, BLOCK_SIZE>>>(d_blockSums, d_output, N);

  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);

  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy result back to host
  cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);

  // Debug: Print some block sums for analysis
  printf("\nDEBUG INFORMATION:\n");
  printf("First few block sums after first kernel:\n");
  for (unsigned int i = 0; i < std::min(5u, numBlocks); i++) {
    printf("  Block %u: %f\n", i, h_blockSums[i]);
  }

  printf("Last few block sums after first kernel:\n");
  for (unsigned int i = std::max(0u, numBlocks - 5); i < numBlocks; i++) {
    printf("  Block %u: %f\n", i, h_blockSums[i]);
  }

  printf("\nFirst few block sums after second kernel:\n");
  for (unsigned int i = 0; i < std::min(5u, numBlocks); i++) {
    printf("  Block %u: %f\n", i, h_blockSums_after[i]);
  }

  printf("Last few block sums after second kernel:\n");
  for (unsigned int i = std::max(0u, numBlocks - 5); i < numBlocks; i++) {
    printf("  Block %u: %f\n", i, h_blockSums_after[i]);
  }

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

      // Print only the first few errors for debugging
      if (significant_errors <= 5) {
        printf("Error at index %u: GPU = %f, CPU = %f, diff = %f\n", i,
               h_output_gpu[i], h_output_cpu[i], error);
      }
    }
  }

  printf("Verification results:\n");
  printf("  Max error: %e\n", max_error);
  printf("  Mean error: %e\n", total_error / N);
  printf("  Significant errors: %d out of %u elements\n", significant_errors,
         N);

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
  free(h_blockSums);
  free(h_blockSums_after);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_blockSums);

  return 0;
}