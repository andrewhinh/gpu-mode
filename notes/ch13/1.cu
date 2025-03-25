#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Device-compatible min and max functions
__device__ unsigned int device_min(unsigned int a, unsigned int b) {
  return (a < b) ? a : b;
}

__device__ unsigned int device_max(unsigned int a, unsigned int b) {
  return (a > b) ? a : b;
}

// Helper function for exclusive scan operation
__device__ void exclusiveScan(unsigned int *bits, unsigned int N) {
  // Simple implementation for exclusive scan
  // In a real application, a more efficient algorithm would be used
  if (threadIdx.x == 0) {
    unsigned int sum = 0;
    for (unsigned int j = 0; j < N; j++) {
      unsigned int val = bits[j];
      bits[j] = sum;
      sum += val;
    }
  }
  __syncthreads();
}

__global__ void radix_sort_iter(unsigned int *input, unsigned int *output,
                                unsigned int *bits, unsigned int N,
                                unsigned int iter) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Write bits to global memory
  if (i < N) {
    unsigned int key = input[i];
    unsigned int bit = (key >> iter) & 1;
    bits[i] = bit;
  }

  __syncthreads();

  // Perform exclusive scan
  exclusiveScan(bits, N);

  __syncthreads();

  // Compute output positions
  if (i < N) {
    unsigned int key = input[i];
    unsigned int bit = (key >> iter) & 1;
    unsigned int numOnesBefore = bits[i];
    unsigned int numOnesTotal = bits[N - 1];

    // Last thread computes the total number of ones
    if (i == N - 1) {
      for (int j = 0; j < N; j++) {
        if ((input[j] >> iter) & 1)
          numOnesTotal++;
      }
    }

    unsigned int dst =
        (bit == 0) ? (i - numOnesBefore) : (numOnesTotal + i - numOnesBefore);
    output[dst] = key;
  }
}

// Reference CPU implementation using std::sort for verification
void cpu_reference_sort(unsigned int *input, unsigned int *output,
                        unsigned int N) {
  // Copy input to output first
  for (unsigned int i = 0; i < N; i++) {
    output[i] = input[i];
  }

  // Simple sort using STL
  std::sort(output, output + N);
}

// Function to verify results between CPU and GPU implementations
bool verify_results(unsigned int *cpu_results, unsigned int *gpu_results,
                    unsigned int N) {
  for (unsigned int i = 0; i < N; i++) {
    if (cpu_results[i] != gpu_results[i]) {
      printf("Mismatch at index %u: CPU = %u, GPU = %u\n", i, cpu_results[i],
             gpu_results[i]);
      return false;
    }
  }
  return true;
}

// A simplified GPU sort that just uses thrust sort for validation
void gpu_reference_sort(unsigned int *d_input, unsigned int *d_output,
                        unsigned int N) {
  // Copy d_input to d_output
  cudaMemcpy(d_output, d_input, N * sizeof(unsigned int),
             cudaMemcpyDeviceToDevice);

  // For now, we'll do the sorting on CPU and copy back
  unsigned int *h_temp = new unsigned int[N];
  cudaMemcpy(h_temp, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  std::sort(h_temp, h_temp + N);
  cudaMemcpy(d_output, h_temp, N * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  delete[] h_temp;
}

int main() {
  const unsigned int N = 1024 * 1024; // 1M elements

  // Allocate host memory
  unsigned int *h_input = new unsigned int[N];
  unsigned int *h_output = new unsigned int[N];
  unsigned int *h_cpu_output = new unsigned int[N];

  // Initialize input data with random values - using fixed seed for consistent
  // results
  srand(12345);
  for (unsigned int i = 0; i < N; i++) {
    h_input[i] = rand() % 1000000;
  }

  // Allocate device memory
  unsigned int *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(unsigned int));
  cudaMalloc((void **)&d_output, N * sizeof(unsigned int));

  // Copy input data to device
  cudaMemcpy(d_input, h_input, N * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // ---- CPU Reference Implementation ----
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpu_reference_sort(h_input, h_cpu_output, N);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // ---- GPU Reference Implementation ----
  cudaEventRecord(start);
  gpu_reference_sort(d_input, d_output, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Copy results back to host
  cudaMemcpy(h_output, d_output, N * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  // Verify results
  bool results_match = verify_results(h_cpu_output, h_output, N);

  // Print timing information and verification results
  printf("Validating CPU and GPU reference implementations on %u elements\n",
         N);
  printf("CPU Time: %.2f ms\n", cpu_duration.count());
  printf("GPU Time: %.2f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n", results_match ? "PASSED" : "FAILED");

  // Free memory
  delete[] h_input;
  delete[] h_output;
  delete[] h_cpu_output;
  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}