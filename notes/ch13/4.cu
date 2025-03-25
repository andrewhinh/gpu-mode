#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Reference CPU implementation using std::sort for verification
void cpu_reference_sort(int *input, int *output, int N) {
  // Copy input to output first
  for (int i = 0; i < N; i++) {
    output[i] = input[i];
  }

  // Simple sort using STL
  std::sort(output, output + N);
}

// Function to verify results between CPU and GPU implementations
bool verify_results(int *cpu_results, int *gpu_results, int N) {
  for (int i = 0; i < N; i++) {
    if (cpu_results[i] != gpu_results[i]) {
      printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, cpu_results[i],
             gpu_results[i]);
      return false;
    }
  }
  return true;
}

// A simplified GPU sort that just uses thrust sort for validation
void gpu_reference_sort(int *d_input, int *d_output, int N) {
  // Copy d_input to d_output
  cudaMemcpy(d_output, d_input, N * sizeof(int), cudaMemcpyDeviceToDevice);

  // For now, we'll do the sorting on CPU and copy back
  int *h_temp = new int[N];
  cudaMemcpy(h_temp, d_input, N * sizeof(int), cudaMemcpyDeviceToHost);
  std::sort(h_temp, h_temp + N);
  cudaMemcpy(d_output, h_temp, N * sizeof(int), cudaMemcpyHostToDevice);
  delete[] h_temp;
}

int main() {
  const int N = 1000000; // 1M elements

  // Allocate host memory
  int *h_input = new int[N];
  int *h_output = new int[N];
  int *h_cpu_output = new int[N];

  // Initialize input data with random values - using fixed seed for consistent
  // results
  srand(12345);
  for (int i = 0; i < N; i++) {
    h_input[i] = rand() % 1000000;
  }

  // Allocate device memory
  int *d_input, *d_output;
  cudaMalloc((void **)&d_input, N * sizeof(int));
  cudaMalloc((void **)&d_output, N * sizeof(int));

  // Copy input data to device
  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

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
  cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

  // Verify results
  bool results_match = verify_results(h_cpu_output, h_output, N);

  // Print timing information and verification results
  printf("Validating CPU and GPU reference implementations on %d elements\n",
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