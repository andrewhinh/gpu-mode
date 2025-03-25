#include "../common.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits.h>

#define TILE_SIZE 256

// Device-compatible min and max functions
__device__ __host__ int device_min(int a, int b) { return (a < b) ? a : b; }

__device__ __host__ int device_max(int a, int b) { return (a > b) ? a : b; }

__device__ __host__ void co_rank(int k,        // Position in the merged array
                                 const int *A, // Array A
                                 int m, const int *B,     // Array B
                                 int n, int *i, int *j) { // Output parameters
  int i_low = (k - n > 0) ? (k - n) : 0;
  int i_high = (k < m) ? k : m;

  while (i_low < i_high) {
    int i_mid = (i_low + i_high) / 2;
    int j_mid = k - i_mid;

    if (j_mid > 0 && i_mid < m && B[j_mid - 1] > A[i_mid]) {
      i_low = i_mid + 1;
    } else {
      i_high = i_mid;
    }
  }

  *i = i_low;
  *j = k - i_low;
}

__global__ void merge_kernel(const int *A, int A_length, const int *B,
                             int B_length, int *C, int C_length,
                             int *merged_offset) {
  // Calculate the range of C elements this block is responsible for
  int C_per_block = (C_length + gridDim.x - 1) / gridDim.x;
  int C_begin = blockIdx.x * C_per_block;
  int C_end = device_min(C_begin + C_per_block, C_length);

  // Find corresponding positions in A and B for this partition
  int A_begin, B_begin;
  co_rank(C_begin, A, A_length, B, B_length, &A_begin, &B_begin);

  int A_end, B_end;
  co_rank(C_end, A, A_length, B, B_length, &A_end, &B_end);

  // Store partition information for debugging
  if (threadIdx.x == 0) {
    merged_offset[blockIdx.x * 4] = C_begin;
    merged_offset[blockIdx.x * 4 + 1] = C_end;
    merged_offset[blockIdx.x * 4 + 2] = A_begin;
    merged_offset[blockIdx.x * 4 + 3] = B_begin;
  }

  // Make sure we don't go out of bounds
  A_begin = device_min(A_begin, A_length);
  B_begin = device_min(B_begin, B_length);
  A_end = device_min(A_end, A_length);
  B_end = device_min(B_end, B_length);

  // Each thread processes elements from C_begin + threadIdx.x with stride
  // blockDim.x
  for (int c_idx = C_begin + threadIdx.x; c_idx < C_end; c_idx += blockDim.x) {
    // Find positions in A and B for this output position
    int a_pos, b_pos;
    co_rank(c_idx - C_begin, A + A_begin, A_end - A_begin, B + B_begin,
            B_end - B_begin, &a_pos, &b_pos);

    a_pos += A_begin;
    b_pos += B_begin;

    // Make sure we don't go out of bounds
    bool a_valid = (a_pos < A_end);
    bool b_valid = (b_pos < B_end);

    if (a_valid && (!b_valid || A[a_pos] <= B[b_pos])) {
      C[c_idx] = A[a_pos];
    } else if (b_valid) {
      C[c_idx] = B[b_pos];
    }
  }
}

void merge_cpu(const int *A, int A_length, const int *B, int B_length, int *C) {
  int i = 0, j = 0, k = 0;

  while (i < A_length && j < B_length) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }

  // Copy remaining elements
  while (i < A_length) {
    C[k++] = A[i++];
  }

  while (j < B_length) {
    C[k++] = B[j++];
  }
}

int main() {
  const int A_LENGTH = 1024 * 1024;
  const int B_LENGTH = 1024 * 512; // Half the size of A
  const int C_LENGTH = A_LENGTH + B_LENGTH;

  size_t A_bytes = A_LENGTH * sizeof(int);
  size_t B_bytes = B_LENGTH * sizeof(int);
  size_t C_bytes = C_LENGTH * sizeof(int);

  // Host memory allocation
  int *h_A = (int *)malloc(A_bytes);
  int *h_B = (int *)malloc(B_bytes);
  int *h_C_gpu = (int *)malloc(C_bytes);
  int *h_C_cpu = (int *)malloc(C_bytes);
  int *h_merged_offset = (int *)malloc(1024 * 4 * sizeof(int)); // For debugging

  // Initialize sorted input arrays
  for (int i = 0; i < A_LENGTH; i++) {
    h_A[i] = 2 * i; // Even numbers
  }

  for (int i = 0; i < B_LENGTH; i++) {
    h_B[i] = 2 * i + 1; // Odd numbers
  }

  // CPU timing
  cudaEvent_t start_cpu, stop_cpu;
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);

  cudaEventRecord(start_cpu);
  merge_cpu(h_A, A_LENGTH, h_B, B_LENGTH, h_C_cpu);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);

  float cpu_time = 0.0f;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  printf("CPU execution time: %f milliseconds\n", cpu_time);

  // Device memory allocation
  int *d_A, *d_B, *d_C, *d_merged_offset;
  cudaMalloc((void **)&d_A, A_bytes);
  cudaMalloc((void **)&d_B, B_bytes);
  cudaMalloc((void **)&d_C, C_bytes);
  cudaMalloc((void **)&d_merged_offset, 1024 * 4 * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice);

  // GPU timing
  cudaEvent_t start_gpu, stop_gpu, start_kernel, stop_kernel;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);

  cudaEventRecord(start_gpu);

  // Launch kernel
  int threads_per_block = 256;
  int blocks_per_grid =
      std::min(1024, (C_LENGTH + threads_per_block - 1) / threads_per_block);

  cudaEventRecord(start_kernel);
  merge_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_A, A_LENGTH, d_B, B_LENGTH, d_C, C_LENGTH, d_merged_offset);
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
  printf("GPU kernel execution time: %f milliseconds\n", kernel_time);

  // Copy results back to host
  cudaMemcpy(h_C_gpu, d_C, C_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_merged_offset, d_merged_offset, 1024 * 4 * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_time = 0.0f;
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  printf("GPU total execution time: %f milliseconds\n", gpu_time);

  // Verify results
  bool correct = true;
  int first_mismatch_idx = -1;
  for (int i = 0; i < C_LENGTH; i++) {
    if (h_C_gpu[i] != h_C_cpu[i]) {
      correct = false;
      first_mismatch_idx = i;
      break;
    }
  }

  if (correct) {
    printf("Verification PASSED!\n");
  } else {
    printf("Verification FAILED!\n");
    printf("First mismatch at index %d: GPU = %d, CPU = %d\n",
           first_mismatch_idx, h_C_gpu[first_mismatch_idx],
           h_C_cpu[first_mismatch_idx]);

    // Print a few values around the mismatch for context
    int start_idx = std::max(0, first_mismatch_idx - 5);
    int end_idx = std::min(C_LENGTH - 1, first_mismatch_idx + 5);

    printf("Values around mismatch (index: GPU/CPU):\n");
    for (int i = start_idx; i <= end_idx; i++) {
      printf("%d: %d/%d%s\n", i, h_C_gpu[i], h_C_cpu[i],
             (i == first_mismatch_idx) ? " <-- MISMATCH" : "");
    }
  }

  // Print speedup
  printf("Speedup (CPU vs GPU kernel): %.2fx\n", cpu_time / kernel_time);
  printf("Speedup (CPU vs GPU total): %.2fx\n", cpu_time / gpu_time);

  // Clean up
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_merged_offset);

  free(h_A);
  free(h_B);
  free(h_C_gpu);
  free(h_C_cpu);
  free(h_merged_offset);

  // Clean up CUDA events
  cudaEventDestroy(start_cpu);
  cudaEventDestroy(stop_cpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);

  return 0;
}