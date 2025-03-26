#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void jdsSpMV(const int *__restrict__ jds_row_ptr,
                        const int *__restrict__ col_idx,
                        const float *__restrict__ data,
                        const int *__restrict__ perm,
                        const float *__restrict__ x, float *__restrict__ y,
                        int num_rows, int max_row_nnz) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows) {
    int orig_row = perm[row];
    float sum = 0.0f;

    for (int j = 0; j < max_row_nnz; j++) {
      // Skip if this row has fewer non-zeros than the current jagged diagonal
      if (j >= jds_row_ptr[row + 1] - jds_row_ptr[row]) {
        break;
      }

      int idx = jds_row_ptr[row] + j;
      if (idx >= jds_row_ptr[row + 1]) {
        break;
      }

      sum += data[idx] * x[col_idx[idx]];
    }

    y[orig_row] = sum;
  }
}

// GPU implementation of JDS SpMV
void jdsSpMVHost(int *jds_row_ptr, int *col_idx, float *data, int *perm,
                 float *x, float *y, int num_rows, int num_cols,
                 int max_row_nnz) {

  int *d_jds_row_ptr, *d_col_idx, *d_perm;
  float *d_data, *d_x, *d_y;

  cudaMalloc(&d_jds_row_ptr, (num_rows + 1) * sizeof(int));
  cudaMalloc(&d_col_idx, jds_row_ptr[num_rows] * sizeof(int));
  cudaMalloc(&d_data, jds_row_ptr[num_rows] * sizeof(float));
  cudaMalloc(&d_perm, num_rows * sizeof(int));
  cudaMalloc(&d_x, num_cols * sizeof(float));
  cudaMalloc(&d_y, num_rows * sizeof(float));

  cudaMemcpy(d_jds_row_ptr, jds_row_ptr, (num_rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx, jds_row_ptr[num_rows] * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, data, jds_row_ptr[num_rows] * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_perm, perm, num_rows * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(d_y, 0, num_rows * sizeof(float));

  int blockSize = 256;
  int gridSize = (num_rows + blockSize - 1) / blockSize;
  jdsSpMV<<<gridSize, blockSize>>>(d_jds_row_ptr, d_col_idx, d_data, d_perm,
                                   d_x, d_y, num_rows, max_row_nnz);

  cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_jds_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_data);
  cudaFree(d_perm);
  cudaFree(d_x);
  cudaFree(d_y);
}

// CPU implementation of JDS SpMV for verification
void jdsSpMVHost_CPU(int *jds_row_ptr, int *col_idx, float *data, int *perm,
                     float *x, float *y, int num_rows, int num_cols,
                     int max_row_nnz) {
  // Initialize output vector to zero
  for (int i = 0; i < num_rows; i++) {
    y[i] = 0.0f;
  }

  // Process each JDS row
  for (int row = 0; row < num_rows; row++) {
    int orig_row = perm[row];
    float sum = 0.0f;

    for (int j = 0; j < max_row_nnz; j++) {
      // Skip if this row has fewer non-zeros than the current jagged diagonal
      if (j >= jds_row_ptr[row + 1] - jds_row_ptr[row]) {
        break;
      }

      int idx = jds_row_ptr[row] + j;
      if (idx >= jds_row_ptr[row + 1]) {
        break;
      }

      sum += data[idx] * x[col_idx[idx]];
    }

    y[orig_row] = sum;
  }
}

// Function to verify results
bool verifyResults(float *y_cpu, float *y_gpu, int num_rows,
                   float tolerance = 1e-5f) {
  for (int i = 0; i < num_rows; i++) {
    if (fabsf(y_cpu[i] - y_gpu[i]) > tolerance) {
      printf("Result mismatch at index %d: CPU = %f, GPU = %f\n", i, y_cpu[i],
             y_gpu[i]);
      return false;
    }
  }
  return true;
}

int main() {
  int num_rows = 4;
  int num_cols = 4;
  int max_row_nnz = 2;

  // Original matrix:
  // 1 0 7 0
  // 0 0 8 0
  // 0 4 3 0
  // 2 0 0 1

  int perm[4] = {0, 2, 3, 1}; // Rows in order of non-zero count
  int jds_row_ptr[5] = {0, 2, 4, 6,
                        7}; // Max 2 non-zeros per row, so num_rows+1 entries

  // Column indices and data arranged by jagged diagonals
  // First jagged diagonal (first non-zero in each row):
  // Row 0 (orig 0): col 0, val 1
  // Row 1 (orig 2): col 1, val 4
  // Row 2 (orig 3): col 0, val 2
  // Row 3 (orig 1): col 2, val 8

  // Second jagged diagonal (second non-zero in each row):
  // Row 0 (orig 0): col 2, val 7
  // Row 1 (orig 2): col 2, val 3
  // Row 2 (orig 3): col 3, val 1

  int col_idx[7] = {// First jagged diagonal
                    0, 1, 0, 2,
                    // Second jagged diagonal
                    2, 2, 3};

  float data[7] = {// First jagged diagonal
                   1.0f, 4.0f, 2.0f, 8.0f,
                   // Second jagged diagonal
                   7.0f, 3.0f, 1.0f};

  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y_gpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float y_cpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure CPU implementation time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  jdsSpMVHost_CPU(jds_row_ptr, col_idx, data, perm, x, y_cpu, num_rows,
                  num_cols, max_row_nnz);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // Measure GPU implementation time
  cudaEventRecord(start);
  jdsSpMVHost(jds_row_ptr, col_idx, data, perm, x, y_gpu, num_rows, num_cols,
              max_row_nnz);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Verify results
  bool results_match = verifyResults(y_cpu, y_gpu, num_rows);

  // Print timing and verification results
  printf(
      "JDS SpMV Performance (%d rows, %d columns, %d max non-zeros per row):\n",
      num_rows, num_cols, max_row_nnz);
  printf("CPU Time: %.4f ms\n", cpu_duration.count());
  printf("GPU Time: %.4f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n\n", results_match ? "PASSED" : "FAILED");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
