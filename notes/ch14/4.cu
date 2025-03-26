#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ellSpMV(int num_rows, int num_cols_per_row,
                        const int *__restrict__ ell_indices,
                        const float *__restrict__ ell_data,
                        const float *__restrict__ x, float *__restrict__ y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < num_rows) {
    float sum = 0.0f;

    for (int col = 0; col < num_cols_per_row; col++) {
      int idx = row + col * num_rows;
      int j = ell_indices[idx];

      if (j >= 0) {
        sum += ell_data[idx] * x[j];
      }
    }

    y[row] = sum;
  }
}

// GPU implementation of hybrid ELL-COO SpMV
void hybridELLCOOSpMV(int num_rows, int num_cols, int ell_cols_per_row,
                      int *ell_indices, float *ell_data, int *coo_row,
                      int *coo_col, float *coo_val, int coo_nnz, float *x,
                      float *y) {

  int *d_ell_indices;
  float *d_ell_data;

  float *d_x, *d_y;

  cudaMalloc(&d_ell_indices, num_rows * ell_cols_per_row * sizeof(int));
  cudaMalloc(&d_ell_data, num_rows * ell_cols_per_row * sizeof(float));
  cudaMalloc(&d_x, num_cols * sizeof(float));
  cudaMalloc(&d_y, num_rows * sizeof(float));

  cudaMemset(d_y, 0, num_rows * sizeof(float));

  cudaMemcpy(d_ell_indices, ell_indices,
             num_rows * ell_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ell_data, ell_data, num_rows * ell_cols_per_row * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (num_rows + blockSize - 1) / blockSize;
  ellSpMV<<<gridSize, blockSize>>>(num_rows, ell_cols_per_row, d_ell_indices,
                                   d_ell_data, d_x, d_y);

  cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < coo_nnz; i++) {
    y[coo_row[i]] += coo_val[i] * x[coo_col[i]];
  }

  cudaFree(d_ell_indices);
  cudaFree(d_ell_data);
  cudaFree(d_x);
  cudaFree(d_y);
}

// CPU implementation of hybrid ELL-COO SpMV for verification
void hybridELLCOOSpMV_CPU(int num_rows, int num_cols, int ell_cols_per_row,
                          int *ell_indices, float *ell_data, int *coo_row,
                          int *coo_col, float *coo_val, int coo_nnz, float *x,
                          float *y) {
  // Initialize output vector to zero
  for (int i = 0; i < num_rows; i++) {
    y[i] = 0.0f;
  }

  // ELL portion
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < ell_cols_per_row; col++) {
      int idx = row + col * num_rows;
      int j = ell_indices[idx];

      if (j >= 0) {
        y[row] += ell_data[idx] * x[j];
      }
    }
  }

  // COO portion
  for (int i = 0; i < coo_nnz; i++) {
    y[coo_row[i]] += coo_val[i] * x[coo_col[i]];
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

  int ell_cols_per_row = 2;

  int ell_indices[8] = {0, -1, 2, 0, 2, -1, -1, -1};
  float ell_data[8] = {1.0f, 0.0f, 2.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f};

  int coo_nnz = 3;
  int coo_row[3] = {1, 2, 3};
  int coo_col[3] = {1, 3, 3};
  float coo_val[3] = {5.0f, 9.0f, 7.0f};

  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y_gpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float y_cpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure CPU implementation time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  hybridELLCOOSpMV_CPU(num_rows, num_cols, ell_cols_per_row, ell_indices,
                       ell_data, coo_row, coo_col, coo_val, coo_nnz, x, y_cpu);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // Measure GPU implementation time
  cudaEventRecord(start);
  hybridELLCOOSpMV(num_rows, num_cols, ell_cols_per_row, ell_indices, ell_data,
                   coo_row, coo_col, coo_val, coo_nnz, x, y_gpu);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Verify results
  bool results_match = verifyResults(y_cpu, y_gpu, num_rows);

  // Print timing and verification results
  printf("Hybrid ELL-COO SpMV Performance (%d rows, %d columns, %d ELL cols "
         "per row, %d COO nnz):\n",
         num_rows, num_cols, ell_cols_per_row, coo_nnz);
  printf("CPU Time: %.4f ms\n", cpu_duration.count());
  printf("GPU Time: %.4f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n\n", results_match ? "PASSED" : "FAILED");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
