#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void computeHistogram(int *row_indices, int *row_counts, int nnz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nnz) {
    atomicAdd(&row_counts[row_indices[idx]], 1);
  }
}

__global__ void exclusivePrefixSum(int *input, int *output, int size) {
  extern __shared__ int temp[];

  int tid = threadIdx.x;
  int offset = 1;

  int ai, bi;
  if (tid < size) {
    temp[tid] = input[tid];
  } else {
    temp[tid] = 0;
  }
  __syncthreads();

  for (int d = size / 2; d > 0; d /= 2) {
    if (tid < d) {
      ai = offset * (2 * tid + 1) - 1;
      bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
    __syncthreads();
  }

  if (tid == 0)
    temp[size - 1] = 0;
  __syncthreads();

  for (int d = 1; d < size; d *= 2) {
    offset /= 2;
    if (tid < d) {
      ai = offset * (2 * tid + 1) - 1;
      bi = offset * (2 * tid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
    __syncthreads();
  }

  if (tid < size) {
    output[tid] = temp[tid];
  }
}

// GPU implementation of COO to CSR conversion
void cooToCSR(int *coo_row, int *coo_col, int *coo_val, int *csr_row_ptr,
              int *csr_col, int *csr_val, int nnz, int num_rows) {

  int *d_coo_row, *d_coo_col, *d_coo_val;
  int *d_csr_row_ptr, *d_row_counts;

  cudaMalloc(&d_coo_row, nnz * sizeof(int));
  cudaMalloc(&d_coo_col, nnz * sizeof(int));
  cudaMalloc(&d_coo_val, nnz * sizeof(int));
  cudaMalloc(&d_csr_row_ptr, (num_rows + 1) * sizeof(int));
  cudaMalloc(&d_row_counts, num_rows * sizeof(int));

  cudaMemcpy(d_coo_row, coo_row, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_col, coo_col, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coo_val, coo_val, nnz * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_row_counts, 0, num_rows * sizeof(int));

  int blockSize = 256;
  int gridSize = (nnz + blockSize - 1) / blockSize;
  computeHistogram<<<gridSize, blockSize>>>(d_coo_row, d_row_counts, nnz);

  cudaMemset(d_csr_row_ptr, 0, (num_rows + 1) * sizeof(int));
  cudaMemcpy(d_csr_row_ptr, d_row_counts, num_rows * sizeof(int),
             cudaMemcpyDeviceToDevice);

  int *row_counts = new int[num_rows];
  cudaMemcpy(row_counts, d_row_counts, num_rows * sizeof(int),
             cudaMemcpyDeviceToHost);

  csr_row_ptr[0] = 0;
  for (int i = 0; i < num_rows; i++) {
    csr_row_ptr[i + 1] = csr_row_ptr[i] + row_counts[i];
  }

  cudaMemcpy(d_csr_row_ptr, csr_row_ptr, (num_rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  int *row_offsets = new int[num_rows];
  for (int i = 0; i < num_rows; i++) {
    row_offsets[i] = csr_row_ptr[i];
  }

  for (int i = 0; i < nnz; i++) {
    int row = coo_row[i];
    int dest = row_offsets[row]++;
    csr_col[dest] = coo_col[i];
    csr_val[dest] = coo_val[i];
  }

  delete[] row_counts;
  delete[] row_offsets;

  cudaFree(d_coo_row);
  cudaFree(d_coo_col);
  cudaFree(d_coo_val);
  cudaFree(d_csr_row_ptr);
  cudaFree(d_row_counts);
}

// CPU implementation of COO to CSR conversion for verification
void cooToCSR_CPU(int *coo_row, int *coo_col, int *coo_val, int *csr_row_ptr,
                  int *csr_col, int *csr_val, int nnz, int num_rows) {
  // First, count number of non-zeros in each row
  int *row_counts = new int[num_rows]();
  for (int i = 0; i < nnz; i++) {
    row_counts[coo_row[i]]++;
  }

  // Calculate row pointers
  csr_row_ptr[0] = 0;
  for (int i = 0; i < num_rows; i++) {
    csr_row_ptr[i + 1] = csr_row_ptr[i] + row_counts[i];
  }

  // Reset row counts to use as current indices
  int *row_offsets = new int[num_rows];
  for (int i = 0; i < num_rows; i++) {
    row_offsets[i] = csr_row_ptr[i];
  }

  // Fill in column indices and values
  for (int i = 0; i < nnz; i++) {
    int row = coo_row[i];
    int dest = row_offsets[row]++;
    csr_col[dest] = coo_col[i];
    csr_val[dest] = coo_val[i];
  }

  delete[] row_counts;
  delete[] row_offsets;
}

// Function to verify CSR results
bool verifyCSR(int *csr_row_ptr1, int *csr_col1, int *csr_val1,
               int *csr_row_ptr2, int *csr_col2, int *csr_val2, int num_rows,
               int nnz) {
  // Verify row pointers
  for (int i = 0; i <= num_rows; i++) {
    if (csr_row_ptr1[i] != csr_row_ptr2[i]) {
      printf("Row pointer mismatch at index %d: CPU = %d, GPU = %d\n", i,
             csr_row_ptr1[i], csr_row_ptr2[i]);
      return false;
    }
  }

  // Verify column indices and values
  for (int i = 0; i < nnz; i++) {
    if (csr_col1[i] != csr_col2[i]) {
      printf("Column index mismatch at index %d: CPU = %d, GPU = %d\n", i,
             csr_col1[i], csr_col2[i]);
      return false;
    }
    if (csr_val1[i] != csr_val2[i]) {
      printf("Value mismatch at index %d: CPU = %d, GPU = %d\n", i, csr_val1[i],
             csr_val2[i]);
      return false;
    }
  }

  return true;
}

int main() {
  int num_rows = 4;
  int nnz = 7;

  int coo_row[7] = {0, 0, 1, 2, 2, 3, 3};
  int coo_col[7] = {0, 2, 2, 1, 2, 0, 3};
  int coo_val[7] = {1, 7, 8, 4, 3, 2, 1};

  int *csr_row_ptr_gpu = new int[num_rows + 1];
  int *csr_col_gpu = new int[nnz];
  int *csr_val_gpu = new int[nnz];

  int *csr_row_ptr_cpu = new int[num_rows + 1];
  int *csr_col_cpu = new int[nnz];
  int *csr_val_cpu = new int[nnz];

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure CPU implementation time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cooToCSR_CPU(coo_row, coo_col, coo_val, csr_row_ptr_cpu, csr_col_cpu,
               csr_val_cpu, nnz, num_rows);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // Measure GPU implementation time
  cudaEventRecord(start);
  cooToCSR(coo_row, coo_col, coo_val, csr_row_ptr_gpu, csr_col_gpu, csr_val_gpu,
           nnz, num_rows);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Verify results
  bool results_match =
      verifyCSR(csr_row_ptr_cpu, csr_col_cpu, csr_val_cpu, csr_row_ptr_gpu,
                csr_col_gpu, csr_val_gpu, num_rows, nnz);

  // Print timing and verification results
  printf("COO to CSR Conversion Performance (%d rows, %d non-zeros):\n",
         num_rows, nnz);
  printf("CPU Time: %.4f ms\n", cpu_duration.count());
  printf("GPU Time: %.4f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n\n", results_match ? "PASSED" : "FAILED");

  delete[] csr_row_ptr_gpu;
  delete[] csr_col_gpu;
  delete[] csr_val_gpu;
  delete[] csr_row_ptr_cpu;
  delete[] csr_col_cpu;
  delete[] csr_val_cpu;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
