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

int main() {
  int num_rows = 4;
  int nnz = 7;

  int coo_row[7] = {0, 0, 1, 2, 2, 3, 3};
  int coo_col[7] = {0, 2, 2, 1, 2, 0, 3};
  int coo_val[7] = {1, 7, 8, 4, 3, 2, 1};

  int *csr_row_ptr = new int[num_rows + 1];
  int *csr_col = new int[nnz];
  int *csr_val = new int[nnz];

  cooToCSR(coo_row, coo_col, coo_val, csr_row_ptr, csr_col, csr_val, nnz,
           num_rows);

  printf("CSR Row Pointers: ");
  for (int i = 0; i <= num_rows; i++) {
    printf("%d ", csr_row_ptr[i]);
  }
  printf("\nCSR Column Indices: ");
  for (int i = 0; i < nnz; i++) {
    printf("%d ", csr_col[i]);
  }
  printf("\nCSR Values: ");
  for (int i = 0; i < nnz; i++) {
    printf("%d ", csr_val[i]);
  }
  printf("\n");

  delete[] csr_row_ptr;
  delete[] csr_col;
  delete[] csr_val;

  return 0;
}
