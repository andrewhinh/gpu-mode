#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int co_rank(int k, int *A, int m, int *B, int n) {
  int i = fminf(k, m);

  int i_low = fmaxf(0, k - n);
  int j_low = fmaxf(0, k - m);
  int i_high = fminf(k, m);
  int j_high = fminf(k, n);

  while (i_low < i_high) {
    int i_mid = (i_low + i_high) / 2;
    int j_mid = k - i_mid;

    if (j_mid > 0 && i_mid < m && B[j_mid - 1] > A[i_mid]) {
      i_low = i_mid + 1;
    } else {
      i_high = i_mid;
    }
  }

  i = i_low;
  return i;
}

__device__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
  int i = 0, j = 0, k = 0;

  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) {
    C[k++] = A[i++];
  }
  while (j < n) {
    C[k++] = B[j++];
  }
}

__global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C,
                                   int tile_size) {
  extern __shared__ int sharedAB[];
  int *A_S = &sharedAB[0];
  int *B_S = &sharedAB[tile_size];

  int C_curr = blockIdx.x * ceilf((m + n) / float(gridDim.x));
  int C_next =
      fminf((blockIdx.x + 1) * ceilf((m + n) / float(gridDim.x)), (m + n));

  if (threadIdx.x == 0) {
    A_S[0] =
        co_rank(C_curr, A, m, B, n); // Make block-level co-rank values visible
    A_S[1] = co_rank(C_next, A, m, B, n); // to other threads in the block
  }

  __syncthreads();

  int A_curr = A_S[0];
  int A_next = A_S[1];
  int B_curr = C_curr - A_curr;
  int B_next = C_next - A_next;

  __syncthreads();

  int counter = 0;
  int C_length = C_next - C_curr;
  int A_length = A_next - A_curr;
  int B_length = B_next - B_curr;
  int total_iteration = ceilf(float(C_length) / tile_size);
  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;

  while (counter < total_iteration) {
    for (int i = 0; i < tile_size; i += blockDim.x) {
      if (i + threadIdx.x < A_length - A_consumed) {
        A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
      }
    }

    for (int i = 0; i < tile_size; i += blockDim.x) {
      if (i + threadIdx.x < B_length - B_consumed) {
        B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
      }
    }

    __syncthreads();

    int c_curr = threadIdx.x * (tile_size / blockDim.x);
    int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
    c_curr =
        (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
    c_next =
        (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

    int a_curr = co_rank(c_curr, A_S, fminf(tile_size, A_length - A_consumed),
                         B_S, fminf(tile_size, B_length - B_consumed));
    int b_curr = c_curr - a_curr;
    int a_next = co_rank(c_next, A_S, fminf(tile_size, A_length - A_consumed),
                         B_S, fminf(tile_size, B_length - B_consumed));
    int b_next = c_next - a_next;

    merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr,
                     b_next - b_curr, C + C_curr + C_completed + c_curr);

    counter++;
    C_completed += tile_size;
    A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
    B_consumed = C_completed - A_consumed;

    __syncthreads();
  }
}

void parallel_merge(int *A, int m, int *B, int n, int *C, int numBlocks,
                    int threadsPerBlock) {
  int tile_size = 1024;
  int sharedMemSize = 2 * tile_size * sizeof(int);

  merge_tiled_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      A, m, B, n, C, tile_size);
  cudaDeviceSynchronize();
}

__global__ void merge_sort_kernel(int *input, int *output, int n, int width) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int start = tid * 2 * width;

  if (start >= n)
    return;

  int mid = fminf(start + width, n);
  int end = fminf(start + 2 * width, n);

  int i = start;
  int j = mid;
  int k = start;

  while (i < mid && j < end) {
    if (input[i] <= input[j]) {
      output[k++] = input[i++];
    } else {
      output[k++] = input[j++];
    }
  }
  while (i < mid) {
    output[k++] = input[i++];
  }
  while (j < end) {
    output[k++] = input[j++];
  }
}

void parallel_merge_sort(int *d_data, int *d_temp, int n) {
  int *d_in = d_data;
  int *d_out = d_temp;

  int threadsPerBlock = 256;

  // Initial distribution of elements (each thread handles one element)
  // For the first pass, width=1, so each thread merges two single elements

  for (int width = 1; width < n; width *= 2) {
    int numBlocks =
        (n + 2 * width * threadsPerBlock - 1) / (2 * width * threadsPerBlock);

    merge_sort_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, n, width);
    cudaDeviceSynchronize();

    // Swap input and output for the next iteration
    int *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  // If result is in d_temp, copy back to d_data
  if (d_in != d_data) {
    cudaMemcpy(d_data, d_in, n * sizeof(int), cudaMemcpyDeviceToDevice);
  }
}

void advanced_parallel_merge_sort(int *d_data, int *d_temp, int n) {
  int *d_in = d_data;
  int *d_out = d_temp;

  int threadsPerBlock = 256;

  int mergeThreshold = 1024;

  for (int width = 1; width < n; width *= 2) {
    if (width < mergeThreshold) {
      int numBlocks =
          (n + 2 * width * threadsPerBlock - 1) / (2 * width * threadsPerBlock);
      merge_sort_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, n, width);
    } else {
      for (int i = 0; i < n; i += 2 * width) {
        int mid = fminf(i + width, n);
        int end = fminf(i + 2 * width, n);

        if (mid < end) {
          parallel_merge(d_in + i, mid - i, d_in + mid, end - mid, d_out + i,
                         32, threadsPerBlock);
        } else {
          cudaMemcpy(d_out + i, d_in + i, (n - i) * sizeof(int),
                     cudaMemcpyDeviceToDevice);
        }
      }
    }

    cudaDeviceSynchronize();

    int *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  if (d_in != d_data) {
    cudaMemcpy(d_data, d_in, n * sizeof(int), cudaMemcpyDeviceToDevice);
  }
}

int main() {
  int n = 1000000;
  int *h_data = (int *)malloc(n * sizeof(int));

  for (int i = 0; i < n; i++) {
    h_data[i] = rand() % 1000000;
  }

  int *d_data;
  int *d_temp;
  cudaMalloc((void **)&d_data, n * sizeof(int));
  cudaMalloc((void **)&d_temp, n * sizeof(int));

  cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  advanced_parallel_merge_sort(d_data, d_temp, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Parallel merge sort completed in %.2f ms\n", milliseconds);

  cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

  bool sorted = true;
  for (int i = 1; i < n; i++) {
    if (h_data[i] < h_data[i - 1]) {
      sorted = false;
      printf("Error: Data is not sorted at index %d\n", i);
      break;
    }
  }

  if (sorted) {
    printf("Sort verification successful!\n");
  }

  free(h_data);
  cudaFree(d_data);
  cudaFree(d_temp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}