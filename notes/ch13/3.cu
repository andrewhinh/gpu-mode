__global__ void radix_sort_iter_coarsened(unsigned int *input,
                                          unsigned int *output,
                                          unsigned int *bits, unsigned int N,
                                          unsigned int iter,
                                          int items_per_thread) {
  const unsigned int tid = threadIdx.x;
  const unsigned int block_offset = blockIdx.x * blockDim.x * items_per_thread;

  unsigned int keys[8]; // 8 items per thread
  unsigned int thread_bits[8];

#pragma unroll
  for (int k = 0; k < items_per_thread; k++) {
    const unsigned int i = block_offset + tid + k * blockDim.x;
    if (i < N) {
      keys[k] = input[i];
      thread_bits[k] = (keys[k] >> iter) & 1;
      bits[i] = thread_bits[k];
    }
  }

  __syncthreads();

  exclusiveScan(bits, N);

  __syncthreads();

#pragma unroll
  for (int k = 0; k < items_per_thread; k++) {
    const unsigned int i = block_offset + tid + k * blockDim.x;
    if (i < N) {
      unsigned int numOnesBefore = bits[i];
      unsigned int numOnesTotal = bits[N];
      unsigned int dst = (thread_bits[k] == 0)
                             ? (i - numOnesBefore)
                             : (numOnesTotal + i - numOnesBefore);
      output[dst] = keys[k];
    }
  }
}
