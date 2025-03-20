__global__ void radix_sort_iter_multibit(unsigned int *input,
                                         unsigned int *output,
                                         unsigned int *bits, unsigned int N,
                                         unsigned int iter,
                                         unsigned int radix_bits) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int key;
  unsigned int digit;

  const unsigned int MAX_RADIX_SIZE = 256; // 2^8, supporting up to 8-bit radix

  __shared__ unsigned int digit_counts[MAX_RADIX_SIZE];

  const unsigned int radix_size = 1 << radix_bits;

  if (threadIdx.x < radix_size) {
    digit_counts[threadIdx.x] = 0;
  }
  __syncthreads();

  if (i < N) {
    key = input[i];
    digit = (key >> (iter * radix_bits)) & (radix_size - 1);
    atomicAdd(&digit_counts[digit], 1);
  }
  __syncthreads();

  exclusiveScan(digit_counts, radix_size);

  if (i < N) {
    key = input[i];
    digit = (key >> (iter * radix_bits)) & (radix_size - 1);
    unsigned int pos = atomicAdd(&digit_counts[digit], 1);
    output[pos] = key;
  }
}