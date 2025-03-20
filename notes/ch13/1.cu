__global__ void radix_sort_iter(unsigned int *input, unsigned int *output,
                                unsigned int *bits, unsigned int N,
                                unsigned int iter) {
  extern __shared__ unsigned int shared_data[];
  unsigned int *shared_keys = shared_data;
  unsigned int *shared_bits = &shared_data[blockDim.x];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + tid;
  unsigned int key = 0, bit = 0;

  if (i < N) {
    key = input[i];
    bit = (key >> iter) & 1;
    shared_keys[tid] = key;
    shared_bits[tid] = bit;
  }
  __syncthreads();

  exclusiveScan(bits, N);

  __syncthreads();

  if (i < N) {
    unsigned int numOnesBefore = bits[i];
    unsigned int numOnesTotal = bits[N - 1];
    unsigned int dst =
        (bit == 0) ? (i - numOnesBefore) : (numOnesTotal + i - numOnesBefore);
    output[dst] = shared_keys[tid];
  }
}