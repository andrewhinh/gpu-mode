int counter = 0; // iteration counter
int C_length = C_next - C_curr;
int A_length = A_next - A_curr;
int B_length = B_next - B_curr;
int total_iteration = ceil((C_length) / tile_size); // total iteration
int C_completed = 0;
int A_consumed = 0;
int B_consumed = 0;

while (counter < total_iteration) {
  /* Calculate co-rank to determine elements needed for this iteration */
  int end_pos = min(C_completed + tile_size, C_length);
  int i_needed, j_needed;
  co_rank(end_pos, A, A_length - A_consumed, B, B_length - B_consumed,
          &i_needed, &j_needed);

  /* loading only necessary A elements into shared memory */
  for (int i = 0; i < tile_size; i += blockDim.x) {
    if (i + threadIdx.x < i_needed) {
      A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
    }
  }

  /* loading only necessary B elements into shared memory */
  for (int i = 0; i < tile_size; i += blockDim.x) {
    if (i + threadIdx.x < j_needed) {
      B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
    }
  }

  syncthreads();

  // Rest of the code remains the same
  // ...

  // Update consumed counts at the end of the iteration
  A_consumed += i_needed;
  B_consumed += j_needed;
  C_completed = end_pos;
  counter++;
}