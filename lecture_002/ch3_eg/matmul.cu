#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// 1a
__global__
void matmul_row_kernel(float* M, float* N, float* P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width) {
        float pVal = 0.0;
        for (int c = 0; c < width; ++c) {
            pVal = 0.0;
            for (int k = 0; k < width; ++k) {
                pVal += M[row * width + k] * N[k * width + c];
            }
            P[row * width + c] = pVal;
        }
    }
}

// 1b
__global__
void matmul_col_kernel(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width) {
        float pVal = 0.0;
        for (int r = 0; r < width; ++r) {
            pVal = 0.0;
            for (int k = 0; k < width; ++k) {
                pVal += M[r * width + k] * N[k * width + col];
            }
            P[r * width + col] = pVal;
        }
    }
}

// 1c
// Per-row pros: 
// - mem access for matrix M is more predictable
// Per-row cons:
// - mem access for matrix N is less predictable

// Per-column pros:
// - mem access for matrix N is more predictable
// Per-column cons:
// - mem access for matrix M is less predictable

// 2
__global__
void matmul_vec_kernel(float* M, float* V, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width) {
        float pVal = 0.0;
        for (int k = 0; k < width; ++k) {
            pVal += M[row * width + k] * V[k];
        }
        P[row * width + col] = pVal;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

void matmul_vec_stub(float* M, float* V, float* P, int width) {
    float *M_d, *V_d, *P_d;
    size_t size = width * sizeof(float);

    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&V_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(width, numThreads);

    matmul_vec_kernel<<<numBlocks, numThreads>>>(M_d, V_d, P_d, width);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(V_d);
    cudaFree(P_d);
}

// 3
// a. 16 * 32 = 512
// b. n_blocks = ((300 - 1) / 16 + 1) * ((150 - 1) / 32 + 1) = 111
//    n_threads = 111 * 512 = 56832
// c. 111
// d. 150 * 300 = 45000

// 4
// a. 20 * 400 + 10 = 8010
// b. 10 * 500 + 20 = 5020

// 5. 10 * 400 + 20 * 500 + 5 * 300 = 15500