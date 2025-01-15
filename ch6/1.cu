#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void matmul_vec_kernel(float* A, float* B, float* C, int width, int tileWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__
    float A_s[tileWidth][tileWidth];
    __shared__
    float B_s[tileWidth][tileWidth];

    for (int k = 0; k < width / tileWidth; ++k) {
        A_s[threadIdx.y][threadIdx.x] = A[row * width + (k * tileWidth + threadIdx.x)];
        B_s[threadIdx.x][threadIdx.y] = B[(k * tileWidth + threadIdx.y) * width + col];
        __syncthreads();

        if (col < width && row < width) {
            float pVal = 0.0;
            for (int k = 0; k < tileWidth; ++k) {
                pVal += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
            }
            C[row * width + col] += pVal;
        }
        __syncthreads();
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

    matmul_vec_kernel<<<numBlocks, numThreads>>>(M_d, V_d, P_d, width, 16);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(V_d);
    cudaFree(P_d);
}
