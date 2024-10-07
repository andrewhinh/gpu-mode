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