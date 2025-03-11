#!POPCORN leaderboard matmul
#!POPCORN gpus H100

# This is a submission template for popcorn leaderboard 'matmul'.
# Your task is as follows:
# > Implement a custom matmul function that matches the reference implementation.

# > The function should handle a tuple of input tensors and apply matmul

# > The shapes of all outer and inner dimensions of tensors are multiples of 16

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_fp16.h>
#include <math_functions.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__
void matmul_tile_coarse(half* __restrict__ C, const half* __restrict__ A, const half* __restrict__ B, unsigned int m, unsigned int k, unsigned int n) {
    __shared__ half A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ half B_s[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];

    unsigned int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    unsigned int colStart = blockIdx.x * TILE_WIDTH * COARSE_FACTOR;

    float sum[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }

    unsigned int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        unsigned int aCol = t * TILE_WIDTH + threadIdx.x;
        if (row < m && aCol < k)
            A_s[threadIdx.y][threadIdx.x] = A[row * k + aCol];
        else
            A_s[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        unsigned int bRow = t * TILE_WIDTH + threadIdx.y;
        for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
            unsigned int bCol = colStart + threadIdx.x + c * TILE_WIDTH;
            if (bRow < k && bCol < n)
                B_s[threadIdx.y][threadIdx.x + c * TILE_WIDTH] = B[bRow * n + bCol];
            else
                B_s[threadIdx.y][threadIdx.x + c * TILE_WIDTH] = __float2half(0.0f);
        }
        __syncthreads();

        for (unsigned int j = 0; j < TILE_WIDTH; j++) {
            float aVal = __half2float(A_s[threadIdx.y][j]);
            for (int c = 0; c < COARSE_FACTOR; c++) {
                float bVal = __half2float(B_s[j][threadIdx.x + c * TILE_WIDTH]);
                sum[c] = fmaf(aVal, bVal, sum[c]);
            }
        }
        __syncthreads();
    }
    
    for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
        unsigned int col = colStart + threadIdx.x + c * TILE_WIDTH;
        if (row < m && col < n)
            C[row * n + col] = __float2half(sum[c]);
    }
}

constexpr inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    unsigned int m = A.size(0);
    unsigned int k = A.size(1);
    unsigned int n = B.size(1);
    torch::Tensor C = torch::empty({m, n}, A.options());
    
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(cdiv(n, TILE_WIDTH * COARSE_FACTOR), cdiv(m, TILE_WIDTH));
    
    matmul_tile_coarse<<<grid, block>>>(
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        m, k, n
    );
    
    return C;
}
"""

cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B);"

matmul_extension = load_inline(
    name="matmul_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    A, B = data
    return A @ B
    # TODO: write working + faster kernel
    # return matmul_extension.matmul(A, B)
