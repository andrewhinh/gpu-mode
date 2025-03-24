#!POPCORN leaderboard vectoradd
#!POPCORN gpus A100

# This is a submission template for popcorn leaderboard 'vectoradd'.
# Your task is as follows:
# > Implement a float16 vector addition kernel.

# >

# > Input: tuple(torch.Tensor, torch.Tensor) with tensors of shape (N, N) and type torch.float16. These tensors are from

# > a normal distribution with mean 0 and variance 1.

# > Output: torch.Tensor of shape (N, N) and type torch.float16

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_fp16.h>

__global__
void vector_add_kernel(__half* __restrict__ C,
                                 const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 unsigned int n) {
    unsigned int n_half2 = n / 2;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_half2) {
        half2 a2 = reinterpret_cast<const half2*>(A)[tid];
        half2 b2 = reinterpret_cast<const half2*>(B)[tid];
        half2 c2 = __hadd2(a2, b2);
        reinterpret_cast<half2*>(C)[tid] = c2;
    }
    if ((n & 1) && tid == 0) {
        unsigned int last = n - 1;
        C[last] = __hadd(A[last], B[last]);
    }
}

constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor vector_add(torch::Tensor A, torch::Tensor B) {
    const unsigned int n = A.size(0) * A.size(1);
    torch::Tensor C = torch::empty_like(A);
    unsigned int n_half2 = n / 2;
    unsigned int totalThreads = n_half2;
    const unsigned int threads_per_block = 512;
    dim3 block(threads_per_block);
    dim3 grid(cdiv(totalThreads, threads_per_block));
    vector_add_kernel<<<grid, block>>>(
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        n
    );
    return C;
}
"""
cpp_source = "torch::Tensor vector_add(torch::Tensor A, torch::Tensor B);"

vector_add_extension = load_inline(
    name="vector_add_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_add"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    A, B = data
    return A + B
    # TODO: write faster kernel
    # return vector_add_extension.vector_add(A, B)
