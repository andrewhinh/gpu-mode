#!POPCORN leaderboard vectoradd
#!POPCORN gpus H100

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
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

// 16-byte vector type: 8 halfs = 16 bytes
struct __align__(16) half8 {
    __half x0, x1, x2, x3, x4, x5, x6, x7;
};

__device__ inline half8 half8_add(const half8 &a, const half8 &b) {
    half8 c;
    c.x0 = __hadd(a.x0, b.x0);
    c.x1 = __hadd(a.x1, b.x1);
    c.x2 = __hadd(a.x2, b.x2);
    c.x3 = __hadd(a.x3, b.x3);
    c.x4 = __hadd(a.x4, b.x4);
    c.x5 = __hadd(a.x5, b.x5);
    c.x6 = __hadd(a.x6, b.x6);
    c.x7 = __hadd(a.x7, b.x7);
    return c;
}

__global__
void vecAddCpAsyncKernel(__half* __restrict__ C,
                           const __half* __restrict__ A,
                           const __half* __restrict__ B,
                           unsigned int n) {
    // process data in half8 chunks (8 halfs per chunk)
    unsigned int n_half8 = n / 8;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid;

    extern __shared__ char smem[];
    half8* sharedA = reinterpret_cast<half8*>(smem);
    half8* sharedB = sharedA + blockDim.x;

    {
        unsigned int pos = start;
        if (pos < n_half8) {
            // cast pointers to 64-bit ints so that the inline asm sees a 64-bit operand
            unsigned long long addr_sharedA = reinterpret_cast<unsigned long long>(&sharedA[threadIdx.x]);
            unsigned long long addr_A = reinterpret_cast<unsigned long long>(&(reinterpret_cast<const half8*>(A))[pos]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                         :
                         : "l"(addr_sharedA),
                           "l"(addr_A),
                           "n"(sizeof(half8)));
                           
            unsigned long long addr_sharedB = reinterpret_cast<unsigned long long>(&sharedB[threadIdx.x]);
            unsigned long long addr_B = reinterpret_cast<unsigned long long>(&(reinterpret_cast<const half8*>(B))[pos]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                         :
                         : "l"(addr_sharedB),
                           "l"(addr_B),
                           "n"(sizeof(half8)));
        }
    }
    
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    
    {
        unsigned int pos = start;
        if (pos < n_half8) {
            half8 a_val = sharedA[threadIdx.x];
            half8 b_val = sharedB[threadIdx.x];
            half8 c_val = half8_add(a_val, b_val);
            reinterpret_cast<half8*>(C)[pos] = c_val;
        }
    }
    
    // leftover __half elements (when n % 8 != 0q), have thread 0 process them.
    unsigned int rem = n % 8;
    if (rem && tid == 0) {
        unsigned int start_idx = n - rem;
        for (unsigned int k = 0; k < rem; k++) {
            C[start_idx + k] = __hadd(A[start_idx + k], B[start_idx + k]);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor vector_add(torch::Tensor A, torch::Tensor B) {
    const unsigned int n = A.size(0) * A.size(1);
    torch::Tensor C = torch::empty_like(A);

    unsigned int n_half8 = n / 8;
    unsigned int totalThreads = n_half8;
    const unsigned int threads_per_block = 512;
    dim3 block(threads_per_block);
    dim3 grid(cdiv(totalThreads, threads_per_block));

    unsigned int sharedMemSize = block.x * sizeof(half8) * 2;

    vecAddCpAsyncKernel<<<grid, block, sharedMemSize>>>(
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
    return vector_add_extension.vector_add(A, B)
