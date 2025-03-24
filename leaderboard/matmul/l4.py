#!POPCORN leaderboard matmul
#!POPCORN gpus L4

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
#include <cublas_v2.h>
#include <cuda_fp16.h>

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::empty({M, N}, 
                        torch::TensorOptions()
                           .dtype(A.scalar_type())
                           .device(A.device()));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode to match PyTorch's default
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    // We need to adapt for row-major vs column-major
    // cuBLAS uses column-major format
    // Computing B * A with cuBLAS gives the same result as A * B in row-major
    
    // This is critical for matching PyTorch's precision - compute in FP32
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Call cuBLAS for half precision matmul
    // Using the version that computes in FP32 but stores in FP16
    cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr(), CUDA_R_16F, N,
                A.data_ptr(), CUDA_R_16F, K,
                &beta,
                C.data_ptr(), CUDA_R_16F, N,
                CUDA_R_32F,      // Compute in FP32 precision
                CUBLAS_GEMM_DEFAULT);
    
    cublasDestroy(handle);
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
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_include_paths=["/usr/local/cuda/include"],
    extra_ldflags=["-lcublas"],
)


def custom_kernel(data: input_t) -> output_t:
    A, B = data
    return matmul_extension.matmul(A, B)
