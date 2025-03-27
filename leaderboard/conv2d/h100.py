#!POPCORN leaderboard conv2d
#!POPCORN gpus H100

# This is a submission template for popcorn leaderboard 'conv2d'.
# Your task is as follows:
# > Implement a 2D convolution kernel that matches the reference implementation.

# > The kernel should perform 2D convolution with the given specifications

# > We will benchmark different sizes, kernel sizes, channels and batch sizes but they will all

# > be even numbers with the exception of batch size which can sometimes be 1

# > We assume no padding and striding and instead vary the size of the input and kernel,

# > number of channels, and batch size.

# >

# > Input: Tuple of (input_tensor, kernel)

# >   - input_tensor: 4D tensor of shape (batch, channels, height, width) with arbitrary values

# >   - kernel: 4D tensor of shape (channels, channels, kernelsize, kernelsize) with arbitrary values

# > Output: 4D tensor of shape (batch, channels, height-kernelsize+1, width-kernelsize+1) with convolved values

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cublas_v2.h>
#include <cuda_fp16.h>

constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// More efficient im2col kernel with memory optimization
__global__ void im2col_kernel(
    const float* input,
    float* col,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int kernel_size,
    const int out_h,
    const int out_w,
    const int blocks_per_batch
) {
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= blocks_per_batch * batch_size) return;

    // Determine which batch and output position this thread is responsible for
    const int b = thread_idx / blocks_per_batch;
    const int local_idx = thread_idx % blocks_per_batch;
    
    // Calculate output position in this batch
    const int out_pos = local_idx;
    const int out_w_idx = out_pos % out_w;
    const int out_h_idx = (out_pos / out_w) % out_h;
    
    // Extract each patch element for this output position
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Calculate corresponding input position
                const int in_h_idx = out_h_idx + kh;
                const int in_w_idx = out_w_idx + kw;
                
                // Input tensor index
                const int in_idx = ((b * in_channels + in_c) * in_h + in_h_idx) * in_w + in_w_idx;
                
                // Column index for GEMM:
                // Arrange for col matrix to be (in_channels * kernel_size * kernel_size, batch_size * out_h * out_w)
                const int col_idx = (in_c * kernel_size * kernel_size + kh * kernel_size + kw) * (batch_size * out_h * out_w) + 
                                    (b * out_h * out_w + out_h_idx * out_w + out_w_idx);
                
                // Use exact copy of input values without any computation
                col[col_idx] = input[in_idx];
            }
        }
    }
}

// Direct convolution for small inputs to avoid memory overhead
__global__ void direct_conv_kernel(
    const float* input,
    const float* kernel,
    float* output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int kernel_size,
    const int out_h,
    const int out_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w) return;
    
    // Calculate output indices
    const int out_w_idx = idx % out_w;
    const int out_h_idx = (idx / out_w) % out_h;
    const int out_c = (idx / (out_h * out_w)) % out_channels;
    const int b = idx / (out_channels * out_h * out_w);
    
    // Use double for accumulation to improve precision
    double sum = 0.0;
    
    // Perform convolution manually
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Input position
                const int in_h_idx = out_h_idx + kh;
                const int in_w_idx = out_w_idx + kw;
                
                // Get input value
                const int in_idx = ((b * in_channels + in_c) * in_h + in_h_idx) * in_w + in_w_idx;
                // Get filter value
                const int filter_idx = (((out_c * in_channels) + in_c) * kernel_size + kh) * kernel_size + kw;
                
                // Multiply and accumulate with higher precision
                sum += (double)input[in_idx] * (double)kernel[filter_idx];
            }
        }
    }
    
    // Write accumulated value to output
    const int out_idx = ((b * out_channels + out_c) * out_h + out_h_idx) * out_w + out_w_idx;
    output[out_idx] = (float)sum;
}

torch::Tensor conv2d(torch::Tensor input, torch::Tensor kernel) {
    // Ensure tensors are contiguous
    input = input.contiguous();
    kernel = kernel.contiguous();
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = kernel.size(0);
    const int kernel_size = kernel.size(2);
    const int out_h = in_h - kernel_size + 1;
    const int out_w = in_w - kernel_size + 1;
    
    // Create output tensor
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    // Size heuristic: Use direct convolution for small inputs to save memory
    const long long mem_needed = (long long)batch_size * out_h * out_w * in_channels * kernel_size * kernel_size * sizeof(float);
    const long long threshold = 1000 * 1024 * 1024; // 1GB threshold
    
    if (mem_needed < threshold) {
        // Heuristic for small inputs: use cuBLAS for matrix multiplication
        
        // Step 1: Convert input to columns (im2col)
        torch::Tensor col = torch::zeros({in_channels * kernel_size * kernel_size, batch_size * out_h * out_w}, input.options());
        
        // Launch kernel efficiently by processing one output element per thread
        const int blocks_per_batch = out_h * out_w;
        const int total_blocks = batch_size * blocks_per_batch;
        const int block_size = 256;
        const int grid_size = cdiv(total_blocks, block_size);
        
        im2col_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            col.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            kernel_size,
            out_h,
            out_w,
            blocks_per_batch
        );
        
        // Step 2: Reshape kernel for matrix multiplication
        // [out_channels, in_channels, kernel_size, kernel_size] -> [out_channels, in_channels * kernel_size * kernel_size]
        torch::Tensor kernel_mat = kernel.reshape({out_channels, in_channels * kernel_size * kernel_size});
        
        // Step 3: Perform matrix multiplication using cuBLAS
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);
        // Use PEDANTIC_MATH for higher precision that matches CPU reference implementation
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
        
        // Also disable TF32 explicitly (Ada Lovelace default behavior)
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        #if CUDA_VERSION >= 11000
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // If running on Ampere or newer, disable TF32
        if (prop.major >= 8) {
            cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }
        #endif
        
        // We're calculating output = kernel_mat * col which results in a matrix of shape 
        // [out_channels, batch_size * out_h * out_w]
        // This gets correctly reshaped to [batch_size, out_channels, out_h, out_w]
        torch::Tensor result = torch::zeros({out_channels, batch_size * out_h * out_w}, input.options());
        
        // Perform GEMM for convolution with correct column-major cuBLAS ordering
        // Use CUBLAS_STATUS_SUCCESS check to ensure computation completes correctly
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N,                           // No transpose for B
            CUBLAS_OP_N,                           // No transpose for A
            batch_size * out_h * out_w,            // M: number of columns in result
            out_channels,                          // N: number of rows in result
            in_channels * kernel_size * kernel_size, // K: inner dimension for multiplication
            &alpha,
            col.data_ptr<float>(),                 // B: col
            batch_size * out_h * out_w,            // ldb: leading dimension of B (col)
            kernel_mat.data_ptr<float>(),          // A: kernel_mat
            in_channels * kernel_size * kernel_size, // lda: leading dimension of A (kernel_mat)
            &beta,
            result.data_ptr<float>(),              // C: result
            batch_size * out_h * out_w             // ldc: leading dimension of C (result)
        );
        
        // Check for CUBLAS errors
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS error: %d\n", status);
        }
        
        // Clean up cuBLAS handle
        cublasDestroy(handle);
        
        // Reshape result to output tensor of shape [batch_size, out_channels, out_h, out_w]
        // We need to be careful about the batch dimension here
        // First reshape to [out_channels, batch_size, out_h, out_w]
        auto reshaped = result.reshape({out_channels, batch_size, out_h, out_w});
        // Then permute to get [batch_size, out_channels, out_h, out_w]
        output.copy_(reshaped.permute({1, 0, 2, 3}));
    }
    else {
        // For larger inputs, use direct convolution to avoid OOM
        const int total_threads = batch_size * out_channels * out_h * out_w;
        const int block_size = 512;
        const int grid_size = cdiv(total_threads, block_size);
        
        direct_conv_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            kernel.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_size,
            out_h,
            out_w
        );
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor conv2d(torch::Tensor input, torch::Tensor kernel);
"""

conv2d_extension = load_inline(
    name="conv2d_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d"],
    with_cuda=True,
    extra_cuda_cflags=[
        "-O3",
        # Removed "--use_fast_math" which causes precision differences
        "-gencode=arch=compute_89,code=sm_89",  # For L4
        "--fmad=false",  # Disable fused multiply-add for consistent results
        "--prec-div=true",  # Force precise division
        "--prec-sqrt=true",  # Force precise square roots
    ],
    extra_include_paths=["/usr/local/cuda/include"],
    extra_ldflags=["-lcublas"],
)


def custom_kernel(data: input_t) -> output_t:
    input, kernel = data
    return conv2d_extension.conv2d(input, kernel)
