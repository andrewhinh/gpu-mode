#!POPCORN leaderboard prefixsum
#!POPCORN gpus H100

# This is a submission template for popcorn leaderboard 'prefixsum'.
# Your task is as follows:
# > Implement an inclusive prefix sum (scan) kernel that matches the reference implementation.

# > The kernel should compute the cumulative sum of all elements up to each position.

# > Because of numerical instability, the tolerance is scaled by the square root of the input size.

# >

# > Input:

# > - `data`: A 1D tensor of size `n`

# > Output:

# > - `output`: A 1D tensor of size `n`

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#define BLOCK_SIZE 256
#define SECTION_SIZE BLOCK_SIZE

constexpr inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__ void stream_scan_kernel(float* output, const float* input, unsigned int n, int* flags, float* scan_values, int* blockCounter) {
    __shared__ float temp[SECTION_SIZE];
    __shared__ unsigned int bid_s;
    __shared__ float previous_sum;
    
    unsigned int tid = threadIdx.x;
    
    // dynamic block index assignment to prevent deadlocks
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    
    unsigned int bid = bid_s;
    unsigned int global_idx = bid * blockDim.x + tid;
    
    temp[tid] = (global_idx < n) ? input[global_idx] : 0.0f;
    __syncthreads();
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = tid - stride;
        float val = (idx >= 0) ? temp[idx] : 0.0f;
        __syncthreads();
        if (idx >= 0) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    float local_sum = temp[blockDim.x - 1];
    
    // adjacent synchronization
    if (tid == 0) {
        if (bid == 0) {
            previous_sum = 0.0f;
            scan_values[bid] = local_sum;
            __threadfence(); // ensures that the write to scan_values is visible to other blocks
            atomicAdd(&flags[bid + 1], 1);
        } else {
            // wait for previous block's flag
            while (atomicAdd(&flags[bid], 0) == 0) { }
            previous_sum = scan_values[bid - 1];
            scan_values[bid] = previous_sum + local_sum;
            __threadfence();
            
            if (bid < gridDim.x - 1) {
                atomicAdd(&flags[bid + 1], 1);
            }
        }
    }
    __syncthreads();
    
    if (global_idx < n) {
        output[global_idx] = temp[tid];
        if (bid > 0) {
            output[global_idx] += previous_sum;
        }
    }
}

torch::Tensor prefixsum(torch::Tensor data) {
    const unsigned int n = data.size(0);
    torch::Tensor output = torch::empty_like(data);

    const unsigned int numBlocks = cdiv(n, BLOCK_SIZE);
    
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
    
    torch::Tensor flags = torch::zeros({numBlocks + 1}, options);
    torch::Tensor scan_values = torch::zeros({numBlocks}, float_options);
    torch::Tensor block_counter = torch::zeros({1}, options);
    
    flags[0].fill_(1);
    
    stream_scan_kernel<<<numBlocks, BLOCK_SIZE, 0>>>(
        output.data_ptr<float>(),
        data.data_ptr<float>(),
        n,
        flags.data_ptr<int>(),
        scan_values.data_ptr<float>(),
        block_counter.data_ptr<int>()
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor prefixsum(torch::Tensor data);"

prefixsum_extension = load_inline(
    name="prefixsum_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["prefixsum"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    return prefixsum_extension.prefixsum(data)
