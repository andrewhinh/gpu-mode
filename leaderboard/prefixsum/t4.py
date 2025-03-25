#!POPCORN leaderboard prefixsum
#!POPCORN gpus T4

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
#include <vector>
#include <algorithm>
#include <numeric>

// A small utility: ceiling-division
constexpr inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__forceinline__ __device__ float warpInclusiveScan(float val) {
    unsigned mask = 0xffffffff; // for full warp
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(mask, val, offset, 32);
        if ((threadIdx.x & 31) >= offset) {
            val += y;
        }
    }
    return val;
}

__device__ void blockInclusiveScan(float* sdata, float &val) {
    const int warpId = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;

    __shared__ float warpSums[BLOCK_SIZE / 32];

    float prefix = warpInclusiveScan(val); // warp-level prefix sum

    if (lane == 31) {
        warpSums[warpId] = prefix;
    }
    __syncthreads();

    float blockOffset = 0;
    if (warpId == 0) {
        float warpTotal  = warpSums[lane];
        float warpPrefix = warpInclusiveScan(warpTotal);
        warpSums[lane]   = warpPrefix;
    }
    __syncthreads();

    if (warpId > 0) {
        blockOffset = warpSums[warpId - 1];
    }

    prefix += blockOffset;
    val = prefix;

    sdata[threadIdx.x] = val;
}

// Optimized single-block scan
__global__ void single_block_scan_kernel(
    float* output,
    const float* input,
    unsigned int n
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    
    float myVal = 0.0f;
    if (tid < n) {
        myVal = input[tid];
    }
    
    blockInclusiveScan(sdata, myVal);
    
    if (tid < n) {
        output[tid] = myVal;
    }
}

__global__ void block_scan_kernel(
    float*       output,
    float*       block_sums,
    const float* input,
    unsigned int n
) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid       = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + tid;

    float myVal = 0.0f;
    if (globalIdx < n) {
        myVal = input[globalIdx];
    }

    blockInclusiveScan(sdata, myVal);

    if (globalIdx < n) {
        output[globalIdx] = myVal;
    }

    if (tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = myVal;
    }
}

// Optimized parallel block sums scan that can handle block_sums efficiently
__global__ void block_sums_scan_kernel(
    float* block_sums,
    unsigned int numBlocks
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    
    float myVal = 0.0f;
    if (tid < numBlocks) {
        myVal = block_sums[tid];
    }
    
    blockInclusiveScan(sdata, myVal);
    
    if (tid < numBlocks) {
        block_sums[tid] = myVal;
    }
}

__global__ void add_block_sums(
    float*       output,
    const float* block_sums,
    unsigned int n
) {
    unsigned int tid       = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + tid;

    if (blockIdx.x == 0) {
        // first block has no offset
        return;
    }
    
    float offset = block_sums[blockIdx.x - 1];
    if (globalIdx < n) {
        output[globalIdx] += offset;
    }
}

torch::Tensor prefixsum(torch::Tensor data) {
    const unsigned int n = data.size(0);
    
    // Special case for empty or single element input
    if (n <= 1) {
        return data.clone();
    }
    
    torch::Tensor output = torch::empty_like(data);

    const unsigned int numBlocks = cdiv(n, BLOCK_SIZE);
    
    // Special case for small arrays that fit in a single block
    if (numBlocks == 1) {
        single_block_scan_kernel<<<1, BLOCK_SIZE>>>(
            output.data_ptr<float>(),
            data.data_ptr<float>(),
            n
        );
        return output;
    }

    // For larger arrays, use a parallel approach
    torch::Tensor block_sums = torch::zeros({(long)numBlocks}, data.options());
    
    // Step 1: Scan individual blocks and collect block sums
    block_scan_kernel<<<numBlocks, BLOCK_SIZE>>>(
        output.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        data.data_ptr<float>(),
        n
    );
    
    // Step 2: Scan the block sums using an optimized approach
    if (numBlocks <= BLOCK_SIZE) {
        // If block_sums fit in a single block, use the efficient single block kernel
        block_sums_scan_kernel<<<1, BLOCK_SIZE>>>(
            block_sums.data_ptr<float>(),
            numBlocks
        );
    } else {
        // For larger block_sums, use a recursive approach
        torch::Tensor scanned_block_sums = prefixsum(block_sums);
        block_sums = scanned_block_sums;  // Use the scanned result
    }
    
    // Step 3: Add the scanned block sums back to the output
    add_block_sums<<<numBlocks, BLOCK_SIZE>>>(
        output.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        n
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
