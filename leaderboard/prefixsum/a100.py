#!POPCORN leaderboard prefixsum
#!POPCORN gpus A100

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
// A100 optimized implementation
#define BLOCK_SIZE 512  // Increased from 256 to 512 for A100
#define THREADS_PER_BLOCK BLOCK_SIZE
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define MAX_BLOCKS_MULTIPLIER 24  // Increased from 16 to 24 for A100 (108 SMs)

constexpr inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// Helper function to get next power of 2
__device__ unsigned int nextPowerOfTwo(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// Work-efficient scan within a single block with bank conflict avoidance
__global__ void prescan_block(float* output, const float* input, unsigned int n, float* blockSums) {
    extern __shared__ float temp[];
    
    unsigned int threadId = threadIdx.x;
    unsigned int blockId = blockIdx.x;
    unsigned int blockOffset = blockId * ELEMENTS_PER_BLOCK;
    
    // Load input into shared memory
    unsigned int ai = threadId;
    unsigned int bi = threadId + (ELEMENTS_PER_BLOCK / 2);
    unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    if (blockOffset + ai < n) {
        temp[ai + bankOffsetA] = input[blockOffset + ai];
    } else {
        temp[ai + bankOffsetA] = 0.0f;
    }
    
    if (blockOffset + bi < n) {
        temp[bi + bankOffsetB] = input[blockOffset + bi];
    } else {
        temp[bi + bankOffsetB] = 0.0f;
    }
    
    // Build sum in place up the tree (reduction phase)
    unsigned int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            unsigned int ai = offset * (2 * threadId + 1) - 1;
            unsigned int bi = offset * (2 * threadId + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element and store block sum
    if (threadId == 0) {
        unsigned int lastIndex = ELEMENTS_PER_BLOCK - 1 + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK - 1);
        if (blockSums != NULL) {
            blockSums[blockId] = temp[lastIndex];
        }
        temp[lastIndex] = 0.0f;  // Clear last element for exclusive scan
    }
    
    // Traverse down the tree and build scan (down-sweep phase)
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            unsigned int ai = offset * (2 * threadId + 1) - 1;
            unsigned int bi = offset * (2 * threadId + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results to global memory
    if (blockOffset + ai < n) {
        output[blockOffset + ai] = temp[ai + bankOffsetA];
    }
    
    if (blockOffset + bi < n) {
        output[blockOffset + bi] = temp[bi + bankOffsetB];
    }
}

// Scan block sums
__global__ void scan_block_sums(float* blockSums, unsigned int numBlocks) {
    extern __shared__ float temp[];
    unsigned int threadId = threadIdx.x;
    
    // Load input into shared memory
    if (threadId < numBlocks) {
        temp[threadId] = blockSums[threadId];
    } else {
        temp[threadId] = 0.0f;
    }
    
    // Zero out the rest of shared memory
    unsigned int powerOfTwo = nextPowerOfTwo(numBlocks);
    if (threadId + THREADS_PER_BLOCK < powerOfTwo && threadId + THREADS_PER_BLOCK >= numBlocks) {
        temp[threadId + THREADS_PER_BLOCK] = 0.0f;
    }
    __syncthreads();
    
    // Build sum in place up the tree
    unsigned int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            unsigned int ai = offset * (2 * threadId + 1) - 1;
            unsigned int bi = offset * (2 * threadId + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (threadId == 0) {
        temp[powerOfTwo - 1] = 0.0f;
    }
    
    // Traverse down the tree and build scan
    for (int d = 1; d < powerOfTwo; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            unsigned int ai = offset * (2 * threadId + 1) - 1;
            unsigned int bi = offset * (2 * threadId + 2) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results to block sums array
    if (threadId < numBlocks) {
        blockSums[threadId] = temp[threadId];
    }
}

// Add block sums to each element in the blocks
__global__ void add_block_sums(float* output, const float* blockSums, unsigned int n) {
    unsigned int threadId = threadIdx.x;
    unsigned int blockId = blockIdx.x;
    unsigned int blockOffset = blockId * ELEMENTS_PER_BLOCK;
    
    // Skip the first block as it doesn't need correction
    if (blockId > 0) {
        float sum = blockSums[blockId - 1];
        
        // Add sum to all elements in this block
        if (blockOffset + threadId < n) {
            output[blockOffset + threadId] += sum;
        }
        if (blockOffset + threadId + THREADS_PER_BLOCK < n) {
            output[blockOffset + threadId + THREADS_PER_BLOCK] += sum;
        }
    }
}

// Convert exclusive scan to inclusive scan
__global__ void exclusive_to_inclusive(float* output, const float* exclusive_scan, const float* original_input, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx == 0) {
            // For first element, inclusive scan is just the first input element
            output[idx] = original_input[idx];
        } else {
            // For all other elements, the inclusive scan is the exclusive scan value
            // plus the original input element at current position
            output[idx] = exclusive_scan[idx] + original_input[idx];
        }
    }
}

torch::Tensor prefixsum(torch::Tensor data) {
    const unsigned int n = data.size(0);
    torch::Tensor output = torch::empty_like(data);
    torch::Tensor temp = torch::empty_like(data);  // Temporary buffer for exclusive scan
    
    // Calculate grid dimensions
    const unsigned int numBlocks = cdiv(n, ELEMENTS_PER_BLOCK);
    const unsigned int sharedMemSize = ELEMENTS_PER_BLOCK * sizeof(float) + 
                                      ELEMENTS_PER_BLOCK * sizeof(float) / 16;  // Extra space for bank conflict avoidance
    
    // Handle special case for small arrays
    if (n <= ELEMENTS_PER_BLOCK) {
        // Allocate block sums array (only need one element for small array)
        torch::Tensor blockSums = torch::zeros({1}, data.options());
        
        // Launch the scan kernel for a single block
        prescan_block<<<1, THREADS_PER_BLOCK, sharedMemSize>>>(
            temp.data_ptr<float>(),
            data.data_ptr<float>(),
            n,
            blockSums.data_ptr<float>()
        );
        
        // Convert exclusive scan to inclusive scan
        exclusive_to_inclusive<<<cdiv(n, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
            output.data_ptr<float>(),
            temp.data_ptr<float>(),
            data.data_ptr<float>(),
            n
        );
        
        return output;
    }
    
    // For large arrays, need multi-block scan
    
    // Allocate block sums array
    torch::Tensor blockSums = torch::zeros({numBlocks}, data.options());
    
    // Step 1: Scan each block and collect block sums
    // Use maxBlocks multiplier to define maximum number of concurrent blocks
    const unsigned int maxConcurrentBlocks = MAX_BLOCKS_MULTIPLIER * 108; // 108 SMs on A100
    const unsigned int blocksToUse = min((unsigned int)numBlocks, maxConcurrentBlocks);
    
    prescan_block<<<blocksToUse, THREADS_PER_BLOCK, sharedMemSize>>>(
        temp.data_ptr<float>(),
        data.data_ptr<float>(),
        n,
        blockSums.data_ptr<float>()
    );
    
    // Step 2: Scan the block sums
    // Determine shared memory size for block sums scan
    int blockSumsSharedMemSize = nextPowerOfTwo(numBlocks) * sizeof(float) * 2;
    scan_block_sums<<<1, min(numBlocks, THREADS_PER_BLOCK), blockSumsSharedMemSize>>>(
        blockSums.data_ptr<float>(),
        numBlocks
    );
    
    // Step 3: Add block sums to each element
    add_block_sums<<<blocksToUse, THREADS_PER_BLOCK>>>(
        temp.data_ptr<float>(),
        blockSums.data_ptr<float>(),
        n
    );
    
    // Convert exclusive scan to inclusive scan
    exclusive_to_inclusive<<<cdiv(n, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
        output.data_ptr<float>(),
        temp.data_ptr<float>(),
        data.data_ptr<float>(),
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
