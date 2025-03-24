#!POPCORN leaderboard vectorsum
#!POPCORN gpus H100

# This is a submission template for popcorn leaderboard 'vectorsum'.
# Your task is as follows:
# > Implement a vector sum reduction kernel. This kernel computes the sum of all elements in the input tensor.

# >

# > Input: A tensor of shape `(N,)` with values from a normal distribution with mean 0 and variance 1.

# > Output: A scalar value equal to the sum of all elements in the input tensor.

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void cp_async_f32(float* dst, const float* src) {
    uint32_t dst_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                  :
                  : "r"(dst_addr), "l"(src), "n"(16)
                  : "memory");
}

extern __shared__ float smem_h100[];
__global__ void vector_sum_kernel(const float* __restrict__ input,
                                               float* __restrict__ workspace,
                                               int n) {
    const int tileSize = blockDim.x * 8;
    int globalTileStart = blockIdx.x * tileSize;
    float partialSum = 0.0f;
    int tid = threadIdx.x;
    
    for (int offset = globalTileStart; offset < n; offset += gridDim.x * tileSize) {
        int idx0 = offset + tid * 8;
        float* dst = smem_h100 + tid * 8;
        const float* src = input + idx0;
        if (idx0 + 7 < n) {
            cp_async_f32(dst, src);
            cp_async_f32(dst + 4, src + 4);
        } else {
            for (int j = 0; j < 8; j++) {
                int pos = idx0 + j;
                dst[j] = (pos < n) ? input[pos] : 0.0f;
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            partialSum += smem_h100[tid * 8 + j];
        }
        __syncthreads();
    }
    
    partialSum = warpReduceSum(partialSum);
    
    __shared__ float shared[32];
    int lane = tid & (warpSize - 1);
    int warpId = tid >> 5;
    if (lane == 0)
        shared[warpId] = partialSum;
    __syncthreads();
    int numWarps = blockDim.x / warpSize;
    float blockSum = (tid < numWarps) ? shared[lane] : 0.0f;
    if (tid < numWarps)
        blockSum = warpReduceSum(blockSum);
    if (tid == 0)
        workspace[blockIdx.x] = blockSum;
}

__global__ void reduce_kernel(const float* __restrict__ workspace,
                                     float* __restrict__ output,
                                     int numElements) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < numElements; i += blockDim.x)
         sum += workspace[i];
    sum = warpReduceSum(sum);
    __shared__ float shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;
    if (lane == 0)
         shared[warpId] = sum;
    __syncthreads();
    int numWarps = blockDim.x / warpSize;
    sum = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (threadIdx.x < numWarps)
         sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
         *output = sum;
}

torch::Tensor vector_sum(torch::Tensor image) {
    int n = image.numel();
    torch::Tensor result = torch::zeros({}, image.options().dtype(torch::kFloat));
    
    int blockSize = 1024;
    int gridSize = cdiv(n, blockSize * 8);
    if (gridSize < 1) gridSize = 1;
    
    torch::Tensor workspace_tensor = torch::empty({gridSize}, image.options().dtype(torch::kFloat));
    
    dim3 block(blockSize);
    dim3 grid(gridSize);
    size_t sharedMemSize = blockSize * 8 * sizeof(float);
    
    vector_sum_kernel<<<grid, block, sharedMemSize>>>(
         image.data_ptr<float>(),
         workspace_tensor.data_ptr<float>(),
         n
    );
    
    dim3 block2(1024);
    dim3 grid2(1);
    reduce_kernel<<<grid2, block2>>>(
         workspace_tensor.data_ptr<float>(),
         result.data_ptr<float>(),
         gridSize
    );
    
    return result;
}
"""
cpp_source = "torch::Tensor vector_sum(torch::Tensor image);"

vector_sum_extension = load_inline(
    name="vectorsum_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["vector_sum"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    return vector_sum_extension.vector_sum(data)
