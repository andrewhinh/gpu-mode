#!POPCORN leaderboard vectorsum
#!POPCORN gpus T4

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
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void vector_sum_kernel(const float* __restrict__ input,
                                                   float* __restrict__ workspace,
                                                   int n) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    int num_float4 = n / 4;
    const float4* __restrict__ input4 = reinterpret_cast<const float4*>(input);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int end = num_float4;
    
    int unroll_limit = end - (end % (4 * stride));
    int i = tid;
    #pragma unroll
    for (; i < unroll_limit; i += 4 * stride) {
        float4 a = input4[i];
        float4 b = input4[i + stride];
        float4 c = input4[i + 2 * stride];
        float4 d = input4[i + 3 * stride];
        sum0 += a.x + a.y + a.z + a.w;
        sum1 += b.x + b.y + b.z + b.w;
        sum2 += c.x + c.y + c.z + c.w;
        sum3 += d.x + d.y + d.z + d.w;
    }
    for (; i < end; i += stride) {
        float4 a = input4[i];
        sum0 += a.x + a.y + a.z + a.w;
    }
    float blockSum = sum0 + sum1 + sum2 + sum3;
    
    blockSum = warpReduceSum(blockSum);
    
    __shared__ float shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;
    if (lane == 0)
        shared[warpId] = blockSum;
    __syncthreads();
    
    int numWarps = blockDim.x / warpSize;
    float sum = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (threadIdx.x < numWarps)
        sum = warpReduceSum(sum);
    
    if (threadIdx.x == 0)
        workspace[blockIdx.x] = sum;
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int tail = n & 3;
        float tailSum = 0.0f;
        for (int j = n - tail; j < n; j++) {
            tailSum += input[j];
        }
        workspace[0] += tailSum;
    }
}

__global__ void reduce_kernel(const float* __restrict__ workspace,
                                              float* __restrict__ output,
                                              int numElements) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < numElements; i += blockDim.x) {
        sum += workspace[i];
    }
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

constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor vector_sum(torch::Tensor image) {
    int n = image.numel();
    torch::Tensor result = torch::zeros({}, image.options().dtype(torch::kFloat));
    
    int blockSize = 256;
    int gridSize = cdiv(n, blockSize * 16);
    if (gridSize < 1) gridSize = 1;
    
    torch::Tensor workspace_tensor = torch::empty({gridSize}, image.options().dtype(torch::kFloat));
    
    dim3 block(blockSize);
    dim3 grid(gridSize);
    
    vector_sum_kernel<<<grid, block>>>(
        image.data_ptr<float>(),
        workspace_tensor.data_ptr<float>(),
        n
    );
    
    dim3 block2(256);
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
