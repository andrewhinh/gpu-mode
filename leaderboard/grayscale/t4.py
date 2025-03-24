#!POPCORN leaderboard grayscale
#!POPCORN gpus T4

# This is a submission template for popcorn leaderboard 'grayscale'.
# Your task is as follows:
# > Implement an RGB to grayscale conversion kernel that matches the reference implementation.

# > The kernel should convert square RGB images with even sizes to grayscale using the standard coefficients:

# > Y = 0.2989 R + 0.5870 G + 0.1140 B

# >

# > Input: RGB tensor of shape (H, W, 3) with values in [0, 1]

# > Output: Grayscale tensor of shape (H, W) with values in [0, 1]

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#define DEVICE_INLINE __device__ __forceinline__

// Using float4 as load_type (16 bytes)
using load_type = float4;
constexpr int warp_size = 32;

// Recursive template for loop unrolling
template<int I, int N, typename F>
__device__ __forceinline__
void unroll(F f) {
    if constexpr(I < N) {
        f.template operator()<I>();
        unroll<I + 1, N>(f);
    }
}

template<int block_size, int pixels_per_thread>
__global__ __launch_bounds__(block_size)
void grayscale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int height
) {
    static_assert(pixels_per_thread * sizeof(float) % sizeof(load_type) == 0);

    constexpr int pixels_per_block = pixels_per_thread * block_size;
    constexpr int inputs_per_thread = 3 * pixels_per_thread;

    constexpr int num_stores = pixels_per_thread * sizeof(float) / sizeof(load_type);
    constexpr int num_loads = num_stores * 3;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    const int block_offset = pixels_per_block * bid;
    const int warp_offset = warp * warp_size * pixels_per_thread;

    extern __shared__ float block_inputs[];

    const float* thread_input = input + (block_offset + warp_offset) * 3;
    float* thread_output = output + block_offset + warp_offset;

    float* warp_inputs = block_inputs + warp_offset * 3;

    // Load RGB data from global to shared memory
    // T4 doesn't support cp.async, so use __ldg to optimize loads
    {
        const auto* src = reinterpret_cast<const load_type*>(thread_input) + lane;
        auto* dst = reinterpret_cast<load_type*>(warp_inputs) + lane;

        #pragma unroll
        for (int i = 0; i < num_loads; ++i) {
            // Use __ldcs for cache optimized streaming loads
            dst[i * warp_size] = __ldcs(&src[i * warp_size]);
        }
    }

    __syncthreads();

    constexpr int K = sizeof(load_type) / sizeof(float);

    unroll<0, num_stores>([&]<int i>() {
        float gray[K];

        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float r = warp_inputs[((i * warp_size + lane) * K + j) * 3 + 0];
            const float g = warp_inputs[((i * warp_size + lane) * K + j) * 3 + 1];
            const float b = warp_inputs[((i * warp_size + lane) * K + j) * 3 + 2];

            float x = r * 0.2989f;
            x = fmaf(g, 0.5870f, x);
            x = fmaf(b, 0.1140f, x);
            gray[j] = x;
        }

        const auto* src = reinterpret_cast<const load_type*>(&gray);
        auto* dst = reinterpret_cast<load_type*>(thread_output) + lane;
        dst[i * warp_size] = *src;
    });
}

constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor grayscale(torch::Tensor image) {
    const unsigned int height = image.size(0);
    const unsigned int width = image.size(1);

    torch::Tensor result = torch::empty({height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(image.device()));

    constexpr int block_size = 256;
    constexpr int pixels_per_thread = 4;  // Must be a multiple of sizeof(load_type)/sizeof(float)
    constexpr int pixels_per_block = block_size * pixels_per_thread;
    
    const int num_blocks = cdiv(width * height, pixels_per_block);
    const size_t shared_mem_size = pixels_per_block * 3 * sizeof(float);

    grayscale_kernel<block_size, pixels_per_thread><<<num_blocks, block_size, shared_mem_size>>>(
        image.data_ptr<float>(),
        result.data_ptr<float>(),
        width, height
    );

    return result;
}
"""

cpp_source = "torch::Tensor grayscale(torch::Tensor image);"

grayscale_extension = load_inline(
    name="grayscale_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["grayscale"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    return grayscale_extension.grayscale(data)
