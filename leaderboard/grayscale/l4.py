#!POPCORN leaderboard grayscale
#!POPCORN gpus L4

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
__global__ 
void grayscale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int height
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        unsigned int pixel_idx = y * width + x;
        unsigned int idx = pixel_idx * 3;
        
        float r = __ldg(input + idx);
        float g = __ldg(input + idx + 1);
        float b = __ldg(input + idx + 2);
        
        float gray = fmaf(0.2989f, r, fmaf(0.5870f, g, 0.1140f * b));
        
        output[pixel_idx] = gray;
    }
}

constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor grayscale(torch::Tensor image) {
    const unsigned int height = image.size(0);
    const unsigned int width  = image.size(1);

    torch::Tensor result = torch::empty({height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(image.device()));

    dim3 block(32, 8);
    dim3 grid(
        cdiv(width, block.x),
        cdiv(height, block.y)
    );

    grayscale_kernel<<<grid, block>>>(
        image.data_ptr<float>(),
        result.data_ptr<float>(),
        width,
        height
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
    extra_cuda_cflags=["-O3"],
)


def custom_kernel(data: input_t) -> output_t:
    return grayscale_extension.grayscale(data)
