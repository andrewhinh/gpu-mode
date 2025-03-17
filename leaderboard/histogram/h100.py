#!POPCORN leaderboard histogram
#!POPCORN gpus H100

# This is a submission template for popcorn leaderboard 'histogram'.
# Your task is as follows:
# > Implement a histogram kernel that counts the number of elements falling into each bin across the specified range.

# > The minimum and maximum values of the range are fixed to 0 and 100 respectively.

# > All sizes are multiples of 16 and the number of bins is set to the size of the input tensor divided by 16.

# >

# > Input:

# >   - data: a tensor of shape (size,)

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#define NUM_BINS 256

extern "C" __global__ void histo_private_kernel(const unsigned char* __restrict__ data, const unsigned int length, unsigned int* __restrict__ histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    
    __syncthreads();
    
    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        unsigned char alphabet_position = data[i];
        if(alphabet_position < NUM_BINS) {
            unsigned int bin = alphabet_position;
            if(bin == prevBinIdx) {
                ++accumulator;
            } else {
                if(accumulator > 0) {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    
    if(accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }
    
    __syncthreads();
    
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        if(histo_s[bin] > 0) {
            atomicAdd(&(histo[bin]), histo_s[bin]);
        }
    }
}

torch::Tensor histogram(torch::Tensor input) {
    torch::Tensor uint_output = torch::zeros({256}, torch::kUInt32).cuda();
    
    unsigned const char* __restrict__ data_ptr = input.data_ptr<unsigned char>();
    unsigned int* __restrict__ output_ptr = uint_output.data_ptr<unsigned int>();
    
    const unsigned int blockSize = 256;
    const unsigned int maxBlocks = 114 * 32; // H100 has 114 SMs
    const unsigned int numElements = input.numel();
    const unsigned int numBlocks = min((numElements + blockSize - 1) / blockSize, maxBlocks);
    
    histo_private_kernel<<<numBlocks, blockSize>>>(
        data_ptr,
        numElements,
        output_ptr
    );
    
    torch::Tensor output = uint_output.to(torch::kInt64);
    return output;
}
"""

cpp_source = "torch::Tensor histogram(torch::Tensor input);"

histogram_extension = load_inline(
    name="histogram_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["histogram"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    return histogram_extension.histogram(data)
