#!POPCORN leaderboard sort
#!POPCORN gpus L4

# This is a submission template for popcorn leaderboard 'sort'.
# Your task is as follows:
# > Implement a sort kernel that matches the reference implementation.

# > The kernel should sort the input array in ascending order using a sort algorithm of your choice.

# >

# > Input arrays are generated as random floating-point numbers, where each row of a roughly square matrix

# > is drawn from a normal distribution with a different mean value per row based on the seed and then flattened into a 1D array.

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>

// Even-odd transposition sort kernel for small arrays
__global__ void sort_small_kernel(float* data, int n, int phase) {
    // Calculate global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread compares and potentially swaps adjacent elements
    // Based on even-odd transposition sort (parallel bubble sort)
    int i = 2 * tid + (phase & 1); // Alternate between even and odd indices
    
    if (i < n - 1) {
        if (data[i] > data[i + 1]) {
            // Swap elements
            float temp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = temp;
        }
    }
}

// Bitonic sort kernel for more efficiency with larger arrays
__global__ void bitonic_sort_kernel(float* data, int n, int j, int k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate the indices for comparison
    int ixj = i ^ j;
    
    // Check bounds and if the thread should do work
    if (ixj > i && i < n && ixj < n) {
        // Check direction of sort (increasing or decreasing)
        if ((i & k) == 0) {
            // Increasing
            if (data[i] > data[ixj]) {
                // Swap
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Decreasing
            if (data[i] < data[ixj]) {
                // Swap
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

torch::Tensor sort(torch::Tensor data) {
    if (!data.is_cuda()) {
        throw std::runtime_error("Input tensor must be a CUDA tensor");
    }
    if (data.scalar_type() != torch::kFloat) {
        throw std::runtime_error("Input tensor must be of type float32");
    }
    if (data.dim() != 1) {
        throw std::runtime_error("Input tensor must be 1-dimensional");
    }

    int n = data.numel();
    
    // Create a copy of the input tensor for our sorting
    torch::Tensor output = data.clone();
    float* d_output = output.data_ptr<float>();
    
    // Use the simpler sort for very small arrays
    if (n < 1024) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        // For even-odd sort, we need n phases to ensure the array is sorted
        for (int phase = 0; phase < n; phase++) {
            sort_small_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, n, phase);
            cudaDeviceSynchronize();
        }
    } 
    else {
        // For larger arrays, use bitonic sort
        // Pad the array to the next power of 2 for bitonic sort
        int size = 1;
        while (size < n) size *= 2;
        
        // Create a padded tensor with high values for the padding
        torch::Tensor padded = torch::full({size}, INFINITY, 
                               torch::TensorOptions()
                               .dtype(torch::kFloat)
                               .device(data.device()));
                               
        // Copy the original data to the beginning of the padded tensor
        padded.slice(0, 0, n).copy_(output);
        
        // Get pointer to the padded data
        float* d_padded = padded.data_ptr<float>();
        
        // Set thread and block configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Perform bitonic sort
        for (int k = 2; k <= size; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                bitonic_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_padded, size, j, k);
                cudaDeviceSynchronize();
            }
        }
        
        // Copy the sorted data back to the output tensor (only the original size)
        output = padded.slice(0, 0, n).clone();
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor sort(torch::Tensor data);
"""

sort_extension = load_inline(
    name="sort_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sort"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)


def custom_kernel(data: input_t) -> output_t:
    return sort_extension.sort(data)
