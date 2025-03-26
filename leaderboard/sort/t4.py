#!POPCORN leaderboard sort
#!POPCORN gpus T4

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
#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 4096 // 16KB available on T4 per block
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_MASK ((1 << RADIX_BITS) - 1)

// Float to uint conversion that preserves ordering for radix sort
__device__ unsigned int float_to_uint(float f) {
    unsigned int ui = __float_as_uint(f);
    // Special handling for NaN and negative values
    if ((ui & 0x7F800000) == 0x7F800000) { // NaN or Inf
        return 0xFFFFFFFF; // Place at the end
    }
    return (ui & 0x80000000) ? ~ui : ui ^ 0x80000000;
}

__device__ float uint_to_float(unsigned int ui) {
    // Handle special case for NaN/Inf
    if (ui == 0xFFFFFFFF) {
        return __uint_as_float(0x7F800000); // +Inf
    }
    ui = (ui & 0x80000000) ? ~ui : ui ^ 0x80000000;
    return __uint_as_float(ui);
}

// Convert float array to uint array
__global__ void convert_float_to_uint_kernel(float* input, unsigned int* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = float_to_uint(input[idx]);
    }
}

// Count sort for radix sort
__global__ void count_sort_kernel(unsigned int* input, unsigned int* output, int n, int shift) {
    extern __shared__ unsigned int s_counts[];
    
    // Initialize counts to 0
    for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();
    
    // Count occurrences of each digit
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int digit = (input[i] >> shift) & RADIX_MASK;
        atomicAdd(&s_counts[digit], 1);
    }
    __syncthreads();
    
    // Compute prefix sum (exclusive scan)
    unsigned int sum = 0;
    for (int i = 0; i < RADIX_SIZE; i++) {
        unsigned int val = s_counts[i];
        s_counts[i] = sum;
        sum += val;
    }
    __syncthreads();
    
    // Rearrange data
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int val = input[i];
        unsigned int digit = (val >> shift) & RADIX_MASK;
        unsigned int dest = atomicAdd(&s_counts[digit], 1);
        output[dest] = val;
    }
}

// When we have the unsigned ints sorted, convert back to floats
__global__ void convert_uint_to_float_kernel(unsigned int* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = uint_to_float(input[idx]);
    }
}

// Simple sorting algorithm for small arrays
__global__ void bitonic_sort_kernel(float* data, int n) {
    extern __shared__ float s_bitonic_data[];
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < n) {
        s_bitonic_data[tid] = data[tid];
    }
    __syncthreads();
    
    // Perform bitonic sort
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            
            if (ixj > tid && tid < n && ixj < n) {
                if ((tid & k) == 0) {
                    // Sort in ascending order
                    if (s_bitonic_data[tid] > s_bitonic_data[ixj]) {
                        float temp = s_bitonic_data[tid];
                        s_bitonic_data[tid] = s_bitonic_data[ixj];
                        s_bitonic_data[ixj] = temp;
                    }
                } else {
                    // Sort in descending order
                    if (s_bitonic_data[tid] < s_bitonic_data[ixj]) {
                        float temp = s_bitonic_data[tid];
                        s_bitonic_data[tid] = s_bitonic_data[ixj];
                        s_bitonic_data[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Write results back to global memory
    if (tid < n) {
        data[tid] = s_bitonic_data[tid];
    }
}

// Simple insertion sort for small arrays
__global__ void insertion_sort_kernel(float* data, int start, int n) {
    extern __shared__ float s_insertion_data[];
    
    // Load data into shared memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_insertion_data[i] = data[start + i];
    }
    __syncthreads();
    
    // Only one thread does the sorting
    if (threadIdx.x == 0) {
        for (int i = 1; i < n; i++) {
            float key = s_insertion_data[i];
            int j = i - 1;
            
            while (j >= 0 && s_insertion_data[j] > key) {
                s_insertion_data[j + 1] = s_insertion_data[j];
                j--;
            }
            s_insertion_data[j + 1] = key;
        }
    }
    __syncthreads();
    
    // Write back to global memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[start + i] = s_insertion_data[i];
    }
}

torch::Tensor sort(torch::Tensor data) {
    int n = data.numel();
    torch::Tensor output = data.clone();
    float* d_output = output.data_ptr<float>();
    
    if (n <= 1024) {
        // For small arrays, use shared memory sorting
        if (n <= 256) {
            // Use bitonic sort for very small arrays
            bitonic_sort_kernel<<<1, 256, n * sizeof(float)>>>(d_output, n);
            cudaDeviceSynchronize();
        } else {
            // Split into smaller chunks for larger arrays
            int chunk_size = 256;
            int num_chunks = (n + chunk_size - 1) / chunk_size;
            
            for (int i = 0; i < num_chunks; i++) {
                int start = i * chunk_size;
                int size = min(chunk_size, n - start);
                insertion_sort_kernel<<<1, 128, size * sizeof(float)>>>(d_output, start, size);
            }
            cudaDeviceSynchronize();
            
            // Now merge the sorted chunks with a simple merge sort on CPU
            torch::Tensor temp = output.clone();
            float* h_output = (float*)malloc(n * sizeof(float));
            float* h_temp = (float*)malloc(n * sizeof(float));
            
            cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
            
            for (int width = chunk_size; width < n; width *= 2) {
                for (int i = 0; i < n; i += 2 * width) {
                    int left = i;
                    int mid = min(i + width, n);
                    int right = min(i + 2 * width, n);
                    
                    // Merge [left, mid) and [mid, right)
                    int i1 = left, i2 = mid, j = left;
                    while (i1 < mid && i2 < right) {
                        if (h_output[i1] <= h_output[i2]) {
                            h_temp[j++] = h_output[i1++];
                        } else {
                            h_temp[j++] = h_output[i2++];
                        }
                    }
                    while (i1 < mid) h_temp[j++] = h_output[i1++];
                    while (i2 < right) h_temp[j++] = h_output[i2++];
                }
                
                // Swap pointers
                float* tmp = h_output;
                h_output = h_temp;
                h_temp = tmp;
            }
            
            // Copy result back to device
            cudaMemcpy(d_output, h_output, n * sizeof(float), cudaMemcpyHostToDevice);
            
            free(h_output);
            free(h_temp);
        }
    } else {
        // For larger arrays, use radix sort
        // Allocate device memory for unsigned int arrays and temporary buffer
        torch::Tensor uint_input = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt).device(data.device()));
        torch::Tensor uint_output = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt).device(data.device()));
        torch::Tensor temp_buffer = torch::empty_like(output);
        
        unsigned int* d_uint_input = (unsigned int*)uint_input.data_ptr<int>();
        unsigned int* d_uint_output = (unsigned int*)uint_output.data_ptr<int>();
        float* d_temp = temp_buffer.data_ptr<float>();
        
        // Convert float to uint preserving sort order
        int threads = BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;
        convert_float_to_uint_kernel<<<blocks, threads>>>(d_output, d_uint_input, n);
        cudaDeviceSynchronize();
        
        // Perform radix sort for each byte
        for (int shift = 0; shift < 32; shift += RADIX_BITS) {
            count_sort_kernel<<<blocks, threads, RADIX_SIZE * sizeof(unsigned int)>>>(
                d_uint_input, d_uint_output, n, shift);
            cudaDeviceSynchronize();
            
            // Swap pointers for next iteration
            unsigned int* temp = d_uint_input;
            d_uint_input = d_uint_output;
            d_uint_output = temp;
        }
        
        // Convert sorted uints back to floats
        convert_uint_to_float_kernel<<<blocks, threads>>>(d_uint_input, d_output, n);
        cudaDeviceSynchronize();
        
        // Check for any CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
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
