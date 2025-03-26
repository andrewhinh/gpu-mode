#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void maxPoolForward(const float *input, float *output,
                               int batch_size, int in_channels, int in_height,
                               int in_width, int out_height, int out_width,
                               int kernel_size, int stride, int padding) {

  int n = blockIdx.x;
  int c = blockIdx.y;
  int h_out = blockIdx.z / out_width;
  int w_out = blockIdx.z % out_width;

  if (n < batch_size && c < in_channels && h_out < out_height &&
      w_out < out_width) {

    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    float max_val = -INFINITY;

    for (int kh = 0; kh < kernel_size; kh++) {
      for (int kw = 0; kw < kernel_size; kw++) {
        int h_in = h_in_start + kh;
        int w_in = w_in_start + kw;

        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
          int input_idx =
              ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
          float val = input[input_idx];

          max_val = fmaxf(max_val, val);
        }
      }
    }

    int output_idx =
        ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
    output[output_idx] = max_val;
  }
}

void launchMaxPoolForward(const float *input, float *output, int batch_size,
                          int in_channels, int in_height, int in_width,
                          int kernel_size, int stride, int padding) {

  int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

  dim3 grid(batch_size, in_channels, out_height * out_width);

  maxPoolForward<<<grid, 1>>>(input, output, batch_size, in_channels, in_height,
                              in_width, out_height, out_width, kernel_size,
                              stride, padding);
}

// CPU reference implementation of max pooling forward
void maxPoolForwardCPU(const float *input, float *output, int batch_size,
                       int in_channels, int in_height, int in_width,
                       int kernel_size, int stride, int padding) {

  int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

  // For each output element
  for (int n = 0; n < batch_size; n++) {
    for (int c = 0; c < in_channels; c++) {
      for (int h_out = 0; h_out < out_height; h_out++) {
        for (int w_out = 0; w_out < out_width; w_out++) {

          int h_in_start = h_out * stride - padding;
          int w_in_start = w_out * stride - padding;

          float max_val = -INFINITY;

          // Apply max pooling kernel
          for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int h_in = h_in_start + kh;
              int w_in = w_in_start + kw;

              if (h_in >= 0 && h_in < in_height && w_in >= 0 &&
                  w_in < in_width) {
                int input_idx =
                    ((n * in_channels + c) * in_height + h_in) * in_width +
                    w_in;
                float val = input[input_idx];
                max_val = fmaxf(max_val, val);
              }
            }
          }

          int output_idx =
              ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
          output[output_idx] = max_val;
        }
      }
    }
  }
}

// Function to verify max pooling results between CPU and GPU
bool verifyMaxPoolResults(float *cpu_output, float *gpu_output, int size) {
  const float epsilon = 1e-5;
  bool match = true;

  for (int i = 0; i < size; i++) {
    if (fabs(cpu_output[i] - gpu_output[i]) > epsilon) {
      printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_output[i],
             gpu_output[i]);
      match = false;
    }
  }
  return match;
}

int main() {
  int batch_size = 1;
  int channels = 3;
  int height = 32;
  int width = 32;
  int kernel_size = 2;
  int stride = 2;
  int padding = 0;

  int out_height = (height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (width + 2 * padding - kernel_size) / stride + 1;

  size_t input_size = batch_size * channels * height * width * sizeof(float);
  size_t output_size =
      batch_size * channels * out_height * out_width * sizeof(float);

  float *h_input, *h_output, *h_output_cpu, *d_input, *d_output;

  h_input = (float *)malloc(input_size);
  h_output = (float *)malloc(output_size);
  h_output_cpu = (float *)malloc(output_size);

  // Initialize input data
  for (int i = 0; i < batch_size * channels * height * width; i++) {
    h_input[i] = static_cast<float>(i % 10);
  }

  // Allocate GPU memory
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, output_size);

  // Copy input to GPU
  cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

  // Run CPU implementation and measure time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  maxPoolForwardCPU(h_input, h_output_cpu, batch_size, channels, height, width,
                    kernel_size, stride, padding);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // Create CUDA events for timing GPU implementation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure GPU implementation time
  cudaEventRecord(start);
  launchMaxPoolForward(d_input, d_output, batch_size, channels, height, width,
                       kernel_size, stride, padding);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate GPU elapsed time
  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Copy output from GPU
  cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

  // Verify results
  int output_elements = batch_size * channels * out_height * out_width;
  bool results_match =
      verifyMaxPoolResults(h_output_cpu, h_output, output_elements);

  // Print timing and verification results
  printf(
      "Max Pooling Performance (%d x %d -> %d x %d, kernel=%d, stride=%d):\n",
      height, width, out_height, out_width, kernel_size, stride);
  printf("CPU Time: %.4f ms\n", cpu_duration.count());
  printf("GPU Time: %.4f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n\n", results_match ? "PASSED" : "FAILED");

  free(h_input);
  free(h_output);
  free(h_output_cpu);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}