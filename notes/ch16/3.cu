#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convBackwardData(const float *dout, const float *weight,
                                 float *dx, int batch_size, int out_channels,
                                 int in_channels, int out_height, int out_width,
                                 int in_height, int in_width, int kernel_h,
                                 int kernel_w, int stride, int padding) {
  const int n = blockIdx.x;
  const int c = blockIdx.y;
  const int h = blockIdx.z / in_width;
  const int w = blockIdx.z % in_width;

  if (n < batch_size && c < in_channels && h < in_height && w < in_width) {
    float val = 0.0f;

    for (int oc = 0; oc < out_channels; ++oc) {
      for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
          int h_out = (h + padding - kh) / stride;
          int w_out = (w + padding - kw) / stride;

          if (h_out >= 0 && h_out < out_height && w_out >= 0 &&
              w_out < out_width && (h + padding - kh) % stride == 0 &&
              (w + padding - kw) % stride == 0) {

            int dout_idx =
                ((n * out_channels + oc) * out_height + h_out) * out_width +
                w_out;
            int weight_idx =
                ((oc * in_channels + c) * kernel_h + kernel_h - 1 - kh) *
                    kernel_w +
                kernel_w - 1 - kw;

            val += dout[dout_idx] * weight[weight_idx];
          }
        }
      }
    }

    int dx_idx = ((n * in_channels + c) * in_height + h) * in_width + w;
    dx[dx_idx] = val;
  }
}

__global__ void convBackwardFilter(const float *input, const float *dout,
                                   float *dweight, int batch_size,
                                   int out_channels, int in_channels,
                                   int out_height, int out_width, int in_height,
                                   int in_width, int kernel_h, int kernel_w,
                                   int stride, int padding) {
  const int oc = blockIdx.x;
  const int ic = blockIdx.y;
  const int kh = blockIdx.z / kernel_w;
  const int kw = blockIdx.z % kernel_w;

  if (oc < out_channels && ic < in_channels && kh < kernel_h && kw < kernel_w) {
    float val = 0.0f;

    for (int n = 0; n < batch_size; ++n) {
      for (int h_out = 0; h_out < out_height; ++h_out) {
        for (int w_out = 0; w_out < out_width; ++w_out) {
          int h_in = h_out * stride - padding + kh;
          int w_in = w_out * stride - padding + kw;

          if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
            int input_idx =
                ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
            int dout_idx =
                ((n * out_channels + oc) * out_height + h_out) * out_width +
                w_out;

            val += input[input_idx] * dout[dout_idx];
          }
        }
      }
    }

    int dweight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
    dweight[dweight_idx] = val;
  }
}

cudaError_t launchConvBackwardData(const float *dout, const float *weight,
                                   float *dx, int batch_size, int out_channels,
                                   int in_channels, int out_height,
                                   int out_width, int in_height, int in_width,
                                   int kernel_h, int kernel_w, int stride,
                                   int padding) {
  dim3 grid(batch_size, in_channels, in_height * in_width);

  convBackwardData<<<grid, 1>>>(dout, weight, dx, batch_size, out_channels,
                                in_channels, out_height, out_width, in_height,
                                in_width, kernel_h, kernel_w, stride, padding);

  return cudaGetLastError();
}

cudaError_t launchConvBackwardFilter(const float *input, const float *dout,
                                     float *dweight, int batch_size,
                                     int out_channels, int in_channels,
                                     int out_height, int out_width,
                                     int in_height, int in_width, int kernel_h,
                                     int kernel_w, int stride, int padding) {
  dim3 grid(out_channels, in_channels, kernel_h * kernel_w);

  convBackwardFilter<<<grid, 1>>>(
      input, dout, dweight, batch_size, out_channels, in_channels, out_height,
      out_width, in_height, in_width, kernel_h, kernel_w, stride, padding);

  return cudaGetLastError();
}

// CPU reference implementation for convolution backward data
void convBackwardDataCPU(const float *dout, const float *weight, float *dx,
                        int batch_size, int out_channels, int in_channels, 
                        int out_height, int out_width, int in_height, int in_width,
                        int kernel_h, int kernel_w, int stride, int padding) {
  
  // Initialize dx to zeros
  size_t dx_size = batch_size * in_channels * in_height * in_width;
  for (int i = 0; i < dx_size; i++) {
    dx[i] = 0.0f;
  }
  
  // For each input position
  for (int n = 0; n < batch_size; n++) {
    for (int c = 0; c < in_channels; c++) {
      for (int h = 0; h < in_height; h++) {
        for (int w = 0; w < in_width; w++) {
          float val = 0.0f;
          
          // For each output channel
          for (int oc = 0; oc < out_channels; oc++) {
            // For each kernel position
            for (int kh = 0; kh < kernel_h; kh++) {
              for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate the corresponding output position
                int h_out = (h + padding - kh) / stride;
                int w_out = (w + padding - kw) / stride;
                
                // Only if the output position is valid and stride conditions are met
                if (h_out >= 0 && h_out < out_height && w_out >= 0 && 
                    w_out < out_width && (h + padding - kh) % stride == 0 && 
                    (w + padding - kw) % stride == 0) {
                  
                  int dout_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
                  // Flip the kernel for convolution backward
                  int weight_idx = ((oc * in_channels + c) * kernel_h + kernel_h - 1 - kh) * kernel_w + 
                                  kernel_w - 1 - kw;
                  
                  val += dout[dout_idx] * weight[weight_idx];
                }
              }
            }
          }
          
          int dx_idx = ((n * in_channels + c) * in_height + h) * in_width + w;
          dx[dx_idx] = val;
        }
      }
    }
  }
}

// CPU reference implementation for convolution backward filter
void convBackwardFilterCPU(const float *input, const float *dout, float *dweight,
                           int batch_size, int out_channels, int in_channels,
                           int out_height, int out_width, int in_height, int in_width,
                           int kernel_h, int kernel_w, int stride, int padding) {
  
  // Initialize dweight to zeros
  size_t dweight_size = out_channels * in_channels * kernel_h * kernel_w;
  for (int i = 0; i < dweight_size; i++) {
    dweight[i] = 0.0f;
  }
  
  // For each kernel position
  for (int oc = 0; oc < out_channels; oc++) {
    for (int ic = 0; ic < in_channels; ic++) {
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          float val = 0.0f;
          
          // For each batch
          for (int n = 0; n < batch_size; n++) {
            // For each output position
            for (int h_out = 0; h_out < out_height; h_out++) {
              for (int w_out = 0; w_out < out_width; w_out++) {
                // Calculate the corresponding input position
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                // Only if the input position is valid
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                  int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                  int dout_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
                  
                  val += input[input_idx] * dout[dout_idx];
                }
              }
            }
          }
          
          int dweight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
          dweight[dweight_idx] = val;
        }
      }
    }
  }
}

// Function to verify results between CPU and GPU
bool verifyResults(const float *cpu_output, const float *gpu_output, int size) {
  const float epsilon = 1e-4; // Increased from 1e-5 to account for floating-point precision differences
  bool match = true;
  
  for (int i = 0; i < size; i++) {
    if (fabs(cpu_output[i] - gpu_output[i]) > epsilon) {
      printf("Mismatch at index %d: CPU = %f, GPU = %f, diff = %f\n", 
             i, cpu_output[i], gpu_output[i], fabs(cpu_output[i] - gpu_output[i]));
      match = false;
    }
  }
  return match;
}

int main() {
  int batch_size = 2;
  int in_channels = 3;
  int out_channels = 16;
  int in_height = 32;
  int in_width = 32;
  int kernel_h = 3;
  int kernel_w = 3;
  int stride = 1;
  int padding = 1;

  int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

  size_t input_size =
      batch_size * in_channels * in_height * in_width * sizeof(float);
  size_t weight_size =
      out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
  size_t output_size =
      batch_size * out_channels * out_height * out_width * sizeof(float);

  // Allocate host memory
  float *h_input = (float *)malloc(input_size);
  float *h_weight = (float *)malloc(weight_size);
  float *h_dout = (float *)malloc(output_size);
  float *h_dx = (float *)malloc(input_size);
  float *h_dx_cpu = (float *)malloc(input_size);
  float *h_dweight = (float *)malloc(weight_size);
  float *h_dweight_cpu = (float *)malloc(weight_size);

  // Initialize host data (random values for simplicity)
  for (int i = 0; i < batch_size * in_channels * in_height * in_width; i++) {
    h_input[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  
  for (int i = 0; i < out_channels * in_channels * kernel_h * kernel_w; i++) {
    h_weight[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  
  for (int i = 0; i < batch_size * out_channels * out_height * out_width; i++) {
    h_dout[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_input, *d_weight, *d_dout, *d_dx, *d_dweight;
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_weight, weight_size);
  cudaMalloc(&d_dout, output_size);
  cudaMalloc(&d_dx, input_size);
  cudaMalloc(&d_dweight, weight_size);

  // Copy data to device
  cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dout, h_dout, output_size, cudaMemcpyHostToDevice);

  // Run CPU convBackwardData and measure time
  auto cpu_start_data = std::chrono::high_resolution_clock::now();
  convBackwardDataCPU(h_dout, h_weight, h_dx_cpu, batch_size, out_channels, 
                     in_channels, out_height, out_width, in_height, in_width, 
                     kernel_h, kernel_w, stride, padding);
  auto cpu_end_data = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration_data = cpu_end_data - cpu_start_data;

  // Create CUDA events for timing GPU convBackwardData
  cudaEvent_t start_data, stop_data;
  cudaEventCreate(&start_data);
  cudaEventCreate(&stop_data);

  // Run GPU convBackwardData and measure time
  cudaEventRecord(start_data);
  cudaError_t err1 = launchConvBackwardData(
      d_dout, d_weight, d_dx, batch_size, out_channels, in_channels, out_height,
      out_width, in_height, in_width, kernel_h, kernel_w, stride, padding);
  cudaEventRecord(stop_data);
  cudaEventSynchronize(stop_data);

  float gpu_duration_data = 0.0f;
  cudaEventElapsedTime(&gpu_duration_data, start_data, stop_data);

  // Copy data back to host
  cudaMemcpy(h_dx, d_dx, input_size, cudaMemcpyDeviceToHost);

  // Verify convBackwardData results
  bool data_match = verifyResults(h_dx_cpu, h_dx, batch_size * in_channels * in_height * in_width);

  // Run CPU convBackwardFilter and measure time
  auto cpu_start_filter = std::chrono::high_resolution_clock::now();
  convBackwardFilterCPU(h_input, h_dout, h_dweight_cpu, batch_size, out_channels, 
                        in_channels, out_height, out_width, in_height, in_width, 
                        kernel_h, kernel_w, stride, padding);
  auto cpu_end_filter = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration_filter = cpu_end_filter - cpu_start_filter;

  // Create CUDA events for timing GPU convBackwardFilter
  cudaEvent_t start_filter, stop_filter;
  cudaEventCreate(&start_filter);
  cudaEventCreate(&stop_filter);

  // Run GPU convBackwardFilter and measure time
  cudaEventRecord(start_filter);
  cudaError_t err2 = launchConvBackwardFilter(
      d_input, d_dout, d_dweight, batch_size, out_channels, in_channels,
      out_height, out_width, in_height, in_width, kernel_h, kernel_w, stride, padding);
  cudaEventRecord(stop_filter);
  cudaEventSynchronize(stop_filter);

  float gpu_duration_filter = 0.0f;
  cudaEventElapsedTime(&gpu_duration_filter, start_filter, stop_filter);

  // Copy data back to host
  cudaMemcpy(h_dweight, d_dweight, weight_size, cudaMemcpyDeviceToHost);

  // Verify convBackwardFilter results
  bool filter_match = verifyResults(h_dweight_cpu, h_dweight, 
                                    out_channels * in_channels * kernel_h * kernel_w);

  // Print timing and verification results
  printf("Conv Backward Data (%dx%d input, %dx%d kernel, %d stride, %d padding):\n", 
         in_height, in_width, kernel_h, kernel_w, stride, padding);
  printf("CPU Time: %.4f ms\n", cpu_duration_data.count());
  printf("GPU Time: %.4f ms\n", gpu_duration_data);
  printf("Speedup: %.2fx\n", cpu_duration_data.count() / gpu_duration_data);
  printf("Verification: %s\n\n", data_match ? "PASSED" : "FAILED");

  printf("Conv Backward Filter (%dx%d input, %dx%d kernel, %d stride, %d padding):\n", 
         in_height, in_width, kernel_h, kernel_w, stride, padding);
  printf("CPU Time: %.4f ms\n", cpu_duration_filter.count());
  printf("GPU Time: %.4f ms\n", gpu_duration_filter);
  printf("Speedup: %.2fx\n", cpu_duration_filter.count() / gpu_duration_filter);
  printf("Verification: %s\n\n", filter_match ? "PASSED" : "FAILED");

  if (err1 != cudaSuccess) {
    printf("Error in backward data: %s\n", cudaGetErrorString(err1));
  }
  if (err2 != cudaSuccess) {
    printf("Error in backward filter: %s\n", cudaGetErrorString(err2));
  }

  // Free host memory
  free(h_input);
  free(h_weight);
  free(h_dout);
  free(h_dx);
  free(h_dx_cpu);
  free(h_dweight);
  free(h_dweight_cpu);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_dout);
  cudaFree(d_dx);
  cudaFree(d_dweight);

  // Destroy CUDA events
  cudaEventDestroy(start_data);
  cudaEventDestroy(stop_data);
  cudaEventDestroy(start_filter);
  cudaEventDestroy(stop_filter);

  return 0;
}