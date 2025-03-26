#include <chrono>
#include <cuda_runtime.h>
#include <queue>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 1024
#define MAX_EDGES 4096
#define BLOCK_SIZE 512
#define INF -1

struct Graph {
  int num_vertices;
  int num_edges;
  int *row_ptr;
  int *col_idx;
};

void initGraph(Graph &graph, int num_vertices, int num_edges, int *row_ptr,
               int *col_idx) {
  graph.num_vertices = num_vertices;
  graph.num_edges = num_edges;

  cudaMallocHost((void **)&graph.row_ptr, (num_vertices + 1) * sizeof(int));
  memcpy(graph.row_ptr, row_ptr, (num_vertices + 1) * sizeof(int));

  cudaMallocHost((void **)&graph.col_idx, num_edges * sizeof(int));
  memcpy(graph.col_idx, col_idx, num_edges * sizeof(int));
}

void cleanupGraph(Graph &graph) {
  cudaFreeHost(graph.row_ptr);
  cudaFreeHost(graph.col_idx);
}

__global__ void singleBlockBFSKernel(const int *row_ptr, const int *col_idx,
                                     int *distances, const int num_vertices,
                                     const int num_edges, const int source) {
  __shared__ int s_row_ptr[MAX_VERTICES + 1];
  __shared__ int s_col_idx[MAX_EDGES];
  __shared__ int s_distances[MAX_VERTICES];
  __shared__ int s_frontier[MAX_VERTICES];
  __shared__ int s_next_frontier[MAX_VERTICES];
  __shared__ int s_front, s_rear;
  __shared__ int s_next_size;
  __shared__ int s_visited[MAX_VERTICES];

  int tid = threadIdx.x;

  // Load graph data into shared memory
  for (int i = tid; i < num_vertices + 1; i += blockDim.x) {
    if (i < num_vertices + 1) {
      s_row_ptr[i] = row_ptr[i];
    }
  }

  for (int i = tid; i < num_edges; i += blockDim.x) {
    if (i < num_edges) {
      s_col_idx[i] = col_idx[i];
    }
  }

  // Initialize arrays
  for (int i = tid; i < num_vertices; i += blockDim.x) {
    if (i < num_vertices) {
      s_distances[i] = INF;
      s_visited[i] = 0;
    }
  }

  __syncthreads();

  // Initialize queue with source vertex
  if (tid == 0) {
    s_distances[source] = 0;
    s_visited[source] = 1;
    s_frontier[0] = source;
    s_front = 0;
    s_rear = 1; // One element in the queue
    s_next_size = 0;
  }

  __syncthreads();

  // Process level by level
  while (true) {
    __syncthreads();

    // Check if the queue is empty
    if (tid == 0) {
      if (s_front == s_rear) {
        s_front = -1; // Signal to end the loop
      }
    }

    __syncthreads();

    if (s_front == -1) {
      break;
    }

    // Determine current level
    int current_level = s_distances[s_frontier[s_front]];

    // Process all vertices in the current level
    while (true) {
      __syncthreads();

      if (tid == 0) {
        // Check if we've processed all vertices at the current level
        if (s_front == s_rear ||
            s_distances[s_frontier[s_front]] > current_level) {
          s_front = -2; // Signal to move to the next level
        }
      }

      __syncthreads();

      if (s_front == -2) {
        break;
      }

      // Process vertices in current level in parallel
      for (int idx = tid; idx < (s_rear - s_front); idx += blockDim.x) {
        int queue_idx = (s_front + idx) % MAX_VERTICES;
        if (queue_idx < s_rear) {
          int u = s_frontier[queue_idx];

          // Only process if this vertex is at the current level
          if (s_distances[u] == current_level) {
            // Process all neighbors
            int start = s_row_ptr[u];
            int end = s_row_ptr[u + 1];

            for (int e = start; e < end; e++) {
              int v = s_col_idx[e];

              if (s_visited[v] == 0) {
                if (atomicCAS(&s_visited[v], 0, 1) == 0) {
                  s_distances[v] = current_level + 1;

                  // Add to the next frontier
                  int pos = atomicAdd(&s_next_size, 1);
                  s_next_frontier[pos] = v;
                }
              }
            }
          }
        }
      }

      __syncthreads();

      // Move to the next vertex in the frontier
      if (tid == 0) {
        s_front = s_rear;
      }
    }

    __syncthreads();

    // Prepare for the next level - move next frontier to current frontier
    if (tid == 0) {
      // Reset queue pointers for the next level
      s_front = 0;
      s_rear = s_next_size;

      // Copy next frontier to current frontier
      for (int i = 0; i < s_next_size; i++) {
        s_frontier[i] = s_next_frontier[i];
      }

      s_next_size = 0;
    }

    __syncthreads();
  }

  // Copy final distances back to global memory
  for (int i = tid; i < num_vertices; i += blockDim.x) {
    if (i < num_vertices) {
      distances[i] = s_distances[i];
    }
  }
}

void singleBlockBFS(Graph &graph, int source, int *distances) {
  if (graph.num_vertices > MAX_VERTICES || graph.num_edges > MAX_EDGES) {
    printf("Error: Graph is too large for single-block BFS. Vertices: %d (max "
           "%d), Edges: %d (max %d)\n",
           graph.num_vertices, MAX_VERTICES, graph.num_edges, MAX_EDGES);
    return;
  }

  int *d_distances;
  cudaMalloc((void **)&d_distances, graph.num_vertices * sizeof(int));

  // Initialize distances to INF
  int *h_init_dist = (int *)malloc(graph.num_vertices * sizeof(int));
  for (int i = 0; i < graph.num_vertices; i++) {
    h_init_dist[i] = INF;
  }
  cudaMemcpy(d_distances, h_init_dist, graph.num_vertices * sizeof(int),
             cudaMemcpyHostToDevice);
  free(h_init_dist);

  int *d_row_ptr, *d_col_idx;
  cudaMalloc((void **)&d_row_ptr, (graph.num_vertices + 1) * sizeof(int));
  cudaMalloc((void **)&d_col_idx, graph.num_edges * sizeof(int));

  cudaMemcpy(d_row_ptr, graph.row_ptr, (graph.num_vertices + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int),
             cudaMemcpyHostToDevice);

  int block_size =
      (graph.num_vertices < BLOCK_SIZE) ? graph.num_vertices : BLOCK_SIZE;

  singleBlockBFSKernel<<<1, block_size>>>(d_row_ptr, d_col_idx, d_distances,
                                          graph.num_vertices, graph.num_edges,
                                          source);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  cudaMemcpy(distances, d_distances, graph.num_vertices * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(d_distances);
  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
}

// CPU reference implementation of BFS for verification
void bfsCPU(const Graph &graph, int source, int *distances) {
  // Initialize distances and visited array
  bool *visited = new bool[graph.num_vertices];
  for (int i = 0; i < graph.num_vertices; i++) {
    distances[i] = INF;
    visited[i] = false;
  }

  distances[source] = 0;
  visited[source] = true;

  // Create a queue for BFS
  std::queue<int> q;
  q.push(source);

  while (!q.empty()) {
    int u = q.front();
    q.pop();

    // Visit all the adjacent vertices of u
    for (int i = graph.row_ptr[u]; i < graph.row_ptr[u + 1]; i++) {
      int v = graph.col_idx[i];
      if (!visited[v]) {
        visited[v] = true;
        distances[v] = distances[u] + 1;
        q.push(v);
      }
    }
  }

  delete[] visited;
}

// Function to verify BFS results between CPU and GPU
bool verifyBFSResults(int *cpu_dist, int *gpu_dist, int num_vertices) {
  bool match = true;
  for (int i = 0; i < num_vertices; i++) {
    if (cpu_dist[i] != gpu_dist[i]) {
      printf("Mismatch at vertex %d: CPU = %d, GPU = %d\n", i, cpu_dist[i],
             gpu_dist[i]);
      match = false;
    }
  }
  return match;
}

int main(int argc, char **argv) {
  int num_vertices = 8;
  int num_edges = 15;

  int row_ptr[9] = {0, 2, 5, 6, 8, 9, 11, 12, 15};

  int col_idx[15] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6};

  Graph graph;
  initGraph(graph, num_vertices, num_edges, row_ptr, col_idx);

  // Allocate memory for CPU and GPU results
  int *cpu_dist = (int *)malloc(num_vertices * sizeof(int));
  int *gpu_dist = (int *)malloc(num_vertices * sizeof(int));

  // Run CPU implementation and measure time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  bfsCPU(graph, 0, cpu_dist);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

  // Create CUDA events for timing GPU implementation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Measure GPU implementation time
  cudaEventRecord(start);
  int source = 0;
  singleBlockBFS(graph, source, gpu_dist);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_duration = 0.0f;
  cudaEventElapsedTime(&gpu_duration, start, stop);

  // Verify results
  bool results_match = verifyBFSResults(cpu_dist, gpu_dist, num_vertices);

  // Print timing and verification results
  printf("CPU Time: %.4f ms\n", cpu_duration.count());
  printf("GPU Time: %.4f ms\n", gpu_duration);
  printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration);
  printf("Verification: %s\n\n", results_match ? "PASSED" : "FAILED");

  free(cpu_dist);
  free(gpu_dist);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cleanupGraph(graph);

  return 0;
}