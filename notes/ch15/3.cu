#include <cuda_runtime.h>
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
  __shared__ int s_new_frontier[MAX_VERTICES];
  __shared__ int s_frontier_size;
  __shared__ int s_new_frontier_size;
  __shared__ int s_done;

  int tid = threadIdx.x;

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

  for (int i = tid; i < num_vertices; i += blockDim.x) {
    if (i < num_vertices) {
      s_distances[i] = INF;
    }
  }

  if (tid == 0) {
    s_distances[source] = 0;
    s_frontier[0] = source;
    s_frontier_size = 1;
    s_new_frontier_size = 0;
    s_done = 0;
  }

  __syncthreads();

  int level = 0;

  while (true) {
    __syncthreads();

    if (tid == 0) {
      s_done = (s_frontier_size == 0) ? 1 : 0;
      s_new_frontier_size = 0;
    }

    __syncthreads();

    if (s_done) {
      break;
    }

    for (int i = tid; i < s_frontier_size; i += blockDim.x) {
      int v = s_frontier[i];

      int start = s_row_ptr[v];
      int end = s_row_ptr[v + 1];

      for (int e = start; e < end; e++) {
        int neighbor = s_col_idx[e];

        if (atomicCAS(&s_distances[neighbor], INF, level + 1) == INF) {
          int idx = atomicAdd(&s_new_frontier_size, 1);
          s_new_frontier[idx] = neighbor;
        }
      }
    }

    __syncthreads();

    if (tid == 0) {
      for (int i = 0; i < s_new_frontier_size; i++) {
        s_frontier[i] = s_new_frontier[i];
      }
      s_frontier_size = s_new_frontier_size;
      level++;
    }

    __syncthreads();
  }

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

int main(int argc, char **argv) {
  int num_vertices = 8;
  int num_edges = 15;

  int row_ptr[9] = {0, 2, 5, 6, 8, 9, 11, 12, 15};

  int col_idx[15] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6};

  Graph graph;
  initGraph(graph, num_vertices, num_edges, row_ptr, col_idx);

  int *distances = (int *)malloc(num_vertices * sizeof(int));

  int source = 0;
  singleBlockBFS(graph, source, distances);

  printf("Distances from source vertex %d using single-block BFS:\n", source);
  for (int i = 0; i < num_vertices; i++) {
    if (distances[i] != INF) {
      printf("Vertex %d: Distance %d\n", i, distances[i]);
    } else {
      printf("Vertex %d: Unreachable\n", i);
    }
  }

  free(distances);
  cleanupGraph(graph);

  return 0;
}