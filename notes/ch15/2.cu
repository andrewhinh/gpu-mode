#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INF -1   // Used to mark unvisited vertices
#define ALPHA 14 // Constant for switching from push to pull
#define BETA 24  // Constant for switching from pull to push

struct Graph {
  int num_vertices;
  int num_edges;
  int *row_ptr;
  int *col_idx;
};

struct BFSData {
  int *dist;
  int *visited;  // Changed from bool to int to avoid alignment issues
  int *frontier;
  int *new_frontier;
  int *frontier_size;
  int *new_frontier_size;
};

void initGraph(Graph &graph, int num_vertices, int num_edges, int *row_ptr,
               int *col_idx) {
  graph.num_vertices = num_vertices;
  graph.num_edges = num_edges;

  cudaMalloc((void **)&graph.row_ptr, (num_vertices + 1) * sizeof(int));
  cudaMemcpy(graph.row_ptr, row_ptr, (num_vertices + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&graph.col_idx, num_edges * sizeof(int));
  cudaMemcpy(graph.col_idx, col_idx, num_edges * sizeof(int),
             cudaMemcpyHostToDevice);
}

void initBFSData(BFSData &bfs_data, int num_vertices, int source) {
  int *h_dist = (int *)malloc(num_vertices * sizeof(int));
  int *h_visited = (int *)malloc(num_vertices * sizeof(int));  // Changed from bool to int

  for (int i = 0; i < num_vertices; i++) {
    h_dist[i] = INF;
    h_visited[i] = 0;  // 0 = false, 1 = true
  }
  h_dist[source] = 0;
  h_visited[source] = 1;  // Mark source as visited

  cudaMalloc((void **)&bfs_data.dist, num_vertices * sizeof(int));
  cudaMalloc((void **)&bfs_data.visited, num_vertices * sizeof(int));  // Changed from bool to int
  cudaMalloc((void **)&bfs_data.frontier, num_vertices * sizeof(int));
  cudaMalloc((void **)&bfs_data.new_frontier, num_vertices * sizeof(int));
  cudaMalloc((void **)&bfs_data.frontier_size, sizeof(int));
  cudaMalloc((void **)&bfs_data.new_frontier_size, sizeof(int));

  cudaMemcpy(bfs_data.dist, h_dist, num_vertices * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bfs_data.visited, h_visited, num_vertices * sizeof(int),
             cudaMemcpyHostToDevice);

  int initial_frontier[1] = {source};
  int initial_size = 1;
  cudaMemcpy(bfs_data.frontier, initial_frontier, sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bfs_data.frontier_size, &initial_size, sizeof(int),
             cudaMemcpyHostToDevice);

  int zero = 0;
  cudaMemcpy(bfs_data.new_frontier_size, &zero, sizeof(int),
             cudaMemcpyHostToDevice);

  free(h_dist);
  free(h_visited);
}

void cleanupBFSData(BFSData &bfs_data) {
  cudaFree(bfs_data.dist);
  cudaFree(bfs_data.visited);
  cudaFree(bfs_data.frontier);
  cudaFree(bfs_data.new_frontier);
  cudaFree(bfs_data.frontier_size);
  cudaFree(bfs_data.new_frontier_size);
}

void cleanupGraph(Graph &graph) {
  cudaFree(graph.row_ptr);
  cudaFree(graph.col_idx);
}

__global__ void bfsPushKernel(Graph graph, BFSData bfs_data, int level) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < *bfs_data.frontier_size) {
    int vertex = bfs_data.frontier[tid];

    int start = graph.row_ptr[vertex];
    int end = graph.row_ptr[vertex + 1];

    for (int edge = start; edge < end; edge++) {
      int neighbor = graph.col_idx[edge];

      // Change from bool to int comparison
      if (bfs_data.visited[neighbor] == 0) {
        // Compare and swap with int values instead of bool
        int old_val = atomicCAS(&bfs_data.visited[neighbor], 0, 1);
        if (old_val == 0) {
          int index = atomicAdd(bfs_data.new_frontier_size, 1);
          bfs_data.new_frontier[index] = neighbor;
          bfs_data.dist[neighbor] = level + 1;
        }
      }
    }
  }
}

__global__ void bfsPullKernel(Graph graph, BFSData bfs_data, int level) {
  int vertex = blockIdx.x * blockDim.x + threadIdx.x;

  if (vertex < graph.num_vertices && bfs_data.dist[vertex] == INF) {
    for (int i = 0; i < graph.num_vertices; i++) {
      // Check if vertex i is at the current level and has an edge to vertex
      if (bfs_data.dist[i] == level) {
        int start = graph.row_ptr[i];
        int end = graph.row_ptr[i + 1];

        for (int edge = start; edge < end; edge++) {
          if (graph.col_idx[edge] == vertex) {
            bfs_data.dist[vertex] = level + 1;
            bfs_data.visited[vertex] = 1;  // Changed from bool to int
            int index = atomicAdd(bfs_data.new_frontier_size, 1);
            bfs_data.new_frontier[index] = vertex;
            return;
          }
        }
      }
    }
  }
}

bool shouldUsePull(int frontier_size, int num_edges, int num_vertices,
                   int edges_examined_push) {
  return (frontier_size * ALPHA > num_edges);
}

bool shouldUsePush(int frontier_size, int num_vertices) {
  return (frontier_size * BETA < num_vertices);
}

void directionOptimizedBFS(Graph &graph, BFSData &bfs_data, int source) {
  int level = 0;
  int h_frontier_size = 1; // Start with just the source vertex
  bool using_pull = false;

  int block_size = 256;
  while (h_frontier_size > 0) {
    if (!using_pull && shouldUsePull(h_frontier_size, graph.num_edges,
                                     graph.num_vertices, 0)) {
      using_pull = true;
      printf("Switching to PULL at level %d, frontier size: %d\n", level,
             h_frontier_size);
    } else if (using_pull &&
               shouldUsePush(h_frontier_size, graph.num_vertices)) {
      using_pull = false;
      printf("Switching to PUSH at level %d, frontier size: %d\n", level,
             h_frontier_size);
    }

    int zero = 0;
    cudaMemcpy(bfs_data.new_frontier_size, &zero, sizeof(int),
               cudaMemcpyHostToDevice);

    if (using_pull) {
      int grid_size = (graph.num_vertices + block_size - 1) / block_size;
      bfsPullKernel<<<grid_size, block_size>>>(graph, bfs_data, level);
    } else {
      int grid_size = (h_frontier_size + block_size - 1) / block_size;
      bfsPushKernel<<<grid_size, block_size>>>(graph, bfs_data, level);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      break;
    }

    cudaDeviceSynchronize();

    int *temp_frontier = bfs_data.frontier;
    bfs_data.frontier = bfs_data.new_frontier;
    bfs_data.new_frontier = temp_frontier;

    cudaMemcpy(&h_frontier_size, bfs_data.new_frontier_size, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(bfs_data.frontier_size, &h_frontier_size, sizeof(int),
               cudaMemcpyHostToDevice);

    level++;
  }

  printf("BFS completed after %d levels\n", level);
}

int main(int argc, char **argv) {
  int num_vertices = 8;
  int num_edges = 15;

  int row_ptr[9] = {0, 2, 5, 6, 8, 9, 11, 12, 15};
  int col_idx[15] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6};

  Graph graph;
  initGraph(graph, num_vertices, num_edges, row_ptr, col_idx);

  BFSData bfs_data;
  int source = 0;
  initBFSData(bfs_data, num_vertices, source);

  directionOptimizedBFS(graph, bfs_data, source);

  int *h_dist = (int *)malloc(num_vertices * sizeof(int));
  cudaMemcpy(h_dist, bfs_data.dist, num_vertices * sizeof(int),
             cudaMemcpyDeviceToHost);

  printf("Distances from source vertex %d:\n", source);
  for (int i = 0; i < num_vertices; i++) {
    if (h_dist[i] != INF) {
      printf("Vertex %d: Distance %d\n", i, h_dist[i]);
    } else {
      printf("Vertex %d: Unreachable\n", i);
    }
  }

  free(h_dist);
  cleanupBFSData(bfs_data);
  cleanupGraph(graph);

  return 0;
}