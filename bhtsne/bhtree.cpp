#include "bhtree.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpu_check(ans) gpu_assert((ans), __FILE__, __LINE__)

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

// Declarations of kernel wrappers, defined in kernels.cu
void reset_arrays(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child,
                  int *index, float *left, float *right, float *bottom, float *top, int n, int m);
void compute_bounding_box(int *mutex, float *x, float *y, float *left, float *right, float *bottom, float *top, int n);
void build_quad_tree(float *x, float *y, float *mass, int *count, int *start, int *child, int *index, float *left,
                     float *right, float *bottom, float *top, int n, int m);
void compute_center_of_mass(float *x, float *y, float *mass, int *index, int n);
void sort_particles(int *count, int *start, int *sorted, int *child, int *index, int n);
void compute_forces(float *x, float *y, float *ax, float *ay, float *mass, int *sorted, int *child, float *left,
                    float *right, int n, float g);

// Constructor
BHTree::BHTree(float *Ys, int N, int output_dims) {
    // Allocate memory on host and device.
    num_points = N;
    num_nodes = 2 * N + 12000;

    h_left = new float;
    h_right = new float;
    h_bottom = new float;
    h_top = new float;

    h_mass = new float[num_nodes];
    h_x = new float[num_nodes];
    h_y = new float[num_nodes];
    h_ax = new float[num_nodes];
    h_ay = new float[num_nodes];

    h_child = new int[4 * num_nodes];
    h_start = new int[num_nodes];
    h_sorted = new int[num_nodes];
    h_count = new int[num_nodes];
    h_output = new float[2 * num_nodes];

    gpu_check(cudaMalloc((void **)&d_left, sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_right, sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_bottom, sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_top, sizeof(float)));
    gpu_check(cudaMemset(d_left, 0, sizeof(float)));
    gpu_check(cudaMemset(d_right, 0, sizeof(float)));
    gpu_check(cudaMemset(d_bottom, 0, sizeof(float)));
    gpu_check(cudaMemset(d_top, 0, sizeof(float)));

    gpu_check(cudaMalloc((void **)&d_mass, num_nodes * sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_x, num_nodes * sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_y, num_nodes * sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_ax, num_nodes * sizeof(float)));
    gpu_check(cudaMalloc((void **)&d_ay, num_nodes * sizeof(float)));

    gpu_check(cudaMalloc((void **)&d_index, sizeof(int)));
    gpu_check(cudaMalloc((void **)&d_child, 4 * num_nodes * sizeof(int)));
    gpu_check(cudaMalloc((void **)&d_start, num_nodes * sizeof(int)));
    gpu_check(cudaMalloc((void **)&d_sorted, num_nodes * sizeof(int)));
    gpu_check(cudaMalloc((void **)&d_count, num_nodes * sizeof(int)));
    gpu_check(cudaMalloc((void **)&d_mutex, sizeof(int)));

    gpu_check(cudaMemset(d_start, -1, num_nodes * sizeof(int)));
    gpu_check(cudaMemset(d_sorted, 0, num_nodes * sizeof(int)));

    gpu_check(cudaMalloc((void **)&d_output, 2 * num_nodes * sizeof(float)));

    // Copy input to host
    assert(output_dims == 2);
    for (int i = 0; i < num_points; i++) {
        h_x[i] = Ys[i * output_dims];
        h_y[i] = Ys[i * output_dims + 1];
        h_mass[i] = 1.0;
        h_ax[i] = h_ay[i] = 0.0;
    }

    // Copy from host to device
    gpu_check(cudaMemcpy(d_mass, h_mass, 2 * num_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check(cudaMemcpy(d_x, h_x, 2 * num_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check(cudaMemcpy(d_y, h_y, 2 * num_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check(cudaMemcpy(d_ax, h_ax, 2 * num_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check(cudaMemcpy(d_ay, h_ay, 2 * num_points * sizeof(float), cudaMemcpyHostToDevice));
}

BHTree::~BHTree() {
    printf("destructor\n");

    delete h_left;
    delete h_right;
    delete h_bottom;
    delete h_top;
    delete[] h_mass;
    delete[] h_x;
    delete[] h_y;
    delete[] h_ax;
    delete[] h_ay;
    delete[] h_child;
    delete[] h_start;
    delete[] h_sorted;
    delete[] h_count;
    delete[] h_output;

    gpu_check(cudaFree(d_left));
    gpu_check(cudaFree(d_right));
    gpu_check(cudaFree(d_bottom));
    gpu_check(cudaFree(d_top));

    gpu_check(cudaFree(d_mass));
    gpu_check(cudaFree(d_x));
    gpu_check(cudaFree(d_y));
    gpu_check(cudaFree(d_ax));
    gpu_check(cudaFree(d_ay));

    gpu_check(cudaFree(d_index));
    gpu_check(cudaFree(d_child));
    gpu_check(cudaFree(d_start));
    gpu_check(cudaFree(d_sorted));
    gpu_check(cudaFree(d_count));

    gpu_check(cudaFree(d_mutex));
    gpu_check(cudaFree(d_output));

    cudaDeviceSynchronize();
}

void BHTree::compute_nonedge_forces(float theta) {
    reset_arrays(d_mutex, d_x, d_y, d_mass, d_count, d_start, d_sorted, d_child, d_index, d_left, d_right, d_bottom,
                 d_top, num_points, num_nodes);
    compute_bounding_box(d_mutex, d_x, d_y, d_left, d_right, d_bottom, d_top, num_points);
    build_quad_tree(d_x, d_y, d_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, num_points,
                    num_nodes);
    compute_center_of_mass(d_x, d_y, d_mass, d_index, num_points);
    sort_particles(d_count, d_start, d_sorted, d_child, d_index, num_points);
    compute_forces(d_x, d_y, d_ax, d_ay, d_mass, d_sorted, d_child, d_left, d_right, num_points, 1.0);

    cudaDeviceSynchronize();
}