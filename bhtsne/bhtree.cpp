#include "bhtree.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>

#define gpu_check(ans) gpu_assert((ans), __FILE__, __LINE__)

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

// defined in kernels.cu
void init_cuda(int num_points, float theta);
void to_device(float* points, int num_points);
int compute_nonedge_forces_cuda();
int get_nonedge_force_from_cuda(int num_points, float* neg_forces, float* sum_Q);
void cleanup_cuda();

// Constructor
// Allocate memory on host and device.
BHTree::BHTree(int num_points, float theta) {
    init_cuda(num_points, theta);
    BHTree::num_points = num_points;
}

BHTree::~BHTree() {
    cleanup_cuda();
}

void BHTree::compute_nonedge_forces(float* points) {
    to_device(points, BHTree::num_points);
    int ret_val = compute_nonedge_forces_cuda();
    assert(ret_val == 0);
}

void BHTree::compute_nonedge_forces() {
    int ret_val = compute_nonedge_forces_cuda();
    assert(ret_val == 0);
}

void BHTree::get_nonedge_forces(float* neg_forces, float* sum_Q) {
    int ret_val = get_nonedge_force_from_cuda(BHTree::num_points, neg_forces, sum_Q);
    assert(ret_val == 0);
}