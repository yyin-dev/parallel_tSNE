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

void init();

// Constructor
// Allocate memory on host and device.
BHTree::BHTree(float *Ys, int N, int output_dims) {}

BHTree::~BHTree() { cudaDeviceSynchronize(); }

void BHTree::compute_nonedge_forces(float theta) {
    init(); 
}