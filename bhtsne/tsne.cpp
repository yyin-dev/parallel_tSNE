/*
 *  tsne.cpp
 *  Implementation of both standard and Barnes-Hut-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <chrono>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tsne.h"
#include "vptree.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float> dsec;

// Remove this macro to compute negative forces with GPU
// #define NEG_FORCE_CPU

#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif

void gradient_computation(int num_points, float momentum, float learning_rate);
void exaggerate_perplexity(int num_points, float factor);
void init_gradients(int num_points, int *inp_row_P_host, int *inp_col_P_host, float *inp_val_P_host);
void getFinalPositions(int num_points, float *points);

/*
    Perform t-SNE
        X -- float matrix of size [N, D]
        D -- input dimensionality
        Y -- array to fill with the result of size [N, no_dims]
        no_dims -- target dimentionality
*/
void TSNE::run(float* X, int N, int D, float* Y,
               int no_dims, float perplexity, float theta ,
               int num_threads, int max_iter, int n_iter_early_exag,
               int random_state, bool init_from_Y, int verbose,
               float early_exaggeration, float learning_rate,
               float *final_error) {
    assert(!init_from_Y);

    if (N - 1 < 3 * perplexity) {
        perplexity = (N - 1) / 3;
        if (verbose)
            fprintf(stderr, "Perplexity too large for the number of data points! Adjusting ...\n");
    }

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS(num_threads));
#endif

    /*
        ======================
            Step 1
        ======================
    */

    if (verbose)
        fprintf(stderr, "Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);

    // Set learning parameters
    // set up timer
    float compute_time = 0.;
    int stop_lying_iter = n_iter_early_exag, mom_switch_iter = n_iter_early_exag;
    float momentum = .5, final_momentum = .8;

    // Normalize input data (to prevent numerical problems)
    if (verbose)
        fprintf(stderr, "Computing input similarities...\n");

    auto compute_start = Clock::now();
    zeroMean(X, N, D);
    float max_X = .0;
    for (int i = 0; i < N * D; i++) {
        if (X[i] > max_X) max_X = X[i];
    }
    for (int i = 0; i < N * D; i++) {
        X[i] /= max_X;
    }

    // Compute input similarities
    int* row_P; int* col_P; float* val_P;

    // Compute asymmetric pairwise input similarities
    auto perplexity_start = Clock::now();
    int num_neighbors = (int) (3 * perplexity);
    computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, num_neighbors, verbose);
    float perplexity_time = duration_cast<dsec>(Clock::now() - perplexity_start).count();
    if (verbose)
        fprintf(stderr, "Computing asymmetric pairwise similarities takes %.4f\n", perplexity_time);

    // Symmetrize input similarities
    auto symmetrize_start = Clock::now();
    symmetrizeMatrix(&row_P, &col_P, &val_P, N);
    float sum_P = .0;
    for (int i = 0; i < row_P[N]; i++) {
        sum_P += val_P[i];
    }
    for (int i = 0; i < row_P[N]; i++) {
        val_P[i] /= sum_P;
    }
    float symmetrize_time = duration_cast<dsec>(Clock::now() - symmetrize_start).count();
    if (verbose)
        fprintf(stderr, "Symmetrization takes %.4f\n", symmetrize_time);

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();

    fprintf(stderr, "Perplexity computed in %.4f seconds (sparsity = %f)!\nLearning embedding...\n", compute_time, (float) row_P[N] / ((float) N * (float) N));

    /*
        ======================
            Step 2
        ======================
    */


    BHTree *bhtree = new BHTree(N, theta);
    // gradient kernels make use of bhtree variables
    init_gradients(N, row_P, col_P, val_P);
    // lie about P values
    exaggerate_perplexity(N, early_exaggeration);

    // Initialize solution (randomly)
    for (int i = 0; i < N * no_dims; i++) {
        Y[i] = randn();
    }

    // Perform main training loop
    compute_time = 0.;
    compute_start = Clock::now();
    const int eval_interval = 100;

    bhtree->points_to_device(Y);

    for (int iter = 0; iter < max_iter; iter++) {
        bool need_eval_error = (verbose && ((iter > 0 && iter % eval_interval == 0) || (iter == max_iter - 1)));

        // Compute approximate gradient using GPU
        float error = -1.0f; // error not calculated from GPU
        bhtree->compute_nonedge_forces();
        gradient_computation(N, momentum, learning_rate);

        // Stop lying about the P-values after a while, and switch momentum
        if (iter == stop_lying_iter) {
            exaggerate_perplexity(N, 1.f / early_exaggeration);
        }
        if (iter == mom_switch_iter) {
            momentum = final_momentum;
        }

        // Print out progress
        if (need_eval_error) {
            float time_elapsed = duration_cast<dsec>(Clock::now() - compute_start).count();

            if (iter == 0)
                fprintf(stderr, "Iteration %d: error is %f\n", iter + 1, error);
            else {
                fprintf(stderr, "Iteration %d: error is %f (%d iterations in %.4f seconds)\n", iter + 1, error, eval_interval, time_elapsed - compute_time);
            }
            compute_time = time_elapsed;
        }
    }

    getFinalPositions(N, Y); // requires BH-tree data on GPU
    delete bhtree;

    if (final_error != NULL)
        *final_error = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);

    compute_time = duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Fitting performed in %.4f seconds\n", compute_time);

    free(row_P); row_P = NULL;
    free(col_P); col_P = NULL;
    free(val_P); val_P = NULL;
}

// Evaluate t-SNE cost function (approximately)
float TSNE::evaluateError(int* row_P, int* col_P, float* val_P, float* Y, int N, int no_dims, float theta)
{

    // Get estimate of normalization term
    BHTree *bhtree = new BHTree(N, theta);
    bhtree->points_to_device(Y);
    bhtree->compute_nonedge_forces();
    float sum_Q;
    float* buff = new float[no_dims]();
    bhtree->get_nonedge_forces(buff, &sum_Q);
    delete[] buff;
    delete bhtree;

    // Loop over all edges to compute t-SNE error
    float C = .0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:C)
#endif
    for (int n = 0; n < N; n++) {
        int ind1 = n * no_dims;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {
            float Q = .0;
            int ind2 = col_P[i] * no_dims;
            for (int d = 0; d < no_dims; d++) {
                float b  = Y[ind1 + d] - Y[ind2 + d];
                Q += b * b;
            }
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    return C;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(float* X, int N, int D, int** _row_P, int** _col_P, float** _val_P, float perplexity, int K, int verbose) {

    if (perplexity > K) fprintf(stderr, "Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1) * sizeof(int));
    *_col_P = (int*)    calloc(N * K, sizeof(int));
    *_val_P = (float*) calloc(N * K, sizeof(float));
    if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }

    /*
        row_P -- offsets for `col_P` (i)
        col_P -- K nearest neighbors indices (j)
        val_P -- p_{i | j}
    */

    int* row_P = *_row_P;
    int* col_P = *_col_P;
    float* val_P = *_val_P;

    row_P[0] = 0;
    for (int n = 0; n < N; n++) {
        row_P[n + 1] = row_P[n] + K;
    }

    // Build ball tree on data set
    // This part is very fast
    auto build_tree_start = Clock::now();
    VpTree<DataPoint, euclidean_distance_squared>* tree = new VpTree<DataPoint, euclidean_distance_squared>();
    std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for (int n = 0; n < N; n++) {
        obj_X[n] = DataPoint(D, n, X + n * D);
    }
    tree->create(obj_X);
    float build_tree_time = duration_cast<dsec>(Clock::now() - build_tree_start).count();
    if (verbose)
        fprintf(stderr, "Building tree takes %.4f\n", build_tree_time);

    // Loop over all points to find nearest neighbors
    if (verbose)
        fprintf(stderr, "Computing perplexity with nearest neighbors...\n");

    int steps_completed = 0;
    const int log_freq = 5;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int n = 0; n < N; n++)
    {
        std::vector<float> cur_P(K);
        std::vector<DataPoint> indices;
        std::vector<float> distances;

        // Find nearest neighbors
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        float beta = 1.0;
        float min_beta = -FLT_MAX;
        float max_beta =  FLT_MAX;
        float tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; float sum_P;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++) {
                cur_P[m] = exp(-beta * distances[m + 1]);
            }

            // Compute entropy of current row
            sum_P = FLT_MIN;
            for (int m = 0; m < K; m++) {
                sum_P += cur_P[m];
            }
            float H = .0;
            for (int m = 0; m < K; m++) {
                H += beta * (distances[m + 1] * cur_P[m]);
            }
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            float Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == FLT_MAX || max_beta == -FLT_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -FLT_MAX || min_beta == FLT_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (int m = 0; m < K; m++) {
            cur_P[m] /= sum_P;
        }
        for (int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }

        // Print progress
        if (verbose) {
#ifdef _OPENMP
        #pragma omp atomic
#endif
        ++steps_completed;

        if (steps_completed % (N / log_freq) == 0) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            fprintf(stderr, " - point %d of %d\n", steps_completed, N);
            }
        }
    }

    // Clean up memory
    obj_X.clear();
    delete tree;
}

void TSNE::symmetrizeMatrix(int** _row_P, int** _col_P, float** _val_P, int N) {

    // Get sparse matrix
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    float* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if (row_counts == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    break;
                }
            }
            if (present) {
                row_counts[n]++;
            }
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++) {
        no_elem += row_counts[n];
    }
    // Allocate memory for symmetrized matrix
    int*    sym_row_P = (int*)    malloc((N + 1) * sizeof(int));
    int*    sym_col_P = (int*)    malloc(no_elem * sizeof(int));
    float* sym_val_P = (float*) malloc(no_elem * sizeof(float));
    if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if (offset == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {                                 // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    if (n <= col_P[i]) {                                                // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || (n <= col_P[i])) {
                offset[n]++;
                if (col_P[i] != n) {
                    offset[col_P[i]]++;
                }
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++) {
        sym_val_P[i] /= 2.0;
    }

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}


// Makes data zero-mean
void TSNE::zeroMean(float* X, int N, int D) {

    // Compute data mean
    float* mean = (float*) calloc(D, sizeof(float));
    if (mean == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[n * D + d];
        }
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // if (D == 2) {
    //     printf("CPU: %f, %f\n", mean[0], mean[1]);
    // }

    // Subtract data mean
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X[n * D + d] -= mean[d];
        }
    }
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
float TSNE::randn() {
    float x, radius;
    do {
        x = 2 * (rand() / ((float) RAND_MAX + 1)) - 1;
        float y = 2 * (rand() / ((float) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    return x;
}