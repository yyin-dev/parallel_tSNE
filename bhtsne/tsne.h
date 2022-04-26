/*
 *  tsne.h
 *  Header file for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#ifndef TSNE_H
#define TSNE_H


static inline float sign(float x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class TSNE
{
public:
    void run(float* X, int N, int D, float* Y,
               int no_dims = 2, float perplexity = 30, float theta = .5,
               int num_threads = 1, int max_iter = 1000, int n_iter_early_exag = 250,
               int random_state = 0, bool init_from_Y = false, int verbose = 0,
               float early_exaggeration = 12, float learning_rate = 200,
               float *final_error = NULL);
    void symmetrizeMatrix(int** row_P, int** col_P, float** val_P, int N);
private:
    float computeGradient(int* inp_row_P, int* inp_col_P, float* inp_val_P, float* Y, int N, int D, float* dC, float theta, bool eval_error);
    float evaluateError(int* row_P, int* col_P, float* val_P, float* Y, int N, int no_dims, float theta);
    void zeroMean(float* X, int N, int D);
    void computeGaussianPerplexity(float* X, int N, int D, int** _row_P, int** _col_P, float** _val_P, float perplexity, int K, int verbose);
    float randn();
};

#endif
