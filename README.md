# T-distributed Stochastic Neighbor Embedding

## Summary

We are going to explore parallelizing t-distributed stochastic neighbor embedding (t-SNE) with Barnes-Hut approximation. t-SNE is a dimensionality reduction technique commonly used for high-dimensional data visualization.

## Background

t-SNE is a dimensionality reduction technique for data visualization that keeps similar data points close in the low dimensional embedding. There are three major steps for t-SNE: (1) construct a probability distribution between all pairs of points in the high-dimensional input, (2) define a similar probability distribution between all pairs of points in the low-dimensional embedding, (3) compute the Kullback-Leibler divergence (KL divergence) between the two distributions and use gradient descent to minimize the KL divergence.

The probability distribution between pairs of points in the high-dimensional input is given by a Gaussian distribution. In the embedding space, the Gaussian distribution is replaced with the Student’s t-distribution. 

A brief description of the t-SNE algorithm with Barnes-Hut is as follows:

- Preprocessing

  - Build a ball tree of nearest neighbors using all N high dimensional data points

  - For each data point, iteratively fit a Gaussian kernel on the Euclidean distance to its K nearest neighbors to compute a similarity value

- Iterative fitting

  - Build a quadtree of a randomly initialized low-dimensional solution based on Euclidean distance.
  - Estimate positive forces by iterating through all edges in the tree
  - Estimate negative forces by computing the cluster distances to the center of mass around each point
  - Estimate gradients using positive and negative forces for each value in the low-dimensional solution and update the solution using gradient descent with momentum

Multiple aspects of this algorithm can be parallelized.

- Gaussian kernel fitting is done point wise. 
- This can be parallelized through SIMD or multithreadingQuadtree and Ball tree construction can be parallelized through multithreading
- Gradient estimation and updates can be parallelized through SIMD or multithreading

## Challenges

### Workload

- Memory access when building and accessing the tree can be unpredictable, leading to more cache misses
- Workload imbalance can happen during iterative kernel Gaussian fitting
- High dimensional data can lead to excessive cache misses during pointwise processing

### Constraints

- The iterative gradient descent process has to be executed sequentially.
- Building spatial-partitioned trees may not be fully parallelizable

Working on this project, we can gain a deeper understanding of parallelization by applying what we learned in class to real-world applications.

## Resources

We plan to use C++ as the programming language and will use OpenMP, ISPC and CUDA to parallelize our code. We will start from the reference Barnes-hut t-SNE implementation in https://github.com/lvdmaaten/bhtsne. After fully understanding the reference solution, we will attempt to parallelize the program. We will mainly use GHC machines for testing our parallelized versions.

We made these choices because the GHC cluster provides a reasonable CPU and GPU for us to benchmark our programs without getting too much resource contention. The languages we chose represent the two most common paradigms of parallelizing workloads: multithreading, SIMD and GPU Parallelism.

## Goals and deliverables

### 75% Goal

- Parallelize t-SNE using ISPC and OpenMP
- Compare and contrast their performance on GHC machines

### 100% Goal

- Parallelize t-SNE using CUDA
- Compare and contrast CUDA version performance on GHC machines

### 125% Goal

- Fully optimize our programs and compare its performance with existing parallelized implementations
- Understand the benefits of more advanced versions of t-SNE, such as Flt-SNE in terms of parallelizability

## Schedule

| Date            | Plan                                                         |
| --------------- | ------------------------------------------------------------ |
| Mar 23 - Apr 1  | Understand t-SNE source code                                 |
| Apr 2 - Apr 8   | Parallelize t-SNE using ISPC and OpenMP                      |
| Apr 9 - Apr 15  | Improve the parallized version and evaluate training/inference time on MNIST dataset. Complete Milestone Report (11th) |
| Apr 16 - Apr 22 | Parallelize t-SNE using CUDA                                 |
| Apr 23 - Apr 28 | Perform analysis and benchmarks on finalized parallel versions using MNIST and other datasets |
| Apr 29 - May 5  | Finish report and presentation                               |

## Reference

https://towardsdatascience.com/t-distributed-stochastic-neighbor-embedding-t-sne-bb60ff109561

https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

https://lvdmaaten.github.io/tsne/

https://github.com/lvdmaaten/bhtsne/blob/master/tsne.cpp



