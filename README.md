# Exploring Parallel T-distributed Stochastic Neighbor Embedding - Midterm Report

[Project proposal](docs/proposal.md)

## Completed work

We thoroughly investigated the tSNE algorithm and the open source implementations using Barnet-Hut approximation from https://github.com/lvdmaaten/bhtsne/ and https://github.com/DmitryUlyanov/Multicore-TSNE. We converted the entire codebase from `double` to `float` in consideration of better vectorization.

We created a testbench using Python and Numpy. This testbench includes 5 test fixtures created from MNIST (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and CLIP (https://github.com/openai/CLIP) model embeddings of various sizes for us to quickly test our implementation as well as benchmarking its performance. 

<table>
  <tr>
   <td>Test case
   </td>
   <td>Source
   </td>
   <td>Number of points
   </td>
   <td>Size of point dimension
   </td>
  </tr>
  <tr>
   <td>simple_200x64
   </td>
   <td rowspan="2" >Scikit Learn digit data
   </td>
   <td>200
   </td>
   <td>64
   </td>
  </tr>
  <tr>
   <td>easy_1797x64
   </td>
   <td>1797
   </td>
   <td>64
   </td>
  </tr>
  <tr>
   <td>medium_10000x768
   </td>
   <td rowspan="2" >MNIST (<a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">https://www.kaggle.com/datasets/oddrationale/mnist-in-csv</a>)
   </td>
   <td>10000
   </td>
   <td>768
   </td>
  </tr>
  <tr>
   <td>hard_60000x768
   </td>
   <td>60000
   </td>
   <td>768
   </td>
  </tr>
  <tr>
   <td>hard_62656x512
   </td>
   <td>CLIP (<a href="https://github.com/openai/CLIP">https://github.com/openai/CLIP</a>) image embeddings of a custom dataset
   </td>
   <td>62656
   </td>
   <td>512
   </td>
  </tr>
</table>

The testbench also defines a single-precision floating-point matrix serialization format with reader/writer implemented in both C++ and Python such that we can easily create new test cases and load the computed t-SNE embeddings into Python for visualization and verification.

We reviewed the codebase and profiled the sequential version of the program to identify parallelizable parts. Using `perf`, we identified two computation-heavy loops suitable for parallelization: (1) computing pairwise distance for K Nearest Neighbors search on a ball tree, and (2) gradient computation. By parallelizing these two tasks using OpenMP and ISPC, we achieved significant speedup. Preliminary results are discussed in a later section.

## Updated schedule

We are following our proposed schedule so far - we have finished most implementations for OpenMP and ISPC. The next step is to produce some benchmark results and start working on CUDA.

| Date            | Plan                                       | Assigned to |
| --------------- | ------------------------------------------ | ----------- |
| Apr 9 - Apr 13  | Benchmark OpenMP and ISPC on PSC machine   | Yue         |
| Apr 13 - Apr 15 | Identify computation for CUDA acceleration | Zhe, Yue    |
| Apr 16 - Apr 22 | Implement CUDA acceleration                | Zhe, Yue    |
| Apr 23 - Apr 25 | Benchmark CUDA on PSC                      | Yue         |
| Apr 25 - Apr 28 | Evaluation; Try IPSC+OpenMP+CUDA together  | Zhe         |
| Apr 29 - May 5  | Report and presentation                    | Zhe, Yue    |

## Poster session expectation

We plan to have the following content at the poster session: problem description, parallelization approaches (OpenMP, ISPC, and CUDA), results and evaluation.

Demo: we are not sure if there will be a demo at this stage. We might include a video demo of the embeddings during gradient descent. 

We will include figures showing the results of our parallelization approaches.

## Preliminary results

OpenMP speedup

|                                                              | 2     | 4     | 8    |
| ------------------------------------------------------------ | ----- | ----- | ---- |
| Step 1 (K Nearest Neighbors on Ball tree)<br />47.9s with 1 thread | 1.99x | 4x    | 6.6x |
| Step 2 (gradient descent)<br />45.6s with 1 thread           | 1.53x | 2.25x | 2.8x |
| Total<br />93.7s with 1 thread                               | 1.74x | 2.8x  | 4x   |

ISPC vectorized Euclidean distance computation

|                                                              | AVX1 8x | AVX2 8x | AVX2 16x |
| ------------------------------------------------------------ | ------- | ------- | -------- |
| Step 1 (K Nearest Neighbors on Ball tree)<br />47.9s with 1 thread and no vectorization | 3.30x   | 3.80x   | 4.02x    |

## Issues and concerns

We have 2 major concerns:

- Our general direction of work so far is to locate potentially parallelizable code in the existing implementation and try to get better speedup using available tools for parallelization. Do we need to research and implement more parallelizable algorithms to replace existing code, such as better kNN search?
- The CUDA version of this program could be challenging to implement. For example, we understand that CUDA doesn’t support tree structures very well. We wonder if there’s any good resources for us to learn more about CUDA.
