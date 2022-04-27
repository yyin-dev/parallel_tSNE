/*
CUDA BarnesHut v2.2: Simulation of the gravitational forces
in a galactic cluster using the Barnes-Hut n-body algorithm

Copyright (c) 2011, Texas State University-San Marcos.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University-San Marcos nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>


// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR1 1
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 8
#define FACTOR6 1

#define WARPSIZE 32
#define MAXDEPTH 32

#define DEBGREE 4

// TODO:
// [X] Octree -> Quadtree
// [] sumQ
// [] CPU (+) + GPU (-) integration 
// [] ISPC & OpenMP + CUDA
// [] Perf analysis (after 30th)


/******************************************************************************/

// With octree (8 children):
// childd is aliased with accxd, accyd, and sortd 
// but they never use the same memory locations.
// Explanation:
// Let P be the number of bodies/points, N be the number of nodes.
// N >= 2P. Nodes will be allocated in range [P, N] (see AtomicSub in 
// TreeBuildingKernel) and will be accessed with 8*index+j. 
// With index >= P, 8*index+j >= 8P.
// On the other hand, sortd starts at index < 6*(P + WARPSIZE) and ends at 
// 7*(P + WARPSIZE). As long as P >= 7*WARPSIZE, the two ranges won't collide.
//
// With quadtree (4 children):
// 4*index+j >= 4P.
// By default, accxd starts at 3*inc, accyd starts at 4*inc, sortd starts at 
// 6*inc. We can shift the offsets such that accxd starts at 0, accyd starts
// at inc, and sortd starts at 2*inc. This way, sortd ends at 3*(P+WARPSIZE).
// 3*(P + WARPSIZE) <= 4P holds as long as P >= 3*WARPSIZE.

__constant__ int nnodesd, nbodiesd;
__constant__ float dtimed, dthfd, itolsqd;
__constant__ volatile float *massd, *posxd, *posyd, *accxd, *accyd;
__constant__ volatile float *maxxd, *maxyd, *minxd, *minyd;
__constant__ volatile int *errd, *sortd, *childd, *countd, *startd;

__device__ volatile int stepd, bottomd, maxdepthd, blkcntd;
__device__ volatile float radiusd;


/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernel()
{
  *errd = 0;
  stepd = -1;
  maxdepthd = 1;
  blkcntd = 0;
}


/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
// __launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel()
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = min(minx, val);
    maxx = max(maxx, val);
    val = posyd[j];
    miny = min(miny, val);
    maxy = max(maxy, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = min(minx, sminx[k]);
      smaxx[i] = maxx = max(maxx, smaxx[k]);
      sminy[i] = miny = min(miny, sminy[k]);
      smaxy[i] = maxy = max(maxy, smaxy[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;

    inc = gridDim.x - 1;
    if (inc == atomicInc((unsigned int *)&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = min(minx, minxd[j]);
        maxx = max(maxx, maxxd[j]);
        miny = min(miny, minyd[j]);
        maxy = max(maxy, maxyd[j]);
      }

      // compute 'radius'
      val = max(maxx - minx, maxy - miny);
      radiusd = val * 0.5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      k *= DEBGREE;
      for (i = 0; i < DEBGREE; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}

/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
// __launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel()
{
  register int i, j, k, depth, localmaxdepth, skip, inc;
  register float x, y, r;
  register float px, py;
  register int ch, n, cell, locked, patch;
  register float radius, rootx, rooty;

  // cache root data
  radius = radiusd;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];

  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      n = nnodesd;
      depth = 1;
      r = radius;
      j = 0;
      // determine which child to follow
      if (rootx < px) j = 1;
      if (rooty < py) j += 2;
    }

    // follow path to leaf cell
    ch = childd[n*DEBGREE+j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      j = 0;
      // determine which child to follow
      if (posxd[n] < px) j = 1;
      if (posyd[n] < py) j += 2;
      ch = childd[n*DEBGREE+j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*DEBGREE+j;
      if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
        if (ch == -1) {
          // if null, just insert the new body
          childd[locked] = i;
        } else {  // there already is a body in this position
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              *errd = 1;
              bottomd = nnodesd;
            }
            patch = max(patch, cell);

            x = (j & 1) * r;
            y = ((j >> 1) & 1) * r;
            r *= 0.5f;

            massd[cell] = -1.0f;
            startd[cell] = -1;
            x = posxd[cell] = posxd[n] - r + x;
            y = posyd[cell] = posyd[n] - r + y;
            for (k = 0; k < DEBGREE; k++) childd[cell*DEBGREE+k] = -1;

            if (patch != cell) { 
              childd[n*DEBGREE+j] = cell;
            }

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j += 2;
            childd[cell*DEBGREE+j] = ch;

            n = cell;
            j = 0;
            if (x < px) j = 1;
            if (y < py) j += 2;

            ch = childd[n*DEBGREE+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*DEBGREE+j] = i;
          __threadfence();  // push out subtree
          childd[locked] = patch;
        }

        localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        skip = 1;
      }
    }
    __syncthreads();  // throttle
  }
  
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
// __launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel()
{
  register int i, j, k, ch, inc, missing, cnt, bottom;
  register float m, cm, px, py;
  __shared__ volatile int child[THREADS3 * DEBGREE];

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  missing = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (missing == 0) {
      // new cell, so initialize
      cm = 0.0f;
      px = 0.0f;
      py = 0.0f;
      cnt = 0;
      j = 0;
      for (i = 0; i < DEBGREE; i++) {
        ch = childd[k*DEBGREE+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*DEBGREE+i] = -1;
            childd[k*DEBGREE+j] = ch;
          }
          child[missing*THREADS3+threadIdx.x] = ch;  // cache missing children
          m = massd[ch];
          missing++;
          if (m >= 0.0f) {
            // child is ready
            missing--;
            if (ch >= nbodiesd) {  // count bodies (needed later)
              cnt += countd[ch] - 1;
            }
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
          }
          j++;
        }
      }
      cnt += j;
    }

    if (missing != 0) {
      do {
        // poll missing child
        ch = child[(missing-1)*THREADS3+threadIdx.x];
        m = massd[ch];
        if (m >= 0.0f) {
          // child is now ready
          missing--;
          if (ch >= nbodiesd) {
            // count bodies (needed later)
            cnt += countd[ch] - 1;
          }
          // add child's contribution
          cm += m;
          px += posxd[ch] * m;
          py += posyd[ch] * m;
        }
        // repeat until we are done or child is not ready
      } while ((m >= 0.0f) && (missing != 0));
    }

    if (missing == 0) {
      // all children are ready, so store computed information
      countd[k] = cnt;
      m = 1.0f / cm;
      posxd[k] = px * m;
      posyd[k] = py * m;
      __threadfence();  // make sure data are visible before setting mass
      massd[k] = cm;
      k += inc;  // move on to next cell
    }
  }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
// __launch_bounds__(THREADS4, FACTOR4)
void SortKernel()
{
  register int i, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      for (i = 0; i < DEBGREE; i++) {
        ch = childd[k*DEBGREE+i];
        if (ch >= nbodiesd) {
          // child is a cell
          startd[ch] = start;  // set start ID of child
          start += countd[ch];  // add #bodies in subtree
        } else if (ch >= 0) {
          // child is a body
          sortd[start] = ch;  // record body in 'sorted' array
          start++;
        }
      }
      k -= dec;  // move on to next cell
    }
    __syncthreads();  // throttle
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
// __launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel()
{
  register int i, j, k, n, depth, base, sbase, diff, t;
  register float px, py, ax, ay, dx, dy, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

  if (0 == threadIdx.x) {
    tmp = radiusd;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < maxdepthd; i++) {
      dq[i] = dq[i - 1] * 0.25f;
    }

    if (maxdepthd > MAXDEPTH) {
      *errd = maxdepthd;
    }
  }
  __syncthreads();

  if (maxdepthd <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();

    // iterate over all bodies assigned to thread
    for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
      // cache position info
      px = posxd[i];
      py = posyd[i];

      ax = 0.0f;
      ay = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = j;
      if (sbase == threadIdx.x) {
        node[j] = nnodesd;
        pos[j] = 0;
      }

      while (depth >= j) {
        // stack is not empty
        while ((t = pos[depth]) < DEBGREE) {
          // node on top of stack has more children to process
          n = childd[node[depth]*DEBGREE+t];  // load child pointer
          if (sbase == threadIdx.x) {
            // I'm the first thread in the warp
            pos[depth] = t + 1;
          }
          if (n >= 0) {
            dx = px - posxd[n];
            dy = py - posyd[n] ;
            tmp = dx*dx + dy*dy;  // compute distance squared (plus softening)
            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
              tmp = 1 / (1 + tmp);
              tmp = massd[n] * tmp * tmp;
              ax += dx * tmp;
              ay += dy * tmp;
            } else {
              // push cell onto stack
              depth++;
              if (sbase == threadIdx.x) {
                node[depth] = n;
                pos[depth] = 0;
              }
            }
          } else {
            depth = max(j, depth - 1);  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      }

      // save computed acceleration
      accxd[i] = ax;
      accyd[i] = ay;
    }
  }
}



/******************************************************************************/

static void CudaTest(const char *msg)
{
  cudaError_t e;

  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}



/******************************************************************************/

int init(float* points, int num_points) {
  int i, blocks;
  int nnodes, nbodies;
  int error;
  float dtime, dthf, itolsq;
  float time, timing[7];
  cudaEvent_t start, stop;
  float *mass, *posx, *posy;

  int *errl, *sortl, *childl, *countl, *startl;
  float *massl;
  float *posxl, *posyl;
  float *accxl, *accyl;
  float *maxxl, *maxyl;
  float *minxl, *minyl;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable device\n");
    exit(-1);
  }
  if (deviceProp.major < 2) {
    fprintf(stderr, "Need at least compute capability 2.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }

  // blocks = deviceProp.multiProcessorCount;
  blocks = 32;
  fprintf(stderr, "blocks = %d\n", blocks);

  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
  if (MAXDEPTH > WARPSIZE) {
    fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
    exit(-1);
  }
  if ((THREADS1 <= 0) || (THREADS1 & (THREADS1-1) != 0)) {
    fprintf(stderr, "THREADS1 must be greater than zero and a power of two\n");
    exit(-1);
  }

  cudaGetLastError();  // reset error value
  for (i = 0; i < 7; i++) timing[i] = 0.0f;

  nbodies = num_points;
  printf("nbodies: %d\n", nbodies);

  nnodes = nbodies * 2;
  if (nnodes < 1024*blocks) nnodes = 1024*blocks;
  while ((nnodes & (WARPSIZE-1)) != 0) nnodes++; // nnodes & WARPSIZE-1 == 0
  nnodes--;
  printf("nnodes: %d\n", nnodes);

//   timesteps = atoi(argv[2]);
  dtime = 0.025;  dthf = dtime * 0.5f;
  itolsq = 1.0f / (0.5 * 0.5);

  // allocate memory
  mass = (float *)malloc(sizeof(float) * nbodies);
  if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  posx = (float *)malloc(sizeof(float) * nbodies);
  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy = (float *)malloc(sizeof(float) * nbodies);
  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
 
  for (int i = 0; i < nbodies; i++) {
    mass[i] = 1.0;
    posx[i] = points[2 * i];
    posy[i] = points[2 * i + 1];
  }

  if (cudaSuccess != cudaMalloc((void **)&errl, sizeof(int))) fprintf(stderr, "could not allocate errd\n");  CudaTest("couldn't allocate errd");
  if (cudaSuccess != cudaMalloc((void **)&childl, sizeof(int) * (nnodes+1) * DEBGREE)) fprintf(stderr, "could not allocate childd\n");  CudaTest("couldn't allocate childd");
  if (cudaSuccess != cudaMalloc((void **)&massl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate massd\n");  CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMalloc((void **)&posxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posxd\n");  CudaTest("couldn't allocate posxd");
  if (cudaSuccess != cudaMalloc((void **)&posyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posyd\n");  CudaTest("couldn't allocate posyd");
  if (cudaSuccess != cudaMalloc((void **)&countl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate countd\n");  CudaTest("couldn't allocate countd");
  if (cudaSuccess != cudaMalloc((void **)&startl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate startd\n");  CudaTest("couldn't allocate startd");

  // alias arrays
  int inc = (nbodies + WARPSIZE - 1) / WARPSIZE * WARPSIZE;

  accxl = (float *)childl;
  accyl = (float *)&childl[inc];
  sortl = (int *)&childl[2*inc];


  if (cudaSuccess != cudaMalloc((void **)&maxyl, sizeof(float) * blocks)) fprintf(stderr, "could not allocate maxyd\n");  CudaTest("couldn't allocate maxyd");
  if (cudaSuccess != cudaMalloc((void **)&maxxl, sizeof(float) * blocks)) fprintf(stderr, "could not allocate maxxd\n");  CudaTest("couldn't allocate maxxd");
  if (cudaSuccess != cudaMalloc((void **)&minxl, sizeof(float) * blocks)) fprintf(stderr, "could not allocate minxd\n");  CudaTest("couldn't allocate minxd");
  if (cudaSuccess != cudaMalloc((void **)&minyl, sizeof(float) * blocks)) fprintf(stderr, "could not allocate minyd\n");  CudaTest("couldn't allocate minyd");

  // copy symbol (__constant__)
  if (cudaSuccess != cudaMemcpyToSymbol(nnodesd, &nnodes, sizeof(int))) fprintf(stderr, "copying of nnodes to device failed\n");  CudaTest("nnode copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int))) fprintf(stderr, "copying of nbodies to device failed\n");  CudaTest("nbody copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(errd, &errl, sizeof(void*))) fprintf(stderr, "copying of err to device failed\n");  CudaTest("err copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float))) fprintf(stderr, "copying of dtime to device failed\n");  CudaTest("dtime copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float))) fprintf(stderr, "copying of dthf to device failed\n");  CudaTest("dthf copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float))) fprintf(stderr, "copying of itolsq to device failed\n");  CudaTest("itolsq copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(sortd, &sortl, sizeof(void*))) fprintf(stderr, "copying of sortl to device failed\n");  CudaTest("sortl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(countd, &countl, sizeof(void*))) fprintf(stderr, "copying of countl to device failed\n");  CudaTest("countl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(startd, &startl, sizeof(void*))) fprintf(stderr, "copying of startl to device failed\n");  CudaTest("startl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(childd, &childl, sizeof(void*))) fprintf(stderr, "copying of childl to device failed\n");  CudaTest("childl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(massd, &massl, sizeof(void*))) fprintf(stderr, "copying of massl to device failed\n");  CudaTest("massl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(posxd, &posxl, sizeof(void*))) fprintf(stderr, "copying of posxl to device failed\n");  CudaTest("posxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(posyd, &posyl, sizeof(void*))) fprintf(stderr, "copying of posyl to device failed\n");  CudaTest("posyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(accxd, &accxl, sizeof(void*))) fprintf(stderr, "copying of accxl to device failed\n");  CudaTest("accxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(accyd, &accyl, sizeof(void*))) fprintf(stderr, "copying of accyl to device failed\n");  CudaTest("accyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(maxxd, &maxxl, sizeof(void*))) fprintf(stderr, "copying of maxxl to device failed\n");  CudaTest("maxxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(maxyd, &maxyl, sizeof(void*))) fprintf(stderr, "copying of maxyl to device failed\n");  CudaTest("maxyl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(minxd, &minxl, sizeof(void*))) fprintf(stderr, "copying of minxl to device failed\n");  CudaTest("minxl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(minyd, &minyl, sizeof(void*))) fprintf(stderr, "copying of minyl to device failed\n");  CudaTest("minyl copy to device failed");
  
  // Copy data
  if (cudaSuccess != cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of mass to device failed\n");  CudaTest("mass copy to device failed");
  if (cudaSuccess != cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to device failed\n");  CudaTest("posx copy to device failed");
  if (cudaSuccess != cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to device failed\n");  CudaTest("posy copy to device failed");

  // run timesteps (launch GPU kernels)
  cudaEventCreate(&start);  cudaEventCreate(&stop);  
  cudaEventRecord(start, 0);
  InitializationKernel<<<1, 1>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[0] += time;
  CudaTest("kernel 0 launch failed");

  //////
  cudaEventRecord(start, 0);
  BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[1] += time;
  CudaTest("kernel 1 launch failed");

  cudaEventRecord(start, 0);
  TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[2] += time;
  CudaTest("kernel 2 launch failed");

  cudaEventRecord(start, 0);
  SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[3] += time;
  CudaTest("kernel 3 launch failed");

  cudaEventRecord(start, 0);
  SortKernel<<<blocks * FACTOR4, THREADS4>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[4] += time;
  CudaTest("kernel 4 launch failed");

  cudaEventRecord(start, 0);
  ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>();
  cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
  timing[5] += time;
  CudaTest("kernel 5 launch failed");
  ///////

  CudaTest("kernel launch failed");
  cudaEventDestroy(start);  cudaEventDestroy(stop);

  // transfer result back to CPU
  if (cudaSuccess != cudaMemcpy(&error, errl, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from device failed\n");  CudaTest("err copy from device failed");
  if (cudaSuccess != cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posx from device failed\n");  CudaTest("posx copy from device failed");
  if (cudaSuccess != cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posy from device failed\n");  CudaTest("posy copy from device failed");

  float *accx, *accy;
  accx = (float *)malloc(sizeof(float) * nbodies);
  accy = (float *)malloc(sizeof(float) * nbodies);
  if (cudaSuccess != cudaMemcpy(accx, accxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of accx from device failed\n");  CudaTest("accx copy from device failed");
  if (cudaSuccess != cudaMemcpy(accy, accyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of accy from device failed\n");  CudaTest("accy copy from device failed");

  // print output
  // for (i = 0; i < nbodies; i++) {
  printf("GPU: \n");
  for (int j = 0; j < 10; j++) {
    printf("%.2e %.2e\n", accx[j], accy[j]);
  }
  // }

  free(mass);
  free(posx);
  free(posy);

  cudaFree(errl);
  cudaFree(childl);
  cudaFree(massl);
  cudaFree(posxl);
  cudaFree(posyl);
  cudaFree(countl);
  cudaFree(startl);

  cudaFree(maxxl);
  cudaFree(maxyl);
  cudaFree(minxl);
  cudaFree(minyl);

  return 0;
}
