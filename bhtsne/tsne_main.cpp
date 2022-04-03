#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cassert>
#include <chrono>

#include "tsne.h"

using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

// Ref: 15-618 HW3 CLI Option Parser
static int _argc;
static const char **_argv;

const char *getOptionString(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0) return _argv[i + 1];
    return default_value;
}

int getOptionInt(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0) return atoi(_argv[i + 1]);
    return default_value;
}

float getOptionFloat(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0) return (float)atof(_argv[i + 1]);
    return default_value;
}

/**
 * @brief Get the Output File Name
 * 
 * @param inputFilePath 
 * @return char* 
 */
char* getOutputFileName(const char* inputFilePath) {
  int inputFilePathLen = strlen(inputFilePath);

  // find the last directory slash and exclude it from file name
  for (int i = inputFilePathLen - 1; i >= 0; i--) {
      if (inputFilePath[i] == '/') {
          inputFilePathLen -= i + 1;
          inputFilePath += i + 1;
          break;
      }
  }

  // truncate the trailing .bin suffix
  inputFilePathLen -= 4;
  char *cleanInputFileName = (char *)malloc(inputFilePathLen + 1);
  strncpy(cleanInputFileName, inputFilePath, inputFilePathLen);
  cleanInputFileName[inputFilePathLen] = '\0';

  return cleanInputFileName;
}

// Function that runs the Barnes-Hut implementation of t-SNE
int main(int argc, const char *argv[]) {
  // parse CLI args
  _argc = argc - 1;
  _argv = argv + 1;

  const char *inputFile = getOptionString("-f", nullptr);

  assert(inputFile != nullptr && "Please specify input file");

  // Define some variables
  int dataN, dataDim;
  float *data;
  int rand_seed = 15618;

  bool dataLoaded = TSNE::load_data(inputFile, &data, &dataN, &dataDim);

  // set up timer
  auto compute_start = Clock::now();
  double compute_time = 0;

    // Read the parameters and the dataset
  // if(dataLoaded) {

  //   // Now fire up the SNE implementation
  //   double* Y = (double*) malloc(N * no_dims * sizeof(double));
  //   double* costs = (double*) calloc(N, sizeof(double));
  //       if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  //   TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 250, 250);

  //   // Save the results
  //   TSNE::save_data(Y, landmarks, costs, N, no_dims);

  //   // Clean up the memory
  //   free(data); data = NULL;
  //   free(Y); Y = NULL;
  //   free(costs); costs = NULL;
  //   }

  compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
  printf("Computation Time: %lf.\n", compute_time);
}
