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
  const int randSeed = getOptionInt("-r", 15618);
  const int reducedDim = getOptionInt("-d", 2);
  // original default args
  const int maxIter = getOptionInt("-i", 1000);
  const float perplexity = getOptionFloat("-p", 50.f);
  const float theta = getOptionFloat("-t", 0.5f);

  assert(inputFile != nullptr && "Please specify input file");

  // Define some variables
  int dataN, dataDim;
  float *data;

  // load dataset
  bool dataLoaded = TSNE::loadData(inputFile, &data, &dataN, &dataDim);

  assert(dataLoaded);

  // set up timer
  auto compute_start = Clock::now();
  double compute_time = 0;

  // Now fire up the SNE implementation
  float* dimReducedData = (float*) malloc(dataN * reducedDim * sizeof(float));
  TSNE::run(data, dataN, dataDim, dimReducedData,
            reducedDim, perplexity, theta, randSeed, false, maxIter, 250, 250);

  compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
  printf("Computation Time: %lf.\n", compute_time);

  // save result to file
  char* cleanFileName = getOutputFileName(inputFile);
  TSNE::saveData(cleanFileName, dimReducedData, dataN, reducedDim);
  free(cleanFileName);

  // Clean up the memory
  free(data); data = NULL;
  free(dimReducedData); dimReducedData = NULL;
}
