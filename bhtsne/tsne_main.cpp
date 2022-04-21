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
typedef std::chrono::duration<float> dsec;

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

// Function that loads data from our custom binary file
// Note: this function does a malloc that should be freed elsewhere
bool loadData(const char* fileName, float** data, int* dataN, int* dataDim) {

  // Open file, read first 2 integers, allocate memory, and read the data
  FILE *file;
  if((file = fopen(fileName, "r+b")) == NULL) {
    printf("Error: could not open data file: %s.\n", fileName);
    return false;
  }
  // number of datapoints
  fread(dataN, sizeof(int), 1, file); 
  // original dimensionality
  fread(dataDim, sizeof(int), 1, file);
  *data = (float*) malloc(*dataDim * *dataN * sizeof(float));
  if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  // read the data
  fread(*data, sizeof(float), *dataN * *dataDim, file);
  fclose(file);
  printf("Read %i x %i data matrix successfully!\n", *dataN, *dataDim);
  return true;
}

// Function that saves map to our custom binary file
void saveData(const char* fileName, const char* outDir, float* data, int dataN, int dataDim, int numThreads) {
  int fileNameLen = strlen(fileName);
  int outDirLen = strlen(outDir);
  char *outFilePath = (char *)malloc(fileNameLen + 28 + outDirLen);
  sprintf(outFilePath, "%s/tsne_%dd_%s_%d.bin", outDir, dataDim, fileName, numThreads);

  // Open file, write first 2 integers and then the data
  FILE *file;
  if((file = fopen(outFilePath, "w+b")) == NULL) {
    printf("Error: could not open data file: %s\n", outFilePath);
    return;
  }
  fwrite(&dataN, sizeof(int), 1, file);
  fwrite(&dataDim, sizeof(int), 1, file);
  fwrite(data, sizeof(float), dataN * dataDim, file);
  fclose(file);
  free(outFilePath);
  printf("Wrote %i x %i data matrix successfully!\n", dataN, dataDim);
}

// t-SNE runner
int main(int argc, const char *argv[]) {
  // parse CLI args
  _argc = argc - 1;
  _argv = argv + 1;

  const char *inputFile = getOptionString("-f", nullptr);
  const char *outputDir = getOptionString("-od", "../output");
  const int randSeed = getOptionInt("-r", 15618);
  const int reducedDim = getOptionInt("-d", 2);
  const int numThreads = getOptionInt("-n", 1);
  const int verbose = getOptionInt("-v", 1);
  // original default args
  const int maxIter = getOptionInt("-i", 1000);
  const float perplexity = getOptionFloat("-p", 50.f);
  const float theta = getOptionFloat("-t", 0.5f);

  assert(inputFile != nullptr && "Please specify input file");

  // Define some variables
  int dataN, dataDim;
  float *data;

  // load dataset
  bool dataLoaded = loadData(inputFile, &data, &dataN, &dataDim);

  assert(dataLoaded);

  // set up
  float* dimReducedData = (float*) malloc(dataN * reducedDim * sizeof(float));
  auto compute_start = Clock::now();
  float compute_time = 0;
  TSNE TSNERunner;

  // Now fire up the SNE implementation
  TSNERunner.run(data, dataN, dataDim, dimReducedData,
            reducedDim, perplexity, theta, numThreads, maxIter, 250, randSeed, false, verbose);

  compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
  printf("Computation Time: %.4f seconds.\n", compute_time);

  // save result to file
  char* cleanFileName = getOutputFileName(inputFile);
  saveData(cleanFileName, outputDir, dimReducedData, dataN, reducedDim, numThreads);
  free(cleanFileName);

  // Clean up the memory
  free(data); data = NULL;
  free(dimReducedData); dimReducedData = NULL;
}
