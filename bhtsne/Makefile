APP_NAME=bhtsne

OBJDIR=objs

OBJS += $(OBJDIR)/splittree.o
OBJS += $(OBJDIR)/tsne_main.o
OBJS += $(OBJDIR)/tsne.o

CUDA_OBJDIR = $(OBJDIR)/cuda
CUDA_OBJS += $(CUDA_OBJDIR)/bhtree.o
CUDA_OBJS += $(CUDA_OBJDIR)/kernels.o

NVCC = nvcc -std=c++11
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -Iinclude/ -I.
CXX = g++ -m64
CXXFLAGS = -I. -Iobjs/ -O3 -Wall -Wextra -m64 -std=c++11 -ggdb -fopenmp -Wno-unknown-pragmas
LDFLAGS = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart

default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/
	mkdir -p $(CUDA_OBJDIR)/

$(APP_NAME): dirs $(OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp *.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(CUDA_OBJDIR)/bhtree.o: bhtree.cpp bhtree.h 
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(CUDA_OBJDIR)/kernels.o: kernels.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ $(OBJDIR) $(APP_NAME)

# Run on easy input
easy:
	make clean && make && ./bhtsne -f ../../inputs/easy_1797x64.bin

easy:
	make clean && make && ./bhtsne -f ../../inputs/medium_10000x784.bin -n 8