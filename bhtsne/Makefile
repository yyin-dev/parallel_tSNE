APP_NAME=bhtsne

OBJDIR=objs

OBJS += $(OBJDIR)/distance_ispc.o
OBJS += $(OBJDIR)/gradient_ispc.o
OBJS += $(OBJDIR)/splittree.o
OBJS += $(OBJDIR)/tsne_main.o
OBJS += $(OBJDIR)/tsne.o

CUDA_OBJDIR = $(OBJDIR)/cuda
CUDA_OBJS += $(CUDA_OBJDIR)/bhtree.o
CUDA_OBJS += $(CUDA_OBJDIR)/bhtree_kernels.o

# CUDA
NVCC = nvcc -std=c++11
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -Iinclude/ -I.

# ISPC
ISPC=ispc
# note: change target to sse4 for SSE capable machines
ISPCFLAGS=-O2 --target=avx2-i32x16 --arch=x86-64

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

# CUDA
$(CUDA_OBJDIR)/bhtree.o: bhtree.cpp bhtree.h
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(CUDA_OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

# ISPC
$(OBJDIR)/%_ispc.h $(OBJDIR)/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

clean:
	/bin/rm -rf *~ $(OBJDIR) $(APP_NAME)

# Run on easy input
easy:
	make clean && make && ./bhtsne -f ../../inputs/easy_1797x64.bin

medium:
	make clean && make && ./bhtsne -f ../../inputs/medium_10000x784.bin -n 8

hard:
	make clean && make && ./bhtsne -f ../../inputs/hard_62656x512.bin -n 8