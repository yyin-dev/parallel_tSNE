APP_NAME=bhtsne

OBJDIR=objs

OBJS += $(OBJDIR)/splittree.o
OBJS += $(OBJDIR)/tsne_main.o

CUDA_OBJDIR = $(OBJDIR)/cuda
CUDA_OBJS += $(CUDA_OBJDIR)/cuda_utils.o
CUDA_OBJS += $(CUDA_OBJDIR)/reduce_utils.o
CUDA_OBJS += $(CUDA_OBJDIR)/perplexity_search.o

CXX = g++ -m64 -std=c++14
CXXFLAGS = -I. -Iobjs/ -O3 -Wall -Wextra -m64 -std=c++11 -ggdb -Wno-unknown-pragmas
LDFLAGS = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart -lcuda -lcublas
NVCC = nvcc -std=c++14
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -Iinclude/ -I.

default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/
	mkdir -p $(OBJDIR)/cuda

$(APP_NAME): dirs $(OBJS) $(OBJDIR)/tsne.o $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(OBJDIR)/tsne.o $(CUDA_OBJS) $(LDFLAGS)

# OBJS
$(OBJDIR)/%.o: %.cpp *.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

# CUDA_OBJS
$(CUDA_OBJDIR)/%.o: kernels/%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

# tsne.o
$(OBJDIR)/tsne.o: tsne.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@ 

clean:
	/bin/rm -rf *~ $(OBJDIR) $(APP_NAME)
