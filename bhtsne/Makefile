APP_NAME=bhtsne

OBJDIR=objs

OBJS += $(OBJDIR)/distance_ispc.o
OBJS += $(OBJDIR)/splittree.o
OBJS += $(OBJDIR)/tsne_main.o
OBJS += $(OBJDIR)/tsne.o

ISPC=ispc
# note: change target to sse4 for SSE capable machines
ISPCFLAGS=-O2 --target=avx2-i32x16 --arch=x86-64

CXX = g++ -m64
CXXFLAGS = -I. -Iobjs/ -O3 -Wall -Wextra -m64 -std=c++11 -ggdb -fopenmp -Wno-unknown-pragmas

default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: %.cpp *.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%_ispc.h $(OBJDIR)/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

clean:
	/bin/rm -rf *~ $(OBJDIR) $(APP_NAME)