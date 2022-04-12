APP_NAME=bhtsne

OBJDIR=objs

OBJS += $(OBJDIR)/splittree.o
OBJS += $(OBJDIR)/tsne_main.o
OBJS += $(OBJDIR)/tsne.o

CXX = g++ -m64
CXXFLAGS = -I. -Iobjs/ -O3 -Wall -Wextra -m64 -std=c++11 -ggdb -fopenmp -Wno-unknown-pragmas

default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: %.cpp *.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ $(OBJDIR) $(APP_NAME)