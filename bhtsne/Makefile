APP_NAME=bhtsne

OBJS += splittree.o
OBJS += tsne_main.o
OBJS += tsne.o

CXX = g++
CXXFLAGS = -I. -O3 -Wall -Wextra -m64 -std=c++11 -ggdb -fopenmp -Wno-unknown-pragmas

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp %.h vptree.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME)