APP_NAME=bhtsne

OBJS += splittree.o
OBJS += tsne_main.o
OBJS += tsne.o

CXX = g++
CXXFLAGS = -I. -O3 -Wall -Wextra -std=c++11 -ggdb

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME)