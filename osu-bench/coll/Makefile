CXX = CC
CXXFLAGS= -O3 -std=c++17 
INCLUDES = -I.

TESTS := $(wildcard *.cpp)
TESTS := $(filter-out osu_coll.cpp, $(TESTS))
EXES := $(TESTS:.cpp=.x)

USE_MPL=0
ifeq ($(USE_MPL),1)
MPL_PATH = $(HOME)/builds/Exhanced-mpl-tiny
CXXFLAGS += -DUSE_MPL_CXX -I$(MPL_PATH)/include
endif
USE_MPI_ALLTOALLW=0
ifeq ($(USE_MPI_ALLTOALLW),1)
CXXFLAGS += -DUSE_ALLTOALLW
endif

.PHONY: clean all common

all: common $(EXES)

.SUFFIXES: .x

common:
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c osu_coll.cpp

%.x: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ $^ osu_coll.o

clean:
	rm -rf $(EXES) *.o *.dSYM
