CXX = mpicxx
CXXFLAGS= -O3 -std=c++17 
INCLUDES = -I.

USE_MPL=1
ifeq ($(USE_MPL),1)
MPL_PATH = $(HOME)/builds/Exhanced-mpl-tiny
CXXFLAGS += -DUSE_MPL_CXX -I$(MPL_PATH)/include
endif

TESTS := $(wildcard *.cpp)
TESTS := $(filter-out osu_pt2pt.cpp, $(TESTS))
TESTS := $(filter-out osu_mbw_mr.cpp, $(TESTS))
EXES := $(TESTS:.cpp=.x)

.PHONY: clean all common osu_mbw_mr.x

all: common $(EXES) osu_mbw_mr.x

.SUFFIXES: .x

common:
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c osu_pt2pt.cpp

%.x: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ $^ osu_pt2pt.o

osu_mbw_mr.x: osu_mbw_mr.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(EXES) osu_mbw_mr.x osu_mbw_mr.x.dSYM *.o *.dSYM
