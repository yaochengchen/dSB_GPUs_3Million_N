# dsb-gpu -- GH200 / H100 (sm_90) only.
#
#   make            build both solvers, dense_scaling, selftest, bench and bit_dense_multi
#   make probe      build ./gpu_probe and ./c2c_probe
#   make ARCH=sm_90a   (if you need the arch-specific variant)
#
# Requires CUDA 12.0+ (thread block clusters) and Eigen 3.

CXX       ?= g++
NVCC      ?= nvcc
ARCH      ?= sm_90
EIGEN_INC ?= /usr/include/eigen3
CUDA_HOME ?= /usr/local/cuda

INCLUDES  := -Iinclude -Ithird_party/fasthare -I$(EIGEN_INC) -I$(CUDA_HOME)/include
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra $(INCLUDES)
NVCCFLAGS := -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr \
             -arch=$(ARCH) $(INCLUDES)
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcudart -lcublas

CU_OBJS  := src/kernels.o src/solver.o src/sparse_solver.o src/gemm.o \
            src/dense_capacity.o src/bitfused.o
CXX_OBJS := src/sparse.o src/reduction.o \
            third_party/fasthare/fasthare.o third_party/fasthare/graph.o

.PHONY: all probe clean

all: solve_qplib solve_gset export_fasthare dense_scaling selftest bench bit_dense_multi

solve_qplib: apps/solve_qplib.o src/qplib.o $(CU_OBJS) $(CXX_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

solve_gset: apps/solve_gset.o src/gset.o $(CU_OBJS) $(CXX_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

export_fasthare: apps/export_fasthare.o src/qplib.o src/gset.o \
		src/sparse.o src/reduction.o third_party/fasthare/fasthare.o \
		third_party/fasthare/graph.o
	$(CXX) $^ -o $@

dense_scaling: apps/dense_scaling.o $(CU_OBJS) $(CXX_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

selftest: apps/selftest.o $(CU_OBJS) $(CXX_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

bench: apps/bench.o $(CU_OBJS) $(CXX_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# dense +-1 bit path over 1-2 GPUs with HBM+Grace placement; no fast-math and no FMA
# contraction so that --check can compare with the host reference bit for bit
bit_dense_multi: apps/bit_dense_multi.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -fmad=false -Xcompiler -ffp-contract=off $< -o $@

probe: gpu_probe c2c_probe

gpu_probe: tools/gpu_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# NVLink-C2C read bandwidth of a pinned Grace allocation, as used by the Grace/hybrid placements
c2c_probe: tools/c2c_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f apps/*.o src/*.o third_party/fasthare/*.o \
	      solve_qplib solve_gset export_fasthare dense_scaling selftest bench gpu_probe c2c_probe bit_dense_multi
