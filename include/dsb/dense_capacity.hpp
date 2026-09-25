// Dense GH200 capacity benchmark support.
//
// Unlike Solver, this class never creates an Eigen N x N host matrix.  It
// allocates J directly in HBM or in pinned Grace memory (an anonymous mmap
// under the caller's NUMA policy, pinned with cudaHostRegister and read in
// place by the GPU over NVLink-C2C; never managed memory, which the driver
// would migrate), fills a deterministic dense high-entropy symmetric matrix,
// and invokes the normal dense dSB kernels.
//
// J may be stored FP32 or FP16 (Options::precision).  At batch=1 every step is
// a bandwidth-bound stream of J, so FP16 storage halves the step time and
// moves the HBM boundary from N ~ 176k (FP32) to N ~ 250k on a 144 GB GH200.
//
// MatrixMemory::Hybrid (gemm only) splits J by rows: as many leading rows as
// fit the HBM budget go to cudaMalloc'd HBM, the remaining rows to pinned
// Grace memory.  Each step issues one GEMM per row block, so the
// HBM share of J is streamed at HBM bandwidth and only the tail crosses C2C.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "dsb/gemm.cuh"
#include "dsb/solver.hpp"

namespace dsb {

enum class MatrixMemory { Auto, Hbm, Grace, Hybrid };

const char* to_string(MatrixMemory memory);
MatrixMemory parse_matrix_memory(const std::string& value);

struct DenseCapacityStats {
  std::string requested_memory;
  std::string selected_memory;
  std::string requested_variant;
  std::string selected_variant;
  int cluster_size = 1;
  std::size_t matrix_bytes = 0;
  std::size_t hbm_bytes = 0;
  std::size_t grace_bytes = 0;
  std::size_t hbm_free_before = 0;
  std::size_t hbm_total = 0;
  double generation_s = 0.0;
  float run_ms = 0.0f;
};

class DenseCapacitySolver {
 public:
  DenseCapacitySolver(int n, const Options& options, MatrixMemory memory,
                      double hbm_fraction = 0.85);
  ~DenseCapacitySolver();

  DenseCapacitySolver(const DenseCapacitySolver&) = delete;
  DenseCapacitySolver& operator=(const DenseCapacitySolver&) = delete;

  void run(int steps);
  const DenseCapacityStats& stats() const { return stats_; }
  const LaunchPlan& plan() const { return plan_; }
  int n() const { return n_; }
  int batch() const { return options_.batch; }

 private:
  void allocate_matrix(MatrixMemory requested, double hbm_fraction);
  void allocate_grace(void** pointer, std::size_t bytes);
  void generate_block(void* matrix, int row0, int rows);
  void allocate_state();
  void generate_matrix();
  void initialize_state();
  void free_all() noexcept;

  Options options_;
  int n_ = 0;
  std::size_t ldj_ = 0;
  LaunchPlan plan_;
  MatrixMemory selected_memory_ = MatrixMemory::Hbm;
  DenseCapacityStats stats_;

  void*  j_ = nullptr;        ///< FP32 or FP16 per options_.precision
  void*  j_tail_ = nullptr;   ///< Hybrid: rows [split_rows_, N), pinned Grace memory
  int    split_rows_ = 0;     ///< Hybrid: rows [0, split_rows_) live in j_
  std::size_t grace_alloc_bytes_ = 0;  ///< mmap length of the pinned Grace block
  void*  x_ = nullptr;
  void*  y_ = nullptr;
  void*  scratch_ = nullptr;
  float* pump_ = nullptr;
  std::unique_ptr<GemmStepper> gemm_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
};

}  // namespace dsb
