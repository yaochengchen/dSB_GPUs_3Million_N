// CSR-native discrete Simulated Bifurcation solver.
#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dsb/solver.hpp"
#include "dsb/sparse.hpp"

namespace dsb {

/// FP32 sparse solver supporting three execution schedules over the same CSR
/// matrix. Select CsrRow, CsrBlock, or CsrCluster in Options.
class SparseSolver {
 public:
  SparseSolver(const SparseMatrix& coupling,
               const Options& options = Options());
  ~SparseSolver();

  SparseSolver(const SparseSolver&) = delete;
  SparseSolver& operator=(const SparseSolver&) = delete;

  void run();
  std::vector<double> energies() const;
  int best_replica() const;
  Eigen::VectorXd best_spins() const;

  int n() const { return n_; }
  int batch() const { return options_.batch; }
  float xi() const { return xi_; }
  float last_run_ms() const { return last_ms_; }
  std::size_t coupling_bytes() const { return coupling_bytes_; }
  std::size_t device_bytes() const { return device_bytes_; }
  Variant variant() const { return options_.variant; }
  int cluster_size() const { return cluster_size_; }

 private:
  void prepare_row_graph();
  void free_device() noexcept;

  Options options_;
  int n_ = 0;
  int nnz_ = 0;
  int threads_ = 0;
  int persistent_threads_ = 0;
  int row_blocks_ = 0;
  int cluster_size_ = 1;
  bool row_uses_fused_ = false;
  bool block_uses_shared_state_ = false;
  std::size_t block_smem_bytes_ = 0;
  float xi_ = 0.f;
  float last_ms_ = 0.f;
  std::size_t coupling_bytes_ = 0;
  std::size_t device_bytes_ = 0;
  bool ran_ = false;

  int* d_row_ptr_ = nullptr;
  int* d_col_idx_ = nullptr;
  float* d_values_ = nullptr;
  float* d_x_ = nullptr;
  float* d_y_ = nullptr;
  int8_t* d_sign_ = nullptr;
  int8_t* d_sign_next_ = nullptr;
  float* d_pump_ = nullptr;
  mutable double* d_energy_ = nullptr;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
  cudaGraph_t row_graph_ = nullptr;
  cudaGraphExec_t row_graph_exec_ = nullptr;
};

}  // namespace dsb
