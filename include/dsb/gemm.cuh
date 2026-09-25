// include/dsb/gemm.cuh
//
// The standard dense formulation, kept as a baseline.
//
// Instead of each replica computing its own J @ s, all `batch` sign vectors are
// stacked into an N x B matrix S and one cuBLAS call produces J @ S for every
// replica at once. Each element of J is then read once per step and used B
// times: arithmetic intensity B instead of 1, and the multiply runs on tensor
// cores -- HMMA for FP16 storage, (when `tf32` is set) TF32 for FP32
// storage, and IMMA for INT8 storage.  INT8 is for couplings in {-1, 0, +1}:
// the products are +-1, the int32 accumulation is exact, and the trajectory
// is bit-identical to the FP32 kernels' -- at roughly twice the FP16 rate.  Without tf32 an FP32 GEMM stays on the CUDA cores and cannot beat
// ~24 us/step at N=2000, B=200 on GH200; with it the same call is ~3x faster.
// TF32 keeps 10 mantissa bits, which for a coupling term that is a sum of
// +-1 products means the FP32 and TF32 trajectories diverge (chaotically) but
// the energy statistics do not.  Pass tf32=false for bit-exact comparisons.
//
// The state lives in device memory because the whole GEMM must finish before
// any replica advances. The initial sign kernel and the complete sequence of
// GEMM/update nodes are captured once in a CUDA graph. Repeated solver runs
// therefore submit the entire step loop with one cudaGraphLaunch.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dsb/common.hpp"

namespace dsb {

class GemmStepper {
 public:
  GemmStepper(int n, std::size_t ldj, int batch, Precision precision,
              cudaStream_t stream, bool tf32 = true);
  ~GemmStepper();

  GemmStepper(const GemmStepper&)            = delete;
  GemmStepper& operator=(const GemmStepper&) = delete;

  /// Capture the graph if these buffers/parameters are not already cached.
  /// Graph construction is deliberately separate from run() so callers can
  /// keep capture/instantiation outside their timed region.
  void prepare(void* x, void* y, const void* j, const float* pump,
               float delta, float xi, float dt, int n_steps);

  /// Advance `n_steps` steps. `x`, `y` and `j` are the solver's device buffers
  /// in the usual layout (x[row*B + replica], j row-major with stride ldj).
  void run(void* x, void* y, const void* j, const float* pump, float delta,
           float xi, float dt, int n_steps);

  /// Hybrid placement: rows [0, split_rows) of J live at the `j` pointer
  /// passed to prepare()/run(), rows [split_rows, N) at `j_tail` (same row
  /// stride ldj, may be in Grace memory).  Each step then issues two GEMMs,
  /// one per row block, writing disjoint row ranges of the accumulator.
  /// split_rows <= 0 or >= N (or j_tail == nullptr) disables the split.
  /// FP32/FP16 only.
  void set_row_split(const void* j_tail, int split_rows);

  /// Grace-resident rows of J are copied (cudaMemcpyAsync, C2C) into an HBM
  /// staging buffer of at most `stage_bytes` and multiplied there, one block
  /// at a time, instead of being read by cuBLAS in place.  cuBLAS's kernel
  /// choice is tuned for HBM: reading pinned Grace memory in place, fp16
  /// N=400k ran at 15 GB/s (batch 1) / 7 GB/s (batch 8, 64) while a plain
  /// stream read of the same buffer reaches ~350 GB/s (tools/c2c_probe.cu).
  /// The staged GEMM runs from HBM, so a step costs ~ one C2C copy of the
  /// Grace share of J whatever kernel cuBLAS picks.
  /// `main_in_grace`: the rows at `j` (prepare/run) are Grace-resident
  /// (MatrixMemory::Grace).  The hybrid tail is always staged once this is
  /// set.  FP32/FP16 only.
  void set_grace_staging(std::size_t stage_bytes, bool main_in_grace);

  /// Device memory this stepper allocated on top of the solver's own.
  std::size_t device_bytes() const;

 private:
  int          n_;
  std::size_t  ldj_;
  int          batch_;
  Precision    precision_;
  cudaStream_t stream_;
  bool         tf32_;

  cublasHandle_t handle_ = nullptr;
  void*          d_sign_ = nullptr;  ///< [N*B] same type as J; INT8: [B][ldj]
  float*         d_acc_  = nullptr;  ///< [N*B] FP32 (or exact INT32) J @ S
  void*          d_workspace_ = nullptr;
  std::size_t    workspace_bytes_ = 0;

  const void*     j_tail_     = nullptr;
  int             split_rows_ = 0;
  void*           d_stage_    = nullptr;  ///< HBM staging block (Grace rows)
  int             stage_rows_ = 0;        ///< 0: staging off
  bool            stage_main_ = false;

  cudaGraph_t     graph_      = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  int             graph_steps_ = 0;
  void*           graph_x_ = nullptr;
  void*           graph_y_ = nullptr;
  const void*     graph_j_ = nullptr;
  const float*    graph_pump_ = nullptr;
  float           graph_delta_ = 0.f;
  float           graph_xi_ = 0.f;
  float           graph_dt_ = 0.f;
  const void*     graph_j_tail_ = nullptr;
  int             graph_split_rows_ = 0;
  int             graph_stage_rows_ = 0;
  bool            graph_stage_main_ = false;
};

}  // namespace dsb
