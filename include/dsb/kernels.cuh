// include/dsb/kernels.cuh
//
// Launchers for the discrete Simulated Bifurcation (dSB) time-stepping kernels.
//
// Every variant computes the same thing each step:
//
//     acc = J @ sign(x);  y += (-(delta-p)*x + xi*acc)*dt;  x += dt*y*delta
//
// They differ only in where the state lives and how the per-step barrier is
// done. J itself is a single allocation in all of them -- the question is never
// how many copies of J exist, but how many times each byte of J is dragged out
// of device memory.
//
//   Block       one replica per CUDA block; state in that block's shared
//               memory; barrier is __syncthreads(). 9N bytes. One launch for
//               the whole run. Each block sweeps J row by row, so whether two
//               blocks share a fetch is left entirely to L2.
//
//   Cluster     one replica per thread block cluster (sm_90+); the state is
//               split across C blocks that read each other's sign array over
//               distributed shared memory; barrier is cluster.sync().
//               9*ceil(N/C) bytes per block, so N scales with C.
//
//   Bit         for J in {-1,0,+1} only (every G-set / Max-Cut instance): J is
//               a bitplane, the coupling term is XOR+POPC, and a block owns
//               32 rows x ALL replicas so J is read once per solve and each
//               J word serves every replica.  See dsb/bitfused.cuh.  This is
//               the variant that wins on K2000 and the small/medium G-set.
//
//   Which one wins, and why, is a function of (N, batch) alone:
//     Block/Cluster read J once per replica per step -- fine while J is
//     L2-resident (N <= ~3800 fp32) and batch >= #SMs, hopeless otherwise.
//     Gemm reads J once per step for all replicas and is the right answer for
//     large N or small batch.  Bit reads J once per *solve*.
//
// All of them need the caller to opt in to more than 48 KB of dynamic shared
// memory, which the launchers do via cudaFuncSetAttribute.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dsb/common.hpp"

namespace dsb {

enum class Variant {
  Auto,        ///< Bit for ternary J (Solver), else Gemm; see make_plan
  Block,
  Cluster,
  GlobalSync,  ///< baseline: state in device memory, grid.sync() per step
  Gemm,        ///< baseline: one cuBLAS J @ S per step for all replicas at once
  CsrRow,      ///< one block per CSR row; threads span replicas
  CsrBlock,    ///< one persistent block per replica; thread per row
  CsrCluster,  ///< one persistent Hopper block cluster per replica
  Bit,         ///< J in {-1,0,+1}: bitplane J on chip for the whole run, XOR+POPC
};

/// Kept for callers that want to reason about occupancy: the per-replica
/// Block/Cluster kernels launch `batch` (x cluster) blocks and cannot fill a
/// 132-SM part below this many replicas.  Auto no longer consults it -- Auto
/// is Gemm for real-valued J at every (N, batch); see make_plan.
constexpr int kAutoMinBatchForBlock = 64;

inline bool is_csr_variant(Variant v) {
  return v == Variant::CsrRow || v == Variant::CsrBlock ||
         v == Variant::CsrCluster;
}

/// True when the variant keeps its state in device memory rather than shared
/// memory, and therefore needs an N*B scratch buffer from the caller.
bool needs_scratch(Variant v);

const char* to_string(Variant v);

struct LaunchPlan {
  Variant     variant       = Variant::Block;
  int         cluster_size  = 1;  ///< blocks per replica
  int         rows_per_rank = 0;  ///< rows per block (== N unless clustered)
  std::size_t smem_bytes    = 0;  ///< dynamic shared memory per block
};

/// Largest dynamic shared memory a single block can opt in to, in bytes.
int device_max_smem_optin(int device = -1);

/// Largest N the plain block kernel can handle on this device.
int device_max_n_block(int device = -1);

/// L2 size and the part of it that can be reserved for persisting accesses.
int device_l2_bytes(int device = -1);
int device_max_persisting_l2_bytes(int device = -1);

/// Measured peak read bandwidth in bytes/s. Streams a large buffer, so it takes
/// a moment and allocates ~1 GB. Used to turn a measured step time into "how
/// many times was J actually fetched".
double measure_read_bandwidth(int device = -1);

/// Pick a launch configuration for `n` spins.
/// Throws if `n` does not fit even with a non-portable Hopper cluster of 16.
LaunchPlan make_plan(int n, Precision precision, Variant want = Variant::Auto,
                     int requested_cluster = 0, int device = -1,
                     int batch = 1);

// ---------------------------------------------------------------------------
// L2 persistence
//
// Marks a byte range as worth keeping in L2. The carve-out is capped at
// device_max_persisting_l2_bytes(), which is a fraction of an already small
// cache -- on H100-class hardware that is tens of megabytes against a J that is
// gigabytes. It is worth switching on for small instances and close to useless
// for large ones; the benchmark reports both so you can see where the line is.
//
// RAII: the window and the carve-out are released on destruction.
// ---------------------------------------------------------------------------
class L2Persistence {
 public:
  L2Persistence(cudaStream_t stream, const void* ptr, std::size_t bytes,
                int device = -1);
  ~L2Persistence();

  L2Persistence(const L2Persistence&)            = delete;
  L2Persistence& operator=(const L2Persistence&) = delete;

  /// Bytes actually pinned (0 when the device refused or nothing was asked).
  std::size_t pinned_bytes() const { return pinned_; }

 private:
  cudaStream_t stream_;
  std::size_t  pinned_ = 0;
  bool         active_ = false;
};

// ---------------------------------------------------------------------------
// Time stepping. `x` and `y` are [N*B] with layout x[row*B + replica].
// `j` is [N*ldj] row-major. `pump` is [n_steps] FP32 on the device.
// ---------------------------------------------------------------------------

/// `scratch` is an N*B buffer of the same type, required when
/// needs_scratch(plan.variant) and ignored otherwise.
void launch_steps(__half* x, __half* y, const __half* j, std::size_t ldj,
                  const float* pump, float delta, float xi, float dt, int n,
                  int batch, int n_steps, const LaunchPlan& plan,
                  __half* scratch, cudaStream_t stream);

void launch_steps(float* x, float* y, const float* j, std::size_t ldj,
                  const float* pump, float delta, float xi, float dt, int n,
                  int batch, int n_steps, const LaunchPlan& plan, float* scratch,
                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Ising energy of sign(x):  E[b] = -0.5 * sum_i (J @ s)_i * s_i
// One block per replica; needs N bytes of shared memory.
// ---------------------------------------------------------------------------

void launch_energy(const __half* x, const __half* j, std::size_t ldj, int n,
                   int batch, double* energy, cudaStream_t stream);

void launch_energy(const float* x, const float* j, std::size_t ldj, int n,
                   int batch, double* energy, cudaStream_t stream);

}  // namespace dsb
