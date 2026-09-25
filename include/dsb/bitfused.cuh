// include/dsb/bitfused.cuh
//
// Persistent bit-fused dSB stepper for Ising instances whose couplings are in
// {-1, 0, +1} -- every G-set / Max-Cut instance, K2000 included.
//
// The identity it rests on: for a, b in {-1,+1}^n packed one bit each
// (bit 1 == +1),
//
//     sum_k a_k b_k  ==  n - 2 * popcount(a XOR b)
//
// exactly, in integer arithmetic.  The dSB coupling term J @ sign(x) is such a
// product, so the 2000-term FP32 dot product cuBLAS issues as 2000 FFMAs
// becomes 63 XOR+POPC pairs, and the answer is exact rather than rounded.
//
// Consequences the kernel is built around:
//
//   * J as a bitplane is N^2/8 bytes -- 500 KB at N=2000.  Each block owns a
//     slice of rows (4 KB for 16 rows at N=2000) and keeps it in shared memory
//     for the whole run, so J is read from device memory once per solve, not
//     once per step.
//   * A block owns 16 or 32 rows, i.e. exactly one 16-bit half or one 32-bit
//     word of every replica's sign bitmap; publishing new signs is one
//     coalesced store per block with no conflicts and no atomics.  16 rows
//     per block is preferred (2x the blocks, 125 of 132 SMs busy at N=2000);
//     32 is the fallback when the 16-row plan cannot be co-resident.
//   * The sign bitmap is replica-major.  In shared memory each replica's row
//     is padded to Wp words, Wp = 4 * odd, so the inner loop reads both the J
//     row (a warp broadcast) and the sign row 16 bytes at a time with no bank
//     conflicts between the eight consecutive replicas of a quarter warp.
//
// Layout
//   W       = ceil(N/32)                words per row;  Wp = padded stride
//   R       = rows per block, 16 or 32; blocks = ceil(N/R)
//   block k owns rows [R*k, R*k+R) and, per replica, bits [R*k, R*k+R) of
//   the bitmap: one uint16 (R=16) or one uint32 (R=32)
//   device  : sign bitmap [2][B][W] uint32 (== [B][2W] uint16 for R=16)
//   grid    = ceil(N/R) cooperative blocks, one grid.sync() per step
//   smem    = jbits[R][Wp] (+ mbits[R][Wp]) | sbits[B][Wp] | x[R][B] | y[R][B]
//
//   K2000 (N=2000, B=200, W=63, Wp=68, R=16, single plane), 125 blocks:
//       jbits   16*68*4      =   4352 B
//       sbits   200*68*4     =  54400 B
//       x,y     16*200*4*2   =  25600 B
//                               84416 B   -> two blocks per SM
//
// Two forms:
//   dense_pm1  every off-diagonal J is +-1  -> one plane,  XOR + POPC per word
//   general    J in {0,+-1}                 -> two planes, 2x (AND + POPC)
//
//     general:  acc_r = 2 * (popc(P_r & S) - popc(M_r & S)) - (|P_r| - |M_r|)
//
// Cost model (K2000, GH200): POPC issues at 16/SM/clk, so the dense form is
// 16*200*68 = 217600 POPC per block per step -> 6.9 us at 1.98 GHz with 125
// blocks on 132 SMs; that is the floor of this kernel.  Measured: cuBLAS
// FP16 (HMMA) 10 us, cuBLAS FP32 72 us, the first (scalar-load) version of
// this kernel 16 us.  For the path that beats FP16 outright see INT8 in
// gemm.cu; this kernel's distinction is the 500 KB on-chip J.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dsb {
namespace bitfused {

constexpr int kThreads = 512;

struct Plan {
  int         n         = 0;
  int         batch     = 0;
  int         words     = 0;      ///< ceil(n/32)
  int         wpad      = 0;      ///< shared row stride in words (4 * odd)
  int         rows      = 16;     ///< rows per block: 16 or 32
  int         blocks    = 0;      ///< ceil(n/rows)
  bool        dense_pm1 = false;  ///< every off-diagonal entry is +-1
  std::size_t smem      = 0;      ///< dynamic shared memory per block
};

/// Plan for (n, batch): 16 rows per block when that is co-resident on this
/// device, else 32.  Never throws; check fits() (which is what it used).
Plan make_bit_plan(int n, int batch, bool dense_pm1, int device = -1);

/// True when the plan's blocks can all be resident at once on this device.
bool fits(const Plan& plan, int device = -1);

/// Words of the double-buffered sign scratch (`d_sign`), both buffers.
inline std::size_t sign_words(const Plan& p) {
  return 2u * std::size_t(p.words) * p.batch;
}

/// Bytes of the packed coupling on the device (one or two planes).
inline std::size_t coupling_bytes(const Plan& p) {
  return (p.dense_pm1 ? 1u : 2u) * std::size_t(p.n) * p.words * sizeof(std::uint32_t);
}

// ---- host-side packing ----------------------------------------------------
//
// `coupling(r, c)` must be in {-1, 0, +1}.  Both packers clear the diagonal and
// the padding columns; the kernel accounts for both.

/// One plane: bit set == +1, bit clear == -1.  Requires dense_pm1.
void pack_dense_pm1(const double* coupling, int n, std::size_t ld, int words,
                    std::vector<std::uint32_t>& out);

/// Two planes: plane 0 == (J == +1), plane 1 == (J == -1).
/// rowconst[r] = popcount(P_r) - popcount(M_r).
void pack_pm_planes(const double* coupling, int n, std::size_t ld, int words,
                    std::vector<std::uint32_t>& out,
                    std::vector<std::int32_t>& rowconst);

/// Classify a coupling matrix.  `pm1` is true when every off-diagonal entry is
/// exactly +-1; the return value is true when every entry is in {-1,0,+1}.
bool classify(const double* coupling, int n, std::size_t ld, bool& pm1);

// ---- launch ---------------------------------------------------------------

/// Runs n_steps of dSB in one cooperative launch.
///   d_x, d_y      [n*batch] FP32, x[row*batch + replica]
///   d_j           packed planes from pack_dense_pm1 / pack_pm_planes
///   d_rowconst    [n] int32 from pack_pm_planes; may be nullptr when dense_pm1
///   d_sign        sign_words(plan) uint32 ([2][batch][W]), ZERO-INITIALISED
///                 by the caller (a 16-row plan with an odd block count leaves
///                 the top half of the last word unwritten and relies on 0)
///   d_pump        [n_steps] FP32
void launch_steps(const Plan& plan, float* d_x, float* d_y,
                  const std::uint32_t* d_j, const std::int32_t* d_rowconst,
                  std::uint32_t* d_sign, const float* d_pump, float delta,
                  float xi, float dt, int n_steps, cudaStream_t stream);

/// Ising energy of sign(x) per replica from the packed planes.
///   energy[b] = -0.5 * sum_r acc_r[b] * s_r[b]
void launch_energy(const Plan& plan, const float* d_x, const std::uint32_t* d_j,
                   const std::int32_t* d_rowconst, double* d_energy,
                   cudaStream_t stream);

}  // namespace bitfused
}  // namespace dsb
