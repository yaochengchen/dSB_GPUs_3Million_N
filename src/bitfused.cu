// src/bitfused.cu -- see include/dsb/bitfused.cuh for the design.
//
// Measured on GH200 at K2000 (N=2000, B=200), first version of this kernel:
// 16 us/step against 10 us for cuBLAS FP16.  The budget per block per step
// was 201,600 (XOR, POPC) pairs, each with TWO 32-bit shared loads: POPC at
// 16/SM/clk is 6.4 us and the loads were another 6.4 us that did not overlap.
// This version:
//
//   * keeps the sign bitmap replica-major in shared memory ([B][Wp]) with the
//     row stride Wp chosen so that 16-byte loads from consecutive replicas hit
//     32 distinct banks (Wp = 4 * odd);
//   * pads J rows to the same Wp and reads both operands with 16-byte loads,
//     so the load count drops 4x and the J load is a warp broadcast;
//   * publishes/gathers the device bitmap replica-major too, so the R=16
//     half-word layout is a plain aligned 32-bit read.
//
// The POPC floor (6.4 us at 125 blocks) remains: on Hopper XOR+POPC on the
// CUDA cores is about the same rate as HMMA on the tensor cores.  The exact
// path that beats FP16 outright is INT8 IMMA -- see Precision::INT8 in
// gemm.cu.  This kernel is the smallest-footprint exact path (J = N^2/8 bytes
// on chip for the whole run).
#include <cooperative_groups.h>

#include <stdexcept>
#include <string>

#include "dsb/bitfused.cuh"
#include "dsb/common.hpp"

namespace cg = cooperative_groups;

namespace dsb {
namespace bitfused {
namespace {

// --------------------------------------------------------------------------
// device helpers
// --------------------------------------------------------------------------

/// popcount(J_r XOR S_b) over Wp words, 16 bytes at a time.  Both rows are
/// 16-byte aligned and Wp is a multiple of 4.
__device__ __forceinline__ int pop_xor(const std::uint32_t* __restrict__ jr,
                                       const std::uint32_t* __restrict__ sr,
                                       int wpad) {
  const uint4* ja = reinterpret_cast<const uint4*>(jr);
  const uint4* sa = reinterpret_cast<const uint4*>(sr);
  int pop = 0;
#pragma unroll 2
  for (int q = 0; q < (wpad >> 2); ++q) {
    const uint4 j = ja[q];
    const uint4 s = sa[q];
    pop += __popc(j.x ^ s.x) + __popc(j.y ^ s.y) + __popc(j.z ^ s.z) +
           __popc(j.w ^ s.w);
  }
  return pop;
}

/// popcount(P_r & S_b) - popcount(M_r & S_b)
__device__ __forceinline__ int pop_and_diff(const std::uint32_t* __restrict__ pr,
                                            const std::uint32_t* __restrict__ mr,
                                            const std::uint32_t* __restrict__ sr,
                                            int wpad) {
  const uint4* pa = reinterpret_cast<const uint4*>(pr);
  const uint4* ma = reinterpret_cast<const uint4*>(mr);
  const uint4* sa = reinterpret_cast<const uint4*>(sr);
  int acc = 0;
#pragma unroll 2
  for (int q = 0; q < (wpad >> 2); ++q) {
    const uint4 p = pa[q];
    const uint4 m = ma[q];
    const uint4 s = sa[q];
    acc += (__popc(p.x & s.x) - __popc(m.x & s.x)) +
           (__popc(p.y & s.y) - __popc(m.y & s.y)) +
           (__popc(p.z & s.z) - __popc(m.z & s.z)) +
           (__popc(p.w & s.w) - __popc(m.w & s.w));
  }
  return acc;
}

__device__ __forceinline__ void advance(float& xv, float& yv, float acc,
                                        float delta, float pump, float xi,
                                        float dt) {
  yv += (-(delta - pump) * xv + xi * acc) * dt;
  xv += dt * yv * delta;
  if (fabsf(xv) > 1.f) {
    xv = (xv > 0.f) ? 1.f : -1.f;
    yv = 0.f;
  }
}

/// Device bitmap is [2][B][W] uint32.  A 16-row block owns one uint16 of each
/// replica's row: viewed as [B][2W] uint16 the half index is the block index.
template <int ROWS>
__device__ __forceinline__ void publish(std::uint32_t* gs, std::size_t bufoff,
                                        int words, int blk, int b,
                                        std::uint32_t bits) {
  if (ROWS == 32) {
    gs[bufoff + std::size_t(b) * words + blk] = bits;
  } else {
    std::uint16_t* h = reinterpret_cast<std::uint16_t*>(gs + bufoff);
    h[std::size_t(b) * (2 * words) + blk] = std::uint16_t(bits);
  }
}

// --------------------------------------------------------------------------
// the step kernel
//
//   gx, gy   [n*batch]              x[row*batch + replica]
//   gj       DENSE : [n][W]         bit 1 == +1, bit 0 == -1
//            else  : [2][n][W]      plane 0 == (J==+1), plane 1 == (J==-1)
//   growc    [n]                    popc(P_r) - popc(M_r); unused when DENSE
//   gs       [2][batch][W]          sign bitmap, double buffered, zeroed once
// --------------------------------------------------------------------------

template <bool DENSE, int ROWS>
__global__ __launch_bounds__(kThreads) void step_kernel(
    float* __restrict__ gx, float* __restrict__ gy,
    const std::uint32_t* __restrict__ gj, const std::int32_t* __restrict__ growc,
    std::uint32_t* __restrict__ gs, const float* __restrict__ pump, float delta,
    float xi, float dt, int n, int batch, int words, int wpad, int n_steps) {
#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();

  const int blk  = int(blockIdx.x);
  const int row0 = blk * ROWS;
  const int len  = (n - row0 < ROWS) ? (n - row0) : ROWS;
  const int tid  = int(threadIdx.x);

  extern __shared__ __align__(16) std::uint32_t raw[];
  std::uint32_t* jb = raw;                                        // [ROWS][Wp]
  std::uint32_t* mb = DENSE ? nullptr : jb + std::size_t(ROWS) * wpad;
  std::uint32_t* sb = (DENSE ? jb : mb) + std::size_t(ROWS) * wpad;  // [B][Wp]
  float*         x  = reinterpret_cast<float*>(sb + std::size_t(batch) * wpad);
  float*         y  = x + std::size_t(ROWS) * batch;
  std::int32_t*  rc = reinterpret_cast<std::int32_t*>(y + std::size_t(ROWS) * batch);

  const std::size_t bitmap = std::size_t(batch) * words;  // words per buffer

  // ---- once: this block's slice of J on chip, rows padded to Wp ----------
  {
    const std::size_t plane = std::size_t(n) * words;
    for (int i = tid; i < ROWS * wpad; i += kThreads) {
      const int  r    = i / wpad;
      const int  w    = i - r * wpad;
      const bool live = (r < len) && (w < words);
      jb[i] = live ? gj[std::size_t(row0 + r) * words + w] : 0u;
      if (!DENSE) mb[i] = live ? gj[plane + std::size_t(row0 + r) * words + w] : 0u;
    }
    if (!DENSE)
      for (int r = tid; r < ROWS; r += kThreads)
        rc[r] = (r < len) ? growc[row0 + r] : 0;
  }
  // Pad words of the sign rows are never written again: zero them once.
  for (int i = tid; i < batch * wpad; i += kThreads) sb[i] = 0u;

  // ---- once: this block's rows of the state on chip ----------------------
  for (int i = tid; i < ROWS * batch; i += kThreads) {
    const int  r    = i / batch;
    const int  b    = i - r * batch;
    const bool live = (r < len);
    x[i] = live ? gx[std::size_t(row0 + r) * batch + b] : 0.f;
    y[i] = live ? gy[std::size_t(row0 + r) * batch + b] : 0.f;
  }
  __syncthreads();

  // ---- publish the initial signs into buffer 0 ----------------------------
  for (int b = tid; b < batch; b += kThreads) {
    std::uint32_t bits = 0u;
    for (int r = 0; r < len; ++r)
      if (x[std::size_t(r) * batch + b] > 0.f) bits |= (1u << r);
    publish<ROWS>(gs, 0, words, blk, b, bits);
  }
  grid.sync();

  int buf = 0;
  for (int step = 0; step < n_steps; ++step) {
    // ---- everyone's signs -> shared [B][Wp].  Contiguous, L2-resident. ---
    {
      const std::uint32_t* src = gs + std::size_t(buf) * bitmap;
      for (int i = tid; i < int(bitmap); i += kThreads) {
        const int b = i / words;
        const int w = i - b * words;
        sb[std::size_t(b) * wpad + w] = src[i];
      }
    }
    __syncthreads();

    const float p = pump[step];

    // ---- acc + update in one pass.  acc reads only sb, never x, so writing
    //      x in place keeps this a Jacobi sweep without an inner barrier.
    for (int idx = tid; idx < len * batch; idx += kThreads) {
      const int r = idx / batch;
      const int b = idx - r * batch;

      const std::uint32_t* sr = sb + std::size_t(b) * wpad;
      float acc;
      if (DENSE) {
        // Diagonal bit is clear (== -1) so column r contributed -s_r instead
        // of 0: add s_r back.  Padding columns have both bits clear, count as
        // +1 each, and cancel exactly against (32Wp - n).  Net: n - 2*pop + s_r.
        const int   pop = pop_xor(jb + std::size_t(r) * wpad, sr, wpad);
        const int   gr  = row0 + r;
        const float srf = ((sr[gr >> 5] >> (gr & 31)) & 1u) ? 1.f : -1.f;
        acc = float(n - 2 * pop) + srf;
      } else {
        // Diagonal and padding bits are clear in both planes: no correction.
        const int d = pop_and_diff(jb + std::size_t(r) * wpad,
                                   mb + std::size_t(r) * wpad, sr, wpad);
        acc = float(2 * d - rc[r]);
      }

      float xv = x[idx];
      float yv = y[idx];
      advance(xv, yv, acc, delta, p, xi, dt);
      x[idx] = xv;
      y[idx] = yv;
    }
    __syncthreads();

    // ---- pack and publish this block's bits of every replica ------------
    buf ^= 1;
    {
      const std::size_t bufoff = std::size_t(buf) * bitmap;
      for (int b = tid; b < batch; b += kThreads) {
        std::uint32_t bits = 0u;
        for (int r = 0; r < len; ++r)
          if (x[std::size_t(r) * batch + b] > 0.f) bits |= (1u << r);
        publish<ROWS>(gs, bufoff, words, blk, b, bits);
      }
    }
    grid.sync();
  }

  // ---- state back to device memory ----------------------------------------
  for (int i = tid; i < len * batch; i += kThreads) {
    const int r = i / batch;
    const int b = i - r * batch;
    gx[std::size_t(row0 + r) * batch + b] = x[i];
    gy[std::size_t(row0 + r) * batch + b] = y[i];
  }
#else
  (void)gx; (void)gy; (void)gj; (void)growc; (void)gs; (void)pump;
  (void)delta; (void)xi; (void)dt; (void)n; (void)batch; (void)words;
  (void)wpad; (void)n_steps;
#endif
}

// --------------------------------------------------------------------------
// energy: one block per replica, bitmap for that replica in shared memory,
// J planes streamed from device memory (they are L2-resident at these sizes).
// --------------------------------------------------------------------------

template <bool DENSE>
__global__ __launch_bounds__(256) void energy_kernel(
    const float* __restrict__ gx, const std::uint32_t* __restrict__ gj,
    const std::int32_t* __restrict__ growc, double* __restrict__ energy,
    int n, int batch, int words) {
  const int b   = int(blockIdx.x);
  const int tid = int(threadIdx.x);

  extern __shared__ __align__(16) std::uint32_t sb[];  // [W]
  __shared__ double total;

  for (int w = tid; w < words; w += 256) {
    std::uint32_t word = 0u;
    const int     base = w * 32;
    for (int i = 0; i < 32; ++i) {
      const int r = base + i;
      if (r < n && gx[std::size_t(r) * batch + b] > 0.f) word |= (1u << i);
    }
    sb[w] = word;
  }
  if (tid == 0) total = 0.0;
  __syncthreads();

  const std::size_t plane = std::size_t(n) * words;
  double local = 0.0;
  for (int r = tid; r < n; r += 256) {
    const float sr = ((sb[r >> 5] >> (r & 31)) & 1u) ? 1.f : -1.f;
    int acc;
    if (DENSE) {
      int pop = 0;
      for (int w = 0; w < words; ++w)
        pop += __popc(gj[std::size_t(r) * words + w] ^ sb[w]);
      acc = n - 2 * pop + int(sr);
    } else {
      int d = 0;
      for (int w = 0; w < words; ++w) {
        const std::uint32_t s = sb[w];
        d += __popc(gj[std::size_t(r) * words + w] & s) -
             __popc(gj[plane + std::size_t(r) * words + w] & s);
      }
      acc = 2 * d - growc[r];
    }
    local += double(acc) * double(sr);
  }
  atomicAdd(&total, local);
  __syncthreads();
  if (tid == 0) energy[b] = -0.5 * total;
}

int resolve(int device) {
  if (device < 0) DSB_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

const void* kernel_for(const Plan& p) {
  if (p.rows == 32)
    return p.dense_pm1 ? (const void*)step_kernel<true, 32>
                       : (const void*)step_kernel<false, 32>;
  return p.dense_pm1 ? (const void*)step_kernel<true, 16>
                     : (const void*)step_kernel<false, 16>;
}

/// Row stride of the shared bitmap and J slice, in words: a multiple of 4
/// (16-byte rows) with (Wp/4) odd, so eight consecutive replicas' 16-byte
/// loads land on eight distinct bank groups.
int padded_words(int words) {
  int wp = (words + 3) & ~3;
  if (((wp >> 2) & 1) == 0) wp += 4;
  return wp;
}

Plan plan_with_rows(int n, int batch, bool dense_pm1, int rows) {
  Plan p;
  p.n         = n;
  p.batch     = batch;
  p.words     = (n + 31) / 32;
  p.wpad      = padded_words(p.words);
  p.rows      = rows;
  p.blocks    = (n + rows - 1) / rows;
  p.dense_pm1 = dense_pm1;
  const std::size_t planes = dense_pm1 ? 1u : 2u;
  p.smem = planes * std::size_t(rows) * p.wpad * sizeof(std::uint32_t)
           + std::size_t(batch) * p.wpad * sizeof(std::uint32_t)
           + 2u * std::size_t(rows) * batch * sizeof(float)
           + std::size_t(rows) * sizeof(std::int32_t);
  return p;
}

}  // namespace

// ==========================================================================
// public
// ==========================================================================

bool fits(const Plan& plan, int device) {
  if (plan.n <= 0 || plan.batch <= 0) return false;
  if (plan.rows != 16 && plan.rows != 32) return false;
  device = resolve(device);
  int optin = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(
      &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (plan.smem > std::size_t(optin)) return false;

  int coop = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device));
  if (!coop) return false;

  const void* kernel = kernel_for(plan);
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(plan.smem)) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads,
                                                    plan.smem) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  int sms = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return per_sm >= 1 && plan.blocks <= per_sm * sms;
}

Plan make_bit_plan(int n, int batch, bool dense_pm1, int device) {
  // 16 rows per block doubles the block count (and halves the per-block
  // shared footprint); take it whenever the grid is co-resident.
  Plan p16 = plan_with_rows(n, batch, dense_pm1, 16);
  if (n > 0 && batch > 0 && fits(p16, device)) return p16;
  return plan_with_rows(n, batch, dense_pm1, 32);
}

// --------------------------------------------------------------------------

bool classify(const double* coupling, int n, std::size_t ld, bool& pm1) {
  pm1 = true;
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      if (c == r) continue;
      const double v = coupling[std::size_t(r) * ld + c];
      if (v == 1.0 || v == -1.0) continue;
      pm1 = false;
      if (v != 0.0) return false;
    }
  return true;
}

void pack_dense_pm1(const double* coupling, int n, std::size_t ld, int words,
                    std::vector<std::uint32_t>& out) {
  out.assign(std::size_t(n) * words, 0u);
  for (int r = 0; r < n; ++r) {
    std::uint32_t* row = out.data() + std::size_t(r) * words;
    for (int c = 0; c < n; ++c) {
      if (c == r) continue;
      if (coupling[std::size_t(r) * ld + c] > 0.0) row[c >> 5] |= (1u << (c & 31));
    }
  }
}

void pack_pm_planes(const double* coupling, int n, std::size_t ld, int words,
                    std::vector<std::uint32_t>& out,
                    std::vector<std::int32_t>& rowconst) {
  out.assign(2u * std::size_t(n) * words, 0u);
  rowconst.assign(std::size_t(n), 0);
  const std::size_t plane = std::size_t(n) * words;
  for (int r = 0; r < n; ++r) {
    std::uint32_t* prow = out.data() + std::size_t(r) * words;
    std::uint32_t* mrow = out.data() + plane + std::size_t(r) * words;
    std::int32_t   acc  = 0;
    for (int c = 0; c < n; ++c) {
      if (c == r) continue;
      const double v = coupling[std::size_t(r) * ld + c];
      if (v > 0.0) {
        prow[c >> 5] |= (1u << (c & 31));
        ++acc;
      } else if (v < 0.0) {
        mrow[c >> 5] |= (1u << (c & 31));
        --acc;
      }
    }
    rowconst[std::size_t(r)] = acc;
  }
}

// --------------------------------------------------------------------------

void launch_steps(const Plan& plan, float* d_x, float* d_y,
                  const std::uint32_t* d_j, const std::int32_t* d_rowconst,
                  std::uint32_t* d_sign, const float* d_pump, float delta,
                  float xi, float dt, int n_steps, cudaStream_t stream) {
  if (!fits(plan))
    throw std::runtime_error(
        "bit variant: N=" + std::to_string(plan.n) + " batch=" +
        std::to_string(plan.batch) + " needs " + std::to_string(plan.smem) +
        " B of shared memory per block and " + std::to_string(plan.blocks) +
        " resident blocks; not available on this device");
  if (!plan.dense_pm1 && d_rowconst == nullptr)
    throw std::runtime_error("bit variant: general form needs rowconst");

  const void* kernel = kernel_for(plan);
  DSB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(plan.smem)));

  int   n = plan.n, batch = plan.batch, words = plan.words, wpad = plan.wpad;
  int   steps = n_steps;
  void* args[] = {(void*)&d_x,    (void*)&d_y,   (void*)&d_j,   (void*)&d_rowconst,
                  (void*)&d_sign, (void*)&d_pump, (void*)&delta, (void*)&xi,
                  (void*)&dt,     (void*)&n,     (void*)&batch, (void*)&words,
                  (void*)&wpad,   (void*)&steps};
  DSB_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(unsigned(plan.blocks)),
                                             dim3(kThreads), args, plan.smem,
                                             stream));
}

void launch_energy(const Plan& plan, const float* d_x, const std::uint32_t* d_j,
                   const std::int32_t* d_rowconst, double* d_energy,
                   cudaStream_t stream) {
  const std::size_t smem = std::size_t(plan.words) * sizeof(std::uint32_t);
  if (plan.dense_pm1) {
    energy_kernel<true><<<unsigned(plan.batch), 256, smem, stream>>>(
        d_x, d_j, d_rowconst, d_energy, plan.n, plan.batch, plan.words);
  } else {
    energy_kernel<false><<<unsigned(plan.batch), 256, smem, stream>>>(
        d_x, d_j, d_rowconst, d_energy, plan.n, plan.batch, plan.words);
  }
  DSB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace bitfused
}  // namespace dsb
