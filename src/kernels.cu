// src/kernels.cu
#include <cooperative_groups.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

#include "dsb/kernels.cuh"

namespace cg = cooperative_groups;

namespace dsb {
namespace {

constexpr int kBlockThreads = 1024;
constexpr int kWarps        = kBlockThreads / 32;

// --------------------------------------------------------------------------
// scalar helpers
// --------------------------------------------------------------------------
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ void store_f32(__half& dst, float v) { dst = __float2half_rn(v); }
__device__ __forceinline__ void store_f32(float& dst, float v) { dst = v; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// sign(0) == -1; see the note in dsb/common.hpp.
__device__ __forceinline__ int8_t sign_of(float v) {
  return (v > 0.f) ? int8_t(1) : int8_t(-1);
}

// --------------------------------------------------------------------------
// Partial dot product:  sum_{k < len} row[col0 + k] * s[k]
//
// `row + col0` is 16-byte aligned by construction (ldj, col0 and every tile
// width are multiples of 8), so the bulk is read with 16-byte vector loads: a
// warp pulls 512 B per instruction instead of the 64 B a scalar loop manages.
// --------------------------------------------------------------------------
__device__ __forceinline__ float dot_segment(const __half* __restrict__ row,
                                             const int8_t* __restrict__ s,
                                             int col0, int len, int lane) {
  const __half* base = row + col0;
  const int     nvec = len >> 3;  // 8 halves == 16 B
  const int4*   v    = reinterpret_cast<const int4*>(base);

  float acc = 0.f;
  for (int i = lane; i < nvec; i += 32) {
    int4           w = v[i];
    const __half2* h = reinterpret_cast<const __half2*>(&w);
    const int      c = i << 3;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __half22float2(h[k]);
      acc += f.x * float(s[c + 2 * k]) + f.y * float(s[c + 2 * k + 1]);
    }
  }
  for (int c = (nvec << 3) + lane; c < len; c += 32)
    acc += __half2float(base[c]) * float(s[c]);
  return acc;
}

__device__ __forceinline__ float dot_segment(const float* __restrict__ row,
                                             const int8_t* __restrict__ s,
                                             int col0, int len, int lane) {
  const float*  base = row + col0;
  const int     nvec = len >> 2;  // 4 floats == 16 B
  const float4* v    = reinterpret_cast<const float4*>(base);

  float acc = 0.f;
  for (int i = lane; i < nvec; i += 32) {
    const float4 w = v[i];
    const int    c = i << 2;
    acc += w.x * float(s[c]) + w.y * float(s[c + 1]) + w.z * float(s[c + 2]) +
           w.w * float(s[c + 3]);
  }
  for (int c = (nvec << 2) + lane; c < len; c += 32)
    acc += base[c] * float(s[c]);
  return acc;
}

// --------------------------------------------------------------------------
// One integration step for one row. x and y are updated in place, which is
// safe even though this is a Jacobi sweep: nothing in the step reads x -- the
// coupling term reads the sign snapshot s, taken before any row was touched.
// --------------------------------------------------------------------------
__device__ __forceinline__ void advance_row(float* x, float* y, int row,
                                            float acc, float delta, float pump,
                                            float xi, float dt) {
  float xv = x[row];
  float yv = y[row];
  yv += (-(delta - pump) * xv + xi * acc) * dt;
  xv += dt * yv * delta;
  if (fabsf(xv) > 1.f) {
    xv = (xv > 0.f) ? 1.f : -1.f;
    yv = 0.f;
  }
  x[row] = xv;
  y[row] = yv;
}

// ==========================================================================
// Variant 1: one replica per block, full-row sweep.  9N bytes.
// ==========================================================================
template <class T>
__global__ __launch_bounds__(kBlockThreads) void step_block_kernel(
    T* __restrict__ x_io, T* __restrict__ y_io, const T* __restrict__ j,
    std::size_t ldj, const float* __restrict__ pump, float delta, float xi,
    float dt, int n, int batch, int n_steps) {
  const int replica = blockIdx.x;

  extern __shared__ __align__(16) unsigned char raw[];
  float*  x = reinterpret_cast<float*>(raw);
  float*  y = x + n;
  int8_t* s = reinterpret_cast<int8_t*>(y + n);

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < n; i += kBlockThreads) {
    const std::size_t g = std::size_t(i) * batch + replica;
    x[i] = to_f32(x_io[g]);
    y[i] = to_f32(y_io[g]);
  }
  __syncthreads();

  for (int step = 0; step < n_steps; ++step) {
    for (int i = tid; i < n; i += kBlockThreads) s[i] = sign_of(x[i]);
    __syncthreads();

    const float p = pump[step];
    for (int row = warp; row < n; row += kWarps) {
      const float acc =
          warp_sum(dot_segment(j + std::size_t(row) * ldj, s, 0, n, lane));
      if (lane == 0) advance_row(x, y, row, acc, delta, p, xi, dt);
    }
    __syncthreads();
  }

  for (int i = tid; i < n; i += kBlockThreads) {
    const std::size_t g = std::size_t(i) * batch + replica;
    store_f32(x_io[g], x[i]);
    store_f32(y_io[g], y[i]);
  }
}

// ==========================================================================
// Variant 2: one replica per thread block cluster (sm_90+).  9*ceil(N/C) bytes.
//
// Rank q owns rows [q*R, q*R+len) and keeps x, y and s for those rows locally.
// To build the coupling term for one of its rows it walks the cluster rank by
// rank, mapping each rank's s into its own address space -- that read crosses
// the SM-to-SM network, not L2.
// ==========================================================================
template <class T>
__global__ __launch_bounds__(kBlockThreads) void step_cluster_kernel(
    T* __restrict__ x_io, T* __restrict__ y_io, const T* __restrict__ j,
    std::size_t ldj, const float* __restrict__ pump, float delta, float xi,
    float dt, int n, int batch, int n_steps, int rows_per_rank) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int         csize   = int(cluster.num_blocks());
  const int         rank    = int(cluster.block_rank());
  const int         replica = int(blockIdx.x) / csize;

  const int R    = rows_per_rank;
  const int row0 = rank * R;
  const int len  = (row0 >= n) ? 0 : ((n - row0 < R) ? (n - row0) : R);

  extern __shared__ __align__(16) unsigned char raw[];
  float*  x = reinterpret_cast<float*>(raw);
  float*  y = x + R;
  int8_t* s = reinterpret_cast<int8_t*>(y + R);

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < R; i += kBlockThreads) {
    if (i < len) {
      const std::size_t g = std::size_t(row0 + i) * batch + replica;
      x[i] = to_f32(x_io[g]);
      y[i] = to_f32(y_io[g]);
    } else {
      x[i] = 0.f;
      y[i] = 0.f;
    }
  }
  cluster.sync();

  for (int step = 0; step < n_steps; ++step) {
    for (int i = tid; i < R; i += kBlockThreads) s[i] = sign_of(x[i]);
    cluster.sync();

    const float p = pump[step];
    for (int r = warp; r < len; r += kWarps) {
      const T* row = j + std::size_t(row0 + r) * ldj;
      float    acc = 0.f;
      for (int q = 0; q < csize; ++q) {
        const int col0 = q * R;
        if (col0 >= n) break;
        const int     seg    = (n - col0 < R) ? (n - col0) : R;
        const int8_t* remote = cluster.map_shared_rank(s, q);
        acc += dot_segment(row, remote, col0, seg, lane);
      }
      acc = warp_sum(acc);
      if (lane == 0) advance_row(x, y, r, acc, delta, p, xi, dt);
    }
    cluster.sync();
  }

  for (int i = tid; i < len; i += kBlockThreads) {
    const std::size_t g = std::size_t(row0 + i) * batch + replica;
    store_f32(x_io[g], x[i]);
    store_f32(y_io[g], y[i]);
  }
#else
  (void)x_io; (void)y_io; (void)j; (void)ldj; (void)pump; (void)delta;
  (void)xi; (void)dt; (void)n; (void)batch; (void)n_steps; (void)rows_per_rank;
#endif
}

// ==========================================================================
// Baseline: state in device memory, one grid-wide barrier per step.
//
// This is what the design above replaces. Nothing is staged on chip: the sign
// array is written to device memory, every warp re-reads it from there, and the
// per-step barrier is a cooperative-launch grid.sync() rather than a block
// barrier. One warp per (row, replica) pair.
//
// Note the sign read s[col*batch + replica] is strided by `batch`, so it does
// not coalesce -- that is inherent to keeping the state in device memory with
// this layout, and is part of what makes this the slow rung of the ladder.
// ==========================================================================
template <class T>
__global__ __launch_bounds__(256) void step_globalsync_kernel(
    T* __restrict__ x, T* __restrict__ y, T* __restrict__ s,
    const T* __restrict__ j, std::size_t ldj, const float* __restrict__ pump,
    float delta, float xi, float dt, int n, int batch, int n_steps) {
#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();

  const long long total   = (long long)n * batch;
  const long long tid     = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long threads = (long long)gridDim.x * blockDim.x;
  const int       lane    = int(threadIdx.x) & 31;
  const long long gwarp   = tid >> 5;
  const long long nwarps  = threads >> 5;

  for (int step = 0; step < n_steps; ++step) {
    for (long long i = tid; i < total; i += threads)
      store_f32(s[i], float(sign_of(to_f32(x[i]))));
    grid.sync();

    const float p = pump[step];
    for (long long k = gwarp; k < total; k += nwarps) {
      const int row     = int(k / batch);
      const int replica = int(k % batch);

      const T* jrow = j + std::size_t(row) * ldj;
      float    acc  = 0.f;
      for (int c = lane; c < n; c += 32)
        acc += to_f32(jrow[c]) * to_f32(s[(long long)c * batch + replica]);
      acc = warp_sum(acc);

      if (lane == 0) {
        float xv = to_f32(x[k]);
        float yv = to_f32(y[k]);
        yv += (-(delta - p) * xv + xi * acc) * dt;
        xv += dt * yv * delta;
        if (fabsf(xv) > 1.f) {
          xv = (xv > 0.f) ? 1.f : -1.f;
          yv = 0.f;
        }
        store_f32(x[k], xv);
        store_f32(y[k], yv);
      }
    }
    grid.sync();
  }
#else
  (void)x; (void)y; (void)s; (void)j; (void)ldj; (void)pump; (void)delta;
  (void)xi; (void)dt; (void)n; (void)batch; (void)n_steps;
#endif
}

// ==========================================================================
// Ising energy of sign(x). One block per replica, N bytes of shared memory.
// ==========================================================================
template <class T>
__global__ __launch_bounds__(kBlockThreads) void energy_kernel(
    const T* __restrict__ x_io, const T* __restrict__ j, std::size_t ldj,
    int n, int batch, double* __restrict__ energy) {
  const int replica = blockIdx.x;

  extern __shared__ __align__(16) unsigned char raw[];
  int8_t* s = reinterpret_cast<int8_t*>(raw);

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < n; i += kBlockThreads)
    s[i] = sign_of(to_f32(x_io[std::size_t(i) * batch + replica]));
  __syncthreads();

  double local = 0.0;
  for (int row = warp; row < n; row += kWarps) {
    const float acc =
        warp_sum(dot_segment(j + std::size_t(row) * ldj, s, 0, n, lane));
    if (lane == 0) local += double(acc) * double(s[row]);
  }

  __shared__ double total;
  if (tid == 0) total = 0.0;
  __syncthreads();
  if (lane == 0) atomicAdd(&total, local);
  __syncthreads();
  if (tid == 0) energy[replica] = -0.5 * total;
}

// ==========================================================================
// Read-bandwidth probe. Streams a buffer with 16-byte loads and no reuse.
// ==========================================================================
__global__ void bandwidth_kernel(const float4* __restrict__ src,
                                 std::size_t count, float* __restrict__ sink) {
  std::size_t       i      = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  float             acc    = 0.f;
  for (; i < count; i += stride) {
    const float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Never true; keeps the loop from being optimised away.
  if (acc == 3.4e38f) sink[0] = acc;
}

// --------------------------------------------------------------------------
// Shared memory opt-in. Without this a kernel is capped at 48 KB of dynamic
// shared memory regardless of hardware.
// --------------------------------------------------------------------------
template <class F>
void enable_large_smem(F kernel, std::size_t bytes) {
  DSB_CUDA_CHECK(cudaFuncSetAttribute(
      (const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      int(bytes)));
}

template <class T>
void launch_globalsync(T* x, T* y, T* s, const T* j, std::size_t ldj,
                       const float* pump, float delta, float xi, float dt,
                       int n, int batch, int n_steps, cudaStream_t stream) {
  if (s == nullptr)
    throw std::runtime_error("launch_steps: global-sync variant needs scratch");

  int device = 0;
  DSB_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  DSB_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  auto      kernel  = step_globalsync_kernel<T>;
  const int threads = 256;
  int       per_sm  = 0;
  DSB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, (const void*)kernel, threads, 0));

  // A cooperative launch may not ask for more blocks than can be resident.
  const long long want    = ((long long)n * batch + threads / 32 - 1) / (threads / 32);
  const long long ceiling = (long long)per_sm * prop.multiProcessorCount;
  int             blocks  = int(want < ceiling ? want : ceiling);
  if (blocks < 1) blocks = 1;

  std::size_t ldj_arg   = ldj;
  void*       args[]    = {&x,     &y,      &s,       &j,     &ldj_arg,
                           (void*)&pump,    &delta,   &xi,    &dt,
                           &n,     &batch,  &n_steps};
  DSB_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)kernel, dim3(blocks),
                                             dim3(threads), args, 0, stream));
}

template <class T>
void steps_impl(T* x, T* y, const T* j, std::size_t ldj, const float* pump,
                float delta, float xi, float dt, int n, int batch, int n_steps,
                const LaunchPlan& plan, T* scratch, cudaStream_t stream) {
  if (plan.variant == Variant::Gemm)
    throw std::runtime_error(
        "launch_steps: the GEMM variant is driven by GemmStepper, not here");
  if (plan.variant == Variant::Bit)
    throw std::runtime_error(
        "launch_steps: the bit variant is driven by dsb::bitfused, not here");

  if (plan.variant == Variant::GlobalSync) {
    launch_globalsync(x, y, scratch, j, ldj, pump, delta, xi, dt, n, batch,
                      n_steps, stream);
    return;
  }

  if (plan.variant != Variant::Cluster) {
    auto kernel = step_block_kernel<T>;
    enable_large_smem(kernel, plan.smem_bytes);
    kernel<<<dim3(unsigned(batch)), dim3(kBlockThreads), plan.smem_bytes,
             stream>>>(x, y, j, ldj, pump, delta, xi, dt, n, batch, n_steps);
    DSB_CUDA_CHECK(cudaGetLastError());
    return;
  }

  auto kernel = step_cluster_kernel<T>;
  enable_large_smem(kernel, plan.smem_bytes);
  if (plan.cluster_size > 8) {
    DSB_CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }

  cudaLaunchAttribute attr[1];
  attr[0].id               = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = unsigned(plan.cluster_size);
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim          = dim3(unsigned(batch) * unsigned(plan.cluster_size));
  cfg.blockDim         = dim3(kBlockThreads);
  cfg.dynamicSmemBytes = plan.smem_bytes;
  cfg.stream           = stream;
  cfg.attrs            = attr;
  cfg.numAttrs         = 1;

  DSB_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, x, y, j, ldj, pump, delta, xi,
                                    dt, n, batch, n_steps,
                                    plan.rows_per_rank));
}

template <class T>
void energy_impl(const T* x, const T* j, std::size_t ldj, int n, int batch,
                 double* energy, cudaStream_t stream) {
  const std::size_t smem   = std::size_t(n);
  auto              kernel = energy_kernel<T>;
  enable_large_smem(kernel, smem);
  kernel<<<dim3(unsigned(batch)), dim3(kBlockThreads), smem, stream>>>(
      x, j, ldj, n, batch, energy);
  DSB_CUDA_CHECK(cudaGetLastError());
}

int resolve_device(int device) {
  if (device < 0) DSB_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

}  // namespace

// ==========================================================================
// public
// ==========================================================================

const char* to_string(Variant v) {
  switch (v) {
    case Variant::Block:      return "block";
    case Variant::Cluster:    return "cluster";
    case Variant::GlobalSync: return "global-sync";
    case Variant::Gemm:       return "gemm";
    case Variant::CsrRow:     return "csr-row";
    case Variant::CsrBlock:   return "csr-block";
    case Variant::CsrCluster: return "csr-cluster";
    case Variant::Bit:        return "bit";
    default:                  return "auto";
  }
}

bool needs_scratch(Variant v) {
  return v == Variant::GlobalSync;
}

int device_max_smem_optin(int device) {
  device   = resolve_device(device);
  int bytes = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(
      &bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  return bytes;
}

int device_max_n_block(int device) { return device_max_smem_optin(device) / 9; }

int device_l2_bytes(int device) {
  device = resolve_device(device);
  int bytes = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device));
  return bytes;
}

int device_max_persisting_l2_bytes(int device) {
  device = resolve_device(device);
  int bytes = 0;
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(
      &bytes, cudaDevAttrMaxPersistingL2CacheSize, device));
  return bytes;
}

double measure_read_bandwidth(int device) {
  device = resolve_device(device);
  cudaDeviceProp prop;
  DSB_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const std::size_t bytes = std::size_t(1) << 30;  // 1 GiB
  const std::size_t count = bytes / sizeof(float4);

  float4* buffer = nullptr;
  float*  sink   = nullptr;
  DSB_CUDA_CHECK(cudaMalloc(&buffer, bytes));
  DSB_CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  DSB_CUDA_CHECK(cudaMemset(buffer, 0, bytes));

  const int threads = 256;
  const int blocks  = prop.multiProcessorCount * 32;

  cudaEvent_t begin, end;
  DSB_CUDA_CHECK(cudaEventCreate(&begin));
  DSB_CUDA_CHECK(cudaEventCreate(&end));

  bandwidth_kernel<<<blocks, threads>>>(buffer, count, sink);  // warm-up
  DSB_CUDA_CHECK(cudaDeviceSynchronize());

  const int repeats = 5;
  DSB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < repeats; ++i)
    bandwidth_kernel<<<blocks, threads>>>(buffer, count, sink);
  DSB_CUDA_CHECK(cudaEventRecord(end));
  DSB_CUDA_CHECK(cudaEventSynchronize(end));

  float ms = 0.f;
  DSB_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaFree(buffer);
  cudaFree(sink);

  return double(bytes) * repeats / (double(ms) * 1e-3);
}

LaunchPlan make_plan(int n, Precision precision, Variant want,
                     int requested_cluster, int device, int batch) {
  if (n <= 0) throw std::runtime_error("make_plan: n must be positive");
  if (batch <= 0) throw std::runtime_error("make_plan: batch must be positive");
  device = resolve_device(device);

  const int budget = device_max_smem_optin(device);
  (void)precision;

  auto block_plan = [&]() {
    LaunchPlan plan;
    plan.variant       = Variant::Block;
    plan.cluster_size  = 1;
    plan.rows_per_rank = n;
    plan.smem_bytes = std::size_t(9) * std::size_t(n);
    return plan;
  };

  auto cluster_plan = [&](int csize) {
    LaunchPlan plan;
    plan.variant      = Variant::Cluster;
    plan.cluster_size = csize;
    const int per     = (n + csize - 1) / csize;
    plan.rows_per_rank = (per + 7) & ~7;  // keeps each rank's column offset aligned
    plan.smem_bytes   = std::size_t(9) * std::size_t(plan.rows_per_rank);
    return plan;
  };

  auto require_cluster_support = [&]() {
    int ok = 0;
    DSB_CUDA_CHECK(cudaDeviceGetAttribute(&ok, cudaDevAttrClusterLaunch, device));
    if (!ok)
      throw std::runtime_error(
          "make_plan: this GPU does not support thread block clusters");
  };

  auto require_valid_cluster_size = [&](int csize) {
    if (csize != 2 && csize != 4 && csize != 8 && csize != 16)
      throw std::runtime_error(
          "make_plan: cluster size must be 2, 4, 8, or 16");
  };

  auto check_fits = [&](const LaunchPlan& plan) {
    if (plan.smem_bytes > std::size_t(budget))
      throw std::runtime_error(
          "make_plan: variant " + std::string(to_string(plan.variant)) +
          " at N=" + std::to_string(n) + " needs " +
          std::to_string(plan.smem_bytes) +
          " B of shared memory but this GPU allows at most " +
          std::to_string(budget) + " B per block");
  };

  if (want == Variant::GlobalSync || want == Variant::Gemm ||
      want == Variant::Bit || is_csr_variant(want)) {
    // Neither keeps state on chip, so shared memory places no ceiling on N.
    LaunchPlan plan;
    plan.variant       = want;
    plan.cluster_size  = 1;
    plan.rows_per_rank = n;
    plan.smem_bytes    = 0;
    return plan;
  }

  if (want == Variant::Block) {
    LaunchPlan plan = block_plan();
    check_fits(plan);
    return plan;
  }

  if (want == Variant::Cluster) {
    require_cluster_support();
    if (requested_cluster > 0) {
      require_valid_cluster_size(requested_cluster);
      LaunchPlan plan = cluster_plan(requested_cluster);
      check_fits(plan);
      return plan;
    }
    for (int csize : {2, 4, 8, 16}) {
      LaunchPlan plan = cluster_plan(csize);
      if (plan.smem_bytes <= std::size_t(budget)) return plan;
    }
    check_fits(cluster_plan(16));
  }

  // Auto for real-valued J is Gemm.  Measured on GH200: at N=2000, B=200 the
  // block kernel is L2-bandwidth bound at intensity 1 and 4.9x slower than
  // cuBLAS; at N >= 10k, B=1 cluster runs on <= 8 SMs and is 18-56x slower.
  // There is no (N, batch) at which the per-replica kernels beat a
  // tensor-core GEMM on real-valued J, so they are only run when asked for.
  // (Ternary J takes the Bit path in Solver before make_plan is reached.)
  if (requested_cluster <= 1) {
    LaunchPlan plan;
    plan.variant       = Variant::Gemm;
    plan.cluster_size  = 1;
    plan.rows_per_rank = n;
    plan.smem_bytes    = 0;
    return plan;
  }

  // Auto: plain block when it fits, otherwise the smallest cluster that does.
  if (requested_cluster > 1) {
    require_cluster_support();
    require_valid_cluster_size(requested_cluster);
    LaunchPlan plan = cluster_plan(requested_cluster);
    check_fits(plan);
    return plan;
  }

  LaunchPlan plain = block_plan();
  if (plain.smem_bytes <= std::size_t(budget)) return plain;

  require_cluster_support();
  for (int csize : {2, 4, 8, 16}) {
    LaunchPlan plan = cluster_plan(csize);
    if (plan.smem_bytes <= std::size_t(budget)) return plan;
  }

  throw std::runtime_error(
      "make_plan: N=" + std::to_string(n) +
      " does not fit in shared memory even with a cluster of 16 (the ceiling "
      "here is about " + std::to_string(16 * (budget / 9)) +
      "). Reduce the instance further, or move to a GEMM-style solver.");
}

// --------------------------------------------------------------------------

L2Persistence::L2Persistence(cudaStream_t stream, const void* ptr,
                             std::size_t bytes, int device)
    : stream_(stream) {
  if (ptr == nullptr || bytes == 0) return;
  device = resolve_device(device);

  const std::size_t cap = std::size_t(device_max_persisting_l2_bytes(device));
  if (cap == 0) return;
  pinned_ = std::min(bytes, cap);

  DSB_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pinned_));

  cudaStreamAttrValue value = {};
  value.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
  value.accessPolicyWindow.num_bytes = pinned_;
  value.accessPolicyWindow.hitRatio  = 1.0f;
  value.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  value.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  DSB_CUDA_CHECK(cudaStreamSetAttribute(
      stream_, cudaStreamAttributeAccessPolicyWindow, &value));
  active_ = true;
}

L2Persistence::~L2Persistence() {
  if (!active_) return;
  cudaStreamAttrValue value = {};
  value.accessPolicyWindow.base_ptr  = nullptr;
  value.accessPolicyWindow.num_bytes = 0;
  value.accessPolicyWindow.hitRatio  = 0.f;
  value.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
  value.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
  cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &value);
  cudaCtxResetPersistingL2Cache();
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
}

// --------------------------------------------------------------------------

void launch_steps(__half* x, __half* y, const __half* j, std::size_t ldj,
                  const float* pump, float delta, float xi, float dt, int n,
                  int batch, int n_steps, const LaunchPlan& plan,
                  __half* scratch, cudaStream_t stream) {
  steps_impl(x, y, j, ldj, pump, delta, xi, dt, n, batch, n_steps, plan, scratch,
             stream);
}

void launch_steps(float* x, float* y, const float* j, std::size_t ldj,
                  const float* pump, float delta, float xi, float dt, int n,
                  int batch, int n_steps, const LaunchPlan& plan, float* scratch,
                  cudaStream_t stream) {
  steps_impl(x, y, j, ldj, pump, delta, xi, dt, n, batch, n_steps, plan, scratch,
             stream);
}

void launch_energy(const __half* x, const __half* j, std::size_t ldj, int n,
                   int batch, double* energy, cudaStream_t stream) {
  energy_impl(x, j, ldj, n, batch, energy, stream);
}

void launch_energy(const float* x, const float* j, std::size_t ldj, int n,
                   int batch, double* energy, cudaStream_t stream) {
  energy_impl(x, j, ldj, n, batch, energy, stream);
}

}  // namespace dsb
