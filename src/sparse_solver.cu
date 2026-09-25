#include "dsb/sparse_solver.hpp"

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace dsb {
namespace {

namespace cg = cooperative_groups;

constexpr int kEdgeTile = 256;

// sign(0) == -1; see the note in dsb/common.hpp.
__device__ __forceinline__ int8_t sparse_sign(float value) {
  return value > 0.f ? int8_t(1) : int8_t(-1);
}

__global__ void snapshot_sign_kernel(const float* __restrict__ x,
                                     int8_t* __restrict__ sign,
                                     std::size_t count) {
  const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) sign[index] = sparse_sign(x[index]);
}

__global__ void sparse_step_kernel(
    float* __restrict__ x, float* __restrict__ y,
    const int8_t* __restrict__ sign, int8_t* __restrict__ next_sign,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const float* __restrict__ values,
    float pump, float delta, float xi, float dt, int n, int batch) {
  const int row = int(blockIdx.x);
  if (row >= n) return;
  const int replica = int(blockIdx.y) * blockDim.x + threadIdx.x;
  const bool active = replica < batch;

  __shared__ int columns[kEdgeTile];
  __shared__ float weights[kEdgeTile];
  const int begin = row_ptr[row];
  const int end = row_ptr[row + 1];

  float acc = 0.f;
  for (int tile = begin; tile < end; tile += kEdgeTile) {
    const int count = min(kEdgeTile, end - tile);
    if (threadIdx.x < count) {
      columns[threadIdx.x] = col_idx[tile + threadIdx.x];
      weights[threadIdx.x] = values[tile + threadIdx.x];
    }
    __syncthreads();
    if (active) {
      for (int k = 0; k < count; ++k)
        acc += weights[k] * float(sign[std::size_t(columns[k]) * batch + replica]);
    }
    __syncthreads();
  }

  if (active) {
    const std::size_t index = std::size_t(row) * batch + replica;
    float xv = x[index];
    float yv = y[index];
    yv += (-(delta - pump) * xv + xi * acc) * dt;
    xv += dt * yv * delta;
    if (fabsf(xv) > 1.f) {
      xv = xv > 0.f ? 1.f : -1.f;
      yv = 0.f;
    }
    x[index] = xv;
    y[index] = yv;
    next_sign[index] = sparse_sign(xv);
  }
}

// Row ownership plus full temporal fusion. Each task is a (row, replica-tile)
// pair, so a CSR edge loaded into shared memory serves up to 256 replicas. The
// cooperative grid barrier preserves the Jacobi step boundary while all steps
// remain inside one kernel launch.
//
// The sign array is double buffered: a step reads `cur`, and each task writes
// the sign of its freshly updated x into `nxt` as part of the update, so one
// grid.sync() per step is enough (the previous version snapshotted signs in a
// separate pass and paid two barriers per step -- on large sparse G-set
// instances the barriers, not the arithmetic, are the step time).
__global__ __launch_bounds__(256) void csr_row_fused_kernel(
    float* __restrict__ x, float* __restrict__ y,
    int8_t* __restrict__ sign0, int8_t* __restrict__ sign1,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const float* __restrict__ values,
    const float* __restrict__ pump, float delta, float xi, float dt, int n,
    int batch, int n_steps) {
#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();
  __shared__ int columns[kEdgeTile];
  __shared__ float weights[kEdgeTile];

  const int replica_tiles = (batch + int(blockDim.x) - 1) / int(blockDim.x);
  const long long tasks = (long long)n * replica_tiles;

  // Initial sign snapshot into buffer 0.
  for (long long task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int row = int(task / replica_tiles);
    const int tile = int(task - (long long)row * replica_tiles);
    const int replica = tile * int(blockDim.x) + int(threadIdx.x);
    if (replica < batch) {
      const std::size_t index = std::size_t(row) * batch + replica;
      sign0[index] = sparse_sign(x[index]);
    }
  }
  grid.sync();

  int8_t* cur = sign0;
  int8_t* nxt = sign1;
  for (int step = 0; step < n_steps; ++step) {
    const float p = pump[step];
    for (long long task = blockIdx.x; task < tasks; task += gridDim.x) {
      const int row = int(task / replica_tiles);
      const int tile = int(task - (long long)row * replica_tiles);
      const int replica = tile * int(blockDim.x) + int(threadIdx.x);
      const bool active = replica < batch;
      const int begin = row_ptr[row];
      const int end = row_ptr[row + 1];
      float acc = 0.f;
      for (int edge0 = begin; edge0 < end; edge0 += kEdgeTile) {
        const int count = min(kEdgeTile, end - edge0);
        if (threadIdx.x < count) {
          columns[threadIdx.x] = col_idx[edge0 + threadIdx.x];
          weights[threadIdx.x] = values[edge0 + threadIdx.x];
        }
        __syncthreads();
        if (active) {
          for (int k = 0; k < count; ++k)
            acc += weights[k] *
                   float(cur[std::size_t(columns[k]) * batch + replica]);
        }
        __syncthreads();
      }

      if (active) {
        const std::size_t index = std::size_t(row) * batch + replica;
        float xv = x[index];
        float yv = y[index];
        yv += (-(delta - p) * xv + xi * acc) * dt;
        xv += dt * yv * delta;
        if (fabsf(xv) > 1.f) {
          xv = xv > 0.f ? 1.f : -1.f;
          yv = 0.f;
        }
        x[index] = xv;
        y[index] = yv;
        nxt[index] = sparse_sign(xv);
      }
    }
    grid.sync();
    int8_t* t = cur;
    cur = nxt;
    nxt = t;
  }
#else
  (void)x; (void)y; (void)sign0; (void)sign1; (void)row_ptr; (void)col_idx;
  (void)values; (void)pump; (void)delta; (void)xi; (void)dt; (void)n;
  (void)batch; (void)n_steps;
#endif
}

// One persistent block owns one replica.  This deliberately simple baseline
// assigns one row to each thread and walks that row's neighbours serially.
__global__ void csr_block_global_kernel(
    float* __restrict__ x, float* __restrict__ y,
    int8_t* __restrict__ sign, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const float* __restrict__ values,
    const float* __restrict__ pump, float delta, float xi, float dt, int n,
    int batch, int n_steps) {
  const int replica = int(blockIdx.x);
  for (int step = 0; step < n_steps; ++step) {
    for (int row = int(threadIdx.x); row < n; row += int(blockDim.x)) {
      const std::size_t index = std::size_t(row) * batch + replica;
      sign[index] = sparse_sign(x[index]);
    }
    __syncthreads();

    const float p = pump[step];
    for (int row = int(threadIdx.x); row < n; row += int(blockDim.x)) {
      float acc = 0.f;
      for (int edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge)
        acc += values[edge] *
               float(sign[std::size_t(col_idx[edge]) * batch + replica]);
      const std::size_t index = std::size_t(row) * batch + replica;
      float xv = x[index];
      float yv = y[index];
      yv += (-(delta - p) * xv + xi * acc) * dt;
      xv += dt * yv * delta;
      if (fabsf(xv) > 1.f) {
        xv = xv > 0.f ? 1.f : -1.f;
        yv = 0.f;
      }
      x[index] = xv;
      y[index] = yv;
    }
    __syncthreads();
  }
}

// The fast persistent path keeps one replica's complete state on chip.  It is
// the corrected form of the original fused solver: sign(x) is snapshotted
// before any row is updated, so every step remains a Jacobi sweep.
__global__ void csr_block_shared_kernel(
    float* __restrict__ x_io, float* __restrict__ y_io,
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ values, const float* __restrict__ pump,
    float delta, float xi, float dt, int n, int batch, int n_steps) {
  const int replica = int(blockIdx.x);
  extern __shared__ __align__(16) unsigned char raw[];
  float* sx = reinterpret_cast<float*>(raw);
  float* sy = sx + n;
  int8_t* sign = reinterpret_cast<int8_t*>(sy + n);

  for (int row = int(threadIdx.x); row < n; row += int(blockDim.x)) {
    const std::size_t index = std::size_t(row) * batch + replica;
    sx[row] = x_io[index];
    sy[row] = y_io[index];
  }
  __syncthreads();

  for (int step = 0; step < n_steps; ++step) {
    for (int row = int(threadIdx.x); row < n; row += int(blockDim.x))
      sign[row] = sparse_sign(sx[row]);
    __syncthreads();

    const float p = pump[step];
    for (int row = int(threadIdx.x); row < n; row += int(blockDim.x)) {
      float acc = 0.f;
      for (int edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge)
        acc += values[edge] * float(sign[col_idx[edge]]);
      float xv = sx[row];
      float yv = sy[row];
      yv += (-(delta - p) * xv + xi * acc) * dt;
      xv += dt * yv * delta;
      if (fabsf(xv) > 1.f) {
        xv = xv > 0.f ? 1.f : -1.f;
        yv = 0.f;
      }
      sx[row] = xv;
      sy[row] = yv;
    }
    __syncthreads();
  }

  for (int row = int(threadIdx.x); row < n; row += int(blockDim.x)) {
    const std::size_t index = std::size_t(row) * batch + replica;
    x_io[index] = sx[row];
    y_io[index] = sy[row];
  }
}

__device__ __forceinline__ float csr_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// A persistent Hopper cluster owns one replica.  Rows are distributed across
// ranks and cluster.sync() is the per-step barrier.  State remains in global
// memory, so unlike the dense DSMEM kernel this variant has no N-dependent
// shared-memory ceiling.
__global__ void csr_cluster_step_kernel(
    float* __restrict__ x, float* __restrict__ y,
    int8_t* __restrict__ sign, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const float* __restrict__ values,
    const float* __restrict__ pump, float delta, float xi, float dt, int n,
    int batch, int n_steps) {
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int csize = int(cluster.num_blocks());
  const int rank = int(cluster.block_rank());
  const int replica = int(blockIdx.x) / csize;
  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
  const int warps = int(blockDim.x) >> 5;

  for (int step = 0; step < n_steps; ++step) {
    for (int row = rank * int(blockDim.x) + int(threadIdx.x); row < n;
         row += csize * int(blockDim.x)) {
      const std::size_t index = std::size_t(row) * batch + replica;
      sign[index] = sparse_sign(x[index]);
    }
    cluster.sync();

    const float p = pump[step];
    for (int row = rank * warps + warp; row < n; row += csize * warps) {
      float acc = 0.f;
      for (int edge = row_ptr[row] + lane; edge < row_ptr[row + 1]; edge += 32)
        acc += values[edge] *
               float(sign[std::size_t(col_idx[edge]) * batch + replica]);
      acc = csr_warp_sum(acc);
      if (lane == 0) {
        const std::size_t index = std::size_t(row) * batch + replica;
        float xv = x[index];
        float yv = y[index];
        yv += (-(delta - p) * xv + xi * acc) * dt;
        xv += dt * yv * delta;
        if (fabsf(xv) > 1.f) {
          xv = xv > 0.f ? 1.f : -1.f;
          yv = 0.f;
        }
        x[index] = xv;
        y[index] = yv;
      }
    }
    cluster.sync();
  }
#else
  (void)x; (void)y; (void)sign; (void)row_ptr; (void)col_idx; (void)values;
  (void)pump; (void)delta; (void)xi; (void)dt; (void)n; (void)batch;
  (void)n_steps;
#endif
}

__global__ void sparse_energy_kernel(
    const int8_t* __restrict__ sign, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const float* __restrict__ values,
    int n, int batch, double* __restrict__ energy) {
  const int row = int(blockIdx.x);
  if (row >= n) return;
  const int replica = int(blockIdx.y) * blockDim.x + threadIdx.x;
  if (replica >= batch) return;
  const int begin = row_ptr[row];
  const int end = row_ptr[row + 1];
  float acc = 0.f;
  for (int p = begin; p < end; ++p)
    acc += values[p] *
           float(sign[std::size_t(col_idx[p]) * batch + replica]);
  const double contribution =
      -0.5 * double(acc) * double(sign[std::size_t(row) * batch + replica]);
  atomicAdd(energy + replica, contribution);
}

}  // namespace

SparseSolver::SparseSolver(const SparseMatrix& coupling, const Options& options)
    : options_(options), n_(coupling.n), nnz_(int(coupling.nnz())) {
  coupling.validate();
  if (options_.precision != Precision::FP32)
    throw std::runtime_error("SparseSolver currently requires --precision=fp32");
  if (options_.batch <= 0 || options_.n_steps <= 0)
    throw std::runtime_error("SparseSolver: batch and steps must be positive");
  threads_ = 256;
  int device = 0;
  DSB_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  DSB_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const int requested_threads = std::min(n_, properties.maxThreadsPerBlock);
  persistent_threads_ = std::max(32, ((requested_threads + 31) / 32) * 32);
  if (!is_csr_variant(options_.variant)) options_.variant = Variant::CsrRow;
  if (options_.variant == Variant::CsrRow) {
    int cooperative = 0;
    DSB_CUDA_CHECK(cudaDeviceGetAttribute(
        &cooperative, cudaDevAttrCooperativeLaunch, device));
    if (cooperative) {
      int per_sm = 0;
      DSB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, (const void*)csr_row_fused_kernel, threads_, 0));
      const long long replica_tiles =
          (options_.batch + threads_ - 1) / threads_;
      const long long tasks = (long long)n_ * replica_tiles;
      const long long ceiling =
          (long long)per_sm * properties.multiProcessorCount;
      row_blocks_ = int(std::min(tasks, ceiling));
      row_uses_fused_ = row_blocks_ > 0;
    }
  }
  if (options_.variant == Variant::CsrBlock) {
    block_smem_bytes_ = std::size_t(9) * std::size_t(n_);
    int max_smem = 0;
    DSB_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    block_uses_shared_state_ =
        block_smem_bytes_ <= std::size_t(std::max(0, max_smem));
    if (block_uses_shared_state_ &&
        block_smem_bytes_ > std::size_t(properties.sharedMemPerBlock)) {
      DSB_CUDA_CHECK(cudaFuncSetAttribute(
          (const void*)csr_block_shared_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(block_smem_bytes_)));
    }
  }
  if (options_.variant == Variant::CsrCluster) {
    cluster_size_ = options_.cluster > 0 ? options_.cluster : 8;
    if (cluster_size_ != 2 && cluster_size_ != 4 && cluster_size_ != 8 &&
        cluster_size_ != 16)
      throw std::runtime_error(
          "SparseSolver: CSR cluster size must be 2, 4, 8, or 16");
    int supported = 0;
    DSB_CUDA_CHECK(cudaDeviceGetAttribute(
        &supported, cudaDevAttrClusterLaunch, device));
    if (!supported)
      throw std::runtime_error(
          "SparseSolver: this GPU does not support thread block clusters");
  }

  double sumsq = 0.0;
  for (float value : coupling.values) sumsq += double(value) * double(value);
  xi_ = options_.xi;
  if (!std::isfinite(xi_) || xi_ <= 0.f)
    xi_ = sumsq > 0.0
              ? float(0.5 * std::sqrt(double(n_ - 1)) / std::sqrt(sumsq))
              : 0.f;

  std::vector<float> pump(std::size_t(options_.n_steps));
  for (int step = 0; step < options_.n_steps; ++step)
    pump[std::size_t(step)] = options_.n_steps == 1
                                  ? 0.f
                                  : float(step) / float(options_.n_steps - 1);

  const std::size_t state_count = std::size_t(n_) * options_.batch;
  std::vector<float> x(state_count), y(state_count);
  std::mt19937 rng(options_.seed);
  std::uniform_real_distribution<float> uniform(-0.01f, 0.01f);
  for (std::size_t i = 0; i < state_count; ++i) {
    x[i] = uniform(rng);
    y[i] = uniform(rng);
  }

  coupling_bytes_ = coupling.bytes();
  device_bytes_ = coupling_bytes_ +
                  state_count * (2 * sizeof(float) + sizeof(int8_t)) +
                  pump.size() * sizeof(float) +
                  std::size_t(options_.batch) * sizeof(double);
  if (options_.variant == Variant::CsrRow)
    device_bytes_ += state_count * sizeof(int8_t);  // second sign buffer

  try {
    DSB_CUDA_CHECK(cudaMalloc(&d_row_ptr_, coupling.row_ptr.size() * sizeof(int)));
    DSB_CUDA_CHECK(cudaMalloc(&d_col_idx_, coupling.col_idx.size() * sizeof(int)));
    DSB_CUDA_CHECK(cudaMalloc(&d_values_, coupling.values.size() * sizeof(float)));
    DSB_CUDA_CHECK(cudaMalloc(&d_x_, state_count * sizeof(float)));
    DSB_CUDA_CHECK(cudaMalloc(&d_y_, state_count * sizeof(float)));
    DSB_CUDA_CHECK(cudaMalloc(&d_sign_, state_count * sizeof(int8_t)));
    if (options_.variant == Variant::CsrRow)
      DSB_CUDA_CHECK(cudaMalloc(&d_sign_next_, state_count * sizeof(int8_t)));
    DSB_CUDA_CHECK(cudaMalloc(&d_pump_, pump.size() * sizeof(float)));
    DSB_CUDA_CHECK(cudaMalloc(&d_energy_, std::size_t(options_.batch) * sizeof(double)));
    DSB_CUDA_CHECK(cudaStreamCreate(&stream_));
    DSB_CUDA_CHECK(cudaEventCreate(&begin_));
    DSB_CUDA_CHECK(cudaEventCreate(&end_));

    DSB_CUDA_CHECK(cudaMemcpy(d_row_ptr_, coupling.row_ptr.data(),
                              coupling.row_ptr.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_col_idx_, coupling.col_idx.data(),
                              coupling.col_idx.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_values_, coupling.values.data(),
                              coupling.values.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_x_, x.data(), state_count * sizeof(float),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_y_, y.data(), state_count * sizeof(float),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_pump_, pump.data(), pump.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    if (options_.variant == Variant::CsrRow && !row_uses_fused_)
      prepare_row_graph();
  } catch (...) {
    free_device();
    throw;
  }
}

SparseSolver::~SparseSolver() { free_device(); }

void SparseSolver::prepare_row_graph() {
  if (row_graph_exec_) return;
  if (!d_sign_next_)
    throw std::runtime_error("SparseSolver: csr-row sign buffer is missing");

  const std::size_t count = std::size_t(n_) * options_.batch;
  const int sign_blocks = int((count + 255) / 256);
  const dim3 grid(unsigned(n_),
                  unsigned((options_.batch + threads_ - 1) / threads_));

  DSB_CUDA_CHECK(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  snapshot_sign_kernel<<<sign_blocks, 256, 0, stream_>>>(d_x_, d_sign_, count);
  int8_t* current = d_sign_;
  int8_t* next = d_sign_next_;
  for (int step = 0; step < options_.n_steps; ++step) {
    sparse_step_kernel<<<grid, threads_, 0, stream_>>>(
        d_x_, d_y_, current, next, d_row_ptr_, d_col_idx_, d_values_,
        float(step) / float(std::max(1, options_.n_steps - 1)), options_.delta,
        xi_, options_.dt, n_, options_.batch);
    std::swap(current, next);
  }
  DSB_CUDA_CHECK(cudaStreamEndCapture(stream_, &row_graph_));
  DSB_CUDA_CHECK(
      cudaGraphInstantiate(&row_graph_exec_, row_graph_, nullptr, nullptr, 0));
}

void SparseSolver::free_device() noexcept {
  if (row_graph_exec_) cudaGraphExecDestroy(row_graph_exec_);
  if (row_graph_) cudaGraphDestroy(row_graph_);
  if (d_row_ptr_) cudaFree(d_row_ptr_);
  if (d_col_idx_) cudaFree(d_col_idx_);
  if (d_values_) cudaFree(d_values_);
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_sign_) cudaFree(d_sign_);
  if (d_sign_next_) cudaFree(d_sign_next_);
  if (d_pump_) cudaFree(d_pump_);
  if (d_energy_) cudaFree(d_energy_);
  if (begin_) cudaEventDestroy(begin_);
  if (end_) cudaEventDestroy(end_);
  if (stream_) cudaStreamDestroy(stream_);
  d_row_ptr_ = d_col_idx_ = nullptr;
  d_values_ = d_x_ = d_y_ = d_pump_ = nullptr;
  d_sign_ = d_sign_next_ = nullptr;
  d_energy_ = nullptr;
  row_graph_ = nullptr;
  row_graph_exec_ = nullptr;
  begin_ = end_ = nullptr;
  stream_ = nullptr;
}

void SparseSolver::run() {
  DSB_CUDA_CHECK(cudaEventRecord(begin_, stream_));
  if (options_.variant == Variant::CsrRow) {
    if (row_uses_fused_) {
      float delta = options_.delta;
      float xi = xi_;
      float dt = options_.dt;
      int n = n_;
      int batch = options_.batch;
      int n_steps = options_.n_steps;
      void* args[] = {&d_x_,       &d_y_,      &d_sign_,  &d_sign_next_,
                      &d_row_ptr_, &d_col_idx_, &d_values_, &d_pump_,
                      &delta,      &xi,        &dt,       &n,
                      &batch,      &n_steps};
      DSB_CUDA_CHECK(cudaLaunchCooperativeKernel(
          (const void*)csr_row_fused_kernel, dim3(unsigned(row_blocks_)),
          dim3(unsigned(threads_)), args, 0, stream_));
    } else {
      DSB_CUDA_CHECK(cudaGraphLaunch(row_graph_exec_, stream_));
    }
  } else if (options_.variant == Variant::CsrBlock) {
    if (block_uses_shared_state_) {
      csr_block_shared_kernel<<<unsigned(options_.batch), persistent_threads_,
                                block_smem_bytes_, stream_>>>(
          d_x_, d_y_, d_row_ptr_, d_col_idx_, d_values_, d_pump_,
          options_.delta, xi_, options_.dt, n_, options_.batch,
          options_.n_steps);
    } else {
      csr_block_global_kernel<<<unsigned(options_.batch), persistent_threads_,
                                0, stream_>>>(
          d_x_, d_y_, d_sign_, d_row_ptr_, d_col_idx_, d_values_, d_pump_,
          options_.delta, xi_, options_.dt, n_, options_.batch,
          options_.n_steps);
    }
  } else {
    auto kernel = csr_cluster_step_kernel;
    if (cluster_size_ > 8)
      DSB_CUDA_CHECK(cudaFuncSetAttribute(
          (const void*)kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
          1));
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = unsigned(cluster_size_);
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(unsigned(options_.batch * cluster_size_));
    config.blockDim = dim3(unsigned(threads_));
    config.stream = stream_;
    config.attrs = attribute;
    config.numAttrs = 1;
    DSB_CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel, d_x_, d_y_, d_sign_, d_row_ptr_, d_col_idx_,
        d_values_, d_pump_, options_.delta, xi_, options_.dt, n_,
        options_.batch, options_.n_steps));
  }
  DSB_CUDA_CHECK(cudaGetLastError());
  DSB_CUDA_CHECK(cudaEventRecord(end_, stream_));
  DSB_CUDA_CHECK(cudaEventSynchronize(end_));
  DSB_CUDA_CHECK(cudaEventElapsedTime(&last_ms_, begin_, end_));
  ran_ = true;
}

std::vector<double> SparseSolver::energies() const {
  if (!ran_) throw std::runtime_error("SparseSolver::energies called before run");
  const std::size_t count = std::size_t(n_) * options_.batch;
  const int sign_blocks = int((count + 255) / 256);
  snapshot_sign_kernel<<<sign_blocks, 256, 0, stream_>>>(d_x_, d_sign_, count);
  DSB_CUDA_CHECK(cudaMemsetAsync(d_energy_, 0,
                                 std::size_t(options_.batch) * sizeof(double),
                                 stream_));
  const dim3 grid(unsigned(n_),
                  unsigned((options_.batch + threads_ - 1) / threads_));
  sparse_energy_kernel<<<grid, threads_, 0, stream_>>>(
      d_sign_, d_row_ptr_, d_col_idx_, d_values_, n_, options_.batch, d_energy_);
  DSB_CUDA_CHECK(cudaGetLastError());
  std::vector<double> out(std::size_t(options_.batch));
  DSB_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_energy_,
                                 out.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost, stream_));
  DSB_CUDA_CHECK(cudaStreamSynchronize(stream_));
  return out;
}

int SparseSolver::best_replica() const {
  const std::vector<double> values = energies();
  return int(std::min_element(values.begin(), values.end()) - values.begin());
}

Eigen::VectorXd SparseSolver::best_spins() const {
  const int best = best_replica();
  std::vector<float> column(static_cast<std::size_t>(n_));
  DSB_CUDA_CHECK(cudaMemcpy2DAsync(
      column.data(), sizeof(float), d_x_ + best,
      std::size_t(options_.batch) * sizeof(float), sizeof(float), n_,
      cudaMemcpyDeviceToHost, stream_));
  DSB_CUDA_CHECK(cudaStreamSynchronize(stream_));
  Eigen::VectorXd out(n_);
  for (int row = 0; row < n_; ++row) {
    const float value = column[std::size_t(row)];
    out(row) = value > 0.f ? 1.0 : -1.0;
  }
  return out;
}

}  // namespace dsb
