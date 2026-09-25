// src/gemm.cu
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "dsb/gemm.cuh"

namespace dsb {
namespace {

__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ void store_f32(__half& d, float v) { d = __float2half_rn(v); }
__device__ __forceinline__ void store_f32(float& d, float v) { d = v; }

// sign(0) == -1; see the note in dsb/common.hpp.
__device__ __forceinline__ float sign_of(float v) {
  return (v > 0.f) ? 1.f : -1.f;
}

template <class T>
__global__ void sign_kernel(const T* __restrict__ x, T* __restrict__ s,
                            long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) store_f32(s[i], sign_of(to_f32(x[i])));
}

/// Applies one step to every (row, replica) entry, then writes the sign array
/// the next GEMM will consume -- so a step costs one GEMM and one kernel, not
/// two kernels.
template <class T>
__global__ void update_kernel(T* __restrict__ x, T* __restrict__ y,
                              T* __restrict__ s, const float* __restrict__ acc,
                              const float* __restrict__ pump_step, float delta,
                              float xi, float dt, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;

  const float p  = *pump_step;
  float       xv = to_f32(x[i]);
  float       yv = to_f32(y[i]);

  yv += (-(delta - p) * xv + xi * acc[i]) * dt;
  xv += dt * yv * delta;
  if (fabsf(xv) > 1.f) {
    xv = (xv > 0.f) ? 1.f : -1.f;
    yv = 0.f;
  }
  store_f32(x[i], xv);
  store_f32(y[i], yv);
  store_f32(s[i], sign_of(xv));
}

// ---------------------------------------------------------------------------
// INT8 path.  S is stored REPLICA-major ([B][ldk] int8) because cublasGemmEx
// with CUBLAS_COMPUTE_32I wants A transposed; x, y, acc stay ROW-major
// ([N][B]).  The two kernels below transpose through a 32x33 shared tile so
// both sides are coalesced.  int32 accumulation of +-1 products is exact.
// ---------------------------------------------------------------------------

constexpr int kTile = 32;

/// s[b*ldk + row] = sign(x[row*batch + b])  for the initial step.
__global__ void sign_kernel_int8(const float* __restrict__ x,
                                 std::int8_t* __restrict__ s, int n, int batch,
                                 int ldk) {
  __shared__ std::int8_t tile[kTile][kTile + 1];
  const int row0 = int(blockIdx.y) * kTile;
  const int b0   = int(blockIdx.x) * kTile;
  const int tx = int(threadIdx.x), ty = int(threadIdx.y);  // (32, 8)

  for (int r = ty; r < kTile; r += 8) {
    const int row = row0 + r, b = b0 + tx;
    if (row < n && b < batch)
      tile[r][tx] = std::int8_t(x[std::size_t(row) * batch + b] > 0.f ? 1 : -1);
  }
  __syncthreads();
  for (int c = ty; c < kTile; c += 8) {
    const int b = b0 + c, row = row0 + tx;
    if (row < n && b < batch) s[std::size_t(b) * ldk + row] = tile[tx][c];
  }
}

/// One step for every (row, replica): reads the exact int32 coupling sum,
/// updates x, y in place (FP32, row-major) and writes the next sign into the
/// replica-major int8 S.
__global__ void update_kernel_int8(float* __restrict__ x, float* __restrict__ y,
                                   std::int8_t* __restrict__ s,
                                   const std::int32_t* __restrict__ acc,
                                   const float* __restrict__ pump_step,
                                   float delta, float xi, float dt, int n,
                                   int batch, int ldk) {
  __shared__ std::int8_t tile[kTile][kTile + 1];
  const int row0 = int(blockIdx.y) * kTile;
  const int b0   = int(blockIdx.x) * kTile;
  const int tx = int(threadIdx.x), ty = int(threadIdx.y);
  const float p = *pump_step;

  for (int r = ty; r < kTile; r += 8) {
    const int row = row0 + r, b = b0 + tx;
    if (row < n && b < batch) {
      const std::size_t i  = std::size_t(row) * batch + b;
      float             xv = x[i];
      float             yv = y[i];
      yv += (-(delta - p) * xv + xi * float(acc[i])) * dt;
      xv += dt * yv * delta;
      if (fabsf(xv) > 1.f) {
        xv = (xv > 0.f) ? 1.f : -1.f;
        yv = 0.f;
      }
      x[i]       = xv;
      y[i]       = yv;
      tile[r][tx] = std::int8_t(xv > 0.f ? 1 : -1);
    }
  }
  __syncthreads();
  for (int c = ty; c < kTile; c += 8) {
    const int b = b0 + c, row = row0 + tx;
    if (row < n && b < batch) s[std::size_t(b) * ldk + row] = tile[tx][c];
  }
}

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS error in ") + what + ": " +
                             std::to_string(int(status)));
}

}  // namespace

GemmStepper::GemmStepper(int n, std::size_t ldj, int batch, Precision precision,
                         cudaStream_t stream, bool tf32)
    : n_(n), ldj_(ldj), batch_(batch), precision_(precision), stream_(stream),
      tf32_(tf32) {
  cublas_check(cublasCreate(&handle_), "cublasCreate");
  cublas_check(cublasSetStream(handle_, stream_), "cublasSetStream");
  // FP32 storage + tf32_: the compute type passed to cublasGemmEx below is
  // CUBLAS_COMPUTE_32F_FAST_TF32, which routes the product to the tensor
  // cores.  FP16 storage goes to HMMA regardless.

  const std::size_t count = std::size_t(n_) * batch_;
  if (precision_ == Precision::INT8) {
    // cublasGemmEx + CUBLAS_COMPUTE_32I: m (= batch) and the leading
    // dimensions must be multiples of 4; k is padded to ldj (multiple of 8)
    // and the pad rows of S are zero, as are the pad columns of J.
    if (batch_ % 4 != 0)
      throw std::runtime_error(
          "GemmStepper INT8: batch must be a multiple of 4 (got " +
          std::to_string(batch_) + ")");
    const std::size_t sbytes = std::size_t(batch_) * ldj_;  // [B][ldk] int8
    DSB_CUDA_CHECK(cudaMalloc(&d_sign_, sbytes));
    DSB_CUDA_CHECK(cudaMemset(d_sign_, 0, sbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_acc_, count * sizeof(std::int32_t)));
  } else {
    DSB_CUDA_CHECK(cudaMalloc(&d_sign_, count * coupling_element_size(precision_)));
    DSB_CUDA_CHECK(cudaMalloc(&d_acc_, count * sizeof(float)));
  }

  // Giving cuBLAS an explicit workspace prevents lazy internal allocation
  // while the stream is being captured. Four MiB is the normal cuBLAS default
  // workspace on recent CUDA releases and keeps the extra footprint bounded.
  workspace_bytes_ = std::size_t(4) << 20;
  DSB_CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_bytes_));
  cublas_check(cublasSetWorkspace(handle_, d_workspace_, workspace_bytes_),
               "cublasSetWorkspace");
}

GemmStepper::~GemmStepper() {
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  if (graph_) cudaGraphDestroy(graph_);
  if (d_sign_) cudaFree(d_sign_);
  if (d_acc_) cudaFree(d_acc_);
  if (d_workspace_) cudaFree(d_workspace_);
  if (d_stage_) cudaFree(d_stage_);
  if (handle_) cublasDestroy(handle_);
}

std::size_t GemmStepper::device_bytes() const {
  const std::size_t count = std::size_t(n_) * batch_;
  if (precision_ == Precision::INT8)
    return std::size_t(batch_) * ldj_ + count * sizeof(std::int32_t) +
           workspace_bytes_;
  return count * coupling_element_size(precision_) + count * sizeof(float) +
         workspace_bytes_ +
         std::size_t(stage_rows_) * ldj_ * coupling_element_size(precision_);
}

void GemmStepper::set_row_split(const void* j_tail, int split_rows) {
  if (j_tail == nullptr || split_rows <= 0 || split_rows >= n_) {
    j_tail_ = nullptr;
    split_rows_ = 0;
    return;
  }
  if (precision_ == Precision::INT8)
    throw std::runtime_error("GemmStepper: row split supports FP32/FP16 only");
  j_tail_ = j_tail;
  split_rows_ = split_rows;
}

void GemmStepper::set_grace_staging(std::size_t stage_bytes, bool main_in_grace) {
  if (precision_ == Precision::INT8)
    throw std::runtime_error("GemmStepper: Grace staging supports FP32/FP16 only");
  const std::size_t row_bytes = ldj_ * coupling_element_size(precision_);
  std::size_t rows = std::max<std::size_t>(1, stage_bytes / row_bytes);
  if (rows >= 64) rows -= rows % 64;
  rows = std::min<std::size_t>(rows, std::size_t(n_));
  if (d_stage_) {
    DSB_CUDA_CHECK(cudaFree(d_stage_));
    d_stage_ = nullptr;
  }
  DSB_CUDA_CHECK(cudaMalloc(&d_stage_, rows * row_bytes));
  stage_rows_ = int(rows);
  stage_main_ = main_in_grace;
}

void GemmStepper::prepare(void* x, void* y, const void* j, const float* pump,
                          float delta, float xi, float dt, int n_steps) {
  if (n_steps <= 0)
    throw std::runtime_error("GemmStepper: n_steps must be positive");

  const bool cached = graph_exec_ != nullptr && graph_steps_ == n_steps &&
                      graph_x_ == x && graph_y_ == y && graph_j_ == j &&
                      graph_pump_ == pump && graph_delta_ == delta &&
                      graph_xi_ == xi && graph_dt_ == dt &&
                      graph_j_tail_ == j_tail_ &&
                      graph_split_rows_ == split_rows_ &&
                      graph_stage_rows_ == stage_rows_ &&
                      graph_stage_main_ == stage_main_;
  if (cached) return;

  // A changed step count or buffer set is uncommon, but supported. All users
  // of this class synchronize the timed run before calling it again.
  DSB_CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (graph_exec_) {
    DSB_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
  if (graph_) {
    DSB_CUDA_CHECK(cudaGraphDestroy(graph_));
    graph_ = nullptr;
  }

  const long long total   = (long long)n_ * batch_;
  const int       threads = 256;
  const int       blocks  = int((total + threads - 1) / threads);

  DSB_CUDA_CHECK(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));

  try {
    if (precision_ == Precision::INT8) {
      const dim3 tgrid(unsigned((batch_ + kTile - 1) / kTile),
                       unsigned((n_ + kTile - 1) / kTile));
      const dim3 tblock(kTile, 8);
      sign_kernel_int8<<<tgrid, tblock, 0, stream_>>>(
          static_cast<const float*>(x), static_cast<std::int8_t*>(d_sign_),
          n_, batch_, int(ldj_));
      DSB_CUDA_CHECK(cudaGetLastError());

      // C[m=B x n=N] = A^T[B x k] * B[k x N] with k = ldj (zero padded):
      //   A  = S stored k x m column-major  == [B][ldk] replica-major int8
      //   B  = J stored k x n column-major  == J row-major (J is symmetric)
      //   C  = acc stored m x n column-major == [N][B] row-major int32
      const std::int32_t ialpha = 1, ibeta = 0;
      for (int step = 0; step < n_steps; ++step) {
        cublas_check(
            cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, batch_, n_,
                         int(ldj_), &ialpha, d_sign_, CUDA_R_8I, int(ldj_), j,
                         CUDA_R_8I, int(ldj_), &ibeta, d_acc_, CUDA_R_32I,
                         batch_, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
            "cublasGemmEx(int8)");
        update_kernel_int8<<<tgrid, tblock, 0, stream_>>>(
            static_cast<float*>(x), static_cast<float*>(y),
            static_cast<std::int8_t*>(d_sign_),
            static_cast<const std::int32_t*>(static_cast<const void*>(d_acc_)),
            pump + step, delta, xi, dt, n_, batch_, int(ldj_));
        DSB_CUDA_CHECK(cudaGetLastError());
      }
    } else {
    if (precision_ == Precision::FP16) {
      sign_kernel<__half><<<blocks, threads, 0, stream_>>>(
          static_cast<const __half*>(x), static_cast<__half*>(d_sign_), total);
    } else {
      sign_kernel<float><<<blocks, threads, 0, stream_>>>(
          static_cast<const float*>(x), static_cast<float*>(d_sign_), total);
    }
    DSB_CUDA_CHECK(cudaGetLastError());

    const cudaDataType data_type =
        (precision_ == Precision::FP16) ? CUDA_R_16F : CUDA_R_32F;
    const cublasComputeType_t compute =
        (precision_ == Precision::FP32 && tf32_) ? CUBLAS_COMPUTE_32F_FAST_TF32
                                                 : CUBLAS_COMPUTE_32F;
    const float alpha = 1.f;
    const float beta  = 0.f;

    // Our buffers are row-major N x B, which is column-major B x N. Read that
    // way, acc^T = S^T * J^T is a plain column-major GEMM with m = B, n = k =
    // N, and J's row stride becomes the leading dimension of J^T.  Row i of J
    // is column i of J^T, i.e. output column i of acc^T, so a row range of J
    // is a range of the GEMM's n dimension.
    auto gemm_block = [&](const void* block, int row0, int rows,
                          const char* what) {
      cublas_check(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, batch_,
                                rows, n_, &alpha, d_sign_, data_type, batch_,
                                block, data_type, int(ldj_), &beta,
                                d_acc_ + std::size_t(row0) * batch_,
                                CUDA_R_32F, batch_, compute,
                                CUBLAS_GEMM_DEFAULT),
                   what);
    };
    // Rows [row0, row0 + rows) of J, stored from `block` on.  Grace-resident
    // blocks go through the HBM staging buffer when staging is on; the copy
    // and the GEMM share stream_, so the next copy waits for the GEMM that
    // reads the buffer.
    const std::size_t row_bytes = ldj_ * coupling_element_size(precision_);
    auto gemm_rows = [&](const void* block, int row0, int rows, bool in_grace) {
      if (!in_grace || stage_rows_ == 0) {
        gemm_block(block, row0, rows, "cublasGemmEx");
        return;
      }
      for (int r = 0; r < rows; r += stage_rows_) {
        const int count = std::min(stage_rows_, rows - r);
        DSB_CUDA_CHECK(cudaMemcpyAsync(
            d_stage_, static_cast<const char*>(block) + std::size_t(r) * row_bytes,
            std::size_t(count) * row_bytes, cudaMemcpyDefault, stream_));
        gemm_block(d_stage_, row0 + r, count, "cublasGemmEx(staged)");
      }
    };

    for (int step = 0; step < n_steps; ++step) {
      if (split_rows_ == 0) {
        gemm_rows(j, 0, n_, stage_main_);
      } else {
        // Hybrid: rows [0, R) from the HBM block, rows [R, N) from the tail.
        gemm_rows(j, 0, split_rows_, false);
        gemm_rows(j_tail_, split_rows_, n_ - split_rows_, true);
      }

      if (precision_ == Precision::FP16) {
        update_kernel<__half><<<blocks, threads, 0, stream_>>>(
            static_cast<__half*>(x), static_cast<__half*>(y),
            static_cast<__half*>(d_sign_), d_acc_, pump + step, delta, xi, dt,
            total);
      } else {
        update_kernel<float><<<blocks, threads, 0, stream_>>>(
            static_cast<float*>(x), static_cast<float*>(y),
            static_cast<float*>(d_sign_), d_acc_, pump + step, delta, xi, dt,
            total);
      }
      DSB_CUDA_CHECK(cudaGetLastError());
    }
    }  // FP16 / FP32
  } catch (...) {
    cudaGraph_t abandoned = nullptr;
    cudaStreamEndCapture(stream_, &abandoned);
    if (abandoned) cudaGraphDestroy(abandoned);
    throw;
  }

  DSB_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
  DSB_CUDA_CHECK(
      cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

  graph_steps_ = n_steps;
  graph_x_ = x;
  graph_y_ = y;
  graph_j_ = j;
  graph_pump_ = pump;
  graph_delta_ = delta;
  graph_xi_ = xi;
  graph_dt_ = dt;
  graph_j_tail_ = j_tail_;
  graph_split_rows_ = split_rows_;
  graph_stage_rows_ = stage_rows_;
  graph_stage_main_ = stage_main_;
}

void GemmStepper::run(void* x, void* y, const void* j, const float* pump,
                      float delta, float xi, float dt, int n_steps) {
  prepare(x, y, j, pump, delta, xi, dt, n_steps);
  DSB_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
}

}  // namespace dsb
