#include "dsb/dense_capacity.hpp"

#include <cuda_fp16.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsb {
namespace {

__device__ __forceinline__ unsigned long long mix64(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ void store_value(float* dst, float v) { *dst = v; }
__device__ __forceinline__ void store_value(__half* dst, float v) {
  *dst = __float2half_rn(v);
}

// `matrix` holds rows [row0, row0 + count/ldj) of J; the values depend only
// on the global (row, column), so a row-split matrix is bit-identical to the
// single-block one.
template <class T>
__global__ void generate_dense_symmetric_kernel(
    T* matrix, int n, std::size_t ldj, unsigned long long seed,
    unsigned long long count, unsigned long long row0) {
  const unsigned long long stride =
      (unsigned long long)gridDim.x * blockDim.x;
  const float scale = rsqrtf(float(n));
  for (unsigned long long index =
           (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const unsigned long long local_row = index / ldj;
    const unsigned long long column = index - local_row * ldj;
    const unsigned long long row = row0 + local_row;
    if (column >= (unsigned long long)n || row == column) {
      store_value(matrix + index, 0.f);
      continue;
    }
    const unsigned long long lo = row < column ? row : column;
    const unsigned long long hi = row < column ? column : row;
    const unsigned long long key =
        seed ^ (lo * 0xd6e8feb86659fd93ULL) ^
        (hi * 0xa5a3564e27f8862fULL);
    const unsigned int bits = unsigned(mix64(key) >> 40);
    const float uniform = float(bits) * (1.0f / 8388608.0f) - 1.0f;
    store_value(matrix + index, uniform * scale);
  }
}

template <class T>
__global__ void initialize_state_kernel(T* x, T* y,
                                        unsigned long long count,
                                        unsigned long long seed) {
  const unsigned long long stride =
      (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long index =
           (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const unsigned long long hx = mix64(seed ^ index);
    const unsigned long long hy = mix64((seed + 1) ^ index);
    store_value(x + index,
                (float(unsigned(hx >> 40)) * (1.0f / 8388608.0f) - 1.0f) * 0.01f);
    store_value(y + index,
                (float(unsigned(hy >> 40)) * (1.0f / 8388608.0f) - 1.0f) * 0.01f);
  }
}

// HBM staging block for Grace-resident rows of J (GemmStepper::
// set_grace_staging).  Big enough that copies run at full C2C rate and a step
// is a few hundred GEMMs at most.
constexpr std::size_t kGraceStageBytes = std::size_t(2) << 30;

int launch_blocks() {
  int device = 0;
  DSB_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  DSB_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  return std::max(1, properties.multiProcessorCount * 32);
}

}  // namespace

const char* to_string(MatrixMemory memory) {
  switch (memory) {
    case MatrixMemory::Hbm: return "hbm";
    case MatrixMemory::Grace: return "grace";
    case MatrixMemory::Hybrid: return "hybrid";
    default: return "auto";
  }
}

MatrixMemory parse_matrix_memory(const std::string& value) {
  if (value == "auto") return MatrixMemory::Auto;
  if (value == "hbm") return MatrixMemory::Hbm;
  if (value == "grace") return MatrixMemory::Grace;
  if (value == "hybrid") return MatrixMemory::Hybrid;
  throw std::runtime_error("unknown matrix memory: " + value);
}

DenseCapacitySolver::DenseCapacitySolver(int n, const Options& options,
                                         MatrixMemory memory,
                                         double hbm_fraction)
    : options_(options), n_(n), ldj_(row_stride(n)) {
  if (n_ <= 0 || options_.batch <= 0 || options_.n_steps <= 0)
    throw std::runtime_error("dense capacity: N, batch and steps must be positive");
  if (options_.variant == Variant::Bit || options_.precision == Precision::INT8)
    throw std::runtime_error(
        "dense capacity scaling generates real-valued J; bit/int8 need {-1,0,+1}");
  if (is_csr_variant(options_.variant))
    throw std::runtime_error("dense capacity scaling does not accept CSR variants");
  if (!(hbm_fraction > 0.0 && hbm_fraction <= 1.0))
    throw std::runtime_error("hbm fraction must be in (0,1]");

  plan_ = make_plan(n_, options_.precision, options_.variant,
                    options_.cluster, -1, options_.batch);
  stats_.requested_memory = to_string(memory);
  stats_.requested_variant = to_string(options_.variant);
  stats_.selected_variant = to_string(plan_.variant);
  stats_.cluster_size = plan_.cluster_size;
  if (memory == MatrixMemory::Hybrid && plan_.variant != Variant::Gemm)
    throw std::runtime_error(
        "HYBRID_REQUIRES_GEMM: hybrid HBM+Grace placement is implemented for "
        "the gemm variant only");
  stats_.matrix_bytes = std::size_t(n_) * ldj_ * coupling_element_size(options_.precision);

  try {
    DSB_CUDA_CHECK(cudaStreamCreate(&stream_));
    DSB_CUDA_CHECK(cudaEventCreate(&begin_));
    DSB_CUDA_CHECK(cudaEventCreate(&end_));
    allocate_matrix(memory, hbm_fraction);
    stats_.selected_memory = to_string(selected_memory_);
    allocate_state();
    generate_matrix();
    initialize_state();
  } catch (...) {
    free_all();
    throw;
  }
}

DenseCapacitySolver::~DenseCapacitySolver() { free_all(); }

void DenseCapacitySolver::allocate_grace(void** pointer, std::size_t bytes) {
  // Only meaningful on a coherent CPU-GPU part (GH200): elsewhere pinned host
  // memory is still readable by the GPU, but over PCIe, and the numbers would
  // not mean what the paper says they mean.
  int pageable = 0, host_ptr_ok = 0;
  int device = 0;
  DSB_CUDA_CHECK(cudaGetDevice(&device));
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(
      &pageable, cudaDevAttrPageableMemoryAccess, device));
  if (!pageable)
    throw std::runtime_error(
        "GRACE_UNAVAILABLE: selected GPU cannot directly access pageable system memory");
  DSB_CUDA_CHECK(cudaDeviceGetAttribute(
      &host_ptr_ok, cudaDevAttrCanUseHostPointerForRegisteredMem, device));
  if (!host_ptr_ok)
    throw std::runtime_error(
        "GRACE_UNAVAILABLE: registered host memory is not directly addressable");

  // Pinned system memory that WE allocate, then register -- not
  // cudaMallocManaged and not cudaMallocHost.
  //
  //  * Managed pages are migrated by the driver (read duplication under
  //    SetReadMostly, Hopper access-counter migration without it).  When J is
  //    larger than the free HBM a streaming pass is the worst case for that
  //    cache and every step degrades to a fault-driven copy of the matrix
  //    (~90 GB/s measured) instead of a direct C2C read (~330 GB/s).
  //  * cudaMallocHost pages are allocated inside the driver: they do not
  //    reliably follow numactl --membind, and past ~200 GB the driver falls
  //    back to small pages or to the other Grace socket.  Measured: a 288 GB
  //    tail ran >40 min/point where a 203 GB tail took 3 min.
  //
  // An anonymous mmap is an ordinary user allocation: it follows the caller's
  // NUMA policy, and cudaHostRegister pins the pages where they are and maps
  // them for the GPU.  Nothing ever migrates them.
  //
  // No MADV_HUGEPAGE: the base page here is already 64 KiB (getconf PAGESIZE)
  // and the THP hugepage is 512 MiB (Hugepagesize).  Asking for 512 MiB pages
  // forces the kernel to compact node 1 to assemble them, and while pinning a
  // few hundred GB that compaction transiently DUPLICATES already-pinned pages
  // -- a 320 GB region reached ~485 GB RSS / 640 GB VM and tripped the global
  // OOM killer.  64 KiB pages give plenty of TLB reach on their own.
  //
  // Registered cudaHostRegisterDefault, not ReadOnly: although J is only read
  // during the dSB steps, it is GENERATED by a GPU kernel writing into this
  // very block, so the device mapping must allow writes.
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED)
    throw std::runtime_error("GRACE_MMAP_FAILED: cannot reserve " +
                             std::to_string(bytes) + " bytes of host memory");
  const cudaError_t status =
      cudaHostRegister(p, bytes, cudaHostRegisterDefault);
  if (status != cudaSuccess) {
    munmap(p, bytes);
    throw std::runtime_error(std::string("GRACE_PIN_FAILED: cudaHostRegister: ") +
                             cudaGetErrorString(status));
  }
  *pointer = p;
  grace_alloc_bytes_ = bytes;
}

void DenseCapacitySolver::allocate_matrix(MatrixMemory requested,
                                          double hbm_fraction) {
  std::size_t free_bytes = 0, total_bytes = 0;
  DSB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  stats_.hbm_free_before = free_bytes;
  stats_.hbm_total = total_bytes;

  const std::size_t element = coupling_element_size(options_.precision);
  const std::size_t state_bytes =
      2 * std::size_t(n_) * options_.batch * element +
      std::size_t(options_.n_steps) * sizeof(float);
  const std::size_t budget = std::size_t(double(free_bytes) * hbm_fraction);
  const bool fits_hbm = stats_.matrix_bytes + state_bytes <= budget;

  if (requested == MatrixMemory::Hybrid) {
    // Reserve the GemmStepper buffers too (sign + FP32 accumulator + 4 MiB
    // workspace + the Grace staging block): unlike the all-HBM case the J
    // share is sized to fill the budget, so there is no slack left to absorb
    // them.
    const std::size_t count = std::size_t(n_) * options_.batch;
    const std::size_t reserved =
        state_bytes + count * element + count * sizeof(float) + (4u << 20) +
        kGraceStageBytes;
    const std::size_t row_bytes = ldj_ * element;
    std::size_t rows = budget > reserved ? (budget - reserved) / row_bytes : 0;
    rows -= rows % 64;  // keep the tail block and acc offset nicely aligned
    if (rows >= std::size_t(n_)) {
      selected_memory_ = MatrixMemory::Hbm;       // everything fits
    } else if (rows == 0) {
      selected_memory_ = MatrixMemory::Grace;     // nothing fits
    } else {
      selected_memory_ = MatrixMemory::Hybrid;
      split_rows_ = int(rows);
      const std::size_t head_bytes = rows * row_bytes;
      const std::size_t tail_bytes = stats_.matrix_bytes - head_bytes;
      DSB_CUDA_CHECK(cudaMalloc(&j_, head_bytes));
      allocate_grace(&j_tail_, tail_bytes);
      stats_.hbm_bytes += head_bytes;
      stats_.grace_bytes = tail_bytes;
      return;
    }
  } else {
    selected_memory_ = requested == MatrixMemory::Auto
                           ? (fits_hbm ? MatrixMemory::Hbm : MatrixMemory::Grace)
                           : requested;
  }

  if (selected_memory_ == MatrixMemory::Hbm) {
    if (!fits_hbm && requested == MatrixMemory::Hbm)
      throw std::runtime_error(
          "HBM_OOM_PREDICTED: matrix and state exceed the configured HBM budget");
    DSB_CUDA_CHECK(cudaMalloc(&j_, stats_.matrix_bytes));
    stats_.hbm_bytes += stats_.matrix_bytes;
  } else {
    allocate_grace(&j_, stats_.matrix_bytes);
    stats_.grace_bytes = stats_.matrix_bytes;
  }
}

void DenseCapacitySolver::allocate_state() {
  const std::size_t count = std::size_t(n_) * options_.batch;
  const std::size_t state_bytes = count * coupling_element_size(options_.precision);
  DSB_CUDA_CHECK(cudaMalloc(&x_, state_bytes));
  DSB_CUDA_CHECK(cudaMalloc(&y_, state_bytes));
  DSB_CUDA_CHECK(cudaMalloc(&pump_,
                            std::size_t(options_.n_steps) * sizeof(float)));
  if (plan_.variant == Variant::GlobalSync)
    DSB_CUDA_CHECK(cudaMalloc(&scratch_, state_bytes));
  stats_.hbm_bytes += 2 * state_bytes +
                      std::size_t(options_.n_steps) * sizeof(float);
  if (scratch_) stats_.hbm_bytes += state_bytes;

  std::vector<float> pump(std::size_t(options_.n_steps));
  for (int step = 0; step < options_.n_steps; ++step)
    pump[std::size_t(step)] = options_.n_steps == 1
                                  ? 0.f
                                  : float(step) / float(options_.n_steps - 1);
  DSB_CUDA_CHECK(cudaMemcpyAsync(pump_, pump.data(),
                                 pump.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
  if (plan_.variant == Variant::Gemm) {
    gemm_.reset(new GemmStepper(n_, ldj_, options_.batch,
                                options_.precision, stream_, options_.tf32));
    if (selected_memory_ == MatrixMemory::Hybrid)
      gemm_->set_row_split(j_tail_, split_rows_);
    // Grace-resident rows are staged through HBM rather than read in place
    // by cuBLAS; HBM-only runs are unchanged.
    if (selected_memory_ == MatrixMemory::Grace)
      gemm_->set_grace_staging(kGraceStageBytes, true);
    else if (selected_memory_ == MatrixMemory::Hybrid)
      gemm_->set_grace_staging(kGraceStageBytes, false);
  }
}

void DenseCapacitySolver::generate_block(void* matrix, int row0, int rows) {
  const unsigned long long count =
      (unsigned long long)rows * (unsigned long long)ldj_;
  if (options_.precision == Precision::FP16) {
    generate_dense_symmetric_kernel<__half><<<launch_blocks(), 256, 0, stream_>>>(
        static_cast<__half*>(matrix), n_, ldj_, options_.seed, count,
        (unsigned long long)row0);
  } else {
    generate_dense_symmetric_kernel<float><<<launch_blocks(), 256, 0, stream_>>>(
        static_cast<float*>(matrix), n_, ldj_, options_.seed, count,
        (unsigned long long)row0);
  }
  DSB_CUDA_CHECK(cudaGetLastError());
}

void DenseCapacitySolver::generate_matrix() {
  const auto start = std::chrono::steady_clock::now();
  if (selected_memory_ == MatrixMemory::Hybrid) {
    generate_block(j_, 0, split_rows_);
    generate_block(j_tail_, split_rows_, n_ - split_rows_);
  } else {
    generate_block(j_, 0, n_);
  }
  DSB_CUDA_CHECK(cudaStreamSynchronize(stream_));
  // No cudaMemAdvise here.  The Grace-resident blocks are pinned host memory
  // (see allocate_grace); there is nothing to advise and, in particular, no
  // SetReadMostly: that would re-enable read duplication into HBM.
  const auto end = std::chrono::steady_clock::now();
  stats_.generation_s = std::chrono::duration<double>(end - start).count();
}

void DenseCapacitySolver::initialize_state() {
  const unsigned long long count =
      (unsigned long long)n_ * (unsigned long long)options_.batch;
  if (options_.precision == Precision::FP16) {
    initialize_state_kernel<__half><<<launch_blocks(), 256, 0, stream_>>>(
        static_cast<__half*>(x_), static_cast<__half*>(y_), count,
        options_.seed ^ 0x7265706c696361ULL);
  } else {
    initialize_state_kernel<float><<<launch_blocks(), 256, 0, stream_>>>(
        static_cast<float*>(x_), static_cast<float*>(y_), count,
        options_.seed ^ 0x7265706c696361ULL);
  }
  DSB_CUDA_CHECK(cudaGetLastError());
  DSB_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void DenseCapacitySolver::run(int steps) {
  if (steps <= 0 || steps > options_.n_steps)
    throw std::runtime_error("dense capacity: invalid run step count");
  constexpr float xi = 0.8660254037844386f;  // expected normalization for U(-1,1)/sqrt(N)
  if (plan_.variant == Variant::Gemm)
    gemm_->prepare(x_, y_, j_, pump_, options_.delta, xi, options_.dt, steps);
  DSB_CUDA_CHECK(cudaEventRecord(begin_, stream_));
  if (plan_.variant == Variant::Gemm) {
    gemm_->run(x_, y_, j_, pump_, options_.delta, xi, options_.dt, steps);
  } else if (options_.precision == Precision::FP16) {
    launch_steps(static_cast<__half*>(x_), static_cast<__half*>(y_),
                 static_cast<const __half*>(j_), ldj_, pump_, options_.delta,
                 xi, options_.dt, n_, options_.batch, steps, plan_,
                 static_cast<__half*>(scratch_), stream_);
  } else {
    launch_steps(static_cast<float*>(x_), static_cast<float*>(y_),
                 static_cast<const float*>(j_), ldj_, pump_, options_.delta,
                 xi, options_.dt, n_, options_.batch, steps, plan_,
                 static_cast<float*>(scratch_), stream_);
  }
  DSB_CUDA_CHECK(cudaEventRecord(end_, stream_));
  DSB_CUDA_CHECK(cudaEventSynchronize(end_));
  DSB_CUDA_CHECK(cudaEventElapsedTime(&stats_.run_ms, begin_, end_));
}

void DenseCapacitySolver::free_all() noexcept {
  gemm_.reset();
  auto release_grace = [this](void* p) {
    cudaHostUnregister(p);
    munmap(p, grace_alloc_bytes_);
  };
  if (j_) {
    if (selected_memory_ == MatrixMemory::Grace) release_grace(j_);
    else cudaFree(j_);
  }
  if (j_tail_) release_grace(j_tail_);
  if (x_) cudaFree(x_);
  if (y_) cudaFree(y_);
  if (scratch_) cudaFree(scratch_);
  if (pump_) cudaFree(pump_);
  if (begin_) cudaEventDestroy(begin_);
  if (end_) cudaEventDestroy(end_);
  if (stream_) cudaStreamDestroy(stream_);
  j_ = j_tail_ = x_ = y_ = scratch_ = pump_ = nullptr;
  begin_ = end_ = nullptr;
  stream_ = nullptr;
}

}  // namespace dsb
