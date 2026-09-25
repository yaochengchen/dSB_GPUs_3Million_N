// src/solver.cu
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "dsb/solver.hpp"

namespace dsb {
namespace {

template <class T>
void pack_rows(const Eigen::MatrixXd& src, std::vector<T>& dst, std::size_t ldj);

template <>
void pack_rows<__half>(const Eigen::MatrixXd& src, std::vector<__half>& dst,
                       std::size_t ldj) {
  const int n = int(src.rows());
  dst.assign(std::size_t(n) * ldj, __float2half_rn(0.f));
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      dst[std::size_t(r) * ldj + c] = __float2half_rn(float(src(r, c)));
}

template <>
void pack_rows<std::int8_t>(const Eigen::MatrixXd& src,
                            std::vector<std::int8_t>& dst, std::size_t ldj) {
  const int n = int(src.rows());
  dst.assign(std::size_t(n) * ldj, std::int8_t(0));
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      const double v = src(r, c);
      dst[std::size_t(r) * ldj + c] = std::int8_t(v > 0.0 ? 1 : (v < 0.0 ? -1 : 0));
    }
}

template <>
void pack_rows<float>(const Eigen::MatrixXd& src, std::vector<float>& dst,
                      std::size_t ldj) {
  const int n = int(src.rows());
  dst.assign(std::size_t(n) * ldj, 0.f);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      dst[std::size_t(r) * ldj + c] = float(src(r, c));
}

template <class T>
void pack_state(const Eigen::MatrixXd& src, std::vector<T>& dst);

template <>
void pack_state<__half>(const Eigen::MatrixXd& src, std::vector<__half>& dst) {
  const int n = int(src.rows()), b = int(src.cols());
  dst.resize(std::size_t(n) * b);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < b; ++c)
      dst[std::size_t(r) * b + c] = __float2half_rn(float(src(r, c)));
}

template <>
void pack_state<float>(const Eigen::MatrixXd& src, std::vector<float>& dst) {
  const int n = int(src.rows()), b = int(src.cols());
  dst.resize(std::size_t(n) * b);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < b; ++c) dst[std::size_t(r) * b + c] = float(src(r, c));
}

template <class T>
void unpack_state(const std::vector<T>& src, Eigen::MatrixXd& dst);

template <>
void unpack_state<__half>(const std::vector<__half>& src, Eigen::MatrixXd& dst) {
  const int n = int(dst.rows()), b = int(dst.cols());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < b; ++c)
      dst(r, c) = double(__half2float(src[std::size_t(r) * b + c]));
}

template <>
void unpack_state<float>(const std::vector<float>& src, Eigen::MatrixXd& dst) {
  const int n = int(dst.rows()), b = int(dst.cols());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < b; ++c) dst(r, c) = double(src[std::size_t(r) * b + c]);
}

}  // namespace

// --------------------------------------------------------------------------

Solver::Solver(const Eigen::MatrixXd& coupling, const Options& options)
    : options_(options) {
  if (is_csr_variant(options_.variant))
    throw std::runtime_error(
        "Solver: sparse variant requires dsb::SparseSolver and a CSR matrix");
  if (coupling.rows() != coupling.cols())
    throw std::runtime_error("Solver: coupling matrix must be square");
  n_ = int(coupling.rows());
  if (n_ <= 0) throw std::runtime_error("Solver: empty coupling matrix");
  if (options_.batch <= 0 || options_.n_steps <= 0)
    throw std::runtime_error("Solver: batch and n_steps must be positive");

  ldj_  = row_stride(n_);

  // INT8 storage: exact for couplings in {-1,0,+1}, Gemm only.
  if (options_.precision == Precision::INT8) {
    bool pm1 = false;
    if (!bitfused::classify(coupling.data(), n_,
                            std::size_t(coupling.outerStride()), pm1))
      throw std::runtime_error(
          "Solver: --precision=int8 requires every coupling in {-1, 0, +1}");
    if (options_.variant == Variant::Auto) options_.variant = Variant::Gemm;
    if (options_.variant != Variant::Gemm)
      throw std::runtime_error(
          "Solver: --precision=int8 is only implemented for --variant=gemm");
    if (options_.batch % 4 != 0)
      throw std::runtime_error(
          "Solver: --precision=int8 needs batch to be a multiple of 4");
    // Energies come from the bitplane kernel (exact, and no dense fp J to
    // hand to launch_energy).  Residency of this plan is irrelevant here.
    bit_plan_ = bitfused::make_bit_plan(n_, options_.batch, pm1);
  }

  // Bit-fused path.  Chosen explicitly, or by Auto whenever the matrix is in
  // {-1,0,+1} and the (N, batch) plan fits on chip.  The dynamics are bit-for-
  // bit those of the FP32 kernels (integer coupling sums are exact in FP32),
  // so this changes nothing but the time.
  {
    // classify() reads data[r*ld + c]; on Eigen's column-major storage that
    // is the transpose, which is fine for a membership test over all
    // off-diagonal entries.  Packing below uses an explicit row-major copy.
    bool pm1 = false;
    const bool ternary =
        (options_.variant == Variant::Bit || options_.variant == Variant::Auto) &&
        bitfused::classify(coupling.data(), n_,
                           std::size_t(coupling.outerStride()), pm1);
    if (ternary) {
      bit_plan_ = bitfused::make_bit_plan(n_, options_.batch, pm1);
      const bool ok = bitfused::fits(bit_plan_);
      if (options_.variant == Variant::Bit && !ok)
        throw std::runtime_error(
            "Solver: bit variant does not fit: N=" + std::to_string(n_) +
            " batch=" + std::to_string(options_.batch) + " needs " +
            std::to_string(bit_plan_.smem) + " B shared memory per block and " +
            std::to_string(bit_plan_.blocks) + " co-resident blocks");
      if (ok) {
        options_.variant   = Variant::Bit;
        options_.precision = Precision::FP32;  // x, y are FP32 on this path
      }
    } else if (options_.variant == Variant::Bit) {
      throw std::runtime_error(
          "Solver: bit variant requires every coupling to be in {-1, 0, +1}");
    }
  }

  plan_ = make_plan(n_, options_.precision, options_.variant,
                    options_.cluster, -1, options_.batch);

  // xi is derived from the ORIGINAL double matrix, not from the quantised copy,
  // so switching precision does not silently change the dynamics.
  xi_ = options_.xi;
  if (!std::isfinite(xi_) || xi_ <= 0.f) {
    const double sumsq = coupling.array().square().sum();
    xi_ = (sumsq > 0.0)
              ? float(0.5 * std::sqrt(double(n_ - 1)) / std::sqrt(sumsq))
              : 0.f;
  }

  // Linear pump schedule, kept in FP32 on the device: it is only n_steps long,
  // so there is nothing to gain from storing it in FP16.
  pump_.resize(std::size_t(options_.n_steps));
  if (options_.n_steps == 1) {
    pump_[0] = 0.f;
  } else {
    for (int i = 0; i < options_.n_steps; ++i)
      pump_[i] = float(i) / float(options_.n_steps - 1);
  }

  std::mt19937                           rng(options_.seed);
  std::uniform_real_distribution<double> uni(-0.01, 0.01);
  x0_.resize(n_, options_.batch);
  y0_.resize(n_, options_.batch);
  for (int r = 0; r < n_; ++r)
    for (int c = 0; c < options_.batch; ++c) {
      x0_(r, c) = uni(rng);
      y0_(r, c) = uni(rng);
    }
  x_ = x0_;

  allocate_and_upload(coupling);
}

Solver::~Solver() { free_device(); }

std::size_t Solver::j_bytes() const {
  if (plan_.variant == Variant::Bit) return bitfused::coupling_bytes(bit_plan_);
  return std::size_t(n_) * ldj_ * coupling_element_size(options_.precision);
}

void Solver::allocate_and_upload(const Eigen::MatrixXd& coupling) {
  const std::size_t jbytes =
      std::size_t(n_) * ldj_ * coupling_element_size(options_.precision);
  const std::size_t sbytes =
      std::size_t(n_) * options_.batch * state_element_size(options_.precision);

  if (options_.precision == Precision::INT8) {
    // J as int8 for the IMMA GEMM; bitplanes only for the energy kernel.
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowmajor = coupling;
    std::vector<std::uint32_t> planes;
    std::vector<std::int32_t>  rowconst;
    if (bit_plan_.dense_pm1) {
      bitfused::pack_dense_pm1(rowmajor.data(), n_, std::size_t(n_),
                               bit_plan_.words, planes);
    } else {
      bitfused::pack_pm_planes(rowmajor.data(), n_, std::size_t(n_),
                               bit_plan_.words, planes, rowconst);
    }
    std::vector<std::int8_t> hj;
    pack_rows<std::int8_t>(coupling, hj, ldj_);
    std::vector<float> hx, hy;
    pack_state<float>(x0_, hx);
    pack_state<float>(y0_, hy);

    DSB_CUDA_CHECK(cudaMalloc(&d_j_, jbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_x_, sbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_y_, sbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_pump_, pump_.size() * sizeof(float)));
    DSB_CUDA_CHECK(cudaMalloc(&d_jbits_, planes.size() * sizeof(std::uint32_t)));
    if (!rowconst.empty())
      DSB_CUDA_CHECK(cudaMalloc(&d_rowc_, rowconst.size() * sizeof(std::int32_t)));
    DSB_CUDA_CHECK(cudaStreamCreate(&stream_));
    DSB_CUDA_CHECK(cudaEventCreate(&begin_));
    DSB_CUDA_CHECK(cudaEventCreate(&end_));

    DSB_CUDA_CHECK(cudaMemcpy(d_j_, hj.data(), jbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_x_, hx.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_y_, hy.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_pump_, pump_.data(),
                              pump_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_jbits_, planes.data(),
                              planes.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
    if (!rowconst.empty())
      DSB_CUDA_CHECK(cudaMemcpy(d_rowc_, rowconst.data(),
                                rowconst.size() * sizeof(std::int32_t),
                                cudaMemcpyHostToDevice));
    return;
  }

  if (plan_.variant == Variant::Bit) {
    // No dense J on the device at all: only the packed planes, the state, the
    // pump and the double-buffered sign bitmap.
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowmajor = coupling;
    std::vector<std::uint32_t> planes;
    std::vector<std::int32_t>  rowconst;
    if (bit_plan_.dense_pm1) {
      bitfused::pack_dense_pm1(rowmajor.data(), n_, std::size_t(n_),
                               bit_plan_.words, planes);
    } else {
      bitfused::pack_pm_planes(rowmajor.data(), n_, std::size_t(n_),
                               bit_plan_.words, planes, rowconst);
    }
    const std::size_t sign_words = bitfused::sign_words(bit_plan_);

    DSB_CUDA_CHECK(cudaMalloc(&d_jbits_, planes.size() * sizeof(std::uint32_t)));
    DSB_CUDA_CHECK(cudaMalloc(&d_sbits_, sign_words * sizeof(std::uint32_t)));
    DSB_CUDA_CHECK(cudaMemset(d_sbits_, 0, sign_words * sizeof(std::uint32_t)));
    if (!rowconst.empty())
      DSB_CUDA_CHECK(cudaMalloc(&d_rowc_, rowconst.size() * sizeof(std::int32_t)));
    DSB_CUDA_CHECK(cudaMalloc(&d_x_, sbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_y_, sbytes));
    DSB_CUDA_CHECK(cudaMalloc(&d_pump_, pump_.size() * sizeof(float)));
    DSB_CUDA_CHECK(cudaStreamCreate(&stream_));
    DSB_CUDA_CHECK(cudaEventCreate(&begin_));
    DSB_CUDA_CHECK(cudaEventCreate(&end_));

    DSB_CUDA_CHECK(cudaMemcpy(d_jbits_, planes.data(),
                              planes.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
    if (!rowconst.empty())
      DSB_CUDA_CHECK(cudaMemcpy(d_rowc_, rowconst.data(),
                                rowconst.size() * sizeof(std::int32_t),
                                cudaMemcpyHostToDevice));
    std::vector<float> hx, hy;
    pack_state<float>(x0_, hx);
    pack_state<float>(y0_, hy);
    DSB_CUDA_CHECK(cudaMemcpy(d_x_, hx.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_y_, hy.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_pump_, pump_.data(),
                              pump_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    return;
  }

  DSB_CUDA_CHECK(cudaMalloc(&d_j_, jbytes));
  DSB_CUDA_CHECK(cudaMalloc(&d_x_, sbytes));
  DSB_CUDA_CHECK(cudaMalloc(&d_y_, sbytes));
  DSB_CUDA_CHECK(cudaMalloc(&d_pump_, pump_.size() * sizeof(float)));
  if (needs_scratch(plan_.variant))
    DSB_CUDA_CHECK(cudaMalloc(&d_scratch_, sbytes));
  DSB_CUDA_CHECK(cudaStreamCreate(&stream_));
  DSB_CUDA_CHECK(cudaEventCreate(&begin_));
  DSB_CUDA_CHECK(cudaEventCreate(&end_));

  if (options_.precision == Precision::FP16) {
    std::vector<__half> hj, hx, hy;
    pack_rows<__half>(coupling, hj, ldj_);
    pack_state<__half>(x0_, hx);
    pack_state<__half>(y0_, hy);
    DSB_CUDA_CHECK(cudaMemcpy(d_j_, hj.data(), jbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_x_, hx.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_y_, hy.data(), sbytes, cudaMemcpyHostToDevice));
  } else {
    std::vector<float> hj, hx, hy;
    pack_rows<float>(coupling, hj, ldj_);
    pack_state<float>(x0_, hx);
    pack_state<float>(y0_, hy);
    DSB_CUDA_CHECK(cudaMemcpy(d_j_, hj.data(), jbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_x_, hx.data(), sbytes, cudaMemcpyHostToDevice));
    DSB_CUDA_CHECK(cudaMemcpy(d_y_, hy.data(), sbytes, cudaMemcpyHostToDevice));
  }
  DSB_CUDA_CHECK(cudaMemcpy(d_pump_, pump_.data(),
                            pump_.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
}

void Solver::free_device() noexcept {
  gemm_.reset();
  if (d_jbits_) cudaFree(d_jbits_);
  if (d_rowc_) cudaFree(d_rowc_);
  if (d_sbits_) cudaFree(d_sbits_);
  d_jbits_ = d_sbits_ = nullptr;
  d_rowc_  = nullptr;
  if (d_j_) cudaFree(d_j_);
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_scratch_) cudaFree(d_scratch_);
  if (d_pump_) cudaFree(d_pump_);
  if (begin_) cudaEventDestroy(begin_);
  if (end_) cudaEventDestroy(end_);
  if (stream_) cudaStreamDestroy(stream_);
  d_j_ = d_x_ = d_y_ = d_scratch_ = nullptr;
  d_pump_ = nullptr;
  begin_ = end_ = nullptr;
  stream_ = nullptr;
}

std::size_t Solver::device_bytes() const {
  if (options_.precision == Precision::INT8) {
    std::size_t bytes = std::size_t(n_) * ldj_ +
                        2 * std::size_t(n_) * options_.batch * sizeof(float) +
                        pump_.size() * sizeof(float) +
                        bitfused::coupling_bytes(bit_plan_) +
                        (d_rowc_ ? std::size_t(n_) * sizeof(std::int32_t) : 0);
    if (gemm_) bytes += gemm_->device_bytes();
    return bytes;
  }
  if (plan_.variant == Variant::Bit) {
    return bitfused::coupling_bytes(bit_plan_) +
           (d_rowc_ ? std::size_t(n_) * sizeof(std::int32_t) : 0) +
           bitfused::sign_words(bit_plan_) * sizeof(std::uint32_t) +
           2 * std::size_t(n_) * options_.batch * sizeof(float) +
           pump_.size() * sizeof(float);
  }
  const std::size_t jesz = coupling_element_size(options_.precision);
  const std::size_t sesz = state_element_size(options_.precision);
  std::size_t       bytes = std::size_t(n_) * ldj_ * jesz +
                      2 * std::size_t(n_) * options_.batch * sesz +
                      pump_.size() * sizeof(float);
  if (d_scratch_) bytes += std::size_t(n_) * options_.batch * sesz;
  if (gemm_) bytes += gemm_->device_bytes();
  return bytes;
}

void Solver::run() {
  cudaStream_t stream = stream_;

  // Build/instantiate the GEMM graph once, outside the measured interval.
  // Subsequent calls reuse it and submit all steps with one graph launch.
  if (plan_.variant == Variant::Gemm) {
    if (!gemm_)
      gemm_ = std::unique_ptr<GemmStepper>(new GemmStepper(
          n_, ldj_, options_.batch, options_.precision, stream,
          options_.tf32));
    gemm_->prepare(d_x_, d_y_, d_j_, d_pump_, options_.delta, xi_,
                   options_.dt, options_.n_steps);
  }

  // Scoped so the carve-out is released before we copy the result back.
  {
    L2Persistence pin(stream, options_.l2_persist ? d_j_ : nullptr,
                      options_.l2_persist ? j_bytes() : 0);
    l2_pinned_ = pin.pinned_bytes();

    DSB_CUDA_CHECK(cudaEventRecord(begin_, stream));
    if (plan_.variant == Variant::Bit) {
      bitfused::launch_steps(bit_plan_, static_cast<float*>(d_x_),
                             static_cast<float*>(d_y_), d_jbits_, d_rowc_,
                             d_sbits_, d_pump_, options_.delta, xi_,
                             options_.dt, options_.n_steps, stream);
    } else if (plan_.variant == Variant::Gemm) {
      gemm_->run(d_x_, d_y_, d_j_, d_pump_, options_.delta, xi_, options_.dt,
                 options_.n_steps);
    } else if (options_.precision == Precision::FP16) {
      launch_steps(static_cast<__half*>(d_x_), static_cast<__half*>(d_y_),
                   static_cast<const __half*>(d_j_), ldj_, d_pump_,
                   options_.delta, xi_, options_.dt, n_, options_.batch,
                   options_.n_steps, plan_,
                   static_cast<__half*>(d_scratch_), stream);
    } else {
      launch_steps(static_cast<float*>(d_x_), static_cast<float*>(d_y_),
                   static_cast<const float*>(d_j_), ldj_, d_pump_,
                   options_.delta, xi_, options_.dt, n_, options_.batch,
                   options_.n_steps, plan_,
                   static_cast<float*>(d_scratch_), stream);
    }
    DSB_CUDA_CHECK(cudaEventRecord(end_, stream));
    DSB_CUDA_CHECK(cudaStreamSynchronize(stream));
    DSB_CUDA_CHECK(cudaEventElapsedTime(&last_ms_, begin_, end_));
  }

  const std::size_t count = std::size_t(n_) * options_.batch;
  if (options_.precision == Precision::FP16) {
    std::vector<__half> hx(count);
    DSB_CUDA_CHECK(cudaMemcpy(hx.data(), d_x_, count * sizeof(__half),
                              cudaMemcpyDeviceToHost));
    unpack_state<__half>(hx, x_);
  } else {
    std::vector<float> hx(count);
    DSB_CUDA_CHECK(cudaMemcpy(hx.data(), d_x_, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    unpack_state<float>(hx, x_);
  }
  ran_ = true;
}

std::vector<double> Solver::energies() const {
  double* d_e = nullptr;
  DSB_CUDA_CHECK(cudaMalloc(&d_e, std::size_t(options_.batch) * sizeof(double)));

  cudaStream_t stream = stream_;
  try {
    if (plan_.variant == Variant::Bit || options_.precision == Precision::INT8) {
      bitfused::launch_energy(bit_plan_, static_cast<const float*>(d_x_),
                              d_jbits_, d_rowc_, d_e, stream);
    } else if (options_.precision == Precision::FP16) {
      launch_energy(static_cast<const __half*>(d_x_),
                    static_cast<const __half*>(d_j_), ldj_, n_, options_.batch,
                    d_e, stream);
    } else {
      launch_energy(static_cast<const float*>(d_x_),
                    static_cast<const float*>(d_j_), ldj_, n_, options_.batch,
                    d_e, stream);
    }
    DSB_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    cudaFree(d_e);
    throw;
  }

  std::vector<double> out(static_cast<std::size_t>(options_.batch));
  DSB_CUDA_CHECK(cudaMemcpy(out.data(), d_e, out.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
  cudaFree(d_e);
  return out;
}

int Solver::best_replica() const {
  if (!ran_) throw std::runtime_error("Solver: call run() before best_replica()");
  const std::vector<double> e = energies();
  int                       best = 0;
  for (std::size_t i = 1; i < e.size(); ++i)
    if (e[i] < e[best]) best = int(i);
  return best;
}

Eigen::VectorXd Solver::best_spins() const {
  const int       b = best_replica();
  Eigen::VectorXd s(n_);
  for (int i = 0; i < n_; ++i) {
    const double v = x_(i, b);
    s(i)           = (v > 0.0) ? 1.0 : -1.0;
  }
  return s;
}

// --------------------------------------------------------------------------
// CPU reference
// --------------------------------------------------------------------------

void reference_run(const Eigen::MatrixXd& coupling, const std::vector<float>& pump,
                   float delta, float xi, float dt, Eigen::MatrixXd& x,
                   Eigen::MatrixXd& y) {
  const int n = int(coupling.rows());
  const int b = int(x.cols());
  Eigen::MatrixXd s(n, b);

  for (float p : pump) {
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < b; ++c)
        s(r, c) = (x(r, c) > 0.0) ? 1.0 : -1.0;

    const Eigen::MatrixXd acc = coupling * s;

    for (int r = 0; r < n; ++r)
      for (int c = 0; c < b; ++c) {
        double yv = y(r, c);
        double xv = x(r, c);
        yv += (-(double(delta) - double(p)) * xv + double(xi) * acc(r, c)) * double(dt);
        xv += double(dt) * yv * double(delta);
        if (std::fabs(xv) > 1.0) {
          xv = (xv > 0.0) ? 1.0 : -1.0;
          yv = 0.0;
        }
        x(r, c) = xv;
        y(r, c) = yv;
      }
  }
}

std::vector<double> reference_energies(const Eigen::MatrixXd& coupling,
                                       const Eigen::MatrixXd& x) {
  const int       n = int(coupling.rows());
  const int       b = int(x.cols());
  Eigen::MatrixXd s(n, b);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < b; ++c)
      s(r, c) = (x(r, c) > 0.0) ? 1.0 : -1.0;

  const Eigen::MatrixXd acc = coupling * s;
  std::vector<double>   out(static_cast<std::size_t>(b));
  for (int c = 0; c < b; ++c) {
    double sum = 0.0;
    for (int r = 0; r < n; ++r) sum += acc(r, c) * s(r, c);
    out[std::size_t(c)] = -0.5 * sum;
  }
  return out;
}

}  // namespace dsb
