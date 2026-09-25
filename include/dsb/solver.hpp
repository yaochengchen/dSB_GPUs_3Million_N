// include/dsb/solver.hpp
//
// Discrete Simulated Bifurcation (dSB) solver for dense Ising problems.
//
// Reference for the dynamics:
//   H. Goto et al., "High-performance combinatorial optimization based on
//   classical mechanics", Sci. Adv. 7, eabe7953 (2021).
//
// Given a symmetric coupling matrix J, the solver integrates
//
//     y <- y + (-(delta - p_t) * x + xi * (J @ sign(x))) * dt
//     x <- x + dt * y * delta
//     |x| > 1  =>  x = sign(x), y = 0
//
// for `n_steps` steps with a linear pump schedule p_t = t / (n_steps - 1),
// starting from `batch` independent random initial states ("replicas").
// The reported solution is sign(x) of the lowest-energy replica.

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

#include "dsb/bitfused.cuh"
#include "dsb/common.hpp"
#include "dsb/gemm.cuh"
#include "dsb/kernels.cuh"

namespace dsb {

struct Options {
  int       batch     = 200;               ///< independent replicas
  int       n_steps   = 800;               ///< integration steps
  float     delta     = 1.0f;              ///< detuning
  float     dt        = 1.0f;              ///< step size
  float     xi        = 0.0f;              ///< coupling gain; <= 0 or NaN => auto
  Precision precision = Precision::FP16;   ///< storage for J, x, y
  Variant   variant   = Variant::Auto;     ///< which kernel to use
  int       cluster   = 0;                 ///< blocks per replica; 0 => auto
  bool      l2_persist = false;            ///< pin as much of J in L2 as allowed
  bool      tf32      = true;              ///< Gemm + FP32: use TF32 tensor cores
  uint32_t  seed      = 12345;
};

class Solver {
 public:
  /// `coupling` must be square and is used as-is (no symmetrisation).
  Solver(const Eigen::MatrixXd& coupling, const Options& options = Options());
  ~Solver();

  Solver(const Solver&)            = delete;
  Solver& operator=(const Solver&) = delete;

  /// Run all `n_steps` in a single kernel launch, then copy x back to the host.
  void run();

  /// Ising energy per replica, computed on the GPU from sign(x).
  std::vector<double> energies() const;

  /// Index of the lowest-energy replica. Requires run() to have been called.
  int best_replica() const;

  /// sign(x) of the lowest-energy replica, as +1 / -1 (x == 0 counts as -1).
  Eigen::VectorXd best_spins() const;

  /// Raw state after run(), shape (n, batch).
  const Eigen::MatrixXd& state() const { return x_; }

  /// Initial state the solver was seeded with, shape (n, batch).
  /// Useful for reproducing a run against the CPU reference.
  const Eigen::MatrixXd& initial_x() const { return x0_; }
  const Eigen::MatrixXd& initial_y() const { return y0_; }

  /// Pump schedule actually used, length n_steps.
  const std::vector<float>& pump() const { return pump_; }

  int               n() const { return n_; }
  int               batch() const { return options_.batch; }
  float             xi() const { return xi_; }
  const Options&    options() const { return options_; }
  const LaunchPlan& plan() const { return plan_; }

  /// Bytes of device memory held by this solver.
  std::size_t device_bytes() const;

  /// Bytes occupied by J, including the row padding (packed planes for Bit).
  std::size_t j_bytes() const;

  /// True when this solver runs the bit-fused path (J in {-1,0,+1}).
  bool uses_bit() const { return plan_.variant == Variant::Bit; }
  const bitfused::Plan& bit_plan() const { return bit_plan_; }

  /// GPU time of the last run(), in milliseconds.
  float last_run_ms() const { return last_ms_; }

  /// Bytes of J pinned in L2 for the last run (0 when l2_persist is off).
  std::size_t l2_pinned_bytes() const { return l2_pinned_; }

 private:
  void allocate_and_upload(const Eigen::MatrixXd& coupling);
  void free_device() noexcept;

  Options     options_;
  int         n_   = 0;
  std::size_t ldj_ = 0;
  float       xi_  = 0.f;
  LaunchPlan  plan_;

  Eigen::MatrixXd    x_;
  Eigen::MatrixXd    x0_;
  Eigen::MatrixXd    y0_;
  std::vector<float> pump_;

  void*  d_j_       = nullptr;
  void*  d_x_       = nullptr;
  void*  d_y_       = nullptr;
  void*  d_scratch_ = nullptr;  ///< [N*B], only for the device-memory variants
  float* d_pump_    = nullptr;

  std::unique_ptr<GemmStepper> gemm_;

  // Bit-fused path (Variant::Bit): J lives only as packed planes.
  bitfused::Plan bit_plan_;
  std::uint32_t* d_jbits_ = nullptr;  ///< [planes][N][W]
  std::int32_t*  d_rowc_  = nullptr;  ///< [N], general form only
  std::uint32_t* d_sbits_ = nullptr;  ///< [2][W][B] double-buffered bitmap

  cudaStream_t stream_ = nullptr;
  cudaEvent_t  begin_  = nullptr;
  cudaEvent_t  end_    = nullptr;

  float       last_ms_    = 0.f;
  std::size_t l2_pinned_  = 0;
  bool        ran_        = false;
};

/// Single-threaded double-precision reference implementation of the same
/// dynamics, for correctness checks. Slow: O(n^2) per step per replica.
void reference_run(const Eigen::MatrixXd& coupling, const std::vector<float>& pump,
                   float delta, float xi, float dt, Eigen::MatrixXd& x,
                   Eigen::MatrixXd& y);

/// Ising energy of sign(x) per column, in double precision.
std::vector<double> reference_energies(const Eigen::MatrixXd& coupling,
                                       const Eigen::MatrixXd& x);

}  // namespace dsb
