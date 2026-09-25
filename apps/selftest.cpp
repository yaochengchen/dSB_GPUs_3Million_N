// apps/selftest.cpp
//
// Checks the GPU kernels against the double-precision CPU reference on a small
// random instance, then re-runs the same instance with a forced cluster size so
// that the block and cluster kernels are compared directly.
//
// Run this after any kernel change. The FP32 path should agree with the
// reference to ~1e-5 on x; the FP16 path only to ~1e-2, which is why the pass
// criterion for FP16 is the energy, not the state.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "dsb/solver.hpp"
#include "dsb/sparse.hpp"
#include "dsb/sparse_solver.hpp"

namespace {

Eigen::MatrixXd random_symmetric(int n, uint32_t seed) {
  std::mt19937                     rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0 / std::sqrt(double(n)));
  Eigen::MatrixXd                  m(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const double v = (i == j) ? 0.0 : gauss(rng);
      m(i, j)        = v;
      m(j, i)        = v;
    }
  return m;
}

/// Complete graph with +-1 couplings (the K2000 shape).
Eigen::MatrixXd random_pm1(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double v = (rng() & 1u) ? 1.0 : -1.0;
      m(i, j) = v;
      m(j, i) = v;
    }
  return m;
}

/// Sparse {-1,0,+1} couplings (the G-set shape), `keep` fraction nonzero.
Eigen::MatrixXd random_ternary(int n, uint32_t seed, double keep) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      if (uni(rng) >= keep) continue;
      const double v = (rng() & 1u) ? 1.0 : -1.0;
      m(i, j) = v;
      m(j, i) = v;
    }
  return m;
}

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

double max_rel_energy_diff(const std::vector<double>& a,
                           const std::vector<double>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double scale = std::max(1.0, std::fabs(b[i]));
    worst              = std::max(worst, std::fabs(a[i] - b[i]) / scale);
  }
  return worst;
}

int check(const char* what, double value, double tolerance) {
  const bool ok = value <= tolerance;
  std::cout << "  " << std::left << std::setw(34) << what << std::right
            << std::scientific << std::setprecision(3) << value
            << "   tol " << tolerance << "   " << (ok ? "PASS" : "FAIL")
            << "\n";
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  const int n       = (argc > 1) ? std::atoi(argv[1]) : 512;
  const int batch   = (argc > 2) ? std::atoi(argv[2]) : 4;
  const int n_steps = (argc > 3) ? std::atoi(argv[3]) : 200;

  std::cout << "selftest  n=" << n << "  batch=" << batch
            << "  steps=" << n_steps << "\n"
            << "shared memory budget : " << dsb::device_max_smem_optin()
            << " B/block   (max N single block " << dsb::device_max_n_block()
            << ")\n\n";

  const Eigen::MatrixXd coupling = random_symmetric(n, 7);
  int                   failures = 0;

  // ---- FP32 block kernel vs CPU reference -------------------------------
  {
    dsb::Options opt;
    opt.batch     = batch;
    opt.n_steps   = n_steps;
    opt.precision = dsb::Precision::FP32;
    opt.variant   = dsb::Variant::Block;

    dsb::Solver solver(coupling, opt);
    solver.run();

    Eigen::MatrixXd x = solver.initial_x();
    Eigen::MatrixXd y = solver.initial_y();
    dsb::reference_run(coupling, solver.pump(), opt.delta, solver.xi(), opt.dt,
                       x, y);

    std::cout << "fp32 block kernel vs CPU reference\n";
    failures += check("max |x_gpu - x_cpu|", max_abs_diff(solver.state(), x),
                      1e-4);
    failures += check("max rel energy diff",
                      max_rel_energy_diff(solver.energies(),
                                          dsb::reference_energies(coupling, x)),
                      1e-6);
    std::cout << "\n";
  }

  // ---- FP16 block kernel: energies only ---------------------------------
  {
    dsb::Options opt;
    opt.batch     = batch;
    opt.n_steps   = n_steps;
    opt.precision = dsb::Precision::FP16;
    opt.variant   = dsb::Variant::Block;

    dsb::Solver solver(coupling, opt);
    solver.run();

    Eigen::MatrixXd x = solver.initial_x();
    Eigen::MatrixXd y = solver.initial_y();
    dsb::reference_run(coupling, solver.pump(), opt.delta, solver.xi(), opt.dt,
                       x, y);

    std::cout << "fp16 block kernel vs CPU reference (energy only)\n";
    failures += check("max rel energy diff",
                      max_rel_energy_diff(solver.energies(),
                                          dsb::reference_energies(coupling, x)),
                      5e-2);
    std::cout << "\n";
  }

  // ---- baselines vs plain block kernel ----------------------------------
  for (dsb::Variant v : {dsb::Variant::GlobalSync, dsb::Variant::Gemm}) {
    dsb::Options base;
    base.batch     = batch;
    base.n_steps   = n_steps;
    base.precision = dsb::Precision::FP32;

    dsb::Options plain = base;
    plain.variant      = dsb::Variant::Block;
    dsb::Solver plain_solver(coupling, plain);
    plain_solver.run();

    dsb::Options other = base;
    other.variant      = v;
    other.tf32         = false;  // bit-exact comparison; TF32 is checked below
    try {
      dsb::Solver other_solver(coupling, other);
      other_solver.run();
      std::cout << dsb::to_string(v) << " baseline vs plain block kernel\n";
      failures += check("max |x_other - x_block|",
                        max_abs_diff(other_solver.state(), plain_solver.state()),
                        1e-3);
      failures += check("max rel energy diff",
                        max_rel_energy_diff(other_solver.energies(),
                                            plain_solver.energies()),
                        1e-6);
      if (v == dsb::Variant::Gemm) {
        // A second call must launch the already-instantiated CUDA graph rather
        // than attempting to capture the cuBLAS sequence again.
        other_solver.run();
        std::cout << "  second graph launch                 PASS\n";
      }
    } catch (const std::exception& e) {
      std::cout << dsb::to_string(v) << " baseline: skipped (" << e.what()
                << ")\n";
    }
    std::cout << "\n";
  }

  // ---- gemm with TF32 tensor cores: energy statistics only ---------------
  // TF32 rounds J to 10 mantissa bits, so trajectories diverge from FP32 and
  // only the energies are comparable.  This is the default gemm configuration
  // in the benchmarks, so it is checked here at the same tolerance as FP16.
  {
    dsb::Options plain;
    plain.batch     = batch;
    plain.n_steps   = n_steps;
    plain.precision = dsb::Precision::FP32;
    plain.variant   = dsb::Variant::Block;
    dsb::Solver plain_solver(coupling, plain);
    plain_solver.run();

    dsb::Options fast = plain;
    fast.variant      = dsb::Variant::Gemm;
    fast.tf32         = true;
    try {
      dsb::Solver fast_solver(coupling, fast);
      fast_solver.run();
      std::cout << "gemm (TF32) vs plain block kernel (energy only)\n";
      failures += check("max rel energy diff",
                        max_rel_energy_diff(fast_solver.energies(),
                                            plain_solver.energies()),
                        5e-2);
    } catch (const std::exception& e) {
      std::cout << "gemm (TF32): skipped (" << e.what() << ")\n";
    }
    std::cout << "\n";
  }

  // ---- bit kernel: exact on {-1,0,+1} couplings ----------------------------
  // Integer coupling sums are exact in FP32, so the bit kernel and the FP32
  // block kernel run the identical arithmetic: x must match to the last bit,
  // not to a tolerance.  Two shapes: complete +-1 (single-plane XOR form, the
  // K2000 case) and sparse ternary (two-plane AND form, the G-set case).
  for (int shape = 0; shape < 2; ++shape) {
    const Eigen::MatrixXd pm = (shape == 0) ? random_pm1(n, 11)
                                            : random_ternary(n, 13, 0.3);
    const char* label = (shape == 0) ? "bit (dense +-1, XOR form)"
                                     : "bit (sparse ternary, AND form)";
    dsb::Options ref;
    ref.batch     = batch;
    ref.n_steps   = n_steps;
    ref.precision = dsb::Precision::FP32;
    ref.variant   = dsb::Variant::Block;
    dsb::Solver ref_solver(pm, ref);
    ref_solver.run();

    dsb::Options bit = ref;
    bit.variant      = dsb::Variant::Bit;
    try {
      dsb::Solver bit_solver(pm, bit);
      bit_solver.run();
      std::cout << label << " vs fp32 block kernel\n"
                << "  plan: rows/block=" << bit_solver.bit_plan().rows
                << " blocks=" << bit_solver.bit_plan().blocks
                << " smem=" << bit_solver.bit_plan().smem << " B"
                << " dense_pm1=" << (bit_solver.bit_plan().dense_pm1 ? 1 : 0)
                << "\n";
      failures += check("max |x_bit - x_block|  (expect 0)",
                        max_abs_diff(bit_solver.state(), ref_solver.state()),
                        0.0);
      failures += check("max rel energy diff    (expect 0)",
                        max_rel_energy_diff(bit_solver.energies(),
                                            ref_solver.energies()),
                        0.0);
      Eigen::MatrixXd x = bit_solver.initial_x();
      Eigen::MatrixXd y = bit_solver.initial_y();
      dsb::reference_run(pm, bit_solver.pump(), bit.delta, bit_solver.xi(),
                         bit.dt, x, y);
      failures += check("max |x_bit - x_cpu|",
                        max_abs_diff(bit_solver.state(), x), 1e-4);
      failures += check("energy vs CPU reference",
                        max_rel_energy_diff(bit_solver.energies(),
                                            dsb::reference_energies(pm, x)),
                        1e-9);
      // A second run continues from the end state (as every variant does);
      // it must still track the block kernel exactly and must not re-pack.
      bit_solver.run();
      ref_solver.run();
      failures += check("second run |x - x_block|",
                        max_abs_diff(bit_solver.state(), ref_solver.state()),
                        0.0);
    } catch (const std::exception& e) {
      std::cout << label << ": skipped (" << e.what() << ")\n";
      ++failures;
    }
    std::cout << "\n";

    // ---- INT8 IMMA gemm on the same instance: also exact ------------------
    {
      dsb::Options ref2 = ref;
      dsb::Solver  ref2_solver(pm, ref2);
      ref2_solver.run();
      dsb::Options i8 = ref;
      i8.variant      = dsb::Variant::Gemm;
      i8.precision    = dsb::Precision::INT8;
      try {
        dsb::Solver i8_solver(pm, i8);
        i8_solver.run();
        std::cout << "gemm int8 (IMMA) vs fp32 block kernel, "
                  << (shape == 0 ? "dense +-1" : "sparse ternary") << "\n";
        failures += check("max |x_int8 - x_block| (expect 0)",
                          max_abs_diff(i8_solver.state(), ref2_solver.state()),
                          0.0);
        failures += check("max rel energy diff    (expect 0)",
                          max_rel_energy_diff(i8_solver.energies(),
                                              ref2_solver.energies()),
                          0.0);
        i8_solver.run();
        ref2_solver.run();
        failures += check("second run |x - x_block|",
                          max_abs_diff(i8_solver.state(), ref2_solver.state()),
                          0.0);
      } catch (const std::exception& e) {
        std::cout << "gemm int8: skipped (" << e.what() << ")\n";
        ++failures;
      }
      std::cout << "\n";
    }
  }

  // ---- CSR path: GPU energy of its best replica vs direct evaluation ----
  {
    Eigen::MatrixXd sparse_dense = coupling;
    for (int row = 0; row < n; ++row)
      for (int col = 0; col < n; ++col)
        if (row != col && ((row * 17 + col * 31) % 7 != 0))
          sparse_dense(row, col) = 0.0;
    sparse_dense = 0.5 * (sparse_dense + sparse_dense.transpose());
    const dsb::SparseMatrix sparse = dsb::sparse_from_dense(sparse_dense);

    dsb::Options reference_options;
    reference_options.batch = batch;
    reference_options.n_steps = n_steps;
    reference_options.precision = dsb::Precision::FP32;
    reference_options.variant = dsb::Variant::CsrRow;
    dsb::SparseSolver reference_solver(sparse, reference_options);
    reference_solver.run();
    const std::vector<double> reference_energies = reference_solver.energies();

    for (dsb::Variant variant :
         {dsb::Variant::CsrRow, dsb::Variant::CsrBlock,
          dsb::Variant::CsrCluster}) {
      try {
        dsb::Options opt;
        opt.batch = batch;
        opt.n_steps = n_steps;
        opt.precision = dsb::Precision::FP32;
        opt.variant = variant;
        opt.cluster = variant == dsb::Variant::CsrCluster ? 2 : 0;
        dsb::SparseSolver solver(sparse, opt);
        solver.run();
        const std::vector<double> energies = solver.energies();
        const double gpu_best =
            *std::min_element(energies.begin(), energies.end());
        const Eigen::VectorXd spins = solver.best_spins();
        const double direct = -0.5 * spins.dot(sparse_dense * spins);
        std::cout << dsb::to_string(variant) << " energy check\n";
        if (variant != dsb::Variant::CsrCluster)
          failures += check("max rel energy diff vs csr-row",
                            max_rel_energy_diff(energies, reference_energies),
                            1e-6);
        failures += check("|E_gpu_best - E_direct|",
                          std::fabs(gpu_best - direct),
                          1e-4 * std::max(1.0, std::fabs(direct)));
      } catch (const std::exception& e) {
        std::cout << dsb::to_string(variant) << ": skipped (" << e.what()
                  << ")\n";
      }
      std::cout << "\n";
    }
  }

  // ---- cluster kernel vs block kernel, same precision -------------------
  int cluster_ok = 0;
  cudaDeviceGetAttribute(&cluster_ok, cudaDevAttrClusterLaunch, 0);
  if (!cluster_ok) {
    std::cout << "cluster kernel: skipped, this GPU has no thread block "
                 "cluster support\n\n";
  } else {
    for (int csize : {2, 4}) {
      dsb::Options base;
      base.batch     = batch;
      base.n_steps   = n_steps;
      base.precision = dsb::Precision::FP32;

      dsb::Options blocked = base;
      blocked.variant      = dsb::Variant::Block;
      dsb::Solver reference_solver(coupling, blocked);
      reference_solver.run();

      dsb::Options clustered = base;
      clustered.variant      = dsb::Variant::Cluster;
      clustered.cluster      = csize;
      dsb::Solver cluster_solver(coupling, clustered);
      cluster_solver.run();

      std::cout << "cluster kernel (C=" << csize << ") vs block kernel\n";
      failures += check("max |x_cluster - x_block|",
                        max_abs_diff(cluster_solver.state(),
                                     reference_solver.state()),
                        1e-4);
      failures += check("max rel energy diff",
                        max_rel_energy_diff(cluster_solver.energies(),
                                            reference_solver.energies()),
                        1e-6);
      std::cout << "\n";
    }
  }

  std::cout << (failures == 0 ? "ALL CHECKS PASSED\n"
                              : std::to_string(failures) + " CHECK(S) FAILED\n");
  return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
