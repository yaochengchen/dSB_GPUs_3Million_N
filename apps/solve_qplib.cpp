// apps/solve_qplib.cpp
//
// Reduce a QPLIB instance with FastHare, solve the remainder with dSB on the
// GPU, map the answer back, and report the objective value.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dsb/qplib.hpp"
#include "dsb/reduction.hpp"
#include "dsb/solver.hpp"
#include "dsb/sparse.hpp"
#include "dsb/sparse_solver.hpp"

namespace {

struct Config {
  double                   alpha      = 0.2;
  dsb::Options             solver;
  bool                     reduction    = true;
  bool                     apply_gauge = true;
  bool                     csv         = false;
  bool                     plan_only   = false;
  int                      repeat      = 0;
  int                      repeats     = 1;
  int                      warmup      = 0;
  std::optional<double>    target;
  std::vector<std::string> files;
};

void usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options] <instance.qplib> [...]\n\n"
      << "  --alpha=F        FastHare merge threshold (default 0.2)\n"
      << "  --no-reduction   solve the unreduced standard-form Hamiltonian\n"
      << "  --batch=N        replicas solved in parallel (default 200)\n"
      << "  --steps=N        integration steps (default 800)\n"
      << "  --dt=F           integration step size (default 1.0); pass the same\n"
      << "                   value to the public baseline as --sb-time-step\n"
      << "  --precision=P    fp16 (default), fp32, or int8 (gemm, {-1,0,+1} only)\n"
      << "  --variant=V      bit | csr-row | csr-block | csr-cluster | block |\n"
      << "                   cluster | global-sync | gemm | auto (default)\n"
      << "  --no-tf32        gemm+fp32 on CUDA cores instead of TF32\n"
      << "  --cluster=N      blocks per replica; 0 = auto (default)\n"
      << "  --l2             pin as much of J in L2 as the device allows\n"
      << "  --seed=N         RNG seed (default 12345)\n"
      << "  --repeat=N       repeat index written to CSV (default 0)\n"
      << "  --repeats=N      measured runs in this process (default 1)\n"
      << "  --warmup=N       unmeasured warm-up runs (default 0)\n"
      << "  --target=F       target objective for success/TTS accounting\n"
      << "  --no-gauge       skip the bias-node gauge fix (old behaviour)\n"
      << "  --csv            one CSV row per instance instead of a report\n"
      << "  --plan-only      print the size/launch plan and stop\n";
}

bool starts_with(const std::string& s, const char* p) {
  return s.rfind(p, 0) == 0;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (starts_with(a, "--alpha=")) {
      cfg.alpha = std::stod(a.substr(8));
    } else if (a == "--no-reduction") {
      cfg.reduction = false;
    } else if (starts_with(a, "--batch=")) {
      cfg.solver.batch = std::stoi(a.substr(8));
    } else if (starts_with(a, "--steps=")) {
      cfg.solver.n_steps = std::stoi(a.substr(8));
    } else if (starts_with(a, "--dt=")) {
      cfg.solver.dt = std::stof(a.substr(5));
      if (!(cfg.solver.dt > 0.f))
        throw std::runtime_error("--dt must be positive");
    } else if (starts_with(a, "--cluster=")) {
      cfg.solver.cluster = std::stoi(a.substr(10));
    } else if (starts_with(a, "--seed=")) {
      cfg.solver.seed = uint32_t(std::stoul(a.substr(7)));
    } else if (starts_with(a, "--target=")) {
      cfg.target = std::stod(a.substr(9));
    } else if (starts_with(a, "--repeat=")) {
      cfg.repeat = std::stoi(a.substr(9));
    } else if (starts_with(a, "--repeats=")) {
      cfg.repeats = std::stoi(a.substr(10));
    } else if (starts_with(a, "--warmup=")) {
      cfg.warmup = std::stoi(a.substr(9));
    } else if (starts_with(a, "--precision=")) {
      const std::string p = a.substr(12);
      if (p == "fp16") cfg.solver.precision = dsb::Precision::FP16;
      else if (p == "fp32") cfg.solver.precision = dsb::Precision::FP32;
      else if (p == "int8") cfg.solver.precision = dsb::Precision::INT8;
      else throw std::runtime_error("unknown precision: " + p);
    } else if (starts_with(a, "--variant=")) {
      const std::string v = a.substr(10);
      if (v == "block") cfg.solver.variant = dsb::Variant::Block;
      else if (v == "cluster") cfg.solver.variant = dsb::Variant::Cluster;
      else if (v == "sparse" || v == "csr-row")
        cfg.solver.variant = dsb::Variant::CsrRow;
      else if (v == "csr-block") cfg.solver.variant = dsb::Variant::CsrBlock;
      else if (v == "csr-cluster")
        cfg.solver.variant = dsb::Variant::CsrCluster;
      else if (v == "global-sync") cfg.solver.variant = dsb::Variant::GlobalSync;
      else if (v == "gemm") cfg.solver.variant = dsb::Variant::Gemm;
      else if (v == "bit") cfg.solver.variant = dsb::Variant::Bit;
      else if (v == "auto") cfg.solver.variant = dsb::Variant::Auto;
      else throw std::runtime_error("unknown variant: " + v);
    } else if (a == "--no-tf32") {
      cfg.solver.tf32 = false;
    } else if (a == "--l2") {
      cfg.solver.l2_persist = true;
    } else if (a == "--no-gauge") {
      cfg.apply_gauge = false;
    } else if (a == "--csv") {
      cfg.csv = true;
    } else if (a == "--plan-only") {
      cfg.plan_only = true;
    } else if (starts_with(a, "-")) {
      throw std::runtime_error("unknown option: " + a);
    } else {
      cfg.files.push_back(a);
    }
  }
  if (cfg.files.empty()) {
    usage(argv[0]);
    std::exit(1);
  }
  if (cfg.repeats <= 0 || cfg.warmup < 0)
    throw std::runtime_error("repeats must be positive and warmup nonnegative");
  return cfg;
}

/// Fraction of non-zero entries in a dense matrix. FastHare merges spins, and
/// merging accumulates couplings onto the representative, so the reduced
/// instance is usually denser than the original -- which is the argument for
/// keeping a dense kernel. Worth checking per instance before believing it.
double density(const Eigen::MatrixXd& m) {
  if (m.size() == 0) return 0.0;
  long long nnz = 0;
  for (int c = 0; c < m.cols(); ++c)
    for (int r = 0; r < m.rows(); ++r)
      if (m(r, c) != 0.0) ++nnz;
  return double(nnz) / double(m.size());
}

/// Lift a solution on the reduced instance back to the original spins.
///
/// The standard form carries an extra node at index n that absorbs the
/// external field, so the recovered assignment is only defined up to the sign
/// of that node. Multiplying through by it is what pins the gauge; without it
/// roughly half the instances come back mirrored and score the wrong
/// objective. (--no-gauge reproduces the older, unpinned behaviour.)
Eigen::VectorXd lift_solution(const dsb::Reduction& reduction, int n,
                              const Eigen::VectorXd& reduced_spins,
                              bool apply_gauge, double* gauge_out) {
  std::vector<double> full(reduction.sign.size(), 0.0);
  for (std::size_t i = 0; i < full.size(); ++i) {
    double v = 1.0;
    if (!reduction.fully_reduced) {
      const int mapped = reduction.spin_map[i];
      v = (mapped >= 0 && mapped < int(reduced_spins.size()))
              ? reduced_spins(mapped)
              : 0.0;
    }
    full[i] = v * double(reduction.sign[i]);
  }

  double gauge = 1.0;
  if (apply_gauge && int(full.size()) > n && full[std::size_t(n)] != 0.0)
    gauge = full[std::size_t(n)];
  if (gauge_out) *gauge_out = gauge;

  Eigen::VectorXd spins = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n && i < int(full.size()); ++i)
    spins(i) = full[std::size_t(i)] * gauge;
  return spins;
}

Eigen::MatrixXd standard_solver_coupling(const dsb::QplibInstance& instance) {
  const int n = instance.size();
  Eigen::MatrixXd coupling = Eigen::MatrixXd::Zero(n + 1, n + 1);
  coupling.topLeftCorner(n, n) = instance.scaled_coupling;
  coupling.block(0, n, n, 1) = instance.scaled_field;
  coupling.block(n, 0, 1, n) = instance.scaled_field.transpose();
  return coupling;
}

Eigen::VectorXd lift_standard_solution(const Eigen::VectorXd& standard,
                                       int n, double* gauge_out) {
  const double gauge = standard.size() > n ? standard(n) : 1.0;
  if (gauge_out) *gauge_out = gauge;
  return standard.head(n) * gauge;
}

struct RunResult {
  int         reduced_n        = 0;
  int         cluster_used     = 0;
  double      density_reduced  = 0.0;
  double      gauge            = 1.0;
  double      objective        = 0.0;
  double      preprocess_s     = 0.0;
  double      setup_s          = 0.0;
  double      reconstruction_s = 0.0;
  double      solver_s         = 0.0;
  double      evaluation_s     = 0.0;
  double      total_s          = 0.0;
  double      wall_s           = 0.0;
  double      reduction_ratio  = 0.0;
  std::string variant_used     = "fully-reduced";
  std::string storage          = "none";
  std::size_t gpu_memory_bytes = 0;
};

Eigen::VectorXd solve_matrix(const Eigen::MatrixXd& coupling,
                             const Config& cfg, uint32_t seed,
                             RunResult& result) {
  dsb::Options options = cfg.solver;
  options.seed = seed;
  if (dsb::is_csr_variant(options.variant)) {
    const auto setup_start = std::chrono::steady_clock::now();
    options.precision = dsb::Precision::FP32;
    const dsb::SparseMatrix sparse = dsb::sparse_from_dense(coupling);
    dsb::SparseSolver solver(sparse, options);
    const auto setup_end = std::chrono::steady_clock::now();
    result.setup_s +=
        std::chrono::duration<double>(setup_end - setup_start).count();
    solver.run();
    result.cluster_used = solver.cluster_size();
    result.variant_used = dsb::to_string(solver.variant());
    // Keep `auto` rows distinguishable from explicit-variant rows (as
    // solve_gset does) so the summaries never merge auto->X with manual X.
    if (cfg.solver.variant == dsb::Variant::Auto)
      result.variant_used = "auto->" + result.variant_used;
    result.storage = "csr";
    result.gpu_memory_bytes = solver.device_bytes();
    result.solver_s = double(solver.last_run_ms()) * 1e-3;
    const auto extraction_start = std::chrono::steady_clock::now();
    Eigen::VectorXd spins = solver.best_spins();
    const auto extraction_end = std::chrono::steady_clock::now();
    result.reconstruction_s += std::chrono::duration<double>(
        extraction_end - extraction_start).count();
    return spins;
  }

  const auto setup_start = std::chrono::steady_clock::now();
  dsb::Solver solver(coupling, options);
  const auto setup_end = std::chrono::steady_clock::now();
  result.setup_s +=
      std::chrono::duration<double>(setup_end - setup_start).count();
  solver.run();
  result.cluster_used = solver.plan().cluster_size;
  result.variant_used = dsb::to_string(solver.plan().variant);
  if (cfg.solver.variant == dsb::Variant::Auto)
    result.variant_used = "auto->" + result.variant_used;
  result.storage = "dense";
  result.gpu_memory_bytes = solver.device_bytes();
  result.solver_s = double(solver.last_run_ms()) * 1e-3;
  const auto extraction_start = std::chrono::steady_clock::now();
  Eigen::VectorXd spins = solver.best_spins();
  const auto extraction_end = std::chrono::steady_clock::now();
  result.reconstruction_s += std::chrono::duration<double>(
      extraction_end - extraction_start).count();
  return spins;
}

RunResult run_once(const dsb::QplibInstance& instance, const Config& cfg,
                   uint32_t seed) {
  const int  n  = instance.size();
  const auto t0 = std::chrono::steady_clock::now();

  RunResult result;
  Eigen::VectorXd spins;
  if (cfg.reduction) {
    const dsb::Reduction reduction = dsb::reduce_ising(
        -instance.scaled_coupling, -instance.scaled_field, cfg.alpha);
    result.reduced_n = reduction.fully_reduced ? 0 : reduction.reduced_size();
    result.density_reduced =
        reduction.fully_reduced ? 0.0 : density(reduction.coupling);
    result.preprocess_s = reduction.seconds;
    result.reduction_ratio = 1.0 - double(result.reduced_n) / double(n + 1);

    Eigen::VectorXd reduced_spins;
    if (!reduction.fully_reduced)
      reduced_spins = solve_matrix(-reduction.coupling, cfg, seed, result);
    const auto reconstruction_start = std::chrono::steady_clock::now();
    spins = lift_solution(reduction, n, reduced_spins, cfg.apply_gauge,
                          &result.gauge);
    const auto reconstruction_end = std::chrono::steady_clock::now();
    result.reconstruction_s += std::chrono::duration<double>(
        reconstruction_end - reconstruction_start).count();
  } else {
    const Eigen::MatrixXd coupling = standard_solver_coupling(instance);
    result.reduced_n = n + 1;
    result.density_reduced = density(coupling);
    const Eigen::VectorXd standard = solve_matrix(coupling, cfg, seed, result);
    const auto reconstruction_start = std::chrono::steady_clock::now();
    spins = lift_standard_solution(standard, n, &result.gauge);
    const auto reconstruction_end = std::chrono::steady_clock::now();
    result.reconstruction_s += std::chrono::duration<double>(
        reconstruction_end - reconstruction_start).count();
  }

  const auto te0 = std::chrono::steady_clock::now();
  result.objective = dsb::objective_value(instance, spins);
  const auto te1 = std::chrono::steady_clock::now();
  result.evaluation_s = std::chrono::duration<double>(te1 - te0).count();
  const auto t1 = std::chrono::steady_clock::now();
  result.total_s =
      result.preprocess_s + result.solver_s + result.reconstruction_s;
  result.wall_s = std::chrono::duration<double>(t1 - t0).count();
  return result;
}

}  // namespace

int main(int argc, char** argv) try {
  const Config cfg = parse_args(argc, argv);

  const int max_block_n = dsb::device_max_n_block();
  if (!cfg.csv) {
    std::cout << "shared memory budget : " << dsb::device_max_smem_optin()
              << " B/block\n"
              << "max N (single block) : " << max_block_n << "\n"
              << "precision            : " << dsb::to_string(cfg.solver.precision)
              << "   batch " << cfg.solver.batch << "   steps "
              << cfg.solver.n_steps << "\n\n";
  } else if (cfg.plan_only) {
    std::cout << "instance,n_original,n_reduced,density_original,"
                 "density_reduced,fits_single_block,variant,cluster,smem_bytes\n";
  } else {
    std::cout << "implementation,instance,repeat,seed,n_original,"
                 "n_before_reduction,n_reduced,agents,steps,precision,variant,"
                 "storage,reduction,early_stopping,objective,target,success,"
                 "preprocess_s,setup_s,solver_s,reconstruction_s,evaluation_s,"
                 "total_s,wall_s,reduction_ratio,gpu_memory_bytes\n";
  }

  for (const std::string& path : cfg.files) {
    const dsb::QplibInstance instance = dsb::load_qplib(path);
    const int                n        = instance.size();

    if (cfg.plan_only) {
      Eigen::MatrixXd planned_coupling;
      int reduced_n = n + 1;
      if (cfg.reduction) {
        const dsb::Reduction reduction = dsb::reduce_ising(
            -instance.scaled_coupling, -instance.scaled_field, cfg.alpha);
        reduced_n = reduction.fully_reduced ? 0 : reduction.reduced_size();
        if (!reduction.fully_reduced)
          planned_coupling = reduction.coupling;
      } else {
        planned_coupling = standard_solver_coupling(instance);
      }
      if (cfg.csv) {
        std::cout << instance.name << ',' << n << ',' << reduced_n << ','
                  << std::setprecision(9) << density(instance.coupling) << ','
                  << (reduced_n > 0 ? density(planned_coupling) : 0.0) << ','
                  << ((reduced_n > 0 && reduced_n <= max_block_n) ? 1 : 0)
                  << ',';
        if (reduced_n > 0) {
          try {
            if (dsb::is_csr_variant(cfg.solver.variant)) {
              std::cout << dsb::to_string(cfg.solver.variant) << ','
                        << (cfg.solver.variant == dsb::Variant::CsrCluster
                                ? (cfg.solver.cluster > 0 ? cfg.solver.cluster : 8)
                                : 1)
                        << ",0";
            } else {
              const dsb::LaunchPlan plan =
                  dsb::make_plan(reduced_n, cfg.solver.precision,
                                 cfg.solver.variant, cfg.solver.cluster, -1,
                                 cfg.solver.batch);
              std::cout << dsb::to_string(plan.variant) << ','
                        << plan.cluster_size << ',' << plan.smem_bytes;
            }
          } catch (const std::exception&) {
            std::cout << "error,0,0";
          }
        } else {
          std::cout << "fully-reduced,0,0";
        }
        std::cout << '\n';
      } else {
        std::cout << instance.name << "  n=" << n << "  reduced=" << reduced_n
                  << "  density " << std::fixed << std::setprecision(4)
                  << density(instance.coupling) << " -> "
                  << (reduced_n > 0 ? density(planned_coupling) : 0.0);
        if (reduced_n > 0) {
          try {
            if (dsb::is_csr_variant(cfg.solver.variant)) {
              std::cout << "  " << dsb::to_string(cfg.solver.variant)
                        << "  storage=csr";
            } else {
              const dsb::LaunchPlan plan =
                  dsb::make_plan(reduced_n, cfg.solver.precision,
                                 cfg.solver.variant, cfg.solver.cluster, -1,
                                 cfg.solver.batch);
              std::cout << "  " << dsb::to_string(plan.variant)
                        << "  cluster=" << plan.cluster_size
                        << "  smem=" << plan.smem_bytes << "B";
            }
          } catch (const std::exception& e) {
            std::cout << "  DOES NOT FIT (" << e.what() << ")";
          }
        }
        std::cout << '\n';
      }
      continue;
    }

    // Warm up in this process so CUDA context creation and first-use effects
    // are excluded for both this executable and the public Python baseline.
    for (int warmup = 0; warmup < cfg.warmup; ++warmup)
      (void)run_once(instance, cfg, cfg.solver.seed + uint32_t(warmup));

    for (int run = 0; run < cfg.repeats; ++run) {
      const int      repeat = cfg.repeat + run;
      const uint32_t seed   = cfg.solver.seed + uint32_t(run);
      const RunResult result = run_once(instance, cfg, seed);

      if (cfg.csv) {
        std::cout << "dsb-gpu," << instance.name << ',' << repeat << ','
                  << seed << ',' << n << ',' << (n + 1) << ','
                  << result.reduced_n << ','
                  << cfg.solver.batch << ',' << cfg.solver.n_steps << ','
                  << dsb::to_string(cfg.solver.precision) << ','
                  << result.variant_used << ',' << result.storage << ','
                  << (cfg.reduction ? 1 : 0) << ",0,"
                  << std::setprecision(17)
                  << result.objective << ',';
        if (cfg.target) {
          std::cout << *cfg.target << ','
                    << (result.objective >= *cfg.target ? 1 : 0);
        } else {
          std::cout << ',';
        }
        std::cout << ',' << std::setprecision(9) << result.preprocess_s
                  << ',' << result.setup_s << ',' << result.solver_s << ','
                  << result.reconstruction_s << ',' << result.evaluation_s
                  << ',' << result.total_s << ',' << result.wall_s << ','
                  << result.reduction_ratio << ',' << result.gpu_memory_bytes
                  << "\n";
      } else {
        std::cout << "==== " << instance.name;
        if (cfg.repeats > 1) std::cout << " repeat " << repeat;
        std::cout << " ====\n"
                  << "  seed             : " << seed << "\n"
                  << "  n                : " << n << "\n"
                  << "  reduction        : "
                  << (cfg.reduction ? "FastHare" : "disabled") << "\n"
                  << "  solver size      : "
                  << (result.reduced_n == 0
                          ? std::string("fully reduced")
                          : std::to_string(result.reduced_n))
                  << "   (" << std::fixed << std::setprecision(3)
                  << result.preprocess_s << " s)\n"
                  << "  density         : " << std::setprecision(4)
                  << density(instance.coupling) << " -> "
                  << result.density_reduced << "\n";
        if (result.reduced_n > 0)
          std::cout << "  blocks / replica : " << result.cluster_used << "\n";
        std::cout << "  gauge            : " << std::showpos << result.gauge
                  << std::noshowpos << "\n"
                  << "  objective        : " << std::setprecision(10)
                  << result.objective << "\n"
                  << "  pipeline time    : " << std::setprecision(3)
                  << result.total_s << " s\n"
                  << "  measured wall    : " << result.wall_s << " s\n\n";
      }
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
