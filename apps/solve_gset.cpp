// G-set / edge-list Max-Cut driver with dense and CSR-native dSB paths.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "dsb/gset.hpp"
#include "dsb/reduction.hpp"
#include "dsb/bitfused.cuh"
#include "dsb/solver.hpp"
#include "dsb/sparse_solver.hpp"

namespace {

struct Config {
  double alpha = 0.2;
  dsb::Options solver;
  bool reduction = true;
  bool csv = false;
  bool plan_only = false;
  int repeat = 0;
  int repeats = 1;
  int warmup = 0;
  std::optional<double> target;
  std::vector<std::string> files;
  double bit_density = 0.04;
  double csr_block_degree = 4.5;   // auto: csr-block at or below this avg degree
  Config() { solver.precision = dsb::Precision::FP32; }
};

void usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options] <G-set/edge-list instance> [...]\n\n"
      << "  --alpha=F        FastHare merge threshold (default 0.2)\n"
      << "  --no-reduction   skip FastHare (recommended for scaling)\n"
      << "  --batch=N        replicas solved in parallel (default 200)\n"
      << "  --steps=N        integration steps (default 800)\n"
      << "  --dt=F           integration step size (default 1.0); pass the same\n"
      << "                   value to the public baseline as --sb-time-step\n"
      << "  --precision=P    fp32 (default), fp16 (dense variants), or int8\n"
      << "                   (gemm only; exact for {-1,0,+1} couplings)\n"
      << "  --variant=V      bit | csr-row | csr-block | csr-cluster | block |\n"
      << "                   cluster | global-sync | gemm | auto (default)\n"
      << "                   auto: bit when J is in {-1,0,+1}, the plan fits and\n"
      << "                   density >= --bit-density; else csr-row when sparse;\n"
      << "                   else gemm\n"
      << "  --bit-density=F  auto threshold for the bit path (default 0.04)\n"
      << "  --csr-block-degree=D  auto: csr-block when avg degree 2E/N <= D\n"
      << "                   (default 4.5; measured crossover vs csr-row)\n"
      << "  --no-tf32        gemm+fp32 on CUDA cores instead of TF32\n"
      << "  --cluster=N      blocks per replica; 0 = auto (default)\n"
      << "  --l2             dense path: request L2 persistence\n"
      << "  --seed=N         RNG seed (default 12345)\n"
      << "  --repeat=N       first repeat index written to CSV\n"
      << "  --repeats=N      measured runs in this process\n"
      << "  --warmup=N       unmeasured warm-up runs\n"
      << "  --target=F       target cut for success and TTS99\n"
      << "  --csv            emit machine-readable rows\n"
      << "  --plan-only      report storage/launch plan without solving\n";
}

bool starts_with(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

dsb::Variant parse_variant(const std::string& value) {
  if (value == "sparse" || value == "csr-row") return dsb::Variant::CsrRow;
  if (value == "csr-block") return dsb::Variant::CsrBlock;
  if (value == "csr-cluster") return dsb::Variant::CsrCluster;
  if (value == "block") return dsb::Variant::Block;
  if (value == "cluster") return dsb::Variant::Cluster;
  if (value == "global-sync") return dsb::Variant::GlobalSync;
  if (value == "gemm") return dsb::Variant::Gemm;
  if (value == "bit") return dsb::Variant::Bit;
  if (value == "auto") return dsb::Variant::Auto;
  throw std::runtime_error("unknown variant: " + value);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (starts_with(arg, "--alpha=")) {
      cfg.alpha = std::stod(arg.substr(8));
    } else if (arg == "--no-reduction") {
      cfg.reduction = false;
    } else if (starts_with(arg, "--batch=")) {
      cfg.solver.batch = std::stoi(arg.substr(8));
    } else if (starts_with(arg, "--steps=")) {
      cfg.solver.n_steps = std::stoi(arg.substr(8));
    } else if (starts_with(arg, "--dt=")) {
      cfg.solver.dt = std::stof(arg.substr(5));
      if (!(cfg.solver.dt > 0.f))
        throw std::runtime_error("--dt must be positive");
    } else if (starts_with(arg, "--cluster=")) {
      cfg.solver.cluster = std::stoi(arg.substr(10));
    } else if (starts_with(arg, "--seed=")) {
      cfg.solver.seed = uint32_t(std::stoul(arg.substr(7)));
    } else if (starts_with(arg, "--target=")) {
      cfg.target = std::stod(arg.substr(9));
    } else if (starts_with(arg, "--repeat=")) {
      cfg.repeat = std::stoi(arg.substr(9));
    } else if (starts_with(arg, "--repeats=")) {
      cfg.repeats = std::stoi(arg.substr(10));
    } else if (starts_with(arg, "--warmup=")) {
      cfg.warmup = std::stoi(arg.substr(9));
    } else if (starts_with(arg, "--precision=")) {
      const std::string precision = arg.substr(12);
      if (precision == "fp16") cfg.solver.precision = dsb::Precision::FP16;
      else if (precision == "fp32") cfg.solver.precision = dsb::Precision::FP32;
      else if (precision == "int8") cfg.solver.precision = dsb::Precision::INT8;
      else throw std::runtime_error("unknown precision: " + precision);
    } else if (starts_with(arg, "--variant=")) {
      cfg.solver.variant = parse_variant(arg.substr(10));
    } else if (arg == "--l2") {
      cfg.solver.l2_persist = true;
    } else if (arg == "--no-tf32") {
      cfg.solver.tf32 = false;
    } else if (starts_with(arg, "--bit-density=")) {
      cfg.bit_density = std::stod(arg.substr(14));
    } else if (starts_with(arg, "--csr-block-degree=")) {
      cfg.csr_block_degree = std::stod(arg.substr(19));
    } else if (arg == "--csv") {
      cfg.csv = true;
    } else if (arg == "--plan-only") {
      cfg.plan_only = true;
    } else if (starts_with(arg, "-")) {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      cfg.files.push_back(arg);
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

double density(const dsb::SparseMatrix& matrix) {
  if (matrix.n == 0) return 0.0;
  return double(matrix.nnz()) / (double(matrix.n) * double(matrix.n));
}

template <class Reduction>
Eigen::VectorXd lift_solution(const Reduction& reduction, int n,
                              const Eigen::VectorXd& reduced_spins) {
  Eigen::VectorXd spins = Eigen::VectorXd::Ones(n);
  for (int i = 0; i < n && i < int(reduction.sign.size()); ++i) {
    double value = 1.0;
    if (!reduction.fully_reduced) {
      const int mapped = reduction.spin_map[std::size_t(i)];
      value = (mapped >= 0 && mapped < reduced_spins.size())
                  ? reduced_spins(mapped)
                  : 1.0;
    }
    spins(i) = value * double(reduction.sign[std::size_t(i)]);
  }
  return spins;
}

struct RunResult {
  int reduced_n = 0;
  int cluster_used = 0;
  std::size_t solver_nnz = 0;
  std::size_t coupling_bytes = 0;
  std::size_t gpu_memory_bytes = 0;
  double density_reduced = 0.0;
  double objective = 0.0;
  double preprocess_s = 0.0;
  double setup_s = 0.0;
  double solver_s = 0.0;
  double reconstruction_s = 0.0;
  double evaluation_s = 0.0;
  double total_s = 0.0;
  double wall_s = 0.0;
  double reduction_ratio = 0.0;
  std::string storage = "none";
  std::string variant_used = "fully-reduced";
};

RunResult run_sparse(const dsb::GsetInstance& instance, const Config& cfg,
                     uint32_t seed) {
  const auto total_start = std::chrono::steady_clock::now();
  RunResult result;
  result.storage = "csr";
  Eigen::VectorXd spins;

  if (cfg.reduction) {
    const auto reduction = dsb::reduce_ising(
        instance.coupling, Eigen::VectorXd::Zero(instance.size()), cfg.alpha);
    result.preprocess_s = reduction.seconds;
    result.reduced_n = reduction.fully_reduced ? 0 : reduction.reduced_size();
    result.reduction_ratio =
        1.0 - double(result.reduced_n) / double(instance.size());
    result.density_reduced = reduction.fully_reduced ? 0.0 : density(reduction.coupling);
    if (reduction.fully_reduced) {
      const auto reconstruction_start = std::chrono::steady_clock::now();
      spins = lift_solution(reduction, instance.size(), Eigen::VectorXd());
      const auto reconstruction_end = std::chrono::steady_clock::now();
      result.reconstruction_s = std::chrono::duration<double>(
          reconstruction_end - reconstruction_start).count();
    } else {
      const dsb::SparseMatrix problem = reduction.coupling.scaled(-1.f);
      dsb::Options options = cfg.solver;
      options.seed = seed;
      const auto setup_start = std::chrono::steady_clock::now();
      dsb::SparseSolver solver(problem, options);
      const auto setup_end = std::chrono::steady_clock::now();
      result.setup_s = std::chrono::duration<double>(
          setup_end - setup_start).count();
      solver.run();
      result.solver_s = double(solver.last_run_ms()) * 1e-3;
      result.variant_used = dsb::to_string(solver.variant());
      result.cluster_used = solver.cluster_size();
      result.solver_nnz = problem.nnz();
      result.coupling_bytes = solver.coupling_bytes();
      result.gpu_memory_bytes = solver.device_bytes();
      const auto reconstruction_start = std::chrono::steady_clock::now();
      const Eigen::VectorXd reduced_spins = solver.best_spins();
      spins = lift_solution(reduction, instance.size(), reduced_spins);
      const auto reconstruction_end = std::chrono::steady_clock::now();
      result.reconstruction_s = std::chrono::duration<double>(
          reconstruction_end - reconstruction_start).count();
    }
  } else {
    const dsb::SparseMatrix problem = instance.coupling.scaled(-1.f);
    result.reduced_n = instance.size();
    result.density_reduced = density(problem);
    dsb::Options options = cfg.solver;
    options.seed = seed;
    const auto setup_start = std::chrono::steady_clock::now();
    dsb::SparseSolver solver(problem, options);
    const auto setup_end = std::chrono::steady_clock::now();
    result.setup_s = std::chrono::duration<double>(
        setup_end - setup_start).count();
    solver.run();
    result.solver_s = double(solver.last_run_ms()) * 1e-3;
    result.variant_used = dsb::to_string(solver.variant());
    result.cluster_used = solver.cluster_size();
    result.solver_nnz = problem.nnz();
    result.coupling_bytes = solver.coupling_bytes();
    result.gpu_memory_bytes = solver.device_bytes();
    const auto reconstruction_start = std::chrono::steady_clock::now();
    spins = solver.best_spins();
    const auto reconstruction_end = std::chrono::steady_clock::now();
    result.reconstruction_s = std::chrono::duration<double>(
        reconstruction_end - reconstruction_start).count();
  }

  const auto evaluation_start = std::chrono::steady_clock::now();
  result.objective = dsb::cut_value(instance, spins);
  const auto evaluation_end = std::chrono::steady_clock::now();
  result.evaluation_s =
      std::chrono::duration<double>(evaluation_end - evaluation_start).count();
  result.total_s =
      result.preprocess_s + result.solver_s + result.reconstruction_s;
  result.wall_s =
      std::chrono::duration<double>(evaluation_end - total_start).count();
  return result;
}

// FastHare can merge spins, which turns +-1 couplings into +-2, +-3, ...  The
// bit kernel is exact only on {-1,0,+1}; when a reduced matrix leaves that set
// the dense path falls back to gemm and says so (the CSV `variant` column
// records what actually ran).
dsb::Options dense_options(const Config& cfg, const Eigen::MatrixXd& matrix,
                           uint32_t seed) {
  dsb::Options options = cfg.solver;
  options.seed = seed;
  const bool exact_path = options.variant == dsb::Variant::Bit ||
                          options.precision == dsb::Precision::INT8;
  if (exact_path) {
    bool pm1 = false;
    const bool ternary = dsb::bitfused::classify(
        matrix.data(), int(matrix.rows()), std::size_t(matrix.outerStride()), pm1);
    if (!ternary) {
      std::cerr << "note: couplings left {-1,0,+1} after reduction; "
                   "exact path -> gemm fp16 for this run\n";
      options.variant   = dsb::Variant::Gemm;
      options.precision = dsb::Precision::FP16;
    }
  }
  return options;
}

RunResult run_dense(const dsb::GsetInstance& instance, const Config& cfg,
                    uint32_t seed) {
  const auto total_start = std::chrono::steady_clock::now();
  RunResult result;
  result.storage = "dense";
  Eigen::VectorXd spins;

  if (cfg.reduction) {
    const auto reduction = dsb::reduce_ising(
        instance.coupling, Eigen::VectorXd::Zero(instance.size()), cfg.alpha);
    result.preprocess_s = reduction.seconds;
    result.reduced_n = reduction.fully_reduced ? 0 : reduction.reduced_size();
    result.reduction_ratio =
        1.0 - double(result.reduced_n) / double(instance.size());
    result.density_reduced =
        reduction.fully_reduced ? 0.0 : density(reduction.coupling);
    if (reduction.fully_reduced) {
      const auto reconstruction_start = std::chrono::steady_clock::now();
      spins = lift_solution(reduction, instance.size(), Eigen::VectorXd());
      const auto reconstruction_end = std::chrono::steady_clock::now();
      result.reconstruction_s = std::chrono::duration<double>(
          reconstruction_end - reconstruction_start).count();
    } else {
      const auto setup_start = std::chrono::steady_clock::now();
      const Eigen::MatrixXd reduced_dense = -reduction.coupling.dense();
      const dsb::Options options = dense_options(cfg, reduced_dense, seed);
      dsb::Solver solver(reduced_dense, options);
      const auto setup_end = std::chrono::steady_clock::now();
      result.setup_s = std::chrono::duration<double>(
          setup_end - setup_start).count();
      solver.run();
      result.cluster_used = solver.plan().cluster_size;
      result.variant_used = dsb::to_string(solver.plan().variant);
      result.solver_s = double(solver.last_run_ms()) * 1e-3;
      result.coupling_bytes = solver.j_bytes();
      result.gpu_memory_bytes = solver.device_bytes();
      const auto reconstruction_start = std::chrono::steady_clock::now();
      const Eigen::VectorXd reduced_spins = solver.best_spins();
      spins = lift_solution(reduction, instance.size(), reduced_spins);
      const auto reconstruction_end = std::chrono::steady_clock::now();
      result.reconstruction_s = std::chrono::duration<double>(
          reconstruction_end - reconstruction_start).count();
    }
  } else {
    const Eigen::MatrixXd dense = -instance.coupling.dense();
    result.reduced_n = instance.size();
    result.density_reduced = density(instance.coupling);
    const auto setup_start = std::chrono::steady_clock::now();
    const dsb::Options options = dense_options(cfg, dense, seed);
    dsb::Solver solver(dense, options);
    const auto setup_end = std::chrono::steady_clock::now();
    result.setup_s = std::chrono::duration<double>(
        setup_end - setup_start).count();
    solver.run();
    result.cluster_used = solver.plan().cluster_size;
    result.variant_used = dsb::to_string(solver.plan().variant);
    result.solver_s = double(solver.last_run_ms()) * 1e-3;
    result.coupling_bytes = solver.j_bytes();
    result.gpu_memory_bytes = solver.device_bytes();
    const auto reconstruction_start = std::chrono::steady_clock::now();
    spins = solver.best_spins();
    const auto reconstruction_end = std::chrono::steady_clock::now();
    result.reconstruction_s = std::chrono::duration<double>(
        reconstruction_end - reconstruction_start).count();
  }

  const auto evaluation_start = std::chrono::steady_clock::now();
  result.objective = dsb::cut_value(instance, spins);
  const auto evaluation_end = std::chrono::steady_clock::now();
  result.evaluation_s =
      std::chrono::duration<double>(evaluation_end - evaluation_start).count();
  result.total_s =
      result.preprocess_s + result.solver_s + result.reconstruction_s;
  result.wall_s =
      std::chrono::duration<double>(evaluation_end - total_start).count();
  return result;
}

// Resolve `auto` before anything is densified.  The rule, and why:
//
//   bit      J in {-1,0,+1}, the (N, batch) bit plan is resident on the GPU,
//            and density >= bit_density.  The bit kernel's cost is N*B*W
//            popcounts per step regardless of sparsity, so on very sparse
//            instances the CSR gather is cheaper even when bit would fit.
//   csr-block otherwise, when the average degree 2E/N <= csr_block_degree:
//            one persistent block per replica, no grid sync.  On the v5
//            G-set runs (B=512, every step count) it beat csr-row by
//            1.7-2.1x on all degree<=4 instances (G11/G32/G48/G70/G72/G81)
//            and lost from degree ~5 upwards (G55/G60).
//   csr-row  otherwise, when the instance is sparse (density < 0.25): CSR
//            storage plus a persistent one-barrier-per-step kernel.  This is
//            the large-sparse-G-set path (N >= 7000).
//   gemm     otherwise: dense, real-valued or too dense for CSR.
//
// N^2 dense storage is only ever built on the bit/gemm paths, so a 20000-node
// G-set instance never allocates 3.2 GB of Eigen doubles just to be dispatched.
dsb::Variant resolve_auto(const dsb::GsetInstance& instance, const Config& cfg) {
  if (cfg.solver.variant != dsb::Variant::Auto) return cfg.solver.variant;
  const int n = instance.size();
  const double dens = density(instance.coupling);

  bool ternary = true, pm1 = true;
  for (float v : instance.coupling.values) {
    if (v == 1.f || v == -1.f) continue;
    pm1 = false;
    if (v != 0.f) { ternary = false; break; }
  }
  // "every off-diagonal is +-1" also needs the graph to be complete.
  pm1 = pm1 && instance.coupling.nnz() == std::size_t(n) * std::size_t(n - 1);

  if (cfg.solver.precision == dsb::Precision::INT8) return dsb::Variant::Gemm;
  if (ternary && dens >= cfg.bit_density) {
    const dsb::bitfused::Plan plan =
        dsb::bitfused::make_bit_plan(n, cfg.solver.batch, pm1);
    if (dsb::bitfused::fits(plan)) return dsb::Variant::Bit;
  }
  // nnz() counts both triangles of the symmetric coupling, so nnz/n = 2E/N.
  const double avg_degree = double(instance.coupling.nnz()) / double(n);
  if (avg_degree <= cfg.csr_block_degree) return dsb::Variant::CsrBlock;
  if (dens < 0.25) return dsb::Variant::CsrRow;
  return dsb::Variant::Gemm;
}

RunResult run_once(const dsb::GsetInstance& instance, const Config& cfg_in,
                   uint32_t seed) {
  Config cfg = cfg_in;
  cfg.solver.variant = resolve_auto(instance, cfg_in);
  RunResult result = dsb::is_csr_variant(cfg.solver.variant)
                         ? run_sparse(instance, cfg, seed)
                         : run_dense(instance, cfg, seed);
  // Keep `auto` rows distinguishable from explicit-variant rows in the CSV so
  // the summaries do not merge them.
  if (cfg_in.solver.variant == dsb::Variant::Auto &&
      result.variant_used != "fully-reduced")
    result.variant_used = "auto->" + result.variant_used;
  return result;
}

}  // namespace

int main(int argc, char** argv) try {
  const Config cfg = parse_args(argc, argv);
  if (cfg.csv && !cfg.plan_only) {
    std::cout << "implementation,instance,repeat,seed,n_original,"
                 "n_before_reduction,n_reduced,edges,agents,steps,precision,"
                 "variant,storage,reduction,early_stopping,objective,target,"
                 "success,preprocess_s,setup_s,solver_s,reconstruction_s,"
                 "evaluation_s,total_s,wall_s,reduction_ratio,time_per_step_s,"
                 "edge_updates_per_s,nnz_solver,coupling_bytes,"
                 "gpu_memory_bytes\n";
  } else if (cfg.csv) {
    std::cout << "instance,n_original,edges,nnz,density,variant,storage,"
                 "reduction,coupling_bytes\n";
  }

  for (const std::string& path : cfg.files) {
    const dsb::GsetInstance instance = dsb::load_gset(path);
    if (cfg.plan_only) {
      Config resolved = cfg;
      resolved.solver.variant = resolve_auto(instance, cfg);
      const bool sparse = dsb::is_csr_variant(resolved.solver.variant);
      const std::size_t dense_bytes =
          std::size_t(instance.size()) * instance.size() *
          dsb::coupling_element_size(cfg.solver.precision);
      if (cfg.csv) {
        std::cout << instance.name << ',' << instance.size() << ','
                  << instance.edge_count() << ',' << instance.coupling.nnz() << ','
                  << std::setprecision(9) << density(instance.coupling) << ','
                  << dsb::to_string(resolved.solver.variant) << ','
                  << (sparse ? "csr" : "dense") << ',' << (cfg.reduction ? 1 : 0)
                  << ',' << (sparse ? instance.coupling.bytes() : dense_bytes) << '\n';
      } else {
        std::cout << instance.name << ": N=" << instance.size()
                  << " edges=" << instance.edge_count()
                  << " nnz=" << instance.coupling.nnz()
                  << " density=" << density(instance.coupling)
                  << " variant=" << dsb::to_string(resolved.solver.variant)
                  << " storage=" << (sparse ? "csr" : "dense") << '\n';
      }
      continue;
    }

    for (int warmup = 0; warmup < cfg.warmup; ++warmup)
      (void)run_once(instance, cfg, cfg.solver.seed + uint32_t(warmup));

    for (int run = 0; run < cfg.repeats; ++run) {
      const int repeat = cfg.repeat + run;
      const uint32_t seed = cfg.solver.seed + uint32_t(run);
      const RunResult result = run_once(instance, cfg, seed);
      const double time_per_step = result.solver_s > 0.0
                                       ? result.solver_s / cfg.solver.n_steps
                                       : 0.0;
      const double edge_updates =
          result.storage == "csr" && result.solver_s > 0.0
              ? (0.5 * double(result.solver_nnz) * cfg.solver.batch *
                 cfg.solver.n_steps / result.solver_s)
              : 0.0;

      if (cfg.csv) {
        std::cout << "dsb-gpu," << instance.name << ',' << repeat << ',' << seed
                  << ',' << instance.size() << ',' << instance.size() << ','
                  << result.reduced_n << ','
                  << instance.edge_count() << ',' << cfg.solver.batch << ','
                  << cfg.solver.n_steps << ',' << dsb::to_string(cfg.solver.precision)
                  << ',' << result.variant_used << ',' << result.storage << ','
                  << (cfg.reduction ? 1 : 0) << ",0," << std::setprecision(17)
                  << result.objective << ',';
        if (cfg.target)
          std::cout << *cfg.target << ',' << (result.objective >= *cfg.target ? 1 : 0);
        else
          std::cout << ',';
        std::cout << ',' << std::setprecision(9) << result.preprocess_s
                  << ',' << result.setup_s << ',' << result.solver_s << ','
                  << result.reconstruction_s << ',' << result.evaluation_s
                  << ',' << result.total_s << ',' << result.wall_s << ','
                  << result.reduction_ratio << ',' << time_per_step << ',';
        if (result.storage == "csr") std::cout << edge_updates;
        std::cout << ',' << result.solver_nnz << ',' << result.coupling_bytes << ','
                  << result.gpu_memory_bytes << '\n';
      } else {
        std::cout << "==== " << instance.name << " repeat " << repeat << " ====\n"
                  << "  N / edges        : " << instance.size() << " / "
                  << instance.edge_count() << "\n"
                  << "  variant / storage: " << result.variant_used << " / "
                  << result.storage << "\n"
                  << "  cut value        : " << std::setprecision(12)
                  << result.objective << "\n"
                  << "  GPU solver       : " << std::setprecision(6)
                  << result.solver_s << " s  (" << time_per_step << " s/step)\n"
                  << "  GPU memory       : " << result.gpu_memory_bytes << " bytes\n";
        if (result.storage == "csr")
          std::cout << "  edge updates/s   : " << edge_updates << '\n';
        std::cout << "  pipeline time    : " << result.total_s << " s\n"
                  << "  measured wall    : " << result.wall_s << " s\n\n";
      }
    }
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
