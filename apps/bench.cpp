// apps/bench.cpp
//
// Runs every kernel variant on the same instance and reports, per variant,
// how many times J was effectively dragged out of device memory per step.
//
// The number that matters is the last column, "J fetches/step":
//
//     J fetches/step = (time per step) x (measured read bandwidth) / sizeof(J)
//
//   ~= 1               every byte of J was fetched once and reused by all the
//                      concurrent replicas. L2 is already doing the sharing a
//                      GEMM would do explicitly, and the shared-memory design
//                      wins outright.
//
//   ~= replicas/SM-wave the replicas are out of phase and each one is pulling
//                      its own copy of J through the memory system. This is the
//                      case where tiling, or a GEMM formulation, is worth the
//                      trouble.
//
// It is an estimate, not a counter: it assumes the kernel is bandwidth bound,
// which stops being true for small N. Cross-check with
// `ncu --metrics dram__bytes.sum` before quoting it in a paper.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dsb/solver.hpp"

namespace {

/// Complete +-1 graph, the K2000 shape: what the bit kernel is for.
Eigen::MatrixXd random_pm1(int n, uint32_t seed) {
  std::mt19937    rng(seed);
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double v = (rng() & 1u) ? 1.0 : -1.0;
      m(i, j)        = v;
      m(j, i)        = v;
    }
  return m;
}

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

struct Row {
  std::string variant;
  int         group   = 1;
  bool        l2      = false;
  double      ms_step = 0.0;
  double      fetches = 0.0;
  std::size_t pinned  = 0;
  bool        ok      = true;
  std::string note;
};

Row measure(const Eigen::MatrixXd& coupling, dsb::Options options,
            double bandwidth) {
  Row row;
  row.variant = dsb::to_string(options.variant);
  row.l2      = options.l2_persist;
  try {
    dsb::Solver solver(coupling, options);
    row.group = solver.plan().cluster_size;
    row.variant = dsb::to_string(solver.plan().variant);

    solver.run();  // warm-up, also pays the one-off setup
    solver.run();

    row.ms_step = double(solver.last_run_ms()) / double(options.n_steps);
    row.pinned  = solver.l2_pinned_bytes();
    row.fetches = (row.ms_step * 1e-3 * bandwidth) / double(solver.j_bytes());
  } catch (const std::exception& e) {
    row.ok   = false;
    row.note = e.what();
  }
  return row;
}

void print(const Row& row, double baseline_ms) {
  std::cout << "  " << std::left << std::setw(13) << row.variant;
  std::cout << std::setw(5) << row.group;
  std::cout << std::setw(6) << (row.l2 ? "on" : "off");

  if (!row.ok) {
    std::cout << "  skipped: " << row.note.substr(0, 60) << "\n";
    return;
  }
  std::cout << std::right << std::fixed << std::setprecision(3) << std::setw(10)
            << row.ms_step << std::setw(15) << std::setprecision(1)
            << row.fetches << std::setw(9) << std::setprecision(2)
            << (baseline_ms > 0 ? baseline_ms / row.ms_step : 1.0) << "x";
  if (row.l2 && row.pinned)
    std::cout << "   pinned " << (row.pinned >> 20) << " MB";
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) try {
  const int         n       = (argc > 1) ? std::atoi(argv[1]) : 2000;
  const int         batch   = (argc > 2) ? std::atoi(argv[2]) : 200;
  const int         n_steps = (argc > 3) ? std::atoi(argv[3]) : 100;
  const std::string shape   = (argc > 4) ? argv[4] : "pm1";  // pm1 | gauss
  if (shape != "pm1" && shape != "gauss")
    throw std::runtime_error("usage: bench [N] [batch] [steps] [pm1|gauss]");

  std::cout << "measuring device read bandwidth...\n";
  const double bandwidth = dsb::measure_read_bandwidth();

  std::cout << std::fixed << std::setprecision(0)
            << "  read bandwidth   : " << bandwidth / 1e9 << " GB/s\n"
            << "  L2 cache         : " << dsb::device_l2_bytes() / 1048576.0
            << " MB   (max persisting "
            << dsb::device_max_persisting_l2_bytes() / 1048576.0 << " MB)\n"
            << "  shared per block : " << dsb::device_max_smem_optin() << " B\n\n";

  const Eigen::MatrixXd coupling =
      (shape == "pm1") ? random_pm1(n, 7) : random_symmetric(n, 7);

  dsb::Options base;
  base.batch     = batch;
  base.n_steps   = n_steps;
  base.precision = dsb::Precision::FP16;

  const double j_mb =
      double(std::size_t(n) * dsb::row_stride(n) * 2) / 1048576.0;
  std::cout << "N=" << n << "  batch=" << batch << "  steps=" << n_steps
            << "  shape=" << shape << "  J=" << std::setprecision(1) << j_mb
            << " MB  (" << dsb::to_string(base.precision) << ")\n\n"
            << "  variant      C    L2     ms/step   J fetches/step  "
               "speedup\n"
            << "  -----------------------------------------------------------\n";

  std::vector<Row> rows;
  // The baseline is gemm on tensor cores (FP16 storage -> HMMA).  It is the
  // strongest library formulation and the only fair yardstick; global-sync is
  // kept for the ladder but is not what anything is measured against.
  for (dsb::Variant v : {dsb::Variant::GlobalSync, dsb::Variant::Gemm}) {
    dsb::Options options = base;
    options.variant      = v;
    rows.push_back(measure(coupling, options, bandwidth));
  }
  {
    // gemm with FP32 storage: TF32 on and off, to show the tensor-core step.
    dsb::Options options = base;
    options.variant      = dsb::Variant::Gemm;
    options.precision    = dsb::Precision::FP32;
    options.tf32         = false;
    Row r = measure(coupling, options, bandwidth);
    r.variant += "/fp32";
    rows.push_back(r);
    options.tf32 = true;
    r = measure(coupling, options, bandwidth);
    r.variant += "/tf32";
    rows.push_back(r);
  }
  if (shape == "pm1") {
    dsb::Options options = base;
    options.variant      = dsb::Variant::Gemm;
    options.precision    = dsb::Precision::INT8;
    Row r = measure(coupling, options, bandwidth);
    r.variant += "/int8";
    rows.push_back(r);
    options           = base;
    options.variant   = dsb::Variant::Bit;
    rows.push_back(measure(coupling, options, bandwidth));
  }
  for (dsb::Variant v : {dsb::Variant::Block, dsb::Variant::Cluster}) {
    for (bool l2 : {false, true}) {
      dsb::Options options = base;
      options.variant      = v;
      options.l2_persist   = l2;
      if (v == dsb::Variant::Cluster) options.cluster = 2;
      rows.push_back(measure(coupling, options, bandwidth));
    }
  }

  // Speedups are quoted against gemm (FP16 storage, tensor cores).
  double baseline = 0.0;
  for (const Row& row : rows)
    if (row.ok && row.variant == "gemm") baseline = row.ms_step;
  if (baseline == 0.0)
    for (const Row& row : rows)
      if (row.ok && row.variant == "global-sync") baseline = row.ms_step;
  for (const Row& row : rows) print(row, baseline);

  std::cout << "\n"
            << "speedup is relative to gemm (fp16 storage, HMMA).\n"
            << "J fetches/step near 1 means J was read once per step and shared "
               "across replicas; bit reads J once per SOLVE, so its figure is "
               "< 1 by construction.\n"
            << "gemm gets that by construction; block gets it only if L2 "
               "happens to line the replicas up.\n"
            << "If block's figure is large and gemm is near 1, the GEMM "
               "formulation is worth its launch overhead.\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << "\n";
  return 1;
}
