// Capacity-oriented scaling for dense high-entropy Ising matrices on GH200.

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "dsb/dense_capacity.hpp"

namespace {

struct Config {
  int n = 20000;
  int repeats = 3;
  int warmup_steps = 1;
  double hbm_fraction = 0.85;
  bool csv = false;
  dsb::MatrixMemory memory = dsb::MatrixMemory::Auto;
  dsb::Options solver;
  Config() {
    solver.batch = 1;
    solver.n_steps = 50;
    solver.precision = dsb::Precision::FP32;
    solver.variant = dsb::Variant::Auto;
    solver.seed = 42;
  }
};

bool starts_with(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

dsb::Variant parse_variant(const std::string& value) {
  if (value == "auto") return dsb::Variant::Auto;
  if (value == "block") return dsb::Variant::Block;
  if (value == "cluster") return dsb::Variant::Cluster;
  if (value == "global-sync") return dsb::Variant::GlobalSync;
  if (value == "gemm") return dsb::Variant::Gemm;
  throw std::runtime_error("unknown dense variant: " + value);
}

void usage(const char* program) {
  std::cout
      << "usage: " << program << " [options]\n\n"
      << "  --n=N                 dense problem size (default 20000)\n"
      << "  --batch=N             replicas (default 1)\n"
      << "  --steps=N             measured steps (default 50)\n"
      << "  --repeats=N           measured repeats (default 3)\n"
      << "  --warmup-steps=N      warm-up steps after generation (default 1)\n"
      << "  --seed=N              deterministic matrix/state seed (default 42)\n"
      << "  --variant=V           auto | block | cluster | global-sync | gemm\n"
      << "  --precision=P         fp32 (default) | fp16 storage for J and state\n"
      << "  --no-tf32             gemm+fp32: stay on CUDA cores (no TF32)\n"
      << "  --cluster=N           requested cluster size: 2,4,8,16\n"
      << "  --matrix-memory=M     auto | hbm | grace\n"
      << "  --hbm-fraction=F      maximum fraction of currently free HBM (0,1]\n"
      << "  --csv                 machine-readable output\n";
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (starts_with(argument, "--n=")) {
      config.n = std::stoi(argument.substr(4));
    } else if (starts_with(argument, "--batch=")) {
      config.solver.batch = std::stoi(argument.substr(8));
    } else if (starts_with(argument, "--steps=")) {
      config.solver.n_steps = std::stoi(argument.substr(8));
    } else if (starts_with(argument, "--repeats=")) {
      config.repeats = std::stoi(argument.substr(10));
    } else if (starts_with(argument, "--warmup-steps=")) {
      config.warmup_steps = std::stoi(argument.substr(15));
    } else if (starts_with(argument, "--seed=")) {
      config.solver.seed = uint32_t(std::stoul(argument.substr(7)));
    } else if (starts_with(argument, "--variant=")) {
      config.solver.variant = parse_variant(argument.substr(10));
    } else if (starts_with(argument, "--precision=")) {
      const std::string p = argument.substr(12);
      if (p == "fp16") config.solver.precision = dsb::Precision::FP16;
      else if (p == "fp32") config.solver.precision = dsb::Precision::FP32;
      else throw std::runtime_error("unknown precision: " + p);
    } else if (argument == "--no-tf32") {
      config.solver.tf32 = false;
    } else if (starts_with(argument, "--cluster=")) {
      config.solver.cluster = std::stoi(argument.substr(10));
    } else if (starts_with(argument, "--matrix-memory=")) {
      config.memory = dsb::parse_matrix_memory(argument.substr(16));
    } else if (starts_with(argument, "--hbm-fraction=")) {
      config.hbm_fraction = std::stod(argument.substr(15));
    } else if (argument == "--csv") {
      config.csv = true;
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (config.n <= 0 || config.solver.batch <= 0 ||
      config.solver.n_steps <= 0 || config.repeats <= 0 ||
      config.warmup_steps < 0)
    throw std::runtime_error("N, batch, steps and repeats must be positive");
  return config;
}

std::string clean_error(std::string message) {
  std::replace(message.begin(), message.end(), ',', ';');
  std::replace(message.begin(), message.end(), '\n', ' ');
  return message;
}

std::string failure_status(const std::string& message) {
  if (message.find("HBM_OOM") != std::string::npos ||
      message.find("out of memory") != std::string::npos)
    return "oom";
  if (message.find("shared memory") != std::string::npos ||
      message.find("cluster") != std::string::npos)
    return "launch-limit";
  if (message.find("GRACE_UNAVAILABLE") != std::string::npos)
    return "grace-unavailable";
  return "error";
}

void print_header() {
  std::cout
      << "implementation,n,repeat,seed,batch,steps,precision,"
         "requested_variant,selected_variant,requested_memory,selected_memory,"
         "cluster,matrix_bytes,hbm_bytes,grace_bytes,hbm_free_before,hbm_total,"
         "generation_s,gpu_s,time_per_step_s,dense_interactions_per_s,"
         "effective_matrix_GB_s,status,error\n";
}

void print_failure(const Config& config, const std::string& message) {
  std::cout << "dsb-gpu-dense," << config.n << ",0," << config.solver.seed
            << ',' << config.solver.batch << ',' << config.solver.n_steps
            << ',' << dsb::to_string(config.solver.precision) << ','
            << dsb::to_string(config.solver.variant)
            << ",," << dsb::to_string(config.memory)
            << ",,0,0,0,0,0,0,0,0,0,0,0,"
            << failure_status(message) << ',' << clean_error(message) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  try {
    config = parse_args(argc, argv);
    if (config.csv) print_header();

    dsb::DenseCapacitySolver solver(config.n, config.solver, config.memory,
                                    config.hbm_fraction);
    if (config.warmup_steps > 0) {
      // GEMM graphs are keyed by the number of steps. Warm the exact graph
      // that measured runs reuse; shorter warmups would force a recapture.
      const int warmup_steps =
          solver.plan().variant == dsb::Variant::Gemm
              ? config.solver.n_steps
              : config.warmup_steps;
      solver.run(warmup_steps);
    }

    for (int repeat = 0; repeat < config.repeats; ++repeat) {
      solver.run(config.solver.n_steps);
      const dsb::DenseCapacityStats& stats = solver.stats();
      const double gpu_s = double(stats.run_ms) * 1e-3;
      const double time_per_step = gpu_s / config.solver.n_steps;
      const long double interactions =
          (long double)config.n * (config.n - 1) * config.solver.batch *
          config.solver.n_steps;
      const double interactions_per_s = double(interactions / gpu_s);
      const long double streamed =
          (long double)stats.matrix_bytes * config.solver.batch *
          config.solver.n_steps;
      const double effective_gb_s = double(streamed / gpu_s / 1.0e9L);

      if (config.csv) {
        std::cout << "dsb-gpu-dense," << config.n << ',' << repeat << ','
                  << config.solver.seed << ',' << config.solver.batch << ','
                  << config.solver.n_steps << ','
                  << dsb::to_string(config.solver.precision) << ','
                  << stats.requested_variant << ',' << stats.selected_variant
                  << ',' << stats.requested_memory << ','
                  << stats.selected_memory << ',' << stats.cluster_size << ','
                  << stats.matrix_bytes << ',' << stats.hbm_bytes << ','
                  << stats.grace_bytes << ',' << stats.hbm_free_before << ','
                  << stats.hbm_total << ',' << std::setprecision(9)
                  << stats.generation_s << ',' << gpu_s << ','
                  << time_per_step << ',' << interactions_per_s << ','
                  << effective_gb_s << ",completed,\n";
      } else {
        std::cout << "N=" << config.n << " repeat=" << repeat
                  << " variant=" << stats.selected_variant
                  << " cluster=" << stats.cluster_size
                  << " memory=" << stats.selected_memory
                  << " matrix=" << std::fixed << std::setprecision(2)
                  << double(stats.matrix_bytes) / 1.0e9 << " GB"
                  << " generation=" << stats.generation_s << " s"
                  << " gpu=" << gpu_s << " s"
                  << " step=" << time_per_step << " s\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    if (config.csv) {
      print_failure(config, error.what());
      return 0;
    }
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
