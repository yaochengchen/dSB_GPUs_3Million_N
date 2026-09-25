// Export one deterministic FastHare reduction for use by the public Python
// baseline. The GPU solver still runs the same in-tree reducer itself; this
// file makes the reduced Hamiltonian and lift map portable without adding a
// solver variant or a Python binding.

#include <Eigen/Dense>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dsb/gset.hpp"
#include "dsb/qplib.hpp"
#include "dsb/reduction.hpp"

namespace {

struct Config {
  std::string format;
  std::string input;
  std::string output;
  double alpha = 0.2;
};

bool starts_with(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

void usage(const char* program) {
  std::cout << "usage: " << program
            << " --format=qplib|gset --output=FILE [--alpha=0.2] INSTANCE\n";
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (starts_with(argument, "--format=")) {
      config.format = argument.substr(9);
    } else if (starts_with(argument, "--output=")) {
      config.output = argument.substr(9);
    } else if (starts_with(argument, "--alpha=")) {
      config.alpha = std::stod(argument.substr(8));
    } else if (starts_with(argument, "-")) {
      throw std::runtime_error("unknown option: " + argument);
    } else if (config.input.empty()) {
      config.input = argument;
    } else {
      throw std::runtime_error("only one input instance is accepted");
    }
  }
  if ((config.format != "qplib" && config.format != "gset") ||
      config.output.empty() || config.input.empty()) {
    usage(argv[0]);
    throw std::runtime_error("format, output and input are required");
  }
  return config;
}

template <class Reduction>
void write_prefix(std::ostream& output, const std::string& kind,
                  const std::string& name, int n_original, int n_standard,
                  double alpha, const Reduction& reduction) {
  output << "DSB_FASTHARE_V1\n"
         << "kind " << kind << '\n'
         << "name " << name << '\n'
         << "n_original " << n_original << '\n'
         << "n_standard " << n_standard << '\n'
         << "n_reduced " << reduction.reduced_size() << '\n'
         << "fully_reduced " << (reduction.fully_reduced ? 1 : 0) << '\n'
         << "alpha " << std::setprecision(17) << alpha << '\n'
         << "preprocess_s " << std::setprecision(17) << reduction.seconds
         << '\n'
         << "sign_count " << reduction.sign.size() << '\n'
         << "sign";
  for (int value : reduction.sign) output << ' ' << value;
  output << "\nmap_count " << reduction.spin_map.size() << "\nmap";
  for (int value : reduction.spin_map) output << ' ' << value;
  output << '\n';
}

void write_dense(std::ostream& output, const dsb::Reduction& reduction) {
  std::size_t edge_count = 0;
  for (int row = 0; row < reduction.coupling.rows(); ++row)
    for (int column = row + 1; column < reduction.coupling.cols(); ++column)
      if (reduction.coupling(row, column) != 0.0) ++edge_count;
  output << "edge_count " << edge_count << '\n';
  output << std::setprecision(17);
  for (int row = 0; row < reduction.coupling.rows(); ++row)
    for (int column = row + 1; column < reduction.coupling.cols(); ++column) {
      const double value = reduction.coupling(row, column);
      if (value != 0.0)
        output << "edge " << row << ' ' << column << ' ' << value << '\n';
    }
}

void write_sparse(std::ostream& output,
                  const dsb::SparseReduction& reduction) {
  std::size_t edge_count = 0;
  for (int row = 0; row < reduction.coupling.n; ++row)
    for (int position = reduction.coupling.row_ptr[std::size_t(row)];
         position < reduction.coupling.row_ptr[std::size_t(row + 1)];
         ++position)
      if (row < reduction.coupling.col_idx[std::size_t(position)]) ++edge_count;
  output << "edge_count " << edge_count << '\n';
  output << std::setprecision(17);
  for (int row = 0; row < reduction.coupling.n; ++row)
    for (int position = reduction.coupling.row_ptr[std::size_t(row)];
         position < reduction.coupling.row_ptr[std::size_t(row + 1)];
         ++position) {
      const int column = reduction.coupling.col_idx[std::size_t(position)];
      if (row < column)
        output << "edge " << row << ' ' << column << ' '
               << reduction.coupling.values[std::size_t(position)] << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) try {
  const Config config = parse_args(argc, argv);
  std::ofstream output(config.output);
  if (!output) throw std::runtime_error("cannot create " + config.output);

  if (config.format == "qplib") {
    const dsb::QplibInstance instance = dsb::load_qplib(config.input);
    const dsb::Reduction reduction = dsb::reduce_ising(
        -instance.scaled_coupling, -instance.scaled_field, config.alpha);
    write_prefix(output, "qplib", instance.name, instance.size(),
                 instance.size() + 1, config.alpha, reduction);
    write_dense(output, reduction);
  } else {
    const dsb::GsetInstance instance = dsb::load_gset(config.input);
    const dsb::SparseReduction reduction = dsb::reduce_ising(
        instance.coupling, Eigen::VectorXd::Zero(instance.size()),
        config.alpha);
    write_prefix(output, "gset", instance.name, instance.size(),
                 instance.size(), config.alpha, reduction);
    write_sparse(output, reduction);
  }
  if (!output) throw std::runtime_error("failed while writing " + config.output);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
