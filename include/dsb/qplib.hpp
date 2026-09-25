// include/dsb/qplib.hpp
//
// Reader for the QPLIB binary-quadratic instances (http://qplib.zib.de).
// The transformation to Ising form is the one the original Python pipeline
// used; it is reproduced here unchanged so that results stay comparable.

#pragma once

#include <Eigen/Dense>
#include <string>

namespace dsb {

struct QplibInstance {
  std::string name;

  /// Ising form used for the objective value.
  Eigen::MatrixXd coupling;  // J
  Eigen::VectorXd field;     // h
  double          offset = 0.0;

  /// Same instance rescaled to max|.| == 1, which is what gets handed to the
  /// reducer and the solver.
  Eigen::MatrixXd scaled_coupling;
  Eigen::VectorXd scaled_field;

  int size() const { return int(field.size()); }
};

QplibInstance load_qplib(const std::string& path);

/// Objective value of a +-1 assignment, in the original QPLIB units.
double objective_value(const QplibInstance& instance,
                       const Eigen::VectorXd& spins);

}  // namespace dsb
