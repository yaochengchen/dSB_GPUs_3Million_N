// include/dsb/reduction.hpp
//
// Thin wrapper around FastHare (third_party/fasthare), which merges spins that
// are provably aligned in any ground state and hands back a smaller instance.
//
//   Nguyen et al., "FastHare: Fast Hamiltonian Reduction for Large-scale
//   Quantum Annealing", IEEE QCE 2023.

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "dsb/sparse.hpp"

namespace dsb {

struct Reduction {
  /// True when FastHare collapsed the whole instance; `coupling` is then empty
  /// and the solution is read straight out of `sign`.
  bool fully_reduced = false;

  /// Per node of the standard form: the sign relating it to its representative.
  /// The standard form carries one extra node at index n that encodes the
  /// external field, so sign.size() is n or n+1.
  std::vector<int> sign;

  /// Standard-form node -> reduced-instance node.
  std::vector<int> spin_map;

  /// Reduced dense coupling matrix, symmetric.
  Eigen::MatrixXd coupling;

  double seconds = 0.0;

  int reduced_size() const { return int(coupling.rows()); }
};

/// Reduce the Ising instance (`coupling`, `field`) with FastHare.
/// `alpha` scales how many merge candidates FastHare tries: in
/// `slow_find_NS` it sets `n_tries = min(int(n_tries * alpha), sc.size())`,
/// so LARGER alpha attempts more merges. Values >= 1.0 try every candidate.
Reduction reduce_ising(const Eigen::MatrixXd& coupling,
                       const Eigen::VectorXd& field, double alpha);

struct SparseReduction {
  bool fully_reduced = false;
  std::vector<int> sign;
  std::vector<int> spin_map;
  SparseMatrix coupling;
  double seconds = 0.0;
  int reduced_size() const { return coupling.n; }
};

/// CSR-native FastHare wrapper. The residual is kept sparse.
SparseReduction reduce_ising(const SparseMatrix& coupling,
                             const Eigen::VectorXd& field, double alpha);

}  // namespace dsb
