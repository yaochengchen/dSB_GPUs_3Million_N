// include/dsb/gset.hpp
//
// Reader and objective evaluator for the Stanford G-set Max-Cut format:
//
//   number_of_vertices number_of_edges
//   vertex_u vertex_v weight
//   ...

// Vertex indices in the files are one-based. The normalized adjacency is kept
// in CSR so loading a large sparse graph never creates an N x N allocation.

#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "dsb/sparse.hpp"

namespace dsb {

struct GsetEdge {
  int    u = 0;  // zero-based
  int    v = 0;  // zero-based
  double weight = 0.0;
};

struct GsetInstance {
  std::string name;
  int n = 0;
  std::vector<GsetEdge> edges;

  /// Symmetric weighted adjacency matrix divided by max |edge weight|.
  /// Original weights remain in `edges` for exact objective evaluation.  G-set
  /// normally uses only +/-1, but normalizing here also supports other weights.
  SparseMatrix coupling;

  int size() const { return n; }
  int edge_count() const { return int(edges.size()); }
};

GsetInstance load_gset(const std::string& path);

/// Weighted cut value of a spin assignment.  Negative spins are one side of
/// the partition and zero/positive spins the other, so the result is always a
/// valid cut even in the unlikely event that dSB returns an exact zero.
double cut_value(const GsetInstance& instance, const Eigen::VectorXd& spins);

}  // namespace dsb
