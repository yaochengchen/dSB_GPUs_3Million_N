// src/gset.cpp

// Stanford G-set parser and Max-Cut objective evaluator.

#include "dsb/gset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace dsb {
namespace {

std::string stem(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::string base =
      (slash == std::string::npos) ? path : path.substr(slash + 1);
  const std::size_t dot = base.find_last_of('.');
  return (dot == std::string::npos) ? base : base.substr(0, dot);
}

}  // namespace

GsetInstance load_gset(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  long long n64 = 0;
  long long m64 = 0;
  if (!(in >> n64 >> m64))
    throw std::runtime_error("invalid G-set header in " + path);
  if (n64 <= 0 || n64 > std::numeric_limits<int>::max())
    throw std::runtime_error("invalid vertex count in " + path);
  if (m64 < 0 || m64 > std::numeric_limits<int>::max())
    throw std::runtime_error("invalid edge count in " + path);

  const int n = int(n64);
  const int m = int(m64);

  GsetInstance out;
  out.name = stem(path);
  out.n = n;
  out.edges.reserve(std::size_t(m));
  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<float> values;
  rows.reserve(std::size_t(2) * m);
  columns.reserve(std::size_t(2) * m);
  values.reserve(std::size_t(2) * m);

  double max_abs_weight = 0.0;
  for (int edge_index = 0; edge_index < m; ++edge_index) {
    long long u64 = 0;
    long long v64 = 0;
    double    weight = 0.0;
    if (!(in >> u64 >> v64 >> weight))
      throw std::runtime_error("truncated edge list in " + path +
                               " at edge " + std::to_string(edge_index + 1));
    if (u64 < 1 || u64 > n || v64 < 1 || v64 > n)
      throw std::runtime_error("vertex index out of range in " + path +
                               " at edge " + std::to_string(edge_index + 1));
    if (u64 == v64)
      throw std::runtime_error("self-loop in " + path + " at edge " +
                               std::to_string(edge_index + 1));
    if (!std::isfinite(weight))
      throw std::runtime_error("non-finite edge weight in " + path +
                               " at edge " + std::to_string(edge_index + 1));

    const int u = int(u64 - 1);
    const int v = int(v64 - 1);
    out.edges.push_back({u, v, weight});
    rows.push_back(u);
    columns.push_back(v);
    values.push_back(float(weight));
    rows.push_back(v);
    columns.push_back(u);
    values.push_back(float(weight));
    max_abs_weight = std::max(max_abs_weight, std::abs(weight));
  }

  std::string extra;
  if (in >> extra)
    throw std::runtime_error("extra data after declared edges in " + path);

  if (max_abs_weight == 0.0) max_abs_weight = 1.0;
  const float inverse_scale = float(1.0 / max_abs_weight);
  for (float& value : values) value *= inverse_scale;
  out.coupling = make_sparse_matrix(n, rows, columns, values);
  return out;
}

double cut_value(const GsetInstance& instance,
                 const Eigen::VectorXd& spins) {
  if (spins.size() != instance.size())
    throw std::runtime_error("cut_value: spin vector size mismatch");

  double cut = 0.0;
  for (const GsetEdge& edge : instance.edges) {
    const bool side_u = spins(edge.u) < 0.0;
    const bool side_v = spins(edge.v) < 0.0;
    if (side_u != side_v) cut += edge.weight;
  }
  return cut;
}

}  // namespace dsb
