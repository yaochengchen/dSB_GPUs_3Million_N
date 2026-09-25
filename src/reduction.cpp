// src/reduction.cpp
#include "dsb/reduction.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "fasthare.h"

namespace dsb {
namespace {

using SparseIsing = FastHare::ski;  // vector<tuple<int,int,double>>

FastHare::fasthare_output run_fasthare(const SparseIsing& standard,
                                       double alpha) {
  // Run the original FastHare implementation directly. Matrix storage is
  // handled after reduction, so the same residual can be rebuilt as CSR
  // without changing either FastHare or the v2 sparse dSB kernels.
  FastHare reducer(standard, alpha);
  reducer.run();
  return reducer.get_output();
}

/// Upper-triangular standard form. The external field is folded in as an extra
/// node with index n, so that the reducer sees a field-free problem.
SparseIsing to_standard_form(const Eigen::MatrixXd& coupling,
                             const Eigen::VectorXd& field) {
  const int n = int(field.size());
  if (coupling.rows() != n || coupling.cols() != n)
    throw std::runtime_error("reduce_ising: coupling/field size mismatch");

  SparseIsing out;
  out.reserve(std::size_t(n) * std::size_t(n - 1) / 2 + std::size_t(n));

  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double v = coupling(i, j);
      if (v != 0.0) out.emplace_back(i, j, -v);
    }
  for (int i = 0; i < n; ++i) {
    const double f = field(i);
    if (f != 0.0) out.emplace_back(i, n, -f);
  }
  return out;
}

SparseIsing to_standard_form(const SparseMatrix& coupling,
                             const Eigen::VectorXd& field) {
  coupling.validate();
  const int n = coupling.n;
  if (field.size() != n)
    throw std::runtime_error("reduce_ising: sparse coupling/field size mismatch");
  SparseIsing out;
  out.reserve(coupling.nnz() / 2 + std::size_t(n));
  for (int row = 0; row < n; ++row)
    for (int p = coupling.row_ptr[std::size_t(row)];
         p < coupling.row_ptr[std::size_t(row + 1)]; ++p) {
      const int column = coupling.col_idx[std::size_t(p)];
      if (row < column)
        out.emplace_back(row, column, -coupling.values[std::size_t(p)]);
    }
  for (int i = 0; i < n; ++i)
    if (field(i) != 0.0) out.emplace_back(i, n, -field(i));
  return out;
}

}  // namespace

Reduction reduce_ising(const Eigen::MatrixXd& coupling,
                       const Eigen::VectorXd& field, double alpha) {
  SparseIsing standard = to_standard_form(coupling, field);

  auto result = run_fasthare(standard, alpha);

  const auto& residual = std::get<0>(result);
  const auto& spin_map = std::get<1>(result);
  const auto& sign     = std::get<2>(result);

  Reduction out;
  out.sign          = sign;
  out.seconds       = std::get<3>(result);
  out.fully_reduced = residual.empty();
  if (out.fully_reduced) return out;

  out.spin_map = spin_map;

  int m = 0;
  for (int s : out.spin_map) m = std::max(m, s);
  ++m;

  Eigen::MatrixXd upper = Eigen::MatrixXd::Zero(m, m);
  for (const auto& e : residual) {
    int    u, v;
    double w;
    std::tie(u, v, w) = e;
    if (0 <= u && u < m && 0 <= v && v < m) upper(u, v) = -w;
  }
  out.coupling = upper + upper.transpose();
  return out;
}

SparseReduction reduce_ising(const SparseMatrix& coupling,
                             const Eigen::VectorXd& field, double alpha) {
  SparseIsing standard = to_standard_form(coupling, field);
  auto result = run_fasthare(standard, alpha);

  const auto& residual = std::get<0>(result);
  SparseReduction out;
  out.spin_map = std::get<1>(result);
  out.sign = std::get<2>(result);
  out.seconds = std::get<3>(result);
  out.fully_reduced = residual.empty();
  if (out.fully_reduced) return out;

  int m = 0;
  for (int mapped : out.spin_map) m = std::max(m, mapped);
  ++m;
  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<float> values;
  rows.reserve(2 * residual.size());
  columns.reserve(2 * residual.size());
  values.reserve(2 * residual.size());
  for (const auto& edge : residual) {
    int u, v;
    double weight;
    std::tie(u, v, weight) = edge;
    if (0 <= u && u < m && 0 <= v && v < m) {
      rows.push_back(u);
      columns.push_back(v);
      values.push_back(float(-weight));
      rows.push_back(v);
      columns.push_back(u);
      values.push_back(float(-weight));
    }
  }
  out.coupling = make_sparse_matrix(m, rows, columns, values);
  return out;
}

}  // namespace dsb
