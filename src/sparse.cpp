#include "dsb/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace dsb {

void SparseMatrix::validate() const {
  if (n < 0 || row_ptr.size() != std::size_t(n + 1))
    throw std::runtime_error("SparseMatrix: invalid row_ptr size");
  if (col_idx.size() != values.size())
    throw std::runtime_error("SparseMatrix: index/value size mismatch");
  if (row_ptr.empty() || row_ptr.front() != 0 ||
      row_ptr.back() != int(values.size()))
    throw std::runtime_error("SparseMatrix: invalid row_ptr endpoints");
  for (int row = 0; row < n; ++row) {
    if (row_ptr[std::size_t(row)] > row_ptr[std::size_t(row + 1)])
      throw std::runtime_error("SparseMatrix: row_ptr is not monotonic");
    int previous = -1;
    for (int p = row_ptr[std::size_t(row)];
         p < row_ptr[std::size_t(row + 1)]; ++p) {
      const int column = col_idx[std::size_t(p)];
      if (column < 0 || column >= n || column <= previous)
        throw std::runtime_error("SparseMatrix: invalid or unsorted column");
      if (!std::isfinite(values[std::size_t(p)]))
        throw std::runtime_error("SparseMatrix: non-finite value");
      previous = column;
    }
  }
}

SparseMatrix SparseMatrix::scaled(float factor) const {
  SparseMatrix out = *this;
  for (float& value : out.values) value *= factor;
  return out;
}

Eigen::MatrixXd SparseMatrix::dense() const {
  validate();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, n);
  for (int row = 0; row < n; ++row)
    for (int p = row_ptr[std::size_t(row)];
         p < row_ptr[std::size_t(row + 1)]; ++p)
      out(row, col_idx[std::size_t(p)]) = values[std::size_t(p)];
  return out;
}

SparseMatrix make_sparse_matrix(int n, const std::vector<int>& rows,
                                const std::vector<int>& columns,
                                const std::vector<float>& values) {
  if (n <= 0 || rows.size() != columns.size() || rows.size() != values.size())
    throw std::runtime_error("make_sparse_matrix: invalid input sizes");

  using Entry = std::tuple<int, int, float>;
  std::vector<Entry> entries;
  entries.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (rows[i] < 0 || rows[i] >= n || columns[i] < 0 || columns[i] >= n)
      throw std::runtime_error("make_sparse_matrix: index out of range");
    if (!std::isfinite(values[i]))
      throw std::runtime_error("make_sparse_matrix: non-finite value");
    if (values[i] != 0.f) entries.emplace_back(rows[i], columns[i], values[i]);
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return std::tie(std::get<0>(a), std::get<1>(a)) <
           std::tie(std::get<0>(b), std::get<1>(b));
  });

  SparseMatrix out;
  out.n = n;
  out.row_ptr.assign(std::size_t(n + 1), 0);
  for (std::size_t i = 0; i < entries.size();) {
    const int row = std::get<0>(entries[i]);
    const int col = std::get<1>(entries[i]);
    float sum = 0.f;
    do {
      sum += std::get<2>(entries[i]);
      ++i;
    } while (i < entries.size() && std::get<0>(entries[i]) == row &&
             std::get<1>(entries[i]) == col);
    if (sum != 0.f) {
      out.col_idx.push_back(col);
      out.values.push_back(sum);
      ++out.row_ptr[std::size_t(row + 1)];
    }
  }
  for (int row = 0; row < n; ++row)
    out.row_ptr[std::size_t(row + 1)] += out.row_ptr[std::size_t(row)];
  out.validate();
  return out;
}

SparseMatrix sparse_from_dense(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols() || matrix.rows() <= 0)
    throw std::runtime_error("sparse_from_dense: matrix must be nonempty square");
  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<float> values;
  for (int row = 0; row < matrix.rows(); ++row)
    for (int col = 0; col < matrix.cols(); ++col)
      if (matrix(row, col) != 0.0) {
        rows.push_back(row);
        columns.push_back(col);
        values.push_back(float(matrix(row, col)));
      }
  return make_sparse_matrix(int(matrix.rows()), rows, columns, values);
}

}  // namespace dsb
