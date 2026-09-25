// Host-side CSR matrix used by the sparse dSB path.
#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace dsb {

struct SparseMatrix {
  int n = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;

  int size() const { return n; }
  std::size_t nnz() const { return values.size(); }
  bool empty() const { return n == 0; }
  std::size_t bytes() const {
    return row_ptr.size() * sizeof(int) + col_idx.size() * sizeof(int) +
           values.size() * sizeof(float);
  }

  void validate() const;
  SparseMatrix scaled(float factor) const;
  Eigen::MatrixXd dense() const;
};

/// Build a square CSR matrix. Duplicate entries are sorted and summed.
SparseMatrix make_sparse_matrix(int n, const std::vector<int>& rows,
                                const std::vector<int>& columns,
                                const std::vector<float>& values);

SparseMatrix sparse_from_dense(const Eigen::MatrixXd& matrix);

}  // namespace dsb
