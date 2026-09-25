// src/qplib.cpp
#include "dsb/qplib.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace dsb {
namespace {

struct Edge {
  int    a = 0;
  int    b = 0;
  double w = 0.0;
};

std::string stem(const std::string& path) {
  std::size_t slash = path.find_last_of("/\\");
  std::string base  = (slash == std::string::npos) ? path : path.substr(slash + 1);
  std::size_t dot   = base.find_last_of('.');
  return (dot == std::string::npos) ? base : base.substr(0, dot);
}

/// Key for an (a, b) pair exactly as the file spells it. The Python reference
/// stores quadratic terms in a dict keyed by "a,b", so a repeated (a, b) line
/// keeps only the last weight; reproduce that before accumulating.
long long pack(int a, int b) {
  return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
}

}  // namespace

QplibInstance load_qplib(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::vector<std::string> lines;
  std::string              line;
  while (std::getline(in, line)) lines.push_back(line);
  if (lines.size() < 8) throw std::runtime_error("file too short: " + path);

  // Line 3 is the declared number of binary variables. Vertices that carry
  // only a linear coefficient never appear in a quadratic term, so sizing the
  // problem from the edge list alone silently drops them -- and drops their
  // field entries with them, which is what makes the standard form's extra
  // field node come out with no edges at all.
  int declared_vars = 0;
  std::istringstream(lines.at(3)) >> declared_vars;

  int n_edges = 0;
  std::istringstream(lines.at(4)) >> n_edges;
  if (n_edges < 0) throw std::runtime_error("bad edge count in " + path);

  std::vector<Edge> edges;
  edges.reserve(std::size_t(n_edges));
  for (int i = 5; i < 5 + n_edges; ++i) {
    if (i >= int(lines.size())) throw std::runtime_error("truncated edges: " + path);
    Edge e;
    std::istringstream(lines.at(i)) >> e.a >> e.b >> e.w;
    edges.push_back(e);
  }

  double default_field = 0.0;
  std::istringstream(lines.at(5 + n_edges)) >> default_field;

  int n_fields = 0;
  std::istringstream(lines.at(6 + n_edges)) >> n_fields;
  if (n_fields < 0) throw std::runtime_error("bad field count in " + path);

  std::unordered_map<int, double> field_entries;
  for (int i = 7 + n_edges; i < 7 + n_edges + n_fields; ++i) {
    if (i >= int(lines.size())) throw std::runtime_error("truncated fields: " + path);
    int    idx = 0;
    double val = 0.0;
    std::istringstream(lines.at(i)) >> idx >> val;
    field_entries[idx - 1] = val;  // QPLIB indices are 1-based
  }

  int n = declared_vars;
  for (const Edge& e : edges) n = std::max(n, std::max(e.a, e.b));
  for (const auto& kv : field_entries) n = std::max(n, kv.first + 1);
  if (n <= 0) throw std::runtime_error("no vertices parsed from " + path);

  // Deduplicate on (a, b) as written, then accumulate into the symmetric
  // matrix. Accumulation matters for diagonal terms: QPLIB encodes a linear
  // coefficient as an (i, i) quadratic entry, and the reference pipeline's
  // sparse assembly sums the two mirrored contributions into 2w there. Plain
  // assignment kept only w and halved that vertex's row sum -- which feeds
  // straight into the field below.
  std::unordered_map<long long, double> quadratic;
  quadratic.reserve(edges.size() * 2);
  for (const Edge& e : edges) quadratic[pack(e.a, e.b)] = e.w / 2.0;

  Eigen::MatrixXd j = Eigen::MatrixXd::Zero(n, n);
  for (const auto& kv : quadratic) {
    const int a = static_cast<int>(kv.first >> 32) - 1;
    const int b = static_cast<int>(static_cast<unsigned int>(kv.first)) - 1;
    if (a < 0 || a >= n || b < 0 || b >= n) continue;
    j(a, b) += kv.second;
    j(b, a) += kv.second;
  }

  // The linear coefficients. QPLIB states a default that applies to every
  // variable and then lists only the ones that differ from it, so the default
  // has to be laid down whether or not the exception list is empty. Gating it
  // on a non-empty list zeroed the whole field for any instance that gives a
  // uniform non-zero default -- and a zero field is exactly what leaves the
  // reducer with nothing to merge.
  Eigen::VectorXd h = Eigen::VectorXd::Constant(n, default_field);
  for (const auto& kv : field_entries)
    if (kv.first >= 0 && kv.first < n) h(kv.first) = kv.second;

  // QUBO -> Ising, matching the original pipeline exactly.
  j *= 1.0 / 8.0;
  const double offset = -j.sum() - h.sum() / 2.0;
  h                   = h * 0.5 + 2.0 * j.rowwise().sum();
  j *= -2.0;
  h = -h;

  double scale = std::max(j.cwiseAbs().maxCoeff(), h.cwiseAbs().maxCoeff());
  if (scale == 0.0) scale = 1.0;

  QplibInstance out;
  out.name            = stem(path);
  out.coupling        = j;
  out.field           = h;
  out.offset          = offset;
  out.scaled_coupling = -j / scale;
  out.scaled_field    = -h / scale;

  // A field that is identically zero leaves the standard form's field node
  // isolated, and FastHare then has only low-degree vertices to work with.
  // Say so once rather than letting it look like the reducer underperformed.
  if (out.scaled_field.cwiseAbs().maxCoeff() == 0.0)
    std::cerr << "note: " << out.name
              << " has an identically zero external field; FastHare will see a "
                 "field-free instance and reduce little\n";
  return out;
}

double objective_value(const QplibInstance& instance,
                       const Eigen::VectorXd& spins) {
  const Eigen::VectorXd js   = instance.coupling * spins;
  const double          quad = js.dot(spins) / 2.0;
  const double          lin  = instance.field.dot(spins);
  return -(quad + lin) - instance.offset;
}

}  // namespace dsb
