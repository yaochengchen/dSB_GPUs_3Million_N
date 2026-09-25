// include/dsb/common.hpp
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace dsb {

// Sign convention, everywhere in this code base (kernels, CPU reference,
// energies, reported spins):
//
//     sign(x) = +1 if x > 0, else -1.
//
// Every Ising spin is +-1; x == 0 exactly is a tie and is resolved to -1.  The
// earlier convention sign(0) = 0 gave a "spin" that contributed nothing to its
// neighbours, could not be represented by the 1-bit and int8 paths, and
// produced a divergence between them and the FP32 kernels whenever a clipped
// state was restarted at pump 0 (x_new = xi * acc lands on 0.0 exactly when
// the integer acc is 0).  Nothing else in a normal run can produce x == 0.

/// Storage precision for the coupling matrix.  State (x, y) is FP16 for FP16
/// and FP32 otherwise.  Compute is FP32 in every kernel; INT8 is exact
/// integer arithmetic for couplings in {-1, 0, +1} and is Gemm-only.
enum class Precision { FP16, FP32, INT8 };

inline const char* to_string(Precision p) {
  switch (p) {
    case Precision::FP16: return "fp16";
    case Precision::INT8: return "int8";
    default:              return "fp32";
  }
}

/// Bytes per element of J.
inline std::size_t coupling_element_size(Precision p) {
  return p == Precision::FP16 ? 2 : (p == Precision::INT8 ? 1 : 4);
}

/// Bytes per element of x and y.
inline std::size_t state_element_size(Precision p) {
  return p == Precision::FP16 ? 2 : 4;
}

/// Row stride of J, in elements. Rounded up to a multiple of 8 so that every
/// row start is 16-byte aligned, which is what the vectorised inner loop needs.
inline std::size_t row_stride(int n) {
  return static_cast<std::size_t>((n + 7) & ~7);
}

}  // namespace dsb

#define DSB_CUDA_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                    \
                               cudaGetErrorString(err_) + "  at " + __FILE__ +  \
                               ":" + std::to_string(__LINE__));                 \
    }                                                                           \
  } while (0)
