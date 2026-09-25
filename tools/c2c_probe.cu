// tools/c2c_probe.cu
//
// Measures, on a J allocated exactly like DenseCapacitySolver::allocate_grace
// (anonymous mmap + cudaHostRegister under the caller's numactl policy):
//
//   stream  : a plain coalesced 16-byte read of the whole buffer (C2C ceiling)
//   copy    : the staged path's copies alone (cudaMemcpyAsync Grace -> HBM)
//   staged  : copy each block to an HBM staging buffer, then cublasGemmEx on
//             it -- what GemmStepper does with set_grace_staging
//   chunked : cublasGemmEx reading Grace in place, 65536-row blocks
//   full    : cublasGemmEx reading Grace in place, one call (old behaviour)
//
// J is not generated: registered anonymous pages are zero-filled and neither
// the copies nor the tensor-core GEMM speed depend on the values.
//
//   nvcc -O3 -arch=sm_90 -o c2c_probe tools/c2c_probe.cu -lcublas
//   CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
//       ./c2c_probe N fp16|fp32 [batch=1] [modes=stream,copy,staged] [reps=2]
//
// `full` and `chunked` can take 20-80 s per call at N=400k; they are off by
// default.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,     \
                   cudaGetErrorString(e_));                                \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)
#define CB(x)                                                              \
  do {                                                                     \
    cublasStatus_t s_ = (x);                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "%s:%d %s -> cublas %d\n", __FILE__, __LINE__,  \
                   #x, int(s_));                                           \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

__global__ void stream_read(const uint4* __restrict__ p, std::size_t count,
                            unsigned* out) {
  unsigned acc = 0;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *out = acc;  // keeps the loads alive
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s N fp16|fp32 [batch] [modes] [reps]\n"
                 "  modes: comma list of stream,copy,staged,chunked,full\n",
                 argv[0]);
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const bool fp16 = std::strcmp(argv[2], "fp16") == 0;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 1;
  const std::string modes = argc > 4 ? argv[4] : "stream,copy,staged";
  const int reps = argc > 5 ? std::atoi(argv[5]) : 2;
  auto want = [&](const char* m) {
    return ("," + modes + ",").find(std::string(",") + m + ",") !=
           std::string::npos;
  };

  const std::size_t elem = fp16 ? 2 : 4;
  const std::size_t ldj = std::size_t((n + 7) & ~7);
  const std::size_t row_bytes = ldj * elem;
  const std::size_t bytes = std::size_t(n) * row_bytes;
  const std::size_t stage_bytes = std::size_t(2) << 30;  // = kGraceStageBytes
  std::size_t stage_rows = std::max<std::size_t>(1, stage_bytes / row_bytes);
  if (stage_rows >= 64) stage_rows -= stage_rows % 64;
  stage_rows = std::min<std::size_t>(stage_rows, std::size_t(n));
  const int chunk = 65536;
  std::printf("N=%d %s batch=%d J=%.2f GB stage_rows=%zu modes=%s\n", n,
              fp16 ? "fp16" : "fp32", batch, bytes / 1e9, stage_rows,
              modes.c_str());

  CK(cudaFree(nullptr));

  auto t0 = std::chrono::steady_clock::now();
  void* j = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (j == MAP_FAILED) { std::perror("mmap"); return 1; }
  CK(cudaHostRegister(j, bytes, cudaHostRegisterDefault));
  auto t1 = std::chrono::steady_clock::now();
  std::printf("  register           : %8.2f s\n",
              std::chrono::duration<double>(t1 - t0).count());
  const char* jb = static_cast<const char*>(j);

  int device = 0, sms = 0;
  CK(cudaGetDevice(&device));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

  cudaStream_t stream;
  cudaEvent_t e0, e1;
  CK(cudaStreamCreate(&stream));
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // Times `body` reps times (after one untimed warm-up when `warm`).
  auto measure = [&](const char* name, bool warm, auto body) {
    for (int r = warm ? -1 : 0; r < reps; ++r) {
      CK(cudaEventRecord(e0, stream));
      body();
      CK(cudaEventRecord(e1, stream));
      CK(cudaEventSynchronize(e1));
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      if (r >= 0)
        std::printf("  %-9s [%d]      : %8.3f s  %7.1f GB/s\n", name, r,
                    ms * 1e-3, bytes / (ms * 1e-3) / 1e9);
      std::fflush(stdout);
    }
  };

  unsigned* d_out = nullptr;
  CK(cudaMalloc(&d_out, sizeof(unsigned)));
  const std::size_t count = std::size_t(n) * batch;
  void* d_sign = nullptr;
  float* d_acc = nullptr;
  void* d_ws = nullptr;
  void* d_stage = nullptr;
  const std::size_t ws_bytes = std::size_t(4) << 20;  // same as GemmStepper
  CK(cudaMalloc(&d_sign, count * elem));
  CK(cudaMemset(d_sign, 0, count * elem));
  CK(cudaMalloc(&d_acc, count * sizeof(float)));
  CK(cudaMalloc(&d_ws, ws_bytes));
  CK(cudaMalloc(&d_stage, stage_rows * row_bytes));

  cublasHandle_t h;
  CB(cublasCreate(&h));
  CB(cublasSetStream(h, stream));
  CB(cublasSetWorkspace(h, d_ws, ws_bytes));
  const cudaDataType type = fp16 ? CUDA_R_16F : CUDA_R_32F;
  const cublasComputeType_t compute =
      fp16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
  const float alpha = 1.f, beta = 0.f;
  // acc rows [row0, row0 + rows) from J rows stored at `block`.
  auto gemm = [&](const void* block, std::size_t row0, std::size_t rows) {
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, batch, int(rows), n, &alpha,
                    d_sign, type, batch, block, type, int(ldj), &beta,
                    d_acc + row0 * batch, CUDA_R_32F, batch, compute,
                    CUBLAS_GEMM_DEFAULT));
  };
  auto staged = [&](bool with_gemm) {
    for (std::size_t r = 0; r < std::size_t(n); r += stage_rows) {
      const std::size_t c = std::min(stage_rows, std::size_t(n) - r);
      CK(cudaMemcpyAsync(d_stage, jb + r * row_bytes, c * row_bytes,
                         cudaMemcpyDefault, stream));
      if (with_gemm) gemm(d_stage, r, c);
    }
  };

  if (want("stream"))
    measure("stream", false, [&] {
      stream_read<<<sms * 32, 256, 0, stream>>>(
          reinterpret_cast<const uint4*>(jb), bytes / 16, d_out);
      CK(cudaGetLastError());
    });
  if (want("copy")) measure("copy", false, [&] { staged(false); });
  if (want("staged")) measure("staged", true, [&] { staged(true); });
  if (want("chunked"))
    measure("chunked", true, [&] {
      for (std::size_t r = 0; r < std::size_t(n); r += chunk)
        gemm(jb + r * row_bytes, r, std::min<std::size_t>(chunk, n - r));
    });
  if (want("full")) measure("full", true, [&] { gemm(jb, 0, n); });

  cublasDestroy(h);
  cudaFree(d_stage);
  cudaFree(d_ws);
  cudaFree(d_acc);
  cudaFree(d_sign);
  cudaFree(d_out);
  cudaHostUnregister(j);
  munmap(j, bytes);
  return 0;
}
