// tools/c2c_probe.cu
//
// Separates "Grace memory is slow for this allocation" from "cuBLAS picks a
// bad kernel for this shape".  Allocates J exactly like
// DenseCapacitySolver::allocate_grace (anonymous mmap + cudaHostRegister under
// the caller's numactl policy) and times, on the same buffer:
//
//   1. stream   : a plain coalesced 16-byte read of the whole buffer
//   2. gemm     : the exact cublasGemmEx that GemmStepper issues per step
//   3. chunked  : the same product split into row blocks (n dimension)
//
// J is not generated: cudaHostRegister'd anonymous pages are zero-filled, and
// neither the stream kernel nor the tensor-core GEMM speed depends on values.
// That skips the 1-3 min generation.
//
//   nvcc -O3 -arch=sm_90 -o c2c_probe tools/c2c_probe.cu -lcublas
//   CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
//       ./c2c_probe 400000 fp16 [batch=1] [chunk_rows=65536] [reps=2]

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

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
    std::fprintf(stderr, "usage: %s N fp16|fp32 [batch] [chunk_rows] [reps]\n",
                 argv[0]);
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const bool fp16 = std::strcmp(argv[2], "fp16") == 0;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 1;
  const int chunk = argc > 4 ? std::atoi(argv[4]) : 65536;
  const int reps = argc > 5 ? std::atoi(argv[5]) : 2;

  const std::size_t elem = fp16 ? 2 : 4;
  const std::size_t ldj = std::size_t((n + 7) & ~7);
  const std::size_t bytes = std::size_t(n) * ldj * elem;
  std::printf("N=%d %s batch=%d ldj=%zu J=%.2f GB chunk_rows=%d\n", n,
              fp16 ? "fp16" : "fp32", batch, ldj, bytes / 1e9, chunk);

  CK(cudaFree(nullptr));  // create the context outside the timings

  // ---- allocate exactly like allocate_grace --------------------------------
  auto t0 = std::chrono::steady_clock::now();
  void* j = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (j == MAP_FAILED) { std::perror("mmap"); return 1; }
  CK(cudaHostRegister(j, bytes, cudaHostRegisterDefault));
  auto t1 = std::chrono::steady_clock::now();
  std::printf("  register           : %8.2f s\n",
              std::chrono::duration<double>(t1 - t0).count());

  int device = 0, sms = 0;
  CK(cudaGetDevice(&device));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

  cudaStream_t stream;
  cudaEvent_t e0, e1;
  CK(cudaStreamCreate(&stream));
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto elapsed = [&]() {
    float ms = 0.f;
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return double(ms) * 1e-3;
  };

  // ---- 1. plain streaming read ---------------------------------------------
  unsigned* d_out = nullptr;
  CK(cudaMalloc(&d_out, sizeof(unsigned)));
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0, stream));
    stream_read<<<sms * 32, 256, 0, stream>>>(static_cast<const uint4*>(j),
                                             bytes / 16, d_out);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1, stream));
    const double s = elapsed();
    std::printf("  stream read   [%d] : %8.3f s  %7.1f GB/s\n", r, s,
                bytes / s / 1e9);
  }

  // ---- 2./3. the production GEMM, full and row-chunked ---------------------
  const std::size_t count = std::size_t(n) * batch;
  void* d_sign = nullptr;
  float* d_acc = nullptr;
  void* d_ws = nullptr;
  const std::size_t ws_bytes = std::size_t(4) << 20;  // same as GemmStepper
  CK(cudaMalloc(&d_sign, count * elem));
  CK(cudaMemset(d_sign, 0, count * elem));
  CK(cudaMalloc(&d_acc, count * sizeof(float)));
  CK(cudaMalloc(&d_ws, ws_bytes));

  cublasHandle_t h;
  CB(cublasCreate(&h));
  CB(cublasSetStream(h, stream));
  CB(cublasSetWorkspace(h, d_ws, ws_bytes));
  const cudaDataType type = fp16 ? CUDA_R_16F : CUDA_R_32F;
  const cublasComputeType_t compute =
      fp16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
  const float alpha = 1.f, beta = 0.f;

  auto gemm_rows = [&](int row0, int rows) {
    const char* jb = static_cast<const char*>(j) + std::size_t(row0) * ldj * elem;
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, batch, rows, n, &alpha,
                    d_sign, type, batch, jb, type, int(ldj), &beta,
                    d_acc + std::size_t(row0) * batch, CUDA_R_32F, batch,
                    compute, CUBLAS_GEMM_DEFAULT));
  };

  for (int r = 0; r <= reps; ++r) {  // r == 0 is a warm-up
    CK(cudaEventRecord(e0, stream));
    gemm_rows(0, n);
    CK(cudaEventRecord(e1, stream));
    const double s = elapsed();
    if (r) std::printf("  gemm full     [%d] : %8.3f s  %7.1f GB/s\n", r - 1, s,
                       bytes / s / 1e9);
  }
  for (int r = 0; r <= reps; ++r) {
    CK(cudaEventRecord(e0, stream));
    for (int row0 = 0; row0 < n; row0 += chunk)
      gemm_rows(row0, row0 + chunk <= n ? chunk : n - row0);
    CK(cudaEventRecord(e1, stream));
    const double s = elapsed();
    if (r) std::printf("  gemm chunked  [%d] : %8.3f s  %7.1f GB/s\n", r - 1, s,
                       bytes / s / 1e9);
  }

  cublasDestroy(h);
  cudaFree(d_ws);
  cudaFree(d_acc);
  cudaFree(d_sign);
  cudaFree(d_out);
  cudaHostUnregister(j);
  munmap(j, bytes);
  return 0;
}
