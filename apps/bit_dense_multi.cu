// apps/bit_dense_multi.cu -- dense +-1 dSB with a 1-bit coupling matrix on one or two GH200s.
//
// Batch 1.  J is a random symmetric +-1 matrix stored as one bit per coupling (bit 1 == +1),
// rows padded to a multiple of 128 bits.  The rows are split over the GPUs; on every GPU the
// first `head` rows live in HBM and the remaining `tail` rows in pinned Grace memory on the
// GPU's own NUMA node, staged through a 2 GiB HBM buffer every step -- the same placement as
// MATRIX_MEMORY=hybrid in dense_scaling (src/dense_capacity.cu, src/gemm.cu).  After each step
// the GPUs exchange their part of the sign bitmap (n/8 bytes) over NVLink.
//
//   A_i = sum_j J_ij s_j = n - 2 popc(J_i XOR S) - s_i        (diagonal stored as +1)
//   y  += (-(delta - p_k) x + xi A) dt ;  x += dt y delta ;  |x| > 1 -> x = sgn x, y = 0
//
// The product is exact (integer popcount) and the update uses explicit round-to-nearest
// intrinsics, so results do not depend on the number of GPUs or on the placement, and
// --check compares the final state bit for bit with a host reference (small n only).
//
//   ./bit_dense_multi --n 3000000 --gpus 2 --steps 50 --dry-run      # memory plan only
//   ./bit_dense_multi --n 8192 --gpus 2 --hbm-rows 1024 --check       # exercises staging
//
// --instance mattis: J_ij = xi_i xi_j for a hidden random xi in {+-1}^n.  The ground states are
// s = +-xi with energy -n(n-1)/2, so the run reports whether it was found (CSV column `found`).
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(call)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                   cudaGetErrorString(e_));                                             \
      std::exit(2);                                                                     \
    }                                                                                   \
  } while (0)

typedef std::uint32_t u32;
typedef std::uint64_t u64;

constexpr u64 kStageBytes = u64(2) << 30;  // HBM staging block for Grace rows
constexpr int kRowsPerWarp = 4;            // rows sharing one load of the sign bitmap

// ---------------------------------------------------------------------------------------
// shared host/device definitions
// ---------------------------------------------------------------------------------------
__host__ __device__ inline u32 fmix32(u32 h) {
  h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
  return h;
}

/// hidden solution of the Mattis instance, as a bit (1 == +1)
__host__ __device__ inline u32 xibit(u32 i, u32 seed) { return fmix32(i ^ fmix32(seed + 0x6d2b79f5u)) & 1u; }

/// J_ij as a bit (1 == +1).  Symmetric; the diagonal is stored as +1 and removed in A_i.
/// kind 0: random +-1;  kind 1: Mattis, J_ij = xi_i xi_j.
__host__ __device__ inline u32 jbit(u32 i, u32 j, u32 seed, int kind) {
  if (i == j) return 1u;
  if (kind == 1) return (xibit(i, seed) ^ xibit(j, seed)) ^ 1u;
  const u32 lo = i < j ? i : j, hi = i < j ? j : i;
  return fmix32(fmix32(hi ^ seed) + lo * 0x9e3779b9u) & 1u;
}

__host__ __device__ inline u32 jword(u32 row, u32 w, u32 n, u32 seed, int kind) {
  const u32 c0 = w * 32u;
  if (c0 >= n) return 0u;
  const u32 cnt = (n - c0) < 32u ? (n - c0) : 32u;
  u32 word = 0u;
  for (u32 b = 0; b < cnt; ++b) word |= jbit(row, c0 + b, seed, kind) << b;
  return word;
}

/// initial x or y of row i: uniform(-0.01, 0.01)
__host__ __device__ inline float init_value(u32 i, u32 which, u32 seed) {
#ifdef __CUDA_ARCH__
  const float u = __fmul_rn(float(fmix32(fmix32(i * 2u + which) ^ (seed * 0x27d4eb2du)) >> 8), 5.9604644775390625e-8f);
  return __fmul_rn(__fadd_rn(__fmul_rn(2.f, u), -1.f), 0.01f);
#else
  const float u = float(fmix32(fmix32(i * 2u + which) ^ (seed * 0x27d4eb2du)) >> 8) * 5.9604644775390625e-8f;
  return (2.f * u + -1.f) * 0.01f;
#endif
}

struct Dyn { float delta, xi, dt; };

__host__ __device__ inline void step_one(float& xv, float& yv, int a, float p, Dyn d) {
#ifdef __CUDA_ARCH__
  const float f = __fadd_rn(__fmul_rn(-(d.delta - p), xv), __fmul_rn(d.xi, float(a)));
  yv = __fadd_rn(yv, __fmul_rn(f, d.dt));
  xv = __fadd_rn(xv, __fmul_rn(__fmul_rn(d.dt, yv), d.delta));
#else
  const float f = -(d.delta - p) * xv + d.xi * float(a);   // host built with -ffp-contract=off
  yv = yv + f * d.dt;
  xv = xv + (d.dt * yv) * d.delta;
#endif
  if (fabsf(xv) > 1.f) { xv = xv > 0.f ? 1.f : -1.f; yv = 0.f; }
}

// ---------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------
__global__ void gen_rows(u32* out, u64 ldw, u32 row0, u32 rows, u32 n, u32 seed, int kind) {
  const u64 total = u64(rows) * ldw;
  for (u64 k = u64(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += u64(gridDim.x) * blockDim.x) {
    const u32 r = u32(k / ldw), w = u32(k - u64(r) * ldw);
    out[k] = jword(row0 + r, w, n, seed, kind);
  }
}

/// pop[r] = popc(J_r XOR S) for `rows` rows starting at J (row stride ldw4 uint4).
__global__ void __launch_bounds__(256) popc_rows(const uint4* __restrict__ J, u64 ldw4, u32 rows,
                                                 const uint4* __restrict__ v, int* __restrict__ pop) {
  const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31u;
  const u32 r0 = warp * kRowsPerWarp;
  if (r0 >= rows) return;                                  // warp-uniform
  const u32 nr = min(u32(kRowsPerWarp), rows - r0);
  const uint4* base = J + u64(r0) * ldw4;
  int acc[kRowsPerWarp] = {0, 0, 0, 0};
  for (u64 k = lane; k < ldw4; k += 32) {
    const uint4 s = __ldg(v + k);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      if (u32(r) < nr) {
        const uint4 a = __ldcs(base + u64(r) * ldw4 + k);
        acc[r] += __popc(a.x ^ s.x) + __popc(a.y ^ s.y) + __popc(a.z ^ s.z) + __popc(a.w ^ s.w);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    int t = acc[r];
    for (int o = 16; o; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
    if (lane == 0 && u32(r) < nr) pop[r0 + r] = t;
  }
}

/// local rows [0, rows) = global rows [row0, row0+rows); row0 % 32 == 0.
/// mode 0: initialise x, y;  mode 1: dSB step;  both write the sign bits into v.
__global__ void update_rows(float* x, float* y, const int* pop, u32* v, u32 row0, u32 rows,
                            u32 n, float p, Dyn d, int mode, u32 seed) {
  const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  bool bit = false;
  if (t < rows) {
    float xv, yv;
    if (mode == 0) {
      xv = init_value(row0 + t, 0u, seed);
      yv = init_value(row0 + t, 1u, seed);
    } else {
      xv = x[t]; yv = y[t];
      const int s = xv > 0.f ? 1 : -1;
      step_one(xv, yv, int(n) - 2 * pop[t] - s, p, d);
    }
    x[t] = xv; y[t] = yv;
    bit = xv > 0.f;
  }
  const u32 word = __ballot_sync(0xffffffffu, bit);
  if ((threadIdx.x & 31u) == 0 && t < rows) v[(row0 + t) >> 5] = word;
}

/// esum += sum_i s_i A_i over local rows (A from pop of the final spins)
__global__ void energy_rows(const float* x, const int* pop, u32 rows, u32 n, unsigned long long* esum) {
  const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  long long e = 0;
  if (t < rows) {
    const int s = x[t] > 0.f ? 1 : -1;
    e = (long long)s * (long long)(int(n) - 2 * pop[t] - s);
  }
  for (int o = 16; o; o >>= 1) e += __shfl_down_sync(0xffffffffu, e, o);
  if ((threadIdx.x & 31u) == 0 && e != 0) atomicAdd(esum, (unsigned long long)e);
}

// ---------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------
struct Opt {
  u64 n = 0; int gpus = 2, steps = 50, repeats = 3; u32 seed = 42;
  float dt = 1.1f; double hbm_fraction = 0.85; std::string placement = "auto";
  long long hbm_rows = -1; std::vector<int> numa; bool check = false, dry = false, csv = false;
  std::string instance = "random";
};

struct Gpu {
  int dev = 0, numa = -1; u32 r0 = 0, rows = 0, head = 0, tail = 0, stage_rows = 0;
  u32* d_head = nullptr; u32* h_tail = nullptr; u32* d_tail = nullptr; u64 tail_bytes = 0;
  u32* d_stage = nullptr; float* x = nullptr; float* y = nullptr; int* pop = nullptr;
  u32* v[2] = {nullptr, nullptr}; unsigned long long* esum = nullptr;
  cudaStream_t st = nullptr; cudaEvent_t ev_copy = nullptr, prof[5];
};

static long long node_free_bytes(int node) {
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.find("MemFree:") != std::string::npos) {
      std::istringstream is(line.substr(line.find("MemFree:") + 8));
      long long kb = 0; is >> kb; return kb * 1024;
    }
  }
  return -1;
}

static bool bind_to_node(void* p, u64 bytes, int node) {
  if (node < 0) return true;
  unsigned long mask[16] = {0};
  if (node >= int(sizeof(mask) * 8)) return false;
  mask[node / (8 * sizeof(unsigned long))] |= 1ul << (node % (8 * sizeof(unsigned long)));
  const long r = syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, mask, sizeof(mask) * 8, 0);
  return r == 0;
}

static int blocks_for(u64 threads, int per = 256) { return int((threads + per - 1) / per); }

static void usage() {
  std::fprintf(stderr,
      "bit_dense_multi --n N [--gpus 1|2] [--steps K] [--repeats R] [--dt 1.1] [--seed 42]\n"
      "                [--placement auto|hbm|grace] [--hbm-fraction 0.85] [--hbm-rows R]\n"
      "                [--numa 0,1] [--instance random|mattis] [--check] [--dry-run] [--csv] [--csv-header]\n");
}

static const char* kHeader =
    "implementation,n,gpus,repeat,batch,steps,placement,matrix_bytes,hbm_bytes,grace_bytes,"
    "generation_s,time_per_step_s,effective_matrix_GB_s,energy,checksum,status,instance,expected_energy,found,seed";

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(1); } return argv[++i]; };
    if (a == "--n") o.n = std::stoull(val());
    else if (a == "--gpus") o.gpus = std::stoi(val());
    else if (a == "--steps") o.steps = std::stoi(val());
    else if (a == "--repeats") o.repeats = std::stoi(val());
    else if (a == "--dt") o.dt = std::stof(val());
    else if (a == "--seed") o.seed = u32(std::stoul(val()));
    else if (a == "--placement") o.placement = val();
    else if (a == "--hbm-fraction") o.hbm_fraction = std::stod(val());
    else if (a == "--hbm-rows") o.hbm_rows = std::stoll(val());
    else if (a == "--numa") { std::stringstream ss(val()); std::string t; while (std::getline(ss, t, ',')) o.numa.push_back(std::stoi(t)); }
    else if (a == "--instance") o.instance = val();
    else if (a == "--check") o.check = true;
    else if (a == "--dry-run") o.dry = true;
    else if (a == "--csv") o.csv = true;
    else if (a == "--csv-header") { std::printf("%s\n", kHeader); return 0; }
    else { usage(); return 1; }
  }
  if (o.n < 64 || o.n >= (u64(1) << 32) || o.gpus < 1 || o.gpus > 2 || o.steps < 1) { usage(); return 1; }
  if (o.placement != "auto" && o.placement != "hbm" && o.placement != "grace") { usage(); return 1; }
  if (o.instance != "random" && o.instance != "mattis") { usage(); return 1; }
  const int kind = o.instance == "mattis" ? 1 : 0;
  // Mattis ground-state energy -n(n-1)/2; the random instance has no known optimum (field left empty)
  const std::string expected = kind == 1 ? std::to_string(-(long long)(o.n * (o.n - 1) / 2)) : std::string();
  int ndev = 0; CK(cudaGetDeviceCount(&ndev));
  if (ndev < o.gpus) { std::fprintf(stderr, "need %d GPUs, found %d\n", o.gpus, ndev); return 1; }

  const u32 n = u32(o.n);
  const u64 words = (o.n + 31) / 32, ldw = (words + 3) / 4 * 4, ldw4 = ldw / 4, row_bytes = ldw * 4;
  const u64 matrix_bytes = o.n * row_bytes;
  const Dyn dyn{1.f, float(0.5 / std::sqrt(double(o.n))), o.dt};   // xi = 0.5 sqrt(n-1)/||J||_F
  const u32 per = u32((o.n + o.gpus - 1) / o.gpus + 31) / 32 * 32;

  // ---- plan -------------------------------------------------------------------------------
  std::vector<Gpu> G(o.gpus);
  bool plan_ok = true;
  for (int g = 0; g < o.gpus; ++g) {
    Gpu& q = G[g];
    q.dev = g;
    q.r0 = std::min<u64>(o.n, u64(g) * per);
    q.rows = u32(std::min<u64>(o.n, u64(q.r0) + per) - q.r0);
    CK(cudaSetDevice(q.dev)); CK(cudaFree(0));
    q.numa = g < int(o.numa.size()) ? o.numa[g] : -1;
#if CUDART_VERSION >= 12020
    if (q.numa < 0) { int id = -1; if (cudaDeviceGetAttribute(&id, cudaDevAttrHostNumaId, q.dev) == cudaSuccess) q.numa = id; }
#endif
    if (q.numa < 0) q.numa = g;
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    const u64 small = 2ull * q.rows * 4 + u64(q.rows) * 4 + 2 * row_bytes + (u64(256) << 20);
    const u64 stage = std::min<u64>(u64(q.rows) * row_bytes, std::max<u64>(row_bytes, kStageBytes / row_bytes * row_bytes));
    const double budget = double(fr) * o.hbm_fraction - double(small);
    u64 head = u64(q.rows);
    if (double(head) * row_bytes > budget) head = budget > double(stage) ? u64((budget - stage) / row_bytes) : 0;
    if (o.placement == "grace") head = 0;
    if (o.hbm_rows >= 0) head = std::min<u64>(head, u64(o.hbm_rows));
    if (head < q.rows) head -= head % 32;
    if (o.placement == "hbm" && head < q.rows) { std::fprintf(stderr, "GPU %d: matrix part does not fit in HBM\n", g); plan_ok = false; }
    q.head = u32(head); q.tail = q.rows - q.head;
    q.tail_bytes = u64(q.tail) * row_bytes;
    q.stage_rows = q.tail ? u32(std::min<u64>(q.tail, std::max<u64>(1, kStageBytes / row_bytes))) : 0;
    const long long nf = node_free_bytes(q.numa);
    std::fprintf(stderr, "GPU %d: rows [%u,%u)  HBM %u rows (%.1f GB)  Grace node %d: %u rows (%.1f GB, node free %.1f GB)\n",
                 g, q.r0, q.r0 + q.rows, q.head, q.head * double(row_bytes) / 1e9, q.numa, q.tail,
                 q.tail_bytes / 1e9, nf / 1e9);
    if (q.tail && nf >= 0 && double(q.tail_bytes) > 0.97 * double(nf)) {
      std::fprintf(stderr, "GPU %d: Grace tail exceeds free memory of node %d\n", g, q.numa); plan_ok = false;
    }
  }
  u64 hbm_b = 0, grace_b = 0;
  for (auto& q : G) { hbm_b += u64(q.head) * row_bytes; grace_b += q.tail_bytes; }
  const char* placement = grace_b == 0 ? "hbm" : (hbm_b == 0 ? "grace" : "hybrid");
  std::fprintf(stderr, "n=%llu  matrix %.1f GB (%.1f HBM + %.1f Grace), placement %s, xi=%.3g dt=%.3g\n",
               (unsigned long long)o.n, matrix_bytes / 1e9, hbm_b / 1e9, grace_b / 1e9, placement, dyn.xi, dyn.dt);
  if (!plan_ok || o.dry) {
    if (o.csv) std::printf("dsb-gpu-bitpm1,%llu,%d,0,1,%d,%s,%llu,%llu,%llu,,,,,,%s,%s,%s,,%u\n", (unsigned long long)o.n,
                           o.gpus, o.steps, placement, (unsigned long long)matrix_bytes,
                           (unsigned long long)hbm_b, (unsigned long long)grace_b, plan_ok ? "planned" : "does-not-fit",
                           o.instance.c_str(), expected.c_str(), o.seed);
    return plan_ok ? 0 : 3;
  }

  // ---- allocate and generate -------------------------------------------------------------
  const auto t_gen0 = std::chrono::steady_clock::now();
  for (auto& q : G) {
    CK(cudaSetDevice(q.dev));
    CK(cudaStreamCreateWithFlags(&q.st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&q.ev_copy, cudaEventDisableTiming));
    for (auto& e : q.prof) CK(cudaEventCreate(&e));
    CK(cudaMalloc(&q.x, u64(q.rows) * 4)); CK(cudaMalloc(&q.y, u64(q.rows) * 4));
    CK(cudaMalloc(&q.pop, u64(q.rows) * 4)); CK(cudaMalloc(&q.esum, 8));
    for (auto& b : q.v) { CK(cudaMalloc(&b, row_bytes)); CK(cudaMemset(b, 0, row_bytes)); }
    if (q.head) CK(cudaMalloc(&q.d_head, u64(q.head) * row_bytes));
    if (q.tail) {
      CK(cudaMalloc(&q.d_stage, u64(q.stage_rows) * row_bytes));
      void* p = mmap(nullptr, q.tail_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED) { std::fprintf(stderr, "mmap of %.1f GB failed\n", q.tail_bytes / 1e9); return 4; }
      if (!bind_to_node(p, q.tail_bytes, q.numa)) std::fprintf(stderr, "warning: mbind to node %d failed\n", q.numa);
      CK(cudaHostRegister(p, q.tail_bytes, cudaHostRegisterDefault));
      q.h_tail = static_cast<u32*>(p);
      CK(cudaHostGetDevicePointer((void**)&q.d_tail, p, 0));
    }
    int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, q.dev));
    if (q.head) gen_rows<<<sms * 32, 256, 0, q.st>>>(q.d_head, ldw, q.r0, q.head, n, o.seed, kind);
    if (q.tail) gen_rows<<<sms * 32, 256, 0, q.st>>>(q.d_tail, ldw, q.r0 + q.head, q.tail, n, o.seed, kind);
    CK(cudaGetLastError());
  }
  for (auto& q : G) { CK(cudaSetDevice(q.dev)); CK(cudaStreamSynchronize(q.st)); }
  if (o.gpus == 2) {
    for (int g = 0; g < 2; ++g) {
      int ok = 0; CK(cudaDeviceCanAccessPeer(&ok, G[g].dev, G[1 - g].dev));
      CK(cudaSetDevice(G[g].dev));
      if (ok) { cudaError_t e = cudaDeviceEnablePeerAccess(G[1 - g].dev, 0); if (e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
      else std::fprintf(stderr, "warning: no peer access %d->%d, bitmap exchange goes through the host\n", g, 1 - g);
    }
  }
  const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen0).count();
  std::fprintf(stderr, "allocation + generation: %.1f s\n", gen_s);

  // ---- one integration step (or the final energy product) ---------------------------------
  auto product = [&](Gpu& q, const u32* v, bool prof) {
    if (prof) CK(cudaEventRecord(q.prof[0], q.st));
    if (q.head)
      popc_rows<<<blocks_for(u64((q.head + kRowsPerWarp - 1) / kRowsPerWarp) * 32), 256, 0, q.st>>>(
          reinterpret_cast<const uint4*>(q.d_head), ldw4, q.head, reinterpret_cast<const uint4*>(v), q.pop);
    if (prof) CK(cudaEventRecord(q.prof[1], q.st));
    for (u32 r = 0; r < q.tail; r += q.stage_rows) {
      const u32 cnt = std::min(q.stage_rows, q.tail - r);
      CK(cudaMemcpyAsync(q.d_stage, q.h_tail + u64(r) * ldw, u64(cnt) * row_bytes, cudaMemcpyDefault, q.st));
      popc_rows<<<blocks_for(u64((cnt + kRowsPerWarp - 1) / kRowsPerWarp) * 32), 256, 0, q.st>>>(
          reinterpret_cast<const uint4*>(q.d_stage), ldw4, cnt, reinterpret_cast<const uint4*>(v), q.pop + q.head + r);
    }
    if (prof) CK(cudaEventRecord(q.prof[2], q.st));
  };
  auto exchange = [&](int buf) {             // copy each GPU's bitmap segment into the peer's buffer
    if (o.gpus == 1) return;
    for (int g = 0; g < 2; ++g) {
      Gpu& q = G[g]; Gpu& h = G[1 - g];
      const u64 w0 = q.r0 / 32, nw = (u64(q.r0) + q.rows + 31) / 32 - w0;
      CK(cudaSetDevice(q.dev));
      CK(cudaMemcpyPeerAsync(h.v[buf] + w0, h.dev, q.v[buf] + w0, q.dev, nw * 4, q.st));
      CK(cudaEventRecord(q.ev_copy, q.st));
    }
    for (int g = 0; g < 2; ++g) { CK(cudaSetDevice(G[g].dev)); CK(cudaStreamWaitEvent(G[g].st, G[1 - g].ev_copy, 0)); }
  };
  auto sync_all = [&]() { for (auto& q : G) { CK(cudaSetDevice(q.dev)); CK(cudaStreamSynchronize(q.st)); } };

  std::vector<double> tps;
  std::vector<float> prof_ms(G.size() * 4, 0.f);
  long long energy = 0; u64 checksum = 0;
  const int prof_step = o.steps / 2;
  for (int rep = 0; rep <= o.repeats; ++rep) {             // rep 0 = warm-up, not reported
    for (auto& q : G) {
      CK(cudaSetDevice(q.dev));
      update_rows<<<blocks_for(q.rows), 256, 0, q.st>>>(q.x, q.y, q.pop, q.v[0], q.r0, q.rows, n, 0.f, dyn, 0, o.seed);
      CK(cudaGetLastError());
    }
    exchange(0);
    sync_all();
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < o.steps; ++k) {
      const int cur = k & 1, nxt = cur ^ 1;
      const float p = o.steps > 1 ? float(k) / float(o.steps - 1) : 0.f;
      const bool prof = rep == 1 && k == prof_step;
      for (auto& q : G) { CK(cudaSetDevice(q.dev)); product(q, q.v[cur], prof); }
      for (auto& q : G) {
        CK(cudaSetDevice(q.dev));
        update_rows<<<blocks_for(q.rows), 256, 0, q.st>>>(q.x, q.y, q.pop, q.v[nxt], q.r0, q.rows, n, p, dyn, 1, o.seed);
        if (prof) CK(cudaEventRecord(q.prof[3], q.st));
      }
      exchange(nxt);
      if (prof) for (auto& q : G) { CK(cudaSetDevice(q.dev)); CK(cudaEventRecord(q.prof[4], q.st)); }
    }
    sync_all();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CK(cudaGetLastError());
    if (rep == 1)
      for (size_t g = 0; g < G.size(); ++g)
        for (int j = 0; j < 4; ++j) { CK(cudaSetDevice(G[g].dev)); CK(cudaEventElapsedTime(&prof_ms[g * 4 + j], G[g].prof[j], G[g].prof[j + 1])); }

    // final energy E = -1/2 sum_i s_i A_i and a checksum of the final spins
    const int fin = o.steps & 1;
    for (auto& q : G) {
      CK(cudaSetDevice(q.dev)); CK(cudaMemsetAsync(q.esum, 0, 8, q.st));
      product(q, q.v[fin], false);
      energy_rows<<<blocks_for(q.rows), 256, 0, q.st>>>(q.x, q.pop, q.rows, n, q.esum);
    }
    sync_all();
    long long sum = 0;
    for (auto& q : G) { unsigned long long e = 0; CK(cudaSetDevice(q.dev)); CK(cudaMemcpy(&e, q.esum, 8, cudaMemcpyDeviceToHost)); sum += (long long)e; }
    energy = -sum / 2;
    std::vector<u32> bits(words);
    CK(cudaSetDevice(G[0].dev)); CK(cudaMemcpy(bits.data(), G[0].v[fin], words * 4, cudaMemcpyDeviceToHost));
    checksum = 1469598103934665603ull;
    for (u32 w : bits) { checksum ^= w; checksum *= 1099511628211ull; }
    if (rep == 0) continue;
    tps.push_back(s / o.steps);
    const char* found = kind == 1 ? (std::to_string(energy) == expected ? "1" : "0") : "";
    std::fprintf(stderr, "repeat %d: %.4f s/step  %.1f GB/s  energy %lld%s%s  checksum %016llx\n", rep, s / o.steps,
                 matrix_bytes / (s / o.steps) / 1e9, energy, kind == 1 ? "  expected " : "", expected.c_str(),
                 (unsigned long long)checksum);
    if (o.csv)
      std::printf("dsb-gpu-bitpm1,%llu,%d,%d,1,%d,%s,%llu,%llu,%llu,%.6g,%.9g,%.6g,%lld,%016llx,completed,%s,%s,%s,%u\n",
                  (unsigned long long)o.n, o.gpus, rep - 1, o.steps, placement, (unsigned long long)matrix_bytes,
                  (unsigned long long)hbm_b, (unsigned long long)grace_b, gen_s, s / o.steps,
                  matrix_bytes / (s / o.steps) / 1e9, energy, (unsigned long long)checksum,
                  o.instance.c_str(), expected.c_str(), found, o.seed);
    std::fflush(stdout);
  }
  for (size_t g = 0; g < G.size(); ++g) {
    const Gpu& q = G[g]; const float* m = &prof_ms[g * 4];
    std::fprintf(stderr, "GPU %zu step %d: HBM rows %.2f ms (%.0f GB/s)  Grace rows %.2f ms (%.0f GB/s)  update %.3f ms  exchange %.3f ms\n",
                 g, prof_step, m[0], m[0] > 0 ? q.head * double(row_bytes) / (m[0] * 1e6) : 0.0, m[1],
                 m[1] > 0 ? q.tail_bytes / (m[1] * 1e6) : 0.0, m[2], m[3]);
  }

  // ---- host reference ----------------------------------------------------------------------
  int status = 0;
  if (o.check) {
    if (o.n > 50000) { std::fprintf(stderr, "--check skipped: n > 50000\n"); }
    else {
      std::vector<u32> J(o.n * ldw, 0u);
      for (u32 i = 0; i < n; ++i) for (u64 w = 0; w < words; ++w) J[u64(i) * ldw + w] = jword(i, u32(w), n, o.seed, kind);
      std::vector<float> x(n), y(n); std::vector<u32> v(ldw, 0u), vn(ldw, 0u);
      for (u32 i = 0; i < n; ++i) { x[i] = init_value(i, 0u, o.seed); y[i] = init_value(i, 1u, o.seed); if (x[i] > 0.f) v[i / 32] |= 1u << (i % 32); }
      for (int k = 0; k < o.steps; ++k) {
        const float p = o.steps > 1 ? float(k) / float(o.steps - 1) : 0.f;
        std::fill(vn.begin(), vn.end(), 0u);
        for (u32 i = 0; i < n; ++i) {
          int pc = 0; for (u64 w = 0; w < ldw; ++w) pc += __builtin_popcount(J[u64(i) * ldw + w] ^ v[w]);
          const int s = x[i] > 0.f ? 1 : -1;
          step_one(x[i], y[i], int(n) - 2 * pc - s, p, dyn);
          if (x[i] > 0.f) vn[i / 32] |= 1u << (i % 32);
        }
        v.swap(vn);
      }
      std::vector<float> gx(n);
      for (auto& q : G) { CK(cudaSetDevice(q.dev)); CK(cudaMemcpy(gx.data() + q.r0, q.x, u64(q.rows) * 4, cudaMemcpyDeviceToHost)); }
      u64 diff = 0; for (u32 i = 0; i < n; ++i) diff += std::memcmp(&x[i], &gx[i], 4) != 0;
      u64 cs = 1469598103934665603ull; for (u64 w = 0; w < words; ++w) { cs ^= v[w]; cs *= 1099511628211ull; }
      std::fprintf(stderr, "check vs host reference: %llu of %u positions differ, checksum host %016llx gpu %016llx -> %s\n",
                   (unsigned long long)diff, n, (unsigned long long)cs, (unsigned long long)checksum,
                   (diff == 0 && cs == checksum) ? "PASS" : "FAIL");
      status = (diff == 0 && cs == checksum) ? 0 : 5;
    }
  }
  if (!tps.empty()) {
    std::sort(tps.begin(), tps.end());
    std::fprintf(stderr, "median %.4f s/step over %zu repeats\n", tps[tps.size() / 2], tps.size());
  }
  for (auto& q : G) { if (q.h_tail) { cudaHostUnregister(q.h_tail); munmap(q.h_tail, q.tail_bytes); } }
  return status;
}
