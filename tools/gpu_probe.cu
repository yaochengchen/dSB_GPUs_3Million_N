// tools/gpu_probe.cu
//
// Reports what this GPU actually allows: shared memory opt-in size, thread
// block cluster support, working distributed shared memory, and the resulting
// N ceilings for the 9N-byte state layout used by the solver.
//
//   make probe && ./gpu_probe

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace cg = cooperative_groups;

#define CK(call)                                                             \
  do {                                                                       \
    cudaError_t e_ = (call);                                                  \
    if (e_ != cudaSuccess)                                                    \
      std::printf("  !! %s -> %s\n", #call, cudaGetErrorString(e_));          \
  } while (0)

// Each rank writes its own id into its shared memory, then every rank reads
// rank 0's. If DSMEM works, all outputs come back as 100.
__global__ void dsmem_probe(int* out) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ int smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned    rank    = cluster.block_rank();

  if (threadIdx.x == 0) smem[0] = 100 + int(rank);
  cluster.sync();

  int* remote = cluster.map_shared_rank(smem, 0);
  int  value  = remote[0];
  cluster.sync();

  if (threadIdx.x == 0) out[rank] = value;
#else
  if (threadIdx.x == 0) out[blockIdx.x] = -1;
#endif
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));

  std::printf("=== %s  (sm_%d%d) ===\n", prop.name, prop.major, prop.minor);
  std::printf("  SMs                  : %d\n", prop.multiProcessorCount);
  std::printf("  L2 cache             : %.1f MB\n", prop.l2CacheSize / 1048576.0);
  std::printf("  device memory        : %.1f GB\n",
              prop.totalGlobalMem / 1073741824.0);

  int optin = 0, per_sm = 0, cluster_ok = 0;
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  CK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&cluster_ok, cudaDevAttrClusterLaunch, dev));

  std::printf("\n  shared / SM          : %d B (%.0f KB)\n", per_sm, per_sm / 1024.0);
  std::printf("  shared / block deflt : %zu B (%.0f KB)  <- without opt-in\n",
              prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
  std::printf("  shared / block optin : %d B (%.0f KB)\n", optin, optin / 1024.0);

  std::printf("\n  --- N ceiling, 9N-byte layout (x,y fp32 + sign int8) ---\n");
  std::printf("  single block         : N <= %d\n", optin / 9);
  for (int c : {2, 4, 8})
    std::printf("  cluster of %-2d        : N <= %d\n", c, c * (optin / 9));

  std::printf("\n  cluster launch       : %s\n", cluster_ok ? "YES" : "NO");

  if (cluster_ok) {
    cudaLaunchConfig_t query = {};
    query.gridDim          = dim3(8);
    query.blockDim         = dim3(1024);
    query.dynamicSmemBytes = 0;
    int max_cluster        = 0;
    CK(cudaOccupancyMaxPotentialClusterSize(&max_cluster, (void*)dsmem_probe,
                                            &query));
    std::printf("  max cluster size     : %d  (8 is the portable guarantee)\n",
                max_cluster);

    const int c = (max_cluster >= 8) ? 8 : (max_cluster >= 2 ? max_cluster : 2);
    int*      d_out = nullptr;
    int       h_out[16] = {0};
    CK(cudaMalloc(&d_out, sizeof(h_out)));
    CK(cudaMemset(d_out, 0, sizeof(h_out)));

    cudaLaunchAttribute attr[1];
    attr[0].id               = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = unsigned(c);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim          = dim3(unsigned(c));
    cfg.blockDim         = dim3(32);
    cfg.dynamicSmemBytes = 1024;
    cfg.stream           = nullptr;
    cfg.attrs            = attr;
    cfg.numAttrs         = 1;

    CK(cudaLaunchKernelEx(&cfg, dsmem_probe, d_out));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < c; ++i)
      if (h_out[i] != 100) ok = false;
    std::printf("  DSMEM map_shared_rank: %s  (C=%d, got", ok ? "OK" : "FAIL", c);
    for (int i = 0; i < c; ++i) std::printf(" %d", h_out[i]);
    std::printf(")\n");
    CK(cudaFree(d_out));
  } else {
    std::printf("  -> no cluster kernel here; only the single-block path works\n");
  }

  std::printf("\n  --- dense J footprint ---\n");
  for (int n : {2000, 8192, 25000, 50000, 100000, 200000}) {
    const double gb = 2.0 * double(n) * double(n) / 1073741824.0;
    std::printf("  N=%7d : %8.2f GB in fp16%s\n", n, gb,
                gb > prop.totalGlobalMem / 1073741824.0 ? "   <- does not fit" : "");
  }
  return 0;
}
