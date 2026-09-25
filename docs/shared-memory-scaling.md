# 共享内存 N 上限：GH200 的大内存能不能帮上忙？

**短答：不能。而且问错了对象 —— 那 `5N` 个 float 里没有 J。**

> 注：本文的硬件数字来自已有知识，本次会话无法联网核对。上机前请用
> `deviceQuery` 或 `cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)`
> 亲自确认一遍再写进论文。

---

## 1. 先把"谁占了共享内存"搞清楚

`cdsb_fastshare.cu` 里的 `5*N*sizeof(float)`：

```
x0[N]  y0[N]  x1[N]  y1[N]  sgn[N]      ← 全是 状态向量
```

**J 从头到尾都在 global memory（HBM）里**，kernel 每个 iteration 通过
`const __half* Jrow = J + row*N;` 从全局内存流式读取。共享内存里一个字节的 J 都没有。

所以"把 J 挪到别的内存去以放大 N"这个思路不成立 —— J 本来就在"别的内存"里。
限制 N 的是 **每个 replica 的 5 个长度为 N 的状态数组**，它们必须住在 SM 的片上
scratchpad 里，否则 replica-per-block 这个设计的全部意义（把全局同步塌缩成
block 内 barrier）就没了。

---

## 2. GH200 的共享内存和 H100 一模一样

GH200 = Grace CPU + **同一颗 GH100 GPU die**，compute capability 同为 sm_90。

| | H100 SXM | GH200 |
|---|---|---|
| 每 SM 共享内存 | 228 KB | 228 KB（同一颗 die） |
| 单 block 最大动态 SMEM (opt-in) | 227 KB | 227 KB |
| 不 opt-in 的默认上限 | 48 KB | 48 KB |

**GH200 一个字节的共享内存都不会多给你。** paper-plan 里已经写对了这一点。

Blackwell（B200/GB200, sm_100）也还是 228 KB/SM —— 换代也救不了。

---

## 3. "特殊内存"确实存在，但都不能用来放 x/y

| 候选 | 容量 | 为什么不行 |
|---|---|---|
| **Grace LPDDR5X**（480 GB，NVLink-C2C） | 很大 | 它是 *global memory* 的延伸，不是片上 scratchpad。放 x/y 等于放弃 SMEM 驻留，退回 V3 的全局内存版本 |
| **Blackwell Tensor Memory (TMEM)** | 256 KB/SM | 只能被 `tcgen05` 系列 MMA 指令和 `tcgen05.ld/st` 访问，不是通用 load/store 的地址空间。设计目的是 MMA 累加器，塞状态向量进去既不自然也无文档支持 |
| **L2 cache**（H100 50 MB） | 中等 | 不可编程寻址，只能靠 `cudaAccessPolicyWindow` 做 persisting hint。而且 N=11.6k 时 J 就有 269 MB，L2 装不下 |
| **常量内存 / texture** | 64 KB | 太小，而且只读 |

**唯一真正能放大 N 的片上机制是下一节的 DSMEM，而它是 Hopper 特性 —— H100 和 GH200 都有，不是 GH200 独有的。**

---

## 4. 真正能把 N 推上去的三条路

### 路线 A：先把 footprint 压下来（几乎零成本，先做这个）

见 code-review 的 B4：`x1/y1` 是冗余的（`sgn` 已经承担了 Jacobi 快照），`sgn` 只需要 1 字节。

| 布局 | 每 N 字节 | N 上限（227 KB） | N 上限（48 KB，未 opt-in） |
|---|---|---|---|
| 现状 `x0,y0,x1,y1,sgn` 全 fp32 | 20N | 11,622 | **2,457 ← 现在代码的真实上限** |
| 去掉 ping-pong：`x0,y0,sgn` fp32 | 12N | 19,370 | 4,096 |
| 再把 `sgn` 换 int8 | 9N | **25,827** | 5,461 |
| 激进：`x,y` 存 fp16 + `sgn` int8 | 5N | 46,489 | 9,830 |

最后一行要谨慎 —— x 是积分变量，fp16 存储会累积误差，得配实验 4 的精度对照。
**前三行加起来是 2.2 倍，改动不超过 30 行，而且顺便省掉每步的拷贝和一次 barrier。**

### 路线 B：Thread Block Cluster + 分布式共享内存（DSMEM）

这是 Hopper（sm_90）引入的、你问的那种"特殊内存机制"，**H100 和 GH200 都有**。

一个 cluster 内最多 8 个 block（portable；H100 上 opt-in 可到 16）落在同一个 GPC 上，
可以直接读写彼此的共享内存（SM-to-SM 网络，绕过 L2），并且有 `cluster.sync()` 这个
比 `grid.sync()` 便宜得多的同步原语。

把**一个 replica 铺到一个 cluster** 而不是一个 block：

| cluster 大小 | 可寻址 SMEM | N 上限（9N 布局） |
|---|---|---|
| 1（现状） | 227 KB | 25.8 k |
| 4 | 908 KB | 103 k |
| 8 | 1.77 MB | **206 k** |
| 16（Hopper non-portable opt-in） | 3.55 MB | **412 k** |

代价：`sgn[]` 的跨 block 访问走 DSMEM，延迟高于本地 SMEM（但远低于 L2/HBM）；
同步从 `__syncthreads()` 变成 `cluster.sync()`。

**这条路线对论文的价值很高**：它是 Hopper 独有的、和你的 replica-per-block 设计
天然契合的硬件特性，而且能把"N 上限"这条主线从 11.6k 推到 10 万量级 ——
足够覆盖 G-set 全部实例和绝大多数 QPLIB。

### 路线 C：换成 GEMM / Tensor Core 路径

见下一节 —— 到了大 N，N 上限根本不是瓶颈。

---

## 5. 但是：把 N 推到 10 万是**没有意义**的，除非同时换掉 J 的访问方式

这是整件事最重要的一点，也是论文第 4 节该讲的东西。

replica-per-block 下，**每个 block 每个 iteration 要把整个 J 读一遍**：

- 每 block 每步：`2N²` 字节（FP16）
- 算术强度：**O(1)** —— 每读一个 J 元素只做一次 FMA，另一个操作数还是 ±1

| N | J (fp16) | 单 replica 单步 HBM 流量 | 800 步 | 能放哪 |
|---|---|---|---|---|
| 2,000 | 8 MB | 8 MB | 6.4 GB | L2 装得下，很快 |
| 11,600 | 269 MB | 269 MB | 215 GB | 超 L2（50 MB），每步真走 HBM |
| 25,800 | 1.33 GB | 1.33 GB | 1.07 TB | HBM 够，但每步 ~0.4 ms 只在读 J |
| 100,000 | 20 GB | 20 GB | 16 TB | HBM 够（96 GB），每步 ~6 ms |
| 206,000 | 85 GB | 85 GB | 68 TB | HBM 勉强，**每步 ~25 ms → 800 步 20 秒/replica** |
| 500,000 | 500 GB | 500 GB | 400 TB | 只能放 Grace LPDDR，C2C 450 GB/s → **每步 1.1 s，800 步 15 分钟** |

（按 HBM3 ~3.4 TB/s 估；GH200 的 HBM3/HBM3e 会好一些，但不改变结论）

所以 **GH200 的 480 GB LPDDR 对 J 来说是个陷阱**：

1. 它不限制 N —— 限制 N 的是共享内存；
2. 就算你真把 J 放过去，NVLink-C2C 的 ~450 GB/s 比 HBM 低 7–9 倍，
   而这个 kernel **已经是 J-bandwidth-bound 的**，所以整体会慢 7–9 倍。

**大 N 的正确解法不是更大的内存，是提高算术强度**：把 `J @ S`（S 是 N×B 的 ±1 矩阵）
做成一次真正的 GEMM，J 读一遍被 B 个 replica 共用 —— 算术强度从 O(1) 变成 O(B)。
代价是 replica 之间必须重新全局同步，退回 grid.sync 或多 kernel launch。
这正是 paper-plan 里说的"另一条 GEMM 路径"，也确实该留给下一篇。

---

## 6. 给论文的建议

**第 4 节（GH200 适配，1.5 页）就写成"内存分层的三道墙"，这比"我们在第二个平台上也跑了"有意思得多：**

1. **片上墙（共享内存）**：sm_90 上 227 KB/block。GH200 == H100，换平台没用。
   能动的只有 ① 压 footprint（20N → 9N，路线 A）和 ② Hopper cluster/DSMEM（路线 B）。
   **这是你的贡献。**
2. **HBM 墙**：dense J = 2N² 字节。N ≲ 200k 时 96 GB HBM 够用，
   但每步流量已经让 time-to-solution 不可接受。
3. **C2C 墙**：越过 HBM 进 Grace LPDDR 之后，带宽掉 7–9 倍，
   而这个 kernel 是纯 bandwidth-bound —— **GH200 的大内存在这里帮不上忙，
   这本身就是一个值得报告的负面结果。**

配一张图：横轴 N（log），纵轴 per-step 时间，三条竖线标出三道墙的位置，
再叠上 FastHare 约简后各 QPLIB 实例的 N 落点。**这张图能撑起整节。**

结论句可以直接写："the shared-memory-resident design is bounded by on-chip
scratchpad, not by device memory capacity; consequently the extended
LPDDR5X capacity of Grace-Hopper does not extend the applicable range of
this kernel, and the memory hierarchy beyond HBM is counter-productive for
an O(1)-arithmetic-intensity inner loop."

这是个干净、可验证、审稿人会喜欢的 negative result。

---

## 7. 立刻可以做的两件事

```bash
# 1) 确认这台机器的真实 opt-in 上限
cat > q.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main(){int d=0,v=0;cudaGetDevice(&d);
  cudaDeviceGetAttribute(&v,cudaDevAttrMaxSharedMemoryPerBlockOptin,d);
  printf("optin  = %d B  -> N_max(20N)=%d  N_max(9N)=%d\n",v,v/20,v/9);
  cudaDeviceGetAttribute(&v,cudaDevAttrMaxSharedMemoryPerMultiprocessor,d);
  printf("per SM = %d B\n",v);
  cudaDeviceProp p; cudaGetDeviceProperties(&p,d);
  printf("sm_%d%d  SMs=%d  L2=%d B\n",p.major,p.minor,p.multiProcessorCount,p.l2CacheSize);
  return 0;}
EOF
nvcc -arch=sm_90 q.cu -o q && ./q
```

```cpp
// 2) 在 cdsb_fused_run_fp16() 里补上 opt-in（没有这句，上面的 227 KB 一个字节都用不到）
auto kern = cdsb_fused_dense_fastshare_kernel_fp16<BLOCK>;
CDSB_CUDA_CHECK(cudaFuncSetAttribute(
    kern, cudaFuncAttributeMaxDynamicSharedMemorySizeBytes, (int)smem));
kern<<<dim3(B), dim3(BLOCK), smem, stream>>>(dY,dX,dJ,dP,delta,xi,dt,N,B,iters);
```

---

## 8. 补充（2026-09-19 追加）：cluster/DSMEM 到底能不能用，两台机器分别怎样

> 同样没能联网核对。第 7 节的 `gpu_probe.cu` 是用来在两台机器上一次问清楚的，
> **先跑它再决定**。

### 8.1 两台机器的预期（需实测确认）

| | 2× GH200 | 8× RTX 6000 Pro (Blackwell) |
|---|---|---|
| 架构 | GH100, **sm_90** | GB202, **sm_120** |
| 每 SM 共享内存 | 228 KB | **100 KB**（Ada 血统，不是 228） |
| 单 block opt-in | 227 KB | **~99 KB** |
| Thread Block Cluster | **确定有**（Hopper 原生特性） | **大概率有，但必须实测** |
| DSMEM `map_shared_rank` | **确定有** | **不确定** —— sm_120 砍掉了 sm_100 的 tcgen05/TMEM，是否也砍了 DSMEM 需要实测 |
| tcgen05 / TMEM | 无（Hopper 没有） | 无（只有 sm_100 有） |

**最要紧的一条不是 cluster，是这个：RTX 6000 Pro 每 SM 只有 ~100 KB 共享内存，不是 228 KB。**

| 布局 | GH200 (227 KB) | RTX 6000 Pro (~99 KB) |
|---|---|---|
| 现状 20N（且已 opt-in） | N ≤ 11,622 | N ≤ **5,069** |
| 去 ping-pong 12N | N ≤ 19,370 | N ≤ 8,448 |
| + sgn int8 → 9N | N ≤ 25,827 | N ≤ **11,264** |

也就是说：**RTX 6000 Pro 上做完全部 footprint 优化，才刚好追平 GH200 未优化时的 11.6k。**
论文里"单卡 N 上限"这条线必须按平台分开报，不能混。

（反过来说，8 张卡 × 188 SM 的吞吐远超 2× GH200，做 batch/多实例扫描时 RTX 机器才是主力。）

### 8.2 修正：cluster 有两种用法，我上一轮只说了较弱的那种

**设计 A：把一个 replica 铺到一个 cluster（"放大 N"）**

每个 block 拥有 N/C 个 row，`sgn` 分片存放，跨 block 通过 DSMEM 读。
- 收益：N 上限 × C。
- DSMEM 流量：每 cluster 每步 `N(C-1)` 字节。N=100k、C=8 时约 700 KB ——
  相对于同一步要读的 `2N²/C = 2.5 GB` 的 J，**完全可以忽略**。所以设计 A 是安全的。
- 但：**它不解决任何带宽问题**。N 推到 10 万，每步还是要读 20 GB 的 J。

**设计 B：一个 cluster 装 C 个 replica，共享 J 的 tile（"提高算术强度"）**

C 个 block 各跑一个 replica，但**同步推进同一段 J 的 tile**；tile 由一个 rank（或 TMA
multicast）载入 SMEM，另外 C-1 个 rank 通过 DSMEM 读同一份。
- 收益：**J 的 HBM 流量降到 1/C**，算术强度从 O(1) 变成 O(C)。
- N 上限不变（每 block 仍要放自己 replica 的状态 + 一个 J tile）。

**这个 kernel 是 bandwidth-bound 的，所以设计 B 的价值高于设计 A。**
上一轮我只强调了 A，这里更正。两者也可以叠：cluster 内先分 replica 再分 row。

### 8.3 先别急着写 cluster —— 有个更便宜的可能性要先测掉

现在 B 个 block 跑的是同一段代码、每步一个 `__syncthreads()`，**它们天然大致同相**，
都在同一时刻读 J 的相近区域。也就是说 **L2 可能已经在替你做设计 B 的事了**。

- GH200：L2 50 MB，N ≤ 5,000 时整个 J 能常驻 L2 → 几乎零 HBM 流量；
  N = 11.6k 时 J 是 269 MB，装不下，但 tile 级的时间局部性仍可能命中很多。
- RTX 6000 Pro：L2 更大（GB202 ~128 MB），同相效应可能更明显。

**所以顺序应该是：**

```
ncu --set full --section MemoryWorkloadAnalysis ./app
# 看三个数：
#   dram__bytes.sum                     实际 HBM 流量
#   lts__t_sector_hit_rate.pct          L2 命中率
#   实测流量 / (2*N*N*B*iters)          ← 这个比值就是 L2 帮你省下的倍数
```

如果 L2 命中率已经很高，设计 B 的收益就有限，论文该转向别的论点；
如果命中率低（说明 block 之间已经跑散相），那么**先试一个零成本的办法**：
让所有 block 从不同的 row offset 起步改成从相同 offset 起步、或在 J 的列循环上分 tile 并
在 tile 边界加一次 `__syncthreads()`，把 block 之间"箍"在同一相位 —— 这在任何架构上都能用，
不需要 cluster，也不需要 sm_90。

**这条可能是整篇论文性价比最高的一格消融，而且 RTX 机器上也能做。**

### 8.4 建议的决策顺序

1. 两台机器都跑 `gpu_probe.cu` —— 拿到真实的 optin 大小、cluster 支持、DSMEM 是否可用。
2. 做 footprint 优化（20N → 9N）。**无条件先做，两个平台都受益，改动 30 行。**
3. 用 `ncu` 量 L2 命中率，判断 J 的重复读取到底有多严重。
4. 试"相位对齐"（8.3），零架构依赖。
5. 若 3 显示 L2 帮不上忙、且 GH200 上 DSMEM 实测可用 → 做设计 B，
   作为论文里 "Hopper-specific optimization" 一节。
6. 设计 A（放大 N）只在实验 2 的表显示"FastHare 之后仍有实例卡在 N 上限之上"时才值得做。

**注意第 5 步的一个风险**：如果 DSMEM 在 sm_120 上不可用，那么这节内容就只能在 2× GH200 上跑，
而论文的平台对比（实验 5）会变成"GH200 有一个 RTX 跑不了的优化" —— 这其实是个**好的**论文结构
（把 GH200 一节从"第二平台评估"升级成"Hopper 独有特性带来的优化"），
比原计划的 negative-result 版本更有分量。但前提是 DSMEM 真的在 GH200 上跑通且有实测收益。
