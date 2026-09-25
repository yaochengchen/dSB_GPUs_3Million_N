# dSB_GPU 代码审查（2026-09-19）

对上传的 `dSB_GPU-main.zip` 全量代码的逐文件审查。结论按「必须修 / 应该修 / 论文里要说清楚」分层。

---

## 1. 仓库里到底有几条实现路径

| 文件 | 语言 | J 格式 | 同步方式 | 状态 |
|---|---|---|---|---|
| `dSB_original_python.py` (`CDSB.update`) | PyTorch | sparse CSR | 每步一次 `torch.sparse.mm` + 若干 elementwise kernel | 参考实现 / baseline |
| `optimized_kernel.py` | CUDA 片段（未编译） | CSR | `__syncthreads()`，但 row 跨 block | **不正确**，只是草稿 |
| `cooperative_batched_fused_code.py` | CUDA + `load_inline` | CSR | `grid.sync()`（cooperative launch） | V3 |
| `MultiRowSharedFused.py` | CUDA + `load_inline` | CSR | block 内 `__syncthreads()`，x/y 驻留 SMEM | V4-sparse |
| `cdsb_fastshare_original.cu` | 纯 CUDA | dense FP32 | block 内，SMEM 2N floats，thread-per-row | V5-fp32 |
| `cdsb_fastshare.cu` | 纯 CUDA | dense FP16 | block 内，SMEM 5N floats，warp-per-row | **V6，主线** |

消融实验（paper-plan 实验 1）的四级阶梯是现成的：PyTorch → grid.sync → SMEM-sparse → SMEM-dense-FP16。这点很好，不用补代码。

`dSB_fasthare_qplib.cpp` / `cdsb_fasthare_qplib_pybind.cpp` 是走 pybind 调 Python CDSB 的旧路径；`cdsb_fasthare_qplib.cpp` 是纯 C++/CUDA 的新 driver（`-DUSE_CDSB=1`）。

---

## 2. 必须修（会导致跑不起来或结果错）

### B1. 缺 `cudaFuncSetAttribute` —— 最严重

`cdsb_fastshare.cu:cdsb_fused_run_fp16()` 直接以 `smem = 5*N*sizeof(float)` 启动 kernel，全仓库 grep 不到任何 `cudaFuncSetAttribute`。

CUDA 的动态共享内存**默认上限是 48 KB**，不管硬件有多少。要用到 sm_90 的 227 KB，必须显式 opt-in：

```cpp
cudaFuncSetAttribute(
    cdsb_fused_dense_fastshare_kernel_fp16<BLOCK>,
    cudaFuncAttributeMaxDynamicSharedMemorySizeBytes,
    (int)smem);
```

后果：**现在这份代码在 H100 上的实际 N 上限是 `49152/20 = 2457`，不是 README 写的 11,600。** 超过就 `cudaErrorInvalidValue`。README 的「H100: ~227KB → max N ≈ 11,600」这句话，在当前代码里是假的。

`MultiRowSharedFused.py` 同样的问题，而且更隐蔽：`check_multi_row_shared_memory_support()` 查的是 `prop.sharedMemPerBlock`（返回默认的 48 KB，不是 `sharedMemPerBlockOptin` 的 227 KB），所以它会在 N > 6144 时「优雅地」退回 fallback kernel —— 而 fallback kernel 本身是错的（见 B2）。

**顺带：`__launch_bounds__(1024, 2)` 要求每 SM 驻留 2 个 block，这会把每 block 的 SMEM 硬压到 ≤114 KB。V6 的 launcher 想用满 227 KB，就必须去掉 `minBlocksPerMultiprocessor`。**

### B2. fallback / cooperative kernel 的同步是错的

`MultiRowSharedFused.py` 和 `cooperative_batched_fused_code.py` 里的 `fused_dynamics_fallback_iterations_kernel`：row 被切到多个 block（`blockIdx.x`），但循环里只有 `__syncthreads()`。跨 block 没有任何同步，第 `it+1` 步读到的 `x[col_idx*B+batch]` 是别的 block 任意进度的值 —— 竞态，结果不可复现。

这个 kernel 只有在 `gridDim.x == 1` 时才正确。`optimized_kernel.py` 的 launch 配置里那段「N>1024 时切多个 block」的注释也承认了这点但没解决。**建议：直接删掉 fallback 路径，或加 `assert(gridDim.x == 1)`。** 留着它，审稿人一眼就能挑出来。

### B3. Jacobi 退化成 Gauss–Seidel（paper-plan 已记录，这里补精确定位）

`MultiRowSharedFused.py` 第 ~60–95 行：同一个 `iter` 内，线程写 `shared_x[row] = x_val`，而别的线程在同一个 `iter` 里从 `shared_x[col_idx]` 读符号。部分 row 会读到已经更新过的值。

与 `dSB_original_python.py` 的语义（`torch.sign(self.x)` 一次性快照）不一致。

### B4. `cdsb_fastshare.cu` 的 ping-pong 其实是多余的 —— 可以白赚 40% SMEM

V6 用了 `x0,y0,x1,y1,sgn` 共 5N floats。但仔细看数据流：

- `acc` 只从 `sgn[]` 读，**不从 `x0[]` 读**；
- `x0[row]/y0[row]` 只被「拥有该 row 的 warp 的 lane 0」写；
- `sgn[]` 是在每个 iteration 开头、从 `x0` 一次性快照出来的。

所以 `x0/y0` 就地更新不会破坏 Jacobi 语义 —— 快照已经由 `sgn` 承担了。`x1/y1` 两个缓冲区是冗余的。

删掉后：`5N floats (20N B)` → `3N floats (12N B)`，N 上限 11,622 → **19,370**，而且省掉每步一次的 `x0[row]=x1[row]` 拷贝循环和一次 `__syncthreads()`。

再把 `sgn` 换成 `int8_t`（值只有 -1/0/1）：`8N + N = 9N B` → N 上限 **25,827**。

### B5. FastHare 回代时缺了 bias 节点的规范化（需要验证）

`cdsb_fasthare_qplib.cpp`：`to_standard_form_dense_upper()` 把 `h_i` 编码成第 `n` 号额外 bias 节点。

- `Flag == true`（100% 约简）分支里，代码做了 `solution[i] = last * sign[i]`，其中 `last = sign.back()` —— 即用 bias 节点的自旋做全局规范化。✅
- `Flag == false` 分支里，只做了 `solution[i] = xred[spin_map[i]] * sign[i]`，**没有乘 bias 节点的自旋**。

带 `h` 项的 Ising 能量在全局翻转下不对称，所以少这一步会让一半的实例拿到「镜像解」，`object_energy` 偏掉。两个分支的处理方式不一致本身就是信号。

建议：算完 `solution` 后取 `s_bias = solution[n]`（或 `xred[spin_map[n]] * sign[n]`），然后整体乘 `s_bias`。**跑一个 QPLIB 实例对比 Python 旧路径的 `object_energy` 就能确认。**

---

## 3. 应该修（性能 / 严谨性）

### P1. baseline 被人为拖慢了，加速比不可信

`dSB_original_python.py`：`self.p = torch.linspace(0, 1, self.n_iter)` 建在 **CPU** 上，而 `self.x` 在 GPU 上。循环里 `-(self.delta - self.p[i]) * self.x` 每步都要把一个 CPU 标量喂给 GPU 表达式 —— 每个 iteration 一次隐式 H2D + 同步。800 步就是 800 次同步。

`MultiRowSharedFused.py:compare_with_traditional()` 更严重：

```python
for i in range(n_iterations):
    ...
    J_csr = J.to_sparse_csr()      # ← 每步重新做一次 COO→CSR 转换！
    sparse_term = torch.sparse.mm(J_csr, sign_x)
    y2 = y2 + (-(delta - p_array[i].item()) * x2 + ...)   # ← .item() 强制同步
```

这两条会把 baseline 拖慢几倍到一个数量级。**论文里报这种加速比会被直接拍回来。** 修法：`p_array` 放 GPU、循环外做一次 `to_sparse_csr()`、用 `p_array[i]`（零维 GPU tensor）而不是 `.item()`。

顺带 `p_array = torch.randn(n_iterations)` 也不是真实的线性 pumping schedule，benchmark 应该用 `linspace(0,1,n_iter)`。

### P2. 数值验证的判据是错的

`compare_with_traditional()` 里比较 Gauss–Seidel kernel 和 Jacobi PyTorch 的输出，差异必然很大，然后打印「⚠️ 数值有轻微差异（在GPU并行计算中是正常的）」。这掩盖了 B3。修完 B3 之后，`1e-4` 的判据在 FP32 下是合理的；FP16 存储版要放宽到 `1e-2` 量级，并且**应该比较最终能量而不是逐元素 x**。

### P3. FP16 存 J 的动态范围风险

`pack_J_to_half_()` 把 `double` 直接 `__float2half_rn`。FP16 最大 65504，尾数 11 bit。

QPLIB driver 里 `input_J = -J / scale_factor`（`scale_factor = max|J|`）已经把 J 归一到 [-1,1]，所以溢出不是问题；但 **1e-4 量级的小耦合会在 FP16 下损失一半有效位**，subnormal 以下（<6e-8）直接变 0。

论文实验 4（精度对比）要做的就是这个：报 FP16 存储 vs FP32 全精度在 QPLIB/G-set 上的**解质量差**，而不是 x 的逐元素误差。另外 `auto_set_xi_from_J_()` 是从**已经量化过的** `hJ_` 算 `sum(J²)` 的，和 Python 参考从原始 double 算的 `xi` 会有微小差别 —— 要么统一，要么在论文里说明。

### P4. dense J 的访存模式没有向量化

`for (int col = lane; col < N; col += WARP) { __half2float(Jrow[col]); }` —— 每 lane 每次取 2 字节，一个 warp 一次事务取 64 字节（连续，所以 coalesce 是好的），但只用到了 1/2 的 128B cache line 粒度效率。

改成 `half2`（每 lane 4 字节，warp 128 字节）或 `float4`（8 个 half，warp 512 字节）能显著降低访存指令数。N 为奇数时处理尾巴即可。这是消融实验里很容易拿分的一格。

### P5. 真正的墙不是共享内存，是 J 的重复流式读取

这是整篇论文最该讲清楚的一点。

replica-per-block 的设计下，**每个 block 每个 iteration 都要把整个 J 从 global memory 读一遍**：

- 每 block 每步：`2N²` 字节（FP16）
- 总量：`2N² × B × n_iter`

N = 11,600 时 J 是 269 MB，H100 的 L2 只有 50 MB —— **J 放不进 L2，每步都真的走 HBM**。800 步 × 269 MB ≈ 215 GB 的 HBM 流量，还要乘 batch 的波数。

算术强度是 **O(1)**：每读一个 J 元素只做一次 FMA，且这个 FMA 的另一个操作数是 ±1。这个 kernel 在 N 大的时候是彻头彻尾的 bandwidth-bound，和共享内存够不够无关。

这直接决定了 GH200 那一节该怎么写（见下）。

### P6. 其他小问题

- `gpu_state_dirty_` 从来没被置为 `true`，`upload_x_if_dirty_()` 是死代码。如果用户改了 host 端的 `solver.x` 再调 `calc_energy()`，会静默用旧的 GPU 数据。
- `calc_energy()` 每次调用都 `cudaMalloc` + `cudaFree` 一个 B×double。
- energy kernel 里 `atomicAdd(&block_sum, sum)` 用的是 double 原子加 —— 正确，但每 warp 一次原子操作，N 大时可以先做 warp 内归约再做一次 block 级 shuffle 树。
- `cdsb_fasthare_qplib.cpp` 用 `std::clock()` 计时 —— 那是 CPU time，多线程/GPU 场景下没意义。换 `std::chrono::steady_clock`。
- `MultiRowSharedFused.py` 的 `extra_cuda_cflags` 只编到 `sm_86`，**没有 sm_90**。在 H100 上会走 PTX JIT，性能和 occupancy 都不代表 native。benchmark 前必须加 `-gencode=arch=compute_90,code=sm_90`。
- `MultiRowSharedMemoryFusedJIT._compilation_attempted` 是类变量且第一次失败后永不重试，静默退回 PyTorch —— benchmark 时可能在不知情的情况下测的是 PyTorch。
- 术语：全仓库的 "CDSB" 应为 Goto 2021 的 **dSB**（discrete SB）。README 里「CDSB dynamics: Continuous-time dynamical system」这句是错的，dSB 恰恰是把 `x` 离散成 `sign(x)` 的那个变体。

---

## 4. 修复的优先顺序

1. **B1**（加 `cudaFuncSetAttribute`，去掉 `__launch_bounds__` 的第二参数）—— 不修的话 N > 2457 根本跑不了，实验 2 那张表做不出来。
2. **B4**（删 `x1/y1`，`sgn` 改 int8）—— 顺手把 N 上限推到 ~25k，改动量很小。
3. **B3 + P2**（统一 Jacobi + 真正的一致性验证）。
4. **B5**（bias 规范化，跑一个实例对比旧 Python 路径）。
5. **P1**（修 baseline，重新测加速比）。
6. **P4 / P6**（向量化、sm_90 编译标志）。
