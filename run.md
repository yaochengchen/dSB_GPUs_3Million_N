# dsb-gpu v5 运行说明

所有命令都在仓库根目录（GH200 上是 `~/software/dsb-gpu_v5`）执行。本文件是"怎么跑、结果放哪、论文用了哪些"的唯一说明；代码结构和各条执行路径的原理见 `README.md`，实验协议见 `docs/`。

## 0. 结果布局

所有结果都在 `results/` 下，`dsb_analysis` 只需要这一个根目录（`python run_all.py --root ../dsb-gpu_v5`）。目录含义见 `results/README.md`，概括如下：

| 目录 | 产生脚本 | 内容 |
|---|---|---|
| `results/run_K2000_result_v5`、`run_Gset_result_v5(_supp)`、`run_qplib_result_v5(_supp)` | `run_K2000.sh` / `run_Gset.sh` / `run_qplib.sh` | 主 benchmark：全部 variant × 步数 200–3200，对齐调度的 SB 2.0.0 baseline |
| `results/run_Gset_int8_v6_supplement`、`run_Gset_int8_v7` | `run_Gset.sh` | 大 G-set 实例的 `gemm:int8` 补跑（v7 是 G66 受干扰后的重跑） |
| `results/run_K2000_result_v7_B{16,64,128}` | `run_K2000.sh` | K2000 小 batch 扫描 |
| `results/run_Gset_agents_v5_supp` | 手写循环（§3.4） | 六个大 G-set 实例 B=1024–8192 |
| `results/run_Gset_library_v5_supp`、`run_qplib_library_v5_supp` | `run_Gset.sh` / `run_qplib.sh` | SB 2.0.0 库默认参数对照，K=800 |
| `results/scaling_sparse_v5_s{200,400}` | `scripts/run_scaling.sh` + 手动 baseline | d=100 二部图稀疏 scaling |
| `results/dense_scaling_v5`、`dense_scaling_v9_{grace,hybrid}_{fp16,fp32}` | `scripts/run_dense_scaling.sh` | 稠密 scaling：HBM、Grace-only、hybrid |
| `results/bit_multi_*`、`bit_mattis_*` | `scripts/run_bit_multi.sh` / `run_bit_mattis.sh` | 双卡 ±1 bit 路径到 n=300 万，及 Mattis 正确性校验 |

三个 benchmark 脚本的 `RESULT_ROOT` 默认已经是 `results/run_*_result`；scaling 脚本的 `OUT`/`OUT_DIR` 默认也在 `results/` 下。脚本会跳过带 `.done` 的目录，所以指向已有目录只会补没跑完的 (实例, steps)。

## 1. 编译、自检

```bash
make clean && make -j                          # solve_qplib solve_gset export_fasthare dense_scaling selftest bench bit_dense_multi
make probe                                     # gpu_probe
pip install -r requirements-baseline.txt       # simulated-bifurcation==2.0.0 + scipy（先装好 CUDA 版 PyTorch）
./gpu_probe                                    # shared memory / cluster / DSMEM 的真实上限
./selftest 512 4 50 && ./selftest 2048 8 50    # bit 的差值应为 0.000e+00
python3 -m unittest discover -s tests          # 10 个测试
./solve_gset --help | grep -- --dt             # 确认 --dt / --no-tf32 编进去了
```

需要 CUDA 12.0+（thread block cluster）、cuBLAS、Eigen 3；`bit_dense_multi` 单独用 `-fmad=false` 编译（见 Makefile），否则 `--check` 对不上主机参考。

## 2. 通用约定

**variant 写法** `variant[:precision[:notf32]]`，不写 precision 就是 fp32。各套件固定用下面的集合（已定，不要再改，否则新老结果对不上）：

```bash
V_GSET="auto bit csr-row csr-block gemm gemm:fp32:notf32 gemm:int8 gemm:fp16"
V_QPLIB="auto bit csr-row csr-block block gemm gemm:fp32:notf32 gemm:int8 gemm:fp16"
V_K2000="auto bit csr-row block gemm gemm:fp32:notf32 gemm:int8 gemm:fp16"
STEPS_ALL="200 400 800 1600 3200"
GSET_IDS="1 11 22 32 43 48 55 60 63 70 72 81"; GSET_NEW="58 64 66 77"
QPLIB_IDS="3506 3565 3642 3650 3693 3705 3706 3738 3745 3822 3832 3838 3850 3852 3877 5721 5725 5755 5875"
```

不跑的 variant 及原因（K2000 / G1 / G11 的 µs/step，SB-2.0.0 作参照 169 / 175 / 167）：

| variant | K2000 | G1 | G11 | 原因 |
|---|---:|---:|---:|---|
| `cluster`、`cluster:fp16` | 2090 / 3376 | 346 / 522 | 346 / 522 | 三个实例都比 SB 慢 |
| `global-sync`、`global-sync:fp16` | 9162 / 9527 | 1402 / 1411 | 1400 / 1434 | 同上，诊断用 |
| `csr-cluster` | 9803 | 196 | 115 | K2000、G1 比 SB 慢；G11 上 csr-block 是 3.1 |
| `block:fp16` | 650 | 156 | 156 | 只比 SB 快 5–7%，还比 fp32 的 block 慢 |

**时间步** `DT=auto` 让 `python/suggest_dt.py` 按实例算 `dt = clip(√(1.2/k), 0.25, 1.25)`，`k = ξ·|λ_min(J)|`，并把同一个 dt 传给 C++（`--dt`）和 baseline（`--sb-time-step`）。参考值：QPLIB ≈1.2，G-set 环面 1.15，稀疏 G55–66 0.9，中密度 G22–47 0.7，稠密 G1–10 0.55，K2000 1.1，d=100 二部图 0.5。`dt² k > 约 2.7` 会发散（v4 里 G1 objective=29 就是这个）。ξ 保持 `0.5·√(N−1)/‖J‖_F`。

**baseline 口径** `SB_SCHEDULES=matched`（默认）= 与 dsb-gpu 相同的 dt、`p_k = k/(K−1)`、初值 uniform(−0.01, 0.01)，CSV 里记为 `discrete-matched`；`library` = SB 2.0.0 原默认（dt=0.1，pump `min(k/1000,1)`，初值 uniform(−1,1)），记为 `discrete`。`public:fp16:*` 包本身不支持，每次报 error，可忽略。`REDUCTION_MODES` 只跑 0（FastHare 不再使用）。

**GPU** 脚本默认用物理 GPU 1（`GPU_ID=1`）。同一张卡只跑一个脚本——v5 第一轮有一段计时被另一个进程干扰，表现为整数倍变慢（bit 21.8 → 46 或 69 µs，SB 175 → 2396 µs）。开跑前确认：

```bash
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv   # 除表头外应为空
```

跑完可以用 `python3 python/check_contamination.py` 扫一遍 `results/`，它把同一目录里 bit 每步时间 max/min > 1.05 的文件列出来。

**失败处理** 每个 variant × 精度、每个 baseline 口径都是独立进程，外面套 `timeout`（`VARIANT_TIMEOUT_S`，默认 1800 s）。状态在各目录 `run_status.csv` 和 `<RESULT_ROOT>/run_status_all.csv`：ok、timeout、oom、killed、does-not-fit、unsupported、error。

## 3. Benchmark（K2000、G-set、QPLIB）

### 3.1 主套件（论文表 6–9、12，图 1–4、6）

```bash
DT=auto RESULT_ROOT=results/run_K2000_result_v5 STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="$V_K2000" VARIANT_TIMEOUT_S=3600 ./run_K2000.sh
DT=auto RESULT_ROOT=results/run_Gset_result_v5  STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="$V_GSET"  ./run_Gset.sh  $GSET_IDS
DT=auto RESULT_ROOT=results/run_Gset_result_v5_supp STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="$V_GSET" ./run_Gset.sh $GSET_NEW
DT=auto RESULT_ROOT=results/run_qplib_result_v5 STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="$V_QPLIB" ./run_qplib.sh $QPLIB_IDS
DT=auto RESULT_ROOT=results/run_qplib_result_v5_supp STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="auto bit gemm:fp32:notf32" ./run_qplib.sh $QPLIB_IDS
```

- K2000：3200 步的 csr-block 约 35 分钟，所以 timeout 提到 3600 s。
- G-set 大实例上 `block` 每步都把稠密 J 从 HBM 读一遍（G70/G72/G81 会跑满 1800 s 记成 timeout，其他 variant 照常）。
- QPLIB 的 baseline 只支持 float32；J 不是三值的实例上 `bit` 和 `gemm:int8` 记成 unsupported。
- 每个实例实际用的 dt 在 `<实例>_<B>_<K>/environment.txt` 里（`public_sb_schedules=... dt=...`）。

### 3.2 INT8 补跑（v6/v7）

```bash
DT=auto RESULT_ROOT=results/run_Gset_int8_v6_supplement STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="gemm:int8" ./run_Gset.sh 43 48 55 58 60 63 64
DT=auto RESULT_ROOT=results/run_Gset_int8_v7 STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="gemm:int8" ./run_Gset.sh 66
```

G66 在 v5_supp 里的 INT8（363 µs/step）受干扰，v7 重跑是 98 µs；分析脚本按 `environment.txt` 的开跑时间取最新，所以 v7 自动覆盖。

### 3.3 K2000 小 batch（论文 §7.8、图 8）

```bash
for B in 16 64 128; do
  DT=auto BATCH=$B STEPS_LIST="$STEPS_ALL" CUSTOM_VARIANTS="bit gemm:fp16 gemm:int8 gemm" \
    RESULT_ROOT=results/run_K2000_result_v7_B$B ./run_K2000.sh
done
```

（`run_K2000_result_v6_supp_B*` 那次三个目录全部失败，没有结果行，已删。）

### 3.4 大 G-set 实例的 B 扫描（论文图 7A、B）

只跑 dsb-gpu，不跑 baseline，用 `solve_gset` 直接循环：

```bash
GSET_BIG="55 60 63 70 72 81"
for B in 1024 2048 4096 8192; do for id in $GSET_BIG; do
  f=data/Gset/data/G$id; dt=$(python3 python/suggest_dt.py $f)
  for s in $STEPS_ALL; do
    out=results/run_Gset_agents_v5_supp/G${id}_${B}_${s}; mkdir -p $out
    echo "public_sb_schedules=none dt=$dt batch=$B steps=$s repeats=50 warmup=1" > $out/environment.txt
    for var in auto csr-block; do
      o=$out/dsb_gpu_${var}-fp32_reduction0.csv
      [ -s $o ] || CUDA_VISIBLE_DEVICES=0 timeout 3600 ./solve_gset --csv --variant=$var --precision=fp32 \
        --no-reduction --batch=$B --steps=$s --seed=12345 --repeats=50 --warmup=1 --dt=$dt $f > $o 2> ${o%.csv}.stderr
    done
  done
done; done
```

### 3.5 库默认参数对照（论文 §7.5、图 5）

66 个 G-set 实例（n ≤ 20000）用 csr-row，19 个 QPLIB 用 csr-block，K=800：

```bash
DT=auto SB_SCHEDULES=library RESULT_ROOT=results/run_Gset_library_v5_supp  STEPS_LIST="800" CUSTOM_VARIANTS="csr-row"   ./run_Gset.sh
DT=auto SB_SCHEDULES=library RESULT_ROOT=results/run_qplib_library_v5_supp STEPS_LIST="800" CUSTOM_VARIANTS="csr-block" ./run_qplib.sh $QPLIB_IDS
```

## 4. 大稀疏图 scaling（论文 §7.9、图 9）

只跑 `csr-row`（d=100 正则二部图，最优值 = N·d/2），steps 200 和 400；`DT` 默认 auto，这些图用 0.49。

```bash
SPARSE_SIZES="10000 20000 50000 100000 200000"
for s in 200 400; do
  SIZES="$SPARSE_SIZES" BATCH=200 STEPS=$s OUT_DIR=results/scaling_sparse_v5_s$s scripts/run_scaling.sh
  for n in 10000 20000 50000; do                       # PyTorch baseline 只有放得下的尺寸；N≥100k 稠密 J OOM，本身就是数据
    f=data/scaling/bipartite_N${n}_d100_seed42.txt; dt=$(python3 python/suggest_dt.py "$f")
    python3 python/benchmark_public_gset.py "$f" --agents=200 --steps=$s --repeats=10 --warmup=1 --dtype=float32 \
      --sb-schedule=matched --sb-time-step="$dt" --target=$((n*50)) \
      --output=results/scaling_sparse_v5_s$s/public_N${n}.csv 2> results/scaling_sparse_v5_s$s/public_N${n}.stderr
  done
done
```

`run_large_scale.sh` 的 `MODE=sparse` 测的是同一批图，不用再跑。

## 5. 大稠密图 scaling（论文 §7.10、图 10、表 2、13）

J 是实数值随机矩阵，只测每步时间（dt 不影响）。`auto` 在实数 J 上等于 `gemm`，所以只跑 `gemm`。

```bash
# HBM 段（PyTorch baseline 一起跑）
N_LIST="20000 80000 160000 200000 240000 300000" BATCH_LIST="1 8 64" PRECISION_LIST="fp32 fp16" \
  CPP_MODES="gemm" REPEATS=3 OUT=results/dense_scaling_v5 scripts/run_dense_scaling.sh

# 超出 HBM：Grace-only（MATRIX_MEMORY=auto，放不下就整个矩阵进 pinned Grace 内存，经 2 GiB HBM staging 流式读）
GPU_ID=1 GRACE_NUMA_NODE=1 TIMEOUT_S=3600 MATRIX_MEMORY=auto N_LIST="200000 240000 300000 400000 450000" BATCH_LIST="1 8 64" \
  PRECISION_LIST="fp16" CPP_MODES="gemm" PYTHON_BASELINE=0 REPEATS=3 OUT=results/dense_scaling_v9_grace_fp16 \
  choom -n 1000 -- scripts/run_dense_scaling.sh
GPU_ID=0 GRACE_NUMA_NODE=0 TIMEOUT_S=3600 MATRIX_MEMORY=auto N_LIST="200000 240000 300000" BATCH_LIST="1 8 64" \
  PRECISION_LIST="fp32" CPP_MODES="gemm" PYTHON_BASELINE=0 REPEATS=3 OUT=results/dense_scaling_v9_grace_fp32 \
  choom -n 1000 -- scripts/run_dense_scaling.sh

# 超出 HBM：hybrid（HBM 先填到 0.85，剩下的行进 Grace）
GPU_ID=1 GRACE_NUMA_NODE=1 TIMEOUT_S=3600 MATRIX_MEMORY=hybrid N_LIST="200000 240000 300000 400000 450000" BATCH_LIST="1 8 64" \
  PRECISION_LIST="fp16" CPP_MODES="gemm" PYTHON_BASELINE=0 REPEATS=3 OUT=results/dense_scaling_v9_hybrid_fp16 \
  choom -n 1000 -- scripts/run_dense_scaling.sh
GPU_ID=0 GRACE_NUMA_NODE=0 TIMEOUT_S=3600 MATRIX_MEMORY=hybrid N_LIST="200000 240000 300000" BATCH_LIST="1 8 64" \
  PRECISION_LIST="fp32" CPP_MODES="gemm" PYTHON_BASELINE=0 REPEATS=3 OUT=results/dense_scaling_v9_hybrid_fp32 \
  choom -n 1000 -- scripts/run_dense_scaling.sh
```

- GPU 和 NUMA 节点要配对（`nvidia-smi topo -m`、`numactl --hardware`：这台机器 GPU0↔node 0，GPU1↔node 1），`choom -n 1000` 防止 OOM killer 先杀掉它。
- 检查点：fp32 在 N≥200k、fp16 在 N≥260k 时 `grace_bytes > 0`；同一 N 下 B 从 1 到 64 每步时间几乎不变；Grace-only 段各 n 都是 371 GB/s；hybrid 的每步时间应满足 `T = M_H/4.2 TB/s + M_G/371 GB/s`（误差 ≤4%）。
- `dense_scaling_v5_supp_grace*`、`v7_grace_fp16`、`v8_*` 是 v9 之前的 Grace 版本（managed memory 只有 80–91 GB/s，以及 staging 的第一版），保留作记录，分析脚本按版本号取 v9。

## 6. 双卡 ±1 bit 路径：到 300 万变量（论文 §7.11、图 11、表 14）

`apps/bit_dense_multi.cu`：稠密 ±1 的 J 每个耦合 1 bit，行按卡切分，每张卡先填满自己的 HBM（0.85 × free），剩下的行放本卡 NUMA 节点上的 pinned Grace 内存，每步经 2 GiB HBM staging 读一遍；每步结束两卡通过 NVLink 交换各自的符号位图（n/8 字节）。batch=1。乘积是整数 popcount，更新用显式 round-to-nearest、不做 FMA 合并，所以结果与卡数、放置方式无关。`--instance mattis` 生成 `J_ij = ξ_i ξ_j`（隐藏的随机 ξ），基态能量恰为 `−n(n−1)/2`。

```bash
# 1. 正确性：单卡、双卡、强制把大部分行放进 Grace，三个都应打印 PASS 且校验和相同（n ≤ 50000 才做主机参考）
#    （2026-09-25 三个都 PASS，校验和一致，已写进论文 §4.5）
./bit_dense_multi --n 8192 --gpus 1 --steps 100 --check
./bit_dense_multi --n 8192 --gpus 2 --steps 100 --check
./bit_dense_multi --n 8192 --gpus 2 --steps 100 --check --hbm-rows 1024

# 2. 只看 300 万变量的内存规划，不分配
./bit_dense_multi --n 3000000 --gpus 2 --dry-run

# 3. 速度扫描（随机 ±1 J，K=50，每个 n 3 次重复），结果在 results/bit_multi_<时间戳>/raw.csv
GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" scripts/run_bit_multi.sh

# 4. Mattis 校验（每个 n 三个 seed，每行必须 PASS，否则脚本非零退出），结果在 results/bit_mattis_<时间戳>/raw.csv
GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" scripts/run_bit_mattis.sh
```

2026-09-25 的结果（`bit_multi_20260925T144444Z`、`bit_mattis_20260925T150135Z`）：

| n | 矩阵 | 放置（HBM + Grace） | s/step | 聚合带宽 | Mattis |
|---:|---:|---|---:|---:|---|
| 500 000 | 31 GB | HBM | 0.0035 | 8.95 TB/s | 3/3 PASS |
| 1 000 000 | 125 GB | HBM | 0.0139 | 8.99 TB/s | 3/3 PASS |
| 2 000 000 | 500 GB | 254 + 246 GB | 0.407 | 1.23 TB/s | 3/3 PASS |
| 2 500 000 | 781 GB | 254 + 527 GB | 0.844 | 0.93 TB/s | 3/3 PASS |
| 3 000 000 | 1125 GB | 254 + 871 GB | 1.40 | 0.80 TB/s | 3/3 PASS |

- n=300 万时每步：HBM 行 28.5 ms（4.46 TB/s/卡），Grace 行 1373 ms（GPU0，317 GB/s）/ 1236 ms（GPU1，352 GB/s），更新 0.01 ms，位图交换 0.1 ms；GPU1 等 GPU0 138 ms。步长由较慢的那条 Grace 链路决定。生成加 pin 内存 81 s，不计入。
- 两层模型 `T = M_H/8.97 TB/s + M_G/646 GB/s` 对三个 hybrid 点误差 ≤2%。
- Mattis 全部 PASS 说明双卡 + HBM/Grace 这条完整数据路径（分块暂存、位图交换、能量计算）在 300 万变量上是对的；Mattis 本身规范等价于铁磁体，很容易，只证明实现正确，不说明 dSB 在难题上的求解能力，论文里也是这样写的。
- 同一个 seed 重复跑结果完全一样（动力学确定），所以 `run_bit_mattis.sh` 默认 `REPEATS=1`，要统计就多给 seed：`SEEDS="$(seq 1 20)"`。n=300 万每个 seed 约 2–4 分钟，n ≤ 200 万不到 1 分钟。
- 想要难一点的植入实例，把 `ξ_i ξ_j` 按概率 p 翻转符号即可（`jbit()` 里加一行）；batch>1（位图和 popcount 按 replica 各算一份）只有做 TTS 统计时才值得实现。

## 7. 出图出表

```bash
cd ../dsb_analysis
pip install -r requirements.txt
python run_all.py --root ../dsb-gpu_v5 --out out      # 图在 out/figures，表在 out/tables，关键数字在 out/summary.md
cp out/figures/*.pdf manuscript/figures/ && cp out/tables/*.tex manuscript/tables/
cd manuscript && pdflatex main.tex && pdflatex main.tex
```

重复配置以最新一次为准（benchmark 按 `environment.txt` 的开跑时间，稠密 scaling 按目录版本号，双卡 bit 按目录时间戳）；来源记录在 `out/tables/provenance.csv` 和 `out/summary.md` 末尾。目录里出现 0 字节的 csv 时脚本会打 WARNING，通常是复制没完成，从 GH200 重新拷。

## 8. 跑完需要发回的文件

- benchmark：每个实例目录的 `dsb_gpu_*_reduction0.csv`、`public_*_reduction0.csv`、`environment.txt`，以及 `run_settings.tsv`、`run_status_all.csv`（`summary*.csv` 是脚本自己的汇总，分析不用）；
- 稀疏 scaling：`raw.csv`、`N*.csv`、`public_N*.csv`、`environment.txt`；
- 稠密 scaling：`raw.csv`、`environment.txt`；
- 双卡 bit：`raw.csv`、`environment.txt`、各 `n*.log`。

最省事的做法是整个 `results/` 目录打包。
