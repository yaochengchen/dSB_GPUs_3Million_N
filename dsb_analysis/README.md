# dSB-GPU 结果分析与出图

只需要一个 `--root`：指向 `dsb-gpu_v5` 仓库目录，所有运行结果都在它的 `results/` 下（benchmark 的 `run_*`、`dense_scaling_*`、`scaling_sparse_*`、`bit_multi_*`、`bit_mattis_*`）。`--root` 默认就是 `../dsb-gpu_v5`。

```bash
pip install -r requirements.txt          # pandas matplotlib numpy tabulate
python run_all.py --root ../dsb-gpu_v5 --out out
```

脚本遇到"有 `.done` 但读不到行"的目录会打印 WARNING（0 字节文件通常是复制没完成，去 GH200 上重新拷）；整个套件都没有结果行的目录（如 `run_K2000_result_v6_supp_B*`，三个全失败）也会提示并自动跳过。

跑完 `out/` 里有：

| 路径 | 内容 |
|---|---|
| `out/figures/*.pdf` `*.png` | 全部图（PDF 进论文，PNG 看） |
| `out/tables/benchmark_summary.csv` | 每个 (家族, 实例, B, K, 路径) 一行：中位 µs/step、solver/wall 时间、best/median 目标值、成功数、gap、TTS99、对 SB 的加速比 |
| `out/tables/fastest_path_K3200.csv` | 每个实例 K=3200 时最快的 dsb 路径及其加速比 |
| `out/tables/trajectory_identity.csv` | 各路径与 gemm-TF32 逐 seed 结果相同的比例（命题"三元耦合下精确"的实证） |
| `out/tables/gset_agents_sweep.csv` | 六个大 G-set 实例 B=512…8192 的扫描 |
| `out/tables/gset_library_sweep.csv` | 66 个 G-set 实例，csr-row vs SB 2.0.0 库默认参数 |
| `out/tables/k2000_all_batches.csv` | K2000 B=16/64/128/512 |
| `out/tables/dense_scaling_summary.csv` `sparse_scaling_summary.csv` | 大规模 scaling 汇总 |
| `out/tables/bit_multi_summary.csv` | 双卡 ±1 bit 路径：每个 (实例类型, n, 卡数, 放置) 的矩阵切分、每步时间、带宽，Mattis 命中数 |
| `out/tables/provenance.csv` | 每个配置来自哪个目录、开跑时间 |
| `out/tables/*.tex` | booktabs 表格片段，可直接 `\input` |
| `out/summary.md` | 论文里要引用的关键数字 |

**重合数据以最新为准**：
- benchmark（K2000/G-set/QPLIB）：同一 (实例, B, K, 路径) 出现在多个运行目录时，取 `environment.txt` 第一行记录的开跑时间最晚的那次；没有时间戳的目录（agents 扫描）退回按目录名版本号 `vN`、`supp` 排序（`load.suite_rank`）。例如 G66 的 INT8 和对应 SB 取 `run_Gset_int8_v7`（v5_supp 那次 363 µs/step 是受干扰的数，v7 重跑是 98 µs），G43–G64 的 INT8 取 `run_Gset_int8_v6_supplement`，K2000 小 B 取 `run_K2000_result_v7_B*`。
- 稠密 scaling（没有时间戳）：按 (实现, 精度, B, n, 放置方式) 去重，放置方式分 `auto`（放得下进 HBM，否则整个矩阵进 Grace）和 `hybrid`（HBM 先填满，剩下的行进 Grace）；同一 key 取版本号最大的**已完成**结果，都失败才保留最新的失败记录。所以 Grace 取 `dense_scaling_v9_grace_*`，hybrid 取 `dense_scaling_v9_hybrid_*`。
- 双卡 bit（目录名带时间戳）：按 (实例类型, n, 卡数, seed) 取最新目录；只读 `status=completed` 的行。
- `*.csv.tmp`（中断的半截文件）不读。

其它输出：`tab_capacity.tex`（容量表，HBM/Grace/PyTorch 实测列由数据生成；"bit, dense ±1" 一行的实测列来自双卡 bit 运行，带 † 号）、`tab_grace.tex`（超出 HBM 后 Grace-only 与 hybrid 的每步时间、带宽、加速比）、`tab_bit_multi.tex`（双卡 ±1 bit：n=50 万–300 万的矩阵切分、每步时间、带宽，以及 Mattis 校验 3/3）。`summary.md` 里有 hybrid 的加性模型 $T=M_H/BW_H+M_G/BW_G$ 对比（gemm 误差 ≤4%，双卡 bit 误差 ≤2%），以及 Mattis 命中汇总。

## 图与论文位置

| 文件 | 论文里 | 内容 |
|---|---|---|
| `figS_trajectory_identity` | Fig. 1（§7.1） | 各路径与 TF32 逐 seed 相同比例的热图 |
| `fig1_k2000_dense` | Fig. 2（§7.2） | K2000：各路径时间 vs 步数；K=3200 最终 cut 分布 |
| `figS_k2000_all_paths` | Fig. 3（§7.2） | K2000 全部 15 条路径每步时间 |
| `fig6_gset_qplib` | Fig. 4（§7.3–7.4） | 16 个 G-set + 19 个 QPLIB：最快路径加速比与解质量 |
| `fig7_gset66_library_baseline` | Fig. 5（§7.5） | 66 个 G-set，csr-row vs SB 库默认参数 |
| `fig5_path_selection` | Fig. 6（§7.7） | 每步时间 vs n；(n, 平均度) 平面上最快路径 |
| `fig4_sensitivity` | Fig. 7（§7.8） | B 扫描（时间线性、gap 微降）+ K2000 成功率 vs K |
| `figS_k2000_batch` | Fig. 8（§7.8） | K2000 B=16/64/128/512：bit 在 B≤64 最快 |
| `fig3_sparse_scaling` | Fig. 9（§7.9） | d=100 二部图 CSR scaling：时间、内存、吞吐 |
| `fig8_dense_scaling` | Fig. 10（§7.10） | 稠密 scaling：HBM、Grace-only（黑边）、hybrid（空心菱形），带宽 + 加性模型 |
| `fig9_bit_multi` | Fig. 11（§7.11） | 双卡 ±1 bit 路径：n=50 万–300 万每步时间与带宽，Mattis 点，两层带宽模型 |

表：`tab_capacity`（表 2）、`tab_k2000_K3200`（表 6）、`tab_k2000_steps`（表 7）、`tab_gset_instances`（表 8）、`tab_qplib_instances`（表 9）、`tab_arith_ratio`（表 12）、`tab_grace`（表 13）、`tab_bit_multi`（表 14）。

## 代码结构

```
run_all.py            入口：加载 → 汇总 → 出图 → 出表 → summary.md
dsbplot/load.py       读 raw 行：results/run_*（优先读每个目录里的 dsb_gpu_*.csv / public_*.csv，从文件名恢复
                      auto-> 前缀）、results/dense_scaling*、results/scaling_sparse*、results/bit_m*_*
dsbplot/summarize.py  去重（同一 (实例,B,K,路径) 以最新为准）、按配置取中位数、TTS99、加速比、逐 seed 一致性
dsbplot/figures.py    每张图一个函数；bit_summary / bit_link_bandwidths 是双卡 bit 的汇总
dsbplot/tables.py     LaTeX 片段
dsbplot/style.py      路径 → 颜色/标记/显示名
```

约定：`label` 列统一了路径名：`gemm-tf32`（默认 gemm fp32，走 TF32）、`gemm-fp32`（`--no-tf32`）、`gemm-fp16`、`gemm-int8`、`bit`、`block`、`csr-row`、`csr-block`、`auto`、`public-matched`（SB 2.0.0 对齐调度）、`public-library`（SB 2.0.0 库默认）。成功判定统一用 `objective >= target`（agents 扫描的 CSV 里 target 列是空的，从同实例其他行补）。

## manuscript/

`manuscript/main.tex` 是 v6 稿（在 v5 基础上加入双卡 ±1 bit 路径：§4.5 的多卡实现说明、§5.3 容量表实测列、§6 配置与实例、新的 §7.11 结果小节（表 + 图）、讨论/局限/结论/摘要同步）。重新出图/出表后：

```bash
cp out/figures/*.pdf manuscript/figures/ && cp out/tables/*.tex manuscript/tables/
cd manuscript && pdflatex main.tex && pdflatex main.tex
```
