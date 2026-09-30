# Key numbers (auto-generated)

## K2000 (n=2000, B=512, K=3200, 50 trials)

| path | us/step | speedup vs SB (integration) | speedup (wall) | u/M | median | TTS99 solver (s) |
|---|---|---|---|---|---|---|
| gemm-fp16 | 12.4 | 13.64 | 4.39 | 31/50 | 33337.0 | 0.199 |
| gemm-int8 | 16.0 | 10.59 | 3.30 | 34/50 | 33337.0 | 0.256 |
| gemm-tf32 | 26.7 | 6.34 | 3.35 | 34/50 | 33337.0 | 0.428 |
| bit | 30.1 | 5.63 | 3.80 | 34/50 | 33337.0 | 0.482 |
| auto | 30.1 | 5.63 | 3.80 | 34/50 | 33337.0 | 0.482 |
| gemm-fp32 | 112.4 | 1.51 | 1.28 | 34/50 | 33337.0 | 1.8 |
| public-matched | 169.5 | 1.00 | 1.00 | 9/50 | 33308.0 | 13 |
| block | 770.7 | 0.22 | 0.22 | 34/50 | 33337.0 | 12.3 |
| csr-row | 1298.6 | 0.13 | 0.13 | 27/50 | 33337.0 | 24.9 |

Fraction of seeds bitwise-identical to gemm TF32 at K=3200: auto=1.00, bit=1.00, block=1.00, csr-row=0.52, gemm-fp16=0.42, gemm-fp32=1.00, gemm-int8=1.00, gemm-tf32=1.00, public-matched=0.10

## G-set (K=3200): fastest path per instance vs matched SB 2.0.0

- instances: 16; fastest path counts: {'csr-block': 8, 'gemm-fp16': 4, 'csr-row': 2, 'gemm-int8': 2}
- integration speedup: min 13.5, median 26.9, max 136.7
- wall speedup: min 3.8, median 22.2, max 115.2
- median gap: dsb better on 10, worse on 1, tie on 5 instances
- instances where dsb reaches best-known in >=1 trial: 5; SB: 5
- relaxed target (tightest of 100/99.9/99.5/99 % reached by both): levels {100.0: 5, 99.5: 4, 99.0: 7}; dsb reaches it more often on 6, less often on 1, equally on 9; TTS99 ratio SB/dsb where both resolved: min 5.1, median 26.4, max 224.0
| family   | instance   |   level_pct |   u_dsb |   u_sb |   tts_dsb_s |   tts_sb_s |   tts_ratio |
|:---------|:-----------|------------:|--------:|-------:|------------:|-----------:|------------:|
| G-set    | G1         |       100   |      50 |     50 |      0.0205 |     0.5598 |     27.3173 |
| G-set    | G11        |       100   |      50 |     38 |      0.0099 |     2.2214 |    223.971  |
| G-set    | G22        |       100   |      13 |      5 |      0.6318 |    23.3934 |     37.0282 |
| G-set    | G32        |        99   |      50 |     44 |      0.0222 |     1.5966 |     71.9906 |
| G-set    | G43        |       100   |      50 |     50 |      0.0233 |     0.6139 |     26.2977 |
| G-set    | G48        |       100   |      50 |     50 |      0.0308 |     0.9281 |     30.0956 |
| G-set    | G55        |        99.5 |      50 |     50 |      0.0932 |     2.1192 |     22.7465 |
| G-set    | G58        |        99.5 |      50 |     50 |      0.1363 |     2.1212 |     15.5618 |
| G-set    | G60        |        99.5 |      50 |     50 |      0.1303 |     3.4445 |     26.4446 |
| G-set    | G63        |        99.5 |      50 |     50 |      0.2247 |     3.4541 |     15.3723 |
| G-set    | G64        |        99   |      42 |     50 |      0.6774 |     3.4503 |      5.0932 |
| G-set    | G66        |        99   |       8 |      0 |      2.3741 |   inf      |    inf      |
| G-set    | G70        |        99   |       0 |      0 |    inf      |   inf      |    nan      |
| G-set    | G72        |        99   |      10 |      0 |      2.0922 |   inf      |    inf      |
| G-set    | G77        |        99   |      20 |      0 |      1.4777 |   inf      |    inf      |
| G-set    | G81        |        99   |       0 |      0 |    inf      |   inf      |    nan      |

## QPLIB (K=3200): fastest path per instance vs matched SB 2.0.0

- instances: 19; fastest path counts: {'csr-block': 17, 'gemm-fp16': 2}
- integration speedup: min 17.6, median 58.4, max 85.6
- wall speedup: min 9.1, median 41.7, max 59.3
- median gap: dsb better on 4, worse on 2, tie on 13 instances
- instances where dsb reaches best-known in >=1 trial: 18; SB: 15
- relaxed target (tightest of 100/99.9/99.5/99 % reached by both): levels {100.0: 15, 99.9: 1, 99.5: 3}; dsb reaches it more often on 6, less often on 1, equally on 12; TTS99 ratio SB/dsb where both resolved: min 17.6, median 60.1, max 626.1
| family   | instance   |   level_pct |   u_dsb |   u_sb |   tts_dsb_s |   tts_sb_s |   tts_ratio |
|:---------|:-----------|------------:|--------:|-------:|------------:|-----------:|------------:|
| QPLIB    | QPLIB_3506 |       100   |      50 |     50 |      0.0129 |     0.5813 |     44.9403 |
| QPLIB    | QPLIB_3565 |       100   |      50 |     50 |      0.0139 |     0.5487 |     39.6001 |
| QPLIB    | QPLIB_3642 |       100   |      50 |     40 |      0.0075 |     1.5743 |    209.396  |
| QPLIB    | QPLIB_3650 |       100   |      25 |     26 |      0.0456 |     3.9041 |     85.5613 |
| QPLIB    | QPLIB_3693 |        99.5 |      50 |     50 |      0.0081 |     0.5259 |     64.5684 |
| QPLIB    | QPLIB_3705 |       100   |      50 |     50 |      0.0092 |     0.5533 |     60.1134 |
| QPLIB    | QPLIB_3706 |       100   |      50 |     50 |      0.0095 |     0.5528 |     58.4196 |
| QPLIB    | QPLIB_3738 |       100   |      50 |     50 |      0.0108 |     0.5575 |     51.6398 |
| QPLIB    | QPLIB_3745 |       100   |      50 |     50 |      0.0162 |     0.5235 |     32.3199 |
| QPLIB    | QPLIB_3822 |       100   |      50 |     50 |      0.0117 |     0.5187 |     44.266  |
| QPLIB    | QPLIB_3832 |       100   |      50 |     48 |      0.0076 |     1.1062 |    145.987  |
| QPLIB    | QPLIB_3838 |       100   |      48 |     17 |      0.0208 |     6.5983 |    317.692  |
| QPLIB    | QPLIB_3850 |       100   |      43 |     22 |      0.0257 |     4.3057 |    167.695  |
| QPLIB    | QPLIB_3852 |       100   |      50 |     50 |      0.012  |     0.5546 |     46.3846 |
| QPLIB    | QPLIB_3877 |       100   |      50 |     50 |      0.0085 |     0.5628 |     66.5706 |
| QPLIB    | QPLIB_5721 |        99.5 |       1 |      1 |      7.27   |   127.635  |     17.5565 |
| QPLIB    | QPLIB_5725 |        99.9 |      50 |     49 |      0.0074 |     1.1133 |    149.442  |
| QPLIB    | QPLIB_5755 |        99.5 |      43 |      7 |      0.0277 |    17.3562 |    626.137  |
| QPLIB    | QPLIB_5875 |       100   |      50 |     50 |      0.028  |     0.5682 |     20.3251 |

## G-set large instances, agents sweep (auto path, K=3200): median gap % by B

| instance   |   512 |   1024 |   2048 |   4096 |   8192 |
|:-----------|------:|-------:|-------:|-------:|-------:|
| G55        | 0.175 |  0.165 |  0.155 |  0.146 |  0.136 |
| G60        | 0.204 |  0.19  |  0.183 |  0.169 |  0.162 |
| G63        | 0.399 |  0.386 |  0.375 |  0.366 |  0.355 |
| G70        | 1.324 |  1.293 |  1.282 |  1.267 |  1.241 |
| G72        | 1.056 |  1.027 |  0.999 |  0.999 |  0.97  |
| G81        | 1.11  |  1.074 |  1.067 |  1.038 |  1.024 |

## G-set 66 instances, K=800: csr-row vs SB 2.0.0 library defaults

- integration speedup: min 4.7, median 10.8, max 77.1
- median gap lower for dsb on 64 / 66 instances; SB library reaches best-known on 3 instances, dsb on 30

## Sparse scaling (degree-100 bipartite, csr-row, B=200)

- ns per nonzero per step: 0.152-0.156; GPU memory at n=200000: 561 MB
- n=10000 K=200: SB 0.247 s vs csr-row 0.031 s -> 8.1x
- n=10000 K=400: SB 0.493 s vs csr-row 0.061 s -> 8.1x
- n=20000 K=200: SB 0.879 s vs csr-row 0.063 s -> 14.1x
- n=20000 K=400: SB 1.753 s vs csr-row 0.125 s -> 14.0x
- n=50000 K=200: SB 5.067 s vs csr-row 0.152 s -> 33.2x
- n=50000 K=400: SB 10.103 s vs csr-row 0.305 s -> 33.1x

## K2000 batch sweep (K=3200): us/step, trials at target / 50, TTS99 solver (s)

| label          |     16 |     64 |    128 |    512 |
|:---------------|-------:|-------:|-------:|-------:|
| bit            |   3.18 |   5.68 |   9.06 |  30.11 |
| gemm-fp16      |   6.97 |   6.84 |   7.47 |  12.43 |
| gemm-int8      |   8.55 |   8.8  |   8.99 |  16    |
| gemm-tf32      |  17.92 |  18.21 |  16.15 |  26.72 |
| public-matched | 175.41 | 174.59 | 168.12 | 169.51 |

| label          |   16 |   64 |   128 |   512 |
|:---------------|-----:|-----:|------:|------:|
| bit            |    2 |    4 |     8 |    34 |
| gemm-fp16      |    3 |    6 |    10 |    31 |
| gemm-int8      |    2 |    4 |     8 |    34 |
| gemm-tf32      |    2 |    4 |     8 |    34 |
| public-matched |    0 |    2 |     0 |     9 |

| label          |      16 |     64 |     128 |    512 |
|:---------------|--------:|-------:|--------:|-------:|
| bit            |   1.148 |  1.018 |   0.783 |  0.482 |
| gemm-fp16      |   1.673 |  0.81  |   0.502 |  0.199 |
| gemm-int8      |   3.093 |  1.577 |   0.777 |  0.256 |
| gemm-tf32      |   6.479 |  3.264 |   1.395 |  0.428 |
| public-matched | inf     | 63.133 | inf     | 13.018 |

## Dense scaling (dsb-gpu gemm and PyTorch, K=50): time per step and matrix bandwidth

tier: hbm = matrix in HBM, grace = whole matrix in Grace memory (staged through HBM), hybrid = HBM filled first, remaining rows in Grace memory

|                                       |   ('GBps', 1) |   ('GBps', 8) |   ('GBps', 64) |   ('ms', 1) |   ('ms', 8) |   ('ms', 64) |
|:--------------------------------------|--------------:|--------------:|---------------:|------------:|------------:|-------------:|
| ('dsb-gpu', 'fp16', 20000, 'hbm')     |       4110.87 |       4103.51 |        3986.87 |        0.19 |        0.19 |         0.2  |
| ('dsb-gpu', 'fp16', 80000, 'hbm')     |       4198.62 |       4074.69 |        3881.82 |        3.05 |        3.14 |         3.3  |
| ('dsb-gpu', 'fp16', 160000, 'hbm')    |       4212.27 |       4387.28 |        4228.2  |       12.15 |       11.67 |        12.11 |
| ('dsb-gpu', 'fp16', 200000, 'hbm')    |       4240.65 |       4210.66 |        3802.59 |       18.87 |       19    |        21.04 |
| ('dsb-gpu', 'fp16', 240000, 'hbm')    |       4303.45 |       4269.23 |        4024.04 |       26.77 |       26.98 |        28.63 |
| ('dsb-gpu', 'fp16', 300000, 'grace')  |        371.48 |        362    |         364.41 |      484.54 |      497.24 |       493.95 |
| ('dsb-gpu', 'fp16', 300000, 'hybrid') |       1008.68 |       1003.6  |         992.43 |      178.45 |      179.35 |       181.37 |
| ('dsb-gpu', 'fp16', 400000, 'grace')  |        370.79 |        368.14 |         365.01 |      863.03 |      869.24 |       876.7  |
| ('dsb-gpu', 'fp16', 400000, 'hybrid') |        576.53 |        576.23 |         573.41 |      555.05 |      555.33 |       558.06 |
| ('dsb-gpu', 'fp16', 450000, 'grace')  |        371.45 |        362.84 |         366    |     1090.34 |     1116.2  |      1106.57 |
| ('dsb-gpu', 'fp16', 450000, 'hybrid') |        517.19 |        505.53 |         511.7  |      783.08 |      801.14 |       791.48 |
| ('dsb-gpu', 'fp32', 20000, 'hbm')     |       3991.68 |       3985.49 |        4147.81 |        0.4  |        0.4  |         0.39 |
| ('dsb-gpu', 'fp32', 80000, 'hbm')     |       4195.19 |       3628.43 |        3733.04 |        6.1  |        7.06 |         6.86 |
| ('dsb-gpu', 'fp32', 160000, 'hbm')    |       4200.46 |       3776.17 |        3647.83 |       24.38 |       27.12 |        28.07 |
| ('dsb-gpu', 'fp32', 200000, 'grace')  |        371.78 |        367.51 |         365.86 |      430.37 |      435.36 |       437.32 |
| ('dsb-gpu', 'fp32', 200000, 'hybrid') |       1358.75 |       1315.18 |        1285.34 |      117.76 |      121.66 |       124.48 |
| ('dsb-gpu', 'fp32', 240000, 'grace')  |        371.32 |        366.92 |         364.64 |      620.49 |      627.93 |       631.85 |
| ('dsb-gpu', 'fp32', 240000, 'hybrid') |        749.17 |        734.08 |         727.53 |      307.54 |      313.86 |       316.69 |
| ('dsb-gpu', 'fp32', 300000, 'grace')  |        370.67 |        366.98 |         363.76 |      971.2  |      980.97 |       989.66 |
| ('dsb-gpu', 'fp32', 300000, 'hybrid') |        547.19 |        539.15 |         535.26 |      657.9  |      667.72 |       672.57 |
| ('torch', 'fp16', 20000, 'hbm')       |       2132.36 |       2060.14 |        2069.57 |        0.38 |        0.39 |         0.39 |
| ('torch', 'fp16', 80000, 'hbm')       |       3951.42 |       3963.77 |        3573.2  |        3.24 |        3.23 |         3.58 |
| ('torch', 'fp16', 160000, 'hbm')      |       4222.62 |       4325.95 |        4125.52 |       12.13 |       11.84 |        12.41 |
| ('torch', 'fp16', 200000, 'hbm')      |       4196.57 |       4166.6  |        3739.51 |       19.06 |       19.2  |        21.39 |
| ('torch', 'fp16', 240000, 'hbm')      |       4175.01 |       4227.47 |        3924.23 |       27.59 |       27.25 |        29.36 |
| ('torch', 'fp32', 20000, 'hbm')       |       2834.41 |       1552.06 |         958.11 |        0.56 |        1.03 |         1.67 |
| ('torch', 'fp32', 80000, 'hbm')       |       3848.08 |       1226.76 |        1208.09 |        6.65 |       20.87 |        21.19 |
| ('torch', 'fp32', 160000, 'hbm')      |       4118.19 |       1475.29 |        1393.75 |       24.87 |       69.41 |        73.47 |

Measured bandwidth at B=1: HBM 4225 GB/s (n>=80,000), Grace only 371 GB/s. Hybrid vs additive model T = M_HBM/BW_HBM + M_Grace/BW_Grace:

| precision   |      n |   hbm_GB |   grace_GB |   tps |   tps_model |   model_err_pct |
|:------------|-------:|---------:|-----------:|------:|------------:|----------------:|
| fp16        | 300000 |  127.373 |     52.627 | 0.178 |       0.172 |           3.838 |
| fp16        | 400000 |  127.386 |    192.614 | 0.555 |       0.549 |           1.14  |
| fp16        | 450000 |  127.354 |    277.646 | 0.783 |       0.778 |           0.686 |
| fp32        | 200000 |  127.386 |     32.614 | 0.118 |       0.118 |          -0.182 |
| fp32        | 240000 |  127.365 |    103.035 | 0.308 |       0.308 |          -0.013 |
| fp32        | 300000 |  127.334 |    232.666 | 0.658 |       0.657 |           0.195 |

Not completed (after dedupe):

| impl   | precision   | mode   |   batch | status   | n                    |
|:-------|:------------|:-------|--------:|:---------|:---------------------|
| torch  | fp16        | auto   |       1 | oom      | 300000,400000,450000 |
| torch  | fp16        | auto   |       8 | oom      | 300000,400000,450000 |
| torch  | fp16        | auto   |      64 | oom      | 300000,400000,450000 |
| torch  | fp32        | auto   |       1 | oom      | 200000,240000,300000 |
| torch  | fp32        | auto   |       8 | oom      | 200000,240000,300000 |
| torch  | fp32        | auto   |      64 | oom      | 200000,240000,300000 |

Dense suites used (latest wins): dense_scaling_v5, dense_scaling_v5_supp_grace_fp16, dense_scaling_v5_supp_grace_fp32, dense_scaling_v7_grace_fp16, dense_scaling_v9_grace_fp16, dense_scaling_v9_grace_fp32, dense_scaling_v9_hybrid_fp16, dense_scaling_v9_hybrid_fp32

## Dense +-1 bit path on 2 GH200s (bit_dense_multi, K=50, B=1): matrix split, step time, bandwidth

placement: hbm = whole matrix in HBM (rows split over the GPUs), hybrid = each GPU fills its HBM first and streams the remaining rows from its local Grace memory through a 2 GiB staging buffer

|            n |   gpus | placement   |   matrix_GB |   hbm_GB |   grace_GB |   gen_s |    tps |     GBps | suite                      |
|-------------:|-------:|:------------|------------:|---------:|-----------:|--------:|-------:|---------:|:---------------------------|
| 500000       |      2 | hbm         |      31.256 |   31.256 |      0     |  3.1955 | 0.0035 | 8951.63  | bit_multi_20260925T144444Z |
|      1e+06   |      2 | hbm         |     125.008 |  125.008 |      0     | 12.6443 | 0.0139 | 8991.22  | bit_multi_20260925T144444Z |
|      2e+06   |      2 | hybrid      |     500     |  254.256 |    245.744 | 41.6381 | 0.4069 | 1228.9   | bit_multi_20260925T144444Z |
|      2.5e+06 |      2 | hybrid      |     781.28  |  254.25  |    527.03  | 59.6463 | 0.8436 |  926.087 | bit_multi_20260925T144444Z |
|      3e+06   |      2 | hybrid      |    1125.02  |  254.237 |    870.787 | 81.2733 | 1.4015 |  802.708 | bit_multi_20260925T144444Z |

Aggregate bandwidths (2 GPUs): HBM-resident points 8971 GB/s, Grace share of the hybrid points 646 GB/s (323 GB/s per GPU). Two-tier model on the hybrid points:

|       n |    tps |   tps_model |   model_err_pct |
|--------:|-------:|------------:|----------------:|
| 2e+06   | 0.4069 |      0.4085 |         -0.399  |
| 2.5e+06 | 0.8436 |      0.8436 |          0      |
| 3e+06   | 1.4015 |      1.3754 |          1.8994 |

Mattis check (J_ij = xi_i xi_j, ground-state energy -n(n-1)/2): seeds that reached it, and the step time relative to the random instance of the same n

|            n | placement   |   seeds |   found |   expected_energy |    tps |   tps_random |   tps_ratio |
|-------------:|:------------|--------:|--------:|------------------:|-------:|-------------:|------------:|
| 500000       | hbm         |       3 |       3 |     -124999750000 | 0.0035 |       0.0035 |      0.9998 |
|      1e+06   | hbm         |       3 |       3 |     -499999500000 | 0.0139 |       0.0139 |      1.0004 |
|      2e+06   | hybrid      |       3 |       3 |    -1999999000000 | 0.4073 |       0.4069 |      1.0011 |
|      2.5e+06 | hybrid      |       3 |       3 |    -3124998750000 | 0.8455 |       0.8436 |      1.0023 |
|      3e+06   | hybrid      |       3 |       3 |    -4499998500000 | 1.4009 |       1.4015 |      0.9995 |

15 of 15 Mattis runs found the planted ground state.

## Provenance: configurations taken from a suite newer than v5

| family   | instance   |   agents | label          | suite                       | steps                 | run_date         |
|:---------|:-----------|---------:|:---------------|:----------------------------|:----------------------|:-----------------|
| G-set    | G43        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:12 |
| G-set    | G43        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:12 |
| G-set    | G48        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:15 |
| G-set    | G48        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:15 |
| G-set    | G55        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:19 |
| G-set    | G55        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:19 |
| G-set    | G58        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:24 |
| G-set    | G58        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:24 |
| G-set    | G60        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:32 |
| G-set    | G60        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:32 |
| G-set    | G63        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:42 |
| G-set    | G63        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:42 |
| G-set    | G64        |      512 | gemm-int8      | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:51 |
| G-set    | G64        |      512 | public-matched | run_Gset_int8_v6_supplement | 200,400,800,1600,3200 | 2026-09-23 19:51 |
| G-set    | G66        |      512 | gemm-int8      | run_Gset_int8_v7            | 200,400,800,1600,3200 | 2026-09-23 21:50 |
| G-set    | G66        |      512 | public-matched | run_Gset_int8_v7            | 200,400,800,1600,3200 | 2026-09-23 21:50 |
