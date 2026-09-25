# G-set data and targets

Place the downloaded Stanford G-set instances in `data/Gset/data/`.  No
solution vectors are required: the comparison script reads the target cut from
`best_known.csv`.

The CSV contains the 71 commonly distributed instances (`G1`--`G67`, `G70`,
`G72`, `G77`, and `G81`).  Values are **best-known cuts**, not claims of proven
optimality.  The legacy values and graph sizes follow the commonly used G-set
benchmark compilation:

- https://github.com/0816keisuke/max-cut-problem-benchmark

The following newer reported values override that compilation:

- `G63 = 27047`: https://arxiv.org/abs/2510.21105
- `G72 = 7008`, `G77 = 9940`, `G81 = 14060`:
  https://arxiv.org/abs/2505.18508

The table was checked on 2026-09-20.  Since best-known values can improve,
record the CSV version used with published benchmark results.

Example:

```bash
REPEATS=5 WARMUP=1 BATCH=200 STEPS=800 \
scripts/run_gh200_gset_comparison.sh data/Gset/data/G22
```

An explicit target overrides the table:

```bash
scripts/run_gh200_gset_comparison.sh data/Gset/data/G22 13300
```
