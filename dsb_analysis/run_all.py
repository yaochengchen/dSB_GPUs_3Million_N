#!/usr/bin/env python3
"""Analyse the dSB-GPU benchmark results and produce all figures and tables.

Usage:
    python run_all.py --root ../dsb-gpu_v5 --out out

`--root` is the dsb-gpu checkout; every run is read from its results/ directory (run_K2000_result_v5/,
run_Gset_result_v5/, ..., dense_scaling_*, scaling_sparse_*, bit_multi_*, bit_mattis_*; see dsbplot/load.py).
Repeated runs of the same configuration: the latest run wins.
Everything is written under `--out`:

    out/figures/*.pdf, *.png        figures (PDF for the paper, PNG for viewing)
    out/tables/*.csv                tidy summaries (one row per instance/steps/path)
    out/tables/*.tex                LaTeX booktabs fragments
    out/summary.md                  key numbers quoted in the paper
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsbplot import load, summarize as S, figures as F, tables as T, style

STEPS_MAIN = 3200


def save(fig, out, name):
    fig.savefig(os.path.join(out, "figures", name + ".pdf"))
    fig.savefig(os.path.join(out, "figures", name + ".png"))
    plt.close(fig)
    print("  wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../dsb-gpu_v5", help="dsb-gpu checkout containing results/")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    for d in ("figures", "tables"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    style.apply()

    print("loading ...")
    bench = load.load_benchmarks(args.root)
    dense = load.load_dense_scaling(args.root)
    sparse = load.load_sparse_scaling(args.root)
    bit = load.load_bit_multi(args.root)
    print(f"  {len(bench)} benchmark rows, {len(dense)} dense-scaling rows, {len(sparse)} sparse-scaling rows, "
          f"{len(bit)} bit multi-GPU rows")

    # ---- benchmark families (deduplicated: the newest suite wins for the same instance/B/K/path) ----
    fam_all = pd.concat([S.bench_rows(bench, f) for f in ("K2000", "G-set", "QPLIB")], ignore_index=True)
    is_main = fam_all["family"].map(S.MAIN_B).isna() | (fam_all["agents"] == fam_all["family"].map(S.MAIN_B))
    fam = fam_all[is_main]
    k2000_all = S.summarize(fam_all[fam_all["family"] == "K2000"])          # every B (small-B sweep when present)
    summ = S.with_speedup(S.summarize(fam))
    summ.to_csv(os.path.join(args.out, "tables", "benchmark_summary.csv"), index=False)
    k2000_all.to_csv(os.path.join(args.out, "tables", "k2000_all_batches.csv"), index=False)
    fam_prov = fam.assign(run_date=pd.to_datetime(fam["run_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M"))
    prov = fam_prov.groupby(["family", "instance", "agents", "steps", "label"]).agg(
        suite=("suite", lambda x: ",".join(sorted(set(x)))), run_date=("run_date", "first")).reset_index()
    prov.to_csv(os.path.join(args.out, "tables", "provenance.csv"), index=False)
    fastest = S.fastest_per_instance(summ, STEPS_MAIN)
    fastest.to_csv(os.path.join(args.out, "tables", f"fastest_path_K{STEPS_MAIN}.csv"), index=False)
    ident = S.bitwise_identity(fam)
    ident.to_csv(os.path.join(args.out, "tables", "trajectory_identity.csv"), index=False)

    # agents sweep (G-set large instances, B = 512 ... 8192)
    ag = bench[bench["suite"].isin(["run_Gset_agents_v5_supp", "run_Gset_result_v5"])]
    ag = ag[ag["instance"].isin(set(bench[bench["suite"] == "run_Gset_agents_v5_supp"]["instance"]))]
    agents = S.summarize(S.dedupe(ag).assign(family="G-set"))
    agents.to_csv(os.path.join(args.out, "tables", "gset_agents_sweep.csv"), index=False)

    # library-defaults baseline sweeps
    lib = S.summarize(bench[bench["suite"] == "run_Gset_library_v5_supp"].assign(family="G-set-lib"))
    lib.to_csv(os.path.join(args.out, "tables", "gset_library_sweep.csv"), index=False)
    qlib = S.summarize(bench[bench["suite"] == "run_qplib_library_v5_supp"].assign(family="QPLIB-lib"))
    qlib.to_csv(os.path.join(args.out, "tables", "qplib_library_sweep.csv"), index=False)

    dsum = F.dense_summary(dense) if len(dense) else pd.DataFrame()
    if len(dense):
        dsum.to_csv(os.path.join(args.out, "tables", "dense_scaling_summary.csv"), index=False)
    if len(sparse):
        S.summarize(sparse.assign(family="sparse")).to_csv(os.path.join(args.out, "tables", "sparse_scaling_summary.csv"), index=False)
    bsum = F.bit_summary(bit) if len(bit) else pd.DataFrame()
    if len(bit):
        bsum.to_csv(os.path.join(args.out, "tables", "bit_multi_summary.csv"), index=False)

    print("figures ...")
    save(F.fig_k2000_dense(summ, fam), args.out, "fig1_k2000_dense")
    save(F.fig_k2000_panorama(summ, steps=1600), args.out, "figS_k2000_all_paths")
    save(F.fig_gset_qplib(summ, fastest, STEPS_MAIN), args.out, "fig6_gset_qplib")
    save(F.fig_library_sweep(lib), args.out, "fig7_gset66_library_baseline")
    if len(sparse):
        save(F.fig_sparse_scaling(sparse), args.out, "fig3_sparse_scaling")
    save(F.fig_sensitivity(agents, summ[summ["family"] == "K2000"]), args.out, "fig4_sensitivity")
    save(F.fig_path_selection(summ, fastest, STEPS_MAIN), args.out, "fig5_path_selection")
    if len(dense):
        save(F.fig_dense_scaling(dense), args.out, "fig8_dense_scaling")
    if k2000_all["agents"].nunique() > 1:
        save(F.fig_k2000_batch(k2000_all), args.out, "figS_k2000_batch")
    else:
        print("  skip figS_k2000_batch: K2000 has results at B=512 only")
    save(F.fig_identity(ident, STEPS_MAIN), args.out, "figS_trajectory_identity")
    if len(bit):
        save(F.fig_bit_multi(bit), args.out, "fig9_bit_multi")

    print("tables ...")
    main_paths = ["public-matched", "csr-row", "block", "gemm-fp32", "gemm-tf32", "gemm-int8", "bit", "gemm-fp16"]
    tex = {
        "tab_k2000_K3200.tex": T.k2000_table(summ, 3200, main_paths),
        "tab_k2000_K1600.tex": T.k2000_table(summ, 1600, main_paths),
        "tab_k2000_steps.tex": T.steps_table(summ, main_paths),
        "tab_gset_instances.tex": T.instance_table(fastest, summ, "G-set", STEPS_MAIN),
        "tab_qplib_instances.tex": T.instance_table(fastest, summ, "QPLIB", STEPS_MAIN),
        "tab_arith_ratio.tex": T.arith_table(summ, ident, STEPS_MAIN),
    }
    if len(dense):
        tex["tab_capacity.tex"] = T.capacity_table(dsum, dense, bsum)
        tex["tab_grace.tex"] = T.grace_table(dsum)
    if len(bit):
        tex["tab_bit_multi.tex"] = T.bit_multi_table(bsum)
    for name, txt in tex.items():
        open(os.path.join(args.out, "tables", name), "w").write(txt)
        print("  wrote", name)

    write_summary(args.out, summ, fastest, ident, agents, lib, qlib, dense, dsum, sparse, prov, k2000_all, bit, bsum)
    print("done ->", args.out)


def write_summary(out, summ, fastest, ident, agents, lib, qlib, dense, dsum, sparse, prov, k2000_all, bit, bsum):
    L = []
    k = summ[(summ["family"] == "K2000") & (summ["steps"] == 3200)].set_index("label")
    L.append("# Key numbers (auto-generated)\n")
    L.append("## K2000 (n=2000, B=512, K=3200, 50 trials)\n")
    L.append("| path | us/step | speedup vs SB (integration) | speedup (wall) | u/M | median | TTS99 solver (s) |\n|---|---|---|---|---|---|---|")
    for l in k.sort_values("us_per_step").index:
        r = k.loc[l]
        L.append(f"| {l} | {r.us_per_step:.1f} | {r.speedup_solver:.2f} | {r.speedup_wall:.2f} | {int(r.successes)}/{int(r.runs)} | {r.median_obj:.1f} | {r.tts99_solver_s:.3g} |")
    idk = ident[(ident["family"] == "K2000") & (ident["steps"] == 3200)].set_index("label")["frac_identical"]
    L.append("\nFraction of seeds bitwise-identical to gemm TF32 at K=3200: " + ", ".join(f"{l}={v:.2f}" for l, v in idk.items()))
    for fam in ["G-set", "QPLIB"]:
        f = fastest[fastest["family"] == fam]
        L.append(f"\n## {fam} (K=3200): fastest path per instance vs matched SB 2.0.0\n")
        L.append(f"- instances: {len(f)}; fastest path counts: {f['label'].value_counts().to_dict()}")
        L.append(f"- integration speedup: min {f.speedup_solver.min():.1f}, median {f.speedup_solver.median():.1f}, max {f.speedup_solver.max():.1f}")
        L.append(f"- wall speedup: min {f.speedup_wall.min():.1f}, median {f.speedup_wall.median():.1f}, max {f.speedup_wall.max():.1f}")
        pub = summ[(summ["family"] == fam) & (summ["steps"] == 3200) & (summ["label"] == "public-matched")].set_index("instance").loc[f["instance"]]
        better = (f["median_gap_pct"].values < pub["median_gap_pct"].values - 1e-9).sum()
        worse = (f["median_gap_pct"].values > pub["median_gap_pct"].values + 1e-9).sum()
        L.append(f"- median gap: dsb better on {better}, worse on {worse}, tie on {len(f)-better-worse} instances")
        L.append(f"- instances where dsb reaches best-known in >=1 trial: {(f.successes>0).sum()}; SB: {(pub.successes>0).sum()}")
    a = agents[(agents["steps"] == 3200) & (agents["label"] == "auto")]
    L.append("\n## G-set large instances, agents sweep (auto path, K=3200): median gap % by B\n")
    L.append(a.pivot_table(index="instance", columns="agents", values="median_gap_pct").round(3).to_markdown())
    L.append("\n## G-set 66 instances, K=800: csr-row vs SB 2.0.0 library defaults\n")
    d = lib[lib["label"] == "csr-row"].set_index("instance"); p = lib[lib["label"] == "public-library"].set_index("instance").loc[d.index]
    sp = p["solver_s"] / d["solver_s"]
    L.append(f"- integration speedup: min {sp.min():.1f}, median {sp.median():.1f}, max {sp.max():.1f}")
    L.append(f"- median gap lower for dsb on {(d.median_gap_pct < p.median_gap_pct).sum()} / {len(d)} instances; SB library reaches best-known on {(p.successes>0).sum()} instances, dsb on {(d.successes>0).sum()}")
    if len(sparse):
        s = S.summarize(sparse.assign(family="sparse"))
        c = s[s["label"] == "csr-row"]
        L.append("\n## Sparse scaling (degree-100 bipartite, csr-row, B=200)\n")
        L.append(f"- ns per nonzero per step: {(c.solver_s/c.steps/(2*c.edges)*1e9).min():.3f}-{(c.solver_s/c.steps/(2*c.edges)*1e9).max():.3f}; GPU memory at n=200000: {c[c.n==200000].gpu_mem_bytes.iloc[0]/1e6:.0f} MB")
        pu = s[s["label"] == "public-matched"]
        for _, r in pu.iterrows():
            cc = c[(c.n == r.n) & (c.steps == r.steps)]
            if len(cc):
                L.append(f"- n={int(r.n)} K={int(r.steps)}: SB {r.solver_s:.3f} s vs csr-row {cc.solver_s.iloc[0]:.3f} s -> {r.solver_s/cc.solver_s.iloc[0]:.1f}x")
    if k2000_all["agents"].nunique() > 1:
        kb = k2000_all[(k2000_all["steps"] == 3200) & k2000_all["label"].isin(["bit", "gemm-fp16", "gemm-int8", "gemm-tf32", "public-matched"])]
        L.append("\n## K2000 batch sweep (K=3200): us/step, trials at target / 50, TTS99 solver (s)\n")
        L.append(kb.pivot_table(index="label", columns="agents", values="us_per_step").round(2).to_markdown())
        L.append("")
        L.append(kb.pivot_table(index="label", columns="agents", values="successes").to_markdown())
        L.append("")
        L.append(kb.pivot_table(index="label", columns="agents", values="tts99_solver_s").round(3).to_markdown())
    if len(dense):
        L.append("\n## Dense scaling (dsb-gpu gemm and PyTorch, K=50): time per step and matrix bandwidth\n")
        L.append("tier: hbm = matrix in HBM, grace = whole matrix in Grace memory (staged through HBM), "
                 "hybrid = HBM filled first, remaining rows in Grace memory\n")
        L.append(dsum.assign(ms=dsum.tps * 1e3, GBps=dsum.bw_TBps * 1e3).pivot_table(
            index=["impl", "precision", "n", "tier"], columns="batch", values=["ms", "GBps"]).round(2).to_markdown())
        d1 = dsum[dsum["batch"] == 1]
        bh, bg = F.dense_link_bandwidths(d1)
        m = F.dense_hybrid_model(d1)
        L.append(f"\nMeasured bandwidth at B=1: HBM {bh/1e9:.0f} GB/s (n>=80,000), Grace only {bg/1e9:.0f} GB/s. "
                 "Hybrid vs additive model T = M_HBM/BW_HBM + M_Grace/BW_Grace:\n")
        L.append(m.assign(grace_GB=m.grace_b / 1e9, hbm_GB=m.hbm_b / 1e9)[
            ["precision", "n", "hbm_GB", "grace_GB", "tps", "tps_model", "model_err_pct"]].round(3).to_markdown(index=False))
        bad = dense[dense.status != "completed"].groupby(["impl", "precision", "mode", "batch", "status"]).n.apply(
            lambda x: ",".join(str(int(v)) for v in sorted(x))).reset_index()
        L.append("\nNot completed (after dedupe):\n\n" + (bad.to_markdown(index=False) if len(bad) else "none"))
        L.append("\nDense suites used (latest wins): " + ", ".join(sorted(dense["suite"].unique())))
    if len(bit):
        gpus = int(bsum["gpus"].max())
        L.append(f"\n## Dense +-1 bit path on {gpus} GH200s (bit_dense_multi, K=50, B=1): matrix split, step time, bandwidth\n")
        L.append("placement: hbm = whole matrix in HBM (rows split over the GPUs), hybrid = each GPU fills its HBM first "
                 "and streams the remaining rows from its local Grace memory through a 2 GiB staging buffer\n")
        r = bsum[bsum["instance"] == "random"]
        L.append(r.assign(matrix_GB=r.mb / 1e9, hbm_GB=r.hbm_b / 1e9, grace_GB=r.grace_b / 1e9, GBps=r.bw_TBps * 1e3)[
            ["n", "gpus", "placement", "matrix_GB", "hbm_GB", "grace_GB", "gen_s", "tps", "GBps", "suite"]].round(4).to_markdown(index=False))
        bh, bg = F.bit_link_bandwidths(bsum)
        y = r[r["placement"] == "hybrid"].copy()
        y["tps_model"] = y.hbm_b / bh + y.grace_b / bg
        y["model_err_pct"] = (y.tps / y.tps_model - 1) * 100
        L.append(f"\nAggregate bandwidths ({gpus} GPUs): HBM-resident points {bh/1e9:.0f} GB/s, Grace share of the hybrid points "
                 f"{bg/1e9:.0f} GB/s ({bg/1e9/gpus:.0f} GB/s per GPU). Two-tier model on the hybrid points:\n")
        L.append(y[["n", "tps", "tps_model", "model_err_pct"]].round(4).to_markdown(index=False))
        m = bsum[bsum["instance"] == "mattis"]
        if len(m):
            L.append("\nMattis check (J_ij = xi_i xi_j, ground-state energy -n(n-1)/2): seeds that reached it, and the step time "
                     "relative to the random instance of the same n\n")
            mm = m.set_index("n").join(r.set_index("n")[["tps"]].rename(columns={"tps": "tps_random"}))
            mm["tps_ratio"] = mm.tps / mm.tps_random
            L.append(mm.assign(expected_energy=mm.expected.map(lambda v: f"{v:.0f}"))[
                ["placement", "seeds", "found", "expected_energy", "tps", "tps_random", "tps_ratio"]].round(4).to_markdown())
            L.append(f"\n{int(m.found.sum())} of {int(m.seeds.sum())} Mattis runs found the planted ground state.")
    L.append("\n## Provenance: configurations taken from a suite newer than v5\n")
    newer = prov[~prov["suite"].str.contains(r"_v5(?:_|$)", regex=True)]
    newer = newer.groupby(["family", "instance", "agents", "label", "suite"]).agg(
        steps=("steps", lambda x: ",".join(str(int(v)) for v in sorted(x))), run_date=("run_date", "max")).reset_index()
    L.append(newer.to_markdown(index=False) if len(newer) else "none")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("  wrote summary.md")


if __name__ == "__main__":
    main()
