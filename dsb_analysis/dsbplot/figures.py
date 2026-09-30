"""Figure functions.  Each takes tidy DataFrames and returns a matplotlib Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import COLOR, MARKER, NAME, order

MAIN_PATHS = ["gemm-fp16", "gemm-int8", "gemm-tf32", "gemm-fp32", "bit", "block", "csr-row", "csr-block", "public-matched"]


def _leg(ax, labels, **kw):
    h = [Line2D([], [], color=COLOR[l], marker=MARKER[l], ls="-" if not l.startswith("public") else "--",
                label=NAME[l]) for l in labels]
    ax.legend(handles=h, **kw)


def _panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


# ---------------------------------------------------------------------------
# Fig. 1  K2000 dense comparison
# ---------------------------------------------------------------------------
def fig_k2000_dense(summ: pd.DataFrame, raw: pd.DataFrame, steps_box: int = 3200):
    s = summ[(summ["family"] == "K2000") & (summ["agents"] == 512)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.05, 1]))
    labels = [l for l in MAIN_PATHS if l in set(s["label"])]
    for l in labels:
        d = s[s["label"] == l].sort_values("steps")
        a.plot(d["steps"], d["solver_s"] * 1e3, marker=MARKER[l], color=COLOR[l],
               ls="--" if l.startswith("public") else "-", label=NAME[l])
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks(sorted(s["steps"].unique())); a.set_xticklabels([str(int(x)) for x in sorted(s["steps"].unique())])
    a.set_xlabel("integration steps $K$"); a.set_ylabel("median integration time (ms)")
    a.legend(ncol=2, loc="upper left", fontsize=6.3, handlelength=1.6)
    _panel(a, "A")

    r = raw[(raw["family"] == "K2000") & (raw["steps"] == steps_box) & (raw["agents"] == 512)]
    labels_b = [l for l in labels if l in set(r["label"])]
    rng = np.random.default_rng(0)
    tgt = r["target"].iloc[0]
    for i, l in enumerate(labels_b):
        v = r[r["label"] == l]["objective"].values
        y = i + rng.uniform(-0.28, 0.28, size=len(v))
        b.scatter(v, y, s=9, color=COLOR[l], alpha=0.55, linewidths=0)
        b.plot([np.median(v)] * 2, [i - 0.38, i + 0.38], color="k", lw=1.2)
        b.text(r["objective"].min() - 1, i, f"{int((v >= tgt).sum())}/{len(v)}", va="center", ha="right", fontsize=6.5)
    b.set_yticks(range(len(labels_b))); b.set_yticklabels([NAME[l] for l in labels_b], fontsize=6.8)
    b.axvline(tgt, color="k", ls=":", lw=0.9)
    b.text(tgt, -0.7, f"target {int(tgt)}", ha="right", va="bottom", fontsize=6.5)
    b.text(r["objective"].min() - 1, -0.75, "at target", ha="right", va="bottom", fontsize=6)
    b.set_xlim(r["objective"].min() - 9, tgt + 2)
    b.set_xlabel(f"final cut after $K={steps_box}$ (50 seeds; bar = median)")
    b.invert_yaxis(); b.grid(axis="y", alpha=0)
    _panel(b, "B")
    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# K2000 panorama of all execution paths (per-step time, log)
# ---------------------------------------------------------------------------
def fig_k2000_panorama(summ: pd.DataFrame, steps: int = 1600):
    s = summ[(summ["family"] == "K2000") & (summ["steps"] == steps) & (summ["agents"] == 512) & (summ["label"] != "auto")].copy()
    s = s.sort_values("us_per_step")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = [COLOR[l] for l in s["label"]]
    ax.barh(range(len(s)), s["us_per_step"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([NAME[l] for l in s["label"]], fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("time per integration step (µs), $B=512$")
    pub = s[s["label"] == "public-matched"]["us_per_step"].iloc[0]
    ax.axvline(pub, color="k", ls="--", lw=0.8)
    for i, (v, l) in enumerate(zip(s["us_per_step"], s["label"])):
        ratio = pub / v
        ax.text(v * 1.15, i, f"{v:,.0f} µs  ({ratio:.2g}×)" if ratio >= 1 else f"{v:,.0f} µs  (1/{1/ratio:.2g})",
                va="center", fontsize=6.3)
    ax.set_xlim(right=s["us_per_step"].max() * 30)
    ax.invert_yaxis(); ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# G-set and QPLIB per-instance comparison at fixed steps
# ---------------------------------------------------------------------------
def fig_gset_qplib(summ: pd.DataFrame, fastest: pd.DataFrame, steps: int = 3200, relaxed: pd.DataFrame | None = None):
    """Panels C/D: fraction of trials reaching the target; when `relaxed` is given, the target of each instance is
    its relaxed level (tightest of 100/99.9/99.5/99 % reached by both), annotated where it is below 100 %."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw=dict(height_ratios=[1.25, 1], width_ratios=[16, 19]), sharex="col")
    for j, fam in enumerate(["G-set", "QPLIB"]):
        f = fastest[fastest["family"] == fam].sort_values("n").reset_index(drop=True)
        pub = summ[(summ["family"] == fam) & (summ["steps"] == steps) & (summ["label"] == "public-matched")]
        pub = pub.set_index("instance").loc[f["instance"]]
        x = np.arange(len(f))
        a = axes[0, j]
        a.bar(x, f["speedup_solver"], 0.7, color=[COLOR[l] for l in f["label"]])
        a.set_yscale("log"); a.axhline(1, color="k", lw=0.7)
        a.set_ylabel(f"speedup over SB 2.0.0, $K={steps}$")
        a.set_title(fam, fontsize=9, loc="left")
        for xi, (sp, lab) in enumerate(zip(f["speedup_solver"], f["label"])):
            a.text(xi, sp * 1.12, NAME[lab].split(" (")[0], rotation=90, ha="center", va="bottom", fontsize=5.6)
        a.set_ylim(top=a.get_ylim()[1] * 4)
        b = axes[1, j]
        if relaxed is not None:
            r = relaxed[relaxed["family"] == fam].set_index("instance").loc[f["instance"]]
            ud, us, lv = r["u_dsb"].values, r["u_sb"].values, r["level_pct"].values
            ylab = "trials reaching target (%)"
        else:
            ud, us, lv = f["successes"].values, pub["successes"].values, np.full(len(f), 100.0)
            ylab = "trials reaching best-known (%)"
        runs = f["runs"].values
        b.plot(x, ud / runs * 100, "o", color="C3", label="dsb-gpu (fastest path)")
        b.plot(x, us / runs * 100, "x", color="k", label="SB 2.0.0 (matched)")
        for xi, l in enumerate(lv):
            if l < 100:
                b.text(xi, 105, f"{l:g}%", ha="center", va="bottom", fontsize=4.8, color="0.35", rotation=90)
        b.set_ylim(-5, 135)
        b.set_ylabel(ylab)
        b.legend(fontsize=6.5, loc="center right" if fam == "G-set" else "lower right")
        b.set_xticks(x)
        b.set_xticklabels([f"{i.replace('QPLIB_', '')}\n{int(n)}" for i, n in zip(f["instance"], f["n"])], fontsize=5.0)
        b.set_xlabel("instance / $n$", fontsize=7)
    _panel(axes[0, 0], "A"); _panel(axes[0, 1], "B"); _panel(axes[1, 0], "C"); _panel(axes[1, 1], "D")
    fig.tight_layout(h_pad=0.4, w_pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# 66-instance G-set sweep against the package's library defaults
# ---------------------------------------------------------------------------
def fig_library_sweep(lib: pd.DataFrame):
    dsb = lib[lib["label"] == "csr-row"].set_index("instance")
    pub = lib[lib["label"] == "public-library"].set_index("instance").loc[dsb.index]
    n = dsb["n"].values
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a.scatter(n, pub["solver_s"] / dsb["solver_s"], s=18, color=COLOR["csr-row"], alpha=0.75, linewidths=0)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$n$"); a.set_ylabel("integration speedup, csr-row vs SB 2.0.0")
    a.set_title("SB 2.0.0 with library defaults, $K=800$", fontsize=7.5, loc="left")
    a.axhline(1, color="k", lw=0.7)
    _panel(a, "A")
    b.scatter(n, dsb["median_gap_pct"], s=18, color=COLOR["csr-row"], alpha=0.75, linewidths=0, label="dsb-gpu csr-row")
    b.scatter(n, pub["median_gap_pct"], s=18, marker="x", color="k", alpha=0.75, label="SB 2.0.0 (library defaults)")
    b.set_xscale("log"); b.set_xlabel("$n$"); b.set_ylabel("median gap to best-known (%)")
    b.legend(fontsize=6.5)
    _panel(b, "B")
    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Fig. 3  Sparse scaling on regular bipartite graphs
# ---------------------------------------------------------------------------
def fig_sparse_scaling(sp: pd.DataFrame):
    g = sp.groupby(["label", "steps", "n_original"]).agg(
        solver=("solver_s", "median"), wall=("wall_s", "median"), mem=("gpu_memory_bytes", "median"),
        eups=("edge_updates_per_s", "median"), edges=("edges", "first"), agents=("agents", "first")).reset_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    kmax = g["steps"].max()
    for (l, st), d in g.groupby(["label", "steps"]):
        d = d.sort_values("n_original")
        a.plot(d["n_original"], d["solver"], marker=MARKER[l], color=COLOR[l], ls="-" if st == kmax else ":",
               label=NAME[l] if st == kmax else None)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$n$ (degree 100, $B=200$)"); a.set_ylabel("median integration time (s)")
    ks = sorted(g["steps"].unique())
    h1, l1 = a.get_legend_handles_labels()
    a.legend(h1 + [Line2D([], [], color="grey", ls="-" if k == kmax else ":") for k in ks],
             l1 + [f"$K$={int(k)}" for k in ks], fontsize=6.2, loc="upper left")
    _panel(a, "A")
    d = g[(g["label"] == "csr-row")].groupby("n_original").agg(mem=("mem", "first"), eups=("eups", "median")).reset_index()
    b.plot(d["n_original"], d["mem"] / 1e9, marker="o", color=COLOR["csr-row"], label="csr-row GPU allocation")
    nn = np.array(sorted(set(sp["n_original"])), dtype=float)
    b.plot(nn, 4 * nn ** 2 / 1e9, "--", color="k", label="dense FP32 matrix $4n^2$")
    b.axhline(144, color="grey", lw=0.8, ls=":"); b.text(nn[-1], 120, "GH200 HBM (144 GB)", fontsize=6.5, va="top", ha="right")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("$n$"); b.set_ylabel("memory (GB)")
    b2 = b.twinx()
    b2.plot(d["n_original"], d["eups"] / 1e12, "s-", color="C2", ms=3.5, label="csr-row throughput (right axis)")
    b2.set_ylabel("nonzero updates per second ($10^{12}$/s)", color="C2"); b2.set_ylim(0, 1.0)
    b2.grid(False); b2.spines["right"].set_visible(True)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b2.legend(h1 + h2, l1 + l2, fontsize=6.0, loc="center right", bbox_to_anchor=(1.0, 0.45))
    _panel(b, "B")
    fig.tight_layout(w_pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# Fig. 4  Sensitivity: batch (agents) sweep and step sweep
# ---------------------------------------------------------------------------
def fig_sensitivity(agents: pd.DataFrame, k2000: pd.DataFrame, steps_gap: int = 3200):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    a, b, c = axes
    insts = sorted(agents["instance"].unique(), key=lambda i: agents[agents["instance"] == i]["n"].iloc[0])
    cmap = plt.get_cmap("viridis")
    for k, inst in enumerate(insts):
        d = agents[(agents["instance"] == inst) & (agents["label"] == "csr-block") & (agents["steps"] == steps_gap)].sort_values("agents")
        col = cmap(k / max(1, len(insts) - 1))
        a.plot(d["agents"], d["us_per_step"], marker="o", color=col, label=f"{inst} ($n$={int(d['n'].iloc[0])})")
        d2 = agents[(agents["instance"] == inst) & (agents["label"] == "csr-block") & (agents["steps"] == steps_gap)].sort_values("agents")
        b.plot(d2["agents"], d2["median_gap_pct"], marker="o", color=col)
        b.plot(d2["agents"], d2["best_gap_pct"], marker="o", mfc="none", ls=":", color=col)
    a.set_xscale("log", base=2); a.set_yscale("log", base=2)
    a.set_xlabel("replicas $B$"); a.set_ylabel(f"csr-block time per step (µs), $K$={steps_gap}")
    xs = sorted(agents["agents"].unique()); a.set_xticks(xs); a.set_xticklabels([str(int(x)) for x in xs], fontsize=6.5)
    a.legend(fontsize=5.8, loc="upper left")
    _panel(a, "A")
    b.set_xscale("log", base=2); b.set_xticks(xs); b.set_xticklabels([str(int(x)) for x in xs], fontsize=6.5)
    b.set_xlabel("replicas $B$"); b.set_ylabel(f"gap to best-known (%), $K$={steps_gap}")
    b.plot([], [], "o-", color="grey", label="median over 50 trials"); b.plot([], [], "o:", mfc="none", color="grey", label="best of 50 trials")
    b.legend(fontsize=6)
    _panel(b, "B")
    for l in ["gemm-fp16", "gemm-tf32", "bit", "csr-row", "public-matched"]:
        d = k2000[k2000["label"] == l].sort_values("steps")
        c.plot(d["steps"], d["p_batch"] * 100, marker=MARKER[l], color=COLOR[l], ls="--" if l.startswith("public") else "-",
               label=NAME[l] + (" (= INT8 = block = FP32)" if l == "gemm-tf32" else ""))
    c.set_xscale("log", base=2); ks = sorted(k2000["steps"].unique()); c.set_xticks(ks); c.set_xticklabels([str(int(k)) for k in ks], fontsize=6.5)
    c.set_xlabel("integration steps $K$"); c.set_ylabel("K2000 trials reaching 33337 (%), $B=512$")
    c.legend(fontsize=5.6, loc="upper left")
    _panel(c, "C")
    fig.tight_layout(w_pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Fig. 5  Path-selection region
# ---------------------------------------------------------------------------
def fig_path_selection(summ: pd.DataFrame, fastest: pd.DataFrame, steps: int = 3200):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    s = summ[(summ["steps"] == steps) & (summ["agents"] == 512) & (summ["family"].isin(["G-set", "K2000"]))]
    dense = ["gemm-fp16", "gemm-int8", "gemm-tf32", "gemm-fp32", "bit", "public-matched"]
    for l in dense:
        d = s[s["label"] == l].groupby("n")["us_per_step"].median().reset_index().sort_values("n")
        if d.empty:
            continue
        a.plot(d["n"], d["us_per_step"], marker=MARKER[l], color=COLOR[l], ls="--" if l.startswith("public") else "-", label=NAME[l])
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$n$"); a.set_ylabel(f"time per step (µs), $B=512$, $K$={steps}")
    a.legend(fontsize=5.8, ncol=2, loc="upper left")
    _panel(a, "A")

    f = fastest[fastest["family"].isin(["G-set", "K2000"])].copy()   # QPLIB logs carry no nnz
    nnz = summ.groupby(["family", "instance"])["nnz"].max()
    e = f["edges"].copy()
    e = e.fillna(pd.Series([nnz.get((a, b), np.nan) / 2 for a, b in zip(f["family"], f["instance"])], index=f.index))
    f["deg"] = 2 * e / f["n"]
    # shaded bands: the measured selection rule in degree
    for lo, hi, l in [(1.5, 4.5, "csr-block"), (4.5, 8, "csr-row"), (8, 4000, "gemm-fp16")]:
        b.axhspan(lo, hi, color=COLOR[l], alpha=0.08, lw=0)
    b.text(640, 2.3, "csr-block", ha="left", va="center", fontsize=6.5, color=COLOR["csr-block"])
    b.text(640, 6.2, "csr-row", ha="left", va="center", fontsize=6.5, color=COLOR["csr-row"])
    b.text(640, 250, "tensor-core products\n(FP16 / INT8)", ha="left", va="center", fontsize=6.5, color=COLOR["gemm-fp16"])
    for l in order(f["label"].unique()):
        d = f[f["label"] == l]
        b.scatter(d["n"], d["deg"], marker=MARKER[l], color=COLOR[l], s=34, label=NAME[l], edgecolors="k", linewidths=0.4, zorder=5)
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("$n$"); b.set_ylabel("average degree $2|\\mathcal{E}|/n$")
    b.set_ylim(1.5, 4000); b.set_xlim(600, 30000)
    b.legend(fontsize=6.2, title="fastest path (integration time)", title_fontsize=6.5, loc="upper right")
    for _, r in f.iterrows():
        if r["family"] == "K2000":
            b.annotate("K2000\n(complete graph)", (r["n"], r["deg"]), textcoords="offset points", xytext=(-7, 0),
                       ha="right", va="center", fontsize=6)
    _panel(b, "B")
    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Dense scaling: HBM tier, Grace (NVLink-C2C) tier and the hybrid HBM+Grace split
# ---------------------------------------------------------------------------
DENSE_STYLE = {("dsb-gpu", "fp16"): ("C3", "o", "-"), ("dsb-gpu", "fp32"): ("C0", "^", "-"),
               ("torch", "fp16"): ("C3", "o", "--"), ("torch", "fp32"): ("C0", "^", "--")}
DENSE_NAME = {"dsb-gpu": "dsb-gpu gemm", "torch": "PyTorch dense"}


def dense_summary(dense: pd.DataFrame) -> pd.DataFrame:
    ok = dense[dense["status"] == "completed"]
    g = ok.groupby(["impl", "precision", "batch", "n", "mode", "tier"]).agg(
        tps=("time_per_step_s", "median"), mb=("matrix_bytes", "first"), grace_b=("grace_bytes", "first"),
        suite=("suite", "first")).reset_index()
    g["grace_b"] = g["grace_b"].fillna(0)
    g["hbm_b"] = g["mb"] - g["grace_b"]
    g["bw_TBps"] = g["mb"] / g["tps"] / 1e12
    return g


def dense_link_bandwidths(g: pd.DataFrame) -> tuple[float, float]:
    """Median matrix bandwidth (B/s) of dsb-gpu gemm with the whole matrix in HBM (n >= 80,000) and in Grace memory."""
    d = g[g["impl"] == "dsb-gpu"]
    hbm = d[(d["tier"] == "hbm") & (d["n"] >= 80000)]["bw_TBps"].median() * 1e12
    grace = d[d["tier"] == "grace"]["bw_TBps"].median() * 1e12
    return hbm, grace


def dense_hybrid_model(g: pd.DataFrame) -> pd.DataFrame:
    """Additive model for the hybrid split: T = M_HBM/BW_HBM + M_Grace/BW_Grace, with both bandwidths measured."""
    bh, bg = dense_link_bandwidths(g)
    h = g[(g["impl"] == "dsb-gpu") & (g["tier"] == "hybrid")].copy()
    h["tps_model"] = h["hbm_b"] / bh + h["grace_b"] / bg
    h["model_err_pct"] = (h["tps"] / h["tps_model"] - 1) * 100
    return h


def fig_dense_scaling(dense: pd.DataFrame):
    """Panels A, B: time per step vs n at B=1 for FP32 and FP16 separately (dsb-gpu gemm vs PyTorch);
    Panel C: matrix bandwidth bn^2/T_step for both.  Filled: HBM; black-edged: whole matrix in Grace memory;
    open diamonds: hybrid (HBM filled first).  Grey ticks in C: additive model of the hybrid split."""
    g = dense_summary(dense)
    g1 = g[g["batch"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

    def series(ax, impl, prec, ycol, scale, with_labels):
        col, mk, ls = DENSE_STYLE[(impl, prec)]
        x = g1[(g1["impl"] == impl) & (g1["precision"] == prec)]
        h = x[(x["mode"] == "auto") & (x["tier"] == "hbm")].sort_values("n")
        r = x[(x["mode"] == "auto") & (x["tier"] == "grace")].sort_values("n")
        y = x[x["tier"] == "hybrid"].sort_values("n")
        if impl == "torch":
            ax.plot(h["n"], h[ycol] * scale, color="0.45", marker=mk, ls="--", mfc="white", mec="0.45", ms=4.5, lw=1.0,
                    alpha=0.9, label=f"PyTorch dense {prec.upper()}" if with_labels else None)
            return
        ax.plot(h["n"], h[ycol] * scale, color=col, marker=mk, ls="-", ms=4.5, lw=1.2,
                label=f"gemm {prec.upper()}, HBM" if with_labels else None)
        for part, lab, m2, ls2, mfc, mec in [(r, "whole matrix in Grace memory", mk, ":", col, "k"),
                                             (y, "hybrid HBM + Grace", "D", "-.", "white", col)]:
            if len(part) and len(h):
                j = pd.concat([h.tail(1), part])
                ax.plot(j["n"], j[ycol] * scale, color=col, ls=ls2, lw=1.0, alpha=0.8)
                ax.plot(part["n"], part[ycol] * scale, color=col, marker=m2, ls="none", mfc=mfc, mec=mec, mew=1.0, ms=5,
                        label=f"gemm {prec.upper()}, {lab}" if with_labels else None)

    for ax, prec, letter in [(axes[0], "fp32", "A"), (axes[1], "fp16", "B")]:
        series(ax, "torch", prec, "tps", 1e3, True)
        series(ax, "dsb-gpu", prec, "tps", 1e3, True)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$n$ (dense random $J$)")
        ax.set_ylabel("time per step (ms), $B=1$")
        ax.set_title(prec.upper() + (" (TF32 product for gemm)" if prec == "fp32" else ""), fontsize=7.5, loc="left")
        ax.set_ylim(top=ax.get_ylim()[1] * 30)
        ax.legend(fontsize=5.4, loc="upper left")
        _panel(ax, letter)
    c = axes[2]
    for prec in ("fp32", "fp16"):
        series(c, "torch", prec, "bw_TBps", 1.0, False)
        series(c, "dsb-gpu", prec, "bw_TBps", 1.0, False)
    m = dense_hybrid_model(g1)
    c.plot(m["n"], m["mb"] / m["tps_model"] / 1e12, ls="none", marker="_", ms=9, mew=1.2, color="0.3",
           label="model $M_H/BW_H+M_G/BW_G$", zorder=6)
    bh, bg = dense_link_bandwidths(g1)
    c.axhline(4.9, color="k", ls=":", lw=0.8); c.text(g1["n"].min(), 5.6, "HBM3e peak 4.9 TB/s", fontsize=6.0, va="bottom")
    c.axhline(0.45, color="k", ls=":", lw=0.8)
    c.text(g1["n"].min(), 0.48, "NVLink-C2C 450 GB/s", fontsize=6.0, va="bottom")
    if not np.isnan(bg):
        c.text(g1["n"].min(), bg / 1e12 * 0.85, f"Grace only: {bg/1e9:.0f} GB/s", fontsize=6.0, ha="left", va="top")
    c.set_xscale("log"); c.set_yscale("log"); c.set_ylim(0.1, 12)
    c.set_xlabel("$n$"); c.set_ylabel("matrix bandwidth $bn^2/T_{\\mathrm{step}}$ (TB/s)")
    c.set_title("FP32 (blue) and FP16 (red)", fontsize=7.5, loc="left")
    c.legend(fontsize=5.6, loc="upper right")
    _panel(c, "C")
    fig.tight_layout(w_pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# K2000 batch-size sweep (only drawn when B != 512 data exist)
# ---------------------------------------------------------------------------
def fig_k2000_batch(k2000_all: pd.DataFrame, steps: int = 3200):
    s = k2000_all[(k2000_all["steps"] == steps)]
    labels = ["bit", "gemm-fp16", "gemm-int8", "gemm-tf32", "public-matched"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for l in labels:
        d = s[s["label"] == l].sort_values("agents")
        if not len(d):
            continue
        kw = dict(marker=MARKER[l], color=COLOR[l], ls="--" if l.startswith("public") else "-", label=NAME[l])
        a.plot(d["agents"], d["us_per_step"], **kw)
        if l != "public-matched":
            b.plot(d["agents"], d["us_per_step"] / d["agents"] * 1e3, **kw)
    xs = sorted(s["agents"].unique())
    for ax in (a, b):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("replicas $B$")
        ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
    a.set_ylabel(f"time per step (µs), K2000, $K$={steps}")
    a.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=2)
    b.set_ylabel("time per replica-step (ns)")
    _panel(a, "A"); _panel(b, "B")
    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Bitwise identity of trajectories across paths
# ---------------------------------------------------------------------------
def fig_identity(ident: pd.DataFrame, steps: int = 3200):
    d = ident[(ident["steps"] == steps) & (ident["family"].isin(["K2000", "G-set", "QPLIB"]))]
    d = d[~d["label"].isin(["auto"])]
    piv = d.pivot_table(index="label", columns="instance", values="frac_identical")
    cols = sorted(piv.columns, key=lambda c: (c.startswith("QPLIB"), int("".join(ch for ch in c if ch.isdigit()) or 0)))
    piv = piv.reindex(columns=cols).reindex(order(piv.index))
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("QPLIB_", "Q") for c in cols], rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([NAME[l] for l in piv.index], fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("seeds identical to gemm TF32", fontsize=6.5)
    ax.set_title(f"Trajectory identity across execution paths ($K$={steps}); grey = path not run / not applicable", fontsize=7.5, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dense +-1 bit path over two GH200s (bit_dense_multi): HBM, then HBM + Grace on every GPU
# ---------------------------------------------------------------------------
def bit_summary(bit: pd.DataFrame) -> pd.DataFrame:
    """One row per (instance, n, gpus, placement): median step time over repeats and seeds, matrix split, and for
    the Mattis instances the number of seeds that reached the planted ground state."""
    g = bit.groupby(["instance", "n", "gpus", "placement"]).agg(
        tps=("time_per_step_s", "median"), mb=("matrix_bytes", "first"), hbm_b=("hbm_bytes", "first"),
        grace_b=("grace_bytes", "first"), gen_s=("generation_s", "median"), steps=("steps", "first"),
        runs=("time_per_step_s", "size"), seeds=("seed", "nunique"), found=("found", "sum"),
        expected=("expected_energy", "first"), suite=("suite", "first")).reset_index()
    g["bw_TBps"] = g["mb"] / g["tps"] / 1e12
    return g


def bit_link_bandwidths(g: pd.DataFrame) -> tuple[float, float]:
    """Aggregate (all GPUs) bandwidths of the bit path: HBM from the HBM-resident points, Grace from the hybrid
    points after subtracting the HBM share at the HBM rate (B/s)."""
    r = g[g["instance"] == "random"]
    h = r[r["placement"] == "hbm"]
    bh = (h["mb"] / h["tps"]).median() if len(h) else np.nan
    y = r[r["placement"] == "hybrid"]
    bg = (y["grace_b"] / (y["tps"] - y["hbm_b"] / bh)).median() if len(y) and not np.isnan(bh) else np.nan
    return bh, bg


def fig_bit_multi(bit: pd.DataFrame):
    """Panel A: time per step of the dense +-1 bit path on two GH200s against n, with the two-tier model
    T = M_H/BW_H + M_G/BW_G from the two measured aggregate bandwidths.  Panel B: coupling-matrix bandwidth."""
    g = bit_summary(bit)
    r = g[g["instance"] == "random"].sort_values("n")
    m = g[g["instance"] == "mattis"].sort_values("n")
    bh, bg = bit_link_bandwidths(g)
    gpus = int(r["gpus"].max())
    col = COLOR["bit"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ycol, scale in [(a, "tps", 1.0), (b, "bw_TBps", 1.0)]:
        for tier, mk, mfc, lab in [("hbm", "D", col, f"matrix in HBM ({gpus} GPUs)"),
                                   ("hybrid", "D", "white", f"HBM filled first, remainder in Grace memory")]:
            d = r[r["placement"] == tier]
            ax.plot(d["n"], d[ycol] * scale, ls="none", marker=mk, color=col, mfc=mfc, mec=col, mew=1.0, ms=5, label=lab)
        if len(m):
            mm = m.set_index("n").reindex(r["n"])
            ax.plot(r["n"], mm[ycol].values * scale, ls="none", marker="x", color="k", ms=5, mew=0.9,
                    label="Mattis instances (planted optimum found)")
    nn = np.logspace(np.log10(r["n"].min()), np.log10(r["n"].max()), 200)
    hb = r[r["placement"] == "hybrid"]["hbm_b"].median()          # HBM share of the hybrid points (fixed budget)
    mb = nn ** 2 / 8
    mh = np.minimum(mb, hb)
    tmodel = mh / bh + (mb - mh) / bg
    a.plot(nn, tmodel, ls=":", color="0.4", lw=1.0, label="$M_H/BW_H+M_G/BW_G$ (two measured bandwidths)")
    b.plot(nn, mb / tmodel / 1e12, ls=":", color="0.4", lw=1.0)
    for _, row in r[r["placement"] == "hybrid"].iterrows():
        a.annotate(f"{row['grace_b']/1e9:.0f} GB in Grace", (row["n"], row["tps"]), textcoords="offset points",
                   xytext=(9, -4), fontsize=5.8, color="0.3", ha="left", va="center")
    a.set_xlim(right=r["n"].max() * 2.2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$n$ (dense $\\pm1$ $J$, one bit per coupling)")
    a.set_ylabel(f"time per step (s), $B=1$, {gpus} GH200")
    a.legend(fontsize=5.6, loc="lower right")
    a.set_ylim(a.get_ylim()[0] * 0.6, a.get_ylim()[1] * 1.5)
    _panel(a, "A")
    b.axhline(gpus * 4.9, color="k", ls=":", lw=0.8)
    b.text(r["n"].min(), gpus * 4.9 * 1.12, f"{gpus}$\\times$ HBM3e peak", fontsize=6.2, va="bottom")
    b.axhline(gpus * 0.45, color="k", ls=":", lw=0.8)
    b.text(r["n"].min(), gpus * 0.45 * 1.08, f"{gpus}$\\times$ NVLink-C2C 450 GB/s", fontsize=6.2, va="bottom")
    b.text(r["n"].max(), bg / 1e12 * 0.9, f"Grace share streamed at {bg/1e9:.0f} GB/s ({gpus} GPUs)",
           fontsize=6.0, ha="right", va="top")
    b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(0.3, 20)
    b.set_xlabel("$n$"); b.set_ylabel("matrix bandwidth $n^2/8\\,/\\,T_{\\mathrm{step}}$ (TB/s)")
    for ax in (a, b):
        ax.set_xlim(4e5, 5e6)
        ax.set_xticks([5e5, 1e6, 2e6, 3e6]); ax.set_xticklabels(["0.5M", "1M", "2M", "3M"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _panel(b, "B")
    fig.tight_layout(w_pad=1.5)
    return fig
