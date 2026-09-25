"""LaTeX table fragments (booktabs) generated from the summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import NAME

TEX_NAME = {k: v.replace("µ", "\\textmu ") for k, v in NAME.items()}
TEX_NAME.update({"gemm-fp16": "\\texttt{gemm} FP16", "gemm-int8": "\\texttt{gemm} INT8", "gemm-tf32": "\\texttt{gemm} TF32",
                 "gemm-fp32": "\\texttt{gemm} FP32 (no TF32)", "bit": "\\texttt{bit}", "block": "\\texttt{block}",
                 "csr-row": "\\texttt{csr-row}", "csr-block": "\\texttt{csr-block}", "auto": "\\texttt{auto}",
                 "public-matched": "SB 2.0.0 (matched)", "public-library": "SB 2.0.0 (library)"})


def _f(x, nd=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "---"
    return f"{x:.{nd}f}"


def _tts(x):
    return "unresolved" if np.isinf(x) else f"{x:.3g}"


def k2000_table(summ: pd.DataFrame, steps: int, labels) -> str:
    s = summ[(summ["family"] == "K2000") & (summ["steps"] == steps)].set_index("label")
    B = int(s["agents"].iloc[0]); runs = int(s["runs"].iloc[0]); tgt = int(s["target"].iloc[0])
    rows = []
    for l in labels:
        if l not in s.index:
            continue
        r = s.loc[l]
        rows.append(f"{TEX_NAME[l]} & {_f(r['us_per_step'])} & {_f(r['solver_s']*1e3)} & "
                    f"{int(r['best'])} & {_f(r['median_obj'])} & {int(r['successes'])}/{runs} & {_tts(r['tts99_solver_s'])} \\\\")
    body = "\n".join(rows)
    return (f"% auto-generated: K2000, B={B}, K={steps}, target {tgt}, matched SB 2.0.0 baseline\n"
            "\\begin{tabular}{lrrrrrr}\\toprule\n"
            "Path & \\textmu s/step & Solve (ms) & Best & Median & $u/M$ & TTS$_{99}$ (s)\\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n")


def steps_table(summ: pd.DataFrame, labels) -> str:
    s = summ[summ["family"] == "K2000"]
    ks = sorted(s["steps"].unique())
    head = " & ".join(f"$K={int(k)}$" for k in ks)
    out = [f"\\begin{{tabular}}{{l{'r'*len(ks)}}}\\toprule", f" & {head}\\\\\\midrule",
           f"\\multicolumn{{{len(ks)+1}}}{{l}}{{\\emph{{Median integration time}} (ms)}}\\\\"]
    for l in labels:
        d = s[s["label"] == l].set_index("steps")
        if d.empty:
            continue
        out.append(f"{TEX_NAME[l]} & " + " & ".join(_f(d.loc[k, 'solver_s'] * 1e3, 2) if k in d.index else "---" for k in ks) + "\\\\")
    out.append("\\midrule")
    out.append(f"\\multicolumn{{{len(ks)+1}}}{{l}}{{\\emph{{Median objective}} / trials reaching target}}\\\\")
    for l in labels:
        d = s[s["label"] == l].set_index("steps")
        if d.empty:
            continue
        out.append(f"{TEX_NAME[l]} & " + " & ".join(
            f"{_f(d.loc[k,'median_obj'])} ({int(d.loc[k,'successes'])})" if k in d.index else "---" for k in ks) + "\\\\")
    out.append("\\bottomrule\\end{tabular}")
    return "% auto-generated: K2000 step-budget sweep, B=512, 50 trials per cell\n" + "\n".join(out) + "\n"


def instance_table(fastest: pd.DataFrame, summ: pd.DataFrame, family: str, steps: int) -> str:
    f = fastest[fastest["family"] == family].sort_values("n")
    pub = summ[(summ["family"] == family) & (summ["steps"] == steps) & (summ["label"] == "public-matched")].set_index("instance")
    nnz = summ[summ["family"] == family].groupby("instance")["nnz"].max()
    has_deg = f["edges"].notna().any() or nnz.notna().any()
    rows = []
    for _, r in f.iterrows():
        p = pub.loc[r["instance"]]
        e = r["edges"] if not np.isnan(r["edges"]) else nnz.get(r["instance"], np.nan) / 2
        deg = 2 * e / r["n"]
        inst = r['instance'].replace('_', '\\_')
        degcol = f"{_f(deg)} & " if has_deg else ""
        rows.append(f"{inst} & {int(r['n'])} & {degcol}{int(r['agents'])} & {_f(r['dt'],2)} & {TEX_NAME[r['label']]} & "
                    f"{_f(r['us_per_step'])} & {_f(p['us_per_step'])} & {_f(r['speedup_solver'])} & "
                    f"{_f(r['median_gap_pct'],3)} & {_f(p['median_gap_pct'],3)} & {int(r['successes'])}/{int(p['successes'])} \\\\")
    return (f"% auto-generated: {family}, K={steps}, fastest dsb-gpu path (integration time) vs matched SB 2.0.0\n"
            + ("\\begin{tabular}{lrrrrlrrrrrr}\\toprule\n" if has_deg else "\\begin{tabular}{lrrrlrrrrrr}\\toprule\n")
            + "Instance & $n$ & " + ("deg & " if has_deg else "") + "$B$ & $\\Delta t$ & Fastest path & \\textmu s/step & SB \\textmu s/step & Speedup & Gap (\\%) & SB gap (\\%) & $u$ dsb/SB\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def arith_table(summ: pd.DataFrame, ident: pd.DataFrame, steps: int) -> str:
    """Runtime ratio of each arithmetic mode against gemm TF32 on the same instance (median over instances)."""
    labels = ["gemm-tf32", "gemm-fp32", "gemm-int8", "gemm-fp16", "bit", "block", "csr-row", "csr-block"]
    s = summ[summ["steps"] == steps]
    ref = s[s["label"] == "gemm-tf32"].set_index(["family", "instance"])["solver_s"]
    rows = []
    for l in labels:
        d = s[s["label"] == l].set_index(["family", "instance"])
        ratio = (ref.reindex(d.index) / d["solver_s"]).dropna()
        cells = []
        for fam in ["K2000", "G-set", "QPLIB"]:
            v = ratio[ratio.index.get_level_values(0) == fam]
            cells.append(f"{v.median():.2f} ({len(v)})" if len(v) else "---")
        rows.append(f"{TEX_NAME[l]} & " + " & ".join(cells) + "\\\\")
    return ("% auto-generated: ratio = t(gemm TF32)/t(path) on the same instance, median over instances (count)\n"
            "\\begin{tabular}{lccc}\\toprule\nPath & K2000 & G-set & QPLIB\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def capacity_table(dense_sum: pd.DataFrame, dense_all: pd.DataFrame, bit_sum: pd.DataFrame | None = None) -> str:
    """Analytic dense capacity and the measured HBM / Grace operating points of dsb-gpu gemm; the dense +-1 bit row
    takes its measured points from bit_dense_multi (two GPUs, marked with a dagger)."""
    hbm_budget, total = 0.85 * 142.5e9, 0.85 * 142.5e9 + 480e9
    fmt = [("FP32 / TF32", 4, "fp32"), ("FP16", 2, "fp16"), ("INT8", 1, None),
           ("bit, ternary", 0.25, None), ("bit, dense $\\pm1$", 0.125, None)]
    d = dense_sum[dense_sum["impl"] == "dsb-gpu"]
    t = dense_all[dense_all["impl"] == "torch"]
    rows = []
    for name, b, prec in fmt:
        n1, n2 = np.sqrt(hbm_budget / b), np.sqrt(total / b)
        cells = ["\\blank", "\\blank", "\\blank"]
        if b == 0.125 and bit_sum is not None and len(bit_sum):
            r = bit_sum[bit_sum["instance"] == "random"]
            h, g = r[r["placement"] == "hbm"], r[r["placement"] == "hybrid"]
            if len(h):
                cells[0] = f"{int(h['n'].max()):,}".replace(",", "{,}") + "$^\\dagger$"
            if len(g):
                cells[1] = f"{int(g['n'].max()):,}".replace(",", "{,}") + f" ({g['mb'].max()/1e9:.0f}\\,GB)$^\\dagger$"
        if prec:
            h = d[(d["precision"] == prec) & (d["tier"] == "hbm")]["n"].max()
            g = d[(d["precision"] == prec) & d["tier"].isin(["grace", "hybrid"])]
            tok = t[(t["precision"] == prec) & (t["status"] == "completed")]["n"].max()
            cells[0] = f"{int(h):,}".replace(",", "{,}")
            cells[1] = f"{int(g['n'].max()):,}".replace(",", "{,}") + f" ({g['mb'].max()/1e9:.0f}\\,GB)" if len(g) else "\\blank"
            cells[2] = f"{int(tok):,}".replace(",", "{,}")
        bs = {4: "4", 2: "2", 1: "1", 0.25: "$1/4$", 0.125: "$1/8$"}[b]
        big = lambda v: f"{round(v, -3):,.0f}".replace(",", "{,}")
        rows.append(f"{name} & {bs} & {big(n1)} & {big(n2)} & " + " & ".join(cells) + " \\\\")
    return ("% auto-generated: analytic n_max = sqrt(M/b); measured = largest n that ran (dsb-gpu gemm, K=50; "
            "dagger = bit_dense_multi on two GH200s, K=50, B=1)\n"
            "\\begin{tabular}{lcccccc}\\toprule\n"
            "Format & Bytes/coupling & $n_{\\max}$, 121\\,GB & $n_{\\max}$, 601\\,GB & HBM, ran & HBM+Grace, ran & PyTorch, ran\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def grace_table(dense_sum: pd.DataFrame) -> str:
    """Beyond HBM: dsb-gpu gemm with the whole matrix in Grace memory and with the hybrid HBM+Grace split."""
    d = dense_sum[dense_sum["impl"] == "dsb-gpu"]
    gr = d[(d["mode"] == "auto") & (d["tier"] == "grace")].set_index(["precision", "n", "batch"])
    hy = d[d["tier"] == "hybrid"].set_index(["precision", "n", "batch"])
    keys = sorted(set(k[:2] for k in gr.index) | set(k[:2] for k in hy.index), key=lambda k: (k[0] != "fp32", k[1]))
    rows = []
    for prec, n in keys:
        def cell(tab, b, col="tps", fmt=lambda v: f"{v:.3f}"):
            return fmt(tab.loc[(prec, n, b), col]) if (prec, n, b) in tab.index else "---"
        mb = (gr if (prec, n, 1) in gr.index else hy).loc[(prec, n, 1), "mb"] / 1e9
        g_bw = cell(gr, 1, "bw_TBps", lambda v: f"{v*1e3:.0f}")
        h_share = cell(hy, 1, "grace_b", lambda v: f"{v/1e9:.0f}")
        sp = (f"{gr.loc[(prec, n, 1), 'tps'] / hy.loc[(prec, n, 1), 'tps']:.2f}"
              if (prec, n, 1) in gr.index and (prec, n, 1) in hy.index else "---")
        rows.append(f"{prec.upper()} & {n:,} & {mb:.0f} & {cell(gr, 1)} & {cell(gr, 64)} & {g_bw} & "
                    f"{h_share} & {cell(hy, 1)} & {cell(hy, 64)} & {cell(hy, 1, 'bw_TBps', lambda v: f'{v*1e3:.0f}')} & {sp} \\\\"
                    .replace(",", "{,}", 1))
    return ("% auto-generated: dsb-gpu gemm beyond the HBM limit, K=50, median of 3 repetitions; GB/s = bn^2/T_step at B=1\n"
            "\\begin{tabular}{lrrrrrrrrrr}\\toprule\n"
            " & & & \\multicolumn{3}{c}{Grace only} & \\multicolumn{4}{c}{Hybrid HBM+Grace} & \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-10}\n"
            "Operands & $n$ & Matrix (GB) & $B=1$ (s) & $B=64$ (s) & GB/s & In Grace (GB) & $B=1$ (s) & $B=64$ (s) & GB/s & Speedup\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def bit_multi_table(bit_sum: pd.DataFrame) -> str:
    """Dense +-1 bit path on 1-2 GH200s: matrix split, step time and bandwidth on random instances, and the
    Mattis check (seeds that reached the planted ground state -n(n-1)/2)."""
    r = bit_sum[bit_sum["instance"] == "random"].set_index("n").sort_index()
    m = bit_sum[bit_sum["instance"] == "mattis"].set_index("n")
    gpus = int(bit_sum["gpus"].max())
    rows = []
    for n, x in r.iterrows():
        place = {"hbm": "HBM", "hybrid": "HBM + Grace", "grace": "Grace"}[x["placement"]]
        if n in m.index:
            y = m.loc[n]
            mat_cell = f"{int(y['found'])}/{int(y['seeds'])} & {y['tps']:.4f}"
        else:
            mat_cell = "\\blank & \\blank"
        ncell = f"{int(n):,}".replace(",", "{,}")
        rows.append(f"{ncell} & {x['mb']/1e9:.0f} & {place} & {x['hbm_b']/1e9:.0f} & {x['grace_b']/1e9:.0f} & "
                    f"{x['tps']:.4f} & {x['mb']/x['tps']/1e9:.0f} & {mat_cell} \\\\")
    return (f"% auto-generated: bit_dense_multi, {gpus} GH200s, K=50, B=1; random +-1 J: median of 3 repetitions; "
            "Mattis: one run per seed, found = energy equals -n(n-1)/2\n"
            "\\begin{tabular}{rrlrrrrrr}\\toprule\n"
            " & & & \\multicolumn{2}{c}{Split (GB)} & \\multicolumn{2}{c}{Random $\\pm1$ $J$} & \\multicolumn{2}{c}{Mattis} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "$n$ & Matrix (GB) & Placement & HBM & Grace & $T_{\\mathrm{step}}$ (s) & GB/s & Found & $T_{\\mathrm{step}}$ (s)\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
