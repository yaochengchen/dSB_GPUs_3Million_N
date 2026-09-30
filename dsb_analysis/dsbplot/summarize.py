"""Tidy per-configuration summaries built from raw rows."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .load import suite_rank, RELAXED_LEVELS

# benchmark family of a suite directory (None = not a main benchmark suite)
FAMILY_PATTERNS = [("K2000", r"^run_K2000_result"), ("G-set", r"^run_Gset_(result|int8)"), ("QPLIB", r"^run_qplib_result")]
MAIN_B = {"K2000": 512, "G-set": 512}      # K2000 / G-set main tables use B=512; QPLIB uses B chosen by n


def family_of(suite: str):
    for fam, pat in FAMILY_PATTERNS:
        if re.search(pat, suite):
            return fam
    return None


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one run per (instance, agents, steps, label): the latest one.  Latest = the start time recorded in
    the run directory's environment.txt; runs without a time stamp fall back to load.suite_rank."""
    t = df["run_time"].fillna(-np.inf) if "run_time" in df.columns else pd.Series(-np.inf, index=df.index)
    rank = [(ti,) + suite_rank(s) for ti, s in zip(t, df["suite"])]
    order = {r: i for i, r in enumerate(sorted(set(rank)))}
    df = df.assign(_pr=[order[r] for r in rank])
    best = df.groupby(["instance", "agents", "steps", "label"])["_pr"].transform("max")
    return df[df["_pr"] == best].drop(columns="_pr")


def bench_rows(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Deduplicated rows of one benchmark family (all batch sizes)."""
    fam = df["suite"].map(family_of)
    return dedupe(df[fam == family].copy()).assign(family=family)


def tts99(succ: float, n: int, t_median: float) -> float:
    """Operational TTS99 (ceil, at least one trial); inf when no success."""
    if succ <= 0:
        return np.inf
    p = min(succ / n, 1.0)
    if p >= 1.0:
        return t_median
    r = int(np.ceil(np.log(0.01) / np.log(1 - p)))
    return t_median * max(1, r)


def summarize(df: pd.DataFrame, by=("family", "instance", "agents", "steps", "label")) -> pd.DataFrame:
    by = [c for c in by if c in df.columns]
    for c in ("dt", "nnz_solver", "edges", "gpu_memory_bytes", "auto_choice"):
        if c not in df.columns:
            df = df.assign(**{c: np.nan if c != "auto_choice" else ""})
    g = df.groupby(by, dropna=False)
    out = g.agg(
        runs=("objective", "size"),
        n=("n_original", "first"),
        edges=("edges", "first"),
        nnz=("nnz_solver", "first"),
        target=("target", "first"),
        dt=("dt", "first"),
        auto_choice=("auto_choice", lambda s: ",".join(sorted(set(s) - {""}))),
        us_per_step=("us_per_step", "median"),
        solver_s=("solver_s", "median"),
        wall_s=("wall_s", "median"),
        total_s=("total_s", "median"),
        setup_s=("setup_s", "median"),
        best=("objective", "max"),
        median_obj=("objective", "median"),
        q1_obj=("objective", lambda s: s.quantile(0.25)),
        q3_obj=("objective", lambda s: s.quantile(0.75)),
        successes=("success", "sum"),
        **{f"successes_{t}": (f"success_{t}", "sum") for t in RELAXED_LEVELS},
        median_gap_pct=("gap_pct", "median"),
        best_gap_pct=("gap_pct", "min"),
        gpu_mem_bytes=("gpu_memory_bytes", "median"),
    ).reset_index()
    out["p_batch"] = out["successes"] / out["runs"]
    out["tts99_solver_s"] = [tts99(s, r, t) for s, r, t in zip(out["successes"], out["runs"], out["solver_s"])]
    out["tts99_wall_s"] = [tts99(s, r, t) for s, r, t in zip(out["successes"], out["runs"], out["wall_s"])]
    for t in RELAXED_LEVELS:
        out[f"tts99_{t}_solver_s"] = [tts99(s, r, tm) for s, r, tm in zip(out[f"successes_{t}"], out["runs"], out["solver_s"])]
    return out


def relaxed_targets(fastest: pd.DataFrame, summary: pd.DataFrame, steps: int,
                    ref_label: str = "public-matched") -> pd.DataFrame:
    """Per instance: the tightest target level (100 %, then 99.9, 99.5, 99 % of the best-known value) at which BOTH the
    fastest custom path and the reference reach the target in at least one trial, so that TTS99 is defined for both;
    99 % if neither level qualifies.  Returns the level and the two success counts and TTS99 values at that level."""
    levels = [("100", "", 1.0)] + [(t, f"_{t}", th) for t, th in RELAXED_LEVELS.items()]
    ref = summary[(summary["steps"] == steps) & (summary["label"] == ref_label)].set_index(["family", "instance"])
    rows = []
    for _, r in fastest.iterrows():
        p = ref.loc[(r["family"], r["instance"])]
        for tag, suf, th in levels:
            if r[f"successes{suf}"] > 0 and p[f"successes{suf}"] > 0:
                break
        rows.append(dict(family=r["family"], instance=r["instance"], level_pct=th * 100,
                         u_dsb=int(r[f"successes{suf}"]), u_sb=int(p[f"successes{suf}"]),
                         tts_dsb_s=r[f"tts99{suf}_solver_s"], tts_sb_s=p[f"tts99{suf}_solver_s"]))
    out = pd.DataFrame(rows)
    out["tts_ratio"] = out["tts_sb_s"] / out["tts_dsb_s"]
    return out


def with_speedup(summary: pd.DataFrame, ref_label: str = "public-matched") -> pd.DataFrame:
    """Add solver/wall speedup columns relative to the reference label of the same (instance, agents, steps)."""
    key = ["family", "instance", "agents", "steps"]
    key = [k for k in key if k in summary.columns]
    ref = summary[summary["label"] == ref_label][key + ["solver_s", "wall_s"]].rename(
        columns={"solver_s": "ref_solver_s", "wall_s": "ref_wall_s"})
    out = summary.merge(ref, on=key, how="left")
    out["speedup_solver"] = out["ref_solver_s"] / out["solver_s"]
    out["speedup_wall"] = out["ref_wall_s"] / out["wall_s"]
    return out


def fastest_per_instance(summary: pd.DataFrame, steps: int, metric: str = "solver_s",
                         exclude=("auto", "public-matched", "public-library")) -> pd.DataFrame:
    s = summary[(summary["steps"] == steps) & ~summary["label"].isin(exclude)]
    idx = s.groupby(["family", "instance"])[metric].idxmin()
    return s.loc[idx].reset_index(drop=True)


def bitwise_identity(df: pd.DataFrame, ref_label: str = "gemm-tf32") -> pd.DataFrame:
    """Fraction of seeds whose final objective equals the reference path's, per instance/steps/label."""
    rows = []
    for (fam, inst, ag, st), grp in df.groupby(["family", "instance", "agents", "steps"]):
        piv = grp.pivot_table(index="seed", columns="label", values="objective", aggfunc="first")
        if ref_label not in piv:
            continue
        ref = piv[ref_label]
        for lab in piv.columns:
            m = piv[lab].notna() & ref.notna()
            if m.sum() == 0:
                continue
            rows.append(dict(family=fam, instance=inst, agents=ag, steps=st, label=lab,
                             n_seeds=int(m.sum()), frac_identical=float((piv[lab][m] == ref[m]).mean())))
    return pd.DataFrame(rows)
