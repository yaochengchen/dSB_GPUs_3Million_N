"""Load every raw result row under the results/ directory of one dsb-gpu tree.

Layout (`--root` is the dsb-gpu checkout; every run lives under `<root>/results/`):

    <root>/results/run_<suite>/<instance>_<batch>_<steps>/dsb_gpu_*_reduction*.csv, public_*_reduction*.csv
                                                            (raw.csv as fallback)        run_Gset.sh / run_qplib.sh / run_K2000.sh
    <root>/results/dense_scaling*/raw.csv                                                scripts/run_dense_scaling.sh
    <root>/results/scaling_sparse*/raw.csv (+ public_N*.csv)                             scripts/run_scaling.sh
    <root>/results/bit_multi_*/raw.csv, bit_mattis_*/raw.csv                             scripts/run_bit_multi.sh / run_bit_mattis.sh

Overlaps between suites: when the same configuration was run more than once, the LATEST run wins.
Benchmark runs: ordered by the start time on the first line of each run directory's environment.txt, then by
`suite_rank` (version number vN in the suite name, supplement flag, name) where no time stamp was written.
Dense scaling (no time stamps): ordered by `suite_rank`; completed runs are preferred over failed ones.
Bit multi-GPU runs: the directory name carries the start time, the latest directory wins per (instance, n, gpus, seed).
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def suite_rank(suite: str) -> tuple:
    m = re.search(r"_v(\d+)", suite)
    return (int(m.group(1)) if m else 0, int("supp" in suite), suite)


def results_dir(root: str) -> str:
    d = os.path.join(root, "results")
    if not os.path.isdir(d):
        raise SystemExit(f"{d} does not exist: --root must be the dsb-gpu checkout that contains results/")
    return d


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "implementation" not in df.columns:
        return None
    return df


def _read_env(env_path: str) -> tuple[float, float]:
    """(dt, run start as POSIX seconds) from a run directory's environment.txt; NaN where absent."""
    try:
        txt = open(env_path).read()
    except OSError:
        return np.nan, np.nan
    m = re.search(r"\bdt=([0-9.]+)", txt)
    dt = float(m.group(1)) if m else np.nan
    first = txt.splitlines()[0].strip() if txt else ""
    t = np.nan
    for fmt in ("%a %b %d %I:%M:%S %p UTC %Y", "%a %b %d %H:%M:%S UTC %Y", "%a %d %b %Y %I:%M:%S %p UTC"):
        try:
            t = datetime.strptime(first, fmt).replace(tzinfo=timezone.utc).timestamp()
            break
        except ValueError:
            pass
    return dt, t


def _load_run_dir(d: str) -> list[pd.DataFrame]:
    # per-entry CSVs are preferred over raw.csv (QPLIB v5 raw.csv labels the auto entry with the chosen
    # variant only, which collides with the explicit run of that variant).  *.csv.tmp = interrupted, ignored.
    files = sorted(glob.glob(os.path.join(d, "dsb_gpu_*_reduction*.csv"))) + \
            sorted(glob.glob(os.path.join(d, "public_*_reduction*.csv")))
    dfs = []
    for f in files:
        df = _read_csv(f)
        if df is None:
            continue
        m = re.match(r"dsb_gpu_(.+?)-(fp32|fp16|int8)(-notf32)?_reduction", os.path.basename(f))
        if m and m.group(1) == "auto":
            v = df["variant"].astype(str)
            df["variant"] = np.where(v.str.startswith("auto->"), v, "auto->" + v)
        dfs.append(df)
    if not dfs and os.path.exists(os.path.join(d, "raw.csv")):
        df = _read_csv(os.path.join(d, "raw.csv"))
        dfs = [df] if df is not None else []
    return dfs


def load_benchmarks(root: str) -> pd.DataFrame:
    frames, empty_suites, done_empty = [], [], []
    for run_dir in sorted(p for p in glob.glob(os.path.join(results_dir(root), "run_*")) if os.path.isdir(p)):
        suite = os.path.basename(run_dir)
        n_suite = 0
        for d in sorted(glob.glob(os.path.join(run_dir, "*_*_*"))):
            if not os.path.isdir(d):
                continue
            dfs = _load_run_dir(d)
            n = sum(len(x) for x in dfs)
            if n:
                dt, t = _read_env(os.path.join(d, "environment.txt"))
                for df in dfs:
                    frames.append(df.assign(suite=suite, run_dir=os.path.basename(d), dt=dt, run_time=t))
            elif os.path.exists(os.path.join(d, ".done")):
                done_empty.append(f"{suite}/{os.path.basename(d)}")
            n_suite += n
        if n_suite == 0:
            empty_suites.append(suite)
    for s in empty_suites:
        print(f"  WARNING: {s} contains no result rows (every entry failed or was interrupted)")
    if done_empty:
        print(f"  WARNING: {len(done_empty)} run directories are marked .done but have no readable rows "
              f"(empty or zero-byte files), e.g. {done_empty[0]}")
    if not frames:
        return pd.DataFrame()
    return _normalise(pd.concat(frames, ignore_index=True))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["is_public"] = df["implementation"].astype(str).str.startswith("simulated-bifurcation")
    v = df["variant"].astype(str)
    df["auto_choice"] = np.where(v.str.startswith("auto->"), v.str.replace("auto->", "", regex=False), "")
    df["variant_base"] = np.where(v.str.startswith("auto->"), "auto", v)
    lab = df["variant_base"].copy()
    prec = df["precision"].astype(str)
    lab = np.where((lab == "gemm") & (prec == "fp32"), "gemm-tf32", lab)
    lab = np.where((lab == "gemm") & (prec == "fp16"), "gemm-fp16", lab)
    lab = np.where((lab == "gemm") & (prec == "int8"), "gemm-int8", lab)
    lab = np.where(lab == "gemm-notf32", "gemm-fp32", lab)
    for base in ("block", "cluster", "global-sync"):
        lab = np.where((df["variant_base"] == base) & (prec == "fp16"), base + "-fp16", lab)
    lab = np.where(v == "discrete-matched", "public-matched", lab)
    lab = np.where(v == "discrete", "public-library", lab)
    df["label"] = lab
    for c in ("objective", "target", "solver_s", "wall_s", "total_s", "setup_s", "time_per_step_s",
              "n_original", "edges", "agents", "steps", "gpu_memory_bytes", "coupling_bytes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    tgt = df.dropna(subset=["target"]).groupby("instance")["target"].first()   # agents sweep left target blank
    df["target"] = df["target"].fillna(df["instance"].map(tgt))
    df["success"] = (df["objective"] >= df["target"]).astype(float)           # every suite is a maximisation
    df["us_per_step"] = df["solver_s"] / df["steps"] * 1e6
    df["gap_pct"] = (df["target"] - df["objective"]) / df["target"].abs().clip(lower=1) * 100
    return df


def load_dense_scaling(root: str) -> pd.DataFrame:
    """All dense-scaling rows, deduplicated per (impl, precision, batch, n, mode).

    mode = "hybrid" for MATRIX_MEMORY=hybrid runs (rows split between HBM and Grace memory), else "auto" (HBM while
    it fits, otherwise the whole matrix in Grace memory).  tier = where the matrix actually was: hbm, grace, hybrid.
    The newest suite wins among completed rows; a configuration that never completed keeps the newest failure."""
    frames = []
    for raw in sorted(glob.glob(os.path.join(results_dir(root), "dense_scaling*", "raw.csv"))):
        df = _read_csv(raw)
        if df is not None:
            frames.append(df.assign(suite=os.path.basename(os.path.dirname(raw))))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ("n", "batch", "gpu_s", "time_per_step_s", "matrix_bytes", "hbm_bytes", "grace_bytes", "steps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["impl"] = np.where(df["implementation"].str.contains("pytorch"), "torch", "dsb-gpu")
    req = df["requested_memory"].astype(str) if "requested_memory" in df.columns else pd.Series("", index=df.index)
    sel = df["selected_memory"].astype(str) if "selected_memory" in df.columns else pd.Series("", index=df.index)
    df["mode"] = np.where(req == "hybrid", "hybrid", "auto")
    df["tier"] = np.where(sel == "hybrid", "hybrid", np.where(df["grace_bytes"].fillna(0) > 0, "grace", "hbm"))
    key = ["impl", "precision", "batch", "n", "mode"]
    df["_ok"] = (df["status"] == "completed").astype(int)
    df["_rank"] = df["suite"].map(lambda s: suite_rank(s))
    top = df.sort_values(["_ok", "_rank"]).groupby(key)[["_ok", "_rank"]].last()
    df = df.merge(top.rename(columns={"_ok": "_bok", "_rank": "_brank"}), left_on=key, right_index=True)
    df = df[(df["_ok"] == df["_bok"]) & (df["_rank"] == df["_brank"])]
    return df.drop(columns=["_ok", "_rank", "_bok", "_brank"]).reset_index(drop=True)


def load_sparse_scaling(root: str) -> pd.DataFrame:
    frames = []
    pats = [("scaling_sparse*", "raw.csv"), ("scaling_sparse*", "*/raw.csv"), ("large_scale*", "sparse/*/raw.csv"),
            ("scaling_sparse*", "public_*.csv"), ("large_scale*", "sparse/public_*.csv")]
    for d, f in pats:
        for p in sorted(glob.glob(os.path.join(results_dir(root), d, f))):
            df = _read_csv(p)
            if df is not None:
                frames.append(df.assign(suite=os.path.basename(os.path.dirname(p)), run_dir=os.path.basename(p)))
    if not frames:
        return pd.DataFrame()
    return _normalise(pd.concat(frames, ignore_index=True))


def load_bit_multi(root: str) -> pd.DataFrame:
    """Dense +-1 bit path over 1-2 GPUs (apps/bit_dense_multi.cu): results/bit_multi_*/raw.csv (random +-1 J,
    `instance` = random) and results/bit_mattis_*/raw.csv (planted Mattis instances, `instance` = mattis, with
    `expected_energy` and `found`).  Completed rows only; the latest directory wins per (instance, n, gpus, seed)."""
    frames = []
    for raw in sorted(glob.glob(os.path.join(results_dir(root), "bit_m*_*", "raw.csv"))):
        df = _read_csv(raw)
        if df is not None:
            frames.append(df.assign(suite=os.path.basename(os.path.dirname(raw))))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].astype(str) == "completed"].copy()
    if "instance" not in df.columns:            # header of an older bit_multi run
        df["instance"], df["seed"], df["found"], df["expected_energy"] = "random", 42, "", np.nan
    df["instance"] = df["instance"].fillna("random")
    for c in ("n", "gpus", "batch", "steps", "matrix_bytes", "hbm_bytes", "grace_bytes", "generation_s",
              "time_per_step_s", "effective_matrix_GB_s", "energy", "expected_energy", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["found"] = pd.to_numeric(df["found"], errors="coerce").fillna(0).astype(int)
    df["seed"] = df["seed"].fillna(42).astype(int)
    key = ["instance", "n", "gpus", "seed"]
    latest = df.groupby(key)["suite"].transform("max")
    return df[df["suite"] == latest].reset_index(drop=True)
