"""Flag benchmark result files whose `bit` timings were disturbed by another process on the GPU.

A disturbed run shows integer-multiple slowdowns of a few repeats (e.g. bit 21.8 -> 46 or 69 us/step), so the
max/min ratio of the per-step time over the repeats in one file is a cheap detector.  Run from the repository
root after a campaign:

    python3 python/check_contamination.py            # scans results/run_*/ and results/*_v*/
"""
import glob
import pandas as pd

TIME_CANDIDATES = ["time_per_step_s", "median_time_per_step_s", "gpu_time_per_step_s", "step_time_s", "time_per_step"]


def per_step(r):
    for c in TIME_CANDIDATES:
        if c in r.columns:
            return pd.to_numeric(r[c], errors="coerce")
    if "gpu_s" in r.columns and "steps" in r.columns:
        return pd.to_numeric(r["gpu_s"], errors="coerce") / pd.to_numeric(r["steps"], errors="coerce")
    if "solver_s" in r.columns and "steps" in r.columns:
        return pd.to_numeric(r["solver_s"], errors="coerce") / pd.to_numeric(r["steps"], errors="coerce")
    return None


files = sorted(set(glob.glob("results/run_*/**/*.csv", recursive=True) +
                   glob.glob("results/*_v*/**/*.csv", recursive=True)))

bad = skipped = checked = 0
for f in files:
    try:
        r = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        skipped += 1
        continue
    if "variant" not in r.columns:
        skipped += 1
        continue
    t = per_step(r)
    if t is None:
        skipped += 1
        continue
    mask = r["variant"].astype(str).str.contains("bit", na=False)
    if "status" in r.columns:
        mask &= r["status"].astype(str).eq("completed")
    b = t[mask].dropna()
    b = b[b > 0]
    if len(b) < 2:
        continue
    checked += 1
    if b.max() / b.min() > 1.05:
        bad += 1
        print(f"disturbed: {f}   bit per-step min={b.min()*1e6:.1f} us  max={b.max()*1e6:.1f} us  n={len(b)}")

print(f"checked {checked} files with bit rows, skipped {skipped}")
print("all clean" if bad == 0 else f"{bad} files need a re-run")
