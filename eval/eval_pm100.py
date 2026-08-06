#!/usr/bin/env python3
"""
Validation against Dataset D: PM100 — HPC Job Power Dataset
(Antici et al., SC'23 — Zenodo:10127767 | github.com/francescoantici/PM100-data)

231,116 jobs from Marconi100 HPC (IBM Power9 + NVIDIA V100).
Each job: allocated CPU cores, memory, job type → time-series power.

Mapping to EnergetiScope feature schema:
  cpu_request_mcpu  ← ncpus * 1000
  mem_request_mib   ← mem_gb * 1024
  workload_kind     ← "Job"  (all HPC batch jobs)
  (workload_name treated as queue name for group-aware CV)

This is a cross-domain transfer test: validates that KNN on resource specs → power
generalises beyond Kubernetes-native infrastructure.

Usage:
    python eval/eval_pm100.py [--data data/external/pm100] [--download]
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ZENODO_URL = "https://zenodo.org/api/records/10127767/files/job_table.parquet/content"
PARQUET_FILENAME = "job_table.parquet"


# ── helpers ──────────────────────────────────────────────────────────────────

def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def cv_metrics(X, y, model_factory, groups=None, n_splits=5, random_state=42):
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))

    maes, rmses, r2s, mapes = [], [], [], []
    for tr, va in splits:
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[va])
        maes.append(mean_absolute_error(y[va], p))
        rmses.append(np.sqrt(mean_squared_error(y[va], p)))
        r2s.append(r2_score(y[va], p))
        mapes.append(mape(y[va], p))
    return {
        "MAE":  (np.mean(maes),  np.std(maes)),
        "RMSE": (np.mean(rmses), np.std(rmses)),
        "R2":   (np.mean(r2s),   np.std(r2s)),
        "MAPE": (np.mean(mapes), np.std(mapes)),
    }


def print_metrics(label, metrics):
    print(f"\n  [{label}]")
    for k, (mean, std) in metrics.items():
        print(f"    {k:6s}: {mean:10.3f} ± {std:.3f}")


# ── download ──────────────────────────────────────────────────────────────────

def download_pm100(dest_dir):
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, PARQUET_FILENAME)
    if os.path.exists(dest):
        print(f"  [SKIP] Already downloaded: {dest}")
        return dest
    print("  Downloading PM100 from Zenodo (~287 MB)...")
    with requests.get(ZENODO_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%  ({downloaded/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print(f"\n  [OK] Saved to {dest}")
    return dest


# ── data loading & feature engineering ────────────────────────────────────────

def load_pm100(path, max_rows=None):
    """
    Load PM100 Parquet and map HPC job columns to EnergetiScope-analogous features.

    PM100 actual schema (Marconi100 HPC, job_table.parquet):
      num_cores_alloc       — allocated CPU cores (int64)
      mem_alloc             — allocated memory (MB units)
      num_nodes_alloc       — number of compute nodes
      node_power_consumption — array of per-node power readings (W)
      partition             — SLURM partition (queue name)
      run_time              — job runtime in seconds
      num_gpus_alloc        — allocated GPU count

    Returns DataFrame with columns:
      cpu_request_mcpu, mem_request_mib, walltime_s, gpu_count,
      queue (grouping key), avg_power_w (label — mean per-node power × nodes)
    """
    df = pd.read_parquet(path)
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # node_power_consumption is stored as a dict {node_id: power_w}
    # Compute mean per-node power and total job power
    def mean_power(val):
        if val is None:
            return float("nan")
        if isinstance(val, dict):
            vals = list(val.values())
        elif hasattr(val, '__iter__') and not isinstance(val, str):
            vals = list(val)
        else:
            return float(val)
        if len(vals) == 0:
            return float("nan")
        return float(np.nanmean([float(v) for v in vals]))

    print("  Computing per-job mean power from node_power_consumption arrays...")
    df["avg_power_w"] = df["node_power_consumption"].apply(mean_power)

    out = pd.DataFrame()
    out["cpu_request_mcpu"] = pd.to_numeric(df["num_cores_alloc"], errors="coerce") * 1000.0
    out["mem_request_mib"]  = pd.to_numeric(df["mem_alloc"], errors="coerce")         # already MB
    out["num_nodes"]        = pd.to_numeric(df["num_nodes_alloc"], errors="coerce")
    out["walltime_s"]       = pd.to_numeric(df["run_time"], errors="coerce")
    out["gpu_count"]        = pd.to_numeric(df["num_gpus_alloc"], errors="coerce").fillna(0)
    out["avg_power_w"]      = df["avg_power_w"]
    out["queue"]            = df["partition"].astype(str)

    # Filter: must have valid cpu, power, and completed job
    out = out.dropna(subset=["cpu_request_mcpu", "avg_power_w"])
    out = out[(out["cpu_request_mcpu"] > 0) & (out["avg_power_w"] > 0)]

    return out


# ── main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(df):
    print(f"\n  Samples after filtering: {len(df)}")
    print(f"  CPU range: [{df['cpu_request_mcpu'].min():.0f}, {df['cpu_request_mcpu'].max():.0f}] mcpu")
    print(f"  Power range: [{df['avg_power_w'].min():.1f}, {df['avg_power_w'].max():.1f}] W")
    print(f"  Queue groups: {df['queue'].nunique()}")

    feature_cols = ["cpu_request_mcpu"]
    if "mem_request_mib" in df.columns and df["mem_request_mib"].std() > 0:
        feature_cols.append("mem_request_mib")
    if "walltime_s" in df.columns and df["walltime_s"].notna().sum() > len(df) * 0.5:
        feature_cols.append("walltime_s")

    print(f"  Features used: {feature_cols}")

    # Subsample for speed if very large
    if len(df) > 50_000:
        df_eval = df.sample(50_000, random_state=42)
        print("  Subsampled to 50,000 rows for CV speed")
    else:
        df_eval = df.copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_eval[feature_cols].fillna(0).values)
    y = df_eval["avg_power_w"].values

    n_splits = 5
    print(f"\n  {n_splits}-fold cross-validation\n")

    knn5  = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=5,  metric="euclidean"), n_splits=n_splits)
    knn10 = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=10, metric="euclidean"), n_splits=n_splits)
    lr    = cv_metrics(X, y, LinearRegression, n_splits=n_splits)

    print_metrics("KNN k=5  (EnergetiScope default)", knn5)
    print_metrics("KNN k=10", knn10)
    print_metrics("Linear Regression", lr)

    # Group-aware CV by queue (mimics leave-one-workload-type-out)
    n_queues = df_eval["queue"].nunique()
    if n_queues >= 3:
        groups = df_eval["queue"].values
        n_g_splits = min(5, n_queues)
        knn5_g = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=5, metric="euclidean"),
                             groups=groups, n_splits=n_g_splits)
        print_metrics(f"KNN k=5  (group-by-queue CV, {n_g_splits} folds)", knn5_g)

    # Mean baseline
    mean_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    print(f"\n  [Mean Baseline]  MAE = {mean_mae:.3f} W")

    return {"KNN_k5": knn5, "LinearRegression": lr, "n_samples": len(df_eval)}


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/external/pm100",
                   help="Directory containing PM100.parquet")
    p.add_argument("--download", action="store_true",
                   help="Download PM100.parquet from Zenodo if not present")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Limit rows loaded (for quick testing)")
    args = p.parse_args()

    parquet_path = os.path.join(args.data, PARQUET_FILENAME)

    if not os.path.exists(parquet_path):
        if args.download:
            parquet_path = download_pm100(args.data)
        else:
            sys.exit(
                f"[ERROR] {parquet_path} not found.\n"
                "Re-run with --download to fetch from Zenodo (~287 MB), or\n"
                "manually download from https://zenodo.org/records/10127767"
            )

    print("\n" + "="*65)
    print("Dataset D — PM100 HPC Job Power Dataset (Zenodo:10127767)")
    print("="*65)
    print("Cross-domain transfer test: HPC resource specs → node power")
    print("(IBM Power9 + V100; analogous to K8s pod CPU/memory → power)")

    df = load_pm100(parquet_path, max_rows=args.max_rows)
    results = run_evaluation(df)

    print("\n" + "="*65)
    print("Summary: Dataset D (PM100)")
    print("="*65)
    knn5 = results["KNN_k5"]
    print("  KNN k=5:")
    print(f"    MAE  = {knn5['MAE'][0]:.2f} W   RMSE = {knn5['RMSE'][0]:.2f} W")
    print(f"    R²   = {knn5['R2'][0]:.4f}      MAPE = {knn5['MAPE'][0]:.1f}%")
    lr = results["LinearRegression"]
    print("  Linear Regression:")
    print(f"    MAE  = {lr['MAE'][0]:.2f} W   R²   = {lr['R2'][0]:.4f}")
    print()


if __name__ == "__main__":
    main()
