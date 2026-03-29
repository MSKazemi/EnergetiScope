#!/usr/bin/env python3
"""
Master evaluation runner — runs all available experiments and produces a
combined results table suitable for inclusion in the paper.

Usage:
    python eval/eval_all.py [--pm100-download]
"""
import argparse
import os
import sys
import subprocess
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def cv5(X, y, model_factory):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes, rmses, r2s, mapes = [], [], [], []
    for tr, va in kf.split(X, y):
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[va])
        maes.append(mean_absolute_error(y[va], p))
        rmses.append(np.sqrt(mean_squared_error(y[va], p)))
        r2s.append(r2_score(y[va], p))
        mapes.append(mape(y[va], p))
    return np.mean(maes), np.mean(rmses), np.mean(r2s), np.mean(mapes)


# ── Dataset A ─────────────────────────────────────────────────────────────────

def run_dataset_a(data_dir="data/external/zenodo_14332659"):
    results = []
    csv_files = [
        os.path.join(data_dir, "data", f"_{ts}-BootstrapInitializer-containerUtilization.csv")
        for ts in ["1729087088", "1729099594", "1729113884"]
    ]
    frames = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"cpuSeconds": "cpu_seconds", "powerWatts": "power_w"})
    df = df.dropna(subset=["cpu_seconds", "power_w"])
    df = df[(df["cpu_seconds"] > 0) & (df["power_w"] > 0)]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[["cpu_seconds"]].values)
    y = df["power_w"].values

    knn_mae, knn_rmse, knn_r2, knn_mape = cv5(X, y, lambda: KNeighborsRegressor(n_neighbors=5))
    lr_mae,  lr_rmse,  lr_r2,  lr_mape  = cv5(X, y, LinearRegression)

    return {
        "dataset": "Zenodo:14332659\n(BootstrapInitializer)",
        "domain":  "K8s / stress-ng",
        "n": len(df),
        "feature": "CPU utilisation (cpu_seconds/15s)",
        "label": "Node power [W] (iDRAC)",
        "knn_mae":  knn_mae,  "knn_rmse":  knn_rmse,
        "knn_r2":   knn_r2,   "knn_mape":  knn_mape,
        "lr_mae":   lr_mae,   "lr_r2":     lr_r2,
    }


# ── Dataset D (PM100) ─────────────────────────────────────────────────────────

def run_pm100(data_dir="data/external/pm100", download=False, max_rows=50_000):
    import importlib.util
    parquet = os.path.join(data_dir, "job_table.parquet")

    if not os.path.exists(parquet):
        if download:
            import requests
            url = "https://zenodo.org/api/records/10127767/files/job_table.parquet/content"
            os.makedirs(data_dir, exist_ok=True)
            print("  Downloading PM100 (~287 MB)...", flush=True)
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(parquet, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
            print()
        else:
            return None

    df = pd.read_parquet(parquet)
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)

    # Detect columns
    # PM100 actual schema: num_cores_alloc, mem_alloc, node_power_consumption (array)
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

    df["avg_power_w"]      = df["node_power_consumption"].apply(mean_power)
    df["cpu_request_mcpu"] = pd.to_numeric(df["num_cores_alloc"], errors="coerce") * 1000
    df["mem_mib"]          = pd.to_numeric(df["mem_alloc"], errors="coerce")
    df["walltime_s"]       = pd.to_numeric(df["run_time"], errors="coerce")

    feature_cols = ["cpu_request_mcpu", "mem_mib", "walltime_s"]
    df = df.dropna(subset=["cpu_request_mcpu", "avg_power_w"])
    df = df[(df["cpu_request_mcpu"] > 0) & (df["avg_power_w"] > 0)]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0).values)
    y = df["avg_power_w"].values

    knn_mae, knn_rmse, knn_r2, knn_mape = cv5(X, y, lambda: KNeighborsRegressor(n_neighbors=5))
    lr_mae,  lr_rmse,  lr_r2,  lr_mape  = cv5(X, y, LinearRegression)

    return {
        "dataset": "Zenodo:10127767\n(PM100 HPC)",
        "domain":  "HPC / IBM Power9",
        "n": len(df),
        "feature": f"{feature_cols}",
        "label": "Avg node power [W]",
        "knn_mae":  knn_mae,  "knn_rmse":  knn_rmse,
        "knn_r2":   knn_r2,   "knn_mape":  knn_mape,
        "lr_mae":   lr_mae,   "lr_r2":     lr_r2,
    }


# ── Existing EnergetiScope model ───────────────────────────────────────────────

def run_existing_model(feat_path="data/features.parquet",
                       label_path="data/kepler_labels.parquet"):
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        return None

    df_feat  = pd.read_parquet(feat_path)
    df_label = pd.read_parquet(label_path)
    df = df_feat.merge(df_label, on=["namespace", "workload_kind", "workload_name"])
    df = df[df["energy_step_j"].notna() & np.isfinite(df["energy_step_j"]) & (df["energy_step_j"] > 0)]

    if len(df) < 5:
        return None

    X = np.stack(df["features"].to_numpy())
    y = df["energy_step_j"].astype(float).values

    knn_mae, knn_rmse, knn_r2, knn_mape = cv5(X, y, lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine"))
    lr_mae,  lr_rmse,  lr_r2,  lr_mape  = cv5(X, y, LinearRegression)

    return {
        "dataset": "EnergetiScope\n(in-cluster Kepler)",
        "domain":  "K8s / Kepler labels",
        "n": len(df),
        "feature": "Encoded manifest vector (9-dim)",
        "label": "energy_step_j [J] (Kepler)",
        "knn_mae":  knn_mae,  "knn_rmse":  knn_rmse,
        "knn_r2":   knn_r2,   "knn_mape":  knn_mape,
        "lr_mae":   lr_mae,   "lr_r2":     lr_r2,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_table(rows):
    print("\n" + "="*90)
    print("COMBINED RESULTS TABLE")
    print("="*90)
    fmt_h = f"{'Dataset':<30} {'Domain':<18} {'n':>6}  {'KNN MAE':>10} {'KNN RMSE':>10} {'KNN R²':>8} {'KNN MAPE':>9} {'LR MAE':>10} {'LR R²':>8}"
    print(fmt_h)
    print("-"*90)
    for r in rows:
        ds   = r["dataset"].replace("\n", " ")
        dom  = r["domain"]
        n    = r["n"]
        print(f"{ds:<30} {dom:<18} {n:>6}  "
              f"{r['knn_mae']:>10.3f} {r['knn_rmse']:>10.3f} "
              f"{r['knn_r2']:>8.4f} {r['knn_mape']:>8.1f}% "
              f"{r['lr_mae']:>10.3f} {r['lr_r2']:>8.4f}")
    print("="*90)


def save_csv(rows, path="eval/results_summary.csv"):
    flat = []
    for r in rows:
        flat.append({
            "dataset":   r["dataset"].replace("\n", " "),
            "domain":    r["domain"],
            "n_samples": r["n"],
            "knn_k5_mae":  round(r["knn_mae"],  4),
            "knn_k5_rmse": round(r["knn_rmse"], 4),
            "knn_k5_r2":   round(r["knn_r2"],   4),
            "knn_k5_mape": round(r["knn_mape"],  2),
            "lr_mae":      round(r["lr_mae"],    4),
            "lr_r2":       round(r["lr_r2"],     4),
        })
    pd.DataFrame(flat).to_csv(path, index=False)
    print(f"\n  Results saved to {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pm100-download", action="store_true",
                   help="Download PM100 from Zenodo (~287 MB) if not present")
    p.add_argument("--skip-existing",  action="store_true",
                   help="Skip in-cluster model evaluation")
    args = p.parse_args()

    results = []

    print("\n[1/3] Running Dataset A (Zenodo:14332659)...")
    r = run_dataset_a()
    if r:
        results.append(r)
        print(f"  KNN k=5  MAE={r['knn_mae']:.3f} W  R²={r['knn_r2']:.4f}  MAPE={r['knn_mape']:.1f}%")
    else:
        print("  [SKIP] Data not found. Run: download zenodo_14332659 first.")

    print("\n[2/3] Running Dataset D (PM100 HPC)...")
    r = run_pm100(download=args.pm100_download)
    if r:
        results.append(r)
        print(f"  KNN k=5  MAE={r['knn_mae']:.3f} W  R²={r['knn_r2']:.4f}  MAPE={r['knn_mape']:.1f}%")
    else:
        if not args.pm100_download:
            print("  [SKIP] PM100 not found. Re-run with --pm100-download to fetch (~287 MB).")
        else:
            print("  [SKIP] PM100 download failed.")

    if not args.skip_existing:
        print("\n[3/3] Running in-cluster EnergetiScope model...")
        r = run_existing_model()
        if r:
            results.append(r)
            print(f"  KNN k=5  MAE={r['knn_mae']:.3f} J  R²={r['knn_r2']:.4f}  MAPE={r['knn_mape']:.1f}%")
        else:
            print("  [SKIP] Not enough in-cluster data (need >5 valid rows).")

    if results:
        print_table(results)
        save_csv(results)
    else:
        print("\n[WARN] No datasets produced results. Check data paths.")


if __name__ == "__main__":
    main()
