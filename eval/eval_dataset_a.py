#!/usr/bin/env python3
"""
Validation against Dataset A: Container Energy Observability Replication Package
(Pijnacker et al., arXiv:2504.10702 — Zenodo:14332659)

Two sub-experiments from this dataset:
  1. BootstrapInitializer: cpuSeconds per 15s window → powerWatts (iDRAC ground truth)
     → Validates that KNN regression can learn the CPU-utilization → power curve.
  2. single-stressor: stress-ng --cpu 32 pod; Kepler namespace power vs. iDRAC node power
     → Validates namespace-level energy attribution accuracy.

Usage:
    python eval/eval_dataset_a.py --data data/external/zenodo_14332659
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ── helpers ──────────────────────────────────────────────────────────────────

def mape(y_true, y_pred, eps=1e-6):
    """Mean Absolute Percentage Error (skips near-zero true values)."""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def cv_metrics(X, y, model_factory, n_splits=5, random_state=42):
    """Run n-fold CV and return dict of mean ± std metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes, rmses, r2s, mapes = [], [], [], []
    for tr, va in kf.split(X):
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
        print(f"    {k:6s}: {mean:8.3f} ± {std:.3f}")


# ── Experiment 1: BootstrapInitializer ────────────────────────────────────────

def load_bootstrap(data_dir):
    """
    Load all BootstrapInitializer containerUtilization CSVs.
    Each row: cpuSeconds (float), powerWatts (float, iDRAC ground truth).
    Returns a combined DataFrame.
    """
    ts_ids = ["1729087088", "1729099594", "1729113884"]
    frames = []
    for ts in ts_ids:
        path = os.path.join(data_dir, "data", f"_{ts}-BootstrapInitializer-containerUtilization.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing: {path}")
            continue
        df = pd.read_csv(path)
        df["experiment_id"] = ts
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No BootstrapInitializer files found in {data_dir}/data/")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"cpuSeconds": "cpu_seconds", "powerWatts": "power_w"})
    # Drop rows with missing values
    combined = combined.dropna(subset=["cpu_seconds", "power_w"])
    return combined


def experiment_bootstrap(data_dir):
    """
    Validate KNN (k=5) vs. Linear Regression vs. Mean Baseline on:
      feature = cpu_seconds (CPU utilisation over 15s window)
      label   = power_w    (iDRAC node power in Watts)

    This validates the core mechanism: resource utilisation → power is learnable
    by the KNN approach used in EnergetiScope.
    """
    print("\n" + "="*65)
    print("Experiment 1 — BootstrapInitializer (iDRAC ground truth)")
    print("="*65)

    df = load_bootstrap(data_dir)
    print(f"  Loaded {len(df)} samples across {df['experiment_id'].nunique()} runs")
    print(f"  cpu_seconds range: [{df['cpu_seconds'].min():.2f}, {df['cpu_seconds'].max():.2f}]")
    print(f"  power_w     range: [{df['power_w'].min():.1f}, {df['power_w'].max():.1f}] W")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[["cpu_seconds"]].values)
    y = df["power_w"].values

    n_splits = 5
    print(f"\n  5-fold cross-validation (n={len(df)})\n")

    # KNN k=5 (same as EnergetiScope default)
    knn5 = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=5, metric="euclidean"), n_splits)
    print_metrics("KNN k=5  (EnergetiScope default)", knn5)

    # KNN k=3
    knn3 = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=3, metric="euclidean"), n_splits)
    print_metrics("KNN k=3", knn3)

    # KNN k=10
    knn10 = cv_metrics(X, y, lambda: KNeighborsRegressor(n_neighbors=10, metric="euclidean"), n_splits)
    print_metrics("KNN k=10", knn10)

    # Linear regression baseline
    lr = cv_metrics(X, y, LinearRegression, n_splits)
    print_metrics("Linear Regression (baseline)", lr)

    # Mean baseline
    mean_mae = np.mean([mean_absolute_error([np.mean(y)] * len(y), y)])
    print(f"\n  [Mean Baseline]")
    print(f"    MAE   : {mean_mae:8.3f} (predict mean for all)")

    # Extended feature set: cpuSeconds + cpuSeconds^2 (nonlinearity)
    X2 = np.column_stack([X, X**2])
    knn5_ext = cv_metrics(X2, y, lambda: KNeighborsRegressor(n_neighbors=5, metric="euclidean"), n_splits)
    print_metrics("KNN k=5 + quadratic feature", knn5_ext)

    return {
        "dataset": "Zenodo:14332659 BootstrapInitializer",
        "n_samples": len(df),
        "feature": "cpu_seconds (iDRAC gt)",
        "KNN_k5": knn5,
        "KNN_k3": knn3,
        "LinearRegression": lr,
    }


# ── Experiment 2: single-stressor ─────────────────────────────────────────────

def experiment_single_stressor(data_dir):
    """
    stress-ng --cpu 32 pod in namespace 'stress'.
    Compare:
      - Kepler namespace-level attribution ('stress' column) vs.
      - iDRAC node power (ground truth)

    This tests Kepler's measurement quality — if Kepler over/underestimates,
    EnergetiScope's labels are biased accordingly.
    """
    print("\n" + "="*65)
    print("Experiment 2 — single-stressor: Kepler vs. iDRAC")
    print("="*65)

    pow_path  = os.path.join(data_dir, "data", "single-stressor--namespace-power.csv")
    node_path = os.path.join(data_dir, "data", "single-stressor--node-power.csv")

    if not os.path.exists(pow_path) or not os.path.exists(node_path):
        print("  [SKIP] Required files not found.")
        return None

    df_ns   = pd.read_csv(pow_path)
    df_node = pd.read_csv(node_path)

    results = []
    # 3 runs: no suffix (.0), .1, .2
    suffixes   = ["", ".1", ".2"]
    run_labels = ["Run 1", "Run 2", "Run 3"]

    for suf, lbl in zip(suffixes, run_labels):
        ns_col    = "stress" + suf
        kepler_col = "kepler" + suf
        idrac_col  = "idrac" + suf

        if ns_col not in df_ns.columns:
            continue

        # Align by timestamp (Unnamed: 0)
        ts_ns   = df_ns["Unnamed: 0"].values
        ts_node = df_node["Unnamed: 0"].values
        common_ts = np.intersect1d(ts_ns, ts_node)

        ns_pow  = df_ns.loc[df_ns["Unnamed: 0"].isin(common_ts), ns_col].values
        kep_tot = df_node.loc[df_node["Unnamed: 0"].isin(common_ts), kepler_col].values
        idr     = df_node.loc[df_node["Unnamed: 0"].isin(common_ts), idrac_col].values

        # Only look at the active stress phase (ts >= 300s = 5 min ramp)
        ts_active = common_ts >= 300
        if ts_active.sum() < 3:
            ts_active = np.ones(len(common_ts), dtype=bool)

        ns_active  = ns_pow[ts_active]
        kep_active = kep_tot[ts_active]
        idr_active = idr[ts_active]

        valid = np.isfinite(ns_active) & np.isfinite(idr_active) & (idr_active > 0)
        if valid.sum() < 3:
            print(f"  {lbl}: insufficient valid data")
            continue

        # Kepler node total vs. iDRAC
        kep_err  = mean_absolute_error(idr_active[valid], kep_active[valid])
        kep_mape_val = mape(idr_active[valid], kep_active[valid])

        print(f"\n  {lbl}:")
        print(f"    Stress namespace power (Kepler): mean={ns_active[valid].mean():.1f} W")
        print(f"    Node power iDRAC (ground truth): mean={idr_active[valid].mean():.1f} W")
        print(f"    Kepler node-total MAE vs. iDRAC: {kep_err:.2f} W")
        print(f"    Kepler node-total MAPE vs. iDRAC: {kep_mape_val:.1f}%")

        # Fraction of node power attributed to stress workload
        stress_frac = ns_active[valid].mean() / idr_active[valid].mean() * 100
        print(f"    Stress workload share of node power: {stress_frac:.1f}%")

        results.append({
            "run": lbl,
            "kepler_mae_w": kep_err,
            "kepler_mape_pct": kep_mape_val,
            "stress_share_pct": stress_frac,
        })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/external/zenodo_14332659",
                   help="Path to extracted Zenodo:14332659 directory")
    args = p.parse_args()

    if not os.path.isdir(args.data):
        sys.exit(f"[ERROR] data directory not found: {args.data}\n"
                 "Run from project root or pass --data path.")

    r1 = experiment_bootstrap(args.data)
    r2 = experiment_single_stressor(args.data)

    print("\n" + "="*65)
    print("Summary: Dataset A (Zenodo:14332659)")
    print("="*65)
    if r1:
        knn5 = r1["KNN_k5"]
        print(f"  BootstrapInitializer KNN k=5:")
        print(f"    MAE  = {knn5['MAE'][0]:.2f} W   RMSE = {knn5['RMSE'][0]:.2f} W")
        print(f"    R²   = {knn5['R2'][0]:.4f}      MAPE = {knn5['MAPE'][0]:.1f}%")
    if r2:
        avg_kep_mae = np.mean([x["kepler_mae_w"] for x in r2])
        print(f"  Kepler attribution mean MAE vs. iDRAC: {avg_kep_mae:.2f} W")

    print()


if __name__ == "__main__":
    main()
