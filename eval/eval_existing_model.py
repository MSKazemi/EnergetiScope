#!/usr/bin/env python3
"""
Evaluate the existing EnergetiScope KNN model on in-cluster training data.
Produces cross-validation metrics and a prediction vs. ground-truth report.

Usage:
    python eval/eval_existing_model.py
    python eval/eval_existing_model.py --features data/features.parquet \
                                        --labels data/kepler_labels.parquet \
                                        --encoder data/encoder.joblib
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
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def print_metrics(label, mae, rmse, r2, mape_val, n):
    print(f"  [{label}]  n={n}")
    print(f"    MAE  = {mae:.4f}   RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}   MAPE = {mape_val:.2f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--labels",   default="data/kepler_labels.parquet")
    p.add_argument("--encoder",  default="data/encoder.joblib")
    p.add_argument("--target",   default="energy_step_j",
                   choices=["energy_step_j", "avg_power_w", "total_energy_j"])
    args = p.parse_args()

    # Load
    df_feat  = pd.read_parquet(args.features)
    df_label = pd.read_parquet(args.labels)

    # Merge on workload identity
    df = df_feat.merge(
        df_label,
        on=["namespace", "workload_kind", "workload_name"],
        how="inner"
    )

    # Filter to valid labels
    df = df[df[args.target].notna() & np.isfinite(df[args.target]) & (df[args.target] > 0)]

    print("\n" + "="*65)
    print("Existing EnergetiScope Model — In-cluster Validation")
    print("="*65)
    print(f"  Target: {args.target}")
    print(f"  Total valid rows: {len(df)}")
    print(f"  Unique workloads: {df['workload_name'].nunique()}")
    print(f"  Label range: [{df[args.target].min():.2f}, {df[args.target].max():.2f}]")

    if len(df) < 5:
        print("\n  [WARN] Too few samples for meaningful evaluation.")
        print("  Run the full data collection pipeline to get more training data.")
        return

    X = np.stack(df["features"].to_numpy())
    y = df[args.target].astype(float).values
    groups = df["workload_name"].astype(str).values
    n_groups = df["workload_name"].nunique()

    # --- Leave-One-Workload-Out CV (if enough groups) ---
    if n_groups >= 3:
        logo = LeaveOneGroupOut()
        maes, rmses, r2s, mapes = [], [], [], []
        for tr, va in logo.split(X, y, groups):
            m = KNeighborsRegressor(n_neighbors=min(5, len(tr)), metric="cosine")
            m.fit(X[tr], y[tr])
            pva = m.predict(X[va])
            maes.append(mean_absolute_error(y[va], pva))
            rmses.append(np.sqrt(mean_squared_error(y[va], pva)))
            r2s.append(r2_score(y[va], pva))
            mapes.append(mape(y[va], pva))

        print(f"\n  Leave-One-Workload-Out CV ({n_groups} groups):")
        print_metrics(
            "KNN k=5 cosine (EnergetiScope)",
            np.mean(maes), np.mean(rmses), np.mean(r2s), np.mean(mapes), len(df)
        )

    # --- K-fold CV ---
    n_splits = min(5, n_groups)
    if n_splits >= 2:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        maes, rmses, r2s, mapes = [], [], [], []
        for tr, va in gkf.split(X, y, groups):
            m = KNeighborsRegressor(n_neighbors=5, metric="cosine")
            m.fit(X[tr], y[tr])
            pva = m.predict(X[va])
            maes.append(mean_absolute_error(y[va], pva))
            rmses.append(np.sqrt(mean_squared_error(y[va], pva)))
            r2s.append(r2_score(y[va], pva))
            mapes.append(mape(y[va], pva))

        print(f"\n  {n_splits}-Fold Group CV:")
        print_metrics(
            "KNN k=5 cosine (EnergetiScope)",
            np.mean(maes), np.mean(rmses), np.mean(r2s), np.mean(mapes), len(df)
        )

        # Comparison baselines
        lr_maes, lr_r2s = [], []
        for tr, va in gkf.split(X, y, groups):
            lr = LinearRegression()
            lr.fit(X[tr], y[tr])
            pva = lr.predict(X[va])
            lr_maes.append(mean_absolute_error(y[va], pva))
            lr_r2s.append(r2_score(y[va], pva))
        print_metrics("Linear Regression (baseline)",
                      np.mean(lr_maes), 0.0, np.mean(lr_r2s), 0.0, len(df))

        mean_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
        print(f"  [Mean Baseline]  MAE = {mean_mae:.4f}")

    # --- Ablation: k sensitivity ---
    print(f"\n  k-sensitivity (5-fold, all data):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in [1, 3, 5, 10, 15]:
        maes = []
        for tr, va in kf.split(X, y):
            m = KNeighborsRegressor(n_neighbors=min(k, len(tr)), metric="cosine")
            m.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[va], m.predict(X[va])))
        print(f"    k={k:2d}  MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # --- Per-workload breakdown ---
    print(f"\n  Per-workload median prediction (train-on-all):")
    m_full = KNeighborsRegressor(n_neighbors=5, metric="cosine")
    m_full.fit(X, y)
    preds = m_full.predict(X)
    for wl in df["workload_name"].unique():
        mask = df["workload_name"] == wl
        y_w = y[mask]
        p_w = preds[mask]
        print(f"    {wl[:40]:40s}  n={mask.sum():3d}  "
              f"true_med={np.median(y_w):.2f}  pred_med={np.median(p_w):.2f}  "
              f"MAE={mean_absolute_error(y_w, p_w):.2f}")

    print()


if __name__ == "__main__":
    main()
