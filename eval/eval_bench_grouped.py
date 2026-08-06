#!/usr/bin/env python3
"""Canonical evaluation of the EnergetiScope benchmark.

Supersedes the bench row of the old ``eval/eval_all.py``, whose ``cv5()``
helper used ``KFold(shuffle=True)`` with **no grouping** and therefore let
rows of the same workload fall on both sides of a split.

Protocol here (matches what the paper claims):
  * GroupKFold, grouping key = ``workload_name``
  * metrics computed on POOLED out-of-fold predictions, so every row is
    predicted exactly once by a model that never saw its workload
  * per-fold MAE reported as mean +/- sd for uncertainty

Run:  python eval/eval_bench_grouped.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

FEATURES = "data/bench/features.parquet"
LABELS = "data/bench/labels.parquet"
TARGET = "energy_step_j"
MIN_ROWS_PER_GROUP = 5  # drops one-off CronJob instances with a single sample


def mape(y_true, y_pred, eps=1e-6):
    m = np.abs(y_true) > eps
    return 100.0 * np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m]))


def load():
    f = pd.read_parquet(FEATURES)
    l = pd.read_parquet(LABELS)
    df = f.merge(l, on=["namespace", "workload_kind", "workload_name"], how="inner")
    df = df[df[TARGET].notna() & np.isfinite(df[TARGET]) & (df[TARGET] > 0)]
    df = df.groupby("workload_name").filter(lambda x: len(x) >= MIN_ROWS_PER_GROUP)
    return df


MODELS = {
    "KNN (k=5, cosine)": lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine"),
    "Linear Regression": LinearRegression,
    "Gradient Boosted Trees": lambda: GradientBoostingRegressor(random_state=0),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
}


def main():
    df = load()
    X = np.stack(df["features"].to_numpy())
    y = df[TARGET].astype(float).to_numpy()
    groups = df["workload_name"].astype(str).to_numpy()
    n_groups = df["workload_name"].nunique()

    print("=" * 78)
    print("EnergetiScope benchmark - GroupKFold, pooled out-of-fold predictions")
    print("=" * 78)
    print(f"rows={len(y)}  workload groups={n_groups}  "
          f"target={TARGET}  mean={y.mean():.1f} J  sd={y.std():.1f} J")
    print(f"feature dim={X.shape[1]}  non-constant columns="
          f"{int((X.std(axis=0) > 0).sum())}")

    gkf = GroupKFold(n_splits=min(5, n_groups))
    rows = []
    for name, factory in MODELS.items():
        oof = np.zeros_like(y)
        fold_mae = []
        for tr, va in gkf.split(X, y, groups):
            m = factory().fit(X[tr], y[tr])
            p = m.predict(X[va])
            oof[va] = p
            fold_mae.append(mean_absolute_error(y[va], p))

        # workload-level: average OOF prediction and truth per workload
        agg = pd.DataFrame({"g": groups, "y": y, "p": oof}).groupby("g").mean()

        rows.append(dict(
            model=name,
            mae=mean_absolute_error(y, oof),
            mae_sd=float(np.std(fold_mae)),
            rmse=float(np.sqrt(mean_squared_error(y, oof))),
            r2=r2_score(y, oof),
            mape=mape(y, oof),
            wl_mae=mean_absolute_error(agg["y"], agg["p"]),
            wl_r2=r2_score(agg["y"], agg["p"]),
            wl_mape=mape(agg["y"].to_numpy(), agg["p"].to_numpy()),
        ))

    print(f"\n{'model':<24} {'MAE (J)':>16} {'RMSE':>8} {'R2':>7} {'MAPE':>7}"
          f" | {'wl-MAE':>8} {'wl-R2':>7} {'wl-MAPE':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{r['model']:<24} {r['mae']:>8.0f} +/- {r['mae_sd']:<4.0f} "
              f"{r['rmse']:>8.0f} {r['r2']:>7.3f} {r['mape']:>6.1f}% | "
              f"{r['wl_mae']:>8.0f} {r['wl_r2']:>7.3f} {r['wl_mape']:>7.1f}%")

    # mean-predictor reference: what R2=0 corresponds to
    print(f"\nmean-predictor MAE = "
          f"{mean_absolute_error(y, np.full_like(y, y.mean())):.0f} J (R2 = 0 by definition)")

    out = pd.DataFrame(rows)
    out.to_csv("eval/results_bench_grouped.csv", index=False)
    print("\n[OK] wrote eval/results_bench_grouped.csv")


if __name__ == "__main__":
    main()
