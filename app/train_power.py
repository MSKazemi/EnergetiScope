#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


def make_model(name, neighbors):
    if name == "gbt":
        return GradientBoostingRegressor(random_state=0)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    if name == "linear":
        return LinearRegression()
    return KNeighborsRegressor(n_neighbors=neighbors, metric="cosine")


def train_model(df, target, neighbors=5, model="gbt", verbose=True):
    """Cross-validate and fit a regressor on encoded workload specs.

    Rows are repeated per-interval measurements of the same workload, so
    folds are split by ``workload_name`` (GroupKFold): a random row split
    would place the same workload on both sides and report memorization
    as generalization.

    Metrics are computed on POOLED out-of-fold predictions rather than
    averaged per fold. A fold whose validation group has near-zero label
    variance makes per-fold R2 diverge, and the average of those is
    uninterpretable even when the predictions are good.

    Returns the model refit on all rows.
    """
    X = np.stack(df["features"].to_numpy())
    y = df[target].astype(float).to_numpy()
    keep = np.isfinite(y)
    X, y, df = X[keep], y[keep], df.loc[keep]
    if len(y) == 0:
        raise SystemExit(f"No finite values for target '{target}'. "
                         f"Try --target energy_step_j or fix labels.")

    groups = df["workload_name"].astype(str).to_numpy()
    n_groups = max(1, df["workload_name"].nunique())
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise SystemExit("Need at least 2 distinct workloads for CV. Collect more data.")

    oof, maes = np.zeros_like(y, dtype=float), []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model(model, neighbors)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        maes.append(mean_absolute_error(y[va], oof[va]))

    if verbose:
        print(f"[{model}] CV MAE: {np.mean(maes):.1f} ± {np.std(maes):.1f} "
              f"| pooled OOF R2: {r2_score(y, oof):.3f}")

    final = make_model(model, neighbors)
    final.fit(X, y)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)  # ./data/train_rows.parquet
    p.add_argument("--target", choices=["avg_power_w","energy_step_j","total_energy_j"], default="avg_power_w")
    p.add_argument("--out", required=True)   # model path
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--model", choices=["gbt","knn","rf","linear"], default="gbt",
                   help="regressor family; gbt is the most accurate on the "
                        "benchmark (see eval/eval_bench_grouped.py)")
    args = p.parse_args()

    model = train_model(pd.read_parquet(args.train), args.target,
                        args.neighbors, args.model)
    joblib.dump(model, args.out)
    print(f"[OK] saved model to {args.out}")

if __name__ == "__main__":
    main()
