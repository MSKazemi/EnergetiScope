#!/usr/bin/env python3
"""
Model improvement experiments for EnergetiScope paper:
1. KNN k-sensitivity (k=1,3,5,7,10,15,20)
2. Gradient Boosted Trees vs KNN
3. KNN prediction intervals (uncertainty estimation)

Runs on both SBERT and numeric-only conditions using standard 5-fold CV
and GroupKFold CV.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import json, csv, os

DATA_DIR = "data/bench"
OUT_CSV = "eval/model_experiments_results.csv"


def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def load_data(condition):
    """Load train_rows parquet for a given condition (unfiltered, matching ablation)."""
    suffix = "sbert" if condition == "sbert" else "no_sbert"
    path = os.path.join(DATA_DIR, f"train_rows_{suffix}.parquet")
    df = pd.read_parquet(path)
    X = np.stack(df["features"].to_numpy())
    y = df["energy_step_j"].astype(float).to_numpy()
    groups = df["workload_name"].astype(str).to_numpy()
    keep = np.isfinite(y) & (y > 0)
    return X[keep], y[keep], groups[keep]


def cv_evaluate(model_cls, model_kwargs, X, y, groups, cv_method="kfold"):
    """Run CV and return dict of metrics."""
    if cv_method == "kfold":
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X, y)
    else:
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(5, n_groups))
        splits = cv.split(X, y, groups)

    maes, r2s, mapes_list = [], [], []
    for tr, va in splits:
        m = model_cls(**model_kwargs)
        m.fit(X[tr], y[tr])
        pva = m.predict(X[va])
        maes.append(mean_absolute_error(y[va], pva))
        r2s.append(r2_score(y[va], pva))
        mapes_list.append(mape(y[va], pva))

    return {
        "mae": np.mean(maes),
        "mae_std": np.std(maes),
        "r2": np.mean(r2s),
        "r2_std": np.std(r2s),
        "mape": np.mean(mapes_list),
    }


def knn_prediction_intervals(X_train, y_train, X_test, k=5):
    """Compute prediction intervals from KNN neighbor distances and variance."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)

    predictions = []
    lower = []
    upper = []
    neighbor_stds = []

    for i in range(len(X_test)):
        neighbor_vals = y_train[indices[i]]
        pred = np.mean(neighbor_vals)
        std = np.std(neighbor_vals)
        predictions.append(pred)
        lower.append(pred - 1.96 * std)
        upper.append(pred + 1.96 * std)
        neighbor_stds.append(std)

    return {
        "predictions": np.array(predictions),
        "lower": np.array(lower),
        "upper": np.array(upper),
        "std": np.array(neighbor_stds),
    }


def main():
    results = []

    for condition in ["sbert", "no_sbert"]:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")
        X, y, groups = load_data(condition)
        n_groups = len(np.unique(groups))
        print(f"  n={len(y)}, groups={n_groups}, y range=[{y.min():.1f}, {y.max():.1f}]")

        # --- Experiment 1: KNN k-sensitivity ---
        print(f"\n  [Exp 1] KNN k-sensitivity (5-fold CV):")
        for k in [1, 3, 5, 7, 10, 15, 20]:
            res = cv_evaluate(
                KNeighborsRegressor,
                {"n_neighbors": k, "metric": "cosine"},
                X, y, groups, "kfold"
            )
            print(f"    k={k:2d}  MAE={res['mae']:.1f} ±{res['mae_std']:.1f}  "
                  f"R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
            results.append({
                "condition": condition, "model": f"KNN_k{k}",
                "cv_method": "KFold", **res, "n": len(y), "n_groups": n_groups
            })

        # Also GroupKFold for best k values
        print(f"\n  [Exp 1b] KNN k-sensitivity (GroupKFold):")
        for k in [3, 5, 10]:
            res = cv_evaluate(
                KNeighborsRegressor,
                {"n_neighbors": k, "metric": "cosine"},
                X, y, groups, "groupkfold"
            )
            print(f"    k={k:2d}  MAE={res['mae']:.1f} ±{res['mae_std']:.1f}  "
                  f"R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
            results.append({
                "condition": condition, "model": f"KNN_k{k}",
                "cv_method": "GroupKFold", **res, "n": len(y), "n_groups": n_groups
            })

        # --- Experiment 2: Gradient Boosted Trees ---
        print(f"\n  [Exp 2] Gradient Boosted Trees (5-fold CV):")
        for n_est, lr, depth in [(100, 0.1, 5), (200, 0.1, 5), (200, 0.05, 7)]:
            label = f"GBT_{n_est}t_lr{lr}_d{depth}"
            res = cv_evaluate(
                GradientBoostingRegressor,
                {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth,
                 "random_state": 42},
                X, y, groups, "kfold"
            )
            print(f"    {label:30s}  MAE={res['mae']:.1f}  "
                  f"R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
            results.append({
                "condition": condition, "model": label,
                "cv_method": "KFold", **res, "n": len(y), "n_groups": n_groups
            })

        # GBT GroupKFold
        print(f"\n  [Exp 2b] GBT best config (GroupKFold):")
        for n_est, lr, depth in [(200, 0.1, 5)]:
            label = f"GBT_{n_est}t_lr{lr}_d{depth}"
            res = cv_evaluate(
                GradientBoostingRegressor,
                {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth,
                 "random_state": 42},
                X, y, groups, "groupkfold"
            )
            print(f"    {label:30s}  MAE={res['mae']:.1f}  "
                  f"R²={res['r2']:.4f}  MAPE={res['mape']:.1f}%")
            results.append({
                "condition": condition, "model": label,
                "cv_method": "GroupKFold", **res, "n": len(y), "n_groups": n_groups
            })

        # --- Experiment 3: Uncertainty estimation ---
        print(f"\n  [Exp 3] KNN prediction intervals (k=5):")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_coverages = []
        all_widths = []
        for tr, va in kf.split(X, y):
            pi = knn_prediction_intervals(X[tr], y[tr], X[va], k=5)
            in_interval = (y[va] >= pi["lower"]) & (y[va] <= pi["upper"])
            coverage = in_interval.mean()
            avg_width = np.mean(pi["upper"] - pi["lower"])
            all_coverages.append(coverage)
            all_widths.append(avg_width)

        mean_coverage = np.mean(all_coverages)
        mean_width = np.mean(all_widths)
        print(f"    95% PI coverage: {mean_coverage:.1%} (target: 95%)")
        print(f"    Mean interval width: {mean_width:.1f} J")
        print(f"    Mean prediction std: {mean_width/3.92:.1f} J")
        results.append({
            "condition": condition, "model": "KNN_k5_uncertainty",
            "cv_method": "KFold", "mae": 0, "r2": 0, "mape": 0,
            "mae_std": 0, "r2_std": 0,
            "n": len(y), "n_groups": n_groups,
            "pi_coverage": mean_coverage, "pi_width": mean_width
        })

    # Save all results
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
