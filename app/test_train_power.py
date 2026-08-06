
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from app.train_power import make_model, train_model


def _dummy_frame(n_workloads=10, rows_per_workload=3, dim=10, seed=0):
    """Several rows per workload, so GroupKFold has something to group on."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_workloads):
        base = rng.random(dim)
        for _ in range(rows_per_workload):
            rows.append({
                "features": base + rng.normal(0, 0.01, dim),
                "total_energy_j": float(base.sum() + rng.normal(0, 0.01)),
                "workload_name": f"workload{w}",
            })
    return pd.DataFrame(rows)


class TestMakeModel(unittest.TestCase):

    def test_families(self):
        self.assertIsInstance(make_model("gbt", 5), GradientBoostingRegressor)
        self.assertIsInstance(make_model("knn", 5), KNeighborsRegressor)

    def test_unknown_name_falls_back_to_knn(self):
        self.assertIsInstance(make_model("nonsense", 5), KNeighborsRegressor)

    def test_neighbors_is_honoured(self):
        self.assertEqual(make_model("knn", 7).n_neighbors, 7)


class TestTrainModel(unittest.TestCase):

    def test_default_is_gradient_boosted(self):
        model = train_model(_dummy_frame(), "total_energy_j", verbose=False)
        self.assertIsInstance(model, GradientBoostingRegressor)

    def test_knn_can_be_selected(self):
        model = train_model(_dummy_frame(), "total_energy_j", neighbors=5,
                            model="knn", verbose=False)
        self.assertIsInstance(model, KNeighborsRegressor)

    def test_returns_fitted_model(self):
        df = _dummy_frame()
        model = train_model(df, "total_energy_j", verbose=False)
        preds = model.predict(np.stack(df["features"].to_numpy()))
        self.assertEqual(len(preds), len(df))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_rejects_single_workload(self):
        """GroupKFold needs >= 2 groups; one workload must not silently pass."""
        with self.assertRaises(SystemExit):
            train_model(_dummy_frame(n_workloads=1), "total_energy_j", verbose=False)

    def test_rejects_all_nan_target(self):
        df = _dummy_frame()
        df["total_energy_j"] = np.nan
        with self.assertRaises(SystemExit):
            train_model(df, "total_energy_j", verbose=False)


if __name__ == '__main__':
    unittest.main()
