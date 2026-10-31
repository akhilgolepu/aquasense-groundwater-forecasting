"""
Leakage guards (LEAKAGE_AUDIT.md): fail if the split loses its embargo, a scaler sees the test period,
a sequence input looks past its origin, or a spatial fold shares wells with its training folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import sequence_baseline
from src.research.benchmark import (
    add_expanding_climatology,
    assign_split,
    build_supervised,
    complete_grid,
)
from src.research.config import HORIZONS, TRAIN_END_YEAR
from src.research.leakage_audit import MAX_LAG_CAMPAIGNS
from src.research.sequence_baseline import rain_scaler, sequence_tensor, static_matrix
from src.research.skill_robustness import N_FOLDS, spatial_folds

ROOT = Path(__file__).resolve().parents[2]
REAL = (ROOT / "data" / "processed" / "gwl_campaigns.parquet").exists()


def _campaigns(n_wells=3, first_year=1994, last_year=2025, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(first_year * 4, last_year * 4 + 4)
    rows = [pd.DataFrame({"well_id": f"w{w}", "campaign_idx": idx, "depth_mbgl": 10 + rng.normal(0, 1, len(idx)).cumsum() * 0.1,
                          "year": idx // 4, "season": idx % 4}) for w in range(n_wells)]
    return pd.concat(rows, ignore_index=True)


def _catalog(n_wells=3):
    return pd.DataFrame({"well_id": [f"w{w}" for w in range(n_wells)], "latitude": 20.0 + np.arange(n_wells),
                         "longitude": 78.0 + np.arange(n_wells), "well_depth_m": 50.0})


def _check_gap(df: pd.DataFrame, h: int):
    split = df["split"] if "split" in df else assign_split(df)
    train, cal, test = (df.loc[split == s, "target_idx"] for s in ("train", "cal", "test"))
    assert train.max() < cal.min() <= cal.max() < test.min()
    assert test.min() - train.max() >= h + MAX_LAG_CAMPAIGNS


@pytest.mark.parametrize("h", HORIZONS)
def test_train_and_test_targets_are_separated_by_an_embargo(h):
    grid = add_expanding_climatology(complete_grid(_campaigns()))
    df = build_supervised(grid, _catalog(), h).dropna(subset=["y_target"])
    _check_gap(df, h)


@pytest.mark.skipif(not REAL, reason="processed data not built")
@pytest.mark.parametrize("h", HORIZONS)
def test_real_split_has_the_embargo(h):
    from src.research.skill_decomposition import load_dataset

    _check_gap(load_dataset(h), h)


def test_static_scaler_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    train = np.arange(200) < 120
    base = static_matrix(df, ["a", "b"], train)
    changed = df.copy()
    changed.loc[~train, ["a", "b"]] = 1e6  # wild values in the evaluation rows only
    after = static_matrix(changed, ["a", "b"], train)
    np.testing.assert_array_equal(base[train], after[train])


def test_rain_scaler_ignores_years_after_training():
    grid = pd.DataFrame({"campaign_idx": np.arange(2010 * 4, 2024 * 4), "rain_since_prev": np.linspace(0, 500, 56)})
    base = rain_scaler(grid)
    changed = grid.copy()
    changed.loc[changed["campaign_idx"] // 4 > TRAIN_END_YEAR, "rain_since_prev"] = 1e5
    assert rain_scaler(changed) == base


def test_sequence_inputs_ignore_readings_after_the_origin(tmp_path, monkeypatch):
    camp = _campaigns(n_wells=1, first_year=2012, last_year=2023)
    origin = 2020 * 4 + 1
    rain = camp[["well_id", "campaign_idx"]].assign(rain_since_prev=np.linspace(50, 400, len(camp)))

    def tensor(c, r):
        c.to_parquet(tmp_path / "gwl_campaigns.parquet")
        r.to_parquet(tmp_path / "forcing_features.parquet")
        monkeypatch.setattr(sequence_baseline, "DATA_PROCESSED_DIR", tmp_path)
        rows = build_supervised(add_expanding_climatology(complete_grid(c)), _catalog(1), 1)
        rows = rows[rows["campaign_idx"] == origin].reset_index(drop=True)
        return sequence_tensor(rows, with_rain=True)

    base = tensor(camp, rain)
    later_c, later_r = camp.copy(), rain.copy()
    later_c.loc[later_c["campaign_idx"] > origin, "depth_mbgl"] = 999.0
    later_r.loc[later_r["campaign_idx"] > origin, "rain_since_prev"] = 9999.0
    np.testing.assert_array_equal(base, tensor(later_c, later_r))
    assert np.isfinite(base).all() and base[0, -1, 0] == 0.0  # newest step is the origin itself (relative depth 0)


def test_spatial_folds_never_share_wells():
    rng = np.random.default_rng(2)
    n = 400
    wells = pd.DataFrame({"well_id": [f"w{i}" for i in range(n)], "latitude": rng.uniform(8, 35, n),
                          "longitude": rng.uniform(68, 97, n)})
    rows = wells.loc[rng.integers(0, n, 5000)].reset_index(drop=True)
    folds = spatial_folds(rows)
    assert folds.groupby("well_id")["fold"].nunique().max() == 1
    assert set(folds["fold"]) == set(range(N_FOLDS))
    for k in range(N_FOLDS):
        assert not set(folds.loc[folds["fold"] == k, "well_id"]) & set(folds.loc[folds["fold"] != k, "well_id"])


@pytest.mark.skipif(not REAL, reason="processed data not built")
def test_real_spatial_folds_never_share_wells():
    from src.research.skill_decomposition import load_dataset

    folds = spatial_folds(load_dataset(1))
    assert folds.groupby("well_id")["fold"].nunique().max() == 1
