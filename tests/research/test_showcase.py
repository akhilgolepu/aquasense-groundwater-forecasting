"""Tests for src/research/model_showcase.py: pairwise statistics and scores on synthetic forecasts."""

import numpy as np
import pandas as pd
import pytest

from src.research.model_showcase import campaign_mae, model_scores, pairwise


def _frame(seed=0, n_wells=40, n_camp=12):
    rng = np.random.default_rng(seed)
    rows = [(w, t) for w in range(n_wells) for t in range(n_camp)]
    d = pd.DataFrame(rows, columns=["well_id", "target_idx"])
    d["well_id"] = "w" + d["well_id"].astype(str)
    d["y_target"] = rng.normal(10, 2, len(d))
    d["persistence"] = d["y_target"] + rng.normal(0, 1.0, len(d))
    d["M0 memory"] = d["y_target"] + rng.normal(0, 0.5, len(d))  # better than persistence
    d["M2 + ENSO/IOD"] = d["y_target"] + rng.normal(0, 0.05, len(d))  # nearly perfect
    return d


def test_scores_rank_models_and_bracket_the_estimate():
    d = _frame()
    rows, _ = model_scores(d, ["persistence", "m0", "m2"])
    s = {r["model"]: r for r in rows}
    assert s["persistence"]["skill"] == pytest.approx(0.0)
    assert s["m2"]["skill"] > 0.99
    assert 0 < s["m0"]["skill"] < 1
    assert s["m0"]["skill_lo"] <= s["m0"]["skill"] <= s["m0"]["skill_hi"]
    assert s["m0"]["rows"] == len(d) and s["m0"]["wells"] == 40 and s["m0"]["campaigns"] == 12


def test_pairwise_counts_campaign_wins_both_ways():
    d = _frame()
    keys = ["persistence", "m0", "m2"]
    _, draws = model_scores(d, keys)
    p = {(r["a"], r["b"]): r for r in pairwise(d, keys, draws)}
    assert len(p) == 6
    assert p[("m0", "m2")]["mse_reduction"] > 0.98
    assert p[("m0", "m2")]["b_wins"] == 12 and p[("m2", "m0")]["b_wins"] == 0
    assert p[("persistence", "m0")]["lo"] > 0  # m0 clearly better
    assert p[("m0", "persistence")]["hi"] < 0
    assert p[("m0", "m2")]["sign_p"] < 0.001


def test_campaign_mae_has_one_row_per_campaign():
    d = _frame()
    c = campaign_mae(d, ["persistence", "m2"])
    assert len(c) == 12 and all(r["m2"] < r["persistence"] for r in c)
