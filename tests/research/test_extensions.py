"""Tests for Phase 7: CFSv2 forecast features (causality, coverage) and neighbour features."""

import numpy as np
import pandas as pd
import pytest

from src.research.forcing_features import N_MONTHS, campaign_month_index
from src.research.model_extensions import NB_RADIUS_KM, neighbour_features
from src.research.seasonal_forecast import (
    MAX_LEAD,
    forecast_climatology,
    forecast_features,
)

N_CELLS = 2


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    F = rng.gamma(2.0, 50.0, size=(N_MONTHS, MAX_LEAD + 1, N_CELLS)).astype(np.float32)
    imd_norm = np.full((1, 12), 100.0)
    cell_of = pd.Series([1.0], index=["w1"])
    imd_row_of = pd.Series([0.0], index=["w1"])
    return F, imd_norm, cell_of, imd_row_of


KEYS = pd.DataFrame({"well_id": ["w1"], "campaign_idx": [2020 * 4 + 1]})  # May 2020


def test_forecast_uses_only_the_issue_before_the_origin_month():
    F, imd_norm, cell_of, imd_row_of = _inputs()
    clim = forecast_climatology(F)
    base = forecast_features(KEYS, cell_of, F, clim, imd_norm, imd_row_of)
    t0 = int(campaign_month_index(KEYS["campaign_idx"].to_numpy())[0])
    later = F.copy()
    later[t0:] *= 3.0  # forecasts issued in or after the campaign month
    pd.testing.assert_frame_equal(base, forecast_features(KEYS, cell_of, later, clim, imd_norm, imd_row_of))
    earlier = F.copy()
    earlier[t0 - 1] *= 3.0  # the forecast actually used
    assert not np.allclose(base["fc_rain_ratio_h1"], forecast_features(KEYS, cell_of, earlier, clim, imd_norm, imd_row_of)["fc_rain_ratio_h1"])


def test_forecast_ratio_and_coverage():
    F, imd_norm, cell_of, imd_row_of = _inputs()
    out = forecast_features(KEYS, cell_of, F, F[:12], imd_norm, imd_row_of)  # pass any [12, lead, cell] climatology
    # May → August window = 3 months, fully reached; 12-month window reaches 9 months
    assert out["fc_coverage_h1"].iloc[0] == pytest.approx(1.0)
    assert out["fc_coverage_h4"].iloc[0] == pytest.approx(MAX_LEAD / 12)
    # A forecast equal to its own climatology is a ratio of 1 and the IMD normal in mm
    t0 = int(campaign_month_index(KEYS["campaign_idx"].to_numpy())[0])
    clim = np.repeat(F[t0 - 1][None], 12, axis=0)
    same = forecast_features(KEYS, cell_of, F, clim, imd_norm, imd_row_of)
    assert same["fc_rain_ratio_h2"].iloc[0] == pytest.approx(1.0)
    n_win = campaign_month_index(KEYS["campaign_idx"].to_numpy() + 2)[0] - t0
    assert same["fc_rain_h2"].iloc[0] == pytest.approx(100.0 * n_win)


def test_unmapped_well_gets_missing_forecast():
    F, imd_norm, _, imd_row_of = _inputs()
    out = forecast_features(KEYS, pd.Series([np.nan], index=["w1"]), F, forecast_climatology(F), imd_norm, imd_row_of)
    assert out["fc_rain_ratio_h1"].isna().all()


def test_neighbour_features_exclude_self_and_far_wells():
    df = pd.DataFrame({
        "campaign_idx": [100] * 4,
        "latitude": [20.0, 20.1, 20.2, 25.0],   # the fourth well is ~550 km away
        "longitude": [78.0, 78.0, 78.0, 78.0],
        "anom_t": [1.0, 2.0, 3.0, 50.0],
        "d1": [0.0, 0.0, 0.0, 0.0],
        "d4": [0.0, 0.0, 0.0, 0.0],
    })
    nb = neighbour_features(df)
    assert nb.loc[0, "nb_anom_mean"] == pytest.approx(2.5)  # wells 1 and 2, not itself, not the far one
    assert nb.loc[0, "nb_n"] == 2
    assert nb.loc[3, "nb_n"] == 0 and np.isnan(nb.loc[3, "nb_anom_mean"])
    assert (nb["nb_dist_km"].dropna() <= NB_RADIUS_KM).all()
