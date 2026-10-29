"""
Research Pipeline — Phase 7.1: Seasonal rainfall forecasts (G4)
How much of the oracle-rainfall headroom can a real, operational seasonal forecast capture?

Source: NCEP CFSv2 monthly precipitation from the North American Multi-Model Ensemble (NMME), via the
IRI Data Library (open access, no key): hindcasts 1982–2010 and real-time forecasts from 2011,
ensemble mean, 1° grid, leads 0.5–9.5 months.

Causality: a campaign reading can be taken early in its month, so the forecast used at an origin is
the one **issued in the month before the campaign month** (as for every past window). Lead 1.5 of that
forecast is the campaign month itself; the longest lead (9.5) reaches 9 months past the origin, which
covers the 3- and 6-month windows fully and 9 of the 12 months of the 12-month window.

Features (per well, origin campaign and horizon h, mirroring the oracle arm):
  fc_rain_ratio_h{h}  forecast rain over the covered months / the model's own lead-dependent climatology
                      (fitted on issue years ≤ SPI_FIT_END_YEAR) — a bias-corrected forecast anomaly
  fc_rain_h{h}        that ratio × the well's IMD normal over the whole origin → target window (mm)
  fc_coverage_h{h}    share of the window's months the forecast reaches
"""

import warnings

import numpy as np
import pandas as pd

from src.research.config import (
    DATA_PROCESSED_DIR,
    EXTERNAL_DIR,
    FORCING_FIRST_YEAR,
    HORIZONS,
    SPI_FIT_END_YEAR,
)
from src.research.forcing_features import (
    N_MONTHS,
    campaign_month_index,
    nearest_valid_cells,
)

CFS_DIR = EXTERNAL_DIR / "cfsv2"
IRI_BASE = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/"
SOURCES = {"hindcast": ".HINDCAST/.MONTHLY/.prec", "forecast": ".FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY/.prec"}
SUBSET = "/X/66/99/RANGEEDGES/Y/5/38/RANGEEDGES/[M]average/data.nc"  # India box, ensemble mean
MAX_LEAD = 9  # lead 9.5 → 9 months after the issue month
S_OFFSET = (FORCING_FIRST_YEAR - 1960) * 12  # IRI "months since 1960-01-01" → months since Jan FORCING_FIRST_YEAR


def download(force: bool = False):
    import requests

    CFS_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        out = CFS_DIR / f"{name}.nc"
        if out.exists() and not force:
            continue
        print(f"Downloading CFSv2 {name} from the IRI Data Library ...")
        r = requests.get(IRI_BASE + path + SUBSET, timeout=3600)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"  {out} ({len(r.content) / 1e6:.1f} MB)")


def load_cfs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (F[issue month, lead 0..MAX_LEAD, lat, lon] in mm/month, lat, lon). Issue months are indexed
    like the forcing table (months since Jan FORCING_FIRST_YEAR); real-time forecasts override hindcasts.
    """
    import xarray as xr

    F, lat, lon = None, None, None
    for name in ("hindcast", "forecast"):
        ds = xr.open_dataset(CFS_DIR / f"{name}.nc", decode_times=False)
        da = ds["prec"].transpose("S", "L", "Y", "X")
        if F is None:
            lat, lon = da["Y"].values, da["X"].values
            F = np.full((N_MONTHS, MAX_LEAD + 1, len(lat), len(lon)), np.nan, dtype=np.float32)
        for i, s in enumerate(np.round(da["S"].values).astype(int) - S_OFFSET):
            if not 0 <= s < N_MONTHS:
                continue
            for j, L in enumerate(da["L"].values):
                lead = int(L)  # L = 0.5, 1.5, … → 0, 1, …
                if lead <= MAX_LEAD:
                    target = s + lead
                    year, month = FORCING_FIRST_YEAR + target // 12, target % 12 + 1
                    days = pd.Period(f"{year}-{month:02d}").days_in_month
                    F[s, lead] = da.values[i, j] * days
    return F, lat, lon


def forecast_climatology(F: np.ndarray) -> np.ndarray:
    """Mean forecast per (issue calendar month, lead, cell) over issue years ≤ SPI_FIT_END_YEAR."""
    issue_year = FORCING_FIRST_YEAR + np.arange(F.shape[0]) // 12
    fit = issue_year <= SPI_FIT_END_YEAR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(F[fit & (np.arange(F.shape[0]) % 12 == c)], axis=0) for c in range(12)])


def forecast_features(keys: pd.DataFrame, cell_of: pd.Series, F: np.ndarray, F_clim: np.ndarray,
                      imd_norm: np.ndarray, imd_row_of: pd.Series) -> pd.DataFrame:
    """
    keys: (well_id, campaign_idx). F/F_clim: [issue, lead, cell] and [calendar month, lead, cell] with cells
    flattened; cell_of maps well → CFS cell. imd_norm: [IMD row, calendar month] rainfall normal (mm);
    imd_row_of maps well → IMD row. Uses only F[issue = origin month − 1]. Pure; unit-tested for leakage.
    """
    out = keys[["well_id", "campaign_idx"]].copy().reset_index(drop=True)
    cell = cell_of.reindex(out["well_id"]).to_numpy()
    irow = imd_row_of.reindex(out["well_id"]).to_numpy()
    ok_well = ~np.isnan(cell.astype(float)) & ~np.isnan(irow.astype(float))
    cell = np.where(ok_well, cell, 0).astype(np.int64)
    irow = np.where(ok_well, irow, 0).astype(np.int64)
    t0 = campaign_month_index(out["campaign_idx"].to_numpy())
    issue = t0 - 1
    ok_issue = (issue >= 0) & (issue < F.shape[0])
    issue_c = np.clip(issue, 0, F.shape[0] - 1)
    for h in HORIZONS:
        t1 = campaign_month_index(out["campaign_idx"].to_numpy() + h)
        n_win = t1 - t0
        fc, clim, norm_all = np.zeros(len(out)), np.zeros(len(out)), np.zeros(len(out))
        n_cov = np.zeros(len(out), dtype=int)
        for k in range(int(n_win.max())):
            in_win = k < n_win
            norm_all += np.where(in_win, imd_norm[irow, (t0 + k) % 12], 0.0)
            lead = k + 1  # month t0 + k is lead k + 1 of the forecast issued in t0 − 1
            if lead > MAX_LEAD:
                continue
            use = in_win
            fc += np.where(use, F[issue_c, lead, cell], 0.0)
            clim += np.where(use, F_clim[issue_c % 12, lead, cell], 0.0)
            n_cov += use
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fc / clim
        ratio[~(ok_well & ok_issue)] = np.nan
        out[f"fc_rain_ratio_h{h}"] = ratio
        out[f"fc_rain_h{h}"] = ratio * norm_all
        out[f"fc_coverage_h{h}"] = n_cov / n_win
    return out


def imd_normals() -> tuple[np.ndarray, pd.Series]:
    """Monthly IMD rainfall normals (≤ SPI_FIT_END_YEAR) for every cell used by a well, and the well → row map."""
    monthly = pd.read_parquet(DATA_PROCESSED_DIR / "forcing_monthly.parquet")
    cells = pd.read_parquet(DATA_PROCESSED_DIR / "well_rain_cells.parquet")
    monthly = monthly[monthly["month"].dt.year <= SPI_FIT_END_YEAR]
    norm = monthly.groupby(["cell", monthly["month"].dt.month])["rain_mm"].mean().unstack().sort_index()
    row_of_cell = pd.Series(np.arange(len(norm)), index=norm.index)
    imd_row_of = cells.set_index("well_id")["cell"].map(row_of_cell)
    return norm.to_numpy(), imd_row_of


def run_seasonal_forecast():
    print("=== Phase 7.1: CFSv2 seasonal forecast features ===")
    download()
    F, lat, lon = load_cfs()
    have = ~np.isnan(F[:, 1]).all(axis=(1, 2))
    first, last = np.flatnonzero(have)[[0, -1]]
    print(f"  CFSv2 issue months {FORCING_FIRST_YEAR + first // 12}-{first % 12 + 1:02d} … "
          f"{FORCING_FIRST_YEAR + last // 12}-{last % 12 + 1:02d}; {int((~have[first:last + 1]).sum())} missing in between")
    F_clim = forecast_climatology(F)
    n_cells = len(lat) * len(lon)
    F_flat, clim_flat = F.reshape(F.shape[0], F.shape[1], n_cells), F_clim.reshape(12, F.shape[1], n_cells)

    catalog = pd.read_parquet(DATA_PROCESSED_DIR / "station_catalog.parquet")
    valid = ~np.isnan(np.nanmean(F[:, 1], axis=0))
    cells = nearest_valid_cells(catalog["latitude"].to_numpy(), catalog["longitude"].to_numpy(), lat, lon, valid)
    cell_of = pd.Series(np.where(cells >= 0, cells, np.nan), index=catalog["well_id"])
    imd_norm, imd_row_of = imd_normals()

    campaigns = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet")
    span = campaigns.groupby("well_id")["campaign_idx"].agg(["min", "max"])
    keys = pd.DataFrame({
        "well_id": np.repeat(span.index.values, (span["max"] - span["min"] + 1).values),
        "campaign_idx": np.concatenate([np.arange(a, b + 1) for a, b in zip(span["min"], span["max"])]),
    })
    feats = forecast_features(keys, cell_of, F_flat, clim_flat, imd_norm, imd_row_of)
    feats.to_parquet(DATA_PROCESSED_DIR / "seasonal_forecast_features.parquet", index=False)
    print(f"Saved seasonal_forecast_features.parquet: {len(feats):,} rows; "
          f"coverage {', '.join(f'h={h} {feats[f'fc_rain_ratio_h{h}'].notna().mean():.0%}' for h in HORIZONS)}")


if __name__ == "__main__":
    run_seasonal_forecast()
