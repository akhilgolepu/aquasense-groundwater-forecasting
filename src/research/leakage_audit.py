"""
Research Pipeline — Leakage audit
Checks the forecasting setup for information leaking from the test period, from nearby wells or from
choices made after seeing test results. Every number in LEAKAGE_AUDIT.md comes from this module.

  inventory    split sizes, target/origin/input date ranges and the train → test gap per horizon
  qc           the ingestion outlier filter uses each well's whole-series median and MAD; refit M0/M2 with
               (a) the range filter only and (b) the MAD filter fitted on readings up to TRAIN_END_YEAR
  spatial      wells shared between folds, distance to the nearest training well, and M2 skill under
               time-only, region-only and time + region hold-out on the same test rows
  sanity       shuffled target, within-well time shuffle, leave-one-feature-group-out, train-vs-test gap
  cal          out-of-sample forecasts of every candidate model for the calibration years (2016–2018)
  selection    which equal-weight combination the calibration years would have chosen
  bootstrap    CIs of the combinations' MSE reduction against XGBoost M2, the best single model and the oracle
  facts        dataset sizes and years quoted in RESULTS.md
  value        economic value of warnings across assumed cost–loss ratios, not only at the peak

XGBoost refits reuse each arm's tuned depth, minimum child weight and tree count from
reports/skill/tuning.json, so differences come from the data, not from re-tuning.
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.research.config import (
    DATA_PROCESSED_DIR,
    HORIZONS,
    RANDOM_SEED,
    SKILL_REPORTS_DIR,
    TRAIN_END_YEAR,
)
from src.research.ingest import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    MAD_LIMIT,
    build_campaign_series,
    harmonise_depth,
    load_raw_observations,
)
from src.research.skill_decomposition import (
    ARMS,
    BASE_PARAMS,
    FEATURE_GROUPS,
    ci,
    load_dataset,
)
from src.research.skill_robustness import spatial_folds

AUDIT_DIR = SKILL_REPORTS_DIR.parent / "audit"
N_THREADS = int(os.environ.get("GW_THREADS", "8"))
SEEDS = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]
EVAL_COLS = ["seasonal_naive", "clim_target", "anomaly_persistence"]
MAX_LAG_CAMPAIGNS = 4  # y_lag4, d4, 12-month rain windows and SPI-12 all reach one year before the origin
ABLATION_GROUPS = ["Groundwater memory", "Season & climatology", "Location & borehole",
                   "Past rainfall & temperature", "ENSO/IOD"]


# ── Shared helpers ──────────────────────────────────────────────────────────

def tuned(h: int, arm: str) -> dict:
    with open(SKILL_REPORTS_DIR / "tuning.json") as f:
        t = json.load(f)[str(h)][arm]
    return {"max_depth": t["best"]["max_depth"], "min_child_weight": t["best"]["min_child_weight"],
            "n_estimators": t["n_trees_final"]}


def fit_fixed(train: pd.DataFrame, feats: list[str], h: int, arm: str, seed: int = RANDOM_SEED,
              target: np.ndarray | None = None) -> xgb.XGBRegressor:
    params = {**BASE_PARAMS, "random_state": seed, "n_jobs": N_THREADS, **tuned(h, arm)}
    m = xgb.XGBRegressor(**params)
    m.fit(train[feats], (train["y_target"] - train["y_t"]) if target is None else target)
    return m


def test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The Phase 2 common evaluation rows: test split with every baseline defined."""
    return df[df["split"] == "test"].dropna(subset=EVAL_COLS).copy()


def skill(frame: pd.DataFrame, col: str) -> float:
    mse = np.mean((frame[col] - frame["y_target"]) ** 2)
    return float(1 - mse / np.mean((frame["y_t"] - frame["y_target"]) ** 2))


def mse_reduction(frame: pd.DataFrame, a: str, b: str) -> float:
    return float(1 - np.mean((frame[b] - frame["y_target"]) ** 2) / np.mean((frame[a] - frame["y_target"]) ** 2))


def save(name: str, rows) -> pd.DataFrame:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT_DIR / f"{name}.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


# ── 1a/1b Inventory and gap ─────────────────────────────────────────────────

def run_inventory() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        df = load_dataset(h)
        for split in ("train", "cal", "test"):
            s = df[df["split"] == split]
            rows.append({"h": h, "split": split, "rows": len(s), "wells": s["well_id"].nunique(),
                         "first_target": f"{s['target_idx'].min() // 4}-{'JMAN'[s['target_idx'].min() % 4]}",
                         "last_target": f"{s['target_idx'].max() // 4}-{'JMAN'[s['target_idx'].max() % 4]}",
                         "first_origin_year": int(s["campaign_idx"].min() // 4),
                         "earliest_input_year": int((s["campaign_idx"].min() - MAX_LAG_CAMPAIGNS) // 4)})
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        gap = int(te["target_idx"].min() - tr["target_idx"].max())
        rows.append({"h": h, "split": "gap (campaigns)", "rows": gap,
                     "wells": h + MAX_LAG_CAMPAIGNS, "first_target": "required = h + max lag",
                     "last_target": "ok" if gap >= h + MAX_LAG_CAMPAIGNS else "VIOLATED"})
        # Do any test rows' inputs reach into the training targets? Allowed (it is the past), but reported.
        rows.append({"h": h, "split": "test inputs before 2016", "rows": int(((te["campaign_idx"] - MAX_LAG_CAMPAIGNS) // 4 <= TRAIN_END_YEAR).sum())})
    return save("inventory", rows)


# ── 1a QC filter: whole-series statistics ───────────────────────────────────

def qc_campaigns(variant: str) -> tuple[pd.DataFrame, dict]:
    """Rebuilds the manual-well campaign table with an alternative outlier filter."""
    raw = load_raw_observations()
    catalog = pd.read_parquet(DATA_PROCESSED_DIR / "station_catalog.parquet")
    manual = set(catalog.loc[catalog["network"] == "manual", "well_id"])
    obs = harmonise_depth(raw)
    obs = obs[obs["well_id"].isin(manual)]
    obs = obs[(obs["depth_mbgl"] >= DEPTH_MIN_M) & (obs["depth_mbgl"] <= DEPTH_MAX_M)]
    n_range = len(obs)
    if variant == "train-period MAD":
        ref = obs[obs["timestamp"].dt.year <= TRAIN_END_YEAR].groupby("well_id")["depth_mbgl"]
        stats = pd.DataFrame({"med": ref.median(), "mad": ref.apply(lambda s: (s - s.median()).abs().median()),
                              "n": ref.size()})
        stats = stats[stats["n"] >= 4]
        med = obs["well_id"].map(stats["med"])
        mad = obs["well_id"].map(stats["mad"]).clip(lower=0.25)
        z = (obs["depth_mbgl"] - med).abs() / (1.4826 * mad)
        obs = obs[~(z > MAD_LIMIT)]  # wells without ≥ 4 training-period readings keep the range filter only
    return build_campaign_series(obs), {"readings_after_range_filter": n_range, "readings_kept": len(obs)}


def mixed_sign_wells() -> dict:
    """harmonise_depth picks each station's sign from its whole series; how many manual wells have mixed signs?"""
    raw = load_raw_observations()
    catalog = pd.read_parquet(DATA_PROCESSED_DIR / "station_catalog.parquet")
    raw = raw[raw["well_id"].isin(catalog.loc[catalog["network"] == "manual", "well_id"])]
    share = raw.groupby("well_id")["water_level_raw"].apply(lambda s: (s < 0).mean())
    return {"manual_wells": len(share), "wells_with_any_negative": int((share > 0).sum()),
            "wells_majority_negative": int((share > 0.5).sum())}


def run_qc() -> pd.DataFrame:
    rows = []
    current = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet")
    variants = {"current (whole-series MAD)": (current, {})}
    for v in ("range only", "train-period MAD"):
        variants[v] = qc_campaigns(v)
    print("mixed-sign manual wells:", mixed_sign_wells())
    for name, (camp, info) in variants.items():
        for h in HORIZONS:
            df = load_dataset(h, campaigns=camp)
            train, test = df[df["split"] == "train"], test_rows(df)
            for arm in ("M0 memory", "M2 + ENSO/IOD"):
                test[arm] = test["y_t"] + fit_fixed(train, ARMS[arm], h, arm).predict(test[ARMS[arm]])
            rows.append({"qc": name, "h": h, "campaign_readings": len(camp), "train_rows": len(train),
                         "test_rows": len(test), "test_wells": test["well_id"].nunique(),
                         "skill_M0": skill(test, "M0 memory"), "skill_M2": skill(test, "M2 + ENSO/IOD"),
                         "mse_reduction_M2_vs_M0": mse_reduction(test, "M0 memory", "M2 + ENSO/IOD"), **info})
    return save("qc_sensitivity", rows)


# ── 1c Spatial hold-out ─────────────────────────────────────────────────────

def nearest_km(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    from sklearn.neighbors import BallTree

    tree = BallTree(np.deg2rad(b[["latitude", "longitude"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.deg2rad(a[["latitude", "longitude"]].to_numpy()), k=1)
    return d[:, 0] * 6371.0


def run_spatial() -> tuple[pd.DataFrame, pd.DataFrame]:
    arm = "M2 + ENSO/IOD"
    dist_rows, skill_rows, binned = [], [], []
    for h in HORIZONS:
        df = spatial_folds(load_dataset(h))
        evals = []
        for k in sorted(df["fold"].unique()):
            test = test_rows(df[df["fold"] == k])
            other, own = df[df["fold"] != k], df[df["fold"] == k]
            settings = {
                "time only (all wells, train ≤ 2015)": df[df["split"] == "train"],
                "region only (other blocks, all years)": other,
                "time + region (other blocks, train ≤ 2015)": other[other["split"] == "train"],
            }
            for name, train in settings.items():
                test[name] = test["y_t"] + fit_fixed(train, ARMS[arm], h, arm).predict(test[ARMS[arm]])
            strict = settings["time + region (other blocks, train ≤ 2015)"]
            wells_tr = strict.drop_duplicates("well_id")
            wells_te = test.drop_duplicates("well_id")
            shared = set(wells_tr["well_id"]) & set(wells_te["well_id"])
            d = nearest_km(wells_te, wells_tr)
            dist_rows.append({"h": h, "fold": int(k), "test_wells": len(wells_te), "train_wells": len(wells_tr),
                              "shared_wells": len(shared), "min_km": float(d.min()), "p10_km": float(np.percentile(d, 10)),
                              "median_km": float(np.median(d)), "max_km": float(d.max()),
                              "test_wells_with_own_training_history": int(wells_te["well_id"].isin(
                                  own.loc[own["split"] == "train", "well_id"]).sum())})
            test["nearest_train_km"] = test["well_id"].map(dict(zip(wells_te["well_id"], d)))
            evals.append(test)
        ev = pd.concat(evals)
        for name in settings:
            skill_rows.append({"h": h, "setting": name, "test_rows": len(ev), "test_wells": ev["well_id"].nunique(),
                               "skill_M2": skill(ev, name)})
        bins = pd.cut(ev["nearest_train_km"], [0, 10, 25, 50, np.inf], right=False)
        for b, g in ev.groupby(bins, observed=True):
            binned.append({"h": h, "nearest_training_well": str(b), "test_rows": len(g), "test_wells": g["well_id"].nunique(),
                           "skill_time_only": skill(g, "time only (all wells, train ≤ 2015)"),
                           "skill_time_region": skill(g, "time + region (other blocks, train ≤ 2015)")})
    save("spatial_distances", dist_rows)
    save("spatial_by_distance", binned)
    return save("spatial_settings", skill_rows), None


# ── 1d Sanity experiments ───────────────────────────────────────────────────

def time_shuffled_campaigns(seed: int) -> pd.DataFrame:
    """Permutes each well's readings across its own campaigns, destroying temporal order but not the values."""
    camp = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet").copy()
    rng = np.random.default_rng(seed)
    camp["depth_mbgl"] = camp.groupby("well_id")["depth_mbgl"].transform(lambda s: rng.permutation(s.to_numpy()))
    return camp


def run_sanity() -> pd.DataFrame:
    rows = []
    arm = "M2 + ENSO/IOD"
    for h in HORIZONS:
        df = load_dataset(h)
        train, test = df[df["split"] == "train"], test_rows(df)
        base = {"persistence": skill(test, "y_t"), "seasonal naive": skill(test, "seasonal_naive"),
                "climatology": skill(test, "clim_target"), "anomaly persistence": skill(test, "anomaly_persistence")}
        for name, v in base.items():
            rows.append({"h": h, "experiment": "baseline", "variant": name, "seed": None, "test_skill": v})
        for seed in SEEDS:
            m = fit_fixed(train, ARMS[arm], h, arm, seed)
            test["full"] = test["y_t"] + m.predict(test[ARMS[arm]])
            fit_pred = train["y_t"] + m.predict(train[ARMS[arm]])
            train_skill = float(1 - np.mean((fit_pred - train["y_target"]) ** 2) / np.mean((train["y_t"] - train["y_target"]) ** 2))
            rows.append({"h": h, "experiment": "full M2", "variant": arm, "seed": seed,
                         "train_skill": train_skill, "test_skill": skill(test, "full")})
            m0 = fit_fixed(train, ARMS["M0 memory"], h, "M0 memory", seed)
            p0 = train["y_t"] + m0.predict(train[ARMS["M0 memory"]])
            test["m0"] = test["y_t"] + m0.predict(test[ARMS["M0 memory"]])
            rows.append({"h": h, "experiment": "full M0", "variant": "M0 memory", "seed": seed,
                         "train_skill": float(1 - np.mean((p0 - train["y_target"]) ** 2) / np.mean((train["y_t"] - train["y_target"]) ** 2)),
                         "test_skill": skill(test, "m0")})

            rng = np.random.default_rng(seed)
            shuffled = rng.permutation((train["y_target"] - train["y_t"]).to_numpy())
            ms = fit_fixed(train, ARMS[arm], h, arm, seed, target=shuffled)
            test["shuf"] = test["y_t"] + ms.predict(test[ARMS[arm]])
            rows.append({"h": h, "experiment": "shuffled target", "variant": arm, "seed": seed, "test_skill": skill(test, "shuf")})

            for group in ABLATION_GROUPS:
                feats = [c for c in ARMS[arm] if c not in FEATURE_GROUPS[group]]
                ma = fit_fixed(train, feats, h, arm, seed)
                test["abl"] = test["y_t"] + ma.predict(test[feats])
                rows.append({"h": h, "experiment": "drop group", "variant": group, "seed": seed,
                             "test_skill": skill(test, "abl"), "change_vs_full": skill(test, "abl") - skill(test, "full")})

            ts = load_dataset(h, campaigns=time_shuffled_campaigns(seed))
            tr_s, te_s = ts[ts["split"] == "train"], ts[ts["split"] == "test"].dropna(subset=["anomaly_persistence"]).copy()
            mt = fit_fixed(tr_s, ARMS[arm], h, arm, seed)
            te_s["pred"] = te_s["y_t"] + mt.predict(te_s[ARMS[arm]])
            # With order destroyed, the best possible forecast is the well's mean, so compare with climatology
            te_s["clim"] = te_s["clim_target"]
            rows.append({"h": h, "experiment": "within-well time shuffle", "variant": arm, "seed": seed,
                         "test_skill": skill(te_s, "pred"), "climatology_skill_same_rows": skill(te_s, "clim")})
            print(f"[h={h}] seed {seed} done", flush=True)
    out = save("sanity", rows)
    summary = (out.groupby(["h", "experiment", "variant"], dropna=False)[["train_skill", "test_skill", "change_vs_full"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(AUDIT_DIR / "sanity_summary.csv")
    print(summary.to_string())
    return out


# ── Selection: which combination would the calibration years have picked? ───

CANDIDATES = ["M2 + ENSO/IOD", "RF M2", "LSTM + forcing", "Chronos-2 + past rain"]


def run_cal_predictions() -> pd.DataFrame:
    """
    Out-of-sample forecasts for the calibration years (2016–2018) from every candidate model, all trained on
    target years ≤ 2015: XGBoost M2 (Phase 2 run), RF M2 (refitted exactly as in reference_models), Chronos-2 +
    past rain (zero-shot) and LSTM + forcing (sequence_baseline, which now also predicts these rows).
    """
    from sklearn.ensemble import RandomForestRegressor

    from src.research.reference_models import chronos_predictions

    cached = DATA_PROCESSED_DIR / "audit_cal_base.parquet"  # RF + Chronos part, slow on CPU
    if cached.exists():
        return _add_lstm(pd.read_parquet(cached))
    base = pd.read_parquet(DATA_PROCESSED_DIR / "skill_predictions.parquet")
    cal = base[base["split"] == "cal"][["well_id", "campaign_idx", "h", "y_t", "y_target", "target_idx", "M2 + ENSO/IOD"]]
    parts = []
    for h in HORIZONS:
        df = load_dataset(h)
        train, ev = df[df["split"] == "train"], df[df["split"] == "cal"].copy()
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=N_THREADS,
                                   random_state=RANDOM_SEED)
        rf.fit(train[ARMS["M2 + ENSO/IOD"]], train["y_target"] - train["y_t"])
        ev["RF M2"] = ev["y_t"] + rf.predict(ev[ARMS["M2 + ENSO/IOD"]])
        parts.append(ev[["well_id", "campaign_idx", "RF M2"]].assign(h=h))
        print(f"[h={h}] RF M2 on {len(ev):,} calibration rows", flush=True)
    out = cal.merge(pd.concat(parts), on=["well_id", "campaign_idx", "h"], how="left")
    origins = out[["well_id", "campaign_idx"]].drop_duplicates().reset_index(drop=True)
    ch = chronos_predictions(origins, only=["Chronos-2 + past rain"])
    long = pd.concat([ch[["well_id", "campaign_idx", f"Chronos-2 + past rain|h{h}"]]
                      .rename(columns={f"Chronos-2 + past rain|h{h}": "Chronos-2 + past rain"}).assign(h=h) for h in HORIZONS])
    out = out.merge(long, on=["well_id", "campaign_idx", "h"], how="left")
    out.to_parquet(cached, index=False)
    return _add_lstm(out)


def _add_lstm(out: pd.DataFrame) -> pd.DataFrame:
    seq_path = DATA_PROCESSED_DIR / "sequence_cal_predictions.parquet"
    if seq_path.exists():
        out = out.merge(pd.read_parquet(seq_path)[["well_id", "campaign_idx", "h", "LSTM + forcing"]],
                        on=["well_id", "campaign_idx", "h"], how="left")
    out.to_parquet(DATA_PROCESSED_DIR / "audit_cal_predictions.parquet", index=False)
    print(f"Saved audit_cal_predictions.parquet ({len(out):,} rows)")
    return out


def test_predictions() -> pd.DataFrame:
    """Every model's test forecasts on the rows the extension results use (same rows as reports/extensions/)."""
    base = pd.read_parquet(DATA_PROCESSED_DIR / "skill_predictions.parquet")
    base = base[base["split"] == "test"].dropna(subset=["anomaly_persistence"])
    keys = ["well_id", "campaign_idx", "h"]
    out = base.merge(pd.read_parquet(DATA_PROCESSED_DIR / "reference_predictions.parquet"), on=keys, how="left")
    out = out.merge(pd.read_parquet(DATA_PROCESSED_DIR / "sequence_predictions.parquet"), on=keys, how="left")
    ext = pd.read_parquet(DATA_PROCESSED_DIR / "extension_predictions.parquet")
    return out.merge(ext.drop(columns="target_idx"), on=keys, how="inner")


def run_selection() -> pd.DataFrame:
    """Scores every equal-weight mean of 1–4 candidate models on the calibration years (2016–2018) and the test years."""
    cal = pd.read_parquet(DATA_PROCESSED_DIR / "audit_cal_predictions.parquet")
    ext = test_predictions()
    rows = []
    for h in HORIZONS:
        parts = {"calibration 2016–2018": cal[cal["h"] == h].dropna(subset=CANDIDATES),
                 "test 2019–2025": ext[ext["h"] == h].dropna(subset=CANDIDATES)}
        for period, d in parts.items():
            for r in range(1, len(CANDIDATES) + 1):
                for combo in combinations(CANDIDATES, r):
                    pred = d[list(combo)].mean(axis=1)
                    mse = np.mean((pred - d["y_target"]) ** 2)
                    rows.append({"h": h, "period": period, "combination": " | ".join(combo), "n_models": r,
                                 "rows": len(d), "skill": float(1 - mse / np.mean((d["y_t"] - d["y_target"]) ** 2))})
    out = pd.DataFrame(rows)
    out["rank_in_period"] = out.groupby(["h", "period"])["skill"].rank(ascending=False).astype(int)
    out = out.sort_values(["h", "period", "rank_in_period"])
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT_DIR / "combination_selection.csv", index=False)
    print(out[out["rank_in_period"] <= 3].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


# ── Bootstrap of the combination gains ──────────────────────────────────────

SINGLES = ["M0 memory", "M2 + ENSO/IOD", "RF M2", "LSTM memory", "LSTM + forcing", "Chronos-2 + past rain"]


def _boot_ci(d: pd.DataFrame, model: str, ref: str, by: str) -> list[float]:
    """95 % CI of the MSE reduction 1 − MSE(model)/MSE(ref), resampling wells or whole target campaigns."""
    groups, inv = np.unique(d[by].to_numpy(), return_inverse=True)
    se_m = np.bincount(inv, weights=(d[model] - d["y_target"]).to_numpy() ** 2)
    se_r = np.bincount(inv, weights=(d[ref] - d["y_target"]).to_numpy() ** 2)
    rng = np.random.default_rng(RANDOM_SEED)
    w = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)) for _ in range(1000)]).astype(float)
    return ci(1 - (w @ se_m) / (w @ se_r))


def run_bootstrap() -> pd.DataFrame:
    """
    Test-period MSE reduction, with well- and campaign-resampled 95 % CIs, of (a) the combination the calibration
    years select (combination_selection.csv) and (b) the post-hoc mean of M2, LSTM and Chronos-2, against XGBoost M2,
    the best single model on the test rows (the hardest comparison) and the oracle-rainfall model M3.
    """
    ext = test_predictions()
    sel = pd.read_csv(AUDIT_DIR / "combination_selection.csv")
    rows = []
    for h in HORIZONS:
        d = ext[ext["h"] == h].dropna(subset=CANDIDATES + SINGLES + ["M3 + oracle rain"]).copy()
        chosen = sel[(sel["h"] == h) & (sel["period"] == "calibration 2016–2018") & (sel["rank_in_period"] == 1)]["combination"].iloc[0]
        d["calibration-selected"] = d[chosen.split(" | ")].mean(axis=1)
        d["post hoc M2 + LSTM + Chronos-2"] = d[["M2 + ENSO/IOD", "LSTM + forcing", "Chronos-2 + past rain"]].mean(axis=1)
        best_single = max(SINGLES, key=lambda c: skill(d, c))
        for model, label in (("calibration-selected", chosen), ("post hoc M2 + LSTM + Chronos-2", "M2 + LSTM + Chronos-2")):
            for ref in ("M2 + ENSO/IOD", best_single, "M3 + oracle rain"):
                rows.append({"h": h, "months": 3 * h, "model": model, "combination": label, "reference": ref,
                             "best_single_model": best_single, "test_rows": len(d), "test_wells": d["well_id"].nunique(),
                             "campaigns": d["target_idx"].nunique(), "skill_model": skill(d, model),
                             "skill_reference": skill(d, ref), "mse_reduction": mse_reduction(d, ref, model),
                             "ci95_wells": _boot_ci(d, model, ref, "well_id"),
                             "ci95_campaigns": _boot_ci(d, model, ref, "target_idx")})
    out = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT_DIR / "combination_bootstrap.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(out.drop(columns=["test_wells", "campaigns"]).to_string(index=False))
    return out


# ── Dataset facts quoted in RESULTS.md ──────────────────────────────────────

def run_facts() -> pd.DataFrame:
    raw = load_raw_observations()
    catalog = pd.read_parquet(DATA_PROCESSED_DIR / "station_catalog.parquet")
    obs = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_observations_clean.parquet")
    camp = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet")
    rows = [("raw readings", len(raw)), ("raw stations", raw["well_id"].nunique()),
            ("raw first reading", f"{raw['timestamp'].min():%Y-%m-%d}"), ("raw last reading", f"{raw['timestamp'].max():%Y-%m-%d}"),
            ("readings after quality control", len(obs)), ("stations after quality control", obs["well_id"].nunique()),
            ("manual (campaign) wells", int((catalog["network"] == "manual").sum())),
            ("telemetry (DWLR) wells, not used for forecasting", int((catalog["network"] == "DWLR").sum())),
            ("manual-well campaign values", len(camp)), ("manual wells with campaign values", camp["well_id"].nunique()),
            ("first campaign year", int(camp["year"].min())), ("last campaign year", int(camp["year"].max()))]
    for h in HORIZONS:
        df = load_dataset(h)
        te = test_rows(df)
        rows += [(f"h={h}: wells with at least one forecast row", df["well_id"].nunique()),
                 (f"h={h}: test rows (common evaluation set)", len(te)), (f"h={h}: test wells", te["well_id"].nunique()),
                 (f"h={h}: test target campaigns", te["target_idx"].nunique())]
    return save("facts", [{"fact": k, "value": v} for k, v in rows])


# ── Economic value: sensitivity to the assumed cost–loss ratio ──────────────

VALUE_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def run_value() -> pd.DataFrame:
    """
    Relative economic value needs a user's cost–loss ratio α = C/L, which the data do not contain. Summarises
    the decision-value curves (reports/decision/value_curves.csv; M2 forecasts, rolling-recalibrated
    probabilities, rule "warn when p ≥ α") across α instead of quoting only the peak.
    """
    v = pd.read_csv(SKILL_REPORTS_DIR.parent / "decision" / "value_curves.csv")
    v = v[(v["arm"] == "M2 + ENSO/IOD") & (v["probability"] == "rolling isotonic")]
    rows = []
    for (event, h), g in v.groupby(["event_id", "h"]):
        g = g.sort_values("alpha")
        peak = g.loc[g["value"].idxmax()]
        pos, sig = g.loc[g["value"] > 0, "alpha"], g.loc[g["value_lo"] > 0, "alpha"]
        row = {"event": event, "h": h, "months": 3 * h, "peak_value": peak["value"], "alpha_at_peak": peak["alpha"],
               "alpha_range_value_positive": f"{pos.min():.2f}–{pos.max():.2f}" if len(pos) else "none",
               "alpha_range_ci_above_zero": f"{sig.min():.2f}–{sig.max():.2f}" if len(sig) else "none"}
        for a in VALUE_ALPHAS:
            r = g.iloc[(g["alpha"] - a).abs().argmin()]
            row[f"V@{a}"] = f"{r['value']:.2f} [{r['value_lo']:.2f}, {r['value_hi']:.2f}]"
        rows.append(row)
    return save("value_sensitivity", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["inventory", "qc", "spatial", "sanity", "cal", "selection", "bootstrap", "value", "facts"])
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for step in args.steps:
            print(f"=== Leakage audit: {step} ===", flush=True)
            {"inventory": run_inventory, "qc": run_qc, "spatial": run_spatial, "sanity": run_sanity,
             "cal": run_cal_predictions, "selection": run_selection, "bootstrap": run_bootstrap, "value": run_value, "facts": run_facts}[step]()
