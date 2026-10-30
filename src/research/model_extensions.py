"""
Research Pipeline — Phase 7: Model extensions (G4–G8)
Tests the changes a reviewer is most likely to ask for, on the Phase 2 test rows and with the Phase 2
tuning protocol and statistics:

  G4  M2 + CFSv2 forecast  M2 + the operational seasonal rainfall forecast issued before the origin
                           (src.research.seasonal_forecast); how much of the M2 → M3 oracle gap does it close?
  G5  Global LSTM          sequence deep-learning baseline (src.research.sequence_baseline), if run
  G6  Datum jumps          how much of the remaining error is one-step jumps > 20 m (likely data errors)
  G7  M2 + neighbours      mean anomaly and recent change of the K nearest wells read in the same campaign
  G8  Combination          equal-weight mean of XGBoost M2 and Chronos-2 + past rain (+ RF M2); no weights fitted

G4 and G8 are repeated in the two Phase 2.7 robustness settings (train ≤ 2017; unseen 2° spatial blocks).
Outputs go to reports/extensions/; `extension_predictions.parquet` holds the new arms on the test rows.
"""

import json
import warnings

import numpy as np
import pandas as pd

from src.research.config import DATA_PROCESSED_DIR, HORIZONS, RANDOM_SEED
from src.research.config import EXTENSIONS_REPORTS_DIR as EXT_REPORTS_DIR
from src.research.skill_decomposition import (
    ARMS,
    JUMP_SENSITIVITY_M,
    campaign_sign_test,
    ci,
    cluster_bootstrap,
    load_dataset,
    mae_bootstrap,
    panel_dm,
    tune_and_fit,
)
from src.research.skill_robustness import BLOCK_DEG, N_FOLDS

FORECAST = ["fc_rain", "fc_rain_ratio"]
NEIGHBOURS = ["nb_anom_mean", "nb_d1_mean", "nb_d4_mean", "nb_n", "nb_dist_km"]
NB_K, NB_RADIUS_KM = 8, 75.0
CFS_ARM = "M2 + CFSv2 forecast"
NB_ARM = "M2 + neighbours"
ENS = "Mean of M2 and Chronos-2"
ENS3 = "Mean of M2, Chronos-2 and RF"
ENS_ALL = "Mean of M2, LSTM and Chronos-2"
CHRONOS = "Chronos-2 + past rain"
LSTM = ["LSTM memory", "LSTM + forcing"]


# ── Features ────────────────────────────────────────────────────────────────

def neighbour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (well, origin) row: summary of the K nearest *other* wells within NB_RADIUS_KM that were read
    in the same origin campaign. Uses only origin-time readings, like the well's own memory features.
    """
    from sklearn.neighbors import BallTree

    parts = []
    for _, g in df.groupby("campaign_idx"):
        g = g.dropna(subset=["anom_t"])
        if len(g) < 2:
            continue
        tree = BallTree(np.deg2rad(g[["latitude", "longitude"]].to_numpy()), metric="haversine")
        dist, idx = tree.query(np.deg2rad(g[["latitude", "longitude"]].to_numpy()), k=min(NB_K + 1, len(g)))
        dist, idx = dist[:, 1:] * 6371.0, idx[:, 1:]  # drop the well itself
        near = dist <= NB_RADIUS_KM
        cols = {}
        for name, src in (("nb_anom_mean", "anom_t"), ("nb_d1_mean", "d1"), ("nb_d4_mean", "d4")):
            v = np.where(near, g[src].to_numpy()[idx], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cols[name] = np.nanmean(v, axis=1)
        cols["nb_n"] = near.sum(axis=1)
        with np.errstate(invalid="ignore"):
            cols["nb_dist_km"] = np.where(near, dist, 0).sum(axis=1) / np.where(cols["nb_n"] > 0, cols["nb_n"], np.nan)
        parts.append(pd.DataFrame(cols, index=g.index))
    return pd.concat(parts)


def extended_dataset(h: int) -> pd.DataFrame:
    df = load_dataset(h)
    fc = pd.read_parquet(DATA_PROCESSED_DIR / "seasonal_forecast_features.parquet")
    fc = fc[["well_id", "campaign_idx", f"fc_rain_h{h}", f"fc_rain_ratio_h{h}"]].rename(
        columns={f"fc_rain_h{h}": "fc_rain", f"fc_rain_ratio_h{h}": "fc_rain_ratio"})
    df = df.merge(fc, on=["well_id", "campaign_idx"], how="left")
    return df.join(neighbour_features(df))


def add_combinations(df: pd.DataFrame) -> pd.DataFrame:
    df[ENS] = (df["M2 + ENSO/IOD"] + df[CHRONOS]) / 2
    df[ENS3] = (df["M2 + ENSO/IOD"] + df[CHRONOS] + df["RF M2"]) / 3
    if "LSTM + forcing" in df:
        df[ENS_ALL] = (df["M2 + ENSO/IOD"] + df["LSTM + forcing"] + df[CHRONOS]) / 3
    return df


# ── Statistics ──────────────────────────────────────────────────────────────

def compare(test: pd.DataFrame, a: str, b: str, h: int, setting: str = "Main test 2019–2025") -> dict:
    mse = lambda c: float(np.mean((test[c] - test["y_target"]) ** 2))
    mae = lambda c: float(np.mean(np.abs(test[c] - test["y_target"])))
    mse_p = mse("persistence")
    stable = test[(test["y_target"] - test["y_t"]).abs() <= JUMP_SENSITIVITY_M]
    se = lambda c: np.mean((stable[c] - stable["y_target"]) ** 2)
    return {"setting": setting, "h": h, "a": a, "b": b, "n": len(test),
            "skill_a": 1 - mse(a) / mse_p, "skill_b": 1 - mse(b) / mse_p,
            "mse_reduction": 1 - mse(b) / mse(a), "mae_reduction": 1 - mae(b) / mae(a),
            "mae_reduction_ci": mae_bootstrap(test, a, b), "dm_mae_p": panel_dm(test, a, b, "absolute")["p_value"],
            **campaign_sign_test(test, a, b), "mse_reduction_without_jumps": float(1 - se(b) / se(a))}


def jump_attribution(test: pd.DataFrame, cols: list[str], h: int) -> list[dict]:
    jump = (test["y_target"] - test["y_t"]).abs() > JUMP_SENSITIVITY_M
    out = []
    for c in cols:
        sq = (test[c] - test["y_target"]) ** 2
        sq_p = (test["persistence"] - test["y_target"]) ** 2
        out.append({"h": h, "model": c, "jump_rows": int(jump.sum()), "jump_row_share": float(jump.mean()),
                    "jump_share_of_squared_error": float(sq[jump].sum() / sq.sum()),
                    "skill_all": float(1 - sq.mean() / sq_p.mean()),
                    "skill_without_jumps": float(1 - sq[~jump].mean() / sq_p[~jump].mean())})
    return out


def rain_forecast_skill(df: pd.DataFrame, h: int) -> list[dict]:
    """Rank correlation of the CFSv2 forecast anomaly with the rain that actually fell (oracle), test origins."""
    from scipy.stats import spearmanr

    t = df[df["split"] == "test"].drop_duplicates(["well_id", "campaign_idx"])
    rows = []
    for season, g in [("all", t)] + [(s, g) for s, g in t.groupby("origin_season")]:
        d = g[["fc_rain_ratio", "oracle_rain_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        label = season if season == "all" else ["Jan", "May", "Aug", "Nov"][int(season)]
        rows.append({"h": h, "origin": label, "n": len(d),
                     "spearman": float(spearmanr(d["fc_rain_ratio"], d["oracle_rain_ratio"]).statistic)})
    return rows


# ── Experiments ─────────────────────────────────────────────────────────────

def main_test(h: int, base: pd.DataFrame, ref: pd.DataFrame, seq: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = extended_dataset(h)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()
    tuning = {}
    for arm, feats in ((CFS_ARM, ARMS["M2 + ENSO/IOD"] + FORECAST), (NB_ARM, ARMS["M2 + ENSO/IOD"] + NEIGHBOURS)):
        model, info = tune_and_fit(train, feats)
        test[arm] = test["y_t"] + model.predict(test[feats])
        tuning[arm] = info
        print(f"[h={h}] {arm}: depth={info['best']['max_depth']} trees={info['n_trees_final']}")
    b = base[(base["h"] == h) & (base["split"] == "test")].dropna(subset=["anomaly_persistence"])
    m = b.merge(test[["well_id", "campaign_idx", CFS_ARM, NB_ARM]], on=["well_id", "campaign_idx"])
    m = m.merge(ref[ref["h"] == h].drop(columns="h"), on=["well_id", "campaign_idx"], how="left")
    if seq is not None:
        m = m.merge(seq[seq["h"] == h].drop(columns="h"), on=["well_id", "campaign_idx"], how="left")
    m = add_combinations(m).dropna(subset=[CHRONOS, "RF M2"] + ([LSTM[0]] if seq is not None else []))
    return m, df, tuning


def robustness(h: int) -> pd.DataFrame:
    """M2 and M2 + CFSv2 in the alternative split and in unseen spatial blocks; Chronos-2 for the combination."""
    df = extended_dataset(h)
    ref = pd.read_parquet(DATA_PROCESSED_DIR / "reference_predictions.parquet")
    ref = ref[ref["h"] == h][["well_id", "campaign_idx", CHRONOS, "RF M2"]]
    arms = {"M2 + ENSO/IOD": ARMS["M2 + ENSO/IOD"], CFS_ARM: ARMS["M2 + ENSO/IOD"] + FORECAST,
            "M3 + oracle rain": ARMS["M3 + oracle rain"]}
    parts = []

    train, test = df[df["target_year"] <= 2017], df[df["target_year"] >= 2020].copy()
    for arm, feats in arms.items():
        model, _ = tune_and_fit(train, feats, inner_fit_end_year=2014)
        test[arm] = test["y_t"] + model.predict(test[feats])
    parts.append(test.assign(setting="Split: train ≤2017, test 2020–25"))
    print(f"[h={h}] robustness: alternative split done")

    block = (np.floor(df["latitude"] / BLOCK_DEG).astype(int).astype(str) + "_"
             + np.floor(df["longitude"] / BLOCK_DEG).astype(int).astype(str))
    blocks = np.array(sorted(block.unique()))
    fold_of = dict(zip(blocks, np.random.default_rng(RANDOM_SEED).permutation(len(blocks)) % N_FOLDS))
    df["fold"] = block.map(fold_of)
    for k in range(N_FOLDS):
        train = df[(df["fold"] != k) & (df["split"] == "train")]
        test = df[(df["fold"] == k) & (df["split"] == "test")].copy()
        for arm, feats in arms.items():
            model, _ = tune_and_fit(train, feats)
            test[arm] = test["y_t"] + model.predict(test[feats])
        parts.append(test.assign(setting="Spatial blocks (unseen regions)"))
        print(f"[h={h}] robustness: spatial fold {k + 1}/{N_FOLDS} done")

    out = pd.concat(parts).dropna(subset=["anomaly_persistence"])
    out = out.merge(ref, on=["well_id", "campaign_idx"], how="inner")
    # The RF reference was fitted on the main split only, so the combination here is M2 (refitted) + Chronos-2
    out[ENS] = (out["M2 + ENSO/IOD"] + out[CHRONOS]) / 2
    return out


def run_main_test():
    print("=== Phase 7: Model extensions — main test ===")
    EXT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(DATA_PROCESSED_DIR / "skill_predictions.parquet")
    ref = pd.read_parquet(DATA_PROCESSED_DIR / "reference_predictions.parquet")
    seq_path = DATA_PROCESSED_DIR / "sequence_predictions.parquet"
    seq = pd.read_parquet(seq_path) if seq_path.exists() else None

    metrics, comps, jumps, rain_skill, boot_rows, preds, tuning = [], [], [], [], [], [], {}
    for h in HORIZONS:
        m, df, tuning[str(h)] = main_test(h, base, ref, seq)
        rain_skill += rain_forecast_skill(df, h)
        models = (["M0 memory", "M2 + ENSO/IOD", CFS_ARM, NB_ARM, "M3 + oracle rain", "RF M2", CHRONOS]
                  + (LSTM if seq is not None else []) + [ENS, ENS3] + ([ENS_ALL] if seq is not None else []))
        mse_p = np.mean((m["persistence"] - m["y_target"]) ** 2)
        boot = cluster_bootstrap(m, models)
        for c in models:
            err = m[c] - m["y_target"]
            metrics.append({"h": h, "months_ahead": 3 * h, "model": c, "n": len(m), "rmse": float(np.sqrt(np.mean(err ** 2))),
                            "mae": float(np.mean(np.abs(err))), "skill_vs_persistence": float(1 - np.mean(err ** 2) / mse_p),
                            "skill_ci": ci(1 - boot[c] / boot["persistence"])})
        pairs = [("M2 + ENSO/IOD", CFS_ARM), ("M2 + ENSO/IOD", NB_ARM), ("M2 + ENSO/IOD", ENS), ("M2 + ENSO/IOD", ENS3)]
        if seq is not None:
            pairs += [("M0 memory", "LSTM memory"), ("M2 + ENSO/IOD", "LSTM + forcing"), ("M2 + ENSO/IOD", ENS_ALL),
                      ("LSTM + forcing", ENS_ALL)]
        for a, b in pairs:
            comps.append(compare(m, a, b, h))
        gap = np.mean((m["M2 + ENSO/IOD"] - m["y_target"]) ** 2) - np.mean((m["M3 + oracle rain"] - m["y_target"]) ** 2)
        for c in (CFS_ARM, ENS) + ((ENS_ALL,) if seq is not None else ()):
            closed = np.mean((m["M2 + ENSO/IOD"] - m["y_target"]) ** 2) - np.mean((m[c] - m["y_target"]) ** 2)
            boot_rows.append({"h": h, "model": c, "share_of_oracle_gap_closed": float(closed / gap)})
        jumps += jump_attribution(m, ["persistence", "M0 memory", "M2 + ENSO/IOD", ENS], h)
        keep = [CFS_ARM, NB_ARM, ENS, ENS3] + ([ENS_ALL] if seq is not None else [])
        preds.append(m[["well_id", "campaign_idx", "target_idx", *keep]].assign(h=h))

    pd.DataFrame(metrics).to_csv(EXT_REPORTS_DIR / "metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(EXT_REPORTS_DIR / "comparisons.csv", index=False)
    pd.DataFrame(jumps).to_csv(EXT_REPORTS_DIR / "jump_attribution.csv", index=False)
    pd.DataFrame(rain_skill).to_csv(EXT_REPORTS_DIR / "cfsv2_rain_skill.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(EXT_REPORTS_DIR / "oracle_gap_closed.csv", index=False)
    pd.concat(preds).to_parquet(DATA_PROCESSED_DIR / "extension_predictions.parquet", index=False)
    with open(EXT_REPORTS_DIR / "tuning.json", "w") as f:
        json.dump(tuning, f, indent=2)

    print(pd.DataFrame(metrics).pivot_table(index="model", columns="h", values="skill_vs_persistence").round(3).to_string())
    for r in comps:
        print(f"  h={r['h']} {r['b']:<30} vs {r['a']:<14} MSE {r['mse_reduction']:+.1%}  MAE {r['mae_reduction']:+.1%}  "
              f"sign {r['campaigns_b_better']}/{r['campaigns']} p={r['sign_p_value']:.2g}  DM-MAE p={r['dm_mae_p']:.2g}")
    print(pd.DataFrame(rain_skill).round(3).to_string(index=False))
    print(pd.DataFrame(boot_rows).round(3).to_string(index=False))



def run_robustness():
    import os

    from src.research.skill_decomposition import BASE_PARAMS

    print("=== Phase 7: Model extensions — robustness ===")
    # GW_XGB_DEVICE=cuda fits XGBoost on the GPU (54 tuning runs). Every arm compared here is refitted in this
    # run, so comparisons stay like-for-like; the main test keeps the CPU to match the Phase 2 predictions.
    if os.environ.get("GW_XGB_DEVICE"):
        BASE_PARAMS["device"] = os.environ["GW_XGB_DEVICE"]
        print(f"  XGBoost device: {BASE_PARAMS['device']}")
    EXT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    seq_path = DATA_PROCESSED_DIR / "sequence_robustness_predictions.parquet"  # from sequence_baseline --robustness
    seq = pd.read_parquet(seq_path) if seq_path.exists() else None
    rows = []
    for h in HORIZONS:
        cached = DATA_PROCESSED_DIR / f"extension_robustness_h{h}.parquet"  # resumable: one file per horizon
        if cached.exists():
            r = pd.read_parquet(cached)
        else:
            r = robustness(h)
            r.to_parquet(cached, index=False)
        pairs = [("M2 + ENSO/IOD", CFS_ARM), ("M2 + ENSO/IOD", ENS), ("M2 + ENSO/IOD", "M3 + oracle rain")]
        if seq is not None:
            r = r.merge(seq[seq["h"] == h].drop(columns="h"), on=["well_id", "campaign_idx", "setting"], how="inner")
            r[ENS_ALL] = (r["M2 + ENSO/IOD"] + r["LSTM + forcing"] + r[CHRONOS]) / 3
            pairs += [("M2 + ENSO/IOD", "LSTM + forcing"), ("M2 + ENSO/IOD", ENS_ALL)]
        for setting, g in r.groupby("setting"):
            for a, b in pairs:
                rows.append(compare(g, a, b, h, setting))
    rob = pd.DataFrame(rows)
    rob.to_csv(EXT_REPORTS_DIR / "robustness.csv", index=False)
    for r in rows:
        print(f"  {r['setting']:<34} h={r['h']} {r['b']:<26} MSE {r['mse_reduction']:+.1%}  MAE {r['mae_reduction']:+.1%}  "
              f"sign {r['campaigns_b_better']}/{r['campaigns']} p={r['sign_p_value']:.2g}")


if __name__ == "__main__":
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # --main or --robustness runs one part (the robustness part does not need the LSTM); default runs both
        if "--robustness" not in sys.argv:
            run_main_test()
        if "--main" not in sys.argv:
            run_robustness()
