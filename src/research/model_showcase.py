"""
Research Pipeline — Model showcase for the dashboard
Puts every forecasting model on the same test rows and writes what the dashboard's Models section needs.

  main         skill (MSE and MAE) with well-clustered 95 % CIs, per-campaign MAE, every pairwise comparison
               (MSE reduction with CI, campaign sign test) and skill by depth, region and target season
  robustness   skill in the alternative split (train ≤ 2017, test 2020–2025) and in unseen 2° spatial blocks, for
               the models refitted in each setting. Random Forest M2 is refitted here (cached), so the combination
               chosen on the calibration years can be scored in both settings without seeing the test regions.
  replay       each model's test forecasts per well, for the well explorer (frontend/public/data/models/)
  timing       wall-clock training / inference time per model at the 6-month horizon on this machine

Rows are the common evaluation rows of the model extensions (31,468 / 30,424 / 32,472). The combination is the
one chosen on 2016–2018 (reports/audit/combination_selection.csv); the post-hoc combinations are not shown.
Outputs: reports/showcase/*.csv|json, frontend/public/data/models.json, frontend/public/data/models/{bucket}.json.
"""

import argparse
import json
import os
import time
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.research.benchmark import DEPTH_CLASS_NAMES
from src.research.config import (
    DATA_PROCESSED_DIR,
    HORIZONS,
    RANDOM_SEED,
    SKILL_REPORTS_DIR,
)
from src.research.export_dashboard import (
    FRONTEND_DATA_DIR,
    N_BUCKETS,
    bucket,
    label,
    strict_json,
)
from src.research.leakage_audit import AUDIT_DIR, test_predictions
from src.research.skill_decomposition import ARMS, load_dataset

SHOWCASE_DIR = SKILL_REPORTS_DIR.parent / "showcase"
N_BOOT = 1000
N_THREADS = int(os.environ.get("GW_THREADS", "8"))
COMBO = "Combination chosen on 2016–2018"

# key → (prediction column, family). Families drive colour and badges on the dashboard.
MODELS = {
    "persistence": ("persistence", "baseline"),
    "climatology": ("clim_target", "baseline"),
    "anomaly": ("anomaly_persistence", "baseline"),
    "m0": ("M0 memory", "boosting"),
    "m2": ("M2 + ENSO/IOD", "boosting"),
    "rf": ("RF M2", "learner"),
    "lstm_mem": ("LSTM memory", "learner"),
    "lstm": ("LSTM + forcing", "learner"),
    "chronos": ("Chronos-2 + past rain", "learner"),
    "combo": (COMBO, "combination"),
    "snapshot": ("C snapshot control", "reference"),
    "cfsv2": ("M2 + CFSv2 forecast", "reference"),
    "m3": ("M3 + oracle rain", "reference"),
}
# Models refitted (or zero-shot) in the robustness settings
ROBUST_MODELS = ["persistence", "climatology", "anomaly", "m2", "rf", "lstm", "chronos", "combo", "cfsv2", "m3"]
SETTINGS = {"Split: train ≤2017, test 2020–25": "split", "Spatial blocks (unseen regions)": "blocks"}
REPLAY_MODELS = ["persistence", "m0", "m2", "rf", "lstm", "chronos", "combo", "cfsv2", "m3"]
STRATA = {"depth": "depth_class", "region": "region", "season": "target_season"}


# ── Data ────────────────────────────────────────────────────────────────────

def selected_members() -> dict[int, list[str]]:
    sel = pd.read_csv(AUDIT_DIR / "combination_selection.csv")
    top = sel[(sel["period"] == "calibration 2016–2018") & (sel["rank_in_period"] == 1)]
    return {int(r.h): r.combination.split(" | ") for r in top.itertuples()}


def main_table() -> pd.DataFrame:
    """Test forecasts of every showcase model on the common evaluation rows."""
    t = test_predictions()
    members = selected_members()
    parts = []
    for h in HORIZONS:
        d = t[t["h"] == h].dropna(subset=["RF M2", "Chronos-2 + past rain", "LSTM + forcing", "LSTM memory"]).copy()
        extra = load_dataset(h)[["well_id", "campaign_idx", "clim_target"]]
        d = d.merge(extra, on=["well_id", "campaign_idx"], how="left")
        d[COMBO] = d[members[h]].mean(axis=1)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    cols = [c for c, _ in MODELS.values()]
    assert out[cols].notna().all().all(), out[cols].isna().sum()[lambda s: s > 0]
    return out


def rf_robustness(h: int) -> pd.DataFrame:
    """Random Forest M2 refitted in both robustness settings (same settings as reference_models)."""
    from sklearn.ensemble import RandomForestRegressor

    from src.research.skill_robustness import N_FOLDS, spatial_folds

    cached = DATA_PROCESSED_DIR / f"showcase_rf_robustness_h{h}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    df = spatial_folds(load_dataset(h))
    feats = ARMS["M2 + ENSO/IOD"]
    runs = [("Split: train ≤2017, test 2020–25", df["target_year"] <= 2017, df["target_year"] >= 2020)]
    runs += [("Spatial blocks (unseen regions)", (df["fold"] != k) & (df["split"] == "train"),
              (df["fold"] == k) & (df["split"] == "test")) for k in range(N_FOLDS)]
    parts = []
    for setting, train, test in runs:
        tr, te = df[train], df[test].copy()
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=N_THREADS,
                                   random_state=RANDOM_SEED)
        rf.fit(tr[feats], tr["y_target"] - tr["y_t"])
        te["RF M2"] = te["y_t"] + rf.predict(te[feats])
        parts.append(te[["well_id", "campaign_idx", "RF M2"]].assign(setting=setting))
        print(f"  [h={h}] RF refitted: {setting}, {len(te):,} test rows", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cached, index=False)
    return out


def robustness_table() -> pd.DataFrame:
    seq = pd.read_parquet(DATA_PROCESSED_DIR / "sequence_robustness_predictions.parquet")
    members = selected_members()
    parts = []
    for h in HORIZONS:
        ext = pd.read_parquet(DATA_PROCESSED_DIR / f"extension_robustness_h{h}.parquet").drop(columns=["RF M2"])
        keys = ["well_id", "campaign_idx", "setting"]
        d = ext.merge(rf_robustness(h), on=keys, how="inner")
        d = d.merge(seq[seq["h"] == h].drop(columns="h"), on=keys, how="inner")
        d = d.dropna(subset=["anomaly_persistence", "clim_target", "Chronos-2 + past rain"])
        d[COMBO] = d[members[h]].mean(axis=1)
        parts.append(d.assign(h=h))
    return pd.concat(parts, ignore_index=True)


# ── Statistics ──────────────────────────────────────────────────────────────

def boot_weights(wells: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(wells)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(float)


def model_scores(d: pd.DataFrame, keys: list[str]) -> tuple[list[dict], dict]:
    """Skill and MAE skill with well-clustered CIs; also returns per-well error sums for pairwise CIs."""
    wells, inv = np.unique(d["well_id"].to_numpy(), return_inverse=True)
    W = boot_weights(wells)
    count = np.bincount(inv, minlength=len(wells)).astype(float)
    y = d["y_target"].to_numpy()
    sq, ab = {}, {}
    for k in keys:
        e = d[MODELS[k][0]].to_numpy() - y
        sq[k] = np.bincount(inv, weights=e ** 2, minlength=len(wells))
        ab[k] = np.bincount(inv, weights=np.abs(e), minlength=len(wells))
    denom = W @ count
    mse_draw = {k: (W @ sq[k]) / denom for k in keys}
    mae_draw = {k: (W @ ab[k]) / denom for k in keys}
    rows = []
    for k in keys:
        mse, mae = sq[k].sum() / len(d), ab[k].sum() / len(d)
        mse_p, mae_p = sq["persistence"].sum() / len(d), ab["persistence"].sum() / len(d)
        s_draw = 1 - mse_draw[k] / mse_draw["persistence"]
        a_draw = 1 - mae_draw[k] / mae_draw["persistence"]
        rows.append({"model": k, "rows": len(d), "wells": len(wells), "campaigns": d["target_idx"].nunique(),
                     "skill": 1 - mse / mse_p, "skill_lo": float(np.percentile(s_draw, 2.5)),
                     "skill_hi": float(np.percentile(s_draw, 97.5)), "mae_skill": 1 - mae / mae_p,
                     "mae_skill_lo": float(np.percentile(a_draw, 2.5)), "mae_skill_hi": float(np.percentile(a_draw, 97.5)),
                     "rmse": float(np.sqrt(mse)), "mae": float(mae)})
    return rows, mse_draw


def pairwise(d: pd.DataFrame, keys: list[str], mse_draw: dict) -> list[dict]:
    """For every ordered pair (a, b): how much b reduces a's error, and in how many campaigns b has the lower MAE."""
    y = d["y_target"]
    camp_mae = {k: (d[MODELS[k][0]] - y).abs().groupby(d["target_idx"]).mean() for k in keys}
    mse = {k: float(np.mean((d[MODELS[k][0]] - y) ** 2)) for k in keys}
    mae = {k: float(np.mean((d[MODELS[k][0]] - y).abs())) for k in keys}
    rows = []
    for a, b in permutations(keys, 2):
        draws = 1 - mse_draw[b] / mse_draw[a]
        diff = camp_mae[a] - camp_mae[b]
        wins, n = int((diff > 0).sum()), int((diff != 0).sum())
        rows.append({"a": a, "b": b, "mse_reduction": 1 - mse[b] / mse[a],
                     "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
                     "mae_reduction": 1 - mae[b] / mae[a], "b_wins": wins, "campaigns": n,
                     "sign_p": float(binomtest(wins, n).pvalue) if n else 1.0})
    return rows


def campaign_mae(d: pd.DataFrame, keys: list[str]) -> list[dict]:
    g = d.groupby("target_idx")
    out = []
    for t, gc in g:
        rec = {"t": label(t), "n": len(gc)}
        rec.update({k: float(np.mean(np.abs(gc[MODELS[k][0]] - gc["y_target"]))) for k in keys})
        out.append(rec)
    return out


def strata(d: pd.DataFrame, keys: list[str]) -> list[dict]:
    d = d.assign(depth_name=d["depth_class"].map(lambda v: DEPTH_CLASS_NAMES[int(v)]),
                 season_name=d["target_season"].map(lambda v: ["Jan", "May", "Aug", "Nov"][int(v)]))
    cols = {"depth": "depth_name", "region": "region", "season": "season_name"}
    out = []
    for name, col in cols.items():
        for value, g in d.groupby(col):
            mp = np.mean((g["persistence"] - g["y_target"]) ** 2)
            for k in keys:
                out.append({"stratum": name, "value": str(value), "model": k, "rows": len(g),
                            "wells": g["well_id"].nunique(),
                            "skill": float(1 - np.mean((g[MODELS[k][0]] - g["y_target"]) ** 2) / mp)})
    return out


# ── Steps ───────────────────────────────────────────────────────────────────

def run_main() -> dict:
    d = main_table()
    keys = list(MODELS)
    out = {"scores": [], "pairs": [], "campaigns": [], "strata": []}
    for h in HORIZONS:
        dh = d[d["h"] == h]
        scores, draws = model_scores(dh, keys)
        out["scores"] += [{**r, "h": h, "setting": "main"} for r in scores]
        out["pairs"] += [{**r, "h": h} for r in pairwise(dh, keys, draws)]
        out["campaigns"] += [{**r, "h": h} for r in campaign_mae(dh, keys)]
        out["strata"] += [{**r, "h": h} for r in strata(dh, keys)]
        print(f"[h={h}] main: {len(dh):,} rows, {dh['well_id'].nunique():,} wells", flush=True)
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        pd.DataFrame(v).to_csv(SHOWCASE_DIR / f"{k}.csv", index=False)
    d.to_parquet(DATA_PROCESSED_DIR / "showcase_predictions.parquet", index=False)
    return out


def run_robustness() -> list[dict]:
    d = robustness_table()
    rows = []
    for (h, setting), g in d.groupby(["h", "setting"]):
        scores, _ = model_scores(g, ROBUST_MODELS)
        rows += [{**r, "h": int(h), "setting": SETTINGS[setting]} for r in scores]
        print(f"[h={h}] {setting}: {len(g):,} rows", flush=True)
    pd.DataFrame(rows).to_csv(SHOWCASE_DIR / "robustness.csv", index=False)
    return rows


def run_replay():
    d = pd.read_parquet(DATA_PROCESSED_DIR / "showcase_predictions.parquet")
    d["b"] = d["well_id"].map(bucket)
    out_dir = FRONTEND_DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [MODELS[k][0] for k in REPLAY_MODELS]
    for b in range(N_BUCKETS):
        g = d[d["b"] == b].sort_values(["well_id", "h", "target_idx"])
        payload: dict = {}
        for (well, h), gw in g.groupby(["well_id", "h"]):
            payload.setdefault(well, {})[str(h)] = {
                "t": [label(t) for t in gw["target_idx"]], "y": gw["y_target"].round(2).tolist(),
                **{k: gw[c].round(2).tolist() for k, c in zip(REPLAY_MODELS, cols)}}
        with open(out_dir / f"{b}.json", "w") as f:
            json.dump(strict_json(payload), f, separators=(",", ":"), allow_nan=False)
    print(f"Saved per-well replays for {d['well_id'].nunique():,} wells to {out_dir}/")


def run_timing() -> dict:
    """Wall-clock time of each model's full training procedure (or inference) at the 6-month horizon."""
    import torch

    from src.research.reference_models import chronos_predictions
    from src.research.sequence_baseline import fit_predict, sequence_tensor
    from src.research.skill_decomposition import INNER_FIT_END_YEAR, tune_and_fit

    h = 2
    df = load_dataset(h)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    out = {"h": h, "train_rows": len(train), "test_rows": len(test),
           "cpu_threads": N_THREADS, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def clock(name, fn):
        t0 = time.perf_counter()
        fn()
        out[name] = round(time.perf_counter() - t0, 1)
        print(f"  {name}: {out[name]} s", flush=True)

    clock("m0", lambda: tune_and_fit(train, ARMS["M0 memory"]))
    clock("m2", lambda: tune_and_fit(train, ARMS["M2 + ENSO/IOD"]))

    def rf():
        from sklearn.ensemble import RandomForestRegressor
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=N_THREADS,
                              random_state=RANDOM_SEED).fit(train[ARMS["M2 + ENSO/IOD"]], train["y_target"] - train["y_t"])
    clock("rf", rf)

    d = df.reset_index(drop=True)
    is_train = (d["split"] == "train").to_numpy()
    inner = is_train & (d["target_year"] <= INNER_FIT_END_YEAR).to_numpy()
    is_test = (d["split"] == "test").to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = sequence_tensor(d, True)
    clock("lstm", lambda: fit_predict(d, seq, "M2 + ENSO/IOD", is_train, inner, is_test, device, "[timing] LSTM"))
    origins = test[["well_id", "campaign_idx"]].drop_duplicates().reset_index(drop=True)
    clock("chronos", lambda: chronos_predictions(origins, only=["Chronos-2 + past rain"]))
    out["chronos_origins"] = len(origins)
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    (SHOWCASE_DIR / "timing.json").write_text(json.dumps(out, indent=2))
    return out


def run_export():
    """Writes frontend/public/data/models.json from reports/showcase/."""
    read = lambda name: pd.read_csv(SHOWCASE_DIR / f"{name}.csv")
    scores = pd.concat([read("scores"), read("robustness")])
    timing_path = SHOWCASE_DIR / "timing.json"
    members = selected_members()
    feats = {"m0": len(ARMS["M0 memory"]), "m2": len(ARMS["M2 + ENSO/IOD"]), "rf": len(ARMS["M2 + ENSO/IOD"]),
             "cfsv2": len(ARMS["M2 + ENSO/IOD"]) + 2, "m3": len(ARMS["M3 + oracle rain"]),
             "snapshot": len(ARMS["C snapshot control"])}
    short = {c: k for k, (c, _) in MODELS.items()}
    payload = {
        "source": "reports/showcase/ (src/research/model_showcase.py), from the saved test predictions of every model",
        "models": [{"key": k, "column": c, "family": fam, "inputs": feats.get(k)} for k, (c, fam) in MODELS.items()],
        "combination": {str(h): [short[m] for m in members[h]] for h in members},
        "scores": scores.to_dict("records"),
        "pairs": read("pairs").to_dict("records"),
        "campaigns": read("campaigns").to_dict("records"),
        "strata": read("strata").to_dict("records"),
        "timing": json.loads(timing_path.read_text()) if timing_path.exists() else None,
    }
    with open(FRONTEND_DATA_DIR / "models.json", "w") as f:
        json.dump(strict_json(payload), f, separators=(",", ":"), allow_nan=False)
    print(f"Saved {FRONTEND_DATA_DIR / 'models.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["main", "robustness", "replay", "timing", "export"])
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for step in args.steps:
            print(f"=== Model showcase: {step} ===", flush=True)
            {"main": run_main, "robustness": run_robustness, "replay": run_replay, "timing": run_timing,
             "export": run_export}[step]()
