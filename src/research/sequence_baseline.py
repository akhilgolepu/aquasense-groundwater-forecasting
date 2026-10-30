"""
Research Pipeline — Phase 7.2: Global sequence deep-learning baseline (G5)
Does a recurrent network that reads each well's campaign sequence beat memory features + gradient
boosting under the same time-blocked protocol?

One LSTM per horizon, trained on all wells at once (a "global" model, as in Malakar et al. 2021):
  sequence  last SEQ_LEN campaigns: depth relative to the current depth, observed flag, season one-hot
            (+ rain since the previous campaign in the forcing variant)
  static    the arm's tabular features at the origin (standardised on training rows, NaN → 0 + flag)
  output    change in depth to the target campaign (same target as the XGBoost arms)

  LSTM memory     static = M0 features             ↔ compare with XGBoost M0
  LSTM + forcing  static = M2 features, rain steps ↔ compare with XGBoost M2

Protocol mirrors `tune_and_fit`: fit on target years ≤ 2012, early-stop on 2013–2015, then refit on all
training years for the selected number of epochs. Each variant is the mean of N_SEEDS networks.
Writes `sequence_predictions.parquet` (test rows of the Phase 2 predictions).
"""

import warnings

import numpy as np
import pandas as pd

from src.research.benchmark import add_expanding_climatology, complete_grid
from src.research.config import DATA_PROCESSED_DIR, HORIZONS, RANDOM_SEED
from src.research.skill_decomposition import ARMS, INNER_FIT_END_YEAR, load_dataset

SEQ_LEN = 16          # campaigns (4 years)
DEPTH_SCALE = 5.0     # metres; relative depths are divided by this and clipped to ±10
HIDDEN = 64
BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 6
N_SEEDS = 3
VARIANTS = {"LSTM memory": ("M0 memory", False), "LSTM + forcing": ("M2 + ENSO/IOD", True)}


def sequence_tensor(df: pd.DataFrame, with_rain: bool) -> np.ndarray:
    """[rows, SEQ_LEN, channels] for each (well, origin) in df, oldest step first."""
    campaigns = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet")
    grid = add_expanding_climatology(complete_grid(campaigns))[["well_id", "campaign_idx", "depth_mbgl", "season"]]
    if with_rain:
        rain = pd.read_parquet(DATA_PROCESSED_DIR / "forcing_features.parquet")[["well_id", "campaign_idx", "rain_since_prev"]]
        grid = grid.merge(rain, on=["well_id", "campaign_idx"], how="left")
        mu, sd = grid["rain_since_prev"].mean(), grid["rain_since_prev"].std()
        grid["rain_z"] = ((grid["rain_since_prev"] - mu) / sd).fillna(0.0)
    by_well = grid.groupby("well_id")
    pos = pd.Series(np.arange(len(grid)), index=pd.MultiIndex.from_frame(grid[["well_id", "campaign_idx"]]))
    rows = pos.reindex(pd.MultiIndex.from_frame(df[["well_id", "campaign_idx"]])).to_numpy()

    y_t = df["y_t"].to_numpy()
    steps = []
    for k in range(SEQ_LEN - 1, -1, -1):  # oldest → newest
        depth = by_well["depth_mbgl"].shift(k).to_numpy()[rows]
        season = ((df["campaign_idx"].to_numpy() - k) % 4).astype(int)
        rel = np.clip((depth - y_t) / DEPTH_SCALE, -10, 10)
        obs = ~np.isnan(rel)
        ch = [np.where(obs, rel, 0.0), obs.astype(float)] + [(season == s).astype(float) for s in range(4)]
        if with_rain:
            ch.append(by_well["rain_z"].shift(k).fillna(0.0).to_numpy()[rows])
        steps.append(np.stack(ch, axis=1))
    return np.stack(steps, axis=1).astype(np.float32)


def static_matrix(df: pd.DataFrame, feats: list[str], train_mask: np.ndarray) -> np.ndarray:
    x = df[feats].to_numpy(dtype=float)
    mu, sd = np.nanmean(x[train_mask], axis=0), np.nanstd(x[train_mask], axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    flags = np.isnan(z)[:, np.isnan(x[train_mask]).any(axis=0)]
    return np.concatenate([np.nan_to_num(z), flags.astype(float)], axis=1).astype(np.float32)


def _net(n_ch: int, n_static: int):
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_ch, HIDDEN, batch_first=True, num_layers=2, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(HIDDEN + n_static, 128), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, seq, static):
            import torch
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)

    return Net()


def _train(seq, static, y, val, epochs, seed, device):
    """Trains on (seq, static, y); with `val` early-stops and returns the best epoch, else trains `epochs`."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _net(seq.shape[2], static.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    S, X, Y = (torch.from_numpy(a).to(device) for a in (seq, static, y))
    best, best_epoch, bad = np.inf, 0, 0
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(Y)), max(1, len(Y) // BATCH)):
            b = torch.from_numpy(b).to(device)
            opt.zero_grad()
            loss = torch.mean((net(S[b], X[b]) - Y[b]) ** 2)
            loss.backward()
            opt.step()
        if val is not None:
            v = float(np.mean((_predict(net, val[0], val[1], device) - val[2]) ** 2))
            if v < best:
                best, best_epoch, bad = v, epoch, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    return net, best_epoch


def _predict(net, seq, static, device):
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), 8192):
            out.append(net(torch.from_numpy(seq[i:i + 8192]).to(device),
                           torch.from_numpy(static[i:i + 8192]).to(device)).cpu().numpy())
    return np.concatenate(out)


def fit_predict(df: pd.DataFrame, seq: np.ndarray, arm: str, is_train: np.ndarray, inner: np.ndarray,
                is_test: np.ndarray, device: str, label: str) -> np.ndarray:
    """Early-stops on train rows outside `inner`, refits on all train rows; mean over N_SEEDS; returns depth forecasts."""
    y = (df["y_target"] - df["y_t"]).to_numpy()
    y_sd = float(np.std(y[is_train]))
    yz = (y / y_sd).astype(np.float32)
    static = static_matrix(df, ARMS[arm], is_train)
    val = is_train & ~inner
    preds = []
    for s in range(N_SEEDS):
        seed = RANDOM_SEED + s
        _, n_ep = _train(seq[inner], static[inner], yz[inner], (seq[val], static[val], yz[val]), None, seed, device)
        n_ep = max(1, int(np.ceil(n_ep * is_train.sum() / inner.sum())))
        net, _ = _train(seq[is_train], static[is_train], yz[is_train], None, n_ep, seed, device)
        preds.append(_predict(net, seq[is_test], static[is_test], device) * y_sd)
        print(f"{label} seed {s}: {n_ep} epochs", flush=True)
    return df.loc[is_test, "y_t"].to_numpy() + np.mean(preds, axis=0)


def run_sequence_robustness():
    """
    LSTM + forcing in the two Phase 2.7 robustness settings (train ≤ 2017 / test 2020–25; unseen 2° spatial
    blocks, same folds as skill_robustness), so the model combinations can be checked there too.
    Writes `sequence_robustness_predictions.parquet`; one cache file per (horizon, setting, fold).
    """
    import torch

    from src.research.skill_robustness import BLOCK_DEG, N_FOLDS

    print("=== Phase 7.2b: LSTM + forcing in the robustness settings ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = DATA_PROCESSED_DIR / "sequence_cache"
    cache.mkdir(exist_ok=True)
    name, (arm, with_rain) = "LSTM + forcing", VARIANTS["LSTM + forcing"]
    out = []
    for h in HORIZONS:
        df = load_dataset(h).reset_index(drop=True)
        seq = sequence_tensor(df, with_rain)
        year = df["target_year"].to_numpy()
        block = (np.floor(df["latitude"] / BLOCK_DEG).astype(int).astype(str) + "_"
                 + np.floor(df["longitude"] / BLOCK_DEG).astype(int).astype(str))
        blocks = np.array(sorted(block.unique()))
        fold_of = dict(zip(blocks, np.random.default_rng(RANDOM_SEED).permutation(len(blocks)) % N_FOLDS))
        fold = block.map(fold_of).to_numpy()
        is_train, is_test = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
        runs = [("Split: train ≤2017, test 2020–25", 0, year <= 2017, year <= 2014, year >= 2020)]
        runs += [("Spatial blocks (unseen regions)", k, (fold != k) & is_train,
                  (fold != k) & is_train & (year <= INNER_FIT_END_YEAR), (fold == k) & is_test) for k in range(N_FOLDS)]
        for setting, k, train, inner, test in runs:
            cached = cache / f"robust_h{h}_{'split' if setting.startswith('Split') else 'block'}{k}.npy"
            if cached.exists():
                pred = np.load(cached)
            else:
                pred = fit_predict(df, seq, arm, train, inner, test, device, f"[h={h}] {setting} {k}")
                np.save(cached, pred)
            out.append(df.loc[test, ["well_id", "campaign_idx"]].assign(h=h, setting=setting, **{name: pred}))
    pd.concat(out).to_parquet(DATA_PROCESSED_DIR / "sequence_robustness_predictions.parquet", index=False)
    print("Saved sequence_robustness_predictions.parquet")


def run_sequence_baseline():
    import os

    import torch

    print("=== Phase 7.2: Global LSTM baseline ===")
    # GW_DEVICE=cpu keeps the GPU free for other jobs; each (horizon, variant) is cached as it finishes
    device = os.environ.get("GW_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("GW_THREADS", "8")))
    cache = DATA_PROCESSED_DIR / "sequence_cache"
    cache.mkdir(exist_ok=True)
    base = pd.read_parquet(DATA_PROCESSED_DIR / "skill_predictions.parquet")
    out = []
    for h in HORIZONS:
        df = load_dataset(h).reset_index(drop=True)
        is_train = (df["split"] == "train").to_numpy()
        inner = is_train & (df["target_year"] <= INNER_FIT_END_YEAR).to_numpy()
        is_test = (df["split"] == "test").to_numpy()
        res = df.loc[is_test, ["well_id", "campaign_idx"]].copy()
        for name, (arm, with_rain) in VARIANTS.items():
            cached = cache / f"h{h}_{name.replace(' ', '_').replace('+', 'plus')}.npy"
            if cached.exists():
                res[name] = np.load(cached)
                print(f"[h={h}] {name}: cached")
                continue
            seq = sequence_tensor(df, with_rain)
            res[name] = fit_predict(df, seq, arm, is_train, inner, is_test, device, f"[h={h}] {name}")
            np.save(cached, res[name].to_numpy())
        out.append(res.assign(h=h))
    pred = pd.concat(out)
    pred.to_parquet(DATA_PROCESSED_DIR / "sequence_predictions.parquet", index=False)

    for h in HORIZONS:
        b = base[(base["h"] == h) & (base["split"] == "test")].dropna(subset=["anomaly_persistence"])
        m = b.merge(pred[pred["h"] == h], on=["well_id", "campaign_idx"])
        mse_p = np.mean((m["persistence"] - m["y_target"]) ** 2)
        line = "  ".join(f"{c} {1 - np.mean((m[c] - m['y_target']) ** 2) / mse_p:.3f}"
                         for c in ["M0 memory", "M2 + ENSO/IOD", *VARIANTS])
        print(f"h={h} skill: {line}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import sys

        run_sequence_robustness() if "--robustness" in sys.argv else run_sequence_baseline()
