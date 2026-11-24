"""
Figures, tables and number macros for the IEEE paper (paper/main.tex).

Every figure, table and number in the paper is generated here from reports/ and data/processed/, so the
paper cannot drift from the results. Output:
  paper/figures/{strata,coverage,value}.pdf
  paper/tables/{skill,models,calib,value}.tex
  paper/numbers.tex          \\newcommand macros for every number quoted in the text

Style: 8 pt sans-serif text, validated colour-blind-safe categorical slots.
"""

import json

import matplotlib
import numpy as np
import pandas as pd

from src.research.calibration import CALIBRATION_REPORTS_DIR
from src.research.config import (
    DATA_PROCESSED_DIR,
    EXTENSIONS_REPORTS_DIR,
    FORCING_REPORTS_DIR,
    SKILL_REPORTS_DIR,
)
from src.research.decision_value import DECISION_REPORTS_DIR
from src.research.leakage_audit import AUDIT_DIR as AUDIT_REPORTS_DIR
from src.research.regions import REGIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = DATA_PROCESSED_DIR.parent.parent / "paper"
FIG, TAB = PAPER / "figures", PAPER / "tables"
MM = 1 / 25.4
FULL = 174 * MM
BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#102a33", "#6a7d83"
ARM_LABELS = {"anomaly_persistence": "Anomaly persistence", "C snapshot control": "C: snapshot rainfall",
              "M0 memory": "M0: memory", "M1 + past forcing": "M1: + past forcing",
              "M2 + ENSO/IOD": "M2: + ENSO/IOD", "M3 + oracle rain": "M3: + oracle rain"}
H_LABEL = {1: "3 months", 2: "6 months", 4: "12 months"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titleweight": "bold",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Where forcing helps ───────────────────────────────────────────

def fig_strata():
    st = pd.read_csv(SKILL_REPORTS_DIR / "strata.csv")
    arms = ["M2 + ENSO/IOD", "M3 + oracle rain"]
    blocks = {"target_season": ["Jan", "May", "Aug", "Nov"], "depth_class": ["< 5 m", "5–15 m", "15–40 m", "≥ 40 m"],
              "region": REGIONS}
    titles = {"target_season": "(a) Target campaign", "depth_class": "(b) Depth at origin", "region": "(c) Region"}
    cols = [(a, h) for a in arms for h in (1, 2, 4)]
    fig, axes = plt.subplots(3, 1, figsize=(FULL * 0.8, 140 * MM), gridspec_kw={"hspace": 0.75})
    for ax, (k, vals) in zip(axes, blocks.items()):
        mat = np.array([[st[(st["stratum"] == k) & (st["value"] == v) & (st["arm"] == a) & (st["h"] == h)]["mse_reduction_vs_M0"].mean()
                         for a, h in cols] for v in vals]) * 100
        im = ax.imshow(mat, cmap="RdBu", vmin=-28, vmax=28, aspect="auto")
        ax.set_yticks(range(len(vals)), vals)
        ax.set_xticks(range(len(cols)), [f"{a.split()[0]}\n{H_LABEL[h]}" for a, h in cols], fontsize=7)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{round(v):+d}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 15 else INK)
        ax.set_title(titles[k], loc="left", fontsize=8)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("MSE reduction vs M0 (%)")
    save(fig, "strata")


# ── Conditional coverage ──────────────────────────────────────────

def fig_coverage():
    cov = pd.read_csv(CALIBRATION_REPORTS_DIR / "coverage.csv")
    methods = {"Split conformal": ORANGE, "Mondrian D×R": GREY, "CQR": "#6250d6", "Mondrian D×R + recal + ACI": BLUE}
    depths = ["< 5 m", "5–15 m", "15–40 m", "≥ 40 m"]
    regions = REGIONS
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 96 * MM), sharey="row")
    for col, h in enumerate((1, 2, 4)):
        for row, (gv, groups) in enumerate((("Depth", depths), ("Region", regions))):
            ax = axes[row, col]
            sub = cov[(cov["h"] == h) & (cov["nominal"] == 0.8) & (cov["group_var"] == gv)]
            ax.axhspan(77, 83, color="#e8eded", lw=0, zorder=0)
            ax.axhline(80, color=INK, lw=0.6, ls="--")
            for i, (m, c) in enumerate(methods.items()):
                v = [sub[(sub["method"] == m) & (sub["group"] == g)]["coverage"].mean() * 100 for g in groups]
                ax.plot(np.arange(len(groups)) + (i - 1.5) * 0.07, v, "o-", color=c, ms=3, lw=0.9, label=m if col == 0 and row == 0 else None)
            ax.set_xticks(range(len(groups)), [g.replace("Indo-Gangetic plain", "IGP").replace("Himalaya & north-east", "Him. & NE")
                                                .replace("Arid north-west", "Arid NW") for g in groups], fontsize=6.8)
            if row == 0:
                ax.set_title(H_LABEL[h])
            if col == 0:
                ax.set_ylabel(f"80 % coverage by {gv.lower()} (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "coverage")


# ── Economic value ────────────────────────────────────────────────

def fig_value():
    v = pd.read_csv(DECISION_REPORTS_DIR / "value_curves.csv")
    v = v[v["probability"] == "rolling isotonic"]
    arms = {"M0 memory": BLUE, "M2 + ENSO/IOD": AQUA, "M3 + oracle rain": INK}
    events = {"E1": "Deeper than usual (E1)", "E3": "Below 8 m (E3)"}
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 96 * MM), sharex=True, sharey=True)
    for row, (ev, name) in enumerate(events.items()):
        for col, h in enumerate((1, 2, 4)):
            ax = axes[row, col]
            for a, c in arms.items():
                s = v[(v["h"] == h) & (v["event_id"] == ev) & (v["arm"] == a)]
                ax.fill_between(s["alpha"], s["value_lo"], s["value_hi"], color=c, alpha=0.18, lw=0)
                ax.plot(s["alpha"], s["value"], color=c, lw=1, label=ARM_LABELS[a])
            ax.axhline(0, color=INK, lw=0.6)
            ax.set_ylim(-0.15, 0.8)
            if row == 0:
                ax.set_title(H_LABEL[h])
            if row == 1:
                ax.set_xlabel("Cost–loss ratio C/L")
            if col == 0:
                ax.set_ylabel(f"{name}\nrelative value V")
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "value")


# ── Numbers quoted in the text ──────────────────────────────────────────────

def numbers():
    cat = pd.read_parquet(DATA_PROCESSED_DIR / "station_catalog.parquet")
    camp = pd.read_parquet(DATA_PROCESSED_DIR / "gwl_campaigns.parquet")
    dm = pd.read_csv(SKILL_REPORTS_DIR / "diebold_mariano.csv").set_index(["h", "a", "b"])
    cov = pd.read_csv(CALIBRATION_REPORTS_DIR / "coverage.csv")
    vc = pd.read_csv(DECISION_REPORTS_DIR / "value_curves.csv")
    vc = vc[vc["probability"] == "rolling isotonic"]
    trig = pd.read_csv(DECISION_REPORTS_DIR / "triggers.csv")
    trig = trig[(trig["probability"] == "rolling isotonic") & (trig["arm"] == "M2 + ENSO/IOD")]
    sanity = json.loads((FORCING_REPORTS_DIR / "sanity.json").read_text())
    boot = {(b["h"], b["model"]): b for b in json.loads((SKILL_REPORTS_DIR / "bootstrap.json").read_text()) if b["stat"] == "skill_vs_persistence"}

    def split_cov(group, h=1):
        return cov[(cov["method"] == "Split conformal") & (cov["h"] == h) & (cov["nominal"] == 0.8) & (cov["group"] == group)]["coverage"].iloc[0]

    aci = cov[(cov["method"] == "Mondrian D×R + recal + ACI") & cov["nominal"].isin([0.8, 0.9]) & cov["group_var"].isin(["Depth", "Region"])]
    aci_worst = {h: (g["coverage"] - g["nominal"]).abs().max() * 100 for h, g in aci.groupby("h")}
    v = lambda h, ev, a, al: vc[(vc["h"] == h) & (vc["event_id"] == ev) & (vc["arm"] == a) & (vc["alpha"] == al)].iloc[0]
    t = lambda h, ev, tr: trig[(trig["h"] == h) & (trig["event_id"] == ev) & (trig["trigger"] == tr)].iloc[0]
    g6 = dm.loc[(2, "M0 memory", "M2 + ENSO/IOD")]
    vmax = lambda ev, h: vc[(vc["event_id"] == ev) & (vc["h"] == h) & (vc["arm"] == "M2 + ENSO/IOD")]["value"].max()
    oracle = [dm.loc[(h, "M2 + ENSO/IOD", "M3 + oracle rain")]["mse_reduction"] * 100 for h in (1, 2, 4)]
    N = {
        "nStations": f"{len(cat):,}", "nManual": f"{(cat['network'] == 'manual').sum():,}",
        "nDWLR": f"{(cat['network'] == 'DWLR').sum():,}", "nObs": f"{int(cat['n_obs'].sum()):,}",
        "nCampaigns": f"{len(camp):,}", "nCampaignWells": f"{camp['well_id'].nunique():,}",
        "snapMedian": f"{sanity['snapshot_vs_imd_12m']['median_snap_cum4_mm']:.1f}",
        "imdMedian": f"{sanity['snapshot_vs_imd_12m']['median_imd_rain_12m_mm']:,.0f}",
        "rechargeShare": f"{sanity['recharge_vs_jjas']['share_positive'] * 100:.0f}",
        "skillMzeroThree": f"{boot[(1, 'M0 memory')]['estimate']:.2f}", "skillMzeroSix": f"{boot[(2, 'M0 memory')]['estimate']:.2f}",
        "skillMzeroTwelve": f"{boot[(4, 'M0 memory')]['estimate']:.2f}",
        "gainSixMSE": f"{g6['mse_reduction'] * 100:.1f}", "gainSixMAE": f"{g6['mae_reduction'] * 100:.1f}",
        "gainSixWins": f"{int(g6['campaigns_b_better'])}", "gainSixCampaigns": f"{int(g6['campaigns'])}",
        "oracleMin": f"{min(oracle):.1f}", "oracleMax": f"{max(oracle):.1f}",  # main split only
        "oracleRobustMin": f"{min(o * 100 for o in pd.read_csv(SKILL_REPORTS_DIR / 'robustness.csv').query('a == \"M2 + ENSO/IOD\" and b == \"M3 + oracle rain\"')['mse_reduction']):.1f}",
        "splitShallow": f"{split_cov('< 5 m') * 100:.0f}", "splitDeep": f"{split_cov('≥ 40 m') * 100:.0f}",
        "aciWorstThree": f"{aci_worst[1]:.1f}", "aciWorstSix": f"{aci_worst[2]:.1f}", "aciWorstTwelve": f"{aci_worst[4]:.2f}",
        "vmaxEoneThree": f"{vmax('E1', 1):.2f}", "vmaxEthreeSix": f"{vmax('E3', 2):.2f}",
        "valEoneThreeTrig": f"{v(1, 'E1', 'M2 + ENSO/IOD', 0.3)['value']:.2f}",
        "hitEoneThree": f"{t(1, 'E1', 0.3)['hit_rate'] * 100:.0f}", "farEoneThree": f"{t(1, 'E1', 0.3)['false_alarm_ratio'] * 100:.0f}",
        "hitEthreeSix": f"{t(2, 'E3', 0.2)['hit_rate'] * 100:.0f}", "farEthreeSix": f"{t(2, 'E3', 0.2)['false_alarm_ratio'] * 100:.0f}",
        "valEthreeSixTrig": f"{v(2, 'E3', 'M2 + ENSO/IOD', 0.2)['value']:.2f}",
    }
    N.update(audit_numbers())
    PAPER.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by src/research/paper_figures.py — do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{val}}}" for k, val in N.items()]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    return N


def audit_numbers() -> dict:
    """Macros for the model extensions and the leakage audit (reports/extensions/, reports/audit/)."""
    facts = pd.read_csv(AUDIT_REPORTS_DIR / "facts.csv").set_index("fact")["value"]
    fw = [int(facts[f"h={h}: wells with at least one forecast row"]) for h in (1, 2, 4)]
    tw = [int(facts[f"h={h}: test wells"]) for h in (1, 2, 4)]
    sp = pd.read_csv(AUDIT_REPORTS_DIR / "spatial_settings.csv")
    strict = sp[sp["setting"].str.startswith("time + region")].set_index("h")["skill_M2"]
    region_only = sp[sp["setting"].str.startswith("region only")].set_index("h")["skill_M2"]
    cb = pd.read_csv(AUDIT_REPORTS_DIR / "combination_bootstrap.csv")
    cb = cb[cb["model"] == "calibration-selected"]
    c = lambda h, ref: cb[(cb["h"] == h) & (cb["reference"] == ref)].iloc[0]
    best = lambda h: cb[cb["h"] == h]["best_single_model"].iloc[0]
    num = lambda x: f"{x:.1f}".replace("-", "$-$")
    pct_ci = lambda r: "{}\\,\\% [{}, {}]".format(num(r["mse_reduction"] * 100), *(num(x * 100) for x in json.loads(r["ci95_wells"])))
    ext = pd.read_csv(EXTENSIONS_REPORTS_DIR / "metrics.csv").set_index(["h", "model"])["skill_vs_persistence"]
    cmp_ = pd.read_csv(EXTENSIONS_REPORTS_DIR / "comparisons.csv").set_index(["h", "a", "b"])
    rain = pd.read_csv(EXTENSIONS_REPORTS_DIR / "cfsv2_rain_skill.csv")
    rain = rain[rain["origin"] == "all"].set_index("h")["spearman"]
    qc = pd.read_csv(AUDIT_REPORTS_DIR / "qc_sensitivity.csv").set_index(["qc", "h"])
    vs = pd.read_csv(AUDIT_REPORTS_DIR / "value_sensitivity.csv").set_index(["event", "h"])
    lstm6 = cmp_.loc[(2, "M2 + ENSO/IOD", "LSTM + forcing")]
    return {
        "nForecastWellsMin": f"{min(fw):,}", "nForecastWellsMax": f"{max(fw):,}",
        "nTestWellsMin": f"{min(tw):,}", "nTestWellsMax": f"{max(tw):,}",
        "spatialStrictThree": f"{strict[1]:.3f}", "spatialStrictSix": f"{strict[2]:.3f}",
        "spatialStrictTwelve": f"{strict[4]:.3f}", "spatialRegionOnlySix": f"{region_only[2]:.3f}",
        "combSkillThree": f"{c(1, 'persistence')['mse_reduction']:.3f}",
        "combSkillSix": f"{c(2, 'persistence')['mse_reduction']:.3f}",
        "combSkillTwelve": f"{c(4, 'persistence')['mse_reduction']:.3f}",
        "combGainThree": pct_ci(c(1, "M2 + ENSO/IOD")), "combGainSix": pct_ci(c(2, "M2 + ENSO/IOD")),
        "combGainTwelve": pct_ci(c(4, "M2 + ENSO/IOD")),
        "combVsBestThree": pct_ci(c(1, best(1))), "combVsBestSix": pct_ci(c(2, best(2))),
        "combVsBestTwelve": pct_ci(c(4, best(4))),
        "combVsOracleThree": pct_ci(c(1, "M3 + oracle rain")), "combVsOracleSix": pct_ci(c(2, "M3 + oracle rain")),
        "lstmSkillThree": f"{ext[(1, 'LSTM + forcing')]:.3f}", "lstmSkillSix": f"{ext[(2, 'LSTM + forcing')]:.3f}",
        "lstmSkillTwelve": f"{ext[(4, 'LSTM + forcing')]:.3f}",
        "lstmGainSix": f"{lstm6['mse_reduction'] * 100:.1f}", "lstmGainSixWins": f"{int(lstm6['campaigns_b_better'])}",
        "lstmGainSixP": f"{lstm6['sign_p_value']:.2f}",
        "lstmMemGainSix": f"{cmp_.loc[(2, 'M0 memory', 'LSTM memory')]['mse_reduction'] * 100:.1f}",
        "lstmMemGainSixWins": f"{int(cmp_.loc[(2, 'M0 memory', 'LSTM memory')]['campaigns_b_better'])}",
        "cfsRhoThree": f"{rain[1]:.2f}", "cfsRhoSix": f"{rain[2]:.2f}", "cfsRhoTwelve": f"{rain[4]:.2f}",
        "qcRangeGainSix": f"{qc.loc[('range only', 2), 'mse_reduction_M2_vs_M0'] * 100:.1f}",
        "qcTrainSkillSix": f"{qc.loc[('train-period MAD', 2), 'skill_M2']:.3f}",
        "valueEoneThreeRange": vs.loc[("E1", 1), "alpha_range_ci_above_zero"].replace("–", "--"),
        "valueEthreeSixRange": vs.loc[("E3", 2), "alpha_range_ci_above_zero"].replace("–", "--"),
    }


# ── Tables (one column each) ────────────────────────────────────────────────

def ieee_table(path, caption, label, header, rows, colspec, note=None):
    """Single-column IEEE table (booktabs, footnotesize)."""
    lines = [r"\begin{table}[!t]", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip\parbox{\columnwidth}{\scriptsize " + note + "}"]
    lines += [r"\end{table}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / path).write_text("\n".join(lines) + "\n")


def ieee_tables():
    """Four tables that fit one column each; every value is read from reports/."""
    f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
    boot = {(b["h"], b["model"]): b for b in json.loads((SKILL_REPORTS_DIR / "bootstrap.json").read_text())}
    dm = pd.read_csv(SKILL_REPORTS_DIR / "diebold_mariano.csv").set_index(["h", "a", "b"])

    # I. Nested arms: skill against persistence, and the forcing and oracle gains
    arms = [("anomaly_persistence", "Anomaly persistence"), ("M0 memory", "M0 memory"),
            ("C snapshot control", "C one-day rainfall"), ("M1 + past forcing", "M1 + past weather"),
            ("M2 + ENSO/IOD", "M2 + ENSO/IOD"), ("M3 + oracle rain", "M3 + future rain$^\\ast$")]
    rows = [[name] + [f3(boot[(h, k)]["estimate"]) for h in (1, 2, 4)] for k, name in arms]
    for a, b, name in (("M0 memory", "M2 + ENSO/IOD", "M2 vs M0 (\\%)"), ("M2 + ENSO/IOD", "M3 + oracle rain", "M3 vs M2 (\\%)")):
        rows.append([name] + [f"{dm.loc[(h, a, b)]['mse_reduction'] * 100:.1f} ({int(dm.loc[(h, a, b)]['campaigns_b_better'])}/{int(dm.loc[(h, a, b)]['campaigns'])})"
                              for h in (1, 2, 4)])
    n = pd.read_csv(SKILL_REPORTS_DIR / "metrics.csv").groupby("h")["n"].first()
    ieee_table("skill.tex", "Skill against persistence of the nested XGBoost arms, 2019--2025", "tab:skill",
               ["Model", "3 mo", "6 mo", "12 mo"], rows, "lrrr",
               f"Rows: {n[1]:,} / {n[2]:,} / {n[4]:,} forecasts. Bottom rows: MSE reduction (campaigns better). "
               "$^\\ast$Uses rain that falls after the forecast; an upper bound, not a forecast.")

    # II. Every model on the same rows (model showcase)
    sc = pd.concat([pd.read_csv(SKILL_REPORTS_DIR.parent / "showcase" / f) for f in ("scores.csv", "robustness.csv")])
    names = [("anomaly", "Anomaly persistence"), ("chronos", "Chronos-2 (zero-shot)"), ("m0", "XGBoost M0"),
             ("rf", "Random Forest M2"), ("m2", "XGBoost M2"), ("cfsv2", "M2 + CFSv2 forecast"),
             ("lstm", "LSTM + rainfall"), ("combo", "Combination$^\\dagger$"), ("m3", "M3 + future rain$^\\ast$")]
    rows = []
    for k, name in names:
        cells = []
        for setting in ("main", "blocks"):
            for h in (1, 2, 4):
                r = sc[(sc["model"] == k) & (sc["h"] == h) & (sc["setting"] == setting)]
                cells.append(f3(r["skill"].iloc[0]) if len(r) else "--")
        rows.append([name] + cells)
    nm = sc[(sc["setting"] == "main") & (sc["model"] == "m2")].set_index("h")["rows"]
    ieee_table("models.tex", "Skill of every model on identical rows: test years (left) and unseen 2$^\\circ$ regions (right)",
               "tab:models", ["Model", "3", "6", "12", "3", "6", "12"], rows, "lrrrrrr",
               f"Horizons in months. Rows: {nm[1]:,} / {nm[2]:,} / {nm[4]:,} (both settings). "
               "$^\\dagger$Equal-weight mean chosen on 2016--2018: M2 + RF + LSTM (3, 6 mo), M2 + RF + Chronos-2 (12 mo). "
               "$^\\ast$Not a forecast. Baselines and Chronos-2 are not trained, so they do not change; -- not refitted there.")

    # III. Interval calibration
    cov = pd.read_csv(CALIBRATION_REPORTS_DIR / "coverage.csv")
    methods = [("Gaussian", "Gaussian"), ("Split conformal", "Split conformal"), ("Mondrian D×R", "Pools D$\\times$R, fixed"),
               ("CQR", "CQR"), ("Mondrian D×R + recal + ACI", "Pools D$\\times$R, online + ACI")]
    rows = []
    for k, name in methods:
        cells = []
        for h in (1, 2, 4):
            g = cov[(cov["method"] == k) & (cov["h"] == h)]
            overall = g[(g["group_var"] == "All") & (g["nominal"] == 0.8)]["coverage"].iloc[0]
            worst = (g[g["group_var"].isin(["Depth", "Region"]) & g["nominal"].isin([0.8, 0.9])]
                     .eval("abs(coverage - nominal)").max())
            cells += [f"{overall * 100:.1f}", f"{worst * 100:.1f}"]
        rows.append([name] + cells)
    ieee_table("calib.tex", "Coverage of 80\\,\\% intervals (\\%) and the worst miss over depth and region classes at 80 and 90\\,\\% (points)",
               "tab:calib", ["Method", "Cov.", "Worst", "Cov.", "Worst", "Cov.", "Worst"], rows, "lrrrrrr",
               "Column pairs: 3, 6 and 12 months. D: depth class; R: region.")

    # IV. Warnings under assumed cost-loss ratios
    scores = pd.DataFrame(json.loads((DECISION_REPORTS_DIR / "scores.json").read_text()))
    scores = scores[(scores["arm"] == "M2 + ENSO/IOD") & (scores["probability"] == "rolling isotonic")]
    vs = pd.read_csv(AUDIT_REPORTS_DIR / "value_sensitivity.csv").set_index(["event", "h"])
    ev = {"E1": "Deeper than usual", "E2": "SGI $<-1$", "E3": "Below 8\\,m"}
    rows = []
    for e, name in ev.items():
        for h in (1, 2, 4):
            sc_ = scores[(scores["event_id"] == e) & (scores["h"] == h)].iloc[0]
            v = vs.loc[(e, h)]
            rows.append([name if h == 1 else "", f"{3 * h}", f"{sc_['bss']:.2f}", f"{sc_['roc_auc']:.2f}",
                         v["V@0.1"].split(" ")[0].replace("-", "$-$"), v["V@0.3"].split(" ")[0].replace("-", "$-$"),
                         v["alpha_range_ci_above_zero"].replace("–", "--")])
    ieee_table("value.tex", "Warnings from M2 with rolling recalibration: Brier skill, ROC AUC and relative value $V$ at assumed cost--loss ratios",
               "tab:value", ["Event", "Mo", "BSS", "AUC", "$V_{0.1}$", "$V_{0.3}$", "$V>0$ for $C/L$"], rows, "lrrrrrl",
               "No costs were measured: $C/L$ is assumed. Last column: ratios whose 95\\,\\% interval of $V$ is above zero.")


def run_paper_figures():
    print("=== Paper figures, tables and numbers ===")
    for fn in (fig_strata, fig_coverage, fig_value):
        fn()
        print(f"  {fn.__name__}")
    ieee_tables()
    N = numbers()
    print(f"Saved 3 figures, 4 tables and {len(N)} number macros to {PAPER}/")


if __name__ == "__main__":
    run_paper_figures()
