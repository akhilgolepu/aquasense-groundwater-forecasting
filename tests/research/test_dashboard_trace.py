"""
Phase 5 exit check: every number the dashboard shows is traceable to a file in reports/.
Compares frontend/public/data/*.json (written by src.research.export_dashboard) with the report files.
Skipped when the dashboard data or reports have not been generated.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "frontend" / "public" / "data"
REPORTS = ROOT / "reports"

needs = pytest.mark.skipif(not (DATA / "skill.json").exists() or not (REPORTS / "skill" / "bootstrap.json").exists(),
                           reason="run src.research.export_dashboard first")


def load(name):
    return json.loads((DATA / name).read_text())


@needs
def test_skill_matches_bootstrap_report():
    boot = {(b["h"], b["model"]): b for b in json.loads((REPORTS / "skill" / "bootstrap.json").read_text())
            if b["stat"] == "skill_vs_persistence"}
    for row in load("skill.json")["skill"]:
        b = boot[(row["h"], row["arm"])]
        assert row["skill"] == pytest.approx(b["estimate"], abs=6e-4)
        assert row["lo"] == pytest.approx(b["ci"][0], abs=6e-4) and row["hi"] == pytest.approx(b["ci"][1], abs=6e-4)


@needs
def test_gains_match_diebold_mariano_report():
    dm = pd.read_csv(REPORTS / "skill" / "diebold_mariano.csv").set_index(["h", "a", "b"])
    for row in load("skill.json")["gains"]:
        ref = dm.loc[(row["h"], row["a"], row["b"])]
        assert row["mse_reduction"] == pytest.approx(ref["mse_reduction"], abs=6e-5)
        assert row["campaigns_b_better"] == ref["campaigns_b_better"]


@needs
def test_coverage_matches_calibration_report():
    cov = pd.read_csv(REPORTS / "calibration" / "coverage.csv").set_index(["h", "method", "nominal", "group_var", "group"])
    for row in load("calibration.json")["coverage"]:
        ref = cov.loc[(row["h"], row["method"], row["nominal"], row["group_var"], row["group"])]
        assert row["coverage"] == pytest.approx(ref["coverage"], abs=6e-5)
        assert row["n"] == ref["n"]


@needs
def test_value_curves_and_triggers_match_decision_report():
    vc = pd.read_csv(REPORTS / "decision" / "value_curves.csv")
    vc = vc[vc["probability"] == "rolling isotonic"].set_index(["h", "event_id", "arm", "alpha"])
    payload = load("decision.json")
    for row in payload["value"]:
        assert row["value"] == pytest.approx(vc.loc[(row["h"], row["event_id"], row["arm"], row["alpha"]), "value"], abs=6e-5)
    # the dashboard's trigger table must agree with the report's trigger table where they overlap
    trig = pd.read_csv(REPORTS / "decision" / "triggers.csv")
    trig = trig[(trig["probability"] == "rolling isotonic") & (trig["arm"] == "M2 + ENSO/IOD")].set_index(["h", "event_id", "trigger"])
    checked = 0
    for row in payload["triggers"]:
        key = (row["h"], row["event"], round(row["trigger"], 2))
        if key in trig.index:
            assert row["hit_rate"] == pytest.approx(trig.loc[key, "hit_rate"], abs=6e-5)
            assert row["false_alarm_ratio"] == pytest.approx(trig.loc[key, "false_alarm_ratio"], abs=6e-5)
            checked += 1
    assert checked >= 9


CATALOG = ROOT / "data" / "processed" / "station_catalog.parquet"


@needs
@pytest.mark.skipif(not CATALOG.exists(), reason="run src.research.ingest first (fresh clone)")
def test_summary_counts_match_processed_data():
    s = load("summary.json")
    cat = pd.read_parquet(CATALOG)
    assert s["n_stations"] == len(cat) and s["n_obs"] == int(cat["n_obs"].sum())
    assert all(k in s["sources"] for k in ("skill", "calibration", "decision"))


@needs
def test_every_dashboard_file_is_strict_json():
    """Browsers reject NaN/Infinity in JSON; a single one blanks the map (found in the leakage audit)."""
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    for path in [*DATA.glob("*.json"), *(DATA / "wells").glob("*.json"), *(DATA / "models").glob("*.json")]:
        json.loads(path.read_text(), parse_constant=reject)


MODELS_JSON = DATA / "models.json"
needs_models = pytest.mark.skipif(not MODELS_JSON.exists(), reason="run src.research.model_showcase first")


@needs_models
def test_model_skills_match_the_extension_report():
    """Each model's main-test skill on the dashboard equals reports/extensions/metrics.csv (computed separately)."""
    m = json.loads(MODELS_JSON.read_text())
    column = {r["key"]: r["column"] for r in m["models"]}
    ext = pd.read_csv(REPORTS / "extensions" / "metrics.csv").set_index(["h", "model"])["skill_vs_persistence"]
    checked = 0
    for s in m["scores"]:
        key = (s["h"], column[s["model"]])
        if s["setting"] == "main" and key in ext.index:
            assert s["skill"] == pytest.approx(ext[key], abs=1e-9), key
            checked += 1
    assert checked == 24  # 8 models also scored in reports/extensions, 3 horizons


@needs_models
def test_combination_matches_the_leakage_audit():
    m = json.loads(MODELS_JSON.read_text())
    cb = pd.read_csv(REPORTS / "audit" / "combination_bootstrap.csv")
    cb = cb[cb["model"] == "calibration-selected"]
    sel = pd.read_csv(REPORTS / "audit" / "combination_selection.csv")
    column = {r["key"]: r["column"] for r in m["models"]}
    for h in (1, 2, 4):
        skill = next(s["skill"] for s in m["scores"] if s["h"] == h and s["setting"] == "main" and s["model"] == "combo")
        ref = cb[(cb["h"] == h) & (cb["reference"] == "persistence")]["mse_reduction"].iloc[0]
        assert skill == pytest.approx(ref, abs=1e-9)
        pair = next(p for p in m["pairs"] if p["h"] == h and p["a"] == "m2" and p["b"] == "combo")
        row = cb[(cb["h"] == h) & (cb["reference"] == "M2 + ENSO/IOD")].iloc[0]
        assert pair["mse_reduction"] == pytest.approx(row["mse_reduction"], abs=1e-9)
        assert [pair["lo"], pair["hi"]] == pytest.approx(json.loads(row["ci95_wells"]), abs=1e-9)
        chosen = sel[(sel["h"] == h) & (sel["period"] == "calibration 2016–2018") & (sel["rank_in_period"] == 1)]
        assert [column[k] for k in m["combination"][str(h)]] == chosen["combination"].iloc[0].split(" | ")


@needs_models
def test_showcase_never_shows_post_hoc_combinations():
    m = json.loads(MODELS_JSON.read_text())
    columns = {r["column"] for r in m["models"]}
    assert not columns & {"Mean of M2, LSTM and Chronos-2", "Mean of M2 and Chronos-2", "Mean of M2, Chronos-2 and RF"}


@needs_models
def test_robustness_skills_match_the_extension_report():
    """Same rows and skills as reports/extensions/robustness.csv for the models refitted there."""
    m = json.loads(MODELS_JSON.read_text())
    ext = pd.read_csv(REPORTS / "extensions" / "robustness.csv")
    setting = {"split": "Split: train ≤2017, test 2020–25", "blocks": "Spatial blocks (unseen regions)"}
    column = {r["key"]: r["column"] for r in m["models"]}
    checked = 0
    for s in m["scores"]:
        if s["setting"] == "main" or s["model"] not in ("m2", "lstm", "cfsv2", "m3"):
            continue
        e = ext[(ext["h"] == s["h"]) & (ext["setting"] == setting[s["setting"]])]
        assert s["rows"] == e["n"].iloc[0]
        ref = e["skill_a"].iloc[0] if s["model"] == "m2" else e[e["b"] == column[s["model"]]]["skill_b"].iloc[0]
        assert s["skill"] == pytest.approx(ref, abs=1e-9)
        checked += 1
    assert checked == 24
