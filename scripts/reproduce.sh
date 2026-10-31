#!/usr/bin/env bash
# Rebuilds every research artefact from the raw CGWB file and checks it.
#   bash scripts/reproduce.sh                    # tests → ingest → benchmark
#   bash scripts/reproduce.sh --forcing          # also download/refresh forcing data and features (Phase 1)
#   bash scripts/reproduce.sh --forcing --phase2 # also the skill decomposition, reference models,
#                                                #   robustness and figures (Phase 2, ~45 min, GPU for Chronos-2)
#   … --phase3                                   # also uncertainty calibration and figures (Phase 3, needs Phase 2)
#   … --phase4                                   # also decision value and figures (Phase 4, needs Phases 2–3)
#   … --phase7                                   # also the model extensions: CFSv2 forecasts, LSTM, combination
#                                                #   (Phase 7, needs Phase 2; GPU recommended, ~1 h)
#   … --audit                                    # also the leakage audit (LEAKAGE_AUDIT.md; needs Phases 2–4 and 7)
# The conda environment is "anpr" unless GW_CONDA_ENV names another one.
set -euo pipefail
cd "$(dirname "$0")/.."

FORCING=0
PHASE2=0
PHASE3=0
PHASE4=0
PHASE7=0
AUDIT=0
for arg in "$@"; do
  case "$arg" in
    --forcing) FORCING=1 ;;
    --phase2) PHASE2=1 ;;
    --phase3) PHASE3=1 ;;
    --phase4) PHASE4=1 ;;
    --phase7) PHASE7=1 ;;
    --audit) AUDIT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GW_CONDA_ENV:-anpr}"
# Third-party pytest plugins on PYTHONPATH (e.g. ROS) must not be auto-loaded into the research tests.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python -m pytest tests/research

# Ingest first: forcing features need the station catalog it writes.
python -m src.research.ingest

if [[ $FORCING == 1 ]]; then
  python -m src.research.forcing_download
  python -m src.research.forcing_features
fi

python -m src.research.benchmark

python - <<'EOF'
import json
m = json.load(open("reports/benchmark/benchmark_metrics.json"))
rmse = m["horizons"]["1"]["metrics"]["XGBoost (memory-only)"]["rmse"]
print(f"Check: XGBoost memory-only RMSE at h=1 = {rmse:.3f} m (pilot reference 2.406 m)")
assert abs(rmse - 2.406) < 0.0005, "pilot benchmark changed — investigate before continuing"
EOF

if [[ $PHASE2 == 1 ]]; then
  python -m src.research.skill_decomposition
  python -m src.research.reference_models
  python -m src.research.skill_robustness
  python -m src.research.skill_figures
  python - <<'EOF'
import pandas as pd
m = pd.read_csv("reports/skill/metrics.csv").set_index(["h", "model"])
skill = m.loc[(2, "M2 + ENSO/IOD"), "skill_vs_persistence"]
print(f"Check: M2 skill vs persistence at 6 months = {skill:.3f} (Phase 2 reference 0.572)")
assert abs(skill - 0.572) < 0.002, "Phase 2 result changed — investigate before continuing"
EOF
fi

if [[ $PHASE3 == 1 ]]; then
  python -m src.research.calibration
  python -m src.research.calibration_figures
  python - <<'PY'
import pandas as pd
c = pd.read_csv("reports/calibration/coverage.csv")
a = c[c["group_var"].isin(["Depth", "Region"]) & c["nominal"].isin([0.8, 0.9])
      & (c["method"] == "Mondrian D×R + recal + ACI")]
worst = (a["coverage"] - a["nominal"]).abs().max() * 100
print(f"Check: worst depth/region miss for D×R + recal + ACI = {worst:.2f} points (Phase 3 reference 3.02)")
assert abs(worst - 3.02) < 0.05, "Phase 3 result changed — investigate before continuing"
PY
fi

if [[ $PHASE4 == 1 ]]; then
  python -m src.research.decision_value
  python -m src.research.decision_figures
  python - <<'PY'
import pandas as pd
v = pd.read_csv("reports/decision/value_curves.csv")
r = v[(v["probability"] == "rolling isotonic") & (v["arm"] == "M2 + ENSO/IOD") & (v["event_id"] == "E1")
      & (v["h"] == 1) & (v["alpha"] == 0.3)].iloc[0]
print(f"Check: value of E1 warnings at 3 months, C/L = 0.3 = {r['value']:.3f} (Phase 4 reference 0.369)")
assert abs(r["value"] - 0.369) < 0.002, "Phase 4 result changed — investigate before continuing"
PY
fi

if [[ $PHASE7 == 1 ]]; then
  python -m src.research.seasonal_forecast
  python -m src.research.sequence_baseline
  python -m src.research.sequence_baseline --robustness
  python -m src.research.model_extensions --main
  GW_XGB_DEVICE=cuda python -m src.research.model_extensions --robustness
  python -m src.research.paper_figures
  python - <<'PY'
import pandas as pd
m = pd.read_csv("reports/extensions/metrics.csv").set_index(["h", "model"])
skill = m.loc[(2, "Mean of M2, LSTM and Chronos-2"), "skill_vs_persistence"]
print(f"Check: combination skill vs persistence at 6 months = {skill:.3f} (reference 0.598, post-hoc combination)")
assert abs(skill - 0.598) < 0.005, "Model-extension result changed — investigate before continuing"
PY
fi

if [[ $AUDIT == 1 ]]; then
  python -m src.research.leakage_audit inventory qc spatial sanity cal selection bootstrap value facts
fi

echo "Reproduction complete."
