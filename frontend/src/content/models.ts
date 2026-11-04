/**
 * Model descriptions for the Models section. Numbers shown next to them come from models.json
 * (src/research/model_showcase.py); the verdicts summarise RESULTS.md and must stay in step with it.
 */
import type { ModelFamily, ModelKey } from "@/lib/research-data";

export interface ModelInfo {
  name: string;
  short: string;
  reads: string;
  learns: string;
  good: string;
  limits: string;
  verdict: string;
}

export const FAMILY: Record<
  ModelFamily,
  { label: string; badge: string; color: string; help: string }
> = {
  baseline: {
    label: "Simple benchmarks",
    badge: "Simple benchmark",
    color: "var(--none)",
    help: "Rules a person could apply by hand. A model is only useful if it beats them.",
  },
  boosting: {
    label: "Gradient boosting",
    badge: "Main model",
    color: "var(--s1)",
    help: "XGBoost trees trained on every well at once.",
  },
  learner: {
    label: "Other learners",
    badge: "Comparison",
    color: "var(--s3)",
    help: "A random forest, a neural network and a pretrained foundation model, on the same rows.",
  },
  combination: {
    label: "Our best forecast",
    badge: "Best verified",
    color: "var(--s2)",
    help: "An equal-weight average of models, chosen on 2016–2018 before the test years were scored.",
  },
  reference: {
    label: "Reference only",
    badge: "Reference only",
    color: "var(--ink-3)",
    help: "Shown for comparison, not as forecasts you could issue.",
  },
};

export const MODEL_INFO: Record<ModelKey, ModelInfo> = {
  persistence: {
    name: "Persistence",
    short: "Persistence",
    reads: "The latest reading only.",
    learns: "Nothing: it says the next reading will equal this one.",
    good: "Hard to beat for water tables that change slowly; it is the zero of every skill score here.",
    limits: "Misses every seasonal rise and fall.",
    verdict: "The bar every model has to clear. Skill is the share of its error a model removes.",
  },
  climatology: {
    name: "Seasonal climatology",
    short: "Climatology",
    reads: "The well's own earlier readings for the target season.",
    learns: "Nothing: it forecasts the well's usual level for that season.",
    good: "Captures the yearly cycle with no model at all.",
    limits: "Ignores where the water is now, so it fails in unusual years and at 12 months.",
    verdict: "Better than persistence at 6 months, worse at 12.",
  },
  anomaly: {
    name: "Anomaly persistence",
    short: "Anomaly persistence",
    reads: "The well's usual level for the target season and how far it is from usual today.",
    learns: "Nothing: it assumes today's departure from usual carries on.",
    good: "Combines the seasonal cycle with the current state; the strongest hand rule.",
    limits: "Cannot learn how fast departures fade, or anything about rainfall.",
    verdict: "The strongest simple rule; every trained model beats it clearly.",
  },
  m0: {
    name: "XGBoost, water levels only (M0)",
    short: "XGBoost M0",
    reads:
      "Current depth, the last four readings, recent changes, season, the well's usual level, location and borehole depth.",
    learns:
      "Gradient-boosted decision trees, trained on all wells together to predict the change in depth.",
    good: "Fast, accurate on tables of features, and handles missing readings.",
    limits: "Knows nothing about rainfall or the climate.",
    verdict: "Groundwater memory alone already removes most of persistence's error.",
  },
  m2: {
    name: "XGBoost with rainfall and climate (M2)",
    short: "XGBoost M2",
    reads:
      "Everything M0 reads, plus past rainfall, drought indices, temperature and El Niño / Indian Ocean Dipole values.",
    learns: "The same gradient boosting, with the same settings search on 2013–2015.",
    good: "Shows which inputs matter, and is the model behind the ranges and warnings.",
    limits: "Only knows rain that has already fallen.",
    verdict:
      "Rainfall and climate improve on M0 at every horizon. It powers the ranges and warnings.",
  },
  rf: {
    name: "Random Forest",
    short: "Random Forest",
    reads: "The same inputs as XGBoost M2.",
    learns: "300 independent decision trees, averaged.",
    good: "Robust and needs little tuning; makes different mistakes from boosting, which helps when combined.",
    limits: "On its own, no better than XGBoost.",
    verdict: "About as good as XGBoost alone, and useful inside the combination.",
  },
  lstm_mem: {
    name: "LSTM, water levels only",
    short: "LSTM (levels only)",
    reads: "The well's last 16 readings as a sequence, plus M0's inputs.",
    learns: "A two-layer recurrent neural network; the average of three training runs.",
    good: "Reads the whole recent history, not just four lags.",
    limits: "Slower to train and harder to explain than trees.",
    verdict: "Clearly beats XGBoost when both see only water levels.",
  },
  lstm: {
    name: "LSTM with rainfall",
    short: "LSTM",
    reads: "The last 16 readings and the rain between them, plus M2's inputs.",
    learns: "The same recurrent network, early-stopped like the trees.",
    good: "The best single model at 3 and 6 months.",
    limits: "Its gain over XGBoost M2 is not significant once rainfall is included.",
    verdict: "Best single model at 3 and 6 months; the combination is not clearly better.",
  },
  chronos: {
    name: "Chronos-2 foundation model",
    short: "Chronos-2",
    reads: "Each well's past readings and the rain between them.",
    learns:
      "Nothing on our data: a model pretrained by Amazon on many other time series, used as it is.",
    good: "Works on a new well with no training at all.",
    limits: "Trails every trained model here.",
    verdict: "Beats the simple rules with zero training, but trails the trained models.",
  },
  combo: {
    name: "Combination chosen on 2016–2018",
    short: "Combination",
    reads: "The forecasts of the models it averages.",
    learns:
      "No weights: an equal average of the combination that scored best on the calibration years.",
    good: "The highest test skill of any real forecast, chosen without looking at the test years.",
    limits: "Needs every member model to run, and is not clearly better than the LSTM alone.",
    verdict: "Best real forecast: 3–5% less error than XGBoost M2, within noise of the LSTM.",
  },
  snapshot: {
    name: "XGBoost with the file's one-day rainfall",
    short: "One-day rainfall",
    reads: "M0's inputs plus the rainfall column of the public data file.",
    learns: "The same gradient boosting.",
    good: "Shows why an earlier study concluded that rainfall does not help.",
    limits: "That column is the rain on the day of each reading, a median of 3.6 mm a year.",
    verdict: "Adds nothing over water levels alone: the problem was the column, not the rainfall.",
  },
  cfsv2: {
    name: "XGBoost with a real seasonal forecast (CFSv2)",
    short: "M2 + CFSv2",
    reads: "M2's inputs plus the NCEP CFSv2 rainfall forecast issued the month before.",
    learns: "The same gradient boosting.",
    good: "The honest test of whether today's seasonal forecasts can be used.",
    limits: "At the scale of one well, the forecast rain barely matches the rain that falls.",
    verdict: "No reliable gain: tested, and it does not help.",
  },
  m3: {
    name: "XGBoost with the rain that actually fell (M3)",
    short: "Perfect rainfall",
    reads: "M2's inputs plus the rain that fell between the forecast and the target.",
    learns: "The same gradient boosting.",
    good: "Shows the most a perfect seasonal rainfall forecast could add.",
    limits: "Not a real forecast: that rain is unknown when the forecast is made.",
    verdict:
      "The ceiling. The best real forecast comes within 1–2% of it at 3 and 6 months, not at 12.",
  },
};

export const CARD_ORDER: ModelKey[] = [
  "combo",
  "lstm",
  "m2",
  "rf",
  "lstm_mem",
  "m0",
  "chronos",
  "anomaly",
  "climatology",
  "persistence",
  "cfsv2",
  "snapshot",
  "m3",
];
