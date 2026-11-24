# Verified results

Every number here was recomputed from code on 2026-10-30/31 and is read from the file named next to it. The
[leakage audit](LEAKAGE_AUDIT.md) explains how the evaluation was checked and what changed.

**Metric.** *Skill* = 1 − MSE(model) / MSE(persistence), where persistence forecasts that the water level stays
where it is; 0 means no better than persistence, 1 is perfect. *Gain* = 1 − MSE(model) / MSE(reference).
Brackets are 95 % intervals from a bootstrap that resamples whole wells (1,000 draws unless stated).

**Test set.** Forecast targets in 2019–2025 (models trained on targets up to 2015, calibrated on 2016–2018). The
common evaluation set has 25,340 / 24,073 / 32,472 forecasts from 3,249 / 3,175 / 3,370 wells at the 3 / 6 / 12-
month horizons, over 24 / 24 / 25 target campaigns (`reports/skill/metrics.csv`, `reports/audit/facts.csv`).

## Headline

| Claim | Verified number | Test set | Source |
|---|---|---|---|
| Data | 414,106 quality-controlled readings from 6,438 CGWB stations, January 1994 – September 2025. Forecasts use the 5,516 manually read wells (4,734–4,989 have at least one forecast row); the 922 telemetry wells are not used for forecasting | — | `reports/audit/facts.csv` |
| Memory-only XGBoost (M0) beats persistence | skill 0.536 [0.503, 0.563] / 0.549 [0.506, 0.586] / 0.201 [0.166, 0.236] | common set | `reports/skill/bootstrap.json` |
| Monsoon rainfall, temperature and ENSO/IOD (M2) add to memory (M0) | MSE gain 1.1 % [−0.4, 2.7] / **5.0 % [3.3, 6.8]** / 1.3 % [−0.1, 2.9]; MAE gain 2.0 / 3.6 / 1.5 % (all intervals above 0); better in 16/24, **22/24**, 17/25 campaigns | common set | `reports/skill/diebold_mariano.csv`, `bootstrap.json` |
| A perfect rainfall forecast (M3, uses future rain; reference only) would add more | MSE gain over M2 5.5 % [4.0, 7.0] / 6.1 % [2.8, 8.7] / 9.6 % [7.7, 11.4] | common set | `reports/skill/bootstrap.json` |
| Real seasonal forecasts (NCEP CFSv2) capture none of that | MSE gain over M2 +2.5 / −0.8 / −0.5 %, better in 14/24, 10/24, 10/25 campaigns | 31,468 / 30,424 / 32,472 | `reports/extensions/comparisons.csv` |
| **Best verified forecast: equal-weight combination chosen on the calibration years** (XGBoost M2 + Random Forest + LSTM; at 12 months XGBoost M2 + Random Forest + Chronos-2) | skill **0.534 / 0.590 / 0.235**; MSE gain over XGBoost M2 **4.7 % [3.3, 6.1] / 4.2 % [2.9, 5.6] / 3.0 % [0.7, 5.1]** | 31,468 / 30,424 / 32,472 | `reports/audit/combination_bootstrap.csv` |
| …against the best single model (the LSTM) | MSE gain 0.9 % [−1.5, 3.7] / 0.8 % [−1.8, 4.0] / 1.3 % [−0.1, 2.5]: **within noise** | same | same |
| …against the perfect-rainfall model M3 | MSE gain −1.3 % [−3.0, 0.3] / −2.0 % [−4.9, 0.8] / −7.3 % [−10.5, −4.2]: not distinguishable at 3 and 6 months, worse at 12 | same | same |
| Combination chosen on the calibration years, in the robustness settings (Random Forest refitted in each) | Later test years (train ≤ 2017, test 2020–2025): skill 0.497 [0.461, 0.527] / 0.530 [0.488, 0.567] / 0.225 [0.170, 0.277], against XGBoost M2 0.479 / 0.512 / 0.210 and the LSTM 0.487 / 0.518 / 0.210. Unseen 2° regions: 0.525 [0.495, 0.551] / 0.576 [0.541, 0.605] / 0.219 [0.175, 0.258], against 0.507 / 0.554 / 0.162 and 0.517 / 0.569 / 0.200. Highest in every cell, but the intervals overlap with the LSTM's; not tested pairwise | 22,444 / 21,609 / 23,651 and 31,468 / 30,424 / 32,472 | `reports/showcase/robustness.csv` |
| Forecasting regions the model has never seen (2° blocks held out as well as years) | M2 skill 0.536 / 0.550 / 0.183, against 0.541 / 0.572 / 0.212 with years held out only | common set | `reports/audit/spatial_settings.csv` |
| One interval width fails for deep wells | 80 % split-conformal interval covers 88 % of readings in wells shallower than 5 m (18,323 readings) but 46 % in wells 40 m or deeper (539) at 3 months | all test rows | `reports/calibration/coverage.csv` |
| Intervals pooled by depth × region and updated online (adaptive conformal inference) hold for every depth and region class | largest miss from the 80 % and 90 % targets over 4 depth and 4 region classes: **1.6 / 1.7 / 3.0 percentage points** (the 3.0 is the Himalaya and north-east at 12 months, 83.0 % at 80 %) | 32,869 / 31,548 / 33,091 | `reports/calibration/coverage.csv` |
| Warnings have economic value, **estimated under assumed cost–loss ratios** (no costs were measured) | "Deeper than usual" at 3 months: relative value 0.37 [0.34, 0.39] for an assumed ratio of 0.3, peak 0.43 over all ratios, positive with intervals above 0 for ratios 0.04–0.92. Pump-suction (8 m) event at 6 months: 0.69 at ratio 0.1, positive for 0.02–0.70 | 27,140 forecasts (E1, 3 months) | `reports/audit/value_sensitivity.csv`, `reports/decision/` |

## What changed from earlier claims

| Earlier claim | Status |
|---|---|
| "The combination cuts error about 6 % below the best single model" | **Withdrawn.** The 6 % (6.1 / 6.2 / 3.5 %) was the post-hoc mean of XGBoost, LSTM and Chronos-2 against XGBoost M2, not against the best single model, and that combination was chosen after seeing the test years. The calibration-selected combination gains 4.2–4.7 % over XGBoost at 3–6 months and is within noise of the LSTM. |
| "The combination matches a perfect-rainfall model at 3 and 6 months" | **Weakened.** The calibration-selected combination is 1–2 % worse than the perfect-rainfall model at 3 and 6 months; the difference is not statistically significant. It is clearly worse at 12 months. |
| "The LSTM with forcing beats XGBoost significantly at 6 months" | **Withdrawn** after the LSTM was retrained with a training-only rain scaler: 3.5 % MSE gain, better in 15 of 24 campaigns (p = 0.31). The memory-only LSTM still beats memory-only XGBoost (5.7 %, 21 of 24 campaigns). |
| "About 6,400 wells" | **Corrected.** 6,438 stations are in the data; 5,516 manual wells are forecast, and 3,175–3,370 wells are in the test set at each horizon. |
| "Economic value of up to 0.43 / 0.69" | **Relabelled.** These are peaks over an assumed cost–loss ratio, not measured savings; see the sensitivity table below. |
| "Within about 3 points of target coverage" | **Confirmed** (largest miss 3.02 points, one cell). |

## Economic value under assumed cost–loss ratios

No costs were measured. A user who pays C to avoid a loss L acts when the forecast probability p ≥ C/L. The table
gives the relative economic value (1 = perfect forecast, 0 = always or never acting) of warnings from the forcing
model M2 with rolling-isotonic probabilities, with 95 % intervals, at several assumed ratios
(`reports/audit/value_sensitivity.csv`, from `python -m src.research.leakage_audit value`). E1: depth beyond the
well's seasonal 80th percentile; E2: standardised groundwater index below −1; E3: water falling below 8 m in wells
at 8 m or shallower.

| Event, horizon | C/L = 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | C/L range with CI above 0 |
|---|---|---|---|---|---|---|---|---|
| E1, 3 months | 0.06 [0.04, 0.08] | 0.15 [0.13, 0.17] | 0.35 [0.33, 0.38] | 0.37 [0.34, 0.39] | 0.18 [0.15, 0.20] | 0.07 [0.05, 0.09] | 0.01 [0.00, 0.02] | 0.04–0.92 |
| E1, 6 months | 0.04 [0.02, 0.06] | 0.10 [0.08, 0.12] | 0.27 [0.25, 0.29] | 0.27 [0.24, 0.30] | 0.12 [0.10, 0.14] | 0.04 [0.02, 0.06] | 0.01 [−0.00, 0.01] | 0.06–0.84 |
| E1, 12 months | 0.01 [0.01, 0.02] | 0.04 [0.03, 0.05] | 0.21 [0.19, 0.22] | 0.17 [0.14, 0.19] | 0.11 [0.09, 0.13] | 0.05 [0.04, 0.07] | 0.01 [−0.00, 0.02] | 0.06–0.98 |
| E2, 3 months | 0.14 [0.11, 0.17] | 0.27 [0.23, 0.30] | 0.44 [0.41, 0.48] | 0.34 [0.31, 0.38] | 0.20 [0.16, 0.23] | 0.07 [0.03, 0.10] | 0.01 [−0.01, 0.03] | 0.04–0.84 |
| E2, 6 months | 0.10 [0.08, 0.13] | 0.19 [0.16, 0.22] | 0.37 [0.33, 0.40] | 0.26 [0.22, 0.29] | 0.13 [0.10, 0.16] | 0.06 [0.04, 0.09] | 0.01 [−0.02, 0.03] | 0.04–0.82 |
| E2, 12 months | 0.05 [0.03, 0.06] | 0.14 [0.12, 0.16] | 0.32 [0.28, 0.36] | 0.22 [0.19, 0.26] | 0.13 [0.09, 0.16] | 0.06 [0.03, 0.09] | 0.00 [−0.02, 0.02] | 0.06–0.86 |
| E3, 3 months | 0.60 [0.57, 0.62] | 0.65 [0.63, 0.67] | 0.50 [0.47, 0.52] | 0.40 [0.37, 0.43] | 0.19 [0.16, 0.22] | −0.01 [−0.04, 0.02] | −0.00 [−0.02, 0.01] | 0.02–0.64 |
| E3, 6 months | 0.61 [0.58, 0.63] | 0.69 [0.67, 0.71] | 0.58 [0.55, 0.61] | 0.49 [0.45, 0.52] | 0.29 [0.25, 0.32] | 0.06 [0.02, 0.09] | −0.01 [−0.03, 0.01] | 0.02–0.70 |
| E3, 12 months | 0.54 [0.51, 0.57] | 0.43 [0.39, 0.46] | 0.23 [0.20, 0.27] | 0.10 [0.06, 0.14] | −0.02 [−0.06, 0.01] | −0.01 [−0.03, 0.01] | 0.00 [0.00, 0.00] | 0.02–0.36 |

For E1 and E2 the value is positive, with intervals above zero, for almost any ratio a user might have, and for E3
for ratios up to about 0.6–0.7 at 3–6 months. The table does **not** support quoting one number as "the" value:
depending on the assumed ratio, 3-month E1 warnings are worth anything from close to zero up to 0.43 of a perfect
forecast.

## Limitations

The test years are thin after 2023 because most manual records end in 2022–2023. Squared-error results depend on
how gross reading errors are handled: with no outlier filter at all, skill falls by about 0.06 and the 6-month
rainfall gain from 5.0 % to 1.7 % ([LEAKAGE_AUDIT §2.1](LEAKAGE_AUDIT.md#21-outlier-filter-uses-each-wells-whole-record)).
Pumping is not observed. Probability recalibration and two interval variants were added after looking at test
results. All results come from one public compilation of CGWB data whose compiler and download query are not
recorded.

## Reproduce these numbers

```bash
bash scripts/reproduce.sh --forcing --phase2 --phase3 --phase4 --phase7 --audit
```
