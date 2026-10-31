# Leakage audit

This audit checks whether any forecast in this project could have seen information it would not have at the
time of forecasting: from the future (time), from nearby wells (space), or from choices made after looking at
the test years (selection). Every number below comes from `python -m src.research.leakage_audit <step>`; the
tables are in [`reports/audit/`](reports/audit/). Skill means the reduction in mean squared error (MSE)
relative to persistence (forecasting that the level stays where it is), on the test years 2019–2025 unless
stated otherwise.

**Summary.** No time leakage was found in the main evaluation. Three problems were found:

1. The ingestion outlier filter used each well's whole record, test years included. Refitting with a filter
   that uses only training years gives equal or slightly higher skill, so it did not inflate the results. It is
   documented here and left in place.
2. The LSTM's rainfall input was standardised with the mean and spread of all years. This is fixed (training
   years only) and the LSTM was retrained. Its 6-month skill moved from 0.589 to 0.587.
3. The model combination quoted as the headline (mean of XGBoost, LSTM and Chronos-2) was chosen after seeing
   test results. Choosing the combination on the calibration years instead picks a different combination with
   a lower test skill. The headline in [RESULTS.md](RESULTS.md) now uses that calibration-selected combination,
   and the claim that the combination beats the best single model by about 6 % is withdrawn.

## 1. What was checked

### Splits and horizons

| | |
|---|---|
| Unit | one row per (well, origin campaign); manual wells are read four times a year (Jan, May, Aug, Nov) |
| Horizons | h = 1, 2, 4 campaigns (about 3, 6, 12 months) |
| Split | by the **year of the forecast target**, the same for every well: train ≤ 2015, calibration 2016–2018, test 2019–2025 |
| Tuning | inner split inside the training years: fit ≤ 2012, early-stop on 2013–2015, refit on all training years |
| Robustness | alternative split (train ≤ 2017, test 2020–2025); five-fold 2° × 2° spatial blocks with the time split |

Train and test share wells (it is a panel), but never target years.

| Horizon | Train rows / wells | Cal rows | Test rows (all / common evaluation set) | Last train target | First test target | Gap |
|---|---|---|---|---|---|---|
| 3 months | 148,459 / 4,282 | 27,477 | 32,869 / 25,340 | Nov 2015 | Jan 2019 | 13 campaigns |
| 6 months | 143,751 / 4,259 | 27,086 | 31,548 / 24,073 | Nov 2015 | Jan 2019 | 13 campaigns |
| 12 months | 136,640 / 4,047 | 26,603 | 33,091 / 32,472 | Nov 2015 | Jan 2019 | 13 campaigns |

**Embargo.** The longest look-back of any feature is four campaigns (lag-4 depth, change over a year,
12-month rain windows, SPI-12). The required gap of horizon + longest look-back is 5, 6 and 8 campaigns, and the
actual gap is 13 (`reports/audit/inventory.csv`). No embargo had to be added. A test (`test_leakage.py`) fails if
the gap shrinks below the requirement.

### Features

| Feature | Computed from | Leak risk | Status |
|---|---|---|---|
| Current depth, 4 lags, 1- and 4-campaign changes | readings at or before the origin (`shift`) | none | perturbation tests: changing later readings leaves them unchanged |
| Seasonal climatology, anomaly | expanding per-well mean of readings up to and including the origin (≥ 2 values) | none | tested |
| Record length | cumulative count up to the origin | none | tested |
| Latitude, longitude, borehole depth | static station metadata | none | — |
| Rain windows, last monsoon, SPI, temperature anomaly | gridded months that end the month **before** the campaign month | none | tested; SPI gamma fits and monthly normals use years ≤ 2015 only |
| ENSO / IOD indices | month before the origin minus 0, 3, 6 months | none | tested |
| Oracle rain (M3 only) | rain between origin and target | **by design** | used only by the "perfect-rainfall" reference model M3 and Chronos-2 "future rain"; never by any other model |
| CFSv2 seasonal forecast | forecast issued the month before the campaign month; bias correction on issues ≤ 2015 | none | tested |
| Neighbour features | other wells read in the same origin campaign | none | tested |
| No centred windows | — | — | none exist: all windows are trailing (`window_sum` ends at the origin; snapshot rain uses trailing `rolling`) |
| Gap filling | missing campaigns stay missing (NaN); no interpolation | none | — |

**Scalers and normalisers.** The LSTM's tabular inputs were standardised on training rows only. Its rainfall
sequence channel used the mean and spread of all years; this is now fitted on years ≤ 2015 (`rain_scaler`, tested).
XGBoost and Random Forest need no scaling.

**Selection.** Hyper-parameters and early stopping use only training years. Conformal intervals are calibrated
on 2016–2018. The online variant adds a test residual only after its target has been observed (target ≤ T − h),
which is tested by perturbing later outcomes. Rolling probability recalibration uses only outcomes observed at
issue time (tested). Warning thresholds use readings up to 2018. Chronos-2 sees each well's readings up to the
origin only. Equal-weight combinations fit no weights, but **which models to combine was chosen after seeing the
test results**; see section 4.

## 2. Problems found

### 2.1 Outlier filter uses each well's whole record

`ingest.remove_outliers` drops readings more than 8 robust standard deviations from the well's median, with the
median and MAD computed over the whole record, test years included. That decides which test readings exist. Two
leak-free alternatives were refitted with each model's tuned settings (`reports/audit/qc_sensitivity.csv`):

| Outlier filter | Test rows (3 / 6 / 12 m) | M0 skill | M2 skill | M2 vs M0 (MSE) |
|---|---|---|---|---|
| Current: whole-record median/MAD | 25,340 / 24,073 / 32,472 | 0.536 / 0.549 / 0.201 | 0.541 / 0.572 / 0.212 | 1.1 / 5.0 / 1.3 % |
| Median/MAD from readings ≤ 2015 only | 25,103 / 23,827 / 32,247 | 0.545 / 0.555 / 0.199 | 0.551 / 0.579 / 0.211 | 1.5 / 5.5 / 1.5 % |
| Physical range only (no outlier filter) | 25,550 / 24,244 / 32,678 | 0.476 / 0.499 / 0.183 | 0.482 / 0.508 / 0.195 | 1.1 / 1.7 / 1.4 % |

The leak-free version scores as high or higher, so the headline numbers are not inflated and the pipeline was
not rerun with it. **Inconvenient result:** with no outlier filter at all, about 870 extra readings (many of
them datum jumps of tens of metres) cut skill by about 0.06 and the 6-month gain from rainfall and climate
indices falls from 5.0 % to 1.7 % of MSE. Squared-error results depend on how gross errors are handled; the
mean-absolute-error results and campaign sign tests in `reports/skill/` are less sensitive.

The station's sign convention (`harmonise_depth`) also uses the whole record. It affects 14 of 5,516 manual
wells (majority-negative readings) and decides only the sign, not which readings are kept.

### 2.2 LSTM rainfall channel scaled with all years

Fixed in `sequence_baseline.rain_scaler`; the LSTM was retrained on the main split and in both robustness
settings. LSTM + forcing skill: 0.529 / 0.589 / 0.221 before → 0.529 / 0.587 / 0.221 after.

### 2.3 Dashboard data (not leakage)

Serving the production dashboard showed that `stations.json` contained `NaN`, which browsers reject; the map
never loaded. Fixed in the export, with a strict-JSON test.

## 3. Spatial check

Wells are grouped into 2° × 2° blocks (about 220 km) and the blocks into five folds. Each well belongs to
exactly one fold, and **no test well appears in its fold's training data** (0 shared wells in all 15
horizon × fold combinations; `reports/audit/spatial_distances.csv`). Distance from each test well to the
nearest training well: minimum 3.3 km, 10th percentile 19–23 km, median 53–62 km.

M2 on identical test rows (`reports/audit/spatial_settings.csv`):

| Training data | 3 months | 6 months | 12 months |
|---|---|---|---|
| Time only: all wells, targets ≤ 2015 (main result) | 0.541 | 0.572 | 0.212 |
| Region only: other blocks, **all years including 2019–2025** | 0.577 | 0.599 | 0.275 |
| Time + region: other blocks, targets ≤ 2015 | 0.536 | 0.550 | 0.183 |

Holding out regions as well as years costs 0.005, 0.022 and 0.029 of skill; that is the honest number for
wells in regions the model has never seen. Holding out regions but not years scores *higher* than the main
result, because the model learns the 2019–2025 weather from other wells: a spatial-only split would overstate
skill. The cost of the regional hold-out does not depend strongly on distance to the nearest training well,
except at 12 months: for test wells more than 50 km from any training well, skill falls from 0.207 (time only)
to 0.157 (time + region) (`spatial_by_distance.csv`).

## 4. Selection: the model combination

The combinations were tried after the individual models' test results were known. To see what an honest
procedure would have chosen, every model was also run on the calibration years 2016–2018 (all trained on
targets ≤ 2015), and every equal-weight mean of 1–4 candidates (XGBoost M2, Random Forest M2, LSTM + forcing,
Chronos-2 + past rain) was ranked there (`reports/audit/combination_selection.csv`):

| Horizon | Chosen on 2016–2018 | Its test skill | Post-hoc headline (M2 + LSTM + Chronos-2) test skill |
|---|---|---|---|
| 3 months | M2 + RF + LSTM | 0.534 | 0.541 |
| 6 months | M2 + RF + LSTM | 0.590 | 0.598 |
| 12 months | M2 + RF + Chronos-2 | 0.235 | 0.239 |

The post-hoc combination ranks 5th or lower on the calibration years at 3 and 6 months. Using the
calibration-selected combination (`reports/audit/combination_bootstrap.csv`, 95 % intervals from 1,000
well-resampled draws; campaign-resampled intervals are in the file):

| MSE reduction of the calibration-selected combination vs | 3 months | 6 months | 12 months |
|---|---|---|---|
| XGBoost M2 | 4.7 % [3.3, 6.1] | 4.2 % [2.9, 5.6] | 3.0 % [0.7, 5.1] |
| Best single model on the test rows (LSTM + forcing; LSTM memory at 12 m) | 0.9 % [−1.5, 3.7] | 0.8 % [−1.8, 4.0] | 1.3 % [−0.1, 2.5] |
| Perfect-rainfall model M3 | −1.3 % [−3.0, 0.3] | −2.0 % [−4.9, 0.8] | −7.3 % [−10.5, −4.2] |

So combining beats XGBoost reliably, but **its gain over the best single model is within noise**, and it is
statistically indistinguishable from, but slightly below, the perfect-rainfall model at 3 and 6 months (clearly
below at 12 months). The earlier "≈ 6 % better than the best single model" was measured against XGBoost, not
the best single model, and used the post-hoc combination.

## 5. Sanity experiments

Three seeds each (42, 43, 44); mean ± standard deviation (`reports/audit/sanity_summary.csv`).

| Experiment | 3 months | 6 months | 12 months | Expected if no leak |
|---|---|---|---|---|
| M2, real data (test skill) | 0.542 ± 0.002 | 0.569 ± 0.003 | 0.214 ± 0.003 | — |
| M2 trained on shuffled targets | −0.011 ± 0.009 | −0.033 ± 0.031 | −0.032 ± 0.025 | ≤ 0 ✔ |
| Readings shuffled in time within each well: M2 / per-well climatology | 0.432 / 0.425 | 0.448 / 0.423 | 0.437 / 0.437 | M2 ≈ climatology ✔ |
| Persistence / seasonal naive / climatology / anomaly persistence | 0 / 0.194 / 0.100 / 0.404 | 0 / 0.404 / 0.332 / 0.444 | 0 / 0 / −0.091 / 0 | below M2 ✔ |
| Train-period skill (in-sample), M2 | 0.730 | 0.756 | 0.454 | above test (expected for boosting) |

With time order destroyed, the model cannot beat each well's average, so it is not exploiting any feature that
encodes the target. Leave-one-feature-group-out (change in M2 test skill):

| Dropped group | 3 months | 6 months | 12 months |
|---|---|---|---|
| Groundwater memory | −0.128 | −0.131 | −0.217 |
| Season and climatology | −0.031 | −0.026 | +0.002 |
| Location and borehole depth | −0.001 | −0.006 | −0.012 |
| Past rainfall and temperature | −0.003 | −0.005 | −0.006 |
| ENSO / IOD | −0.001 | −0.008 | −0.006 |

No single group gives an implausible lift; the model rests on groundwater memory, as it should.

## 6. Tests that guard against leakage

`tests/research/test_leakage.py` (new) and earlier tests fail if leakage is reintroduced:

- train, calibration and test targets are ordered and separated by at least horizon + look-back (synthetic and
  real data);
- the LSTM's tabular scaler and rain scaler ignore values outside the training rows / years;
- LSTM sequence inputs at an origin are unchanged when every later reading and rain value is changed;
- spatial folds never share a well (synthetic and real data);
- existing: tabular features, forcing features, CFSv2 features, online conformal pools and rolling recalibration
  are unchanged by perturbing anything after the origin; an end-to-end test truncates the real data at three
  cut-offs and requires identical features.

## 7. Remaining limitations

- The outlier filter still uses each well's whole record (section 2.1). Switching to the training-period filter
  would not lower any headline number but would change every table slightly; it has not been done.
- Design choices other than the combination were also made with the test years visible: probability
  recalibration was added after the raw reliability diagram was seen, two interval variants are marked as post
  hoc, and the warning triggers 0.2 and 0.3 are illustrative. None changes the main conclusions, but none is a
  clean out-of-sample test.
- Chronos-2 is a pretrained foundation model; whether its pretraining corpus included this public CGWB
  compilation is unknown.
- Test years are thin after 2023 because most manual records end in 2022–2023.
- The 2° blocks are a coarse spatial hold-out; the nearest training well can be 3 km away.
- Results come from one public compilation of CGWB data whose compiler and query are not recorded.
