# Diabetic Readmission: Informative Missingness, Patient-Level Leakage, and Subgroup Equity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rashijain/diabetic-readmission-audit/blob/main/notebooks/ResidencyProject.ipynb)

PhDAI 733 team project — Deliverables 2 and 3.
**Authors:** Sai Teja Bandaru, Rashi Jain

Predicting 30-day readmission in diabetic inpatients: auditing three methodological
assumptions the literature usually takes for granted (Deliverable 2), then optimizing the
model and testing it once on held-out patients (Deliverable 3).

## Dataset

Diabetes 130-US Hospitals (Strack et al., 2014) — 101,766 encounters, 130 US hospitals,
1999–2008, 50 variables, 71,518 distinct patients.

Not committed to this repo. `src/data_loader.py` downloads it on first run and asserts
the expected shape before proceeding.

## Headline results

| Research question | Result |
|---|---|
| RQ1 — does encounter-level CV inflate scores? | **No.** Max inflation 0.0009 ROC-AUC against a fold SD of 0.004–0.007. |
| RQ2 — drop, impute, or encode missingness? | **Encode.** Monotone drop < impute < explicit across both tree models and both metrics. |
| RQ3 — is the model equitable? | **No.** FNR 0.798 for patients with no recorded race vs 0.519 for Hispanic patients. |

Final model: histogram gradient boosting, Regime C.
ROC-AUC 0.6759 · PR-AUC 0.2352 · Brier 0.0957 · ECE 0.0038 (5-fold patient-grouped).

Most actionable finding: preprocessing was worth +0.0054 PR-AUC, hyperparameter tuning
across 12 configurations +0.0008. Representation beats model search roughly sevenfold here.

## Deliverable 3 — optimization and held-out evaluation

Patients, not encounters, are split 80/20 before anything else. Searching, calibration
experiments and the choice of operating threshold all happen on the development side;
the held-out set is scored once.

| Question | Result |
|---|---|
| Does tuning help? | **Marginally.** 60 sampled configurations moved PR-AUC from 0.2281 to 0.2322, under half the fold SD of 0.0090. |
| Then why tune? | **Downside protection.** The 60 candidates spanned 0.1900–0.2322 — an 18% relative gap, almost all of it learning rate. |
| Does it hold out? | **Yes.** ROC-AUC 0.6750, PR-AUC 0.2395, ECE 0.0059 on 19,802 unseen encounters. |
| Is it worth deploying? | **As a ranking aid.** At the top-15% cut-off: precision 0.2494, 2.23× lift, ~4 contacts per readmission caught. |
| Is it equitable? | **No, and it replicates.** Patients with no recorded race are flagged at 5.98% and miss 89.5% of their true readmissions. |

The cut-off is fixed at the 85th percentile of development out-of-fold predictions
(p = 0.1701) and applied unchanged to the test set, where it flags 14.9% — so the
threshold transfers. At that point the model is 81.3% accurate, *below* the 88.8% you get
by flagging nobody, which is the clearest argument in the project for why average
precision rather than accuracy was the objective throughout.

Isotonic recalibration was tried and rejected: it cost 0.02 PR-AUC to buy 0.002 ECE.

## Reproducing

```bash
pip install -r requirements.txt
python -m src.data_loader      # downloads + validates the dataset
python -m src.evaluation       # Deliverable 2 experiments  -> outputs/
python -m src.visualization    # Deliverable 2 figures      -> outputs/
python -m src.optimization     # Deliverable 3 search + held-out evaluation
python -m src.d3_figures       # Deliverable 3 figures
```

Or open `notebooks/ResidencyProject.ipynb` in Google Colab and run all cells.
Deliverable 2 takes ~10–15 minutes; the Deliverable 3 search adds ~30 minutes, since it
fits 60 configurations across 3 folds.

All parameters live in `config.yaml`. Random seed is fixed at 42 throughout; every table
and figure in the report regenerates from committed code with no manual editing.

## Layout

```
config.yaml                 experiment parameters (clinical and modelling blocks)
src/data_loader.py          ingestion, cohort construction, feature engineering
src/preprocessing.py        the three missingness regimes as leakage-safe Pipelines
src/evaluation.py           leakage ablation, regime comparison, tuning, subgroup audit
src/visualization.py        Deliverable 2 figures
src/optimization.py         dev/test split, searches, held-out evaluation, operating point
src/d3_figures.py           Deliverable 3 figures
notebooks/                  Colab runner (ResidencyProject.ipynb)
outputs/                    generated tables (.csv) and figures (.png)
```

## Notes on the data

Missing values are encoded three different ways: `'?'` in administrative fields, `'None'`
in `A1Cresult` / `max_glu_serum` (test not ordered), `'Unknown/Invalid'` in `gender`.
Parsing without declaring all three leaves ~97% of `weight` populated with the literal
string `'?'`, which any encoder treats as a valid category.
