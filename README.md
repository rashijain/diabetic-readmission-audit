# Diabetic Readmission: Informative Missingness, Patient-Level Leakage, and Subgroup Equity

PhDAI 733 team project — Deliverable 2.
**Authors:** Sai Teja Bandaru, Rashi Jain

Predicting 30-day readmission in diabetic inpatients, and auditing three methodological
assumptions that the literature usually takes for granted.

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

## Reproducing

```bash
pip install -r requirements.txt
python -m src.data_loader      # downloads + validates the dataset
python -m src.evaluation       # runs all experiments -> outputs/
python -m src.visualization    # regenerates figures -> outputs/
```

Or open `notebooks/main_experiments.ipynb` in Google Colab and run all cells (~10–15 min).

All parameters live in `config.yaml`. Random seed is fixed at 42 throughout; every table
and figure in the report regenerates from committed code with no manual editing.

## Layout

```
config.yaml                 experiment parameters, split by owner
src/data_loader.py          ingestion, cohort construction, feature engineering
src/preprocessing.py        the three missingness regimes as leakage-safe Pipelines
src/evaluation.py           leakage ablation, regime comparison, tuning, subgroup audit
src/visualization.py        figure generation
notebooks/                  Colab runner
outputs/                    generated tables (.csv) and figures (.png)
```

## Division of work

- **Rashi Jain** — clinical framing, cohort exclusion rules, ICD-9 grouping, interpretation
  of the missingness diagnostics, deployment constraints.
- **Sai Teja Bandaru** — ingestion and parsing, leakage-safe pipeline architecture,
  validation design and ablation, model comparison and tuning, repository setup.
- **Joint** — research questions, subgroup audit design, written report.

## Notes on the data

Missing values are encoded three different ways: `'?'` in administrative fields, `'None'`
in `A1Cresult` / `max_glu_serum` (test not ordered), `'Unknown/Invalid'` in `gender`.
Parsing without declaring all three leaves ~97% of `weight` populated with the literal
string `'?'`, which any encoder treats as a valid category.
