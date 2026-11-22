# Review notes

Review of `ResidencyProject.ipynb` (commit `723a289`) after executing the pipeline
independently against the full 101,766-row dataset.

The notebook **runs clean end to end** — no errors, and its results reproduce. What
follows is one substantive correction, two robustness items, and a reproduction check.

## 1. `weight` is not an informative-missingness feature — correction

`config.yaml` lists `weight` among the MNAR features on the assumption that all five
high-missingness columns carry signal. Measured against the data, it does not:

| Feature | Missing | Readmit if measured | if missing | chi-square p |
|---|---|---|---|---|
| **weight** | **96.85%** | **11.39%** | **11.39%** | **1.000** |
| max_glu_serum | 94.81% | 12.75% | 11.31% | 1.7e-03 |
| A1Cresult | 83.05% | 9.94% | 11.68% | 8.7e-11 |
| medical_specialty | 48.94% | 10.96% | 11.83% | 1.6e-05 |
| payer_code | 39.66% | 11.19% | 11.69% | 1.6e-02 |

`weight` is the most incomplete column in the dataset and carries no signal whatsoever —
the readmission rate is identical to two decimal places either way.

It is therefore reframed as a **null control** rather than removed. Its null result is
precisely what shows the other four reflect clinical signal rather than an artefact of
high missingness. Reporting it as MNAR alongside A1Cresult would have been false.

Note also that the two laboratory fields point in **opposite** directions: for
`A1Cresult` the untested patients are at higher risk, for `max_glu_serum` the tested
patients are. No single "missingness as acuity" story covers both.

## 2. Robustness — neither of these broke the run

**Legacy UCI URL.** `archive.ics.uci.edu/ml/machine-learning-databases/00296/` predates
the UCI site restructure. It still resolved on this run, but it is the deprecated path.
Now: current static URL first, legacy as fallback, plus an assertion on the expected
(101766, 50) shape so a wrong source file fails loudly rather than silently.

**seaborn `palette` without `hue`.** One call —
`sns.barplot(data=mnar_df, y="Feature", x="Missing_%", palette="viridis", ...)` — uses a
pattern deprecated in seaborn 0.13 and removed in 0.14. Colab currently ships 0.13, so it
warns rather than fails. Fixed by passing `hue` explicitly with `legend=False`.

## 3. Reproduction check

Two independent executions, same conclusions:

| | Notebook run | Independent run |
|---|---|---|
| Cohort | 99,343 enc. / 69,990 patients / 11.39% | identical |
| Max leakage inflation (ROC-AUC) | 0.0002 | 0.0009 |
| HistGBDT by regime (drop / impute / explicit) | 0.6716 / 0.6735 / 0.6754 | 0.6721 / 0.6738 / 0.6759 |
| Post-hoc calibration ECE (none / Platt / isotonic) | 0.0035 / 0.0092 / 0.0059 | 0.0039 / 0.0086 / 0.0073 |

Both runs support the same three findings:

- **Patient-level leakage produces no measurable inflation.** Maximum difference 0.0002
  (notebook) and 0.0009 (independent), against a fold-to-fold SD of 0.004–0.007. The
  effect is far below the noise floor of the estimate. Patient-grouped splitting is
  correct in principle and retained, but it is not where this cohort's validity is decided.
- **Explicit missingness encoding wins**, monotonically across both tree models and both
  metrics, though the margin is small (+0.0038 ROC-AUC, +0.0054 PR-AUC).
- **Post-hoc calibration makes calibration worse.** The model is already well calibrated;
  Platt and isotonic both increase ECE. Not adopted.

Hyperparameter tuning was also run separately: twelve configurations under three-fold
patient-grouped CV returned +0.0008 PR-AUC against a fold SD of 0.0043 — also null.
Preprocessing was worth roughly seven times more than model configuration here.

## 4. Refactor

The notebook is reorganised into `src/` modules with parameters lifted into `config.yaml`,
split into a clinical block and a modelling block so ownership is explicit and any change
is a reviewable diff rather than an inline edit. The notebook remains as the Colab runner.
