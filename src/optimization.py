"""Deliverable 3: hyperparameter optimization and held-out evaluation.

The design separates the data once and never crosses it again:

  * Patients - not encounters - are split 80/20 into a development set and a
    held-out test set. Searching, calibration experiments and the choice of
    operating threshold all happen on development data.
  * The test set is scored once, at the end, by `evaluate_holdout`.

Every parameter below comes from the `deliverable3` block of config.yaml.
Re-running this module reproduces the reported numbers exactly: the search
objective, the fold count, the sampled space and the seed are all fixed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit,
                                     RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline

from .data_loader import CFG, ROOT, build_cohort, load_raw
from .evaluation import expected_calibration_error
from .preprocessing import build_preprocessor

SEED = CFG["project"]["random_state"]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

D3 = CFG["deliverable3"]
REGIME = D3["regime"]                 # selected in Deliverable 2
FOLDS = D3["search_splits"]
SCORING = CFG["validation"]["scoring"]

# Six hyperparameters, sampled rather than gridded. Random search covers an
# unevenly important space more efficiently than an exhaustive grid of the same
# budget (Bergstra & Bengio, 2012); here learning rate carries nearly all of the
# variation, so most grid points would be spent re-confirming the rest.
SEARCH_SPACE = {
    "clf__learning_rate":     loguniform(*D3["space"]["learning_rate"]),
    "clf__max_iter":          randint(*D3["space"]["max_iter"]),
    "clf__max_leaf_nodes":    randint(*D3["space"]["max_leaf_nodes"]),
    "clf__min_samples_leaf":  randint(*D3["space"]["min_samples_leaf"]),
    "clf__l2_regularization": loguniform(*D3["space"]["l2_regularization"]),
    "clf__max_features":      D3["space"]["max_features"],
}


# --------------------------------------------------------------------------- #
# data split                                                                   #
# --------------------------------------------------------------------------- #
def split_dev_test(df: pd.DataFrame, cfg=CFG, verbose: bool = True):
    """Split on patient_nbr so no patient appears on both sides.

    Splitting on encounters would place a patient's January stay in the
    development set and their March stay in the test set, which is the same
    leakage surface examined in Deliverable 2 - only this time it would corrupt
    the final estimate rather than a cross-validated one.
    """
    y, groups = df.target.values, df.patient_nbr.values
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["deliverable3"]["test_size"],
                                 random_state=SEED)
    dev_idx, test_idx = next(splitter.split(df, y, groups=groups))
    dev = df.iloc[dev_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(dev.patient_nbr) & set(test.patient_nbr)
    assert not overlap, f"{len(overlap)} patients appear in both splits"

    if verbose:
        for name, part in (("development", dev), ("held-out   ", test)):
            print(f"{name} {len(part):,} encounters / {part.patient_nbr.nunique():,} "
                  f"patients / readmission rate {part.target.mean()*100:.2f}%")
        print("patient-disjoint: confirmed")
    return dev, test


def _pipe(clf, regime: str = REGIME) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(regime)[0]), ("clf", clf)])


def _cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)


def _columns(regime: str = REGIME) -> list[str]:
    return build_preprocessor(regime)[1]


def default_models(cfg=CFG) -> dict:
    """The three candidates at their config.yaml defaults, as carried from Deliverable 2."""
    m = cfg["models"]
    return {
        "LogisticRegression": LogisticRegression(**m["logistic_regression"], random_state=SEED),
        "RandomForest":       RandomForestClassifier(**m["random_forest"], random_state=SEED),
        "HistGBDT":           HistGradientBoostingClassifier(**m["hist_gbdt"], random_state=SEED),
    }


# --------------------------------------------------------------------------- #
# stage 1: untuned baselines                                                   #
# --------------------------------------------------------------------------- #
def baselines(dev: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Score the config.yaml defaults on development folds, as the bar tuning must clear."""
    cols, rows = _columns(), []
    for name, clf in default_models().items():
        t0 = time.time()
        scores = cross_validate(_pipe(clf), dev[cols], dev.target.values,
                                groups=dev.patient_nbr.values, cv=_cv(),
                                scoring=SCORING)["test_score"]
        rows.append(dict(Stage="baseline", Model=name,
                         CV_PR_AUC=round(scores.mean(), 4), SD=round(scores.std(), 4),
                         Params=json.dumps(_default_params(name))))
        if verbose:
            print(f"  {name:20s} PR-AUC {scores.mean():.4f} +/- {scores.std():.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(rows)


def _default_params(name: str) -> dict:
    key = {"LogisticRegression": "logistic_regression",
           "RandomForest": "random_forest", "HistGBDT": "hist_gbdt"}[name]
    return CFG["models"][key]


# --------------------------------------------------------------------------- #
# stage 2: searches                                                            #
# --------------------------------------------------------------------------- #
def search_histgbdt(dev: pd.DataFrame, verbose: bool = True):
    """Randomized search over the six-parameter space. Returns (summary_row, all_candidates).

    refit=False: the winner is refitted on the full development set later, by
    `fit_final`, so nothing here is carried forward as a fitted object.
    """
    cols = _columns()
    search = RandomizedSearchCV(
        _pipe(HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)),
        SEARCH_SPACE, n_iter=D3["n_iter"], scoring=SCORING, cv=_cv(),
        random_state=SEED, refit=False)
    t0 = time.time()
    search.fit(dev[cols], dev.target.values, groups=dev.patient_nbr.values)

    candidates = (pd.DataFrame(search.cv_results_)
                  [["params", "mean_test_score", "std_test_score", "rank_test_score"]]
                  .sort_values("mean_test_score", ascending=False)
                  .reset_index(drop=True))
    candidates.to_csv(OUT / "d3_search_histgbdt.csv", index=False)

    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    row = dict(Stage="randomized_search", Model="HistGBDT",
               CV_PR_AUC=round(search.best_score_, 4),
               SD=round(float(candidates.loc[0, "std_test_score"]), 4),
               Params=json.dumps(best))
    if verbose:
        lo, hi = candidates.mean_test_score.min(), candidates.mean_test_score.max()
        print(f"  {D3['n_iter']} candidates in {time.time()-t0:.0f}s")
        print(f"  best PR-AUC {search.best_score_:.4f}")
        print(f"  spread {lo:.4f} to {hi:.4f}  ({100*(hi-lo)/hi:.0f}% relative)")
        print(f"  best params {best}")
    return row, candidates


def search_grid(dev: pd.DataFrame, model: str, verbose: bool = True):
    """Grid search for the two reference models, so the comparison stays fair."""
    cols = _columns()
    if model == "LogisticRegression":
        estimator = LogisticRegression(max_iter=D3["logreg_max_iter"], random_state=SEED)
        grid = {f"clf__{k}": v for k, v in D3["grids"]["logistic_regression"].items()}
    elif model == "RandomForest":
        estimator = RandomForestClassifier(n_estimators=D3["rf_n_estimators"],
                                           n_jobs=-1, random_state=SEED)
        grid = {f"clf__{k}": v for k, v in D3["grids"]["random_forest"].items()}
    else:
        raise ValueError(f"no grid defined for {model!r}")

    search = GridSearchCV(_pipe(estimator), grid, scoring=SCORING, cv=_cv(), refit=False)
    t0 = time.time()
    search.fit(dev[cols], dev.target.values, groups=dev.patient_nbr.values)

    results = (pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
               .sort_values("mean_test_score", ascending=False).reset_index(drop=True))
    results.to_csv(OUT / f"d3_search_{model.lower()}.csv", index=False)

    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    row = dict(Stage="grid_search", Model=model, CV_PR_AUC=round(search.best_score_, 4),
               SD=round(float(results.loc[0, "std_test_score"]), 4), Params=json.dumps(best))
    if verbose:
        print(f"  {model:20s} best PR-AUC {search.best_score_:.4f} {best} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return row, results


def tuning_summary(rows: list[dict]) -> pd.DataFrame:
    """Collect baseline and search rows into the table reported as Table 1."""
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "d3_tuning_summary.csv", index=False)
    return summary


def best_params(summary: pd.DataFrame | None = None) -> dict:
    """The winning HistGBDT configuration, read back from the saved summary."""
    if summary is None:
        summary = pd.read_csv(OUT / "d3_tuning_summary.csv")
    row = summary.query("Stage == 'randomized_search' and Model == 'HistGBDT'").iloc[0]
    return json.loads(row["Params"])


# --------------------------------------------------------------------------- #
# stage 3: final models and the held-out score                                 #
# --------------------------------------------------------------------------- #
def fit_final(dev: pd.DataFrame, params: dict, verbose: bool = True) -> dict:
    """Fit the four candidates reported in Table 3 on the full development set.

    The isotonic variant is calibrated on a patient-disjoint slice of the
    development data, never on the test set.
    """
    cols = _columns()
    y = dev.target.values
    models = {}

    for name, clf in (("Tuned HistGBDT",
                       HistGradientBoostingClassifier(**params, early_stopping=False,
                                                      random_state=SEED)),
                      ("Untuned HistGBDT",
                       HistGradientBoostingClassifier(**CFG["models"]["hist_gbdt"],
                                                      random_state=SEED))):
        t0 = time.time()
        models[name] = _pipe(clf).fit(dev[cols], y)
        if verbose:
            print(f"  fitted {name} [{time.time()-t0:.0f}s]", flush=True)

    fit_idx, cal_idx = next(GroupShuffleSplit(1, test_size=D3["calibration_fraction"],
                                              random_state=SEED)
                            .split(dev, y, groups=dev.patient_nbr.values))
    base = _pipe(HistGradientBoostingClassifier(**params, early_stopping=False,
                                                random_state=SEED))
    base.fit(dev.iloc[fit_idx][cols], y[fit_idx])
    models["Tuned + isotonic"] = CalibratedClassifierCV(
        FrozenEstimator(base), method="isotonic").fit(dev.iloc[cal_idx][cols], y[cal_idx])

    logreg_c = json.loads(pd.read_csv(OUT / "d3_tuning_summary.csv")
                          .query("Model == 'LogisticRegression' and Stage == 'grid_search'")
                          .iloc[0]["Params"])["C"]
    models["LogReg (tuned)"] = _pipe(LogisticRegression(
        C=logreg_c, max_iter=D3["logreg_max_iter"], random_state=SEED)).fit(dev[cols], y)
    return models


def evaluate_holdout(models: dict, test: pd.DataFrame, verbose: bool = True):
    """Score every fitted candidate on the held-out set. Returns (table, probabilities)."""
    cols, y = _columns(), test.target.values
    rows, probs = [], {}
    for name, model in models.items():
        p = model.predict_proba(test[cols])[:, 1]
        probs[name] = p
        rows.append(dict(Model=name,
                         ROC_AUC=round(roc_auc_score(y, p), 4),
                         PR_AUC=round(average_precision_score(y, p), 4),
                         Brier=round(brier_score_loss(y, p), 4),
                         ECE=round(expected_calibration_error(y, p), 4)))
        if verbose:
            print(f"  {name:20s} " + "  ".join(f"{k}={v}" for k, v in rows[-1].items()
                                               if k != "Model"), flush=True)
    table = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    table.to_csv(OUT / "d3_test_metrics.csv", index=False)
    return table, probs


# --------------------------------------------------------------------------- #
# stage 4: operating point                                                     #
# --------------------------------------------------------------------------- #
def operating_stats(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Everything a discharge team would ask about one cut-off."""
    flagged = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    ppv = tp / (tp + fp) if tp + fp else 0.0
    return dict(threshold=round(float(threshold), 4),
                flagged=int(flagged.sum()), flagged_pct=round(100 * flagged.mean(), 2),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision=round(ppv, 4),
                recall=round(tp / (tp + fn), 4) if tp + fn else None,
                specificity=round(tn / (tn + fp), 4) if tn + fp else None,
                FNR=round(fn / (tp + fn), 4) if tp + fn else None,
                F1=round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0,
                accuracy=round((tp + tn) / len(y), 4),
                lift=round(ppv / y.mean(), 2) if ppv else 0.0,
                contacts_per_catch=round(1 / ppv, 1) if ppv else None)


def choose_operating_point(dev: pd.DataFrame, params: dict, verbose: bool = True):
    """Fix the probability cut-off on development data, never on the test set.

    Capacity, not probability, defines the decision: the team can follow up a
    fixed share of discharges, so the threshold is the matching quantile of
    out-of-fold development predictions. Taking it from the test set instead
    would be choosing the operating point with the answer in hand.
    """
    cols, capacity = _columns(), D3["capacity"]
    oof = cross_val_predict(
        _pipe(HistGradientBoostingClassifier(**params, early_stopping=False, random_state=SEED)),
        dev[cols], dev.target.values, groups=dev.patient_nbr.values,
        cv=_cv(), method="predict_proba")[:, 1]
    np.save(OUT / "d3_dev_oof.npy", oof)

    threshold = float(np.quantile(oof, 1 - capacity))
    if verbose:
        print(f"  top {capacity:.0%} of development encounters -> cut-off {threshold:.4f}")
    return threshold, oof


def capacity_sweep(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """What the team would get at other staffing levels, for Table 4 and Figure 4A."""
    rows = []
    for capacity in D3["capacity_grid"]:
        stats = operating_stats(y, p, float(np.quantile(p, 1 - capacity)))
        stats["capacity"] = capacity
        rows.append(stats)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "d3_capacity_sweep.csv", index=False)
    return sweep


# --------------------------------------------------------------------------- #
# stage 5: audit                                                               #
# --------------------------------------------------------------------------- #
def subgroup_audit_test(test: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per-group performance at the deployed cut-off, on data used for nothing else."""
    y, rows = test.target.values, []
    for column in D3["audit_attributes"]:
        values = test[column].fillna("Not recorded")
        for group in sorted(values.unique()):
            mask = (values == group).values
            if mask.sum() < D3["min_subgroup_n"] or y[mask].sum() < 10:
                continue
            stats = operating_stats(y[mask], p[mask], threshold)
            rows.append(dict(Attribute=column, Group=group, N=int(mask.sum()),
                             Prevalence=round(float(y[mask].mean()), 4),
                             ROC_AUC=round(roc_auc_score(y[mask], p[mask]), 4),
                             Flag_rate=round(stats["flagged_pct"] / 100, 4),
                             Recall=stats["recall"], FNR=stats["FNR"],
                             Precision=stats["precision"]))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "d3_subgroup_test.csv", index=False)
    return audit


def feature_importance(model, test: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Permutation importance: the drop in average precision when one column is shuffled.

    Measured on a test subsample for runtime, and reported as a check on what the
    model leans on rather than as a causal statement.
    """
    cols = _columns()
    sample = test.sample(min(D3["importance_sample"], len(test)), random_state=SEED)
    result = permutation_importance(model, sample[cols], sample.target.values,
                                    n_repeats=D3["importance_repeats"],
                                    scoring=SCORING, random_state=SEED)
    importance = (pd.DataFrame({"Feature": cols,
                                "Importance": result.importances_mean,
                                "SD": result.importances_std})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))
    importance.to_csv(OUT / "d3_permutation_importance.csv", index=False)
    if verbose:
        print(importance.head(8).to_string(index=False))
    return importance


# --------------------------------------------------------------------------- #
# script entry point                                                           #
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    df = build_cohort(load_raw())
    dev, test = split_dev_test(df)

    print("\nstage 1  untuned baselines")
    rows = baselines(dev).to_dict("records")

    print(f"\nstage 2  randomized search, HistGBDT ({D3['n_iter']} candidates)")
    hgb_row, _ = search_histgbdt(dev)
    rows.append(hgb_row)

    print("\nstage 3  grid searches")
    for model in ("LogisticRegression", "RandomForest"):
        rows.append(search_grid(dev, model)[0])
    summary = tuning_summary(rows)
    params = best_params(summary)

    print("\nstage 4  final fits and the held-out score")
    models = fit_final(dev, params)
    _, probs = evaluate_holdout(models, test)
    p_test = probs["Tuned HistGBDT"]
    np.save(OUT / "d3_test_probs.npy", p_test)
    test[["patient_nbr", "target"] + D3["audit_attributes"]].to_csv(
        OUT / "d3_test_index.csv", index=False)

    print("\nstage 5  operating point and audit")
    threshold, _ = choose_operating_point(dev, params)
    point = operating_stats(test.target.values, p_test, threshold)
    pd.DataFrame([point]).to_csv(OUT / "d3_operating_point.csv", index=False)
    print("  held-out at that cut-off:", json.dumps(point))
    print(f"  accuracy of flagging nobody: {1 - test.target.mean():.4f}")

    capacity_sweep(test.target.values, p_test)
    print(subgroup_audit_test(test, p_test, threshold).to_string(index=False))
    feature_importance(models["Tuned HistGBDT"], test)

    print(f"\nELAPSED {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
