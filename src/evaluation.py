"""Experiments: leakage ablation (RQ1), missingness regimes (RQ2), tuning, subgroup audit (RQ3)."""
from __future__ import annotations
import pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .data_loader import CFG, ROOT, load_raw, build_cohort
from .preprocessing import build_preprocessor, REGIMES

SEED = CFG["project"]["random_state"]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)


def get_models(cfg=CFG):
    m = cfg["models"]
    return {
        "LogisticRegression": LogisticRegression(**m["logistic_regression"], random_state=SEED),
        "RandomForest":       RandomForestClassifier(**m["random_forest"], n_jobs=-1, random_state=SEED),
        "HistGBDT":           HistGradientBoostingClassifier(**m["hist_gbdt"], random_state=SEED),
    }


def expected_calibration_error(y, p, n_bins=None):
    n_bins = n_bins or CFG["validation"]["ece_bins"]
    edges = np.linspace(0, 1, n_bins + 1); err = 0.0; n = len(y)
    for i in range(n_bins):
        hi_inclusive = i == n_bins - 1
        m = (p >= edges[i]) & (p <= edges[i+1]) if hi_inclusive else (p >= edges[i]) & (p < edges[i+1])
        if m.sum():
            err += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(err)


def missingness_diagnostics(df, cfg=CFG) -> pd.DataFrame:
    """Test whether missingness in the high-missingness features is informative.

    For each feature, compares the 30-day readmission rate among encounters where it
    is recorded against those where it is absent, with a chi-square test of
    independence. A significant result means the absence of the value is itself
    predictive, so dropping or imputing the column destroys signal.
    """
    from scipy.stats import chi2_contingency

    rows = []
    for column in cfg["data"]["mnar_cols"]:
        missing = df[column].isna()
        _, p_value, _, _ = chi2_contingency(pd.crosstab(missing, df.target))
        measured_rate = df.loc[~missing, "target"].mean()
        missing_rate = df.loc[missing, "target"].mean()
        rows.append({
            "Feature": column,
            "Missing_pct": round(missing.mean() * 100, 2),
            "Readmit_measured": round(measured_rate * 100, 2),
            "Readmit_missing": round(missing_rate * 100, 2),
            "Delta": round((missing_rate - measured_rate) * 100, 2),
            "chi2_p": p_value,
        })
    out = pd.DataFrame(rows).sort_values("Missing_pct", ascending=False)
    out.to_csv(OUT / "mnar_diagnostics.csv", index=False)
    return out


def run_cv(df, regime, split, cfg=CFG, verbose=True):
    """split: 'leaky_encounter' (encounter-level) or 'grouped_patient' (grouped on patient_nbr)."""
    y, groups = df.target.values, df.patient_nbr.values
    k = cfg["validation"]["n_splits"]
    if split == "leaky_encounter":
        folds = list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(df, y))
    elif split == "grouped_patient":
        folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(df, y, groups=groups))
    else:
        raise ValueError(split)

    results, oof_preds = [], {}
    for name, est in get_models(cfg).items():
        t0 = time.time(); oof = np.zeros(len(df)); fold = {m: [] for m in ("auc","pr","brier","ece")}
        for tr, va in folds:
            prep, cols = build_preprocessor(regime, cfg)
            pipe = Pipeline([("prep", prep), ("clf", est)]).fit(df.iloc[tr][cols], y[tr])
            p = pipe.predict_proba(df.iloc[va][cols])[:, 1]
            oof[va] = p
            fold["auc"].append(roc_auc_score(y[va], p))
            fold["pr"].append(average_precision_score(y[va], p))
            fold["brier"].append(brier_score_loss(y[va], p))
            fold["ece"].append(expected_calibration_error(y[va], p))
        oof_preds[name] = oof
        results.append(dict(Split=split, Regime=regime, Model=name,
                            ROC_AUC=np.mean(fold["auc"]), ROC_SD=np.std(fold["auc"]),
                            PR_AUC=np.mean(fold["pr"]), Brier=np.mean(fold["brier"]),
                            ECE=np.mean(fold["ece"])))
        if verbose:
            print(f"  {name:20s} AUC={np.mean(fold['auc']):.4f} "
                  f"PR={np.mean(fold['pr']):.4f} [{time.time()-t0:.0f}s]", flush=True)
    return pd.DataFrame(results), oof_preds


def tune(df, regime="explicit_mnar", cfg=CFG):
    """Grid search on the selected model. Objective is average precision, not accuracy."""
    y, groups = df.target.values, df.patient_nbr.values
    prep, cols = build_preprocessor(regime, cfg)
    cv = StratifiedGroupKFold(cfg["validation"]["tuning_splits"], shuffle=True, random_state=SEED)
    base_clf = HistGradientBoostingClassifier(min_samples_leaf=cfg["models"]["hist_gbdt"]["min_samples_leaf"],
                                              random_state=SEED)
    gs = GridSearchCV(Pipeline([("prep", build_preprocessor(regime, cfg)[0]), ("clf", base_clf)]),
                      dict(cfg["tuning_grid"]), scoring=cfg["validation"]["scoring"], cv=cv, refit=True)
    gs.fit(df[cols], y, groups=groups)
    untuned = Pipeline([("prep", build_preprocessor(regime, cfg)[0]),
                        ("clf", HistGradientBoostingClassifier(**cfg["models"]["hist_gbdt"], random_state=SEED))])
    base = cross_val_score(untuned, df[cols], y, groups=groups, cv=cv,
                           scoring=cfg["validation"]["scoring"]).mean()
    print(f"best {gs.best_params_}")
    print(f"tuned {gs.best_score_:.4f} vs untuned {base:.4f} -> delta {gs.best_score_-base:+.4f}")
    return gs, base


def subgroup_audit(df, oof_by_regime, cfg=CFG):
    """Audit out-of-fold predictions by recorded race, age band and payer.

    df must carry a RangeIndex (data_loader resets it) so that .loc labels from
    groupby align with positional indexing into the out-of-fold arrays.
    """
    thr = cfg["validation"]["clinical_decision_threshold"]
    aud = df[["target", "race", "age_group", "payer_code"]].copy()
    aud["race_grp"] = aud.race.fillna("Missing")
    top = aud.payer_code.value_counts().nlargest(5).index
    aud["payer_grp"] = aud.payer_code.apply(lambda x: "Missing" if pd.isna(x) else (x if x in top else "Other"))

    rows = []
    for regime, by_model in oof_by_regime.items():
        probs = by_model["HistGBDT"]
        for attr in ("race_grp", "age_group", "payer_grp"):
            for value, idx in aud.groupby(attr).groups.items():
                pos = np.asarray(idx)
                ys, ps = aud.loc[idx, "target"].values, probs[pos]
                if len(ys) < 100 or len(np.unique(ys)) < 2:
                    continue
                flagged = (ps >= thr).astype(int)
                fn = ((ys == 1) & (flagged == 0)).sum()
                tp = ((ys == 1) & (flagged == 1)).sum()
                rows.append(dict(Regime=regime, Attribute=attr, Subgroup=value, N=len(ys),
                                 Prevalence=ys.mean(), ROC_AUC=roc_auc_score(ys, ps),
                                 ECE=expected_calibration_error(ys, ps),
                                 FNR_at_threshold=fn / (fn + tp) if (fn + tp) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "subgroup_audit.csv", index=False)
    return out


def main():
    df = build_cohort(load_raw())
    print("\nMissingness diagnostics")
    print(missingness_diagnostics(df).to_string(index=False))
    frames, oof_by_regime = [], {}
    print("\nRQ1 - leakage ablation")
    for split in ("leaky_encounter", "grouped_patient"):
        print(f" {split}:")
        r, o = run_cv(df, "explicit_mnar", split)
        frames.append(r)
        if split == "grouped_patient":
            oof_by_regime["explicit_mnar"] = o
    print("\nRQ2 - missingness regimes")
    for regime in ("drop_mnar", "standard_impute"):
        print(f" {regime}:")
        r, o = run_cv(df, regime, "grouped_patient")
        frames.append(r); oof_by_regime[regime] = o
    allres = pd.concat(frames, ignore_index=True)
    allres.to_csv(OUT / "all_results.csv", index=False)
    pickle.dump(oof_by_regime, open(OUT / "oof.pkl", "wb"))
    print("\nTuning"); tune(df)
    print("\nRQ3 - subgroup audit")
    sub = subgroup_audit(df, oof_by_regime)
    print(sub[(sub.Attribute == "race_grp") & (sub.Regime == "explicit_mnar")]
          [["Subgroup", "N", "ROC_AUC", "ECE", "FNR_at_threshold"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
