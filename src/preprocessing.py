"""The three missingness regimes, each as a leakage-safe ColumnTransformer.

Every transform that learns a parameter sits inside the Pipeline so it is fitted on
training folds only. Fitting a scaler or imputer before splitting is a silent source
of optimistic bias (Kaufman et al., 2012).
"""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .data_loader import CFG

NUMERIC = ["age_midpoint","time_in_hospital","num_lab_procedures","num_procedures",
           "num_medications","number_outpatient","number_emergency","number_inpatient",
           "number_diagnoses","num_med_changes","num_active_meds","total_prior_visits"]

BASE_CAT = ["race","gender","admission_type_id","discharge_disposition_id",
            "admission_source_id","diag_1_grp","diag_2_grp","diag_3_grp",
            "change","diabetesMed","insulin","metformin"]

REGIMES = ("drop_mnar", "standard_impute", "explicit_mnar")


def build_preprocessor(regime: str, cfg=CFG):
    """Return (ColumnTransformer, feature_columns) for one missingness regime."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    miss = cfg["clinical"]["missingness_cols"]
    min_freq = cfg["validation"]["min_category_freq"]

    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])

    if regime == "drop_mnar":            # A: discard the high-missingness features
        cat, imp = BASE_CAT, SimpleImputer(strategy="most_frequent")
    elif regime == "standard_impute":    # B: keep them, impute modally (erases the signal)
        cat, imp = BASE_CAT + miss, SimpleImputer(strategy="most_frequent")
    else:                                # C: keep them, encode absence as a category
        cat, imp = BASE_CAT + miss, SimpleImputer(strategy="constant", fill_value="Missing")

    cat_pipe = Pipeline([
        ("imp", imp),
        # min_frequency pools rare levels: medical_specialty alone has 73.
        ("ohe", OneHotEncoder(min_frequency=min_freq, handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat_pipe, cat)], remainder="drop"), NUMERIC + cat
