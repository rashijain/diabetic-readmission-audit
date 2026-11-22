"""Ingestion, cohort construction and feature engineering.

Clinical parameters (exclusion rules, ICD-9 grouping) live in config.yaml.
This module applies them; it does not define them.
"""
from __future__ import annotations
import io, zipfile, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
ROOT = Path(__file__).resolve().parents[1]

# Missing values are encoded three ways in the raw file. keep_default_na=False with an
# explicit list makes parsing deterministic without corrupting the 'No' medication flags.
NA_VALUES = ["", "?", "None", "NA", "NaN", "null", "Unknown/Invalid"]


def download(cfg=CFG) -> Path:
    csv = ROOT / cfg["data"]["local_csv"]
    if csv.exists():
        return csv
    csv.parent.mkdir(parents=True, exist_ok=True)
    for url in cfg["data"]["sources"]:
        try:
            raw = urllib.request.urlopen(url, timeout=180).read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = next(n for n in zf.namelist() if n.endswith("diabetic_data.csv"))
                csv.write_bytes(zf.read(name))
            return csv
        except Exception as e:                       # noqa: BLE001
            print(f"  source failed ({type(e).__name__}): {url}")
    raise RuntimeError("Could not download the dataset; place diabetic_data.csv in data/ manually.")


def load_raw(cfg=CFG) -> pd.DataFrame:
    df = pd.read_csv(download(cfg), keep_default_na=False, na_values=NA_VALUES, low_memory=False)
    expected = tuple(cfg["data"]["expected_shape"])
    assert df.shape == expected, f"expected {expected}, got {df.shape} - wrong source file?"
    return df


def map_icd9(code) -> object:
    """Collapse 700+ ICD-9 codes into the nine categories used by Strack et al. (2014)."""
    if pd.isna(code):
        return np.nan
    s = str(code).strip()
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if (390 <= v <= 459) or v == 785:  return "Circulatory"
    if (460 <= v <= 519) or v == 786:  return "Respiratory"
    if (520 <= v <= 579) or v == 787:  return "Digestive"
    if 250 <= v < 251:                 return "Diabetes"
    if 800 <= v <= 999:                return "Injury"
    if 710 <= v <= 739:                return "Musculoskeletal"
    if (580 <= v <= 629) or v == 788:  return "Genitourinary"
    if 140 <= v <= 239:                return "Neoplasms"
    return "Other"


MED_COLS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide",
            "glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose",
            "miglitol","troglitazone","tolazamide","insulin","glyburide-metformin",
            "glipizide-metformin","glimepiride-pioglitazone","metformin-rosiglitazone",
            "metformin-pioglitazone"]


def build_cohort(raw: pd.DataFrame, cfg=CFG, verbose: bool = True) -> pd.DataFrame:
    c = cfg["clinical"]
    df = raw.copy()
    n0 = len(df)
    df = df[~df.discharge_disposition_id.isin(c["expired_hospice_ids"])].copy()
    df.drop(columns=c["zero_variance_cols"], errors="ignore", inplace=True)
    df["target"] = (df.readmitted == c["target_label"]).astype(int)
    df["age_midpoint"] = df.age.map({f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)})
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_grp"] = df[col].apply(map_icd9)
    meds = [m for m in MED_COLS if m in df.columns]
    df["num_med_changes"]    = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["num_active_meds"]    = (df[meds] != "No").sum(axis=1)
    df["total_prior_visits"] = df.number_outpatient + df.number_emergency + df.number_inpatient
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)
    df["age_group"] = pd.cut(df.age_midpoint, bins=[0, 49, 69, 100],
                             labels=["<50", "50-69", "70+"]).astype(str)
    # Required: downstream subgroup indexing mixes .loc labels with positional numpy
    # arrays of out-of-fold predictions. Without this the two diverge after the
    # exclusions above and the audit raises IndexError.
    df = df.reset_index(drop=True)
    if verbose:
        rep = df.duplicated("patient_nbr", keep=False).sum()
        print(f"raw {n0:,} -> cohort {len(df):,} encounters, {df.patient_nbr.nunique():,} patients")
        print(f"rows from repeat patients: {rep:,} ({100*rep/len(df):.1f}%)  <- leakage surface")
        print(f"30-day readmission rate  : {df.target.mean()*100:.2f}%")
    return df


if __name__ == "__main__":
    build_cohort(load_raw())
