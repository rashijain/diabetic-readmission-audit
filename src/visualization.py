"""Figure generation for the readmission analysis.

Reads the tables and out-of-fold predictions written by `evaluation.main()` and
regenerates every figure in the report. Run after evaluation:

    python -m src.evaluation
    python -m src.visualization
"""
from __future__ import annotations

import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import CFG, ROOT, load_raw, build_cohort

OUT = ROOT / "outputs"
DPI = 300
REGIME_ORDER = ["drop_mnar", "standard_impute", "explicit_mnar"]
REGIME_COLOURS = {
    "drop_mnar": "#7570b3",
    "standard_impute": "#d95f02",
    "explicit_mnar": "#1b9e77",
}
REGIME_LABELS = {
    "drop_mnar": "A: drop MNAR features",
    "standard_impute": "B: mode imputation",
    "explicit_mnar": "C: explicit MNAR indicators",
}


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def plot_missingness(mnar: pd.DataFrame) -> None:
    """Figure 1: missingness prevalence, and readmission rate by measurement status."""
    data = mnar.sort_values("Missing_pct")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    sns.barplot(
        data=data, y="Feature", x="Missing_pct",
        hue="Feature", palette="viridis", legend=False, ax=axes[0],
    )
    axes[0].set_title("A. Missingness prevalence", fontweight="bold")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 112)
    for i, value in enumerate(data["Missing_pct"]):
        axes[0].text(value + 1.5, i, f"{value:.1f}%", va="center", fontsize=9)

    melted = data.melt(
        id_vars=["Feature", "chi2_p"],
        value_vars=["Readmit_measured", "Readmit_missing"],
        var_name="Status", value_name="Rate",
    )
    melted["Status"] = melted["Status"].map({
        "Readmit_measured": "Measured / documented",
        "Readmit_missing": "Missing / not ordered",
    })
    sns.barplot(data=melted, y="Feature", x="Rate", hue="Status",
                palette="Set2", ax=axes[1])
    axes[1].set_title("B. Informative missingness: 30-day readmission rate",
                      fontweight="bold")
    axes[1].set_xlabel("30-day readmission rate (%)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 16)
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Annotate significance so the weight null result is visible on the figure itself.
    for i, feature in enumerate(data["Feature"]):
        p_value = data.loc[data["Feature"] == feature, "chi2_p"].iloc[0]
        label = "n.s." if p_value >= 0.05 else f"p={p_value:.0e}"
        axes[1].text(13.6, i, label, va="center", fontsize=8,
                     style="italic", color="#444444")

    fig.tight_layout()
    _save(fig, "fig1_mnar.png")


def plot_leakage_and_regimes(results: pd.DataFrame) -> None:
    """Figure 2: leakage ablation, and PR-AUC across the three missingness regimes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    leakage = results[results["Regime"] == "explicit_mnar"].copy()
    leakage["Split"] = leakage["Split"].map({
        "leaky_encounter": "Encounter-level (leaky)",
        "grouped_patient": "Patient-grouped",
    })
    sns.barplot(data=leakage, x="Model", y="ROC_AUC", hue="Split",
                palette=["#d95f02", "#1b9e77"], ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.4f", fontsize=8, padding=2)
    axes[0].set_ylim(0.60, 0.70)
    axes[0].set_ylabel("5-fold mean ROC-AUC")
    axes[0].set_title("A. Leakage ablation — no measurable inflation",
                      fontweight="bold")
    axes[0].legend(title="", fontsize=8)

    grouped = results[results["Split"] == "grouped_patient"].copy()
    grouped["Regime"] = pd.Categorical(grouped["Regime"], REGIME_ORDER, ordered=True)
    sns.barplot(data=grouped.sort_values("Regime"), x="Model", y="PR_AUC",
                hue="Regime", palette="Blues", ax=axes[1])
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.4f", fontsize=8, padding=2)
    axes[1].set_ylim(0.19, 0.25)
    axes[1].set_ylabel("5-fold mean PR-AUC")
    axes[1].set_title("B. Missingness regimes (patient-grouped)", fontweight="bold")
    axes[1].legend(title="", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig2_leakage_regimes.png")


def plot_subgroups(subgroups: pd.DataFrame) -> None:
    """Figure 3: false negative rate and calibration error by recorded race."""
    threshold = CFG["validation"]["clinical_decision_threshold"]
    by_race = subgroups[subgroups["Attribute"] == "race_grp"]
    selected = (by_race[by_race["Regime"] == "explicit_mnar"]
                .sort_values("FNR_at_threshold"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    sns.barplot(data=selected, y="Subgroup", x="FNR_at_threshold",
                hue="Subgroup", palette="Reds", legend=False, ax=axes[0])
    for i, value in enumerate(selected["FNR_at_threshold"]):
        axes[0].text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel(f"False negative rate at {threshold:.0%} threshold")
    axes[0].set_ylabel("")
    axes[0].set_title("A. Missed readmissions by recorded race", fontweight="bold")

    heat = (by_race.pivot(index="Subgroup", columns="Regime", values="ECE")
                   .reindex(columns=REGIME_ORDER))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1],
                cbar_kws={"label": "ECE"})
    axes[1].set_title("B. Expected calibration error by subgroup", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.tight_layout()
    _save(fig, "fig3_subgroup.png")


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                thresholds: np.ndarray) -> np.ndarray:
    """Decision curve net benefit (Vickers & Elkin, 2006).

    False positives are penalised at the odds implied by each threshold, so the
    curve answers whether acting on the model beats treating everyone or no one.
    """
    n = len(y_true)
    benefits = []
    for threshold in thresholds:
        flagged = y_prob >= threshold
        true_pos = np.sum(flagged & (y_true == 1))
        false_pos = np.sum(flagged & (y_true == 0))
        benefits.append(true_pos / n - (false_pos / n) * (threshold / (1 - threshold)))
    return np.array(benefits)


def plot_decision_curve(y_true: np.ndarray, oof_by_regime: dict) -> None:
    """Figure 4: net clinical benefit across plausible intervention thresholds."""
    prevalence = y_true.mean()
    thresholds = np.linspace(0.02, 0.35, 67)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds * 100,
            prevalence - (1 - prevalence) * (thresholds / (1 - thresholds)),
            linestyle="--", color="gray", label="Intervene on all")
    ax.plot(thresholds * 100, np.zeros_like(thresholds),
            linestyle=":", color="black", label="Intervene on none")

    for regime in REGIME_ORDER:
        curve = net_benefit(y_true, oof_by_regime[regime]["HistGBDT"], thresholds)
        ax.plot(thresholds * 100, curve, linewidth=2.2,
                color=REGIME_COLOURS[regime], label=REGIME_LABELS[regime])

    ax.set_title("Decision curve analysis — net clinical benefit "
                 "(HistGBDT, patient-grouped out-of-fold)", fontweight="bold")
    ax.set_xlabel("Intervention threshold (% 30-day readmission risk)")
    ax.set_ylabel("Net clinical benefit")
    ax.set_xlim(2, 35)
    ax.set_ylim(-0.02, prevalence + 0.01)
    ax.legend()

    fig.tight_layout()
    _save(fig, "fig4_dca.png")


def main() -> None:
    # Only force a headless backend for the script entry point; importing this
    # module from a notebook must leave the inline backend alone.
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="notebook")
    OUT.mkdir(exist_ok=True)

    mnar = pd.read_csv(OUT / "mnar_diagnostics.csv")
    results = pd.read_csv(OUT / "all_results.csv")
    subgroups = pd.read_csv(OUT / "subgroup_audit.csv")
    with open(OUT / "oof.pkl", "rb") as handle:
        oof_by_regime = pickle.load(handle)

    y_true = build_cohort(load_raw(), verbose=False)["target"].values

    print("generating figures:")
    plot_missingness(mnar)
    plot_leakage_and_regimes(results)
    plot_subgroups(subgroups)
    plot_decision_curve(y_true, oof_by_regime)


if __name__ == "__main__":
    main()
