"""Deliverable 3 figures.

Reads the tables that `src.optimization` writes into outputs/ and regenerates
Figures 1-5 of the Deliverable 3 report. Run after the optimization:

    python -m src.optimization
    python -m src.d3_figures

Kept separate from `src.visualization`, which owns the Deliverable 2 figures,
so neither report's figures can be changed by editing the other's code.
"""
from __future__ import annotations

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .data_loader import CFG, ROOT

OUT = ROOT / "outputs"
DPI = 300
TEAL, CORAL, SLATE, AMBER, GREY = "#2E8B8B", "#E0603A", "#24374B", "#C9A227", "#8395A7"
STYLE = {"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
         "axes.spines.top": False, "axes.spines.right": False}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote outputs/{name}")


def _load():
    """Everything the five figures need, read back from disk rather than recomputed."""
    return dict(
        y=pd.read_csv(OUT / "d3_test_index.csv").target.values,
        p=np.load(OUT / "d3_test_probs.npy"),
        sweep=pd.read_csv(OUT / "d3_capacity_sweep.csv"),
        point=pd.read_csv(OUT / "d3_operating_point.csv").iloc[0],
        search=pd.read_csv(OUT / "d3_search_histgbdt.csv"),
        summary=pd.read_csv(OUT / "d3_tuning_summary.csv"),
        subgroups=pd.read_csv(OUT / "d3_subgroup_test.csv"),
        importance=pd.read_csv(OUT / "d3_permutation_importance.csv"),
    )


def _learning_rates(search: pd.DataFrame) -> np.ndarray:
    """Pull one hyperparameter back out of the saved params column.

    GridSearchCV writes the dict with numpy scalars repr'd inside it, so the
    column is not valid JSON and has to be read with a pattern rather than
    parsed. Only used for plotting.
    """
    pattern = re.compile(r"learning_rate.:\s*(?:np\.float64\()?([0-9.eE+-]+)")
    return search.params.apply(lambda s: float(pattern.search(s).group(1))).values


def plot_search(search: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Figure 1: where the sampled configurations landed, and why."""
    scores = search.mean_test_score.values
    default = float(summary.query("Stage == 'baseline' and Model == 'HistGBDT'")
                    .iloc[0]["CV_PR_AUC"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    axes[0].hist(scores, bins=18, color=TEAL, alpha=0.75, edgecolor="white")
    axes[0].axvline(default, color=CORAL, lw=2, label=f"untuned default ({default:.4f})")
    axes[0].axvline(scores.max(), color=SLATE, lw=2, ls="--",
                    label=f"best of {len(scores)} ({scores.max():.4f})")
    axes[0].set_xlabel("cross-validated PR-AUC")
    axes[0].set_ylabel("candidate configurations")
    axes[0].set_title(f"A. {len(scores)} random configurations", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7.5, frameon=False)

    axes[1].scatter(_learning_rates(search), scores, s=18, color=SLATE, alpha=0.7)
    axes[1].axhline(default, color=CORAL, lw=1.4, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("B. Learning rate dominates the spread", loc="left", fontweight="bold")
    _save(fig, "d3_fig1_search.png")


def plot_roc_pr(y: np.ndarray, p: np.ndarray) -> None:
    """Figure 2: discrimination on the held-out set."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=TEAL, lw=2)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("A. ROC, held-out test set", loc="left", fontweight="bold")

    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, color=TEAL, lw=2)
    axes[1].axhline(y.mean(), color=GREY, ls="--", lw=1)
    axes[1].annotate(f"prevalence {y.mean()*100:.2f}%", (0.52, y.mean() + 0.012),
                     fontsize=7.5, color=GREY)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("B. Precision-recall", loc="left", fontweight="bold")
    _save(fig, "d3_fig2_roc_pr.png")


def plot_calibration_and_confusion(y: np.ndarray, p: np.ndarray, point: pd.Series) -> None:
    """Figure 3: are the probabilities honest, and what happens at the cut-off."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    limit = max(predicted) * 1.05
    axes[0].plot([0, limit], [0, limit], color=GREY, ls="--", lw=1, label="perfect")
    axes[0].plot(predicted, observed, "o-", color=TEAL, lw=1.8, ms=4, label="tuned model")
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed readmission rate")
    axes[0].set_title("A. Calibration (deciles of risk)", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7.5, frameon=False)

    matrix = np.array([[point.TN, point.FP], [point.FN, point.TP]], dtype=float)
    share = matrix / matrix.sum()
    axes[1].imshow(share, cmap="Blues", vmin=0, vmax=0.8)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{int(matrix[i, j]):,}\n{100*share[i, j]:.1f}%",
                         ha="center", va="center", fontsize=9,
                         color="white" if share[i, j] > 0.4 else SLATE)
    axes[1].set_xticks([0, 1], ["not flagged", "flagged"])
    axes[1].set_yticks([0, 1], ["no readmit", "readmitted"])
    axes[1].set_title(f"B. Confusion matrix at the top-"
                      f"{CFG['deliverable3']['capacity']:.0%} cutoff",
                      loc="left", fontweight="bold")
    axes[1].grid(False)
    _save(fig, "d3_fig3_calibration_confusion.png")


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Vickers & Elkin (2006): true positives credited, false positives charged
    at the odds the threshold implies."""
    flagged = p >= threshold
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    return tp / len(y) - (fp / len(y)) * (threshold / (1 - threshold))


def plot_capacity_and_dca(y: np.ndarray, p: np.ndarray, sweep: pd.DataFrame) -> None:
    """Figure 4: the staffing trade-off, and whether acting on the model pays."""
    capacity = CFG["deliverable3"]["capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    axes[0].plot(sweep.capacity * 100, sweep.recall * 100, "o-", color=TEAL, lw=2, ms=4,
                 label="recall (% of readmissions caught)")
    axes[0].plot(sweep.capacity * 100, sweep.precision * 100, "s-", color=CORAL, lw=2, ms=4,
                 label="precision (% of flags correct)")
    axes[0].axvline(capacity * 100, color=SLATE, ls=":", lw=1.4)
    axes[0].annotate("chosen capacity", (capacity * 100 + 0.6, 8), fontsize=7.5, color=SLATE)
    axes[0].set_xlabel("% of discharges followed up")
    axes[0].set_ylabel("percent")
    axes[0].set_title("A. The capacity trade-off", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7.5, frameon=False)

    thresholds = np.linspace(0.01, 0.45, 90)
    prevalence = y.mean()
    axes[1].plot(thresholds, [net_benefit(y, p, t) for t in thresholds],
                 color=TEAL, lw=2, label="tuned model")
    axes[1].plot(thresholds, [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds],
                 color=CORAL, lw=1.6, ls="--", label="follow up everyone")
    axes[1].axhline(0, color=SLATE, lw=1.2, label="follow up no one")
    axes[1].set_ylim(-0.08, 0.05)
    axes[1].set_xlabel("risk threshold")
    axes[1].set_ylabel("net benefit")
    axes[1].set_title("B. Decision curve analysis", loc="left", fontweight="bold")
    axes[1].legend(fontsize=7.5, frameon=False)
    _save(fig, "d3_fig4_capacity_dca.png")


def plot_subgroups_and_importance(subgroups: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Figure 5: who the model misses, and what it leans on."""
    race = subgroups[subgroups.Attribute == "race"].sort_values("FNR")
    top = importance.head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    colours = plt.cm.Reds(np.linspace(0.35, 0.85, len(race)))
    axes[0].barh(race.Group, race.FNR, color=colours, edgecolor="white")
    for i, (value, n) in enumerate(zip(race.FNR, race.N)):
        axes[0].text(value + 0.01, i, f"{value:.3f}  (n={n:,})", va="center",
                     fontsize=7.5, color=SLATE)
    axes[0].set_xlim(0, race.FNR.max() * 1.28)
    axes[0].set_xlabel("false negative rate at the operating point")
    axes[0].set_title("A. Missed readmissions by recorded race", loc="left", fontweight="bold")

    axes[1].barh(top.Feature, top.Importance, xerr=top.SD, color=TEAL, edgecolor="white",
                 error_kw=dict(ecolor=GREY, lw=0.8))
    axes[1].set_xlabel("drop in PR-AUC when the feature is shuffled")
    axes[1].set_title("B. What the model actually uses", loc="left", fontweight="bold")
    _save(fig, "d3_fig5_subgroup_importance.png")


def plot_all() -> None:
    """Every Deliverable 3 figure, for use from the notebook (inline backend kept)."""
    with plt.rc_context(STYLE):
        data = _load()
        plot_search(data["search"], data["summary"])
        plot_roc_pr(data["y"], data["p"])
        plot_calibration_and_confusion(data["y"], data["p"], data["point"])
        plot_capacity_and_dca(data["y"], data["p"], data["sweep"])
        plot_subgroups_and_importance(data["subgroups"], data["importance"])


def main() -> None:
    # Headless only for the script entry point; importing from a notebook must
    # leave the inline backend alone.
    matplotlib.use("Agg")
    OUT.mkdir(exist_ok=True)
    print("generating Deliverable 3 figures:")
    plot_all()


if __name__ == "__main__":
    main()
