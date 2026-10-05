"""ML-4 / VIS-1: model evaluation metrics and performance visualisation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from . import config as C


def compute_metrics(y_true, proba, threshold=C.DECISION_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=130)
    plt.close(fig)


def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log"); ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Class Distribution (highly imbalanced)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    _save(fig, "class_distribution.png")


def plot_amount_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, col, name in [(0, "#4C78A8", "Legitimate"), (1, "#E45756", "Fraud")]:
        ax.hist(np.log1p(df.loc[df[C.TARGET] == cls, "Amount"]), bins=50, alpha=.6,
                color=col, label=name, density=True)
    ax.set_xlabel("log(1 + Amount)"); ax.set_ylabel("Density")
    ax.set_title("Transaction Amount by Class"); ax.legend()
    _save(fig, "amount_distribution.png")


def plot_correlation(df):
    corr = df[C.FEATURES + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET).sort_values()
    top = corr.abs().sort_values(ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(6, 5))
    corr[top].sort_values().plot.barh(ax=ax, color="#72B7B2")
    ax.set_title("Top Feature Correlations with Fraud"); ax.set_xlabel("Pearson r")
    _save(fig, "feature_correlation.png")


def plot_confusion(cm, name, fname="confusion_matrix.png"):
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    _save(fig, fname)


def plot_roc(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(loc=4)
    _save(fig, "roc_curves.png")


def plot_pr(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, p in probas.items():
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(loc=3)
    _save(fig, "pr_curves.png")


def plot_model_comparison(results: dict):
    names = list(results)
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, [results[n][m] for n in names], w, label=m.upper())
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05); ax.set_title("Model Comparison (test set)"); ax.legend(ncol=4)
    _save(fig, "model_comparison.png")


def plot_feature_importance(model, name):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    s = (__import__("pandas").Series(imp, index=C.FEATURES).sort_values().tail(15))
    fig, ax = plt.subplots(figsize=(6, 5))
    s.plot.barh(ax=ax, color="#F58518")
    ax.set_title(f"Top 15 Feature Importances - {name}")
    _save(fig, "feature_importance.png")
