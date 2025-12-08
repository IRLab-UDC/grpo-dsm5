import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass(frozen=True)
class EvalConfig:
    result_file: str = "inference_results.jsonl"
    output_dir: str = "eval_outputs"
    truth_field: str = "gold_all_symptoms"
    pred_field: str = "predicted_symptoms"
    lowercase_tags: bool = False
    per_label_f1_top_k: int = 30
    fig_dpi: int = 160
    annotate_all_confusions: bool = False
    normalize_confusions: bool = True


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_results(result_file: str) -> List[dict]:
    rows: List[dict] = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_text(text: str, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def save_per_label_csv(classes: List[str], y_true_bin: np.ndarray, 
                       y_pred_bin: np.ndarray, file_path: str) -> None:
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for label, p, r, f, s in zip(classes, prec, rec, f1, support):
            writer.writerow([label, f"{p:.6f}", f"{r:.6f}", f"{f:.6f}", int(s)])


def normalize_labels(obj: Union[None, str, Iterable[str]], *, lowercase: bool = False) -> Set[str]:
    if obj is None:
        return set()

    def clean_one(x: str) -> str:
        s = str(x).strip()
        if lowercase:
            s = s.lower()
        return s

    if isinstance(obj, str):
        parts = (seg for semi in obj.split(";") for seg in semi.split(","))
        cleaned = [clean_one(p) for p in parts if p.strip()]
        return set(cleaned)

    try:
        cleaned = [clean_one(x) for x in obj if str(x).strip()]
        return set(cleaned)
    except TypeError:
        s = clean_one(obj)
        return {s} if s else set()


def to_binary(y_true_sets: List[Set[str]], y_pred_sets: List[Set[str]]) -> Tuple[MultiLabelBinarizer, List[str], np.ndarray, np.ndarray]:
    all_labels = sorted(set().union(*y_true_sets, *y_pred_sets))
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_true_bin = mlb.fit_transform(y_true_sets)
    y_pred_bin = mlb.transform(y_pred_sets)
    classes = list(mlb.classes_)
    return mlb, classes, y_true_bin, y_pred_bin


def build_summary_text(classes: List[str], y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> str:
    lines: List[str] = []

    subset_acc = accuracy_score(y_true_bin, y_pred_bin)
    ham = hamming_loss(y_true_bin, y_pred_bin)
    jacc_samples = jaccard_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="weighted", zero_division=0
    )

    lines.append("=== Multilabel Evaluation Summary ===")
    lines.append(f"Num samples: {y_true_bin.shape[0]}")
    lines.append(f"Num labels:  {len(classes)}")
    lines.append("")
    lines.append("Global metrics:")
    lines.append(f"- Subset accuracy (exact match): {subset_acc:.4f}")
    lines.append(f"- Hamming loss:                  {ham:.4f}")
    lines.append(f"- Jaccard (samples average):     {jacc_samples:.4f}")
    lines.append(f"- Precision / Recall / F1 (micro):    {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
    lines.append(f"- Precision / Recall / F1 (macro):    {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    lines.append(f"- Precision / Recall / F1 (weighted): {p_w:.4f} / {r_w:.4f} / {f1_w:.4f}")
    lines.append("")

    report = classification_report(
        y_true_bin, y_pred_bin, target_names=classes, digits=4, zero_division=0
    )
    lines.append("Per-label classification report:")
    lines.append(report)

    return "\n".join(lines)


def _init_matplotlib(dpi: int) -> None:
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def plot_per_label_confusions(classes: List[str], y_true_bin: np.ndarray, 
                              y_pred_bin: np.ndarray, file_path: str, 
                              normalize: bool = True) -> None:
    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin, labels=range(len(classes)))

    n = len(classes)
    cols = min(4, max(1, int(round(n**0.5))))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (label, mat) in enumerate(zip(classes, mcm)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        if normalize:
            mat_disp = mat.astype(float)
            row_sums = mat_disp.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                mat_disp = np.divide(mat_disp, row_sums, out=np.zeros_like(mat_disp), 
                                    where=row_sums != 0) * 100.0
            fmt = ".1f"
            title_suffix = " (row %)"
        else:
            mat_disp = mat
            fmt = "d"
            title_suffix = ""

        sns.heatmap(mat_disp, annot=True, fmt=fmt, cbar=False, cmap="Blues", ax=ax,
                   xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(f"{label}{title_suffix}")
        ax.tick_params(axis="y", rotation=0)

    for extra in range(n, rows * cols):
        r, c = divmod(extra, cols)
        axes[r][c].axis("off")

    plt.suptitle("Per-label 2×2 Confusion Matrices" + 
                (" (rows sum to 100%)" if normalize else ""), y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(file_path)
    plt.close()


def plot_per_label_f1(classes: List[str], y_true_bin: np.ndarray, 
                     y_pred_bin: np.ndarray, file_path: str, 
                     top_k: int = 30) -> None:
    _, _, f1, support = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    order = sorted(range(len(classes)), key=lambda i: (f1[i], support[i]))
    if len(order) > top_k:
        order = order[:top_k]

    labels = [classes[i] for i in order]
    f1_vals = [float(f1[i]) for i in order]

    plt.figure(figsize=(max(8, 0.3 * len(labels)), 6))
    plt.bar(range(len(labels)), f1_vals)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("F1 score")
    plt.title(f"Per-label F1 scores (lowest {len(labels)})")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_multilabel_confusion_matrix(classes: List[str], y_true_bin: np.ndarray,
                                     y_pred_bin: np.ndarray, file_path: str,
                                     annotate: bool = False, 
                                     normalize: bool = True) -> None:
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    for true_vec, pred_vec in zip(y_true_bin, y_pred_bin):
        true_inds = np.where(true_vec == 1)[0]
        pred_inds = np.where(pred_vec == 1)[0]
        for i in true_inds:
            for j in pred_inds:
                cm[i, j] += 1

    if normalize:
        cm_disp = cm.astype(float)
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_disp = np.divide(cm_disp, row_sums, out=np.zeros_like(cm_disp),
                              where=row_sums != 0) * 100.0
        fmt = ".1f" if annotate else ""
        title_suffix = " (row %)"
    else:
        cm_disp = cm
        fmt = "d" if annotate else ""
        title_suffix = ""

    plt.figure(figsize=(max(8, 0.4 * n), max(6, 0.35 * n)))
    ax = sns.heatmap(cm_disp, annot=annotate, fmt=fmt, 
                    xticklabels=classes, yticklabels=classes, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Multilabel Confusion Matrix (all labels){title_suffix}")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def evaluate(cfg: EvalConfig) -> None:
    ensure_dir(cfg.output_dir)
    _init_matplotlib(cfg.fig_dpi)

    if not os.path.exists(cfg.result_file):
        return

    results = load_results(cfg.result_file)

    y_true_sets: List[Set[str]] = []
    y_pred_sets: List[Set[str]] = []

    for item in results:
        truth = normalize_labels(item.get(cfg.truth_field), lowercase=cfg.lowercase_tags)
        pred = normalize_labels(item.get(cfg.pred_field), lowercase=cfg.lowercase_tags)
        y_true_sets.append(truth)
        y_pred_sets.append(pred)

    _, classes, y_true_bin, y_pred_bin = to_binary(y_true_sets, y_pred_sets)

    plot_per_label_confusions(
        classes, y_true_bin, y_pred_bin,
        file_path=os.path.join(cfg.output_dir, "confusions_per_label.png"),
        normalize=cfg.normalize_confusions
    )
    plot_per_label_f1(
        classes, y_true_bin, y_pred_bin,
        file_path=os.path.join(cfg.output_dir, "per_label_f1.png"),
        top_k=cfg.per_label_f1_top_k
    )
    plot_multilabel_confusion_matrix(
        classes, y_true_bin, y_pred_bin,
        file_path=os.path.join(cfg.output_dir, "confusion_matrix_all_labels.png"),
        annotate=cfg.annotate_all_confusions,
        normalize=cfg.normalize_confusions
    )

    summary_text = build_summary_text(classes, y_true_bin, y_pred_bin)
    save_text(summary_text, os.path.join(cfg.output_dir, "eval_summary.txt"))
    save_per_label_csv(
        classes, y_true_bin, y_pred_bin,
        os.path.join(cfg.output_dir, "per_label_metrics.csv")
    )


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Multilabel evaluation for DSM-5 symptom classification")
    p.add_argument("--input", dest="result_file", default="inference_results.jsonl",
                  help="Path to JSONL with results")
    p.add_argument("--output", dest="output_dir", default="eval_outputs",
                  help="Output directory for reports/plots")
    p.add_argument("--truth-field", default="gold_all_symptoms",
                  help="JSON field for ground truth labels")
    p.add_argument("--pred-field", default="predicted_symptoms",
                  help="JSON field for predicted labels")
    p.add_argument("--lowercase", action="store_true",
                  help="Lowercase all tags during normalization")
    p.add_argument("--top-k", dest="per_label_f1_top_k", type=int, default=30,
                  help="How many lowest-F1 labels to plot")
    p.add_argument("--annotate-all", dest="annotate_all_confusions", action="store_true",
                  help="Annotate counts/values in the all-labels confusion heatmap")
    p.add_argument("--dpi", dest="fig_dpi", type=int, default=160,
                  help="Figure DPI for saved images")
    p.add_argument("--raw-confusions", dest="raw_confusions", action="store_true",
                  help="Disable normalization for confusion matrices (show raw counts)")
    
    args = p.parse_args()

    return EvalConfig(
        result_file=args.result_file,
        output_dir=args.output_dir,
        truth_field=args.truth_field,
        pred_field=args.pred_field,
        lowercase_tags=args.lowercase,
        per_label_f1_top_k=args.per_label_f1_top_k,
        fig_dpi=args.fig_dpi,
        annotate_all_confusions=args.annotate_all_confusions,
        normalize_confusions=not args.raw_confusions,
    )


def main() -> None:
    cfg = parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
