"""
Additional experiments for the Climate Frames project.

This script focuses on paper-value additions that do not require new annotation:
  1. Stronger core-frame baselines and ensembles
  2. Threshold tuning for peripheral multi-label prediction
  3. Best-model error analysis for core-frame classification
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

from climate_frames_dataset import DEFAULT_DATA_PATH, load_annotations

hf_logging.set_verbosity_error()

OUTPUT_DIR = r"E:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
CORE_SPLITS = 5
PERIPHERAL_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CoreResult:
    name: str
    true: np.ndarray
    pred: np.ndarray
    fold_accs: list[float]
    fold_f1s: list[float]

    @property
    def mean_acc(self) -> float:
        return float(np.mean(self.fold_accs))

    @property
    def std_acc(self) -> float:
        return float(np.std(self.fold_accs))

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean(self.fold_f1s))

    @property
    def std_macro_f1(self) -> float:
        return float(np.std(self.fold_f1s))


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in batched(texts, batch_size):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(DEVICE)
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.cpu().numpy())
    return np.vstack(outputs)


def core_splits(labels: np.ndarray):
    skf = StratifiedKFold(n_splits=CORE_SPLITS, shuffle=True, random_state=SEED)
    idx = np.arange(len(labels))
    return list(skf.split(idx, labels))


def run_core_tfidf(texts: list[str], labels: np.ndarray, splits) -> CoreResult:
    oof_pred = np.zeros_like(labels)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val = vectorizer.transform([texts[i] for i in val_idx])

        clf = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(x_train, labels[train_idx])
        pred = clf.predict(x_val)
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return CoreResult("TF-IDF LR", labels.copy(), oof_pred, fold_accs, fold_f1s)


def run_core_embedding_lr(embeddings: np.ndarray, labels: np.ndarray, splits) -> tuple[CoreResult, np.ndarray]:
    oof_pred = np.zeros_like(labels)
    oof_prob = np.zeros((len(labels), len(np.unique(labels))), dtype=np.float32)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(embeddings[train_idx], labels[train_idx])
        pred = clf.predict(embeddings[val_idx])
        prob = clf.predict_proba(embeddings[val_idx])
        oof_pred[val_idx] = pred
        oof_prob[val_idx] = prob
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return CoreResult("Frozen ClimateBERT LR", labels.copy(), oof_pred, fold_accs, fold_f1s), oof_prob


def run_core_embedding_svc(embeddings: np.ndarray, labels: np.ndarray, splits) -> CoreResult:
    oof_pred = np.zeros_like(labels)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED)
        clf.fit(embeddings[train_idx], labels[train_idx])
        pred = clf.predict(embeddings[val_idx])
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return CoreResult("Frozen ClimateBERT LinearSVC", labels.copy(), oof_pred, fold_accs, fold_f1s)


def run_core_fusion_lr(texts: list[str], embeddings: np.ndarray, labels: np.ndarray, splits) -> CoreResult:
    oof_pred = np.zeros_like(labels)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])
        x_train = hstack([x_train_tfidf, csr_matrix(embeddings[train_idx])], format="csr")
        x_val = hstack([x_val_tfidf, csr_matrix(embeddings[val_idx])], format="csr")

        clf = LogisticRegression(
            max_iter=3500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(x_train, labels[train_idx])
        pred = clf.predict(x_val)
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return CoreResult("TF-IDF + ClimateBERT Fusion LR", labels.copy(), oof_pred, fold_accs, fold_f1s)


def run_core_probability_ensemble(texts: list[str], embeddings: np.ndarray, labels: np.ndarray, splits):
    n_classes = len(np.unique(labels))
    oof_pred = np.zeros_like(labels)
    oof_prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])

        tfidf_lr = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        emb_lr = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        tfidf_lr.fit(x_train_tfidf, labels[train_idx])
        emb_lr.fit(embeddings[train_idx], labels[train_idx])

        prob = 0.5 * tfidf_lr.predict_proba(x_val_tfidf) + 0.5 * emb_lr.predict_proba(embeddings[val_idx])
        pred = prob.argmax(axis=1)
        oof_prob[val_idx] = prob
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return CoreResult("Soft-Vote Ensemble", labels.copy(), oof_pred, fold_accs, fold_f1s), oof_prob


def prepare_peripheral_targets(df):
    mlb = MultiLabelBinarizer()
    full_y = mlb.fit_transform(df["peripheral_frames"])
    supports = full_y.sum(axis=0)
    keep = supports >= 10
    y = full_y[:, keep]
    labels = [label for label, flag in zip(mlb.classes_, keep) if flag]
    support_map = {label: int(count) for label, count in zip(labels, supports[keep])}
    return y, labels, support_map


def collect_peripheral_probabilities(texts, embeddings, y):
    n_samples, n_labels = y.shape
    tfidf_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    emb_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    splits = list(KFold(n_splits=PERIPHERAL_SPLITS, shuffle=True, random_state=SEED).split(np.arange(n_samples)))

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])

        tfidf_clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SEED,
            ),
            n_jobs=-1,
        )
        emb_clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SEED,
            ),
            n_jobs=-1,
        )

        tfidf_clf.fit(x_train_tfidf, y[train_idx])
        emb_clf.fit(embeddings[train_idx], y[train_idx])
        tfidf_prob[val_idx] = tfidf_clf.predict_proba(x_val_tfidf)
        emb_prob[val_idx] = emb_clf.predict_proba(embeddings[val_idx])

    return tfidf_prob, emb_prob


def evaluate_thresholds(true_y: np.ndarray, prob: np.ndarray, thresholds: list[float]):
    rows = []
    for threshold in thresholds:
        pred = (prob >= threshold).astype(np.int32)
        micro = f1_score(true_y, pred, average="micro", zero_division=0)
        macro = f1_score(true_y, pred, average="macro", zero_division=0)
        rows.append({"threshold": threshold, "micro_f1": float(micro), "macro_f1": float(macro)})
    return rows


def best_threshold(rows: list[dict], metric: str = "macro_f1") -> dict:
    return max(rows, key=lambda item: (item[metric], item["micro_f1"], -abs(item["threshold"] - 0.5)))


def plot_core_comparison(results: list[CoreResult]):
    names = [item.name for item in results]
    macro = [item.mean_macro_f1 for item in results]
    acc = [item.mean_acc for item in results]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, macro, width, label="Macro-F1", color="#4c78a8")
    ax.bar(x + width / 2, acc, width, label="Accuracy", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Additional Core Classification Experiments")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "additional_core_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_peripheral_thresholds(threshold_rows_by_model: dict[str, list[dict]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    palette = {"TF-IDF": "#4c78a8", "ClimateBERT": "#59a14f", "Ensemble": "#e15759"}

    for model_name, rows in threshold_rows_by_model.items():
        thresholds = [item["threshold"] for item in rows]
        micro = [item["micro_f1"] for item in rows]
        macro = [item["macro_f1"] for item in rows]
        axes[0].plot(thresholds, micro, marker="o", label=model_name, color=palette[model_name])
        axes[1].plot(thresholds, macro, marker="o", label=model_name, color=palette[model_name])

    axes[0].set_title("Peripheral Micro-F1 by Threshold")
    axes[1].set_title("Peripheral Macro-F1 by Threshold")
    axes[0].set_ylabel("F1")
    axes[0].set_xlabel("Decision threshold")
    axes[1].set_xlabel("Decision threshold")
    axes[0].set_ylim(0.3, 0.7)
    axes[1].set_ylim(0.3, 0.6)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "peripheral_threshold_sweep.png"), dpi=160, bbox_inches="tight")
    plt.close()


def write_core_error_analysis(best_result: CoreResult, id2label: dict[int, str], texts: list[str]):
    report = classification_report(
        best_result.true,
        best_result.pred,
        labels=sorted(id2label),
        target_names=[id2label[idx] for idx in sorted(id2label)],
        output_dict=True,
        zero_division=0,
    )

    confusion_pairs = {}
    for idx, (true_id, pred_id) in enumerate(zip(best_result.true, best_result.pred)):
        if true_id == pred_id:
            continue
        key = (true_id, pred_id)
        confusion_pairs.setdefault(key, []).append(idx)

    ranked_pairs = sorted(confusion_pairs.items(), key=lambda item: (-len(item[1]), item[0]))
    out_path = os.path.join(OUTPUT_DIR, "additional_core_error_analysis.txt")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Additional Core Error Analysis\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Model: {best_result.name}\n")
        handle.write(
            f"Mean accuracy: {best_result.mean_acc:.4f} +/- {best_result.std_acc:.4f}\n"
            f"Mean macro-F1: {best_result.mean_macro_f1:.4f} +/- {best_result.std_macro_f1:.4f}\n\n"
        )

        handle.write("Per-class recall and F1\n")
        handle.write("-" * 60 + "\n")
        rows = []
        for class_name in [id2label[idx] for idx in sorted(id2label)]:
            rows.append((class_name, report[class_name]["recall"], report[class_name]["f1-score"], int(report[class_name]["support"])))
        rows.sort(key=lambda item: (item[2], item[1], item[0]))
        for class_name, recall, f1_value, support in rows:
            handle.write(f"{class_name:40s} recall={recall:.3f} f1={f1_value:.3f} support={support}\n")

        handle.write("\nTop confusion pairs\n")
        handle.write("-" * 60 + "\n")
        for (true_id, pred_id), indices in ranked_pairs[:10]:
            handle.write(
                f"{id2label[true_id]:35s} -> {id2label[pred_id]:35s} "
                f"count={len(indices)}\n"
            )

        handle.write("\nSample errors\n")
        handle.write("-" * 60 + "\n")
        for (true_id, pred_id), indices in ranked_pairs[:5]:
            handle.write(
                f"\n{ id2label[true_id] } -> { id2label[pred_id] } "
                f"(count={len(indices)})\n"
            )
            for sample_idx in indices[:3]:
                text = texts[sample_idx].replace("\n", " ")
                handle.write(f"  [{sample_idx}] {text[:420]}\n")

    return out_path


def save_results(
    core_results: list[CoreResult],
    peripheral_summary: dict,
    support_map: dict[str, int],
    best_peripheral_label_scores: list[tuple[str, int, float]],
):
    out_path = os.path.join(OUTPUT_DIR, "additional_experiments_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Additional Experiments\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Backbone model: {MODEL_NAME}\n\n")

        handle.write("Core classification\n")
        handle.write("-" * 60 + "\n")
        for result in core_results:
            handle.write(
                f"{result.name:30s} "
                f"acc={result.mean_acc:.4f} +/- {result.std_acc:.4f} "
                f"macro_f1={result.mean_macro_f1:.4f} +/- {result.std_macro_f1:.4f}\n"
            )

        handle.write("\nPeripheral threshold sweeps\n")
        handle.write("-" * 60 + "\n")
        for model_name, summary in peripheral_summary.items():
            default_row = summary["default"]
            best_row = summary["best_macro"]
            handle.write(
                f"{model_name:12s} "
                f"default@0.50 micro={default_row['micro_f1']:.4f} macro={default_row['macro_f1']:.4f} | "
                f"best@{best_row['threshold']:.2f} micro={best_row['micro_f1']:.4f} macro={best_row['macro_f1']:.4f}\n"
            )

        handle.write("\nBest peripheral label-wise F1 (support >= 20)\n")
        handle.write("-" * 60 + "\n")
        for label, support, score in best_peripheral_label_scores:
            handle.write(f"{label:45s} support={support:3d} f1={score:.3f}\n")

        handle.write("\nPeripheral supports retained\n")
        handle.write("-" * 60 + "\n")
        for label, support in sorted(support_map.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{label:45s} {support}\n")

    return out_path


def main():
    print("=" * 60)
    print("  Additional Experiments")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    texts = df["text"].tolist()
    frames = sorted(df["core_frame"].unique())
    label2id = {frame: idx for idx, frame in enumerate(frames)}
    id2label = {idx: frame for frame, idx in label2id.items()}
    core_y = np.array([label2id[label] for label in df["core_frame"]], dtype=np.int64)
    splits = core_splits(core_y)

    print(f"  Loaded {len(df)} merged texts")
    print("  Encoding texts with ClimateBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    embeddings = embed_texts(texts, tokenizer, encoder)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    core_results = []
    tfidf_result = run_core_tfidf(texts, core_y, splits)
    core_results.append(tfidf_result)
    print(f"  {tfidf_result.name}: acc={tfidf_result.mean_acc:.3f} macro-F1={tfidf_result.mean_macro_f1:.3f}")

    emb_lr_result, emb_lr_prob = run_core_embedding_lr(embeddings, core_y, splits)
    core_results.append(emb_lr_result)
    print(f"  {emb_lr_result.name}: acc={emb_lr_result.mean_acc:.3f} macro-F1={emb_lr_result.mean_macro_f1:.3f}")

    emb_svc_result = run_core_embedding_svc(embeddings, core_y, splits)
    core_results.append(emb_svc_result)
    print(f"  {emb_svc_result.name}: acc={emb_svc_result.mean_acc:.3f} macro-F1={emb_svc_result.mean_macro_f1:.3f}")

    fusion_result = run_core_fusion_lr(texts, embeddings, core_y, splits)
    core_results.append(fusion_result)
    print(f"  {fusion_result.name}: acc={fusion_result.mean_acc:.3f} macro-F1={fusion_result.mean_macro_f1:.3f}")

    ensemble_result, ensemble_prob = run_core_probability_ensemble(texts, embeddings, core_y, splits)
    core_results.append(ensemble_result)
    print(f"  {ensemble_result.name}: acc={ensemble_result.mean_acc:.3f} macro-F1={ensemble_result.mean_macro_f1:.3f}")

    best_core = max(core_results, key=lambda item: (item.mean_macro_f1, item.mean_acc))
    error_path = write_core_error_analysis(best_core, id2label, texts)
    plot_core_comparison(core_results)

    peripheral_y, peripheral_labels, support_map = prepare_peripheral_targets(df)
    tfidf_prob, peripheral_emb_prob = collect_peripheral_probabilities(texts, embeddings, peripheral_y)
    peripheral_ensemble_prob = 0.5 * tfidf_prob + 0.5 * peripheral_emb_prob

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_rows_by_model = {
        "TF-IDF": evaluate_thresholds(peripheral_y, tfidf_prob, thresholds),
        "ClimateBERT": evaluate_thresholds(peripheral_y, peripheral_emb_prob, thresholds),
        "Ensemble": evaluate_thresholds(peripheral_y, peripheral_ensemble_prob, thresholds),
    }

    peripheral_summary = {}
    prob_lookup = {"TF-IDF": tfidf_prob, "ClimateBERT": peripheral_emb_prob, "Ensemble": peripheral_ensemble_prob}
    for model_name, rows in threshold_rows_by_model.items():
        default_row = next(item for item in rows if abs(item["threshold"] - 0.50) < 1e-9)
        best_row = best_threshold(rows, metric="macro_f1")
        peripheral_summary[model_name] = {"default": default_row, "best_macro": best_row}

    best_peripheral_model_name, best_peripheral_model_summary = max(
        peripheral_summary.items(),
        key=lambda item: (item[1]["best_macro"]["macro_f1"], item[1]["best_macro"]["micro_f1"]),
    )
    best_peripheral_prob = prob_lookup[best_peripheral_model_name]
    best_peripheral_threshold = best_peripheral_model_summary["best_macro"]["threshold"]
    best_peripheral_pred = (best_peripheral_prob >= best_peripheral_threshold).astype(np.int32)
    label_f1 = f1_score(peripheral_y, best_peripheral_pred, average=None, zero_division=0)
    label_scores = []
    for label, score in zip(peripheral_labels, label_f1):
        support = support_map[label]
        if support >= 20:
            label_scores.append((label, support, float(score)))
    label_scores.sort(key=lambda item: (-item[2], -item[1], item[0]))

    plot_peripheral_thresholds(threshold_rows_by_model)
    results_path = save_results(core_results, peripheral_summary, support_map, label_scores[:20])

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    for result in sorted(core_results, key=lambda item: (-item.mean_macro_f1, -item.mean_acc)):
        print(
            f"  {result.name:30s} "
            f"acc={result.mean_acc:.3f} "
            f"macro-F1={result.mean_macro_f1:.3f}"
        )
    print(
        f"  Best peripheral setting: {best_peripheral_model_name} @ {best_peripheral_threshold:.2f} "
        f"macro-F1={best_peripheral_model_summary['best_macro']['macro_f1']:.3f}"
    )
    print(f"\n  Results -> {results_path}")
    print(f"  Error analysis -> {error_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
