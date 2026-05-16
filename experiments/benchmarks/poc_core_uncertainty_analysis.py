"""
Uncertainty and paired-bootstrap analysis for the strongest merged-CV core models.

This script keeps the focus on the leading classical / fusion baselines that can
be compared on identical out-of-fold predictions:

  1. TF-IDF + LogisticRegression
  2. Frozen ClimateBERT embeddings + LogisticRegression
  3. TF-IDF + ClimateBERT fusion LogisticRegression
  4. Soft-vote ensemble

It adds:
  - bootstrap confidence intervals for accuracy and macro-F1
  - paired bootstrap deltas between the top models
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

from climate_frames.dataset import DEFAULT_DATA_PATH, load_annotations
from climate_frames.paths import OUTPUTS_DIR

hf_logging.set_verbosity_error()

OUTPUT_DIR = str(OUTPUTS_DIR)
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 5
BOOTSTRAPS = 4000

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OOFResult:
    name: str
    true: np.ndarray
    pred: np.ndarray

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.true, self.pred))

    @property
    def macro_f1(self) -> float:
        return float(f1_score(self.true, self.pred, average="macro", zero_division=0))


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    arrays = []
    model.eval()
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
            arrays.append(pooled.cpu().numpy())
    return np.vstack(arrays)


def build_splits(labels: np.ndarray):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    idx = np.arange(len(labels))
    return list(skf.split(idx, labels))


def run_tfidf(texts, labels, splits) -> OOFResult:
    pred = np.zeros_like(labels)
    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val = vectorizer.transform([texts[i] for i in val_idx])
        clf = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED)
        clf.fit(x_train, labels[train_idx])
        pred[val_idx] = clf.predict(x_val)
    return OOFResult("TF-IDF LR", labels.copy(), pred)


def run_emb_lr(embeddings, labels, splits) -> OOFResult:
    pred = np.zeros_like(labels)
    for train_idx, val_idx in splits:
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED)
        clf.fit(embeddings[train_idx], labels[train_idx])
        pred[val_idx] = clf.predict(embeddings[val_idx])
    return OOFResult("Frozen ClimateBERT LR", labels.copy(), pred)


def run_fusion(texts, embeddings, labels, splits) -> OOFResult:
    pred = np.zeros_like(labels)
    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])
        x_train = hstack([x_train_tfidf, csr_matrix(embeddings[train_idx])], format="csr")
        x_val = hstack([x_val_tfidf, csr_matrix(embeddings[val_idx])], format="csr")
        clf = LogisticRegression(max_iter=3500, class_weight="balanced", solver="lbfgs", random_state=SEED)
        clf.fit(x_train, labels[train_idx])
        pred[val_idx] = clf.predict(x_val)
    return OOFResult("Fusion LR", labels.copy(), pred)


def run_soft_vote(texts, embeddings, labels, splits) -> OOFResult:
    pred = np.zeros_like(labels)
    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])
        tfidf_lr = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED)
        emb_lr = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED)
        tfidf_lr.fit(x_train_tfidf, labels[train_idx])
        emb_lr.fit(embeddings[train_idx], labels[train_idx])
        prob = 0.5 * tfidf_lr.predict_proba(x_val_tfidf) + 0.5 * emb_lr.predict_proba(embeddings[val_idx])
        pred[val_idx] = prob.argmax(axis=1)
    return OOFResult("Soft-Vote Ensemble", labels.copy(), pred)


def bootstrap_metric(true: np.ndarray, pred: np.ndarray, metric_name: str, n_boot: int = BOOTSTRAPS):
    rng = np.random.default_rng(SEED)
    n = len(true)
    values = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        if metric_name == "accuracy":
            values[idx] = accuracy_score(true[sample], pred[sample])
        elif metric_name == "macro_f1":
            values[idx] = f1_score(true[sample], pred[sample], average="macro", zero_division=0)
        else:
            raise ValueError(metric_name)
    return {
        "mean": float(values.mean()),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
        "values": values,
    }


def paired_bootstrap_delta(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, metric_name: str, n_boot: int = BOOTSTRAPS):
    rng = np.random.default_rng(SEED + 1)
    n = len(true)
    deltas = np.zeros(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        if metric_name == "accuracy":
            score_a = accuracy_score(true[sample], pred_a[sample])
            score_b = accuracy_score(true[sample], pred_b[sample])
        elif metric_name == "macro_f1":
            score_a = f1_score(true[sample], pred_a[sample], average="macro", zero_division=0)
            score_b = f1_score(true[sample], pred_b[sample], average="macro", zero_division=0)
        else:
            raise ValueError(metric_name)
        deltas[idx] = score_a - score_b

    p_two_sided = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return {
        "mean_delta": float(deltas.mean()),
        "low": float(np.percentile(deltas, 2.5)),
        "high": float(np.percentile(deltas, 97.5)),
        "p_value": min(p_two_sided, 1.0),
    }


def save_plot(results: list[OOFResult], macro_stats: dict[str, dict[str, float]]):
    order = np.arange(len(results))
    centers = [macro_stats[item.name]["mean"] for item in results]
    lower = [macro_stats[item.name]["mean"] - macro_stats[item.name]["low"] for item in results]
    upper = [macro_stats[item.name]["high"] - macro_stats[item.name]["mean"] for item in results]

    plt.figure(figsize=(9, 5))
    plt.errorbar(order, centers, yerr=[lower, upper], fmt="o", color="#cc4125", capsize=5, linewidth=2)
    plt.xticks(order, [item.name for item in results], rotation=15, ha="right")
    plt.ylabel("Macro-F1")
    plt.ylim(0.30, 0.60)
    plt.title("Bootstrap 95% CI for Core Macro-F1")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "core_macro_f1_bootstrap.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main():
    print("=" * 60)
    print("  Core Uncertainty Analysis")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    texts = df["text"].tolist()
    label_names = sorted(df["core_frame"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(label_names)}
    labels = np.array([label2id[label] for label in df["core_frame"]], dtype=np.int64)
    splits = build_splits(labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    embeddings = embed_texts(texts, tokenizer, encoder)

    results = [
        run_tfidf(texts, labels, splits),
        run_emb_lr(embeddings, labels, splits),
        run_fusion(texts, embeddings, labels, splits),
        run_soft_vote(texts, embeddings, labels, splits),
    ]

    acc_stats = {item.name: bootstrap_metric(item.true, item.pred, "accuracy") for item in results}
    macro_stats = {item.name: bootstrap_metric(item.true, item.pred, "macro_f1") for item in results}

    paired = {
        "Fusion LR vs Frozen ClimateBERT LR": paired_bootstrap_delta(results[2].true, results[2].pred, results[1].pred, "macro_f1"),
        "Soft-Vote Ensemble vs Frozen ClimateBERT LR": paired_bootstrap_delta(results[3].true, results[3].pred, results[1].pred, "macro_f1"),
        "Soft-Vote Ensemble vs Fusion LR": paired_bootstrap_delta(results[3].true, results[3].pred, results[2].pred, "macro_f1"),
    }

    plot_path = save_plot(results, macro_stats)
    out_path = os.path.join(OUTPUT_DIR, "core_uncertainty_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Core Uncertainty and Paired Bootstrap Analysis\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Rows evaluated: {len(labels)}\n")
        handle.write(f"Bootstrap resamples: {BOOTSTRAPS}\n\n")

        for item in results:
            handle.write(f"-- {item.name} --\n")
            handle.write(f"Accuracy: {item.accuracy:.4f}  95% CI [{acc_stats[item.name]['low']:.4f}, {acc_stats[item.name]['high']:.4f}]\n")
            handle.write(f"Macro-F1: {item.macro_f1:.4f}  95% CI [{macro_stats[item.name]['low']:.4f}, {macro_stats[item.name]['high']:.4f}]\n\n")

        handle.write("Paired Bootstrap Delta on Macro-F1\n")
        handle.write("-" * 60 + "\n")
        for name, stats in paired.items():
            handle.write(
                f"{name}: delta={stats['mean_delta']:+.4f} "
                f"95% CI [{stats['low']:+.4f}, {stats['high']:+.4f}] "
                f"approx_p={stats['p_value']:.4f}\n"
            )

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    for item in results:
        print(f"  {item.name}: acc={item.accuracy:.3f} macro-F1={item.macro_f1:.3f}")
    print(f"\n  Results -> {out_path}")
    print(f"  Plot -> {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
