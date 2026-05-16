"""
Peripheral-frame experiments with classifier chains and calibrated thresholds.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
import torch

from climate_frames.dataset import DEFAULT_DATA_PATH, load_annotations
from climate_frames.paths import OUTPUTS_DIR

hf_logging.set_verbosity_error()

OUTPUT_DIR = str(OUTPUTS_DIR)
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    model.eval()
    arrays = []
    with torch.no_grad():
        for batch in batched(texts, batch_size):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(DEVICE)
            outputs = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            arrays.append(pooled.cpu().numpy())
    return np.vstack(arrays)


def prepare_targets(df):
    mlb = MultiLabelBinarizer()
    full_y = mlb.fit_transform(df["peripheral_frames"])
    supports = full_y.sum(axis=0)
    keep = supports >= 10
    y = full_y[:, keep]
    labels = [label for label, flag in zip(mlb.classes_, keep) if flag]
    support_map = {label: int(count) for label, count in zip(labels, supports[keep])}
    return y.astype(np.int32), labels, support_map


def evaluate_thresholds(true_y, prob, thresholds):
    rows = []
    for threshold in thresholds:
        pred = (prob >= threshold).astype(np.int32)
        rows.append(
            {
                "threshold": threshold,
                "micro_f1": float(f1_score(true_y, pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(true_y, pred, average="macro", zero_division=0)),
            }
        )
    return rows


def best_threshold(rows):
    return max(rows, key=lambda item: (item["macro_f1"], item["micro_f1"], -abs(item["threshold"] - 0.5)))


def ovr_probability_oof(texts, embeddings, y):
    n_samples, n_labels = y.shape
    tfidf_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    emb_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    fusion_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(n_samples)))

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])
        x_train_fusion = hstack([x_train_tfidf, csr_matrix(embeddings[train_idx])], format="csr")
        x_val_fusion = hstack([x_val_tfidf, csr_matrix(embeddings[val_idx])], format="csr")

        tfidf_clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED),
            n_jobs=-1,
        )
        emb_clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED),
            n_jobs=-1,
        )
        fusion_clf = OneVsRestClassifier(
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED),
            n_jobs=-1,
        )

        tfidf_clf.fit(x_train_tfidf, y[train_idx])
        emb_clf.fit(embeddings[train_idx], y[train_idx])
        fusion_clf.fit(x_train_fusion, y[train_idx])
        tfidf_prob[val_idx] = tfidf_clf.predict_proba(x_val_tfidf)
        emb_prob[val_idx] = emb_clf.predict_proba(embeddings[val_idx])
        fusion_prob[val_idx] = fusion_clf.predict_proba(x_val_fusion)

    return tfidf_prob, emb_prob, fusion_prob


def classifier_chain_oof(texts, embeddings, y, order):
    n_samples, n_labels = y.shape
    tfidf_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    fusion_prob = np.zeros((n_samples, n_labels), dtype=np.float32)
    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(n_samples)))

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        x_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_val_tfidf = vectorizer.transform([texts[i] for i in val_idx])
        x_train_fusion = hstack([x_train_tfidf, csr_matrix(embeddings[train_idx])], format="csr")
        x_val_fusion = hstack([x_val_tfidf, csr_matrix(embeddings[val_idx])], format="csr")

        base = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED)
        tfidf_chain = ClassifierChain(estimator=base, order=order)
        fusion_chain = ClassifierChain(estimator=base, order=order)

        tfidf_chain.fit(x_train_tfidf, y[train_idx])
        fusion_chain.fit(x_train_fusion, y[train_idx])
        tfidf_prob[val_idx] = tfidf_chain.predict_proba(x_val_tfidf)
        fusion_prob[val_idx] = fusion_chain.predict_proba(x_val_fusion)

    return tfidf_prob, fusion_prob


def plot_thresholds(model_rows: dict[str, list[dict]]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    palette = {
        "OVR TF-IDF": "#4c78a8",
        "OVR ClimateBERT": "#59a14f",
        "OVR Fusion": "#f58518",
        "Chain TF-IDF": "#e15759",
        "Chain Fusion": "#b07aa1",
    }
    for name, rows in model_rows.items():
        thresholds = [item["threshold"] for item in rows]
        axes[0].plot(thresholds, [item["micro_f1"] for item in rows], marker="o", label=name, color=palette[name])
        axes[1].plot(thresholds, [item["macro_f1"] for item in rows], marker="o", label=name, color=palette[name])
    axes[0].set_title("Peripheral Micro-F1")
    axes[1].set_title("Peripheral Macro-F1")
    axes[0].set_xlabel("Threshold")
    axes[1].set_xlabel("Threshold")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "peripheral_chain_thresholds.png"), dpi=160, bbox_inches="tight")
    plt.close()


def save_results(model_rows, support_map, best_model_name, best_row, best_label_scores):
    out_path = os.path.join(OUTPUT_DIR, "peripheral_chain_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Peripheral Chain Experiments\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n\n")

        handle.write("Threshold sweep summary\n")
        handle.write("-" * 60 + "\n")
        for name, rows in model_rows.items():
            default = next(item for item in rows if abs(item["threshold"] - 0.50) < 1e-9)
            best = best_threshold(rows)
            handle.write(
                f"{name:15s} "
                f"default@0.50 micro={default['micro_f1']:.4f} macro={default['macro_f1']:.4f} | "
                f"best@{best['threshold']:.2f} micro={best['micro_f1']:.4f} macro={best['macro_f1']:.4f}\n"
            )

        handle.write("\nBest overall setting\n")
        handle.write("-" * 60 + "\n")
        handle.write(
            f"{best_model_name} @ {best_row['threshold']:.2f} "
            f"micro={best_row['micro_f1']:.4f} macro={best_row['macro_f1']:.4f}\n"
        )

        handle.write("\nBest label-wise F1 (support >= 20)\n")
        handle.write("-" * 60 + "\n")
        for label, support, score in best_label_scores:
            handle.write(f"{label:45s} support={support:3d} f1={score:.3f}\n")

        handle.write("\nSupports retained\n")
        handle.write("-" * 60 + "\n")
        for label, support in sorted(support_map.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{label:45s} {support}\n")

    return out_path


def main():
    print("=" * 60)
    print("  Peripheral Chain Experiments")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    texts = df["text"].tolist()
    y, labels, support_map = prepare_targets(df)
    print(f"  Loaded {len(df)} merged texts and {len(labels)} retained peripheral labels")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    embeddings = embed_texts(texts, tokenizer, encoder)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tfidf_prob, emb_prob, fusion_prob = ovr_probability_oof(texts, embeddings, y)
    order = np.argsort(-y.sum(axis=0))
    chain_tfidf_prob, chain_fusion_prob = classifier_chain_oof(texts, embeddings, y, order=order)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    model_rows = {
        "OVR TF-IDF": evaluate_thresholds(y, tfidf_prob, thresholds),
        "OVR ClimateBERT": evaluate_thresholds(y, emb_prob, thresholds),
        "OVR Fusion": evaluate_thresholds(y, fusion_prob, thresholds),
        "Chain TF-IDF": evaluate_thresholds(y, chain_tfidf_prob, thresholds),
        "Chain Fusion": evaluate_thresholds(y, chain_fusion_prob, thresholds),
    }

    best_model_name, best_row = max(
        ((name, best_threshold(rows)) for name, rows in model_rows.items()),
        key=lambda item: (item[1]["macro_f1"], item[1]["micro_f1"]),
    )

    best_prob_lookup = {
        "OVR TF-IDF": tfidf_prob,
        "OVR ClimateBERT": emb_prob,
        "OVR Fusion": fusion_prob,
        "Chain TF-IDF": chain_tfidf_prob,
        "Chain Fusion": chain_fusion_prob,
    }
    best_pred = (best_prob_lookup[best_model_name] >= best_row["threshold"]).astype(np.int32)
    label_f1 = f1_score(y, best_pred, average=None, zero_division=0)
    best_label_scores = []
    for label, support, score in zip(labels, [support_map[label] for label in labels], label_f1):
        if support >= 20:
            best_label_scores.append((label, support, float(score)))
    best_label_scores.sort(key=lambda item: (-item[2], -item[1], item[0]))

    plot_thresholds(model_rows)
    out_path = save_results(model_rows, support_map, best_model_name, best_row, best_label_scores[:20])

    print(f"  Best overall: {best_model_name} @ {best_row['threshold']:.2f} macro-F1={best_row['macro_f1']:.3f}")
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
