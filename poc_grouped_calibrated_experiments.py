"""
Grouped and calibrated core-frame experiments.

This script does three things:
  1. Infer contiguous article-like blocks from the merged workbook order
  2. Evaluate core classifiers with leave-one-block-out validation
  3. Report calibration and selective prediction for the strongest merged-CV models
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
import torch

from climate_frames_dataset import DEFAULT_DATA_PATH, load_annotations

hf_logging.set_verbosity_error()

OUTPUT_DIR = r"E:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
MERGED_SPLITS = 5
EXPECTED_ARTICLE_BLOCKS = 12
MIN_BLOCK_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ProbResult:
    name: str
    true: np.ndarray
    pred: np.ndarray
    prob: np.ndarray
    fold_accs: list[float]
    fold_f1s: list[float]

    @property
    def mean_acc(self) -> float:
        return float(np.mean(self.fold_accs))

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean(self.fold_f1s))


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def infer_article_blocks(texts: list[str], embeddings: np.ndarray, n_blocks: int = EXPECTED_ARTICLE_BLOCKS, min_block: int = MIN_BLOCK_SIZE):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    x = tfidf.fit_transform(texts)
    x = x.tocsr()

    scores = []
    for idx in range(len(texts) - 1):
        emb_sim = cosine(embeddings[idx], embeddings[idx + 1])
        tfidf_sim = float(x[idx].multiply(x[idx + 1]).sum())
        scores.append((idx, 0.5 * emb_sim + 0.5 * tfidf_sim))

    order = sorted(scores, key=lambda item: (item[1], item[0]))
    cuts = []
    for idx, score in order:
        left = idx + 1
        if left < min_block or len(texts) - left < min_block:
            continue
        boundaries = [0] + sorted(cuts) + [len(texts)]
        valid = True
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start < left < end:
                if (left - start) < min_block or (end - left) < min_block:
                    valid = False
                break
        if not valid:
            continue
        cuts.append(left)
        if len(cuts) == n_blocks - 1:
            break

    cuts = sorted(cuts)
    groups = np.zeros(len(texts), dtype=np.int32)
    start = 0
    block_rows = []
    for group_id, end in enumerate(cuts + [len(texts)]):
        groups[start:end] = group_id
        block_rows.append(
            {
                "group_id": group_id,
                "start": start,
                "end": end - 1,
                "size": end - start,
                "cut_score_before": None if group_id == 0 else dict(scores)[start - 1],
            }
        )
        start = end
    return groups, cuts, block_rows, scores


def merged_splits(labels: np.ndarray):
    skf = StratifiedKFold(n_splits=MERGED_SPLITS, shuffle=True, random_state=SEED)
    idx = np.arange(len(labels))
    return list(skf.split(idx, labels))


def group_splits(groups: np.ndarray):
    logo = LeaveOneGroupOut()
    idx = np.arange(len(groups))
    return list(logo.split(idx, groups=groups))


def fit_tfidf_lr(train_texts, train_y):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    x_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=SEED)
    clf.fit(x_train, train_y)
    return vectorizer, clf


def fit_emb_lr(train_x, train_y):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    clf.fit(train_x, train_y)
    return clf


def fit_fusion_lr(train_texts, train_emb, train_y):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    x_tfidf = vectorizer.fit_transform(train_texts)
    x = hstack([x_tfidf, csr_matrix(train_emb)], format="csr")
    clf = LogisticRegression(max_iter=3500, class_weight="balanced", solver="lbfgs", random_state=SEED)
    clf.fit(x, train_y)
    return vectorizer, clf


def evaluate_soft_vote(texts, embeddings, labels, splits) -> ProbResult:
    n_classes = len(np.unique(labels))
    oof_pred = np.zeros_like(labels)
    oof_prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer, tfidf_lr = fit_tfidf_lr([texts[i] for i in train_idx], labels[train_idx])
        emb_lr = fit_emb_lr(embeddings[train_idx], labels[train_idx])
        tfidf_prob = tfidf_lr.predict_proba(vectorizer.transform([texts[i] for i in val_idx]))
        emb_prob = emb_lr.predict_proba(embeddings[val_idx])
        prob = 0.5 * tfidf_prob + 0.5 * emb_prob
        pred = prob.argmax(axis=1)
        oof_prob[val_idx] = prob
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return ProbResult("Soft-Vote Ensemble", labels.copy(), oof_pred, oof_prob, fold_accs, fold_f1s)


def evaluate_fusion(texts, embeddings, labels, splits) -> ProbResult:
    n_classes = len(np.unique(labels))
    oof_pred = np.zeros_like(labels)
    oof_prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer, clf = fit_fusion_lr([texts[i] for i in train_idx], embeddings[train_idx], labels[train_idx])
        x_val = hstack(
            [vectorizer.transform([texts[i] for i in val_idx]), csr_matrix(embeddings[val_idx])],
            format="csr",
        )
        prob = clf.predict_proba(x_val)
        pred = prob.argmax(axis=1)
        oof_prob[val_idx] = prob
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return ProbResult("Fusion LR", labels.copy(), oof_pred, oof_prob, fold_accs, fold_f1s)


def evaluate_stacked(texts, embeddings, labels, outer_splits) -> ProbResult:
    n_classes = len(np.unique(labels))
    oof_pred = np.zeros_like(labels)
    oof_prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in outer_splits:
        train_y = labels[train_idx]
        inner_splits = merged_splits(train_y)
        base_oof = np.zeros((len(train_idx), n_classes * 2), dtype=np.float32)
        train_texts = [texts[i] for i in train_idx]
        train_emb = embeddings[train_idx]

        for inner_train_pos, inner_val_pos in inner_splits:
            vectorizer, tfidf_lr = fit_tfidf_lr([train_texts[i] for i in inner_train_pos], train_y[inner_train_pos])
            emb_lr = fit_emb_lr(train_emb[inner_train_pos], train_y[inner_train_pos])
            tfidf_prob = tfidf_lr.predict_proba(vectorizer.transform([train_texts[i] for i in inner_val_pos]))
            emb_prob = emb_lr.predict_proba(train_emb[inner_val_pos])
            base_oof[inner_val_pos] = np.hstack([tfidf_prob, emb_prob])

        meta = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
        meta.fit(base_oof, train_y)

        vectorizer, tfidf_lr = fit_tfidf_lr(train_texts, train_y)
        emb_lr = fit_emb_lr(train_emb, train_y)
        tfidf_prob_val = tfidf_lr.predict_proba(vectorizer.transform([texts[i] for i in val_idx]))
        emb_prob_val = emb_lr.predict_proba(embeddings[val_idx])
        base_val = np.hstack([tfidf_prob_val, emb_prob_val])
        prob = meta.predict_proba(base_val)
        pred = prob.argmax(axis=1)

        oof_prob[val_idx] = prob
        oof_pred[val_idx] = pred
        fold_accs.append(accuracy_score(labels[val_idx], pred))
        fold_f1s.append(f1_score(labels[val_idx], pred, average="macro", zero_division=0))

    return ProbResult("Stacked Meta-LR", labels.copy(), oof_pred, oof_prob, fold_accs, fold_f1s)


def multiclass_ece(true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += mask.mean() * abs(acc - avg_conf)
    return float(ece)


def selective_curve(true: np.ndarray, prob: np.ndarray, coverages=(0.5, 0.7, 0.9, 1.0)):
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    order = np.argsort(-conf)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(len(true) * coverage)))
        chosen = order[:keep]
        rows.append(
            {
                "coverage": coverage,
                "accuracy": float(accuracy_score(true[chosen], pred[chosen])),
                "macro_f1": float(f1_score(true[chosen], pred[chosen], average="macro", zero_division=0)),
                "mean_conf": float(conf[chosen].mean()),
            }
        )
    return rows


def plot_reliability(results: list[ProbResult]):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    colors = ["#4c78a8", "#f58518", "#59a14f"]

    for idx, result in enumerate(results):
        true_binary = (result.pred == result.true).astype(np.int32)
        conf = result.prob.max(axis=1)
        frac_pos, mean_pred = calibration_curve(true_binary, conf, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=result.name, color=colors[idx % len(colors)])

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability of Core Classifiers")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "core_reliability_diagram.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_selective(results: list[tuple[str, list[dict]]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"Soft-Vote Ensemble": "#4c78a8", "Fusion LR": "#f58518", "Stacked Meta-LR": "#59a14f"}
    for name, rows in results:
        coverage = [item["coverage"] for item in rows]
        accuracy = [item["accuracy"] for item in rows]
        ax.plot(coverage, accuracy, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on retained predictions")
    ax.set_title("Selective Prediction for Core Classifiers")
    ax.set_ylim(0.45, 0.85)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "core_selective_accuracy.png"), dpi=160, bbox_inches="tight")
    plt.close()


def save_results(block_rows, grouped_results: list[ProbResult], merged_results: list[ProbResult], calibration_rows: list[dict], selective_rows: dict[str, list[dict]]):
    out_path = os.path.join(OUTPUT_DIR, "grouped_calibrated_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Grouped and Calibrated Core Experiments\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Article grouping: inferred contiguous topical blocks\n\n")

        handle.write("Inferred article-like blocks\n")
        handle.write("-" * 60 + "\n")
        for row in block_rows:
            handle.write(
                f"group={row['group_id']:2d} start={row['start']:3d} end={row['end']:3d} "
                f"size={row['size']:3d} cut_score_before={row['cut_score_before']}\n"
            )

        handle.write("\nGrouped leave-one-block-out evaluation\n")
        handle.write("-" * 60 + "\n")
        for result in grouped_results:
            handle.write(
                f"{result.name:20s} acc={result.mean_acc:.4f} macro_f1={result.mean_macro_f1:.4f}\n"
            )

        handle.write("\nMerged 5-fold calibration view\n")
        handle.write("-" * 60 + "\n")
        for row in calibration_rows:
            handle.write(
                f"{row['name']:20s} "
                f"acc={row['acc']:.4f} macro_f1={row['macro_f1']:.4f} "
                f"ece={row['ece']:.4f} brier={row['brier']:.4f} nll={row['nll']:.4f}\n"
            )

        handle.write("\nSelective prediction\n")
        handle.write("-" * 60 + "\n")
        for name, rows in selective_rows.items():
            handle.write(f"{name}\n")
            for row in rows:
                handle.write(
                    f"  coverage={row['coverage']:.2f} accuracy={row['accuracy']:.4f} "
                    f"macro_f1={row['macro_f1']:.4f} mean_conf={row['mean_conf']:.4f}\n"
                )
    return out_path


def main():
    print("=" * 60)
    print("  Grouped and Calibrated Experiments")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    texts = df["text"].tolist()
    frames = sorted(df["core_frame"].unique())
    label2id = {frame: idx for idx, frame in enumerate(frames)}
    y = np.array([label2id[item] for item in df["core_frame"]], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    embeddings = embed_texts(texts, tokenizer, encoder)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    groups, cuts, block_rows, _ = infer_article_blocks(texts, embeddings)
    print(f"  Inferred {len(np.unique(groups))} contiguous article-like blocks")
    print(f"  Block sizes: {[row['size'] for row in block_rows]}")

    grouped_split_list = group_splits(groups)
    grouped_results = [
        evaluate_soft_vote(texts, embeddings, y, grouped_split_list),
        evaluate_fusion(texts, embeddings, y, grouped_split_list),
        evaluate_stacked(texts, embeddings, y, grouped_split_list),
    ]

    for result in grouped_results:
        print(f"  Grouped {result.name}: acc={result.mean_acc:.3f} macro-F1={result.mean_macro_f1:.3f}")

    merged_split_list = merged_splits(y)
    merged_results = [
        evaluate_soft_vote(texts, embeddings, y, merged_split_list),
        evaluate_fusion(texts, embeddings, y, merged_split_list),
        evaluate_stacked(texts, embeddings, y, merged_split_list),
    ]

    calibration_rows = []
    selective_rows = {}
    for result in merged_results:
        conf = result.prob.max(axis=1)
        brier = float(np.mean(np.sum((result.prob - np.eye(result.prob.shape[1])[result.true]) ** 2, axis=1)))
        nll = float(log_loss(result.true, result.prob, labels=np.arange(result.prob.shape[1])))
        ece = multiclass_ece(result.true, result.prob)
        calibration_rows.append(
            {
                "name": result.name,
                "acc": result.mean_acc,
                "macro_f1": result.mean_macro_f1,
                "ece": ece,
                "brier": brier,
                "nll": nll,
            }
        )
        selective_rows[result.name] = selective_curve(result.true, result.prob)
        print(f"  Calibration {result.name}: ECE={ece:.3f} Brier={brier:.3f} NLL={nll:.3f}")

    plot_reliability(merged_results)
    plot_selective([(result.name, selective_rows[result.name]) for result in merged_results])

    out_path = save_results(block_rows, grouped_results, merged_results, calibration_rows, selective_rows)
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
