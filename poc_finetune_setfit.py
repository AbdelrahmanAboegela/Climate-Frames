"""
Climate Frames benchmark suite on the merged 2026 workbook.

This script focuses on the parts that are train-ready right now:
  1. Core-frame classification
     - TF-IDF + LogisticRegression
     - Frozen ClimateBERT embeddings + LogisticRegression
     - LoRA sequence classification with class-balanced loss
  2. Dictionary token-extraction sanity check from Token Summary
  3. Peripheral-frame multi-label prediction baseline

Notes:
  - Exact-text duplicates are merged before evaluation to avoid leakage.
  - The SetFit baseline is tracked separately in `poc_setfit_baseline.py`
    so this script focuses on non-SetFit trainable baselines.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

from climate_frames_dataset import (
    DEFAULT_DATA_PATH,
    dataset_profile,
    load_annotations,
    token_summary_by_frame,
)

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

OUTPUT_DIR = r"e:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
N_SPLITS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_stratified_splits(labels: np.ndarray, n_splits: int = N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    indices = np.arange(len(labels))
    return list(skf.split(indices, labels))


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch_texts in batched(texts, batch_size):
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def effective_number_weights(labels: np.ndarray, n_labels: int, beta: float = 0.99) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    weights = weights / weights.mean()
    return torch.tensor(weights.astype(np.float32), device=device)


def run_tfidf_baseline(df: pd.DataFrame, label2id: dict[str, int], splits):
    texts = df["text"].tolist()
    y = np.array([label2id[label] for label in df["core_frame"]])

    all_true: list[int] = []
    all_preds: list[int] = []
    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    print("\n" + "=" * 60)
    print("  Strategy A: TF-IDF + LogisticRegression")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        X_val = vectorizer.transform([texts[i] for i in val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        all_true.extend(y_val.tolist())
        all_preds.extend(preds.tolist())
        print(f"     Fold {fold}: Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}")

    return all_true, all_preds, fold_accs, fold_f1s


def run_embedding_lr_baseline(
    df: pd.DataFrame,
    label2id: dict[str, int],
    splits,
    tokenizer,
    encoder,
):
    texts = df["text"].tolist()
    y = np.array([label2id[label] for label in df["core_frame"]])
    X = embed_texts(texts, tokenizer, encoder)

    all_true: list[int] = []
    all_preds: list[int] = []
    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    print("\n" + "=" * 60)
    print("  Strategy B: Frozen ClimateBERT Embeddings + LogisticRegression")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        all_true.extend(y_val.tolist())
        all_preds.extend(preds.tolist())
        print(f"     Fold {fold}: Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}")

    return all_true, all_preds, fold_accs, fold_f1s, X


def run_lora_cv(df: pd.DataFrame, label2id: dict[str, int], splits, epochs: int = 4):
    texts = df["text"].tolist()
    y = np.array([label2id[label] for label in df["core_frame"]])
    n_labels = len(label2id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_true: list[int] = []
    all_preds: list[int] = []
    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    print("\n" + "=" * 60)
    print("  Strategy C: LoRA Sequence Classification")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        y_train = y[train_idx]
        y_val = y[val_idx]
        class_weights = effective_number_weights(y_train, n_labels=n_labels, beta=0.99)

        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_labels,
            ignore_mismatched_sizes=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        model = get_peft_model(base, lora_cfg).to(device)

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=512)

        train_ds = HFDataset.from_dict(
            {
                "text": [texts[i] for i in train_idx],
                "labels": [int(y[i]) for i in train_idx],
            }
        ).map(tokenize, batched=True, remove_columns=["text"])
        val_ds = HFDataset.from_dict(
            {
                "text": [texts[i] for i in val_idx],
                "labels": [int(y[i]) for i in val_idx],
            }
        ).map(tokenize, batched=True, remove_columns=["text"])

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"newdata_lora_fold{fold}"),
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=25,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=SEED,
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        trainer.train()
        pred_out = trainer.predict(val_ds)
        preds = np.argmax(pred_out.predictions, axis=1)

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        all_true.extend(y_val.tolist())
        all_preds.extend(preds.tolist())

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        print(
            f"     Fold {fold}: Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}  "
            f"Trainable={trainable:,}/{total:,}"
        )

        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_true, all_preds, fold_accs, fold_f1s


def evaluate_token_summary_baseline(df: pd.DataFrame):
    token_summary = token_summary_by_frame(DATA_PATH)
    all_tp = all_fp = all_fn = 0

    for _, row in df.iterrows():
        frame = row["core_frame"]
        text_lower = row["text"].lower()
        gold = set(row["tokens"])
        canon = set(token_summary.get(frame, []))
        pred = {token for token in canon if token in text_lower}

        all_tp += len(gold & pred)
        all_fp += len(pred - gold)
        all_fn += len(gold - pred)

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def run_peripheral_multilabel(df: pd.DataFrame, embeddings: np.ndarray):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["peripheral_frames"])
    supports = Y.sum(axis=0)
    label_names = list(mlb.classes_)
    keep_mask = supports >= 10
    Y = Y[:, keep_mask]
    label_names = [label for label, keep in zip(label_names, keep_mask) if keep]
    supports = supports[keep_mask]
    label_support = {label: int(support) for label, support in zip(label_names, supports)}

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_true = []
    all_pred = []
    fold_micro = []
    fold_macro = []

    print("\n" + "=" * 60)
    print("  Peripheral Multi-Label Baseline")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings), start=1):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SEED,
            )
        )
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_val)

        micro = f1_score(Y_val, pred, average="micro", zero_division=0)
        macro = f1_score(Y_val, pred, average="macro", zero_division=0)
        fold_micro.append(micro)
        fold_macro.append(macro)
        all_true.append(Y_val)
        all_pred.append(pred)
        print(f"     Fold {fold}: Micro-F1={micro:.3f}  Macro-F1={macro:.3f}")

    Y_true = np.vstack(all_true)
    Y_pred = np.vstack(all_pred)
    per_label_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)

    frequent = [(label, support, f1) for label, support, f1 in zip(label_names, supports, per_label_f1) if support >= 20]
    frequent.sort(key=lambda item: (-item[2], -item[1], item[0]))

    return {
        "micro_f1_mean": float(np.mean(fold_micro)),
        "micro_f1_std": float(np.std(fold_micro)),
        "macro_f1_mean": float(np.mean(fold_macro)),
        "macro_f1_std": float(np.std(fold_macro)),
        "label_support": label_support,
        "labels_retained": len(label_names),
        "per_label_f1": {label: float(score) for label, score in zip(label_names, per_label_f1)},
        "top_frequent_labels": frequent[:15],
    }


def plot_confusion(all_true, all_preds, id2label, filename: str, title: str):
    label_ids = sorted(id2label)
    label_names = [id2label[idx] for idx in label_ids]
    short = [name.replace(" and ", "\n& ")[:26] for name in label_names]
    cm = confusion_matrix(all_true, all_preds, labels=label_ids)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short,
        yticklabels=short,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160, bbox_inches="tight")
    plt.close()


def plot_per_class_comparison(results_by_name: dict[str, tuple[list[int], list[int]]], id2label):
    label_ids = sorted(id2label)
    label_names = [id2label[idx] for idx in label_ids]
    short = [name[:24] for name in label_names]

    strategy_names = list(results_by_name)
    per_strategy_f1 = []
    supports = None
    for strategy_name in strategy_names:
        y_true, y_pred = results_by_name[strategy_name]
        report = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        per_strategy_f1.append([report[name]["f1-score"] for name in label_names])
        if supports is None:
            supports = [report[name]["support"] for name in label_names]

    x = np.arange(len(label_names))
    width = 0.24
    offsets = np.linspace(-(len(strategy_names) - 1) / 2, (len(strategy_names) - 1) / 2, len(strategy_names)) * width
    colors = ["#4c78a8", "#59a14f", "#e15759"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, strategy_name in enumerate(strategy_names):
        ax.bar(
            x + offsets[idx],
            per_strategy_f1[idx],
            width,
            label=strategy_name,
            color=colors[idx % len(colors)],
            alpha=0.88,
        )

    for idx, support in enumerate(supports or []):
        ax.text(idx, -0.07, f"n={int(support)}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("F1 score")
    ax.set_ylim(-0.1, 1.02)
    ax.set_title("Core-frame classification on merged newdata - per-class F1")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "finetune_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()


def save_results(
    profile: dict,
    id2label: dict[int, str],
    tfidf_results,
    emb_results,
    lora_results,
    token_baseline,
    peripheral_results,
):
    out_path = os.path.join(OUTPUT_DIR, "finetune_cv_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Climate Frames Benchmark Suite\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Rows evaluated: {profile['rows']}\n")
        handle.write(f"Core labels: {profile['unique_core_frames']}\n")
        handle.write(f"Model family: {MODEL_NAME}\n")
        handle.write(f"CV: Stratified {N_SPLITS}-fold for core, shuffled {N_SPLITS}-fold for peripheral\n\n")

        for title, result in [
            ("TF-IDF + LogisticRegression", tfidf_results),
            ("Frozen ClimateBERT + LogisticRegression", emb_results),
            ("LoRA Sequence Classification", lora_results),
        ]:
            y_true, y_pred, fold_accs, fold_f1s = result
            handle.write(f"-- {title} --\n")
            handle.write(f"Fold accuracy: {[f'{score:.3f}' for score in fold_accs]}\n")
            handle.write(f"Mean accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}\n")
            handle.write(f"Fold macro-F1: {[f'{score:.3f}' for score in fold_f1s]}\n")
            handle.write(f"Mean macro-F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}\n")
            handle.write(
                classification_report(
                    y_true,
                    y_pred,
                    labels=sorted(id2label),
                    target_names=[id2label[idx] for idx in sorted(id2label)],
                    zero_division=0,
                )
            )
            handle.write("\n")

        precision, recall, f1 = token_baseline
        handle.write("-- Dictionary Token Baseline --\n")
        handle.write(f"Precision: {precision:.4f}\n")
        handle.write(f"Recall:    {recall:.4f}\n")
        handle.write(f"F1:        {f1:.4f}\n\n")

        handle.write("-- Peripheral Multi-Label Baseline --\n")
        handle.write(f"Labels retained (support >= 10): {peripheral_results['labels_retained']}\n")
        handle.write(
            f"Micro-F1: {peripheral_results['micro_f1_mean']:.4f} +/- {peripheral_results['micro_f1_std']:.4f}\n"
        )
        handle.write(
            f"Macro-F1: {peripheral_results['macro_f1_mean']:.4f} +/- {peripheral_results['macro_f1_std']:.4f}\n"
        )
        handle.write("Top frequent labels (support >= 20):\n")
        for label, support, score in peripheral_results["top_frequent_labels"]:
            handle.write(f"  {label:45s} support={int(support):3d} f1={score:.3f}\n")


def main():
    print("=" * 60)
    print("  Climate Frames Benchmark Suite")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    profile = dataset_profile(df)
    print(f"  Loaded {profile['rows']} merged texts across {profile['unique_core_frames']} core frames")
    print("\n  Core distribution:")
    for frame, count in profile["core_counts"].items():
        bar = "#" * max(1, int(count / 5))
        print(f"    {frame[:42]:42s} {count:3d} {bar}")

    frames = sorted(df["core_frame"].unique())
    label2id = {frame: idx for idx, frame in enumerate(frames)}
    id2label = {idx: frame for frame, idx in label2id.items()}
    y = np.array([label2id[label] for label in df["core_frame"]])
    splits = build_stratified_splits(y)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)

    tfidf_results = run_tfidf_baseline(df, label2id, splits)
    emb_results = run_embedding_lr_baseline(df, label2id, splits, tokenizer, encoder)
    _, _, _, _, embeddings = emb_results

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_results = run_lora_cv(df, label2id, splits, epochs=4)
    token_baseline = evaluate_token_summary_baseline(df)
    peripheral_results = run_peripheral_multilabel(df, embeddings)

    plot_confusion(
        emb_results[0],
        emb_results[1],
        id2label,
        "finetune_confusion_matrix_embedding_lr.png",
        "Frozen ClimateBERT + LogisticRegression\nNormalized confusion matrix",
    )
    plot_confusion(
        lora_results[0],
        lora_results[1],
        id2label,
        "finetune_confusion_matrix_lora.png",
        "ClimateBERT LoRA\nNormalized confusion matrix",
    )
    plot_per_class_comparison(
        {
            "TF-IDF": (tfidf_results[0], tfidf_results[1]),
            "Frozen ClimateBERT": (emb_results[0], emb_results[1]),
            "LoRA": (lora_results[0], lora_results[1]),
        },
        id2label,
    )
    save_results(
        profile,
        id2label,
        tfidf_results,
        emb_results[:4],
        lora_results,
        token_baseline,
        peripheral_results,
    )

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for name, result in [
        ("TF-IDF", tfidf_results),
        ("Frozen ClimateBERT", emb_results[:4]),
        ("LoRA", lora_results),
    ]:
        _, _, fold_accs, fold_f1s = result
        print(f"  {name:20s} Acc={np.mean(fold_accs):.3f}  Macro-F1={np.mean(fold_f1s):.3f}")
    print(
        f"  Peripheral baseline     Micro-F1={peripheral_results['micro_f1_mean']:.3f}  "
        f"Macro-F1={peripheral_results['macro_f1_mean']:.3f}"
    )
    print(
        f"  Token summary baseline  P={token_baseline[0]:.3f}  "
        f"R={token_baseline[1]:.3f}  F1={token_baseline[2]:.3f}"
    )
    print(f"\n  All outputs -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
