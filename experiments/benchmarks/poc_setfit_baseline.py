"""
SetFit baseline on the merged 2026 Climate Frames dataset.

Environment note:
  - Requires a SetFit-compatible stack.
  - In this workspace, SetFit runs after upgrading setfit and moving
    transformers back to a compatible 4.x release.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from setfit import SetFitModel, Trainer, TrainingArguments

from climate_frames.dataset import DEFAULT_DATA_PATH, load_annotations
from climate_frames.paths import OUTPUTS_DIR

warnings.filterwarnings("ignore")

OUTPUT_DIR = str(OUTPUTS_DIR)
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_SPLITS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_setfit_cv():
    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    frames = sorted(df["core_frame"].unique())
    label2id = {frame: idx for idx, frame in enumerate(frames)}

    texts = df["text"].tolist()
    labels = np.array([label2id[frame] for frame in df["core_frame"]])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        train_ds = Dataset.from_dict(
            {
                "text": [texts[i] for i in train_idx],
                "label": [int(labels[i]) for i in train_idx],
            }
        )
        val_ds = Dataset.from_dict(
            {
                "text": [texts[i] for i in val_idx],
                "label": [int(labels[i]) for i in val_idx],
            }
        )

        model = SetFitModel.from_pretrained(MODEL_ID)
        args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"setfit_fold{fold}"),
            batch_size=(16, 16),
            num_epochs=(1, 8),
            num_iterations=10,
            sampling_strategy="oversampling",
            end_to_end=False,
            show_progress_bar=False,
            report_to="none",
            save_strategy="no",
            eval_strategy="no",
            seed=SEED,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            column_mapping={"text": "text", "label": "label"},
            metric="accuracy",
        )
        trainer.train()

        preds = np.asarray(model.predict([texts[i] for i in val_idx]), dtype=int)
        true = labels[val_idx]
        acc = accuracy_score(true, preds)
        macro_f1 = f1_score(true, preds, average="macro", zero_division=0)
        fold_metrics.append({"fold": fold, "acc": acc, "macro_f1": macro_f1})
        all_true.extend(true.tolist())
        all_pred.extend(preds.tolist())
        print(f"     Fold {fold}: Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}")

    per_class_f1 = {}
    for frame, idx in label2id.items():
        true_bin = np.array(all_true) == idx
        pred_bin = np.array(all_pred) == idx
        tp = int(np.logical_and(true_bin, pred_bin).sum())
        fp = int(np.logical_and(~true_bin, pred_bin).sum())
        fn = int(np.logical_and(true_bin, ~pred_bin).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class_f1[frame] = f1

    return {
        "rows": int(len(df)),
        "core_counts": df["core_frame"].value_counts().to_dict(),
        "folds": fold_metrics,
        "mean_acc": float(np.mean([item["acc"] for item in fold_metrics])),
        "std_acc": float(np.std([item["acc"] for item in fold_metrics])),
        "mean_macro_f1": float(np.mean([item["macro_f1"] for item in fold_metrics])),
        "std_macro_f1": float(np.std([item["macro_f1"] for item in fold_metrics])),
        "per_class_f1": per_class_f1,
    }


def save_results(results):
    out_path = os.path.join(OUTPUT_DIR, "setfit_cv_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("SetFit Cross-Validation Results\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Model: {MODEL_ID}\n")
        handle.write(f"CV folds: {N_SPLITS}\n")
        handle.write(f"Rows: {results['rows']}\n")
        handle.write(f"Core counts: {results['core_counts']}\n\n")
        for fold in results["folds"]:
            handle.write(
                f"Fold {fold['fold']}: acc={fold['acc']:.4f} "
                f"macro_f1={fold['macro_f1']:.4f}\n"
            )
        handle.write("\n")
        handle.write(f"Mean acc:      {results['mean_acc']:.4f} +/- {results['std_acc']:.4f}\n")
        handle.write(f"Mean macro-F1: {results['mean_macro_f1']:.4f} +/- {results['std_macro_f1']:.4f}\n")
        handle.write("\nPer-class F1:\n")
        for frame, score in sorted(results["per_class_f1"].items()):
            handle.write(f"  {frame:40s} {score:.4f}\n")


def main():
    print("=" * 60)
    print("  SetFit Baseline")
    print("=" * 60)
    print(f"  Dataset: {DATA_PATH}")
    print(f"  Model:   {MODEL_ID}")

    results = run_setfit_cv()
    save_results(results)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Mean acc:      {results['mean_acc']:.3f}")
    print(f"  Mean macro-F1: {results['mean_macro_f1']:.3f}")
    print(f"\n  Results -> {os.path.join(OUTPUT_DIR, 'setfit_cv_results.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
