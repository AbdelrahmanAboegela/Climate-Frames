"""
Imbalance-aware LoRA experiments for core-frame classification.

This script revisits the sequence-classification path and asks a narrower
question than the main benchmark:

  Which loss function handles the current class imbalance best?

It compares:
  1. Effective-number weighted cross-entropy (current strong baseline)
  2. Class-balanced focal loss
  3. Logit-adjusted cross-entropy

The goal is not to beat every non-neural fusion model. It is to determine the
best professionally defensible neural fine-tuning recipe with the data already
available.
"""

from __future__ import annotations

import gc
import os
import shutil
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

from climate_frames.dataset import DEFAULT_DATA_PATH, load_annotations
from climate_frames.paths import OUTPUTS_DIR

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

OUTPUT_DIR = str(OUTPUTS_DIR)
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 5
EPOCHS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


@dataclass
class StrategyResult:
    name: str
    true: np.ndarray
    pred: np.ndarray
    fold_accs: list[float]
    fold_f1s: list[float]
    report: str
    confusion: np.ndarray

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


def effective_number_weights(labels: np.ndarray, n_labels: int, beta: float = 0.99) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    weights = weights / weights.mean()
    return torch.tensor(weights.astype(np.float32), device=DEVICE)


def class_priors(labels: np.ndarray, n_labels: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)
    priors = counts / counts.sum()
    return torch.tensor(np.clip(priors, 1e-12, None).astype(np.float32), device=DEVICE)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: torch.Tensor | None, gamma: float = 2.0):
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    if alpha is None:
        alpha_factor = torch.ones_like(pt)
    else:
        alpha_factor = alpha[labels]
    loss = alpha_factor * torch.pow(1.0 - pt, gamma) * ce
    return loss.mean()


def build_splits(labels: np.ndarray):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    idx = np.arange(len(labels))
    return list(skf.split(idx, labels))


def build_dataset(texts: list[str], labels: np.ndarray, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    ds = HFDataset.from_dict({"text": texts, "labels": labels.astype(int).tolist()})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def make_model(n_labels: int):
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
    return get_peft_model(base, lora_cfg).to(DEVICE)


def run_strategy(
    strategy_name: str,
    texts: list[str],
    labels: np.ndarray,
    id2label: dict[int, str],
    splits,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_labels = len(id2label)
    oof_pred = np.zeros_like(labels)
    fold_accs = []
    fold_f1s = []

    print("\n" + "=" * 60)
    print(f"  {strategy_name}")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_labels = labels[train_idx]
        weight_eff = effective_number_weights(train_labels, n_labels=n_labels)
        priors = class_priors(train_labels, n_labels=n_labels)

        train_ds = build_dataset([texts[i] for i in train_idx], labels[train_idx], tokenizer)
        val_ds = build_dataset([texts[i] for i in val_idx], labels[val_idx], tokenizer)

        model = make_model(n_labels)

        class LossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                gold = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                if strategy_name == "Effective-Weight CE":
                    loss = F.cross_entropy(logits, gold, weight=weight_eff)
                elif strategy_name == "Class-Balanced Focal":
                    loss = focal_loss(logits, gold, alpha=weight_eff, gamma=2.0)
                elif strategy_name == "Logit-Adjusted CE":
                    adjusted_logits = logits + torch.log(priors).unsqueeze(0)
                    loss = F.cross_entropy(adjusted_logits, gold)
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")

                return (loss, outputs) if return_outputs else loss

        fold_dir = os.path.join(
            OUTPUT_DIR,
            f"lora_imbalance_{strategy_name.lower().replace(' ', '_').replace('-', '_')}_fold{fold}",
        )
        args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=EPOCHS,
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

        trainer = LossTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        trainer.train()

        pred_out = trainer.predict(val_ds)
        pred = pred_out.predictions.argmax(axis=1)
        oof_pred[val_idx] = pred

        acc = accuracy_score(labels[val_idx], pred)
        macro_f1 = f1_score(labels[val_idx], pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        print(f"     Fold {fold}: Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}")

        del trainer, model, train_ds, val_ds, pred_out
        if os.path.isdir(fold_dir):
            shutil.rmtree(fold_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = classification_report(
        labels,
        oof_pred,
        target_names=[id2label[i] for i in range(n_labels)],
        digits=3,
        zero_division=0,
    )
    confusion = confusion_matrix(labels, oof_pred, labels=np.arange(n_labels))
    return StrategyResult(strategy_name, labels.copy(), oof_pred, fold_accs, fold_f1s, report, confusion)


def save_plot(results: list[StrategyResult]):
    plt.figure(figsize=(9, 5))
    order = np.arange(len(results))
    accs = [item.mean_acc for item in results]
    f1s = [item.mean_macro_f1 for item in results]
    acc_err = [item.std_acc for item in results]
    f1_err = [item.std_macro_f1 for item in results]

    width = 0.35
    plt.bar(order - width / 2, accs, width, yerr=acc_err, label="Accuracy", color="#3c78d8")
    plt.bar(order + width / 2, f1s, width, yerr=f1_err, label="Macro-F1", color="#cc4125")
    plt.xticks(order, [item.name for item in results], rotation=12, ha="right")
    plt.ylabel("Score")
    plt.ylim(0.0, 0.75)
    plt.title("Imbalance-Aware LoRA Strategies")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "lora_imbalance_comparison.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def save_results(results: list[StrategyResult], id2label: dict[int, str]):
    out_path = os.path.join(OUTPUT_DIR, "lora_imbalance_results.txt")
    best = max(results, key=lambda item: item.mean_macro_f1)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Imbalance-Aware LoRA Experiments\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Rows evaluated: {len(results[0].true)}\n")
        handle.write(f"Backbone: {MODEL_NAME}\n")
        handle.write(f"CV: Stratified {N_SPLITS}-fold\n")
        handle.write(f"Epochs per fold: {EPOCHS}\n\n")

        for item in results:
            handle.write(f"-- {item.name} --\n")
            handle.write(f"Fold accuracy: {[f'{x:.3f}' for x in item.fold_accs]}\n")
            handle.write(f"Mean accuracy: {item.mean_acc:.4f} +/- {item.std_acc:.4f}\n")
            handle.write(f"Fold macro-F1: {[f'{x:.3f}' for x in item.fold_f1s]}\n")
            handle.write(f"Mean macro-F1: {item.mean_macro_f1:.4f} +/- {item.std_macro_f1:.4f}\n")
            handle.write(item.report + "\n")

        handle.write("-- Best Strategy --\n")
        handle.write(f"{best.name}\n")
        handle.write(f"Accuracy: {best.mean_acc:.4f}\n")
        handle.write(f"Macro-F1: {best.mean_macro_f1:.4f}\n\n")

        baseline = next(item for item in results if item.name == "Effective-Weight CE")
        handle.write("Delta vs Effective-Weight CE\n")
        handle.write("-" * 60 + "\n")
        for item in results:
            handle.write(
                f"{item.name}: "
                f"accuracy_delta={item.mean_acc - baseline.mean_acc:+.4f} "
                f"macro_f1_delta={item.mean_macro_f1 - baseline.mean_macro_f1:+.4f}\n"
            )

    np.savez(
        os.path.join(OUTPUT_DIR, "lora_imbalance_oof_predictions.npz"),
        labels=results[0].true,
        **{item.name.replace(" ", "_").replace("-", "_").lower(): item.pred for item in results},
    )

    best_cm_path = os.path.join(OUTPUT_DIR, "lora_imbalance_best_confusion.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        best.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[id2label[i] for i in range(len(id2label))],
        yticklabels=[id2label[i] for i in range(len(id2label))],
    )
    plt.title(f"Best LoRA Strategy Confusion Matrix: {best.name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(best_cm_path, dpi=220)
    plt.close()

    return out_path, best, best_cm_path


def main():
    print("=" * 60)
    print("  Imbalance-Aware LoRA Experiments")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    texts = df["text"].tolist()
    labels_text = df["core_frame"].tolist()
    label_names = sorted(df["core_frame"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    labels = np.array([label2id[label] for label in labels_text], dtype=np.int64)
    splits = build_splits(labels)

    strategies = [
        "Effective-Weight CE",
        "Class-Balanced Focal",
        "Logit-Adjusted CE",
    ]
    results = [run_strategy(name, texts, labels, id2label, splits) for name in strategies]
    plot_path = save_plot(results)
    out_path, best, best_cm_path = save_results(results, id2label)

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    for item in results:
        print(f"  {item.name}: acc={item.mean_acc:.3f} macro-F1={item.mean_macro_f1:.3f}")
    print(f"  Best: {best.name}")
    print(f"\n  Results -> {out_path}")
    print(f"  Plot -> {plot_path}")
    print(f"  Confusion -> {best_cm_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
