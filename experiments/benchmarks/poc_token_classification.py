"""
BIO token-classification benchmark for trigger spans.

This script converts merged trigger-token annotations into BIO labels and
fine-tunes ClimateBERT with LoRA for generic trigger detection.
Only rows with at least one mapped trigger span are kept for training.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

from climate_frames.dataset import DEFAULT_DATA_PATH, find_token_spans, load_annotations
from climate_frames.paths import OUTPUTS_DIR

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

OUTPUT_DIR = str(OUTPUTS_DIR)
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {0: "O", 1: "B-TRIGGER", 2: "I-TRIGGER"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def build_examples():
    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    examples = []
    matched_tokens = 0
    total_tokens = 0

    for _, row in df.iterrows():
        spans = find_token_spans(row["text"], row["tokens"])
        total_tokens += len(row["tokens"])
        matched_tokens += len({token for _, _, token in spans})
        if not spans:
            continue
        examples.append(
            {
                "text": row["text"],
                "core_frame": row["core_frame"],
                "tokens": row["tokens"],
                "spans": spans,
            }
        )

    return df, examples, matched_tokens, total_tokens


def tokenize_and_align(example, tokenizer):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    labels = []
    spans = example["spans"]

    for offset in encoding["offset_mapping"]:
        start, end = offset
        if start == end:
            labels.append(-100)
            continue

        label = LABEL2ID["O"]
        for span_start, span_end, _ in spans:
            overlaps = start < span_end and end > span_start
            if not overlaps:
                continue
            label = LABEL2ID["B-TRIGGER"] if start <= span_start else LABEL2ID["I-TRIGGER"]
            break
        labels.append(label)

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    return encoding


def build_hf_dataset(examples, tokenizer):
    encoded = [tokenize_and_align(example, tokenizer) for example in examples]
    return HFDataset.from_dict(
        {
            "input_ids": [item["input_ids"] for item in encoded],
            "attention_mask": [item["attention_mask"] for item in encoded],
            "labels": [item["labels"] for item in encoded],
        }
    )


def class_weights_from_dataset(dataset):
    labels = []
    for row in dataset["labels"]:
        labels.extend(label for label in row if label >= 0)
    counts = np.bincount(labels, minlength=len(ID2LABEL)).astype(np.float64)
    weights = counts.sum() / np.clip(len(ID2LABEL) * counts, 1.0, None)
    weights = weights / weights.mean()
    weights[0] = min(weights[0], 0.35)
    return torch.tensor(weights.astype(np.float32), device=device)


def bio_spans(label_ids: list[int]):
    spans = []
    idx = 0
    while idx < len(label_ids):
        label = label_ids[idx]
        if label in (LABEL2ID["B-TRIGGER"], LABEL2ID["I-TRIGGER"]):
            start = idx
            idx += 1
            while idx < len(label_ids) and label_ids[idx] == LABEL2ID["I-TRIGGER"]:
                idx += 1
            spans.append((start, idx))
            continue
        idx += 1
    return spans


def evaluate_predictions(true_sequences, pred_sequences):
    tp = fp = fn = 0
    token_tp = token_fp = token_fn = 0

    for true_ids, pred_ids in zip(true_sequences, pred_sequences):
        true_spans = set(bio_spans(true_ids))
        pred_spans = set(bio_spans(pred_ids))
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

        for true_id, pred_id in zip(true_ids, pred_ids):
            true_pos = true_id != LABEL2ID["O"]
            pred_pos = pred_id != LABEL2ID["O"]
            if true_pos and pred_pos:
                token_tp += 1
            elif pred_pos and not true_pos:
                token_fp += 1
            elif true_pos and not pred_pos:
                token_fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    token_precision = token_tp / max(token_tp + token_fp, 1)
    token_recall = token_tp / max(token_tp + token_fn, 1)
    token_f1 = 2 * token_precision * token_recall / max(token_precision + token_recall, 1e-9)

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
    }


def run_cv(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = build_hf_dataset(examples, tokenizer)
    labels = np.array([example["core_frame"] for example in examples])
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(len(examples)), labels))

    fold_metrics = []
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_ds = dataset.select(train_idx.tolist())
        val_ds = dataset.select(val_idx.tolist())
        weights = class_weights_from_dataset(train_ds)

        base = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        model = get_peft_model(base, lora_cfg).to(device)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels_tensor = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), labels_tensor.view(-1))
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"token_lora_fold{fold}"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        )
        trainer.train()

        pred_out = trainer.predict(val_ds)
        pred_ids = np.argmax(pred_out.predictions, axis=-1)
        label_ids = pred_out.label_ids

        fold_true = []
        fold_pred = []
        for seq_true, seq_pred in zip(label_ids, pred_ids):
            filtered_true = []
            filtered_pred = []
            for true_id, pred_id in zip(seq_true.tolist(), seq_pred.tolist()):
                if true_id == -100:
                    continue
                filtered_true.append(int(true_id))
                filtered_pred.append(int(pred_id))
            fold_true.append(filtered_true)
            fold_pred.append(filtered_pred)

        metrics = evaluate_predictions(fold_true, fold_pred)
        fold_metrics.append(metrics)
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        print(
            f"     Fold {fold}: entity-F1={metrics['entity_f1']:.3f}  "
            f"token-F1={metrics['token_f1']:.3f}"
        )

        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall = evaluate_predictions(all_true, all_pred)
    return fold_metrics, overall


def save_results(df, examples, matched_tokens, total_tokens, fold_metrics, overall):
    out_path = os.path.join(OUTPUT_DIR, "token_classification_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Token Classification Benchmark\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Merged texts available: {len(df)}\n")
        handle.write(f"Examples retained for BIO training: {len(examples)}\n")
        handle.write(f"Matched trigger types: {matched_tokens}/{max(total_tokens, 1)}\n\n")

        for idx, metrics in enumerate(fold_metrics, start=1):
            handle.write(
                f"Fold {idx}: entity-P={metrics['entity_precision']:.4f} "
                f"entity-R={metrics['entity_recall']:.4f} "
                f"entity-F1={metrics['entity_f1']:.4f} "
                f"token-F1={metrics['token_f1']:.4f}\n"
            )

        handle.write("\nOverall:\n")
        handle.write(f"  Entity precision: {overall['entity_precision']:.4f}\n")
        handle.write(f"  Entity recall:    {overall['entity_recall']:.4f}\n")
        handle.write(f"  Entity F1:        {overall['entity_f1']:.4f}\n")
        handle.write(f"  Token precision:  {overall['token_precision']:.4f}\n")
        handle.write(f"  Token recall:     {overall['token_recall']:.4f}\n")
        handle.write(f"  Token F1:         {overall['token_f1']:.4f}\n")


def main():
    print("=" * 60)
    print("  Token Classification Benchmark")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Dataset: {DATA_PATH}")

    df, examples, matched_tokens, total_tokens = build_examples()
    coverage = len(examples) / max(len(df), 1)
    print(f"  Merged texts: {len(df)}")
    print(f"  BIO-trainable texts: {len(examples)} ({coverage*100:.1f}%)")
    print(f"  Matched trigger types: {matched_tokens}/{max(total_tokens, 1)}")

    fold_metrics, overall = run_cv(examples)
    save_results(df, examples, matched_tokens, total_tokens, fold_metrics, overall)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Entity-F1: {overall['entity_f1']:.3f}")
    print(f"  Token-F1:  {overall['token_f1']:.3f}")
    print(f"\n  Results -> {os.path.join(OUTPUT_DIR, 'token_classification_results.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
