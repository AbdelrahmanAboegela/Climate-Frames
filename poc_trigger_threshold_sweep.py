"""
Threshold sweep for trigger extraction.

This script keeps the same generic LoRA trigger detector as the main token
benchmark, then compares:
  1. plain argmax BIO decoding
  2. thresholded trigger decoding using P(B) + P(I)

The goal is to see whether we can recover better span precision and entity F1
without changing the data.
"""

from __future__ import annotations

import os

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

from climate_frames_dataset import DEFAULT_DATA_PATH, find_token_spans, load_annotations

hf_logging.set_verbosity_error()

OUTPUT_DIR = r"E:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {0: "O", 1: "B-TRIGGER", 2: "I-TRIGGER"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def build_examples():
    df = load_annotations(DATA_PATH, dedupe_mode="merge")
    examples = []
    for _, row in df.iterrows():
        spans = find_token_spans(row["text"], row["tokens"])
        if not spans:
            continue
        examples.append(
            {
                "text": row["text"],
                "core_frame": row["core_frame"],
                "spans": spans,
            }
        )
    return examples


def tokenize_and_align(example, tokenizer):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    labels = []

    for start, end in encoding["offset_mapping"]:
        if start == end:
            labels.append(-100)
            continue

        label = LABEL2ID["O"]
        for span_start, span_end, _ in example["spans"]:
            if not (start < span_end and end > span_start):
                continue
            label = LABEL2ID["B-TRIGGER"] if start <= span_start else LABEL2ID["I-TRIGGER"]
            break
        labels.append(label)

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    return encoding


def build_dataset(examples, tokenizer):
    encoded = [tokenize_and_align(example, tokenizer) for example in examples]
    return HFDataset.from_dict(
        {
            "input_ids": [row["input_ids"] for row in encoded],
            "attention_mask": [row["attention_mask"] for row in encoded],
            "labels": [row["labels"] for row in encoded],
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
    return torch.tensor(weights.astype(np.float32), device=DEVICE)


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


def evaluate_sequences(true_sequences, pred_sequences):
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

    entity_precision = tp / max(tp + fp, 1)
    entity_recall = tp / max(tp + fn, 1)
    entity_f1 = 2 * entity_precision * entity_recall / max(entity_precision + entity_recall, 1e-9)
    token_precision = token_tp / max(token_tp + token_fp, 1)
    token_recall = token_tp / max(token_tp + token_fn, 1)
    token_f1 = 2 * token_precision * token_recall / max(token_precision + token_recall, 1e-9)

    return {
        "entity_precision": float(entity_precision),
        "entity_recall": float(entity_recall),
        "entity_f1": float(entity_f1),
        "token_precision": float(token_precision),
        "token_recall": float(token_recall),
        "token_f1": float(token_f1),
    }


def decode_threshold_sequence(trigger_scores: list[float], threshold: float):
    decoded = []
    active = False
    for score in trigger_scores:
        if score >= threshold:
            decoded.append(LABEL2ID["I-TRIGGER"] if active else LABEL2ID["B-TRIGGER"])
            active = True
        else:
            decoded.append(LABEL2ID["O"])
            active = False
    return decoded


def softmax(logits: np.ndarray):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def run_sweep(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = build_dataset(examples, tokenizer)
    labels = np.array([item["core_frame"] for item in examples])
    splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.arange(len(examples)),
            labels,
        )
    )

    argmax_true = []
    argmax_pred = []
    threshold_data = {threshold: {"true": [], "pred": []} for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]}

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
        model = get_peft_model(base, lora_cfg).to(DEVICE)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels_tensor = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), labels_tensor.view(-1))
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"trigger_threshold_fold{fold}"),
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
        probs = softmax(pred_out.predictions)
        pred_ids = probs.argmax(axis=-1)
        label_ids = pred_out.label_ids

        fold_argmax_true = []
        fold_argmax_pred = []
        fold_threshold_sequences = {threshold: {"true": [], "pred": []} for threshold in threshold_data}

        for seq_true, seq_pred, seq_prob in zip(label_ids, pred_ids, probs):
            filtered_true = []
            filtered_argmax = []
            filtered_scores = []
            for true_id, pred_id, prob in zip(seq_true.tolist(), seq_pred.tolist(), seq_prob.tolist()):
                if true_id == -100:
                    continue
                filtered_true.append(int(true_id))
                filtered_argmax.append(int(pred_id))
                filtered_scores.append(float(prob[LABEL2ID["B-TRIGGER"]] + prob[LABEL2ID["I-TRIGGER"]]))

            fold_argmax_true.append(filtered_true)
            fold_argmax_pred.append(filtered_argmax)
            for threshold in threshold_data:
                decoded = decode_threshold_sequence(filtered_scores, threshold)
                fold_threshold_sequences[threshold]["true"].append(filtered_true)
                fold_threshold_sequences[threshold]["pred"].append(decoded)

        argmax_true.extend(fold_argmax_true)
        argmax_pred.extend(fold_argmax_pred)

        for threshold in threshold_data:
            threshold_data[threshold]["true"].extend(fold_threshold_sequences[threshold]["true"])
            threshold_data[threshold]["pred"].extend(fold_threshold_sequences[threshold]["pred"])

        fold_metrics = evaluate_sequences(fold_argmax_true, fold_argmax_pred)
        print(
            f"     Fold {fold}: argmax entity-F1={fold_metrics['entity_f1']:.3f} "
            f"token-F1={fold_metrics['token_f1']:.3f}"
        )

        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    argmax_metrics = evaluate_sequences(argmax_true, argmax_pred)
    threshold_metrics = {}
    for threshold, payload in threshold_data.items():
        threshold_metrics[threshold] = evaluate_sequences(payload["true"], payload["pred"])

    return argmax_metrics, threshold_metrics


def save_results(argmax_metrics, threshold_metrics):
    out_path = os.path.join(OUTPUT_DIR, "trigger_threshold_sweep_results.txt")
    best_entity = max(threshold_metrics.items(), key=lambda item: (item[1]["entity_f1"], item[1]["token_f1"]))
    best_token = max(threshold_metrics.items(), key=lambda item: (item[1]["token_f1"], item[1]["entity_f1"]))

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Trigger Threshold Sweep\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n\n")

        handle.write("Argmax baseline\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Entity precision: {argmax_metrics['entity_precision']:.4f}\n")
        handle.write(f"Entity recall:    {argmax_metrics['entity_recall']:.4f}\n")
        handle.write(f"Entity F1:        {argmax_metrics['entity_f1']:.4f}\n")
        handle.write(f"Token precision:  {argmax_metrics['token_precision']:.4f}\n")
        handle.write(f"Token recall:     {argmax_metrics['token_recall']:.4f}\n")
        handle.write(f"Token F1:         {argmax_metrics['token_f1']:.4f}\n\n")

        handle.write("Threshold sweep\n")
        handle.write("-" * 60 + "\n")
        for threshold in sorted(threshold_metrics):
            metrics = threshold_metrics[threshold]
            handle.write(
                f"thr={threshold:.2f} "
                f"entity_F1={metrics['entity_f1']:.4f} "
                f"entity_P={metrics['entity_precision']:.4f} "
                f"entity_R={metrics['entity_recall']:.4f} "
                f"token_F1={metrics['token_f1']:.4f}\n"
            )

        handle.write("\nBest thresholds\n")
        handle.write("-" * 60 + "\n")
        handle.write(
            f"Best entity-F1 threshold: {best_entity[0]:.2f} "
            f"(entity_F1={best_entity[1]['entity_f1']:.4f}, token_F1={best_entity[1]['token_f1']:.4f})\n"
        )
        handle.write(
            f"Best token-F1 threshold:  {best_token[0]:.2f} "
            f"(token_F1={best_token[1]['token_f1']:.4f}, entity_F1={best_token[1]['entity_f1']:.4f})\n"
        )

    return out_path, best_entity, best_token


def main():
    print("=" * 60)
    print("  Trigger Threshold Sweep")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    examples = build_examples()
    print(f"  Trainable examples: {len(examples)}")
    argmax_metrics, threshold_metrics = run_sweep(examples)
    out_path, best_entity, best_token = save_results(argmax_metrics, threshold_metrics)

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(
        f"  Argmax baseline: entity-F1={argmax_metrics['entity_f1']:.3f} "
        f"token-F1={argmax_metrics['token_f1']:.3f}"
    )
    print(
        f"  Best entity threshold: {best_entity[0]:.2f} "
        f"entity-F1={best_entity[1]['entity_f1']:.3f} "
        f"token-F1={best_entity[1]['token_f1']:.3f}"
    )
    print(
        f"  Best token threshold:  {best_token[0]:.2f} "
        f"token-F1={best_token[1]['token_f1']:.3f} "
        f"entity-F1={best_token[1]['entity_f1']:.3f}"
    )
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
