"""
Structured trigger decoding with transition-aware Viterbi.

This keeps the same LoRA token detector as the main trigger benchmark, but
replaces argmax decoding with a transition-aware Viterbi pass learned from the
training label sequences in each fold.
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
NUM_LABELS = len(ID2LABEL)
VERY_NEG = -1e4


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
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(np.float64)
    weights = counts.sum() / np.clip(NUM_LABELS * counts, 1.0, None)
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


def softmax(logits: np.ndarray):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def allowed_transition(prev_label: int, next_label: int) -> bool:
    if next_label == LABEL2ID["I-TRIGGER"] and prev_label == LABEL2ID["O"]:
        return False
    return True


def estimate_transition_scores(label_sequences: list[list[int]], smoothing: float = 1.0):
    start_counts = np.full(NUM_LABELS, smoothing, dtype=np.float64)
    trans_counts = np.full((NUM_LABELS, NUM_LABELS), smoothing, dtype=np.float64)
    end_counts = np.full(NUM_LABELS, smoothing, dtype=np.float64)

    for seq in label_sequences:
        if not seq:
            continue
        start_counts[seq[0]] += 1.0
        end_counts[seq[-1]] += 1.0
        for prev_label, next_label in zip(seq[:-1], seq[1:]):
            trans_counts[prev_label, next_label] += 1.0

    for prev_label in range(NUM_LABELS):
        for next_label in range(NUM_LABELS):
            if not allowed_transition(prev_label, next_label):
                trans_counts[prev_label, next_label] = smoothing * 1e-3

    start_scores = np.log(start_counts / start_counts.sum())
    end_scores = np.log(end_counts / end_counts.sum())
    trans_scores = np.log(trans_counts / np.clip(trans_counts.sum(axis=1, keepdims=True), 1e-12, None))
    return start_scores, trans_scores, end_scores


def viterbi_decode(emission_log_probs: np.ndarray, start_scores: np.ndarray, trans_scores: np.ndarray, end_scores: np.ndarray):
    length = emission_log_probs.shape[0]
    dp = np.full((length, NUM_LABELS), VERY_NEG, dtype=np.float64)
    back = np.zeros((length, NUM_LABELS), dtype=np.int32)

    dp[0] = start_scores + emission_log_probs[0]
    dp[0, LABEL2ID["I-TRIGGER"]] = VERY_NEG

    for pos in range(1, length):
        for curr in range(NUM_LABELS):
            candidates = dp[pos - 1] + trans_scores[:, curr]
            best_prev = int(np.argmax(candidates))
            dp[pos, curr] = candidates[best_prev] + emission_log_probs[pos, curr]
            back[pos, curr] = best_prev

    last = int(np.argmax(dp[-1] + end_scores))
    path = [last]
    for pos in range(length - 1, 0, -1):
        last = int(back[pos, last])
        path.append(last)
    path.reverse()
    return path


def run_structured_decoder(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = build_dataset(examples, tokenizer)
    labels = np.array([item["core_frame"] for item in examples])
    splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.arange(len(examples)),
            labels,
        )
    )

    all_true = []
    all_argmax = []
    all_viterbi = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_ds = dataset.select(train_idx.tolist())
        val_ds = dataset.select(val_idx.tolist())
        weights = class_weights_from_dataset(train_ds)

        base = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
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
            output_dir=os.path.join(OUTPUT_DIR, f"structured_trigger_fold{fold}"),
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

        train_label_sequences = []
        for row in train_ds["labels"]:
            train_label_sequences.append([int(label) for label in row if label >= 0])
        start_scores, trans_scores, end_scores = estimate_transition_scores(train_label_sequences)

        pred_out = trainer.predict(val_ds)
        probs = softmax(pred_out.predictions)
        argmax_ids = probs.argmax(axis=-1)
        label_ids = pred_out.label_ids

        fold_true = []
        fold_argmax = []
        fold_viterbi = []
        for seq_true, seq_argmax, seq_prob in zip(label_ids, argmax_ids, probs):
            filtered_true = []
            filtered_argmax = []
            filtered_prob = []
            for true_id, argmax_id, prob in zip(seq_true.tolist(), seq_argmax.tolist(), seq_prob.tolist()):
                if true_id == -100:
                    continue
                filtered_true.append(int(true_id))
                filtered_argmax.append(int(argmax_id))
                filtered_prob.append(prob)

            emission_log_probs = np.log(np.clip(np.asarray(filtered_prob, dtype=np.float64), 1e-12, 1.0))
            filtered_viterbi = viterbi_decode(emission_log_probs, start_scores, trans_scores, end_scores)

            fold_true.append(filtered_true)
            fold_argmax.append(filtered_argmax)
            fold_viterbi.append(filtered_viterbi)

        argmax_metrics = evaluate_sequences(fold_true, fold_argmax)
        viterbi_metrics = evaluate_sequences(fold_true, fold_viterbi)
        print(
            f"     Fold {fold}: argmax entity-F1={argmax_metrics['entity_f1']:.3f} "
            f"viterbi entity-F1={viterbi_metrics['entity_f1']:.3f}"
        )

        all_true.extend(fold_true)
        all_argmax.extend(fold_argmax)
        all_viterbi.extend(fold_viterbi)

        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    argmax_metrics = evaluate_sequences(all_true, all_argmax)
    viterbi_metrics = evaluate_sequences(all_true, all_viterbi)
    return argmax_metrics, viterbi_metrics


def save_results(argmax_metrics, viterbi_metrics):
    out_path = os.path.join(OUTPUT_DIR, "trigger_structured_decoder_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Trigger Structured Decoder\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n\n")

        handle.write("Argmax decoding\n")
        handle.write("-" * 60 + "\n")
        for key in ["entity_precision", "entity_recall", "entity_f1", "token_precision", "token_recall", "token_f1"]:
            handle.write(f"{key}: {argmax_metrics[key]:.4f}\n")

        handle.write("\nTransition-aware Viterbi decoding\n")
        handle.write("-" * 60 + "\n")
        for key in ["entity_precision", "entity_recall", "entity_f1", "token_precision", "token_recall", "token_f1"]:
            handle.write(f"{key}: {viterbi_metrics[key]:.4f}\n")

        handle.write("\nDelta (Viterbi - Argmax)\n")
        handle.write("-" * 60 + "\n")
        for key in ["entity_precision", "entity_recall", "entity_f1", "token_precision", "token_recall", "token_f1"]:
            handle.write(f"{key}: {viterbi_metrics[key] - argmax_metrics[key]:+.4f}\n")

    return out_path


def main():
    print("=" * 60)
    print("  Trigger Structured Decoder")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    examples = build_examples()
    print(f"  Trainable examples: {len(examples)}")
    argmax_metrics, viterbi_metrics = run_structured_decoder(examples)
    out_path = save_results(argmax_metrics, viterbi_metrics)

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(
        f"  Argmax:  entity-F1={argmax_metrics['entity_f1']:.3f} token-F1={argmax_metrics['token_f1']:.3f}"
    )
    print(
        f"  Viterbi: entity-F1={viterbi_metrics['entity_f1']:.3f} token-F1={viterbi_metrics['token_f1']:.3f}"
    )
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
