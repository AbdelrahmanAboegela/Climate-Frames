"""
CRF-based trigger extraction benchmark.

This experiment keeps the current ClimateBERT + LoRA token encoder, but adds a
linear-chain CRF decoder on top of token emissions. It reports both:
  1. raw emission argmax
  2. CRF decoding

This gives a cleaner answer to whether structured sequence modeling improves
frame-trigger extraction on the current dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

from climate_frames_dataset import DEFAULT_DATA_PATH, find_token_spans, load_annotations

hf_logging.set_verbosity_error()

OUTPUT_DIR = r"E:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
SEED = 42
N_SPLITS = 3
EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ID2LABEL = {0: "O", 1: "B-TRIGGER", 2: "I-TRIGGER"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)


@dataclass
class Metrics:
    entity_precision: float
    entity_recall: float
    entity_f1: float
    token_precision: float
    token_recall: float
    token_f1: float


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
    label_mask = []

    for start, end in encoding["offset_mapping"]:
        if start == end:
            labels.append(-100)
            label_mask.append(0)
            continue

        label = LABEL2ID["O"]
        for span_start, span_end, _ in example["spans"]:
            if not (start < span_end and end > span_start):
                continue
            label = LABEL2ID["B-TRIGGER"] if start <= span_start else LABEL2ID["I-TRIGGER"]
            break
        labels.append(label)
        label_mask.append(1)

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    encoding["label_mask"] = label_mask
    return encoding


def prepare_dataset(examples, tokenizer):
    rows = []
    for example in examples:
        encoded = tokenize_and_align(example, tokenizer)
        rows.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["labels"],
                "label_mask": encoded["label_mask"],
                "core_frame": example["core_frame"],
            }
        )
    return rows


def collate_batch(tokenizer, rows):
    model_inputs = tokenizer.pad(
        [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows],
        padding=True,
        return_tensors="pt",
    )
    max_len = model_inputs["input_ids"].shape[1]
    labels = []
    label_mask = []
    for row in rows:
        pad = max_len - len(row["labels"])
        labels.append(row["labels"] + ([-100] * pad))
        label_mask.append(row["label_mask"] + ([0] * pad))
    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    model_inputs["label_mask"] = torch.tensor(label_mask, dtype=torch.bool)
    return model_inputs


class CRFTokenClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        base = AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            # We use the encoder as a feature extractor and handle the
            # token-classification + CRF heads ourselves.
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_cfg)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)
        self.crf = CRF(NUM_LABELS, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))
        outputs = {"emissions": emissions}

        if labels is not None and label_mask is not None:
            crf_labels = labels.masked_fill(labels < 0, LABEL2ID["O"])
            crf_mask = attention_mask.bool()
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")
            outputs["loss"] = loss
        return outputs

    def decode(self, emissions, attention_mask):
        return self.crf.decode(emissions, mask=attention_mask.bool())


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


def evaluate_sequences(true_sequences, pred_sequences) -> Metrics:
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
    return Metrics(entity_precision, entity_recall, entity_f1, token_precision, token_recall, token_f1)


def train_one_epoch(model, loader, optimizer):
    model.train()
    losses = []
    for batch in loader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_model(model, loader):
    model.eval()
    argmax_true = []
    argmax_pred = []
    crf_true = []
    crf_pred = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            label_mask = batch["label_mask"]
            batch_on_device = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = model(**batch_on_device)
            emissions = outputs["emissions"].cpu()
            crf_sequences = model.decode(outputs["emissions"], batch_on_device["attention_mask"])
            argmax_sequences = emissions.argmax(dim=-1)

            for i in range(labels.shape[0]):
                valid = label_mask[i].bool()
                true_seq = labels[i][valid].tolist()
                argmax_seq = argmax_sequences[i][valid].tolist()
                crf_seq = [
                    pred_id
                    for pred_id, keep in zip(crf_sequences[i], valid.tolist())
                    if keep
                ]
                argmax_true.append(true_seq)
                argmax_pred.append(argmax_seq)
                crf_true.append(true_seq)
                crf_pred.append(crf_seq)

    return evaluate_sequences(argmax_true, argmax_pred), evaluate_sequences(crf_true, crf_pred)


def count_trainable(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def run_cv(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    rows = prepare_dataset(examples, tokenizer)
    labels = np.array([row["core_frame"] for row in rows])
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(len(rows)), labels))

    argmax_metrics = []
    crf_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        train_loader = DataLoader(
            train_rows,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(tokenizer, batch),
        )
        val_loader = DataLoader(
            val_rows,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(tokenizer, batch),
        )

        model = CRFTokenClassifier(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        trainable, total = count_trainable(model)
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer)
            print(f"     Fold {fold} Epoch {epoch}: loss={loss:.4f}")

        argmax_score, crf_score = evaluate_model(model, val_loader)
        argmax_metrics.append(argmax_score)
        crf_metrics.append(crf_score)
        print(
            f"     Fold {fold}: argmax entity-F1={argmax_score.entity_f1:.3f} "
            f"crf entity-F1={crf_score.entity_f1:.3f} "
            f"Trainable={trainable:,}/{total:,}"
        )

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return argmax_metrics, crf_metrics


def mean_metrics(items: list[Metrics]) -> Metrics:
    keys = Metrics.__dataclass_fields__.keys()
    values = {key: float(np.mean([getattr(item, key) for item in items])) for key in keys}
    return Metrics(**values)


def save_results(argmax_metrics: list[Metrics], crf_metrics: list[Metrics]):
    out_path = os.path.join(OUTPUT_DIR, "trigger_crf_results.txt")
    argmax_mean = mean_metrics(argmax_metrics)
    crf_mean = mean_metrics(crf_metrics)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("Trigger CRF Benchmark\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Dataset path: {DATA_PATH}\n")
        handle.write("Duplicate handling: exact-text merge\n")
        handle.write(f"Model backbone: {MODEL_NAME}\n\n")

        for name, metrics_list, mean_item in [
            ("Argmax Emission Decoding", argmax_metrics, argmax_mean),
            ("CRF Decoding", crf_metrics, crf_mean),
        ]:
            handle.write(f"{name}\n")
            handle.write("-" * 60 + "\n")
            for idx, item in enumerate(metrics_list, start=1):
                handle.write(
                    f"Fold {idx}: entity_F1={item.entity_f1:.4f} "
                    f"token_F1={item.token_f1:.4f} "
                    f"entity_P={item.entity_precision:.4f} "
                    f"entity_R={item.entity_recall:.4f}\n"
                )
            handle.write(
                f"Mean: entity_F1={mean_item.entity_f1:.4f} token_F1={mean_item.token_f1:.4f} "
                f"entity_P={mean_item.entity_precision:.4f} entity_R={mean_item.entity_recall:.4f}\n\n"
            )

        handle.write("Delta (CRF - Argmax)\n")
        handle.write("-" * 60 + "\n")
        for key in Metrics.__dataclass_fields__.keys():
            handle.write(f"{key}: {getattr(crf_mean, key) - getattr(argmax_mean, key):+.4f}\n")
    return out_path, argmax_mean, crf_mean


def main():
    print("=" * 60)
    print("  Trigger CRF Benchmark")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_PATH}")

    examples = build_examples()
    print(f"  Trainable examples: {len(examples)}")
    argmax_metrics, crf_metrics = run_cv(examples)
    out_path, argmax_mean, crf_mean = save_results(argmax_metrics, crf_metrics)

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(f"  Argmax: entity-F1={argmax_mean.entity_f1:.3f} token-F1={argmax_mean.token_f1:.3f}")
    print(f"  CRF:    entity-F1={crf_mean.entity_f1:.3f} token-F1={crf_mean.token_f1:.3f}")
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
