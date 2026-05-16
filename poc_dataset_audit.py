"""
Dataset audit for the expanded 2026 Climate Frames workbook.

This script produces a paper-ready text summary of:
  - raw vs merged row counts
  - exact-text duplicate structure
  - annotation variation across duplicate texts
  - trigger-token span coverage
  - frame-role alignment issues
  - peripheral-label support
  - leakage sensitivity for core-frame classification
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from climate_frames_dataset import (
    DEFAULT_DATA_PATH,
    find_token_spans,
    load_annotations,
    split_annotation_items,
)

OUTPUT_DIR = r"E:\Frames\poc_outputs"
DATA_PATH = str(DEFAULT_DATA_PATH)
LEGACY_DATA_PATH = Path(r"E:\Frames\12 articles Ann. Core Peripheral RST and FrameNET Structure.xlsx")
SEED = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _stringify_list(values):
    return " | ".join(values) if values else ""


def duplicate_summary(raw_df: pd.DataFrame) -> dict:
    grouped = raw_df.groupby("text", sort=False)
    sizes = grouped.size()
    duplicate_sizes = sizes[sizes > 1]

    row_keys = raw_df.apply(
        lambda row: (
            row["text"],
            row["core_frame"],
            tuple(row["peripheral_frames"]),
            tuple(row["tokens"]),
            row["frame_roles"],
        ),
        axis=1,
    )
    full_row_duplicates = int(row_keys.duplicated().sum())

    peripheral_variant_groups = 0
    token_variant_groups = 0
    role_variant_groups = 0
    core_conflict_groups = 0
    sample_groups = []

    for text, group in grouped:
        if len(group) < 2:
            continue
        core_values = sorted(group["core_frame"].dropna().astype(str).unique().tolist())
        periph_values = sorted({_stringify_list(items) for items in group["peripheral_frames"]})
        token_values = sorted({_stringify_list(items) for items in group["tokens"]})
        role_values = sorted(set(group["frame_roles"].fillna("").astype(str).tolist()))

        if len(core_values) > 1:
            core_conflict_groups += 1
        if len(periph_values) > 1:
            peripheral_variant_groups += 1
        if len(token_values) > 1:
            token_variant_groups += 1
        if len(role_values) > 1:
            role_variant_groups += 1

        if len(sample_groups) < 5 and (len(periph_values) > 1 or len(token_values) > 1 or len(role_values) > 1):
            sample_groups.append(
                {
                    "text": text,
                    "rows": group["row_id"].tolist(),
                    "core": core_values,
                    "peripheral_variants": periph_values,
                    "token_variants": token_values,
                    "role_variants": role_values,
                }
            )

    return {
        "raw_rows": int(len(raw_df)),
        "unique_texts": int(sizes.shape[0]),
        "duplicate_groups": int(duplicate_sizes.shape[0]),
        "rows_in_duplicate_groups": int(duplicate_sizes.sum()),
        "extra_duplicate_rows": int((duplicate_sizes - 1).sum()),
        "max_group_size": int(duplicate_sizes.max()) if not duplicate_sizes.empty else 1,
        "full_row_duplicates": full_row_duplicates,
        "core_conflict_groups": core_conflict_groups,
        "peripheral_variant_groups": peripheral_variant_groups,
        "token_variant_groups": token_variant_groups,
        "role_variant_groups": role_variant_groups,
        "sample_groups": sample_groups,
    }


def trigger_coverage(df: pd.DataFrame) -> dict:
    all_found = 0
    some_found = 0
    none_found = 0
    no_tokens = 0
    matched_types = 0
    total_types = 0

    for _, row in df.iterrows():
        tokens = row["tokens"]
        if not tokens:
            no_tokens += 1
            continue
        spans = find_token_spans(row["text"], tokens)
        matched_token_types = {token for _, _, token in spans}
        total_types += len(tokens)
        matched_types += len(matched_token_types)

        if len(matched_token_types) == len(tokens):
            all_found += 1
        elif matched_token_types:
            some_found += 1
        else:
            none_found += 1

    return {
        "all_found_rows": all_found,
        "some_found_rows": some_found,
        "none_found_rows": none_found,
        "no_token_rows": no_tokens,
        "matched_trigger_types": matched_types,
        "total_trigger_types": total_types,
    }


def role_alignment(df: pd.DataFrame) -> dict:
    rows_with_roles = 0
    rows_with_token_role_counts = 0
    exact_matches = 0
    mismatches = 0

    for _, row in df.iterrows():
        roles = split_annotation_items(row["frame_roles"])
        tokens = row["tokens"]
        if not roles:
            continue
        rows_with_roles += 1
        rows_with_token_role_counts += 1
        if len(tokens) == len(roles):
            exact_matches += 1
        else:
            mismatches += 1

    return {
        "rows_with_roles": rows_with_roles,
        "rows_with_token_role_counts": rows_with_token_role_counts,
        "exact_token_role_matches": exact_matches,
        "token_role_count_mismatches": mismatches,
    }


def peripheral_summary(merged_df: pd.DataFrame) -> dict:
    counts = {}
    label_cardinality = []
    for labels in merged_df["peripheral_frames"]:
        label_cardinality.append(len(labels))
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "labels": len(sorted_counts),
        "avg_labels_per_text": float(np.mean(label_cardinality)) if label_cardinality else 0.0,
        "max_labels_per_text": int(max(label_cardinality)) if label_cardinality else 0,
        "labels_ge_5": int(sum(count >= 5 for _, count in sorted_counts)),
        "labels_ge_10": int(sum(count >= 10 for _, count in sorted_counts)),
        "labels_ge_20": int(sum(count >= 20 for _, count in sorted_counts)),
        "top_labels": sorted_counts[:15],
    }


def tfidf_cv(texts, labels, splits):
    all_true = []
    all_pred = []
    fold_accs = []
    fold_f1s = []

    for train_idx, val_idx in splits:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
        X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        X_val = vectorizer.transform([texts[i] for i in val_idx])
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        fold_accs.append(accuracy_score(y_val, preds))
        fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))
        all_true.extend(y_val.tolist())
        all_pred.extend(preds.tolist())

    return {
        "mean_acc": float(np.mean(fold_accs)),
        "std_acc": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
    }


def leakage_audit(raw_df: pd.DataFrame, merged_df: pd.DataFrame) -> dict:
    raw_frames = sorted(raw_df["core_frame"].unique())
    raw_label2id = {frame: idx for idx, frame in enumerate(raw_frames)}
    raw_labels = np.array([raw_label2id[label] for label in raw_df["core_frame"]])
    raw_texts = raw_df["text"].tolist()
    raw_groups = raw_df["text"].tolist()

    row_splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.arange(len(raw_texts)),
            raw_labels,
        )
    )
    group_splits = list(
        StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.arange(len(raw_texts)),
            raw_labels,
            groups=raw_groups,
        )
    )

    merged_frames = sorted(merged_df["core_frame"].unique())
    merged_label2id = {frame: idx for idx, frame in enumerate(merged_frames)}
    merged_labels = np.array([merged_label2id[label] for label in merged_df["core_frame"]])
    merged_texts = merged_df["text"].tolist()
    merged_splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.arange(len(merged_texts)),
            merged_labels,
        )
    )

    return {
        "raw_row_level": tfidf_cv(raw_texts, raw_labels, row_splits),
        "raw_grouped_by_text": tfidf_cv(raw_texts, raw_labels, group_splits),
        "merged_unique_texts": tfidf_cv(merged_texts, merged_labels, merged_splits),
    }


def legacy_overlap(merged_df: pd.DataFrame) -> dict:
    if not LEGACY_DATA_PATH.exists() or LEGACY_DATA_PATH.resolve() == Path(DATA_PATH).resolve():
        return {}

    legacy_df = load_annotations(LEGACY_DATA_PATH, dedupe_mode="merge")
    merged_texts = set(merged_df["text"])
    legacy_texts = set(legacy_df["text"])
    overlap = merged_texts & legacy_texts

    return {
        "legacy_rows": int(len(legacy_df)),
        "legacy_unique_texts": int(len(legacy_texts)),
        "overlap_texts": int(len(overlap)),
        "new_only_texts": int(len(merged_texts - legacy_texts)),
        "legacy_only_texts": int(len(legacy_texts - merged_texts)),
    }


def save_report(raw_df, merged_df, dupes, coverage, roles, peripherals, leakage, legacy):
    out_path = Path(OUTPUT_DIR) / "dataset_audit_results.txt"
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Climate Frames Dataset Audit\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Active workbook: {DATA_PATH}\n")
        handle.write("Duplicate handling target for modeling: exact-text merge\n\n")

        handle.write("Counts\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Raw annotation rows kept: {len(raw_df)}\n")
        handle.write(f"Merged unique texts:      {len(merged_df)}\n")
        handle.write(f"Core frame inventory:     {merged_df['core_frame'].nunique()}\n")
        handle.write(f"Core counts:              {merged_df['core_frame'].value_counts().to_dict()}\n\n")

        handle.write("Duplicate Structure\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Duplicate text groups:      {dupes['duplicate_groups']}\n")
        handle.write(f"Rows in duplicate groups:   {dupes['rows_in_duplicate_groups']}\n")
        handle.write(f"Extra duplicate rows:       {dupes['extra_duplicate_rows']}\n")
        handle.write(f"Maximum group size:         {dupes['max_group_size']}\n")
        handle.write(f"Full row duplicates:        {dupes['full_row_duplicates']}\n")
        handle.write(f"Core-conflict groups:       {dupes['core_conflict_groups']}\n")
        handle.write(f"Peripheral-variant groups:  {dupes['peripheral_variant_groups']}\n")
        handle.write(f"Token-variant groups:       {dupes['token_variant_groups']}\n")
        handle.write(f"Role-variant groups:        {dupes['role_variant_groups']}\n\n")

        if dupes["sample_groups"]:
            handle.write("Sample duplicate groups with annotation variation\n")
            handle.write("-" * 60 + "\n")
            for idx, sample in enumerate(dupes["sample_groups"], start=1):
                handle.write(f"Sample {idx}\n")
                handle.write(f"Rows: {sample['rows']}\n")
                handle.write(f"Core: {sample['core']}\n")
                handle.write(f"Peripheral variants: {sample['peripheral_variants']}\n")
                handle.write(f"Token variants: {sample['token_variants']}\n")
                handle.write(f"Role variants: {sample['role_variants']}\n")
                handle.write(f"Text: {sample['text'][:400]}\n\n")

        handle.write("Trigger Coverage\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Rows with all trigger types found exactly: {coverage['all_found_rows']}\n")
        handle.write(f"Rows with some trigger types found:        {coverage['some_found_rows']}\n")
        handle.write(f"Rows with no trigger type found:           {coverage['none_found_rows']}\n")
        handle.write(f"Rows with no trigger annotations:          {coverage['no_token_rows']}\n")
        handle.write(
            f"Matched trigger types overall:             "
            f"{coverage['matched_trigger_types']}/{coverage['total_trigger_types']}\n\n"
        )

        handle.write("Role Alignment\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Rows with role annotations:        {roles['rows_with_roles']}\n")
        handle.write(f"Exact token-role count matches:    {roles['exact_token_role_matches']}\n")
        handle.write(f"Token-role count mismatches:       {roles['token_role_count_mismatches']}\n\n")

        handle.write("Peripheral Labels\n")
        handle.write("-" * 60 + "\n")
        handle.write(f"Normalized peripheral labels: {peripherals['labels']}\n")
        handle.write(f"Average labels per text:      {peripherals['avg_labels_per_text']:.3f}\n")
        handle.write(f"Maximum labels per text:      {peripherals['max_labels_per_text']}\n")
        handle.write(f"Labels with support >= 5:     {peripherals['labels_ge_5']}\n")
        handle.write(f"Labels with support >= 10:    {peripherals['labels_ge_10']}\n")
        handle.write(f"Labels with support >= 20:    {peripherals['labels_ge_20']}\n")
        handle.write("Top labels:\n")
        for label, count in peripherals["top_labels"]:
            handle.write(f"  {label:45s} {count}\n")
        handle.write("\n")

        handle.write("Leakage Audit (TF-IDF + LogisticRegression)\n")
        handle.write("-" * 60 + "\n")
        for name, metrics in leakage.items():
            handle.write(
                f"{name:22s} "
                f"acc={metrics['mean_acc']:.4f} +/- {metrics['std_acc']:.4f} "
                f"macro_f1={metrics['mean_macro_f1']:.4f} +/- {metrics['std_macro_f1']:.4f}\n"
            )
        handle.write("\n")

        if legacy:
            handle.write("Legacy Dataset Overlap\n")
            handle.write("-" * 60 + "\n")
            for key, value in legacy.items():
                handle.write(f"{key.replace('_', ' ').title():25s} {value}\n")

    return out_path


def main():
    raw_df = load_annotations(DATA_PATH, dedupe_mode="none")
    merged_df = load_annotations(DATA_PATH, dedupe_mode="merge")

    dupes = duplicate_summary(raw_df)
    coverage = trigger_coverage(raw_df)
    roles = role_alignment(raw_df)
    peripherals = peripheral_summary(merged_df)
    leakage = leakage_audit(raw_df, merged_df)
    legacy = legacy_overlap(merged_df)
    out_path = save_report(raw_df, merged_df, dupes, coverage, roles, peripherals, leakage, legacy)

    print("=" * 60)
    print("  Dataset Audit")
    print("=" * 60)
    print(f"  Raw rows:         {len(raw_df)}")
    print(f"  Merged texts:     {len(merged_df)}")
    print(f"  Duplicate groups: {dupes['duplicate_groups']}")
    print(
        f"  Trigger coverage: {coverage['matched_trigger_types']}/"
        f"{max(coverage['total_trigger_types'], 1)} token types matched exactly"
    )
    print(
        f"  Leakage check: row-level macro-F1={leakage['raw_row_level']['mean_macro_f1']:.3f}, "
        f"grouped macro-F1={leakage['raw_grouped_by_text']['mean_macro_f1']:.3f}, "
        f"merged macro-F1={leakage['merged_unique_texts']['mean_macro_f1']:.3f}"
    )
    print(f"\n  Results -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
