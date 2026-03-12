# ClimateBERT Frame Classification — Research Report

**Project**: Automated Cognitive Frame Classification and Frame-Evoking Token Extraction in Climate Discourse  
**Model**: `climatebert/distilroberta-base-climate-f` (DistilRoBERTa, 82.4M parameters)  
**Dataset**: 232 annotated paragraphs — 12 climate change articles, 6 core frames  
**Status**: Phase 0 complete — base model probing + preliminary fine-tuning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Repository Structure](#3-repository-structure)
4. [Round 1 — Base Model Embedding Analysis](#4-round-1--base-model-embedding-analysis)
5. [Round 2 — Zero-Shot Classification & Frame Taxonomy](#5-round-2--zero-shot-classification--frame-taxonomy)
6. [Round 3 — Layer-wise Probing & Domain Comparison](#6-round-3--layer-wise-probing--domain-comparison)
7. [Round 4 — Attention Head Analysis](#7-round-4--attention-head-analysis)
8. [Phase 0 — Fine-Tuning Results](#8-phase-0--fine-tuning-results)
9. [Key Findings Summary](#9-key-findings-summary)
10. [Architecture Roadmap](#10-architecture-roadmap)
11. [Next Steps](#11-next-steps)

---

## 1. Project Overview

### Goal

Build a system that automatically:

1. **Classifies the cognitive frame** of a climate change paragraph (e.g., *Causal and Attribution Effect*, *Impact and Consequences*)
2. **Extracts the specific tokens** within that paragraph that invoke the frame (e.g., *"greenhouse gas emissions"*, *"fossil fuels"*)
3. **Tags peripheral frames** co-present in the same paragraph (multi-label)
4. **Assigns frame roles** to extracted tokens (e.g., *Cause*, *Agent*, *Source*)

This is distinct from topic classification (what the text is *about*) — it captures *how* climate change is cognitively framed, grounded in FrameNet and cognitive linguistics theory (Entman 1993, Fillmore 1976).

### Why ClimateBERT

`climatebert/distilroberta-base-climate-f` is a DistilRoBERTa model further pre-trained on 2M+ climate-related sentences. It achieves state-of-the-art on climate NLP benchmarks while being 6× smaller than full RoBERTa. Its domain pre-training means it already understands climate-specific terminology before any task-specific fine-tuning.

### Relation to Existing Work

| System | Task | Scale | Difference from ours |
|---|---|---|---|
| CCF Project (Canada) | Topical frame classification | 9.2M sentences, 65 annotators | Classifies *what topic* (Culture/Science/Economy); no token extraction; 120 siloed models |
| Otmakhova ACL 2024 | Climate aspect extraction | Sentence-level | No frame classification; no peripheral frames |
| Badullovich 2025 | Media framing analysis | Document-level | No token extraction; no FrameNet grounding |
| **This work** | **Cognitive frame + token extraction** | **Paragraph-level** | **Unified multi-head; FrameNet-grounded; peripheral frames + roles** |

---

## 2. Dataset Description

### Source

**File**: `12 articles Ann. Core Peripheral RST and FrameNET Structure.xlsx`  
**Sheet**: `Core and Peripheral Annotations`

Manually annotated by a team of computational linguists across 12 climate change news articles from Arabic media (translated to English for annotation).

### Schema

| Column | Type | Description |
|---|---|---|
| Text Segment | string | Full paragraph text |
| Core Frame | string | Primary cognitive frame (1 per paragraph) |
| Peripheral Frames | string (`;`-separated) | Secondary frames co-present |
| Trigger Tokens | string (`;`-separated) | Words/phrases that invoke the frame |
| Frame Roles | string (`,`-separated) | Semantic roles (Cause, Agent, Source, etc.) |

**Second sheet** — `Token Summary`: 47 rows of canonical frame-evoking tokens per frame, used for zero-shot description enhancement and dictionary-based token extraction.

### Class Distribution

| Core Frame | Count | % | Bar |
|---|---|---|---|
| Causal and Attribution Effect | 96 | 41.4% | ████████████████████ |
| Impact and Consequences | 59 | 25.4% | ████████████ |
| Epistemic and Scientific Research | 29 | 12.5% | ██████ |
| Action and Solutions | 29 | 12.5% | ██████ |
| Socio-Political and Economic | 11 | 4.7% | ██ |
| Temporal and Scalar | 8 | 3.4% | █ |
| **Total** | **232** | **100%** | |

**Total annotated frame tokens**: 1,094 across all paragraphs  
**Imbalance ratio**: 12:1 (Causal vs Temporal)

### Dataset Sufficiency Assessment

| Class | n | LoRA Fine-tune | SetFit/Proto |
|---|---|---|---|
| Causal and Attribution Effect | 96 | ✓ Sufficient | ✓ |
| Impact and Consequences | 59 | ✓ Sufficient | ✓ |
| Epistemic and Scientific Research | 29 | ⚠ Marginal | ✓ |
| Action and Solutions | 29 | ⚠ Marginal | ✓ |
| Socio-Political and Economic | 11 | ✗ Insufficient | ⚠ |
| Temporal and Scalar | 8 | ✗ Insufficient | ⚠ |

**Target for robust full fine-tuning**: ≥100 examples per class (~600 total). Currently at 232 — need ~270 more, concentrated on the two minority classes.

---

## 3. Repository Structure

```
E:\Frames\
├── 12 articles Ann. Core Peripheral RST and FrameNET Structure.xlsx   ← MAIN DATASET
├── V4 12 Articles Annotation Core Peripheral Tokens Role RST....xlsx  ← v4 reference
├── March26 Summary of Core and Peripheral Frames Fine Tuning...docx   ← taxonomy spec
├── Sample_Annotation_Format.xlsx                                       ← annotation guide
├── WhatsApp Image 2026-02-25 at 1.32.10 PM.jpeg                        ← CCF pipeline ref
│
├── poc_base_model_analysis.py    ← Round 1: embeddings, silhouette, MLM, attention
├── poc_round2_analysis.py        ← Round 2: zero-shot, intra/inter sim, taxonomy
├── poc_round3_analysis.py        ← Round 3: layer-wise, kNN probe, ClimateBERT vs vanilla
├── poc_round4_attention.py       ← Round 4: attention heads, rollout, entropy
├── poc_finetune_setfit.py        ← Phase 0: LoRA + prototype fine-tuning
│
├── poc_outputs/                  ← All generated figures and text results
│   ├── paragraph_umap_cls_core.png
│   ├── paragraph_umap_mean_core.png
│   ├── token_umap_core.png
│   ├── frame_similarity_heatmap.png
│   ├── frame_taxonomy_heatmap.png
│   ├── frame_taxonomy_umap.png
│   ├── climate_vs_vanilla.png
│   ├── layerwise_silhouette.png
│   ├── intra_inter_similarity.png
│   ├── tokenizer_coverage.png
│   ├── head_frame_attention.png
│   ├── attention_rollout.png
│   ├── attention_heatmap_P0/P2/P15.png
│   ├── cross_attention_matrix.png
│   ├── attention_entropy.png
│   ├── finetune_comparison.png
│   ├── finetune_confusion_matrix_lora.png
│   ├── finetune_confusion_matrix_prototype.png
│   └── *.txt  (knn_probe_results, zero_shot_results, finetune_cv_results, ...)
│
├── ReadMe.md                     ← This document
└── .gitignore
```

**Run order**:
```bash
python poc_base_model_analysis.py   # ~10 min
python poc_round2_analysis.py       # ~5 min
python poc_round3_analysis.py       # ~15 min
python poc_round4_attention.py      # ~20 min
python poc_finetune_setfit.py       # ~30 min (GPU recommended)
```

---

## 4. Round 1 — Base Model Embedding Analysis

**Script**: `poc_base_model_analysis.py`

### What it does

Five modules probing the base ClimateBERT model (no fine-tuning):

1. **UMAP of paragraph embeddings** — CLS token and mean-pooled, colored by core frame
2. **Token-level UMAP** — frame-evoking tokens embedded and clustered
3. **Frame-to-frame cosine similarity heatmap** — which frames share representational space
4. **MLM probing** — mask frame tokens, check if model predicts semantically appropriate replacements
5. **Attention weight analysis** — sample paragraphs, measure attention to frame vs non-frame tokens

### Results

| Metric | Value |
|---|---|
| Silhouette score (CLS embeddings, cosine) | **−0.0136** |
| Silhouette score (mean-pooled embeddings) | **−0.0122** |
| Frame-evoking tokens extracted | 894 |
| MLM probes run | 10 |

**Silhouette = −0.0136** means the 6 core frames are **not linearly separable** in the base representation space — paragraphs from different frames are geometrically interleaved. This is a critical finding: **fine-tuning is non-optional**. The base model, despite its climate domain training, cannot distinguish cognitive frames without task-specific supervision.

MLM probing confirmed ClimateBERT understands climate vocabulary: when frame tokens are masked, it predicts semantically coherent replacements (*"emissions"* → *"pollution"*, *"greenhouse gases"*) rather than random words.

**Outputs**: `paragraph_umap_cls_core.png`, `paragraph_umap_mean_core.png`, `token_umap_core.png`, `frame_similarity_heatmap.png`

---

## 5. Round 2 — Zero-Shot Classification & Frame Taxonomy

**Script**: `poc_round2_analysis.py`

### What it does

1. **Zero-shot frame classification** — embed each paragraph and each frame description; classify by cosine similarity. Frame descriptions are enriched using canonical tokens from the Token Summary sheet.
2. **Intra/inter-frame similarity** — measure cosine similarity within vs. across frame classes
3. **Frame taxonomy UMAP** — embed frame names + peripheral frames, visualize taxonomic structure
4. **Tokenizer coverage** — analyse how the RoBERTa tokenizer handles frame-evoking multi-word expressions

### Results

| Metric | Value |
|---|---|
| Zero-shot Top-1 Accuracy | **19.0%** |
| Zero-shot Top-3 Accuracy | **37.5%** |
| Random baseline | 16.7% |
| Intra-frame cosine similarity | 0.9763 |
| Inter-frame cosine similarity | 0.9727 |
| **Similarity gap (intra − inter)** | **0.0036** |
| Frame tokens kept whole (1 subword) | 604 / 1094 = **55.2%** |
| Frame tokens split (multi-subword) | 490 / 1094 = **44.8%** |

**Zero-shot at 19%** is only marginally above random (17%) — the base model embeddings are not discriminative enough for zero-shot frame classification, even with enriched frame descriptions. The intra/inter gap of 0.0036 is very small, confirming frames occupy overlapping regions of the embedding space.

**Tokenizer coverage**: 44.8% of frame-evoking terms are split into multiple subwords (e.g., *"intergenerational justice"* → 7 subwords; *"international cooperation"* → 5 subwords). This motivates mean-pooling over subword spans rather than using single token embeddings for frame-evoking terms.

**Outputs**: `intra_inter_similarity.png`, `frame_taxonomy_heatmap.png`, `frame_taxonomy_umap.png`, `tokenizer_coverage.png`, `zero_shot_results.txt`

---

## 6. Round 3 — Layer-wise Probing & Domain Comparison

**Script**: `poc_round3_analysis.py`

### What it does

1. **Layer-wise silhouette analysis** — extract embeddings from each of the 7 layers (embedding + 6 transformer layers), compute silhouette score. Identifies which layer contains the most frame-discriminative signal.
2. **k-NN probing (LOO-CV)** — k-nearest-neighbour classifier using leave-one-out cross-validation. Measures whether frames are locally separable even if globally interleaved.
3. **Sentence decomposition** — split paragraphs into sentences; check whether individual sentences carry the same frame signal as the full paragraph.
4. **ClimateBERT vs vanilla DistilRoBERTa** — compare all metrics to a non-domain-adapted model to quantify the value of climate pre-training.

### Results

| Metric | ClimateBERT | Vanilla DistilRoBERTa |
|---|---|---|
| Silhouette (best layer) | −0.0096 (Layer 1, mean-pool) | −0.0254 |
| **1-NN Core Frame Accuracy** | **49.1%** | **45.3%** |
| 3-NN Core Frame Accuracy | 53.4% | — |
| NN Frame Overlap Rate | 78.0% | — |
| Intra/inter gap | 0.0036 | 0.0019 |
| Sentence match rate | 20.1% | — |

**Key findings**:

- **Layer 1 is best** — the earliest transformer layer (closest to raw token embeddings) contains the most frame-discriminative signal. Deep layers (5–6) lose frame specificity to task-agnostic contextual blending. This guides LoRA placement: adapters on layers 1–3 are highest priority.
- **1-NN at 49.1%** — even though frames are not globally separable (silhouette < 0), they are *locally* separable. Each paragraph's nearest neighbour is more likely to share its frame than not. This validates metric/prototypical learning approaches.
- **78.0% frame overlap** — even when the exact core frame doesn't match, the nearest neighbour shares at least one frame (core or peripheral). The representational space is semantically structured.
- **Sentence match rate 20.1%** — individual sentences are poor frame classifiers on their own; frame signal is a paragraph-level phenomenon.
- **ClimateBERT beats vanilla** on all metrics: +3.8pp on 1-NN accuracy, silhouette nearly 2× better, intra/inter gap 89% larger. Domain pre-training is genuinely beneficial.

**Outputs**: `layerwise_silhouette.png`, `climate_vs_vanilla.png`, `knn_probe_results.txt`, `sentence_decomposition.txt`

---

## 7. Round 4 — Attention Head Analysis

**Script**: `poc_round4_attention.py`

### What it does

Five attention-level tests across all 72 heads (6 layers × 12 heads):

1. **Head-level frame attention** — for each head, compute ratio of attention given to frame vs. non-frame tokens
2. **Attention rollout** — aggregate attention through all layers to get the effective CLS → token attention
3. **Per-paragraph attention heatmaps** — token-level visualizations for 3 sample paragraphs
4. **Frame token cross-attention** — measure how much frame tokens attend to each other (last layer)
5. **Attention entropy** — diffuseness of attention per head; correlation with frame focus

### Results

| Metric | Value |
|---|---|
| Global mean frame attention ratio | **0.675×** |
| Paragraphs where frame tokens get more attention | 2 / 227 (0.9%) |
| Attention rollout frame/non-frame ratio | **0.558×** |
| **Best head** | **Layer 1, Head 7 (4.800×)** |
| Second best | Layer 1, Head 11 (1.393×) |
| Heads with ratio > 1.0 (frame-attending) | 5 / 72 (7%) |
| Worst head | Layer 5, Head 6 (0.042×) |
| Frame→Frame attention (last layer) | 0.00572 |
| Frame→Non-frame attention | 0.00876 |
| F→F / F→NF ratio | **0.653×** (no self-clustering) |
| Mean attention entropy | 3.25 bits |
| Entropy vs frame ratio correlation | −0.078 (negligible) |

**Per-layer average attention ratio**:

| Layer | Frame Attention Ratio | Interpretation |
|---|---|---|
| 1 | **1.216×** | ✓ Frame-attending |
| 2 | 0.817× | Neutral |
| 3 | 0.649× | Below average |
| 4 | 0.468× | Avoids frame tokens |
| 5 | 0.256× | Strongly avoids frame tokens |
| 6 | 0.645× | Below average |

**Key findings**:

- **Only Layer 1 averages above 1.0** — early layers do attend to semantically significant (frame-evoking) tokens, but this signal is progressively diluted in deeper layers.
- **The base model suppresses frame tokens globally** (mean 0.675×). Punctuation (`.`, `,`) dominates attention rollout — a known artefact of DistilRoBERTa's pre-training on general text.
- **Layer 1, Head 7 at 4.800×** is a strong frame-attending head that could be a target for LoRA fine-tuning to amplify.
- **No frame→frame self-clustering** in the last layer (0.653× ratio) — frame-evoking tokens do not form internal semantic networks in the base model. Fine-tuning is required to establish this structure.
- **Entropy vs frame ratio correlation = −0.078** — more focused heads do not preferentially attend to frame tokens. Frame sensitivity is not driven by attention sharpness alone.

**Outputs**: `head_frame_attention.png`, `attention_rollout.png`, `attention_heatmap_P*.png`, `cross_attention_matrix.png`, `attention_entropy.png`, `round4_attention_results.txt`

---

## 8. Phase 0 — Fine-Tuning Results

**Script**: `poc_finetune_setfit.py`

### Methodology

Two complementary fine-tuning strategies evaluated via **Stratified 5-Fold Cross-Validation** (no held-out test set — too few samples):

**Strategy A — LoRA Fine-Tune**:
- LoRA adapters (rank=8, α=16, dropout=0.1) on Q and V projections of all 6 layers
- Only 742,662 trainable parameters out of 83,046,156 total (**0.89%**)
- Trained with CrossEntropy + class weights inversely proportional to frequency
- 8 epochs, lr=2e-4, batch size=8, warmup ratio=0.1

**Strategy B — Prototype Classifier (metric learning baseline)**:
- No gradient updates — zero trainable parameters
- Build a class centroid from mean-pooled embeddings of training samples
- Predict via cosine similarity to nearest centroid
- Pure metric learning; establishes the "free" upper bound for embedding quality

**Strategy C — Dictionary-Based Token Extraction**:
- Match canonical tokens from Token Summary sheet against paragraph text (exact substring)
- Evaluates precision/recall/F1 against ground-truth annotated tokens

### Results

#### Frame Classification (5-Fold CV)

| Method | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Random baseline | 16.7% | ~17% | — |
| Zero-shot (base model) | 19.0% | ~17% | Cosine sim to frame descriptions |
| Prototype / Cosine (no training) | 45.3% ± 10.7% | 33.9% ± 10.1% | Free metric baseline |
| **LoRA Fine-Tuned** | **56.1% ± 9.2%** | **42.7% ± 10.2%** | 0.89% trainable params |

#### Per-Class F1 (LoRA, aggregated across 5 folds)

| Frame | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Causal and Attribution Effect | 0.70 | 0.51 | **0.59** | 96 |
| Impact and Consequences | 0.64 | 0.80 | **0.71** | 59 |
| Epistemic and Scientific Research | 0.61 | 0.59 | **0.60** | 29 |
| Action and Solutions | 0.28 | 0.45 | **0.34** | 29 |
| Socio-Political and Economic | 0.31 | 0.36 | **0.33** | 11 |
| Temporal and Scalar | 0.00 | 0.00 | **0.00** | 8 |
| **Macro average** | 0.42 | 0.45 | **0.43** | 232 |

#### Token Extraction (Dictionary-Based)

| Metric | Value |
|---|---|
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | **0.000** |

Token extraction F1 = 0.0% because the Token Summary canonical tokens (high-level category terms) do not overlap with the paragraph-specific ground-truth annotations (context-specific phrases). This confirms that **dictionary-based extraction is insufficient** — a trained BIO token tagger is required for Phase 1.

### Interpretation

- **LoRA jumps from 19% → 56%** with only 0.89% of parameters updated — strong evidence that the fine-tuning direction is correct and the base model has latent frame signal that LoRA unlocks.
- **Temporal and Scalar (n=8) scores 0.00 F1** — critically under-sampled. The model has no capacity to learn from 6–7 training examples per fold. Annotating 90+ more paragraphs for this class is the highest priority.
- **Prototype at 45.3%** is a strong free baseline, confirming the metric learning approach (nearest centroid) is viable and worth combining with LoRA in Phase 1.
- **High fold variance (±9–10%)** is expected at this data scale. Results will stabilise with more data.

**Outputs**: `finetune_confusion_matrix_lora.png`, `finetune_confusion_matrix_prototype.png`, `finetune_comparison.png`, `finetune_cv_results.txt`

---

## 9. Key Findings Summary

| Finding | Detail | Implication |
|---|---|---|
| **Base model cannot separate frames** | Silhouette = −0.0136 | Fine-tuning is mandatory |
| **Local structure exists** | 1-NN = 49.1%, frame overlap = 78% | Metric learning viable |
| **Zero-shot is near-random** | 19.0% vs 17% baseline | Frame semantics not in base embeddings |
| **Layer 1 is best** | Sil = −0.0096, attention ratio = 1.216× | LoRA priority on early layers |
| **Domain pre-training helps** | +3.8pp 1-NN vs vanilla | ClimateBERT is the right base |
| **LoRA works at 232 samples** | 56.1% acc (+37pp vs zero-shot) | Fine-tuning direction confirmed |
| **Minority classes fail** | Temporal F1 = 0.00 | Need 90+ more annotations each |
| **Dict-based tokens fail** | F1 = 0.00 | Need trained BIO tagger |
| **Tokenizer splits 45% of tokens** | 44.8% multi-subword | Mean-pool spans, not single tokens |
| **Attention suppresses frame tokens** | 0.675× global ratio | Fine-tuning must rewire attention |

---

## 10. Architecture Roadmap

### Phase 0 — COMPLETE (232 samples)

- ✅ LoRA (rank=8) sequence classifier — 56.1% accuracy
- ✅ Prototype cosine classifier — 45.3% accuracy
- ✅ Dictionary token extraction — 0.0% F1 (baseline established)

### Phase 1 — Target ~500 samples (80+ per class)

**Requirements**: Annotate ~270 more paragraphs, prioritising Temporal and Scalar (+92) and Socio-Political (+70).

**Architecture**:
```
ClimateBERT (distilroberta-base-climate-f)
    + LoRA adapters (rank=16, Q+V, all 6 layers)
    ├── Head 1: BIO Token Tagger
    │     Linear(768 → 3) + CRF
    │     Tags: B-FRAME_EVOKE, I-FRAME_EVOKE, O
    │     Loss: CRF NLL with class weights
    └── Head 2: Frame Classifier
          Linear(768 → 6) on [CLS]
          Loss: FocalLoss(γ=2) for class imbalance
Joint loss: 0.3·L_token + 0.7·L_frame
```

**Data augmentation**:
- Back-translation via Helsinki-NLP/opus-mt-en-de → de-en (frame-evoking tokens kept fixed)
- EDA (Easy Data Augmentation) on non-frame tokens only

### Phase 2 — Target ~1,000 samples (150+ per class)

**Architecture** (full 3-head):
```
ClimateBERT + LoRA (rank=16)
    ├── Head 1: BIO Token Tagger + CRF
    ├── Head 2: Core Frame Classifier (FocalLoss)
    │     + Peripheral Frame Multi-label (BCEWithLogitsLoss)
    └── Head 3: Contrastive Token Embedding
          Frame-evoking token spans → MLP(768→256→128) → L2-norm
          Soft SupCon loss (frame-overlap-weighted similarity)
Joint loss: λ₁·L_token + λ₂·L_frame + λ₃·L_contrastive
```

Proper 70/15/15 train/val/test split becomes feasible.

### Architecture Design Principles

| Decision | Choice | Reason |
|---|---|---|
| BIO tags | Generic (3 tags) not frame-specific | Too few samples for 18+ tags |
| Frame classifier | FocalLoss(γ=2) | 12:1 imbalance |
| LoRA rank | 8 (Phase 0–1), 16 (Phase 2) | Minimal params, strong regularisation |
| LoRA targets | Q + V projections | Attention rewriting, not MLP |
| Best layers | 1–3 (highest frame signal) | Layer analysis confirms early layers |
| Token pooling | Mean over subword span | 44.8% of tokens are multi-subword |
| Contrastive | Soft SupCon (overlap-weighted) | Handles partial frame overlap |

---

## 11. Next Steps

### Immediate (data)
- [ ] Annotate 92 more *Temporal and Scalar* paragraphs (currently only 8)
- [ ] Annotate 70 more *Socio-Political and Economic* paragraphs (currently only 11)
- [ ] Resolve annotator disagreement between V4 and 12-Ann counts (Action/Solutions: 63 vs 29)
- [ ] Fill in empty RST and Date columns in V4 dataset

### Immediate (modelling)
- [ ] Investigate token extraction failure — inspect Token Summary sheet vs ground-truth alignment
- [ ] Experiment with BIO tagger training on current 232 samples as ablation
- [ ] Add data augmentation pipeline (back-translation)

### Phase 1 (once 500 samples available)
- [ ] Train Phase 1 dual-head LoRA model (BIO + classifier)
- [ ] Evaluate BIO F1 per frame class
- [ ] Compare LoRA rank 8 vs 16 on classification accuracy
- [ ] Report macro-F1 per class with 95% CI (5-fold CV)

### Publication
- [ ] Paper draft requires Phase 1 results (trained model F1 scores)
- [ ] Target: ACL 2027 or EMNLP 2026 (fine-tuning results + annotation study)
- [ ] Minimum viable contribution: 500 paragraphs + LoRA Phase 1 + ablation study

---

*Report generated: March 2026*  
*Model: `climatebert/distilroberta-base-climate-f`*  
*Scripts: 5 POC rounds + Phase 0 fine-tuning*  
*Outputs: 20 figures, 7 result files*

