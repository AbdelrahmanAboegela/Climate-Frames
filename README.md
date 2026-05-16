# ClimateBERT Climate-Frames Research Workspace

Status as of 2026-05-16: the repository is reorganized around `climate_frames/`, `experiments/`, `data/`, `outputs/`, and `requirements/`, and the benchmark suite has been rerun on the current workbook.

## 1. Executive Summary

This workspace studies climate framing with three prediction targets:

1. Core-frame classification at paragraph level
2. Trigger-span extraction for frame-evoking text
3. Peripheral-frame prediction as a multi-label task

The current data is large enough for meaningful benchmarking, but it is not large enough to ignore leakage, class imbalance, or annotation inconsistency. The main conclusions from the full rerun are:

- The defensible evaluation unit is the merged exact-text paragraph, not the raw workbook row.
- ClimateBERT is useful, but the strongest current core-frame system is still a combination of lexical and domain-semantic signals rather than a single end-to-end fine-tuned model.
- The best core macro-F1 under standard merged 5-fold CV is the soft-vote ensemble at `0.5395`, with fusion logistic regression essentially tied at `0.5383`.
- The best all-around core model is the fusion classifier, because it is stronger under grouped robustness checks and much better calibrated (`ECE 0.0250`).
- The best neural fine-tuning result is still below the best fusion model. Among the LoRA variants tried, `Logit-Adjusted CE` reached the best neural macro-F1 (`0.5061`), while `Class-Balanced Focal` reached the best neural accuracy (`0.6007`).
- Trigger extraction is viable. The best entity-level trigger result is now the LoRA + CRF model at `0.3415` entity F1.
- Peripheral prediction is also viable for the better-supported labels. The best current peripheral system is the TF-IDF + ClimateBERT probability ensemble at `0.5125` micro-F1 / `0.4334` macro-F1.
- Frame-role supervision is not ready for modeling, because token and role counts mismatch in `988/1081` raw rows.

This README is written as a full research workspace guide: what data is active, why the evaluation protocol looks the way it does, what each experiment tried to answer, which approaches failed, and which findings are strong enough to anchor a paper.

## 1.1 Quick Start

From the repository root:

```bash
python -m pip install -r requirements/requirements-research.txt
python -m experiments.analysis.poc_dataset_audit
```

Useful conventions:

- run all experiment commands from the repository root so the `climate_frames` and `experiments` packages resolve cleanly
- the default active workbook lives in `data/current/`
- committed benchmark figures and text summaries are written to `outputs/`

## 2. Research Questions

The current repository is organized around five practical questions:

1. Does ClimateBERT already separate climate frames in its base representation space?
2. What is the strongest current model for paragraph-level core-frame classification?
3. Can frame triggers be extracted from the existing trigger annotations with reasonable span quality?
4. Are peripheral frames trainable as a multi-label task with the current support distribution?
5. Which parts of the annotation are already model-ready, and which are still blocked by schema or consistency issues?

## 3. Active Data

### 3.1 Primary workbook

Default workbook used by the scripts:

`E:\Frames\data\current\14 May 2026 12 articles Ann. Core Peripheral RST and FrameNET Structure.xlsx`

### 3.2 Legacy workbook retained for comparison

Legacy workbook:

`E:\Frames\data\legacy\12 articles Ann. Core Peripheral RST and FrameNET Structure.xlsx`

The older 232-row dataset is fully contained in the expanded workbook. After exact-text merge, all `232/232` legacy texts remain present and the new workbook contributes `597` additional unique texts.

### 3.3 Why exact-text merge is the default

The expanded workbook contains repeated exact texts with annotation variation. Those repeats appear to reflect re-annotation or refinement rather than accidental noise, but they create serious leakage if used naively in ordinary row-level cross-validation.

Current audit:

| Quantity | Value |
|---|---:|
| Raw rows kept after cleanup | 1081 |
| Unique merged texts | 829 |
| Exact duplicate groups | 252 |
| Rows inside duplicate groups | 504 |
| Extra duplicate rows beyond first copy | 252 |
| Full row duplicates | 17 |
| Duplicate groups with conflicting core labels | 0 |

Leakage audit with TF-IDF + Logistic Regression:

| Evaluation protocol | Accuracy | Macro-F1 |
|---|---:|---:|
| Raw rows, ordinary stratified CV | 0.6735 | 0.6512 |
| Raw rows, grouped by exact text | 0.5429 | 0.3793 |
| Merged unique texts | 0.5380 | 0.4055 |

This is one of the most important findings in the project. Any paper built on the raw row-level CV numbers would materially overstate model performance.

### 3.4 Label structure

Core frames after merge:

| Core frame | Count |
|---|---:|
| Causal and Attribution Effect | 231 |
| Epistemic and Scientific Research | 209 |
| Impact and Consequences | 183 |
| Action and Solutions | 152 |
| Temporal and Scalar | 39 |
| Socio-Political and Economic | 15 |

Other useful inventory facts:

- Normalized peripheral labels: `40`
- Average peripheral labels per merged text: `1.745`
- Peripheral labels with support `>= 10`: `28`
- Peripheral labels with support `>= 20`: `24`

### 3.5 Annotation readiness by task

The dataset is not equally ready for every modeling target.

| Annotation component | Current status | Evidence |
|---|---|---|
| Core frame | ready | no core-label conflicts across duplicate groups |
| Peripheral frame | partly ready | 28 labels already have support `>= 10` |
| Trigger tokens | mostly ready | only 20 merged texts with trigger annotations fail exact token-span matching |
| Frame roles | not ready | `988/1081` token-role count mismatches |

In other words:

- core classification is ready
- trigger extraction is ready enough to benchmark
- peripheral prediction is ready for head and torso labels
- role modeling should be deferred

## 4. Evaluation Protocol

### 4.1 Standard protocol

Unless noted otherwise, the main benchmark uses:

- merged exact-text paragraphs
- stratified 5-fold CV for core-frame classification
- shuffled 5-fold CV for peripheral multi-label prediction
- stratified 3-fold CV for trigger extraction

### 4.2 Robustness protocol

Because explicit article IDs are not yet wired into the workbook metadata, grouped evaluation currently uses an inferred robustness check:

- `12` contiguous article-like blocks are inferred from workbook order
- leave-one-block-out evaluation is used as a stress test for article-local style/topic leakage

This is useful, but it is still a proxy. It should be described in the paper as a robustness check, not as the final article-aware split.

### 4.3 Uncertainty protocol

The repository now includes bootstrap uncertainty analysis on out-of-fold predictions for the strongest classical/fusion core models:

- `4000` bootstrap resamples
- 95% confidence intervals for accuracy and macro-F1
- paired bootstrap deltas for key model comparisons

One important detail:

- the main benchmark tables below report mean fold scores
- the uncertainty analysis reports aggregate out-of-fold scores and their bootstrap confidence intervals

Those numbers are close, but they are not mathematically identical.

### 4.4 Metrics

Primary metrics by task:

- core classification: accuracy and macro-F1
- trigger extraction: entity precision/recall/F1 and token precision/recall/F1
- peripheral prediction: micro-F1 and macro-F1
- calibration: expected calibration error, Brier score, and negative log-likelihood

Macro-F1 is the most important headline metric for core-frame classification because the class distribution is strongly imbalanced.

## 5. Repository Map

Top-level layout:

- `climate_frames/`
  - shared dataset and path utilities
- `experiments/analysis/`
  - representation, probing, and audit entrypoints
- `experiments/benchmarks/`
  - supervised benchmark and ablation entrypoints
- `data/current/`
  - active workbook used by default
- `data/legacy/`
  - older workbook retained for comparison
- `data/reference/`
  - supporting annotation/reference files
- `outputs/`
  - committed plots and result summaries
- `requirements/`
  - reproducible environment files

### 5.1 Shared data utility

- `climate_frames/dataset.py`
  - central loader for the expanded workbook
  - supports `merge`, `first`, and `none` duplicate modes
  - normalizes token summaries
  - aligns trigger-token strings to text spans

### 5.2 Analysis scripts

- `experiments/analysis/poc_dataset_audit.py`
  - dataset profile, duplicate analysis, leakage audit, trigger coverage, role alignment
- `experiments/analysis/poc_base_model_analysis.py`
  - paragraph UMAPs, token UMAP, frame similarity, MLM probes, sample attention analysis
- `experiments/analysis/poc_round2_analysis.py`
  - zero-shot core classification, taxonomy mapping, tokenizer coverage
- `experiments/analysis/poc_round3_analysis.py`
  - layer-wise probing, k-NN probing, sentence decomposition, ClimateBERT vs vanilla comparison
- `experiments/analysis/poc_round4_attention.py`
  - head-level attention ratios, rollout, cross-attention, entropy

### 5.3 Benchmark scripts

- `experiments/benchmarks/poc_finetune_setfit.py`
  - TF-IDF core baseline
  - frozen ClimateBERT embedding baseline
  - initial LoRA sequence-classification baseline
  - dictionary trigger baseline
  - initial peripheral multi-label baseline
- `experiments/benchmarks/poc_setfit_baseline.py`
  - SetFit baseline on merged exact-text data
- `experiments/benchmarks/poc_token_classification.py`
  - LoRA BIO token classifier for generic trigger detection
- `experiments/benchmarks/poc_additional_experiments.py`
  - fusion and ensemble core models
  - peripheral global threshold sweeps
  - core error analysis
- `experiments/benchmarks/poc_grouped_calibrated_experiments.py`
  - inferred grouped evaluation
  - calibration and selective prediction
- `experiments/benchmarks/poc_peripheral_chain_experiments.py`
  - classifier chains vs simpler peripheral fusion setups
- `experiments/benchmarks/poc_trigger_threshold_sweep.py`
  - thresholded trigger decoding
- `experiments/benchmarks/poc_trigger_structured_decoder.py`
  - transition-aware Viterbi decoding
- `experiments/benchmarks/poc_trigger_crf.py`
  - LoRA encoder with a CRF sequence decoder for trigger spans
- `experiments/benchmarks/poc_lora_imbalance_experiments.py`
  - effective-weight CE vs class-balanced focal vs logit-adjusted LoRA
- `experiments/benchmarks/poc_core_uncertainty_analysis.py`
  - bootstrap CIs and paired bootstrap deltas for the strongest core models

### 5.4 Outputs

All refreshed outputs are written to:

`E:\Frames\outputs\`

The text reports in that folder are the authoritative numeric record for the rerun.

## 6. Dataset Audit Results

Primary output:

- `outputs/dataset_audit_results.txt`

Headline findings:

- duplicate groups preserve core labels, but token and role annotations vary heavily
- trigger annotations are usable enough for supervised trigger modeling
- frame roles are not token-aligned enough for supervised role modeling
- the long tail of peripheral labels is still real, but the head and torso are already trainable

Important audit counts:

| Audit item | Value |
|---|---:|
| Token-variant duplicate groups | 235 |
| Role-variant duplicate groups | 233 |
| Peripheral-variant duplicate groups | 2 |
| Rows with no trigger annotations | 5 |
| Rows with no trigger type found in text | 20 |
| Exact token-role count matches | 93 |
| Token-role count mismatches | 988 |

Paper implication:

- it is reasonable to benchmark core frames, peripheral frames, and trigger extraction now
- it is not reasonable to claim frame-role modeling is ready

## 7. Representation and Probing Analyses

These analyses answer the question: what does ClimateBERT already know before we design stronger supervised models?

### 7.1 Round 1: Base representation structure

Key outputs:

- `outputs/paragraph_umap_cls_core.png`
- `outputs/paragraph_umap_mean_core.png`
- `outputs/token_umap_core.png`
- `outputs/frame_similarity_heatmap.png`
- `outputs/mlm_probing_results.txt`
- `outputs/attention_analysis_results.txt`

Key results:

- CLS silhouette: `-0.0072`
- mean-pool silhouette: `-0.0175`
- token embeddings extracted: `6493`

Interpretation:

- ClimateBERT already places climate language into a coherent domain space.
- That space is not linearly separated enough to solve the six core frames on its own.
- Base semantics help, but supervision still matters.

### 7.2 Round 2: Zero-shot, similarity, taxonomy, and tokenizer coverage

Key outputs:

- `outputs/zero_shot_results.txt`
- `outputs/intra_inter_similarity.png`
- `outputs/frame_taxonomy_heatmap.png`
- `outputs/frame_taxonomy_umap.png`
- `outputs/tokenizer_coverage.png`

Key results:

- zero-shot top-1 core accuracy: `28.6%`
- zero-shot top-3 core accuracy: `68.8%`
- intra-frame similarity: `0.9679`
- inter-frame similarity: `0.9652`
- similarity gap: `0.0027`
- multi-subword trigger expressions: `74.8%`

Interpretation:

- the model has useful domain semantics, but frame discrimination is weak without supervision
- the very small intra-vs-inter similarity gap confirms that frames are not well separated in base embedding space
- trigger modeling needs span-aware token supervision because most trigger expressions are not single lexical items

### 7.3 Round 3: Layer-wise probing and model comparison

Key outputs:

- `outputs/layerwise_silhouette.png`
- `outputs/knn_probe_results.txt`
- `outputs/sentence_decomposition.txt`
- `outputs/climate_vs_vanilla.png`
- `outputs/gradient_saliency.txt`

Key results:

- best CLS silhouette: `0.0000` at embedding layer
- best mean-pool silhouette: `-0.0075` at layer 1
- 1-NN accuracy: `43.8%`
- 3-NN accuracy: `48.1%`
- sentence-to-parent-frame match rate: `26.4%`
- ClimateBERT slightly but consistently beats vanilla DistilRoBERTa

Interpretation:

- early layers contain the most useful frame-separation signal
- ClimateBERT is better than vanilla, but not by a large margin
- many paragraphs contain mixed framing cues, which helps explain why sentence decomposition is not cleanly aligned to the paragraph label

### 7.4 Round 4: Attention analysis

Key outputs:

- `outputs/head_frame_attention.png`
- `outputs/attention_rollout.png`
- `outputs/attention_heatmap_P0.png`
- `outputs/attention_heatmap_P2.png`
- `outputs/attention_heatmap_P15.png`
- `outputs/cross_attention_matrix.png`
- `outputs/attention_entropy.png`
- `outputs/round4_attention_results.txt`

Key results:

- global mean frame-attention ratio across heads: `0.8155`
- strongest head: `Layer 1, Head 7 = 12.521x`
- mean rollout ratio, frame vs non-frame: `0.5461`
- frame tokens get more rollout attention in only `6/809` paragraphs

Interpretation:

- a few early heads are clearly frame-sensitive
- aggregate attention is still not reliable enough to treat as a trigger-extraction method
- attention is useful as descriptive evidence, not as the main predictive mechanism

## 8. Core-Frame Classification

This is the strongest and most complete part of the current project.

### 8.1 Core methods tried and why

The project deliberately tried a range of model families because the dataset sits in an awkward middle ground: it is larger than a toy benchmark, but still small enough that leakage, imbalance, and overfitting matter.

#### A. TF-IDF + Logistic Regression

Purpose:

- measure how much framing can be recovered from surface lexical evidence alone
- establish a cheap baseline that is easy to interpret

What it tells us:

- framing is not purely semantic abstraction
- simple lexical signals are real and surprisingly competitive

#### B. SetFit

Purpose:

- test a fast sentence-transformer approach without full generative prompting
- evaluate whether compact contrastive sentence embeddings are enough for the task

What it tells us:

- SetFit is operational and better than the plain TF-IDF baseline on accuracy
- it does not solve minority-class recall and is not competitive on macro-F1 with the best ClimateBERT-based approaches

#### C. Frozen ClimateBERT Embeddings + Logistic Regression

Purpose:

- test whether domain pretraining alone, without task fine-tuning, already provides a strong paragraph representation

What it tells us:

- this is the strongest single baseline before fusion
- the result is strong enough that any fine-tuned model must meaningfully beat it to justify its extra complexity

#### D. Frozen ClimateBERT Embeddings + LinearSVC

Purpose:

- test whether a margin-based classifier works better than logistic regression on the same embeddings

What it tells us:

- it does not outperform logistic regression here
- probability-aware logistic models remain more useful, especially when later calibration and fusion analyses matter

#### E. Initial LoRA Sequence Classifier

Purpose:

- test whether lightweight domain-task adaptation improves over frozen embeddings
- keep compute manageable while still giving ClimateBERT trainable task-specific capacity

What it tells us:

- LoRA improves headline accuracy
- the initial weighted-CE version does not beat the frozen embedding baseline on macro-F1

#### F. Imbalance-Aware LoRA Variants

Purpose:

- directly address the tail-class problem without collecting new data

Variants tried:

1. Effective-number weighted CE
2. Class-Balanced Focal
3. Logit-Adjusted CE

What they tell us:

- loss design matters for this dataset
- both Class-Balanced Focal and Logit-Adjusted CE improve macro-F1 over the earlier weighted-CE LoRA
- even the best neural loss variants still do not overtake the best fusion/ensemble core models

#### G. TF-IDF + ClimateBERT Fusion Logistic Regression

Purpose:

- concatenate sparse lexical features with dense ClimateBERT paragraph embeddings

What it tells us:

- lexical and domain-semantic signals are complementary
- this is the best single classifier overall once robustness and calibration are considered

#### H. Soft-Vote Ensemble

Purpose:

- combine the posterior probabilities from TF-IDF LR and ClimateBERT LR without early feature concatenation

What it tells us:

- the ensemble is the best headline macro-F1 result under standard merged 5-fold CV
- the gain over fusion is small enough that calibration and grouped robustness become the deciding criteria

#### I. Stacked Meta-Classifier

Purpose:

- test whether a learned combination of base-model probabilities outperforms simple averaging

What it tells us:

- stacking can lift raw accuracy
- it does not deliver the best macro-F1 or calibration
- the simpler fusion model remains easier to defend

### 8.2 Main core results

Main benchmark outputs:

- `outputs/finetune_cv_results.txt`
- `outputs/setfit_cv_results.txt`
- `outputs/additional_experiments_results.txt`
- `outputs/lora_imbalance_results.txt`

Mean fold results:

| Method | Accuracy | Macro-F1 | Reading |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | 0.5380 | 0.4055 | lexical-only baseline |
| SetFit (`all-MiniLM-L6-v2`) | 0.5815 | 0.4100 | workable, but weak on tail classes |
| Frozen ClimateBERT + Logistic Regression | 0.5838 | 0.5115 | strongest single baseline before fusion |
| Frozen ClimateBERT + LinearSVC | 0.5778 | 0.4735 | below embedding LR |
| LoRA, effective-weight CE | 0.5983 | 0.4895 | neural baseline after rerun |
| LoRA, Class-Balanced Focal | 0.6007 | 0.5055 | best neural accuracy |
| LoRA, Logit-Adjusted CE | 0.5766 | 0.5061 | best neural macro-F1 |
| TF-IDF + ClimateBERT Fusion LR | 0.6067 | 0.5383 | best single classifier |
| Soft-Vote Ensemble | 0.6031 | 0.5395 | best standard-CV macro-F1 |

Core takeaway:

- the best core result is still a hybrid rather than a single neural model
- fusion and soft-vote are effectively co-leaders under standard CV
- the best LoRA variants narrow the gap, but they do not close it
- the stacked meta-classifier is reported later in the robustness and calibration section, where its trade-offs can be described on comparable terms

### 8.3 Minority-class behavior

The two rarest core frames remain the critical weakness:

- `Socio-Political and Economic`
- `Temporal and Scalar`

What helped:

- frozen ClimateBERT embeddings helped both minority classes more than TF-IDF alone
- imbalance-aware LoRA losses improved their recall relative to the initial weighted-CE LoRA

What did not happen:

- no model made the rarest class stable enough to claim the problem is solved

Example pattern from the neural loss sweep:

- Logit-Adjusted CE raised recall on the rare classes, which improved macro-F1
- that gain came with a noticeable accuracy cost, showing the usual imbalance trade-off clearly

### 8.4 Error analysis

Primary output:

- `outputs/additional_core_error_analysis.txt`

Most frequent confusion directions:

- Causal and Attribution Effect -> Impact and Consequences
- Causal and Attribution Effect -> Action and Solutions
- Causal and Attribution Effect -> Epistemic and Scientific Research
- Action and Solutions -> Causal and Attribution Effect
- Epistemic and Scientific Research -> Causal and Attribution Effect

Interpretation:

- the classes are semantically adjacent, not arbitrary
- many errors look like framing overlap rather than total semantic failure
- this makes macro-category success plausible even when finer distinctions remain difficult

### 8.5 Robustness, calibration, and selective prediction

Primary outputs:

- `outputs/grouped_calibrated_results.txt`
- `outputs/core_reliability_diagram.png`
- `outputs/core_selective_accuracy.png`

Grouped leave-one-block-out robustness:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Fusion LR | 0.6236 | 0.4957 |
| Soft-Vote Ensemble | 0.6181 | 0.4898 |
| Stacked Meta-LR | 0.6216 | 0.4775 |

Calibration on merged out-of-fold probabilities:

| Method | ECE | Brier | NLL |
|---|---:|---:|---:|
| Fusion LR | 0.0250 | 0.5288 | 1.0233 |
| Soft-Vote Ensemble | 0.1621 | 0.5682 | 1.1414 |
| Stacked Meta-LR | 0.0687 | 0.5299 | 1.0595 |

Selective prediction example:

- Fusion LR at 50% coverage: `0.7319` accuracy / `0.6575` macro-F1

Interpretation:

- the fusion model is the most attractive practical core system right now
- it is strong, stable, and already well calibrated
- if the paper needs one preferred model recommendation, fusion is the cleanest answer

### 8.6 Statistical uncertainty

Primary outputs:

- `outputs/core_uncertainty_results.txt`
- `outputs/core_macro_f1_bootstrap.png`

Bootstrap 95% confidence intervals on out-of-fold predictions:

| Method | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF LR | `0.5380 [0.5030, 0.5718]` | `0.4070 [0.3618, 0.4523]` |
| Frozen ClimateBERT LR | `0.5838 [0.5501, 0.6188]` | `0.5092 [0.4635, 0.5519]` |
| Fusion LR | `0.6068 [0.5730, 0.6393]` | `0.5350 [0.4838, 0.5830]` |
| Soft-Vote Ensemble | `0.6031 [0.5706, 0.6357]` | `0.5368 [0.4856, 0.5848]` |

Paired bootstrap deltas on macro-F1:

- Fusion LR vs Frozen ClimateBERT LR: `+0.0252`, 95% CI `[+0.0067, +0.0437]`, approx `p=0.0050`
- Soft-Vote vs Frozen ClimateBERT LR: `+0.0269`, 95% CI `[+0.0098, +0.0449]`, approx `p=0.0025`
- Soft-Vote vs Fusion LR: `+0.0017`, 95% CI `[-0.0156, +0.0204]`, approx `p=0.8515`

Interpretation:

- both fusion and soft-vote are meaningfully better than the frozen ClimateBERT linear baseline
- the difference between fusion and soft-vote is too small to call meaningful with the current data
- this is one reason the grouped/calibration evidence matters so much for final model choice

## 9. Trigger Extraction

This part of the project asks a different question from core classification:

- not "which frame is present?"
- but "where is the frame evoked in the paragraph?"

This is harder because the current trigger supervision mixes lexical phrases, multi-subword spans, and some alignment noise.

### 9.1 Methods tried and why

#### A. Dictionary trigger baseline

Purpose:

- test whether the token summary alone is enough as a direct lookup system

Outcome:

- very high sparsity
- extremely poor recall

#### B. LoRA BIO token classifier

Purpose:

- learn generic trigger-vs-non-trigger token labels from the merged annotations

Outcome:

- good token recall
- only moderate exact-entity quality

#### C. Thresholded decoding

Purpose:

- check whether simple probability thresholding can improve span extraction beyond argmax decoding

Outcome:

- token F1 improves at high thresholds
- entity F1 does not

#### D. Transition-aware Viterbi decoding

Purpose:

- enforce BIO-style sequence consistency without retraining the encoder

Outcome:

- much better precision
- too much recall loss
- entity F1 effectively unchanged

#### E. LoRA + CRF

Purpose:

- test a genuine structured sequence model rather than post-hoc threshold or Viterbi heuristics

Outcome:

- best current entity-level trigger result in the repository

### 9.2 Trigger results summary

Primary outputs:

- `outputs/token_classification_results.txt`
- `outputs/trigger_threshold_sweep_results.txt`
- `outputs/trigger_structured_decoder_results.txt`
- `outputs/trigger_crf_results.txt`

| Method | Entity F1 | Token F1 | Main reading |
|---|---:|---:|---|
| Dictionary token baseline | 0.0767 | n/a | far too sparse to use directly |
| LoRA BIO token classifier | 0.3129 | 0.5287 | good token detector, weak exact span matching |
| Thresholded decode, best entity setting | 0.3159 | 0.5275 | no real entity gain over argmax |
| Thresholded decode, best token setting | 0.2414 | 0.5682 | better token F1, worse entity spans |
| Transition-aware Viterbi | 0.3137 | 0.5251 | precision improves, recall collapses |
| LoRA + CRF | 0.3415 | 0.4588 | best entity-level span result |

Important nuance:

- if the paper emphasizes token tagging quality, the plain token model remains competitive
- if the paper emphasizes exact trigger-span recovery, the CRF model is the best current choice

### 9.3 What the trigger experiments teach us

1. Trigger extraction is feasible with the current data.
2. Exact-span quality is substantially harder than token detection.
3. Threshold tuning alone is not enough.
4. Structure helps, but the form of structure matters.
5. A learned CRF decoder is better than ad hoc Viterbi constraints for entity-level spans on this dataset.

This is a useful paper result because it moves the story from "can this be modeled at all?" to "which kind of decoder best matches the current annotation quality?"

## 10. Peripheral-Frame Prediction

Peripheral prediction is a multi-label task with a clear long tail, so the main question is not whether every label is solved. The real question is whether the head and torso of the ontology are already trainable enough to report responsibly.

### 10.1 Methods tried and why

#### A. One-vs-rest logistic baselines

Purpose:

- create a simple and interpretable multi-label benchmark
- test lexical and dense semantic signals separately

#### B. Global threshold sweeps

Purpose:

- check whether a single decision threshold is suppressing performance unfairly

#### C. Probability ensemble

Purpose:

- test the same lexical + semantic complementarity observed in core classification

#### D. Classifier chains

Purpose:

- model label dependence explicitly

Outcome:

- chains did not beat the simpler fusion-style systems

### 10.2 Peripheral results summary

Primary outputs:

- `outputs/finetune_cv_results.txt`
- `outputs/additional_experiments_results.txt`
- `outputs/peripheral_chain_results.txt`
- `outputs/peripheral_threshold_sweep.png`
- `outputs/peripheral_chain_thresholds.png`

Best results by family:

| Method | Micro-F1 | Macro-F1 | Reading |
|---|---:|---:|---|
| Initial peripheral baseline | 0.4788 | 0.4138 | trainable starting point |
| TF-IDF tuned threshold | 0.4575 | 0.3610 | weaker than semantic models |
| ClimateBERT tuned threshold | 0.4926 | 0.4302 | strong single-source model |
| Probability ensemble | 0.5125 | 0.4334 | best overall peripheral result |
| OVR fusion | 0.4954 | 0.4321 | strong but below probability ensemble |
| Chain fusion | 0.4862 | 0.4255 | extra complexity not justified |

Most learnable frequent peripheral labels include:

- `extreme weather and climate event`
- `mitigation and emission reduction`
- `policy and government action`
- `human health and public health`
- `environmental and ecological impact`

Interpretation:

- peripheral prediction is already strong enough to report for the better-supported labels
- the same lexical + ClimateBERT complementarity appears here too
- classifier chains are currently an unnecessary complication

## 11. Negative Results That Matter

This repo now has several negative or near-negative results that are worth reporting because they sharpen the paper rather than weaken it.

### 11.1 Raw row-level CV is misleading

This is not a small methodological detail. It changes the benchmark substantially and must be controlled.

### 11.2 SetFit is not the lead method

SetFit works after dependency repair, but it does not challenge the stronger ClimateBERT-based or fusion-based systems on macro-F1.

### 11.3 Full fine-tuning is not automatically better than frozen embeddings

The frozen ClimateBERT embedding baseline remains surprisingly strong. This is an important scientific result because it shows domain pretraining carries a lot of the signal already.

### 11.4 More complex combiners are not always better

Stacking improved raw accuracy but did not beat fusion on the combination of macro-F1, calibration, and grouped robustness.

### 11.5 Thresholding is not a real span solution

It shifts the token/entity trade-off, but it does not solve the entity-level extraction problem.

### 11.6 Classifier chains do not justify themselves here

Peripheral label dependence is real in theory, but the current data does not reward chain complexity enough to use it as the main method.

### 11.7 Role modeling should be deferred

This is a data-readiness issue, not a model choice issue.

## 12. Current Best Interpretations

### 12.1 Best core model for a paper headline

If the paper wants the strongest standard merged-CV macro-F1 number:

- Soft-Vote Ensemble: `0.5395` macro-F1

If the paper wants the most defensible all-around practical model:

- Fusion LR
  - standard merged CV: `0.6067` accuracy / `0.5383` macro-F1
  - grouped robustness: `0.6236` / `0.4957`
  - calibration: `ECE 0.0250`

### 12.2 Best neural story

The best neural fine-tuning story is no longer the initial weighted-CE LoRA model.

It is:

- `Class-Balanced Focal` for the strongest neural accuracy
- `Logit-Adjusted CE` for the strongest neural macro-F1

That gives the paper a more professional claim:

- LoRA can be improved materially by imbalance-aware loss design
- but the current data still favors hybrid lexical + domain-semantic systems overall

### 12.3 Best trigger story

The strongest span-level result is:

- LoRA + CRF: `0.3415` entity F1

The strongest token-level result is still more ambiguous, because thresholding can improve token F1 while hurting entity spans. For a paper centered on trigger-span extraction rather than generic token tagging, the CRF result is the right headline.

### 12.4 Best peripheral story

The cleanest peripheral result is:

- TF-IDF + ClimateBERT probability ensemble: `0.5125` micro-F1 / `0.4334` macro-F1

## 13. Limits and What Is Still Blocked

What is still blocked even after the full rerun:

1. True article-aware grouped evaluation remains incomplete until article IDs are wired in directly.
2. Rare core classes remain too small for strong stability claims.
3. Trigger annotations support modeling, but span precision is still limited by supervision quality.
4. Peripheral long-tail labels remain data-limited.
5. Role modeling is blocked by annotation alignment, not by model choice.

These are not reasons to delay the paper. They are the right limits to state clearly in the paper.

## 14. Recommended Paper Framing

The strongest defensible narrative is now:

1. The project contributes an expanded climate-framing dataset with nontrivial duplicate adjudication issues that require explicit leakage control.
2. ClimateBERT has domain knowledge relevant to climate framing, but base embedding geometry still does not separate frame classes cleanly.
3. Frozen ClimateBERT embeddings are already a strong baseline.
4. The best current core performance comes from combining lexical and ClimateBERT signals rather than relying on a single fine-tuned classifier.
5. Imbalance-aware LoRA improves the neural story but still does not overtake the hybrid fusion systems.
6. Trigger extraction is viable and benefits from structured decoding, with CRF outperforming simpler decoding heuristics on entity F1.
7. Peripheral prediction is feasible for the better-supported part of the ontology.
8. Role modeling is explicitly out of scope for the current data release because the annotation is not yet aligned enough.

Claims to avoid:

- do not present raw row-level CV as the primary benchmark
- do not claim role modeling is ready
- do not claim LoRA is clearly superior overall
- do not oversell the inferred article-block robustness check as true article-level grouping

## 15. Key Output Files

Core benchmarking:

- `outputs/finetune_cv_results.txt`
- `outputs/setfit_cv_results.txt`
- `outputs/additional_experiments_results.txt`
- `outputs/lora_imbalance_results.txt`
- `outputs/grouped_calibrated_results.txt`
- `outputs/core_uncertainty_results.txt`

Trigger modeling:

- `outputs/token_classification_results.txt`
- `outputs/trigger_threshold_sweep_results.txt`
- `outputs/trigger_structured_decoder_results.txt`
- `outputs/trigger_crf_results.txt`

Peripheral modeling:

- `outputs/peripheral_chain_results.txt`

Dataset and analysis:

- `outputs/dataset_audit_results.txt`
- `outputs/zero_shot_results.txt`
- `outputs/knn_probe_results.txt`
- `outputs/round4_attention_results.txt`

Error analysis:

- `outputs/additional_core_error_analysis.txt`

## 16. Reproducibility

### 16.1 Tested stack

The current rerun used:

| Package | Version |
|---|---:|
| Python | 3.11.9 |
| pandas | 2.3.3 |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| torch | 2.6.0+cu124 |
| transformers | 4.57.6 |
| peft | 0.18.1 |
| datasets | 4.7.0 |
| sentence-transformers | 5.2.3 |
| setfit | 1.1.3 |
| pytorch-crf | 0.7.2 |
| openpyxl | 3.1.5 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| umap-learn | 0.5.11 |

Compatibility note:

- SetFit required the repo to remain on a `transformers 4.x` stack.
- This is compatible with the current benchmark scripts, but it may conflict with unrelated tools that expect `transformers 5.x`.

### 16.2 Suggested rerun order

Run the commands below from the repository root.

```bash
python -m experiments.analysis.poc_dataset_audit
python -m experiments.analysis.poc_base_model_analysis
python -m experiments.analysis.poc_round2_analysis
python -m experiments.analysis.poc_round3_analysis
python -m experiments.analysis.poc_round4_attention
python -m experiments.benchmarks.poc_finetune_setfit
python -m experiments.benchmarks.poc_setfit_baseline
python -m experiments.benchmarks.poc_token_classification
python -m experiments.benchmarks.poc_additional_experiments
python -m experiments.benchmarks.poc_grouped_calibrated_experiments
python -m experiments.benchmarks.poc_core_uncertainty_analysis
python -m experiments.benchmarks.poc_lora_imbalance_experiments
python -m experiments.benchmarks.poc_peripheral_chain_experiments
python -m experiments.benchmarks.poc_trigger_threshold_sweep
python -m experiments.benchmarks.poc_trigger_structured_decoder
python -m experiments.benchmarks.poc_trigger_crf
```

All scripts now default to the expanded workbook through `climate_frames/dataset.py`.

If you need to point the code at a different workbook without editing source, set `CLFRAMES_DATA_PATH` before running a module.

## 17. Method References

- [ClimateBERT: A Pretrained Language Model for Climate-Related Text](https://arxiv.org/abs/2110.12010)
- [Efficient Few-Shot Learning Without Prompts (SetFit)](https://arxiv.org/abs/2209.11055)
- [SetFit documentation](https://huggingface.co/docs/setfit/index)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Adjusting Logits for Long-Tail Recognition](https://arxiv.org/abs/2007.07314)
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/handle/20.500.14332/6188)
- [Transformers token classification guide](https://huggingface.co/docs/transformers/main/tasks/token_classification)
- [StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

