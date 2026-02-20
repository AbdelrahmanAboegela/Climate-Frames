"""
ClimateBERT Base Model POC — Round 4: Deep Attention Map Analysis
=================================================================
5 tests analyzing the 72 attention heads (6 layers × 12 heads):

1. Head-level Frame Attention Scores → 6×12 heatmap
2. Attention Rollout (CLS → tokens) → aggregated visualization
3. Per-Paragraph Attention Heatmaps → annotated token-level maps
4. Frame Token Cross-Attention → 2×2 group matrix
5. Attention Entropy Analysis → scatter (entropy vs frame focus)

Run: python poc_round4_attention.py
Output: e:/Frames/poc_outputs/
"""

import os, re, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

warnings.filterwarnings("ignore")

OUT = r"e:\Frames\poc_outputs"
DATA = r"e:\Frames\3 Articles Samples Annotation 2026.xlsx"
MODEL_NAME = "climatebert/distilroberta-base-climate-f"

os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  Data Loading (reused from prior rounds)
# ══════════════════════════════════════════════════════════

def load_data():
    df = pd.read_excel(DATA, engine="openpyxl", sheet_name="Sheet1", usecols=[0, 1, 2, 3])
    df.columns = ["Text", "CoreFrame", "PeripheralFrame", "Tokens"]
    df = df.dropna(subset=["Text"]).reset_index(drop=True)
    records = []
    for _, r in df.iterrows():
        tokens_raw = str(r["Tokens"]) if pd.notna(r["Tokens"]) else ""
        tokens = [t.strip().lower() for t in tokens_raw.split(",") if t.strip()]
        core = str(r["CoreFrame"]).strip() if pd.notna(r["CoreFrame"]) else ""
        core = core.replace("_", " ").strip()
        records.append({
            "text": str(r["Text"]).strip(),
            "core_frame": core,
            "tokens": tokens,
        })
    return records


def find_frame_token_positions(text, frame_tokens, tokenizer, encoding):
    """Map frame-evoking tokens to subword positions in the encoding."""
    input_ids = encoding["input_ids"][0]
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text_lower = text.lower()
    frame_positions = set()

    for ft in frame_tokens:
        ft_lower = ft.lower().strip()
        # Find character position in original text
        start = text_lower.find(ft_lower)
        if start == -1:
            continue
        end = start + len(ft_lower)

        # Map character positions to token positions
        for idx in range(1, len(all_tokens) - 1):  # skip CLS/SEP
            span = encoding.token_to_chars(0, idx)
            if span is None:
                continue
            tok_start, tok_end = span.start, span.end
            # Check overlap
            if tok_start < end and tok_end > start:
                frame_positions.add(idx)

    return frame_positions, all_tokens


# ══════════════════════════════════════════════════════════
#  Test 1: Head-level Frame Attention Scores
# ══════════════════════════════════════════════════════════

def test_head_frame_attention(records, model, tokenizer):
    """For each of 72 heads, compute ratio of attention to frame vs non-frame tokens."""
    print("\n[1/5] Head-level Frame Attention Scores...")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # Accumulate ratios across paragraphs
    head_ratios = np.zeros((n_layers, n_heads))
    head_counts = np.zeros((n_layers, n_heads))

    for i, rec in enumerate(records):
        if not rec["tokens"]:
            continue

        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True,
                        max_length=512, return_offsets_mapping=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_attentions=True)

        frame_pos, all_toks = find_frame_token_positions(
            rec["text"], rec["tokens"], tokenizer, 
            tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=512)
        )

        if len(frame_pos) == 0:
            continue

        seq_len = enc["input_ids"].shape[1]
        non_frame_pos = set(range(1, seq_len - 1)) - frame_pos

        if len(non_frame_pos) == 0:
            continue

        for layer_idx, attn in enumerate(out.attentions):
            # attn shape: [1, n_heads, seq_len, seq_len]
            attn_np = attn[0].cpu().numpy()  # [n_heads, seq_len, seq_len]

            for head_idx in range(n_heads):
                # Attention FROM [CLS] (row 0) TO each token
                cls_attn = attn_np[head_idx, 0, :]  # [seq_len]
                frame_attn = np.mean([cls_attn[p] for p in frame_pos])
                non_frame_attn = np.mean([cls_attn[p] for p in non_frame_pos])

                if non_frame_attn > 0:
                    head_ratios[layer_idx, head_idx] += frame_attn / non_frame_attn
                    head_counts[layer_idx, head_idx] += 1

    # Average
    mask = head_counts > 0
    head_ratios[mask] /= head_counts[mask]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(head_ratios, cmap="RdYlGn", aspect="auto",
                   vmin=0.5, vmax=2.0)
    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i+1}" for i in range(n_layers)])
    ax.set_title("Frame Attention Ratio by Head (CLS → tokens)\n"
                 "Green = head attends MORE to frame tokens | Red = LESS",
                 fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Frame / Non-frame Attention Ratio")

    # Annotate values
    for ly in range(n_layers):
        for hd in range(n_heads):
            val = head_ratios[ly, hd]
            color = "white" if val < 0.8 or val > 1.6 else "black"
            ax.text(hd, ly, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "head_frame_attention.png"), dpi=150)
    plt.close()

    # Find top/bottom heads
    flat = [(head_ratios[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)
    top5 = flat[:5]
    bot5 = flat[-5:]

    results = []
    results.append("Head-level Frame Attention Scores")
    results.append("=" * 50)
    results.append(f"Model: {n_layers} layers × {n_heads} heads = {n_layers * n_heads} heads")
    results.append(f"Paragraphs analyzed: {int(head_counts.max())}")
    results.append(f"\nGlobal mean ratio: {head_ratios.mean():.4f}")
    results.append(f"Std dev: {head_ratios.std():.4f}")
    results.append(f"\nTop 5 frame-attending heads (highest ratio):")
    for val, l, h in top5:
        results.append(f"  Layer {l+1}, Head {h}: {val:.3f}x")
    results.append(f"\nBottom 5 heads (least frame attention):")
    for val, l, h in bot5:
        results.append(f"  Layer {l+1}, Head {h}: {val:.3f}x")

    # Per-layer average
    results.append(f"\nPer-layer average ratio:")
    for l in range(n_layers):
        results.append(f"  Layer {l+1}: {head_ratios[l].mean():.3f}")

    print("  → head_frame_attention.png")
    return head_ratios, "\n".join(results)


# ══════════════════════════════════════════════════════════
#  Test 2: Attention Rollout
# ══════════════════════════════════════════════════════════

def compute_attention_rollout(attentions):
    """
    Compute attention rollout following Abnar & Zuidema (2020).
    Aggregates attention through layers accounting for residual connections.
    """
    # Average across heads for each layer
    result = None
    for attn in attentions:
        # attn: [1, n_heads, seq, seq]
        attn_heads_avg = attn[0].mean(dim=0).cpu().numpy()  # [seq, seq]
        # Add residual connection (identity matrix)
        attn_heads_avg = 0.5 * attn_heads_avg + 0.5 * np.eye(attn_heads_avg.shape[0])
        # Normalize rows
        attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(axis=-1, keepdims=True)

        if result is None:
            result = attn_heads_avg
        else:
            result = result @ attn_heads_avg

    return result


def test_attention_rollout(records, model, tokenizer):
    """Compute attention rollout from CLS and compare frame vs non-frame."""
    print("\n[2/5] Attention Rollout Analysis...")

    rollout_ratios = []
    rollout_frame_means = []
    rollout_nonframe_means = []
    para_labels = []

    for i, rec in enumerate(records):
        if not rec["tokens"]:
            continue

        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=512)
        enc_dev = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc_dev, output_attentions=True)

        frame_pos, all_toks = find_frame_token_positions(
            rec["text"], rec["tokens"], tokenizer, enc
        )

        if len(frame_pos) == 0:
            continue

        seq_len = enc["input_ids"].shape[1]
        non_frame_pos = set(range(1, seq_len - 1)) - frame_pos

        if len(non_frame_pos) == 0:
            continue

        rollout = compute_attention_rollout(out.attentions)
        cls_rollout = rollout[0]  # CLS row

        frame_attn = np.mean([cls_rollout[p] for p in frame_pos])
        non_attn = np.mean([cls_rollout[p] for p in non_frame_pos])

        if non_attn > 0:
            ratio = frame_attn / non_attn
            rollout_ratios.append(ratio)
            rollout_frame_means.append(frame_attn)
            rollout_nonframe_means.append(non_attn)
            para_labels.append(f"P{i}: {rec['core_frame'][:15]}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: ratio per paragraph
    colors = ["#2ca02c" if r > 1.0 else "#d62728" for r in rollout_ratios]
    axes[0].barh(range(len(rollout_ratios)), rollout_ratios, color=colors, alpha=0.8)
    axes[0].axvline(x=1.0, color="black", linestyle="--", alpha=0.5, label="Equal attention")
    axes[0].set_yticks(range(len(para_labels)))
    axes[0].set_yticklabels(para_labels, fontsize=7)
    axes[0].set_xlabel("Frame / Non-frame Rollout Ratio")
    axes[0].set_title("Attention Rollout: Frame Token Focus\n"
                      "Green = more attention to frame tokens", fontweight="bold")
    axes[0].invert_yaxis()

    # Right: grouped bar
    x = np.arange(len(para_labels))
    w = 0.35
    axes[1].barh(x - w/2, rollout_frame_means, w, label="Frame tokens", color="#2ca02c", alpha=0.7)
    axes[1].barh(x + w/2, rollout_nonframe_means, w, label="Non-frame tokens", color="#d62728", alpha=0.7)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(para_labels, fontsize=7)
    axes[1].set_xlabel("Mean Rollout Attention from CLS")
    axes[1].set_title("Rollout Attention: Frame vs Non-frame\n"
                      "(absolute values)", fontweight="bold")
    axes[1].legend()
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "attention_rollout.png"), dpi=150)
    plt.close()

    results = []
    results.append("\nAttention Rollout Analysis")
    results.append("=" * 50)
    results.append(f"Mean rollout ratio (frame/non-frame): {np.mean(rollout_ratios):.4f}")
    results.append(f"Median: {np.median(rollout_ratios):.4f}")
    results.append(f"Paragraphs where frame tokens get MORE attention: "
                   f"{sum(1 for r in rollout_ratios if r > 1.0)}/{len(rollout_ratios)}")
    results.append(f"Range: {min(rollout_ratios):.3f} – {max(rollout_ratios):.3f}")

    print("  → attention_rollout.png")
    return "\n".join(results)


# ══════════════════════════════════════════════════════════
#  Test 3: Per-Paragraph Attention Heatmaps
# ══════════════════════════════════════════════════════════

def test_paragraph_heatmaps(records, model, tokenizer, para_indices=[0, 2, 15]):
    """Generate detailed attention heatmaps for select paragraphs."""
    print("\n[3/5] Per-Paragraph Attention Heatmaps...")

    results = ["\nPer-Paragraph Attention Heatmaps", "=" * 50]

    for pi in para_indices:
        if pi >= len(records):
            continue
        rec = records[pi]
        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=512)
        enc_dev = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc_dev, output_attentions=True)

        frame_pos, all_toks = find_frame_token_positions(
            rec["text"], rec["tokens"], tokenizer, enc
        )

        seq_len = enc["input_ids"].shape[1]

        # Compute rollout
        rollout = compute_attention_rollout(out.attentions)
        cls_rollout = rollout[0, 1:-1]  # skip CLS and SEP
        display_toks = all_toks[1:-1]  # skip CLS and SEP
        frame_pos_shifted = set(p - 1 for p in frame_pos)  # shift for display

        # Truncate for display if too long
        max_display = 60
        if len(display_toks) > max_display:
            display_toks = display_toks[:max_display]
            cls_rollout = cls_rollout[:max_display]
            frame_pos_shifted = set(p for p in frame_pos_shifted if p < max_display)

        # Clean token labels
        clean_toks = [t.replace("Ġ", "▸").replace("Ċ", "↵") for t in display_toks]

        # Color bars: green for frame tokens, gray for others
        bar_colors = ["#2ca02c" if idx in frame_pos_shifted else "#b0b0b0"
                      for idx in range(len(clean_toks))]

        fig, ax = plt.subplots(figsize=(16, max(5, len(clean_toks) * 0.22)))
        y_pos = np.arange(len(clean_toks))
        ax.barh(y_pos, cls_rollout, color=bar_colors, alpha=0.85, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_toks, fontsize=7, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("Attention Rollout from [CLS]", fontsize=11)
        ax.set_title(f"P{pi}: {rec['core_frame']}\n"
                     f"Green = frame-evoking tokens | Gray = other tokens",
                     fontsize=12, fontweight="bold")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2ca02c", label="Frame token"),
                           Patch(facecolor="#b0b0b0", label="Non-frame token")]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        fname = f"attention_heatmap_P{pi}.png"
        plt.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close()

        results.append(f"\nP{pi} ({rec['core_frame']}):")
        results.append(f"  Tokens: {seq_len}, Frame tokens: {len(frame_pos)}")
        top_indices = np.argsort(cls_rollout)[::-1][:5]
        results.append(f"  Top 5 attended tokens (rollout):")
        for ti in top_indices:
            marker = "★" if ti in frame_pos_shifted else " "
            results.append(f"    {marker} {clean_toks[ti]:20s} {cls_rollout[ti]:.4f}")

        print(f"  → {fname}")

    return "\n".join(results)


# ══════════════════════════════════════════════════════════
#  Test 4: Frame Token Cross-Attention
# ══════════════════════════════════════════════════════════

def test_cross_attention(records, model, tokenizer):
    """Analyze attention between frame and non-frame token groups."""
    print("\n[4/5] Frame Token Cross-Attention...")

    # Accumulate 2×2 attention matrix: [from_group, to_group]
    # Groups: frame (F), non-frame (N)
    cross_attn = np.zeros((2, 2))  # [F→F, F→N], [N→F, N→N]
    cross_count = 0

    for i, rec in enumerate(records):
        if not rec["tokens"]:
            continue

        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=512)
        enc_dev = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc_dev, output_attentions=True)

        frame_pos, _ = find_frame_token_positions(
            rec["text"], rec["tokens"], tokenizer, enc
        )

        if len(frame_pos) == 0:
            continue

        seq_len = enc["input_ids"].shape[1]
        non_frame_pos = set(range(1, seq_len - 1)) - frame_pos

        if len(non_frame_pos) == 0:
            continue

        # Use last layer (best for frame knowledge per Round 3)
        last_attn = out.attentions[-1][0].mean(dim=0).cpu().numpy()  # [seq, seq], averaged across heads

        frame_list = sorted(frame_pos)
        non_list = sorted(non_frame_pos)

        # F→F: average attention from frame tokens to frame tokens
        ff = np.mean([last_attn[f1, f2] for f1 in frame_list for f2 in frame_list if f1 != f2]) if len(frame_list) > 1 else 0
        # F→N
        fn = np.mean([last_attn[f, n] for f in frame_list for n in non_list])
        # N→F
        nf = np.mean([last_attn[n, f] for n in non_list for f in frame_list])
        # N→N
        nn = np.mean([last_attn[n1, n2] for n1 in non_list for n2 in non_list if n1 != n2]) if len(non_list) > 1 else 0

        cross_attn += np.array([[ff, fn], [nf, nn]])
        cross_count += 1

    cross_attn /= cross_count

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Frame\nTokens", "Non-Frame\nTokens"]
    im = ax.imshow(cross_attn, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Attention TO →", fontsize=13)
    ax.set_ylabel("Attention FROM ↓", fontsize=13)
    ax.set_title("Cross-Attention Between Token Groups\n(Last Layer, Head-averaged)",
                 fontsize=13, fontweight="bold")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cross_attn[r, c]:.4f}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white" if cross_attn[r, c] > 0.015 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Attention")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cross_attention_matrix.png"), dpi=150)
    plt.close()

    results = []
    results.append("\nFrame Token Cross-Attention (Last Layer)")
    results.append("=" * 50)
    results.append(f"Paragraphs analyzed: {cross_count}")
    results.append(f"\n  Frame→Frame:     {cross_attn[0,0]:.5f}")
    results.append(f"  Frame→Non-frame: {cross_attn[0,1]:.5f}")
    results.append(f"  Non-frame→Frame: {cross_attn[1,0]:.5f}")
    results.append(f"  Non-frame→Non:   {cross_attn[1,1]:.5f}")
    ff_vs_fn = cross_attn[0,0] / cross_attn[0,1] if cross_attn[0,1] > 0 else 0
    results.append(f"\n  Frame→Frame / Frame→Non ratio: {ff_vs_fn:.3f}x")
    results.append(f"  → {'Frame tokens attend to each other MORE' if ff_vs_fn > 1 else 'No strong frame-to-frame preference'}")

    print("  → cross_attention_matrix.png")
    return "\n".join(results)


# ══════════════════════════════════════════════════════════
#  Test 5: Attention Entropy Analysis
# ══════════════════════════════════════════════════════════

def test_entropy(records, model, tokenizer):
    """Compute attention entropy per head and correlate with frame focus."""
    print("\n[5/5] Attention Entropy Analysis...")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    head_entropies = np.zeros((n_layers, n_heads))
    head_frame_ratios = np.zeros((n_layers, n_heads))
    counts = np.zeros((n_layers, n_heads))

    for i, rec in enumerate(records):
        if not rec["tokens"]:
            continue

        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=512)
        enc_dev = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc_dev, output_attentions=True)

        frame_pos, _ = find_frame_token_positions(
            rec["text"], rec["tokens"], tokenizer, enc
        )

        if len(frame_pos) == 0:
            continue

        seq_len = enc["input_ids"].shape[1]
        non_frame_pos = set(range(1, seq_len - 1)) - frame_pos

        if len(non_frame_pos) == 0:
            continue

        for layer_idx, attn in enumerate(out.attentions):
            attn_np = attn[0].cpu().numpy()
            for head_idx in range(n_heads):
                # Entropy of CLS attention distribution
                cls_attn = attn_np[head_idx, 0, :]
                cls_attn = cls_attn + 1e-10  # avoid log(0)
                entropy = -np.sum(cls_attn * np.log2(cls_attn))

                # Frame ratio
                frame_attn = np.mean([cls_attn[p] for p in frame_pos])
                non_attn = np.mean([cls_attn[p] for p in non_frame_pos])
                ratio = frame_attn / non_attn if non_attn > 0 else 1.0

                head_entropies[layer_idx, head_idx] += entropy
                head_frame_ratios[layer_idx, head_idx] += ratio
                counts[layer_idx, head_idx] += 1

    mask = counts > 0
    head_entropies[mask] /= counts[mask]
    head_frame_ratios[mask] /= counts[mask]

    # Flatten for scatter
    ent_flat = head_entropies.flatten()
    ratio_flat = head_frame_ratios.flatten()
    layer_flat = np.repeat(np.arange(n_layers), n_heads)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: scatter (entropy vs frame ratio)
    cmap = plt.cm.viridis
    colors = [cmap(l / (n_layers - 1)) for l in layer_flat]
    sc = axes[0].scatter(ent_flat, ratio_flat, c=layer_flat, cmap="viridis",
                         s=80, alpha=0.8, edgecolors="black", linewidths=0.5)
    axes[0].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Equal attention")
    axes[0].set_xlabel("Attention Entropy (bits)", fontsize=12)
    axes[0].set_ylabel("Frame / Non-frame Ratio", fontsize=12)
    axes[0].set_title("Head Specialization: Entropy vs Frame Focus\n"
                      "Low entropy = focused | High ratio = frame-biased",
                      fontsize=12, fontweight="bold")
    cbar = plt.colorbar(sc, ax=axes[0])
    cbar.set_label("Layer")
    cbar.set_ticks(range(n_layers))
    cbar.set_ticklabels([f"L{i+1}" for i in range(n_layers)])
    axes[0].legend()

    # Annotate notable heads
    for idx in range(len(ent_flat)):
        if ratio_flat[idx] > 1.5 or ratio_flat[idx] < 0.6:
            l = layer_flat[idx]
            h = idx % n_heads
            axes[0].annotate(f"L{l+1}H{h}", (ent_flat[idx], ratio_flat[idx]),
                            fontsize=7, alpha=0.8)

    # Right: entropy heatmap
    im = axes[1].imshow(head_entropies, cmap="coolwarm", aspect="auto")
    axes[1].set_xlabel("Head", fontsize=12)
    axes[1].set_ylabel("Layer", fontsize=12)
    axes[1].set_xticks(range(n_heads))
    axes[1].set_xticklabels([f"H{i}" for i in range(n_heads)])
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels([f"L{i+1}" for i in range(n_layers)])
    axes[1].set_title("Attention Entropy per Head\n"
                      "Blue = focused | Red = diffuse", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], label="Entropy (bits)")

    for ly in range(n_layers):
        for hd in range(n_heads):
            axes[1].text(hd, ly, f"{head_entropies[ly, hd]:.1f}", ha="center",
                        va="center", fontsize=6, color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "attention_entropy.png"), dpi=150)
    plt.close()

    results = []
    results.append("\nAttention Entropy Analysis")
    results.append("=" * 50)
    results.append(f"Mean entropy across all heads: {ent_flat.mean():.2f} bits")
    results.append(f"Most focused head: L{layer_flat[np.argmin(ent_flat)]+1} H{np.argmin(ent_flat) % n_heads} "
                   f"({ent_flat.min():.2f} bits)")
    results.append(f"Most diffuse head: L{layer_flat[np.argmax(ent_flat)]+1} H{np.argmax(ent_flat) % n_heads} "
                   f"({ent_flat.max():.2f} bits)")

    # Correlation
    corr = np.corrcoef(ent_flat, ratio_flat)[0, 1]
    results.append(f"\nCorrelation (entropy vs frame ratio): {corr:.4f}")
    results.append(f"  → {'Focused heads tend to focus on frame tokens' if corr < -0.2 else 'Focused heads tend to focus on non-frame tokens' if corr > 0.2 else 'No strong correlation between focus and frame preference'}")

    print("  → attention_entropy.png")
    return "\n".join(results)


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Round 4: Deep Attention Map Analysis")
    print("=" * 60)

    records = load_data()
    print(f"  Loaded {len(records)} paragraphs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
    model.eval()
    print(f"  Model loaded: {MODEL_NAME}")

    all_results = []

    # Test 1
    head_ratios, r1 = test_head_frame_attention(records, model, tokenizer)
    all_results.append(r1)

    # Test 2
    r2 = test_attention_rollout(records, model, tokenizer)
    all_results.append(r2)

    # Test 3
    r3 = test_paragraph_heatmaps(records, model, tokenizer, para_indices=[0, 2, 15])
    all_results.append(r3)

    # Test 4
    r4 = test_cross_attention(records, model, tokenizer)
    all_results.append(r4)

    # Test 5
    r5 = test_entropy(records, model, tokenizer)
    all_results.append(r5)

    # Save all results
    with open(os.path.join(OUT, "round4_attention_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_results))
    print(f"\n  ✓ All results saved to {OUT}/round4_attention_results.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
