"""
=============================================================================
Hinglish Code-Switching Language ID — Complete Training & Evaluation Pipeline
=============================================================================
Run in Google Colab with a T4 GPU.

SETUP (run this first in Colab):
    !pip install -q "transformers>=4.41" datasets accelerate scikit-learn seaborn matplotlib

Upload these files to Colab:
    - train.py
    - custom_test_hinglish.conll

Then run this script:
    !python hinglish_lid_colab.py

Or copy cells into a notebook.
"""

# ========================== CELL 1: IMPORTS ==========================
import os, json, sys, time, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_recall_fscore_support)
from collections import Counter

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

LABELS = ["EN", "HI", "NE", "OTHER"]
LOSS_TYPES = ["ce", "weighted", "focal"]
LOSS_NAMES = {"ce": "Cross-Entropy", "weighted": "Weighted CE", "focal": "Focal Loss (γ=2)"}
EVAL_SETS = ["lince_val", "lince_heldout_test", "custom_conversational"]
EVAL_NAMES = {"lince_val": "LinCE Val", "lince_heldout_test": "LinCE Held-out Test",
              "custom_conversational": "Custom Conversational"}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================== CELL 2: TRAINING ==========================
def run_training():
    """Run all three training experiments."""
    print("=" * 70)
    print("PHASE 1: TRAINING (3 experiments)")
    print("=" * 70)

    for loss in LOSS_TYPES:
        print(f"\n{'='*60}")
        print(f"  Training with {LOSS_NAMES[loss]} loss")
        print(f"{'='*60}")
        out_dir = f"runs/{loss}"
        cmd = (f"{sys.executable} train.py "
               f"--loss {loss} --seed 42 --out {out_dir} --epochs 4 "
               f"--custom custom_test_hinglish.conll")
        ret = os.system(cmd)
        if ret != 0:
            print(f"[WARNING] Training with {loss} exited with code {ret}")
        else:
            print(f"[OK] {LOSS_NAMES[loss]} training complete → {out_dir}/")

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)


# ========================== CELL 3: LOAD RESULTS ==========================
def load_all_reports():
    """Load JSON reports from all runs."""
    reports = {}
    for loss in LOSS_TYPES:
        reports[loss] = {}
        for eval_set in EVAL_SETS:
            path = f"runs/{loss}/report_{eval_set}.json"
            if os.path.exists(path):
                with open(path) as f:
                    reports[loss][eval_set] = json.load(f)
                print(f"  Loaded {path}")
            else:
                print(f"  [SKIP] {path} not found")
    return reports


# ========================== CELL 4: ABLATION TABLE ==========================
def make_ablation_table(reports):
    """Print and save the main ablation comparison table."""
    print("\n" + "=" * 70)
    print("TABLE: ABLATION COMPARISON ACROSS LOSS FUNCTIONS")
    print("=" * 70)

    header = f"{'Loss Function':<18} {'Eval Set':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for loss in LOSS_TYPES:
        for es in EVAL_SETS:
            if es in reports.get(loss, {}):
                r = reports[loss][es]
                acc = r.get("accuracy", 0)
                f1m = r.get("macro avg", {}).get("f1-score", 0)
                f1w = r.get("weighted avg", {}).get("f1-score", 0)
                row = {
                    "loss": LOSS_NAMES[loss], "eval_set": EVAL_NAMES[es],
                    "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w
                }
                rows.append(row)
                print(f"{LOSS_NAMES[loss]:<18} {EVAL_NAMES[es]:<25} {acc:>10.4f} {f1m:>10.4f} {f1w:>12.4f}")

    # Save as JSON for report generation
    with open(f"{OUTPUT_DIR}/ablation_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    return rows


# ========================== CELL 5: PER-CLASS F1 BAR CHART ==========================
def plot_per_class_f1(reports):
    """Bar chart comparing per-class F1 across loss functions for each eval set."""
    for es in EVAL_SETS:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(LABELS))
        width = 0.25

        has_data = False
        for i, loss in enumerate(LOSS_TYPES):
            if es in reports.get(loss, {}):
                r = reports[loss][es]
                f1s = [r.get(label, {}).get("f1-score", 0) for label in LABELS]
                bars = ax.bar(x + i * width, f1s, width, label=LOSS_NAMES[loss],
                              edgecolor='white', linewidth=0.5)
                # Add value labels on bars
                for bar, val in zip(bars, f1s):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                has_data = True

        if has_data:
            ax.set_xlabel('Language Tag')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'Per-Class F1 Score — {EVAL_NAMES[es]}')
            ax.set_xticks(x + width)
            ax.set_xticklabels(LABELS)
            ax.set_ylim(0, 1.15)
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)

            fname = f"{OUTPUT_DIR}/f1_per_class_{es}.png"
            plt.savefig(fname)
            print(f"  Saved {fname}")
        plt.close()


# ========================== CELL 6: CONFUSION MATRICES ==========================
def plot_confusion_matrices(reports):
    """Generate confusion matrices from the classification reports."""
    # We'll reconstruct approximate confusion matrices from support + F1
    # For proper confusion matrices, we need to re-run predictions.
    # This section generates them from the saved model.
    print("\n  Generating confusion matrices requires re-running inference...")
    print("  (Will be done in the Colab notebook with saved models)")

    # For now, create placeholder structure
    for loss in LOSS_TYPES:
        for es in EVAL_SETS:
            if es not in reports.get(loss, {}):
                continue
            r = reports[loss][es]
            # Extract available stats
            support = [r.get(label, {}).get("support", 0) for label in LABELS]
            precision = [r.get(label, {}).get("precision", 0) for label in LABELS]
            recall = [r.get(label, {}).get("recall", 0) for label in LABELS]
            f1 = [r.get(label, {}).get("f1-score", 0) for label in LABELS]

            # Create a summary stats heatmap instead
            fig, ax = plt.subplots(figsize=(8, 4))
            data = np.array([precision, recall, f1])
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(range(len(LABELS)))
            ax.set_xticklabels(LABELS)
            ax.set_yticks(range(3))
            ax.set_yticklabels(['Precision', 'Recall', 'F1'])

            for i in range(3):
                for j in range(len(LABELS)):
                    ax.text(j, i, f'{data[i, j]:.3f}',
                            ha='center', va='center', fontsize=11,
                            color='white' if data[i, j] > 0.5 else 'black')

            ax.set_title(f'{LOSS_NAMES[loss]} — {EVAL_NAMES[es]}')
            plt.colorbar(im, ax=ax, label='Score')

            fname = f"{OUTPUT_DIR}/metrics_heatmap_{loss}_{es}.png"
            plt.savefig(fname)
            plt.close()
            print(f"  Saved {fname}")


# ========================== CELL 7: GENERATE PROPER CONFUSION MATRICES ==========================
def generate_confusion_matrices_from_model():
    """Re-load saved models and generate real confusion matrices."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    for loss in LOSS_TYPES:
        model_dir = f"runs/{loss}/best"
        if not os.path.exists(model_dir):
            print(f"  [SKIP] {model_dir} not found")
            continue

        print(f"\n  Loading model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()

        # Load and predict on custom test set
        if os.path.exists("custom_test_hinglish.conll"):
            L2I = {l: i for i, l in enumerate(LABELS)}
            sents, all_true, all_pred = [], [], []

            toks, labs = [], []
            for line in open("custom_test_hinglish.conll", encoding="utf-8"):
                line = line.rstrip("\n")
                if not line.strip():
                    if toks:
                        sents.append((toks, labs))
                        toks, labs = [], []
                    continue
                w, t = line.split("\t")
                toks.append(w)
                labs.append(t.strip())
            if toks:
                sents.append((toks, labs))

            for words, true_tags in sents:
                enc = tokenizer(words, is_split_into_words=True,
                                truncation=True, max_length=128,
                                return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits[0]

                prev_wid = None
                word_idx = 0
                for j, wid in enumerate(enc.word_ids(0)):
                    if wid is None or wid == prev_wid:
                        prev_wid = wid
                        continue
                    pred_label = LABELS[logits[j].argmax().item()]
                    if word_idx < len(true_tags):
                        all_true.append(true_tags[word_idx])
                        all_pred.append(pred_label)
                    word_idx += 1
                    prev_wid = wid

            # Build confusion matrix
            cm = confusion_matrix([L2I[t] for t in all_true],
                                  [L2I[t] for t in all_pred],
                                  labels=list(range(len(LABELS))))

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                        linewidths=0.5, linecolor='gray')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix — {LOSS_NAMES[loss]} (Custom Test Set)')

            fname = f"{OUTPUT_DIR}/confusion_matrix_{loss}_custom.png"
            plt.savefig(fname)
            plt.close()
            print(f"  Saved {fname}")

            # Also do error analysis
            errors = []
            idx = 0
            for words, true_tags in sents:
                enc = tokenizer(words, is_split_into_words=True,
                                truncation=True, max_length=128,
                                return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits[0]
                    probs = torch.softmax(logits, dim=-1)

                prev_wid = None
                word_idx = 0
                for j, wid in enumerate(enc.word_ids(0)):
                    if wid is None or wid == prev_wid:
                        prev_wid = wid
                        continue
                    pred_id = logits[j].argmax().item()
                    pred_label = LABELS[pred_id]
                    conf = probs[j][pred_id].item()
                    if word_idx < len(true_tags) and pred_label != true_tags[word_idx]:
                        context = " ".join(words)
                        errors.append({
                            "word": words[word_idx],
                            "true": true_tags[word_idx],
                            "pred": pred_label,
                            "confidence": round(conf, 3),
                            "context": context
                        })
                    word_idx += 1
                    prev_wid = wid

            # Save error analysis
            with open(f"{OUTPUT_DIR}/error_analysis_{loss}.json", "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            print(f"  Saved {OUTPUT_DIR}/error_analysis_{loss}.json ({len(errors)} errors)")


# ========================== CELL 8: CMI ANALYSIS ==========================
def compute_cmi_from_model():
    """Compute Code-Mixing Index for each sentence using the best model."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    # Use the best model (we'll pick the one with highest F1 later)
    best_loss = None
    best_f1 = 0
    for loss in LOSS_TYPES:
        report_path = f"runs/{loss}/report_lince_val.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                r = json.load(f)
                f1 = r.get("macro avg", {}).get("f1-score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_loss = loss

    if best_loss is None:
        print("  [SKIP] No trained models found")
        return

    model_dir = f"runs/{best_loss}/best"
    print(f"\n  Using best model ({LOSS_NAMES[best_loss]}, F1={best_f1:.4f}) from {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # Load custom test sentences
    sents = []
    toks = []
    for line in open("custom_test_hinglish.conll", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks:
                sents.append(toks)
                toks = []
            continue
        w, _ = line.split("\t")
        toks.append(w)
    if toks:
        sents.append(toks)

    cmis = []
    for words in sents:
        enc = tokenizer(words, is_split_into_words=True,
                        truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]

        pred_labels = []
        prev_wid = None
        for j, wid in enumerate(enc.word_ids(0)):
            if wid is None or wid == prev_wid:
                prev_wid = wid
                continue
            pred_labels.append(LABELS[logits[j].argmax().item()])
            prev_wid = wid

        # Compute CMI
        counts = Counter(pred_labels)
        n_total = len(pred_labels)
        n_other = counts.get("OTHER", 0) + counts.get("NE", 0)
        lang_counts = {k: v for k, v in counts.items() if k in ("EN", "HI")}

        if n_total - n_other > 0 and len(lang_counts) > 0:
            max_lang = max(lang_counts.values()) if lang_counts else 0
            cmi = (n_total - max_lang - n_other) / (n_total - n_other) * 100
        else:
            cmi = 0.0
        cmis.append(cmi)

    # Plot CMI distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(cmis, bins=20, color='#4ECDC4', edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.mean(cmis), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(cmis):.1f}%')
    axes[0].axvline(np.median(cmis), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(cmis):.1f}%')
    axes[0].set_xlabel('Code-Mixing Index (%)')
    axes[0].set_ylabel('Number of Sentences')
    axes[0].set_title('CMI Distribution — Custom Test Set')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(cmis, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#45B7D1', alpha=0.7))
    axes[1].set_ylabel('Code-Mixing Index (%)')
    axes[1].set_title('CMI Box Plot')

    plt.suptitle(f'Code-Mixing Analysis ({len(sents)} sentences)', fontsize=14, y=1.02)
    plt.tight_layout()

    fname = f"{OUTPUT_DIR}/cmi_distribution.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")

    # Print stats
    print(f"\n  CMI Statistics:")
    print(f"    Mean:   {np.mean(cmis):.1f}%")
    print(f"    Median: {np.median(cmis):.1f}%")
    print(f"    Std:    {np.std(cmis):.1f}%")
    print(f"    Min:    {np.min(cmis):.1f}%")
    print(f"    Max:    {np.max(cmis):.1f}%")

    # Save
    with open(f"{OUTPUT_DIR}/cmi_scores.json", "w") as f:
        json.dump({"scores": cmis, "mean": np.mean(cmis),
                   "median": float(np.median(cmis)), "std": np.std(cmis)}, f, indent=2)


# ========================== CELL 9: F1 COMPARISON ACROSS MODELS ==========================
def plot_model_comparison(reports):
    """Side-by-side F1 macro comparison across all loss functions and eval sets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(EVAL_SETS))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, loss in enumerate(LOSS_TYPES):
        f1s = []
        for es in EVAL_SETS:
            if es in reports.get(loss, {}):
                f1s.append(reports[loss][es].get("macro avg", {}).get("f1-score", 0))
            else:
                f1s.append(0)
        bars = ax.bar(x + i * width, f1s, width, label=LOSS_NAMES[loss],
                      color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Evaluation Set')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Loss Function Ablation — Macro F1 Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([EVAL_NAMES[es] for es in EVAL_SETS])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fname = f"{OUTPUT_DIR}/f1_macro_comparison.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")


# ========================== CELL 10: DATASET STATISTICS ==========================
def dataset_statistics():
    """Compute and visualize dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    # Custom test set stats
    if os.path.exists("custom_test_hinglish.conll"):
        sents, total_toks = 0, 0
        tag_counts = Counter()
        sent_lens = []
        toks_in_sent = 0

        for line in open("custom_test_hinglish.conll", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                if toks_in_sent > 0:
                    sents += 1
                    sent_lens.append(toks_in_sent)
                    toks_in_sent = 0
                continue
            _, t = line.split("\t")
            tag_counts[t.strip()] += 1
            total_toks += 1
            toks_in_sent += 1
        if toks_in_sent > 0:
            sents += 1
            sent_lens.append(toks_in_sent)

        print(f"\n  Custom Test Set:")
        print(f"    Sentences: {sents}")
        print(f"    Total tokens: {total_toks}")
        print(f"    Avg sentence length: {np.mean(sent_lens):.1f} tokens")
        print(f"    Tag distribution:")
        for tag in LABELS:
            count = tag_counts.get(tag, 0)
            pct = count / total_toks * 100
            print(f"      {tag}: {count} ({pct:.1f}%)")

        # Plot tag distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        sizes = [tag_counts.get(t, 0) for t in LABELS]
        colors_pie = ['#3498db', '#e67e22', '#2ecc71', '#95a5a6']
        axes[0].pie(sizes, labels=LABELS, autopct='%1.1f%%', colors=colors_pie,
                    startangle=90, textprops={'fontsize': 12})
        axes[0].set_title('Custom Test Set — Tag Distribution')

        # Sentence length histogram
        axes[1].hist(sent_lens, bins=range(1, max(sent_lens)+2), color='#3498db',
                     edgecolor='white', linewidth=0.5, align='left')
        axes[1].set_xlabel('Sentence Length (tokens)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Custom Test Set — Sentence Length Distribution')

        plt.tight_layout()
        fname = f"{OUTPUT_DIR}/dataset_statistics.png"
        plt.savefig(fname)
        plt.close()
        print(f"  Saved {fname}")


# ========================== CELL 11: EXAMPLE PREDICTIONS VISUALIZATION ==========================
def visualize_example_predictions():
    """Show color-coded predictions on a few example sentences."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    best_loss = None
    best_f1 = 0
    for loss in LOSS_TYPES:
        report_path = f"runs/{loss}/report_lince_val.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                r = json.load(f)
                f1 = r.get("macro avg", {}).get("f1-score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_loss = loss

    if best_loss is None:
        print("  [SKIP] No trained models found")
        return

    model_dir = f"runs/{best_loss}/best"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # Example sentences to visualize
    examples = [
        "kal meeting ka time kya hai ?",
        "mujhe flight book karni hai Mumbai se Goa tak",
        "bhai ye app bahut slow chal raha hai",
        "please mera password reset kar do",
        "can you mujhe ek doosra option dikha sakte ho ?",
        "Flipkart pe sale kab start hogi ?",
        "good morning , aaj ka schedule kya hai ?",
        "server down hai isliye API response nahi de raha",
    ]

    tag_colors = {"EN": "#3498db", "HI": "#e67e22", "NE": "#2ecc71", "OTHER": "#95a5a6"}

    fig, axes = plt.subplots(len(examples), 1, figsize=(16, len(examples) * 1.2))
    if len(examples) == 1:
        axes = [axes]

    for idx, sent in enumerate(examples):
        words = sent.split()
        enc = tokenizer(words, is_split_into_words=True,
                        truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1)

        pred_labels = []
        confidences = []
        prev_wid = None
        for j, wid in enumerate(enc.word_ids(0)):
            if wid is None or wid == prev_wid:
                prev_wid = wid
                continue
            pred_id = logits[j].argmax().item()
            pred_labels.append(LABELS[pred_id])
            confidences.append(probs[j][pred_id].item())
            prev_wid = wid

        # Draw color-coded sentence
        ax = axes[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        x_pos = 0.02
        for w, tag, conf in zip(words, pred_labels, confidences):
            color = tag_colors.get(tag, '#95a5a6')
            ax.text(x_pos, 0.5, f" {w} ", fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3,
                              edgecolor=color),
                    va='center', ha='left')
            x_pos += len(w) * 0.018 + 0.03

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.3, edgecolor=c,
                                     label=f'{k}')
                       for k, c in tag_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=11,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f'Example Predictions — {LOSS_NAMES[best_loss]} Model', fontsize=14, y=1.02)
    plt.tight_layout()

    fname = f"{OUTPUT_DIR}/example_predictions.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")


# ========================== CELL 12: GENERATE ALL OUTPUTS ==========================
def generate_summary_report(reports):
    """Generate a markdown summary of all results."""
    lines = []
    lines.append("# Hinglish LID — Experiment Results Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Ablation Results\n")
    lines.append("| Loss Function | Eval Set | Accuracy | F1 (Macro) | F1 (Weighted) |")
    lines.append("|---|---|---|---|---|")

    for loss in LOSS_TYPES:
        for es in EVAL_SETS:
            if es in reports.get(loss, {}):
                r = reports[loss][es]
                acc = r.get("accuracy", 0)
                f1m = r.get("macro avg", {}).get("f1-score", 0)
                f1w = r.get("weighted avg", {}).get("f1-score", 0)
                lines.append(f"| {LOSS_NAMES[loss]} | {EVAL_NAMES[es]} | "
                             f"{acc:.4f} | {f1m:.4f} | {f1w:.4f} |")

    lines.append("\n## Per-Class F1 Scores\n")
    for loss in LOSS_TYPES:
        if "lince_val" in reports.get(loss, {}):
            r = reports[loss]["lince_val"]
            lines.append(f"### {LOSS_NAMES[loss]}\n")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|---|---|---|---|---|")
            for label in LABELS:
                if label in r:
                    p = r[label].get("precision", 0)
                    rc = r[label].get("recall", 0)
                    f1 = r[label].get("f1-score", 0)
                    s = r[label].get("support", 0)
                    lines.append(f"| {label} | {p:.4f} | {rc:.4f} | {f1:.4f} | {s} |")
            lines.append("")

    lines.append("\n## Generated Outputs\n")
    for f in sorted(glob.glob(f"{OUTPUT_DIR}/*")):
        lines.append(f"- `{f}`")

    report_text = "\n".join(lines)
    with open(f"{OUTPUT_DIR}/results_summary.md", "w") as f:
        f.write(report_text)
    print(f"\n  Saved {OUTPUT_DIR}/results_summary.md")


# ========================== MAIN ==========================
def main():
    print("=" * 70)
    print("  HINGLISH CODE-SWITCHING LANGUAGE ID — FULL PIPELINE")
    print("=" * 70)

    # Phase 1: Training
    run_training()

    # Phase 2: Load results
    print("\n" + "=" * 70)
    print("PHASE 2: LOADING RESULTS")
    print("=" * 70)
    reports = load_all_reports()

    # Phase 3: Visualizations
    print("\n" + "=" * 70)
    print("PHASE 3: GENERATING VISUALIZATIONS")
    print("=" * 70)

    print("\n--- Dataset Statistics ---")
    dataset_statistics()

    print("\n--- Ablation Table ---")
    make_ablation_table(reports)

    print("\n--- Per-Class F1 Charts ---")
    plot_per_class_f1(reports)

    print("\n--- Model Comparison ---")
    plot_model_comparison(reports)

    print("\n--- Metrics Heatmaps ---")
    plot_confusion_matrices(reports)

    print("\n--- Confusion Matrices (from model) ---")
    try:
        generate_confusion_matrices_from_model()
    except Exception as e:
        print(f"  [ERROR] {e}")

    print("\n--- CMI Analysis ---")
    try:
        compute_cmi_from_model()
    except Exception as e:
        print(f"  [ERROR] {e}")

    print("\n--- Example Predictions ---")
    try:
        visualize_example_predictions()
    except Exception as e:
        print(f"  [ERROR] {e}")

    print("\n--- Summary Report ---")
    generate_summary_report(reports)

    print("\n" + "=" * 70)
    print("  ALL DONE! Check the 'outputs/' directory for all figures and data.")
    print("=" * 70)
    print(f"\nGenerated files:")
    for f in sorted(glob.glob(f"{OUTPUT_DIR}/*")):
        print(f"  📄 {f}")


if __name__ == "__main__":
    main()
