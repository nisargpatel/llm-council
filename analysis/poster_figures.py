"""
Poster figure generation and supplementary analysis.
Produces publication-ready figures and computes metrics not in analyze.py.

Requires: matplotlib, numpy, pandas
Optional: scipy (for ECE confidence intervals)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.trace_parser import (
    parse_adversarial_trace,
    extract_numeric_confidence,
    extract_leading_diagnosis,
)


# ──────────────────────────────────────────────
# COLOR PALETTE (muted, poster-ready)
# ──────────────────────────────────────────────

COLORS = {
    "baseline": "#5B7B9A",      # steel blue
    "adversarial": "#C4785B",   # warm amber/terra cotta
    "structured": "#6B9E78",    # sage green
}

CONDITION_LABELS = {
    "baseline": "Baseline",
    "adversarial": "Adversarial",
    "structured": "Structured",
}


# ──────────────────────────────────────────────
# FIGURE 1: Grouped bar chart — accuracy by difficulty × condition
# ──────────────────────────────────────────────

def figure_accuracy_by_difficulty(df: pd.DataFrame, output_path: str = "data/analysis/fig1_accuracy_by_difficulty.png"):
    """Grouped bar chart: Top-1 accuracy by difficulty level and condition."""

    difficulties = ["easy", "moderate", "hard"]
    conditions = [c for c in ["baseline", "adversarial", "structured"] if c in df["condition"].unique()]

    # Compute accuracy for each cell
    data = {}
    for cond in conditions:
        data[cond] = []
        for diff in difficulties:
            subset = df[(df["condition"] == cond) & (df["difficulty"] == diff)]
            acc = subset["top1_correct"].mean() if len(subset) > 0 else 0
            data[cond].append(acc)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(difficulties))
    width = 0.25
    offsets = np.linspace(-width, width, len(conditions))

    for i, cond in enumerate(conditions):
        bars = ax.bar(x + offsets[i], data[cond], width, 
                      label=CONDITION_LABELS[cond],
                      color=COLORS[cond], edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, data[cond]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Case Difficulty", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Diagnostic Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in difficulties], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved to {output_path}")


# ──────────────────────────────────────────────
# FIGURE 2: Confidence calibration plot
# ──────────────────────────────────────────────

def figure_calibration(df: pd.DataFrame, output_path: str = "data/analysis/fig2_calibration.png"):
    """Calibration plot: stated confidence vs actual accuracy by condition."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 100], [0, 1], "k--", alpha=0.3, linewidth=1, label="Perfect calibration")

    bins = [0, 30, 50, 70, 85, 100]
    bin_centers = [15, 40, 60, 77.5, 92.5]

    conditions = [c for c in ["baseline", "adversarial", "structured"] if c in df["condition"].unique()]

    ece_values = {}

    for cond in conditions:
        subset = df[(df["condition"] == cond) & df["numeric_confidence"].notna()].copy()
        if len(subset) < 5:
            continue

        bin_accs = []
        bin_confs = []
        bin_sizes = []

        for i in range(len(bins) - 1):
            in_bin = subset[(subset["numeric_confidence"] >= bins[i]) & 
                           (subset["numeric_confidence"] < bins[i+1])]
            if len(in_bin) > 0:
                bin_accs.append(in_bin["top1_correct"].mean())
                bin_confs.append(in_bin["numeric_confidence"].mean())
                bin_sizes.append(len(in_bin))
            else:
                bin_accs.append(None)
                bin_confs.append(None)
                bin_sizes.append(0)

        # Plot calibration curve
        valid = [(c, a) for c, a in zip(bin_confs, bin_accs) if c is not None and a is not None]
        if valid:
            confs, accs = zip(*valid)
            ax.plot(confs, accs, "o-", color=COLORS[cond], linewidth=2, markersize=8,
                    label=f"{CONDITION_LABELS[cond]}")

        # Compute ECE
        total = sum(bin_sizes)
        if total > 0:
            ece = sum(
                (bin_sizes[i] / total) * abs((bin_confs[i] or 0) / 100 - (bin_accs[i] or 0))
                for i in range(len(bins) - 1) if bin_sizes[i] > 0
            )
            ece_values[cond] = ece

    ax.set_xlabel("Stated Confidence (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add ECE values to legend
    legend_labels = []
    for cond in conditions:
        if cond in ece_values:
            legend_labels.append(f"{CONDITION_LABELS[cond]} (ECE={ece_values[cond]:.3f})")
    if legend_labels:
        ax.legend(fontsize=10, frameon=False, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved to {output_path}")

    return ece_values


# ──────────────────────────────────────────────
# FIGURE 3: Confidence utility quadrants
# ──────────────────────────────────────────────

def figure_confidence_quadrants(df: pd.DataFrame, confidence_threshold: float = 70,
                                 output_path: str = "data/analysis/fig3_confidence_quadrants.png"):
    """2×2 quadrant analysis: confidence vs accuracy for each condition."""

    conditions = [c for c in ["baseline", "adversarial", "structured"] if c in df["condition"].unique()]

    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    quadrant_data = {}

    for ax, cond in zip(axes, conditions):
        subset = df[(df["condition"] == cond) & df["numeric_confidence"].notna()].copy()
        if len(subset) == 0:
            continue

        high_conf = subset["numeric_confidence"] >= confidence_threshold
        correct = subset["top1_correct"]

        # Four quadrants
        safe = (high_conf & correct).sum()           # high conf + correct
        danger = (high_conf & ~correct).sum()         # high conf + wrong
        caution = (~high_conf & correct).sum()        # low conf + correct
        appropriate = (~high_conf & ~correct).sum()   # low conf + wrong

        total = len(subset)
        quadrant_data[cond] = {
            "safe_automation": int(safe),
            "danger_zone": int(danger),
            "unnecessary_caution": int(caution),
            "appropriate_flag": int(appropriate),
            "total": total,
            "danger_rate": danger / total if total > 0 else 0,
        }

        # Plot as stacked/grouped
        categories = ["High\nConfidence", "Low\nConfidence"]
        correct_vals = [safe / total, caution / total]
        incorrect_vals = [danger / total, appropriate / total]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, correct_vals, width, label="Correct",
                       color="#6B9E78", edgecolor="white")
        bars2 = ax.bar(x + width/2, incorrect_vals, width, label="Incorrect",
                       color="#C4785B", edgecolor="white")

        # Highlight danger zone
        if danger > 0:
            bars2[0].set_edgecolor("red")
            bars2[0].set_linewidth(2)

        # Add counts
        for bar, val, count in zip(bars1, correct_vals, [safe, caution]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"n={count}", ha="center", va="bottom", fontsize=8)
        for bar, val, count in zip(bars2, incorrect_vals, [danger, appropriate]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"n={count}", ha="center", va="bottom", fontsize=8)

        ax.set_title(CONDITION_LABELS[cond], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Proportion of Cases", fontsize=11, fontweight="bold")
    axes[-1].legend(fontsize=9, frameon=False, loc="upper right")

    plt.suptitle(f"Confidence Utility Analysis (threshold: {confidence_threshold}%)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved to {output_path}")

    return quadrant_data


# ──────────────────────────────────────────────
# FIGURE 4: Switch table (rescued vs sabotaged)
# ──────────────────────────────────────────────

def figure_switch_table(df: pd.DataFrame, output_path: str = "data/analysis/fig4_switch_table.png"):
    """Visual switch table showing rescued vs sabotaged by condition."""

    conditions = [c for c in ["adversarial", "structured"] if c in df["condition"].unique()]
    baseline = df[df["condition"] == "baseline"][["case_id", "model", "top1_correct"]].copy()
    baseline.columns = ["case_id", "model", "baseline_correct"]

    switch_data = {}

    for cond in conditions:
        cond_df = df[df["condition"] == cond][["case_id", "model", "top1_correct"]].copy()
        cond_df.columns = ["case_id", "model", f"{cond}_correct"]
        merged = baseline.merge(cond_df, on=["case_id", "model"])

        rescued = (~merged["baseline_correct"] & merged[f"{cond}_correct"]).sum()
        sabotaged = (merged["baseline_correct"] & ~merged[f"{cond}_correct"]).sum()
        both_right = (merged["baseline_correct"] & merged[f"{cond}_correct"]).sum()
        both_wrong = (~merged["baseline_correct"] & ~merged[f"{cond}_correct"]).sum()

        switch_data[cond] = {
            "rescued": int(rescued),
            "sabotaged": int(sabotaged),
            "both_right": int(both_right),
            "both_wrong": int(both_wrong),
            "net": int(rescued - sabotaged),
        }

    # Create figure as a clean table
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")

    headers = ["Condition", "Rescued\n(wrong→right)", "Sabotaged\n(right→wrong)", "Net", "Both\nRight", "Both\nWrong"]
    rows = []
    for cond in conditions:
        d = switch_data[cond]
        net_str = f"+{d['net']}" if d['net'] > 0 else str(d['net'])
        rows.append([CONDITION_LABELS[cond], str(d["rescued"]), str(d["sabotaged"]),
                      net_str, str(d["both_right"]), str(d["both_wrong"])])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j, header in enumerate(headers):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color net column
    for i, cond in enumerate(conditions):
        net = switch_data[cond]["net"]
        cell = table[i + 1, 3]
        if net > 0:
            cell.set_facecolor("#D5F5E3")
        elif net < 0:
            cell.set_facecolor("#FADBD8")
        else:
            cell.set_facecolor("#FDEBD0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved to {output_path}")

    return switch_data


# ──────────────────────────────────────────────
# FIGURE 5: Capability-scaling dot plot with reflection deltas
# ──────────────────────────────────────────────

def figure_capability_scaling(df: pd.DataFrame, output_path: str = "data/analysis/fig5_capability_scaling.png"):
    """Dot plot: models ordered by capability on x-axis, reflection delta on y-axis.

    Shows adversarial delta and structured delta for each model, with brackets
    connecting within-provider pairs (e.g., GPT-5.2 ↔ GPT-5.4).
    """

    # Compute per-model accuracy by condition
    model_accuracy = df.groupby(["model", "condition"])["top1_correct"].mean().unstack(fill_value=0)

    if "baseline" not in model_accuracy.columns:
        print("Warning: no baseline condition found, skipping capability scaling figure")
        return {}

    # Compute deltas
    deltas = pd.DataFrame(index=model_accuracy.index)
    if "adversarial" in model_accuracy.columns:
        deltas["adversarial_delta"] = model_accuracy["adversarial"] - model_accuracy["baseline"]
    if "structured" in model_accuracy.columns:
        deltas["structured_delta"] = model_accuracy["structured"] - model_accuracy["baseline"]
    deltas["baseline_accuracy"] = model_accuracy["baseline"]

    # Order by capability_rank if available, otherwise by baseline accuracy
    if "capability_rank" in df.columns:
        rank_map = df.drop_duplicates("model").set_index("model")["capability_rank"].to_dict()
        deltas["rank"] = deltas.index.map(rank_map)
        deltas = deltas.sort_values("rank")
    else:
        deltas = deltas.sort_values("baseline_accuracy")

    # Short model names for display
    short_names = [m.split("/")[-1] for m in deltas.index]

    # Provider families for bracket connections
    family_map = {}
    if "family" in df.columns:
        family_map = df.drop_duplicates("model").set_index("model")["family"].to_dict()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(deltas))

    # Zero line
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

    # Plot deltas
    offset = 0.15
    if "adversarial_delta" in deltas.columns:
        ax.scatter(x - offset, deltas["adversarial_delta"], color=COLORS["adversarial"],
                   s=100, zorder=5, label="Adversarial Δ", edgecolors="white", linewidth=0.5)
        # Stems
        for xi, val in zip(x, deltas["adversarial_delta"]):
            ax.plot([xi - offset, xi - offset], [0, val], color=COLORS["adversarial"],
                    linewidth=1.5, alpha=0.5)

    if "structured_delta" in deltas.columns:
        ax.scatter(x + offset, deltas["structured_delta"], color=COLORS["structured"],
                   s=100, zorder=5, label="Structured Δ", edgecolors="white", linewidth=0.5)
        for xi, val in zip(x, deltas["structured_delta"]):
            ax.plot([xi + offset, xi + offset], [0, val], color=COLORS["structured"],
                    linewidth=1.5, alpha=0.5)

    # Within-provider brackets
    if family_map:
        families = {}
        for i, model in enumerate(deltas.index):
            fam = family_map.get(model, "")
            if fam:
                families.setdefault(fam, []).append(i)

        for fam, indices in families.items():
            if len(indices) == 2:
                i1, i2 = indices
                # Draw bracket below the plot
                bracket_y = ax.get_ylim()[0] + 0.02
                ax.annotate("", xy=(i1, bracket_y), xytext=(i2, bracket_y),
                            arrowprops=dict(arrowstyle="<->", color="gray",
                                            lw=1.5, connectionstyle="bar,fraction=0.3"))

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Model (ordered by capability)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy Δ vs Baseline", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    # Shade regions
    ylim = ax.get_ylim()
    ax.fill_between([-0.5, len(deltas) - 0.5], 0, ylim[1], color="green", alpha=0.03)
    ax.fill_between([-0.5, len(deltas) - 0.5], ylim[0], 0, color="red", alpha=0.03)
    ax.text(len(deltas) - 0.7, ylim[1] * 0.85, "Reflection helped", fontsize=9,
            ha="right", va="top", color="green", alpha=0.5, style="italic")
    ax.text(len(deltas) - 0.7, ylim[0] * 0.85, "Reflection hurt", fontsize=9,
            ha="right", va="bottom", color="red", alpha=0.5, style="italic")

    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, len(deltas) - 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 5 saved to {output_path}")

    return deltas.to_dict(orient="index")


# ──────────────────────────────────────────────
# SUPPLEMENTAL FIGURE S1: Confidence shift (Phase 1 → Phase 3)
# ──────────────────────────────────────────────

def figure_confidence_shift(df: pd.DataFrame, output_path: str = "data/analysis/figS1_confidence_shift.png"):
    """Arrow plot showing confidence change from Phase 1 to Phase 3 for reflection conditions.

    Each response is an arrow: start = Phase 1 confidence, end = Phase 3 confidence.
    Colored by whether the final diagnosis was correct (green) or incorrect (red).
    Shows whether models appropriately adjust confidence during reflection.
    """

    reflection_df = df[df["condition"].isin(["adversarial", "structured"])].copy()
    reflection_df = reflection_df[
        reflection_df["phase1_confidence"].notna() &
        reflection_df["numeric_confidence"].notna()
    ]

    if len(reflection_df) < 3:
        print("Warning: insufficient data for confidence shift figure")
        return {}

    conditions = [c for c in ["adversarial", "structured"] if c in reflection_df["condition"].unique()]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 6), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    results = {}

    for ax, cond in zip(axes, conditions):
        subset = reflection_df[reflection_df["condition"] == cond].copy()
        subset["shift"] = subset["numeric_confidence"] - subset["phase1_confidence"]

        correct = subset[subset["top1_correct"] == True]
        incorrect = subset[subset["top1_correct"] == False]

        # Plot arrows for each response
        for _, row in incorrect.iterrows():
            ax.annotate("", xy=(1, row["numeric_confidence"]), xytext=(0, row["phase1_confidence"]),
                         arrowprops=dict(arrowstyle="->", color="#C75B3F", alpha=0.4, lw=1.5))

        for _, row in correct.iterrows():
            ax.annotate("", xy=(1, row["numeric_confidence"]), xytext=(0, row["phase1_confidence"]),
                         arrowprops=dict(arrowstyle="->", color="#7AA874", alpha=0.6, lw=2))

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Phase 1", "Phase 3"])
        ax.set_ylabel("Confidence (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 100)
        ax.set_title(f"{cond.title()} Reflection", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#7AA874", lw=2, label=f"Correct (n={len(correct)})"),
            Line2D([0], [0], color="#C75B3F", lw=1.5, label=f"Incorrect (n={len(incorrect)})"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=False)

        results[cond] = {
            "n": len(subset),
            "mean_shift": float(subset["shift"].mean()),
            "correct_mean_shift": float(correct["shift"].mean()) if len(correct) > 0 else None,
            "incorrect_mean_shift": float(incorrect["shift"].mean()) if len(incorrect) > 0 else None,
        }

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure S1 saved to {output_path}")

    return results


# ──────────────────────────────────────────────
# SUPPLEMENTAL FIGURE S2: Response length analysis
# ──────────────────────────────────────────────

def figure_response_length(df: pd.DataFrame, output_path: str = "data/analysis/figS2_response_length.png"):
    """Box plot of response length by condition, with correlation to accuracy annotated."""

    df_len = df.copy()
    df_len["response_length"] = df_len["response"].str.len()

    conditions = sorted(df_len["condition"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Box plot of response length by condition
    data_for_box = [df_len[df_len["condition"] == c]["response_length"].values for c in conditions]
    bp = ax1.boxplot(data_for_box, labels=[c.title() for c in conditions], patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))

    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(COLORS.get(cond, "#888888"))
        patch.set_alpha(0.6)

    ax1.set_ylabel("Response Length (characters)")
    ax1.set_title("A. Response Length by Condition", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Length vs accuracy scatter
    model_cond_stats = df_len.groupby(["model", "condition"]).agg(
        mean_length=("response_length", "mean"),
        accuracy=("top1_correct", "mean")
    ).reset_index()

    for cond in conditions:
        subset = model_cond_stats[model_cond_stats["condition"] == cond]
        ax2.scatter(subset["mean_length"], subset["accuracy"],
                    color=COLORS.get(cond, "#888888"), s=80, label=cond.title(),
                    edgecolors="white", linewidth=0.5, alpha=0.8)

    ax2.set_xlabel("Mean Response Length (characters)")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("B. Length vs Accuracy", fontweight="bold")
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure S2 saved to {output_path}")

    # Stats
    results = {}
    for cond in conditions:
        subset = df_len[df_len["condition"] == cond]
        results[cond] = {
            "mean_length": float(subset["response_length"].mean()),
            "median_length": float(subset["response_length"].median()),
            "std_length": float(subset["response_length"].std()),
        }
    return results


# ──────────────────────────────────────────────
# SUPPLEMENTAL FIGURE S3: Differential diversity
# ──────────────────────────────────────────────

def figure_differential_diversity(df: pd.DataFrame, output_path: str = "data/analysis/figS3_differential_diversity.png"):
    """Bar chart of differential diagnosis diversity by condition.

    Measures: mean number of unique diagnoses in top-5 differential,
    and whether broader differentials correlate with accuracy.
    """
    from backend.trace_parser import extract_differential

    df_div = df.copy()

    def count_differential(row):
        response = row["response"]
        if row["condition"] in ("adversarial", "structured"):
            parsed = parse_adversarial_trace(response)
            text = parsed.get("phase3", response)
        else:
            text = response
        diff = extract_differential(text)
        return len(diff)

    df_div["diff_count"] = df_div.apply(count_differential, axis=1)

    conditions = sorted(df_div["condition"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions))
    means = [df_div[df_div["condition"] == c]["diff_count"].mean() for c in conditions]
    stds = [df_div[df_div["condition"] == c]["diff_count"].std() for c in conditions]
    colors = [COLORS.get(c, "#888888") for c in conditions]

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5,
                  edgecolor="white", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions])
    ax.set_ylabel("Mean Differential Diagnoses Extracted")
    ax.set_title("Differential Diagnosis Breadth by Condition", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate means
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure S3 saved to {output_path}")

    return {c: {"mean": m, "std": s} for c, m, s in zip(conditions, means, stds)}


# ──────────────────────────────────────────────
# SUPPLEMENTAL FIGURE S4: Per-case heatmap (model × case)
# ──────────────────────────────────────────────

def figure_case_heatmap(df: pd.DataFrame, output_path: str = "data/analysis/figS4_case_heatmap.png"):
    """Heatmap showing correctness for each model × case, one panel per condition.

    Reveals which cases are 'reflection-responsive' vs 'reflection-resistant'.
    """

    conditions = sorted(df["condition"].unique())
    models = sorted(df["model"].unique(), key=lambda m: m.split("/")[-1])
    cases = sorted(df["case_id"].unique())

    if len(cases) < 2 or len(models) < 2:
        print("Warning: insufficient data for case heatmap")
        return {}

    n_conds = len(conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(max(4 * n_conds, 8), max(len(cases) * 0.4, 4)),
                              sharey=True)
    if n_conds == 1:
        axes = [axes]

    short_models = [m.split("/")[-1] for m in models]
    short_cases = [c.replace("NEJM_CPC_", "") for c in cases]

    for ax, cond in zip(axes, conditions):
        # Build matrix: cases × models
        matrix = np.full((len(cases), len(models)), np.nan)
        for i, case in enumerate(cases):
            for j, model in enumerate(models):
                row = df[(df["case_id"] == case) & (df["model"] == model) & (df["condition"] == cond)]
                if len(row) > 0:
                    matrix[i, j] = float(row.iloc[0]["top1_correct"])

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#C75B3F", "#7AA874"])  # red=wrong, green=correct

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_models, rotation=45, ha="right", fontsize=8)
        ax.set_title(cond.title(), fontweight="bold")

        if ax == axes[0]:
            ax.set_yticks(range(len(cases)))
            ax.set_yticklabels(short_cases, fontsize=8)

    # Colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#7AA874", label="Correct"),
                       Patch(facecolor="#C75B3F", label="Incorrect")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Diagnostic Accuracy by Case × Model × Condition", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure S4 saved to {output_path}")

    # Identify reflection-responsive cases
    responsive = {}
    for case in cases:
        bl = df[(df["case_id"] == case) & (df["condition"] == "baseline")]["top1_correct"].mean()
        for cond in [c for c in conditions if c != "baseline"]:
            ref = df[(df["case_id"] == case) & (df["condition"] == cond)]["top1_correct"].mean()
            responsive[(case, cond)] = {"baseline_acc": float(bl), "reflection_acc": float(ref),
                                         "delta": float(ref - bl)}

    return responsive


# ──────────────────────────────────────────────
# SUPPLEMENTAL FIGURE S5: Within-provider capability slope chart
# ──────────────────────────────────────────────

def figure_within_provider_slopes(df: pd.DataFrame,
                                   output_path: str = "data/analysis/figS5_within_provider_slopes.png"):
    """Slope chart comparing accuracy between standard and frontier models
    within each provider family, one panel per condition.

    Each provider family is a line connecting standard (left) to frontier (right).
    Shows whether reflection benefit scales with model capability within provider.
    """

    # Infer provider pairs from family column or MODEL_METADATA
    if "family" not in df.columns or "tier" not in df.columns:
        print("Warning: 'family' and 'tier' columns required for within-provider slopes")
        return {}

    conditions = sorted(df["condition"].unique())
    families = sorted(df["family"].dropna().unique())

    if len(families) < 1:
        print("Warning: no provider families found for slope chart")
        return {}

    # Provider family colors
    family_colors = {
        "OpenAI": "#10A37F",
        "Anthropic": "#D4A574",
        "Google": "#4285F4",
    }
    default_colors = ["#555555", "#888888", "#AAAAAA"]

    n_conds = len(conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 6), sharey=True)
    if n_conds == 1:
        axes = [axes]

    results = {}

    for ax, cond in zip(axes, conditions):
        cond_df = df[df["condition"] == cond]

        for i, family in enumerate(families):
            fam_df = cond_df[cond_df["family"] == family]

            # Get standard and frontier accuracy
            standard = fam_df[fam_df["tier"] == "standard"]
            frontier = fam_df[fam_df["tier"] == "frontier"]

            if len(standard) == 0 or len(frontier) == 0:
                continue

            std_acc = standard["top1_correct"].mean() * 100
            fro_acc = frontier["top1_correct"].mean() * 100

            std_model = standard["model"].iloc[0].split("/")[-1]
            fro_model = frontier["model"].iloc[0].split("/")[-1]

            color = family_colors.get(family, default_colors[i % len(default_colors)])

            # Draw slope line
            ax.plot([0, 1], [std_acc, fro_acc], color=color, linewidth=2.5, alpha=0.8,
                    marker="o", markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                    zorder=5)

            # Label endpoints
            ax.annotate(std_model, xy=(0, std_acc), xytext=(-0.15, std_acc),
                        fontsize=8, ha="right", va="center", color=color, fontweight="bold")
            ax.annotate(fro_model, xy=(1, fro_acc), xytext=(1.15, fro_acc),
                        fontsize=8, ha="left", va="center", color=color, fontweight="bold")

            # Store results
            key = f"{family}_{cond}"
            delta = fro_acc - std_acc
            results[key] = {
                "standard_model": std_model, "frontier_model": fro_model,
                "standard_accuracy": round(std_acc, 1), "frontier_accuracy": round(fro_acc, 1),
                "delta": round(delta, 1),
            }

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Standard", "Frontier"], fontsize=11)
        ax.set_ylabel("Top-1 Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(CONDITION_LABELS.get(cond, cond.title()), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=family_colors.get(f, "#888"), lw=2.5, marker="o",
               markersize=8, label=f)
        for f in families if f in family_colors
    ]
    if legend_elements:
        fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Within-Provider Capability Scaling", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure S5 saved to {output_path}")

    return results


def extract_problem_representation(response: str) -> Optional[str]:
    """Extract the problem representation from a structured condition Phase 1.

    Looks for the one-sentence summary after 'PROBLEM REPRESENTATION' or
    similar headers.
    """
    import re

    patterns = [
        # After "Problem Representation:" header
        r"(?:problem\s+representation)[:\s]*\n*\s*(.+?)(?:\n\n|\nB\.|\n##|\n\*\*B)",
        # After "A. PROBLEM REPRESENTATION" with content on next line
        r"A\.\s*PROBLEM\s+REPRESENTATION[:\s]*\n+\s*(.+?)(?:\n\n|\nB\.)",
        # Inline after bold
        r"\*\*Problem\s+Representation[:\s]*\*\*\s*(.+?)(?:\n\n|\n\*\*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            # Clean up: take first sentence or first 300 chars
            if len(text) > 300:
                text = text[:300] + "..."
            return text

    return None


def generate_problem_representation_showcase(df: pd.DataFrame, case_id: str,
                                              output_path: str = "data/analysis/problem_representation_showcase.md") -> str:
    """Generate a side-by-side comparison of problem representations for one case.

    For the poster right column.
    """

    md = f"# Problem Representation Showcase: {case_id}\n\n"

    case_df = df[df["case_id"] == case_id]

    for cond in ["adversarial", "structured"]:
        cond_df = case_df[case_df["condition"] == cond]

        md += f"## {CONDITION_LABELS.get(cond, cond)}\n\n"

        for _, row in cond_df.iterrows():
            model_short = row["model"].split("/")[-1]
            parsed = parse_adversarial_trace(row["response"])

            if cond == "structured":
                pr = extract_problem_representation(row["response"])
                if pr:
                    md += f"**{model_short}:** {pr}\n\n"
                else:
                    md += f"**{model_short}:** [Problem representation not extracted]\n\n"
            else:
                # For adversarial, show the leading diagnosis from Phase 1
                leading = extract_leading_diagnosis(parsed.get("phase1", ""))
                md += f"**{model_short}:** Leading dx: {leading or '[not extracted]'}\n\n"

    # Ground truth
    gt = case_df["ground_truth"].iloc[0] if len(case_df) > 0 else "Unknown"
    md += f"\n## Expert Framing\n\n**Ground truth:** {gt}\n"

    with open(output_path, "w") as f:
        f.write(md)

    print(f"Problem representation showcase saved to {output_path}")
    return md


# ──────────────────────────────────────────────
# BRIER SCORE AND ECE
# ──────────────────────────────────────────────

def compute_calibration_metrics(df: pd.DataFrame) -> dict:
    """Compute Brier score and Expected Calibration Error for each condition."""

    results = {}

    for cond in df["condition"].unique():
        subset = df[(df["condition"] == cond) & df["numeric_confidence"].notna()].copy()
        if len(subset) < 5:
            continue

        conf = subset["numeric_confidence"].values / 100  # normalize to 0-1
        correct = subset["top1_correct"].astype(float).values

        # Brier score: mean squared error of probability estimates
        brier = np.mean((conf - correct) ** 2)

        # ECE: expected calibration error (binned)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            in_bin = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = conf[in_bin].mean()
                ece += (in_bin.sum() / len(conf)) * abs(bin_acc - bin_conf)

        # Maximum Calibration Error
        mce = 0
        for i in range(n_bins):
            in_bin = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = conf[in_bin].mean()
                mce = max(mce, abs(bin_acc - bin_conf))

        results[cond] = {
            "brier_score": float(brier),
            "ece": float(ece),
            "mce": float(mce),
            "n": len(subset),
            "mean_confidence": float(conf.mean()),
            "mean_accuracy": float(correct.mean()),
        }

    return results


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "data/analysis/scored_results.csv"

    # Load scored results (output of analyze.py)
    if results_path.endswith(".csv"):
        df = pd.read_csv(results_path)
    else:
        from analysis.analyze import load_results, score_accuracy
        df = load_results(results_path)
        scores = df.apply(score_accuracy, axis=1, result_type="expand")
        df = pd.concat([df, scores], axis=1)

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating poster figures from {len(df)} results...\n")

    # Figure 1: Accuracy by difficulty
    figure_accuracy_by_difficulty(df)

    # Figure 2: Calibration plot + ECE
    ece_values = figure_calibration(df)
    print(f"ECE values: {ece_values}")

    # Figure 3: Confidence utility quadrants
    quadrant_data = figure_confidence_quadrants(df)
    print(f"Quadrant data: {json.dumps(quadrant_data, indent=2)}")

    # Figure 4: Switch table
    switch_data = figure_switch_table(df)
    print(f"Switch data: {json.dumps(switch_data, indent=2)}")

    # Figure 5: Capability-scaling dot plot
    scaling_data = figure_capability_scaling(df)
    print(f"Scaling data: {json.dumps(scaling_data, indent=2)}")

    # ── Supplemental Figures ──
    print("\nGenerating supplemental figures...\n")

    # Figure S1: Confidence shift (Phase 1 → Phase 3)
    if "phase1_confidence" in df.columns:
        shift_data = figure_confidence_shift(df)
        print(f"Confidence shift data: {json.dumps(shift_data, indent=2)}")

    # Figure S2: Response length analysis
    length_data = figure_response_length(df)
    print(f"Response length data: {json.dumps(length_data, indent=2)}")

    # Figure S3: Differential diversity
    diversity_data = figure_differential_diversity(df)
    print(f"Differential diversity: {json.dumps(diversity_data, indent=2)}")

    # Figure S4: Per-case heatmap
    heatmap_data = figure_case_heatmap(df)
    if heatmap_data:
        # Show most reflection-responsive cases
        responsive = [(k, v) for k, v in heatmap_data.items() if v["delta"] != 0]
        responsive.sort(key=lambda x: abs(x[1]["delta"]), reverse=True)
        print(f"Most reflection-responsive cases: {responsive[:5]}")

    # Figure S5: Within-provider capability slopes
    if "family" in df.columns and "tier" in df.columns:
        slope_data = figure_within_provider_slopes(df)
        print(f"Within-provider slopes: {json.dumps(slope_data, indent=2)}")

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(df)
    print(f"\nCalibration metrics:")
    for cond, metrics in cal_metrics.items():
        print(f"  {cond}: Brier={metrics['brier_score']:.4f}, ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}")

    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    # Problem representation showcase (use first hard case)
    hard_cases = df[df["difficulty"] == "hard"]["case_id"].unique()
    if len(hard_cases) > 0:
        generate_problem_representation_showcase(df, hard_cases[0])

    print(f"\nAll poster figures saved to {output_dir}/")