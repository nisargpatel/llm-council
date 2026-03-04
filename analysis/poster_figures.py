"""
Figure generation and supplementary analysis.
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
# COLOR PALETTE (muted tone)
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
# PROBLEM REPRESENTATION EXTRACTION
# ──────────────────────────────────────────────

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

    For the paper or poster right column.
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

    print(f"Generating paper/poster figures from {len(df)} results...\n")

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

    print(f"\nAll paper figures saved to {output_dir}/")