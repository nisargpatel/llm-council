"""
Statistical analysis for diagnostic reflection experiment.
Compares three conditions: baseline, adversarial self-critique, structured reflection.
Outputs tables and figures.

Key analyses:
1. Primary: McNemar's test for each reflection condition vs baseline
2. Confidence calibration: numeric confidence vs actual accuracy
3. Reasoning coherence: Phase 2 signals that predict right-to-wrong switches
4. Anchoring resistance: escape rate from difficulty-stratified traps
5. Within-provider and capability-scaling comparisons
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import warnings
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import MODEL_METADATA, PROVIDER_PAIRS
from backend.trace_parser import (
    parse_adversarial_trace,
    extract_leading_diagnosis,
    extract_confidence,
    extract_numeric_confidence,
    extract_differential,
    extract_coherence_signals,
    extract_anchoring_features,
)


def load_results(path: str) -> pd.DataFrame:
    """Load experiment results into a DataFrame."""
    with open(path) as f:
        if path.endswith(".jsonl"):
            results = [json.loads(line) for line in f]
        else:
            results = json.load(f)
    return pd.DataFrame(results)


def score_accuracy(row: pd.Series) -> dict:
    """Score a single result against ground truth.

    Returns:
        top1_correct: bool - leading diagnosis matches ground truth
        top3_correct: bool - ground truth appears in top 3 of differential
        top5_correct: bool - ground truth appears in top 5 of differential
        numeric_confidence: float or None - extracted probability estimate
    """
    gt = row["ground_truth"].lower().strip()
    response = row["response"]

    # For reflection conditions, use Phase 3 (revised) diagnosis
    if row["condition"] in ("adversarial", "structured"):
        parsed = parse_adversarial_trace(response)
        final_text = parsed.get("phase3", response)
    else:
        final_text = response

    leading = extract_leading_diagnosis(final_text)
    differential = extract_differential(final_text)
    confidence = extract_numeric_confidence(final_text)

    # Also extract Phase 1 confidence for reflection conditions
    phase1_confidence = None
    if row["condition"] in ("adversarial", "structured"):
        parsed = parse_adversarial_trace(response)
        phase1_confidence = extract_numeric_confidence(parsed.get("phase1", ""))

    # Fuzzy matching — ground truth substring in extracted diagnosis
    def matches(extracted: str, truth: str) -> bool:
        if not extracted:
            return False
        e, t = extracted.lower().strip(), truth.lower().strip()
        return t in e or e in t

    top1 = matches(leading, gt) if leading else False
    top3 = any(matches(d, gt) for d in differential[:3])
    top5 = any(matches(d, gt) for d in differential[:5])

    return {
        "top1_correct": top1,
        "top3_correct": top3 or top1,
        "top5_correct": top5 or top3 or top1,
        "numeric_confidence": confidence,
        "phase1_confidence": phase1_confidence,
        "extracted_diagnosis": leading,
    }


def primary_analysis(df: pd.DataFrame) -> dict:
    """
    Primary analysis: paired comparison of accuracy across conditions.

    Statistical tests:
    - McNemar's test for each reflection condition vs baseline
    - Stratified by difficulty
    - Per-model breakdowns
    """

    # Score all results
    scores = df.apply(score_accuracy, axis=1, result_type="expand")
    df = pd.concat([df, scores], axis=1)

    results = {}
    conditions = df["condition"].unique()

    # ── Overall accuracy by condition ──
    for metric in ["top1_correct", "top3_correct", "top5_correct"]:
        for cond in conditions:
            subset = df[df["condition"] == cond][metric]
            results[f"{cond}_{metric}"] = {
                "mean": subset.mean(),
                "n": len(subset),
            }

    # ── McNemar's test: each reflection condition vs baseline ──
    for cond in [c for c in conditions if c != "baseline"]:
        paired = df[df["condition"].isin(["baseline", cond])].pivot_table(
            index=["case_id", "model"],
            columns="condition",
            values="top1_correct",
            aggfunc="first"
        ).dropna()

        if len(paired) > 0 and "baseline" in paired.columns and cond in paired.columns:
            b_wrong_a_right = ((paired["baseline"] == False) & (paired[cond] == True)).sum()
            b_right_a_wrong = ((paired["baseline"] == True) & (paired[cond] == False)).sum()

            if b_wrong_a_right + b_right_a_wrong > 0:
                mcnemar_stat = (abs(b_wrong_a_right - b_right_a_wrong) - 1)**2 / (b_wrong_a_right + b_right_a_wrong)
                mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            else:
                mcnemar_stat, mcnemar_p = 0, 1.0

            results[f"mcnemar_{cond}_vs_baseline"] = {
                "baseline_wrong_cond_right": int(b_wrong_a_right),
                "baseline_right_cond_wrong": int(b_right_a_wrong),
                "both_right": int(((paired["baseline"] == True) & (paired[cond] == True)).sum()),
                "both_wrong": int(((paired["baseline"] == False) & (paired[cond] == False)).sum()),
                "statistic": mcnemar_stat,
                "p_value": mcnemar_p,
                "net_switches": int(b_wrong_a_right) - int(b_right_a_wrong),
            }

    # ── McNemar's: adversarial vs structured (head-to-head) ──
    if "adversarial" in conditions and "structured" in conditions:
        paired_av = df[df["condition"].isin(["adversarial", "structured"])].pivot_table(
            index=["case_id", "model"],
            columns="condition",
            values="top1_correct",
            aggfunc="first"
        ).dropna()

        if len(paired_av) > 0:
            a_wrong_s_right = ((paired_av["adversarial"] == False) & (paired_av["structured"] == True)).sum()
            a_right_s_wrong = ((paired_av["adversarial"] == True) & (paired_av["structured"] == False)).sum()

            if a_wrong_s_right + a_right_s_wrong > 0:
                stat = (abs(a_wrong_s_right - a_right_s_wrong) - 1)**2 / (a_wrong_s_right + a_right_s_wrong)
                p = 1 - stats.chi2.cdf(stat, df=1)
            else:
                stat, p = 0, 1.0

            results["mcnemar_adversarial_vs_structured"] = {
                "adversarial_wrong_structured_right": int(a_wrong_s_right),
                "adversarial_right_structured_wrong": int(a_right_s_wrong),
                "statistic": stat,
                "p_value": p,
            }

    # ── Stratified by difficulty ──
    for difficulty in ["easy", "moderate", "hard"]:
        subset = df[df["difficulty"] == difficulty]
        for cond in conditions:
            bl = subset[subset["condition"] == cond]["top1_correct"]
            results[f"{difficulty}_{cond}_top1"] = {
                "mean": bl.mean() if len(bl) > 0 else None,
                "n": len(bl),
            }

    # ── By model ──
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        model_results = {}
        for cond in conditions:
            acc = subset[subset["condition"] == cond]["top1_correct"]
            model_results[f"{cond}_mean"] = acc.mean() if len(acc) > 0 else None
        results[f"model_{model}"] = model_results

    return results, df


def confidence_calibration_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze confidence calibration: does numeric confidence predict accuracy?

    Tests:
    - Correlation between stated confidence and actual accuracy
    - Calibration by condition (baseline vs adversarial vs structured)
    - Overconfidence detection: high confidence + wrong answer rate
    - Whether self-reported confidence predicts reflection benefit
    """

    results = {"by_condition": {}, "calibration_bins": {}}

    for cond in df["condition"].unique():
        subset = df[(df["condition"] == cond) & df["numeric_confidence"].notna()].copy()

        if len(subset) < 5:
            continue

        conf = subset["numeric_confidence"].values
        correct = subset["top1_correct"].astype(float).values

        # Point-biserial correlation: confidence vs binary accuracy
        if len(set(correct)) > 1:  # need both correct and incorrect
            r, p = pearsonr(conf, correct)
        else:
            r, p = 0, 1.0

        results["by_condition"][cond] = {
            "n": len(subset),
            "mean_confidence": float(np.mean(conf)),
            "mean_accuracy": float(np.mean(correct)),
            "overconfidence_gap": float(np.mean(conf) / 100 - np.mean(correct)),
            "correlation_r": float(r),
            "correlation_p": float(p),
        }

        # Binned calibration: group by confidence decile
        bins = [0, 30, 50, 70, 85, 100]
        labels = ["0-30%", "30-50%", "50-70%", "70-85%", "85-100%"]
        subset["conf_bin"] = pd.cut(subset["numeric_confidence"], bins=bins, labels=labels, include_lowest=True)

        bin_data = []
        for label in labels:
            bin_subset = subset[subset["conf_bin"] == label]
            if len(bin_subset) > 0:
                bin_data.append({
                    "bin": label,
                    "n": len(bin_subset),
                    "mean_confidence": float(bin_subset["numeric_confidence"].mean()),
                    "actual_accuracy": float(bin_subset["top1_correct"].mean()),
                    "gap": float(bin_subset["numeric_confidence"].mean() / 100 - bin_subset["top1_correct"].mean()),
                })
        results["calibration_bins"][cond] = bin_data

    # ── Does baseline confidence predict reflection benefit? ──
    # For each (case, model), check if baseline confidence correlates with
    # whether reflection helped or hurt
    for cond in [c for c in df["condition"].unique() if c != "baseline"]:
        baseline = df[df["condition"] == "baseline"][["case_id", "model", "top1_correct", "numeric_confidence"]].copy()
        baseline.columns = ["case_id", "model", "baseline_correct", "baseline_confidence"]

        reflection = df[df["condition"] == cond][["case_id", "model", "top1_correct"]].copy()
        reflection.columns = ["case_id", "model", f"{cond}_correct"]

        merged = baseline.merge(reflection, on=["case_id", "model"], how="inner")
        merged = merged[merged["baseline_confidence"].notna()]

        if len(merged) < 10:
            continue

        # Reflection helped = baseline wrong, reflection right
        # Reflection hurt = baseline right, reflection wrong
        merged["reflection_helped"] = (~merged["baseline_correct"] & merged[f"{cond}_correct"]).astype(int)
        merged["reflection_hurt"] = (merged["baseline_correct"] & ~merged[f"{cond}_correct"]).astype(int)
        merged["net_benefit"] = merged["reflection_helped"] - merged["reflection_hurt"]

        # Split by confidence: does reflection help more when confidence is low?
        low_conf = merged[merged["baseline_confidence"] < 60]
        high_conf = merged[merged["baseline_confidence"] >= 60]

        results[f"confidence_predicts_{cond}"] = {
            "n_total": len(merged),
            "low_confidence_n": len(low_conf),
            "low_confidence_net_benefit": float(low_conf["net_benefit"].mean()) if len(low_conf) > 0 else None,
            "low_confidence_helped": int(low_conf["reflection_helped"].sum()) if len(low_conf) > 0 else 0,
            "low_confidence_hurt": int(low_conf["reflection_hurt"].sum()) if len(low_conf) > 0 else 0,
            "high_confidence_n": len(high_conf),
            "high_confidence_net_benefit": float(high_conf["net_benefit"].mean()) if len(high_conf) > 0 else None,
            "high_confidence_helped": int(high_conf["reflection_helped"].sum()) if len(high_conf) > 0 else 0,
            "high_confidence_hurt": int(high_conf["reflection_hurt"].sum()) if len(high_conf) > 0 else 0,
        }

    return results


def coherence_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze reasoning trace coherence in reflection conditions.

    Tests whether coherence signals in Phase 2 predict right-to-wrong switches.
    """

    results = {"conditions": {}}

    for cond in [c for c in df["condition"].unique() if c != "baseline"]:
        cond_df = df[df["condition"] == cond].copy()
        coherence_data = []

        for _, row in cond_df.iterrows():
            parsed = parse_adversarial_trace(row["response"])
            if parsed.get("parse_error"):
                continue

            signals = extract_coherence_signals(parsed["phase2"], parsed["phase3"])

            # Determine if this was a right-to-wrong switch
            baseline_row = df[
                (df["case_id"] == row["case_id"]) &
                (df["model"] == row["model"]) &
                (df["condition"] == "baseline")
            ]

            if len(baseline_row) == 0:
                continue

            baseline_correct = baseline_row.iloc[0].get("top1_correct", False)
            reflection_correct = row.get("top1_correct", False)

            signals["case_id"] = row["case_id"]
            signals["model"] = row["model"]
            signals["baseline_correct"] = baseline_correct
            signals["reflection_correct"] = reflection_correct
            signals["right_to_wrong"] = baseline_correct and not reflection_correct
            signals["wrong_to_right"] = not baseline_correct and reflection_correct
            signals["diagnosis_changed"] = parsed.get("diagnosis_changed")

            coherence_data.append(signals)

        if not coherence_data:
            continue

        coh_df = pd.DataFrame(coherence_data)

        # Do right-to-wrong switches correlate with coherence signals?
        rtw = coh_df[coh_df["right_to_wrong"] == True]
        non_rtw = coh_df[coh_df["right_to_wrong"] == False]

        results["conditions"][cond] = {
            "n_total": len(coh_df),
            "n_right_to_wrong": len(rtw),
            "n_wrong_to_right": len(coh_df[coh_df["wrong_to_right"] == True]),
            "n_diagnosis_changed": int(coh_df["diagnosis_changed"].sum()) if coh_df["diagnosis_changed"].notna().any() else 0,
        }

        # Counterargument strength in RTW vs non-RTW
        if len(rtw) > 0 and len(non_rtw) > 0:
            rtw_strong = (rtw["counterargument_strength"] == "strong").mean()
            non_rtw_strong = (non_rtw["counterargument_strength"] == "strong").mean()
            results["conditions"][cond]["rtw_strong_counter_rate"] = float(rtw_strong)
            results["conditions"][cond]["non_rtw_strong_counter_rate"] = float(non_rtw_strong)

            rtw_dismissal = rtw["dismissal_detected"].mean()
            non_rtw_dismissal = non_rtw["dismissal_detected"].mean()
            results["conditions"][cond]["rtw_dismissal_rate"] = float(rtw_dismissal)
            results["conditions"][cond]["non_rtw_dismissal_rate"] = float(non_rtw_dismissal)

            # Fisher's exact test: strong counterargument × RTW
            a = int((rtw["counterargument_strength"] == "strong").sum())     # strong + RTW
            b = int((rtw["counterargument_strength"] != "strong").sum())     # not strong + RTW
            c = int((non_rtw["counterargument_strength"] == "strong").sum()) # strong + not RTW
            d = int((non_rtw["counterargument_strength"] != "strong").sum()) # not strong + not RTW

            contingency = [[a, b], [c, d]]
            try:
                odds_ratio, fisher_p = stats.fisher_exact(contingency)
                results["conditions"][cond]["fisher_strong_counter"] = {
                    "contingency": contingency,
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(fisher_p),
                }
            except Exception:
                pass

            # Fisher's exact test: dismissal × RTW
            a2 = int(rtw["dismissal_detected"].sum())
            b2 = int((~rtw["dismissal_detected"]).sum())
            c2 = int(non_rtw["dismissal_detected"].sum())
            d2 = int((~non_rtw["dismissal_detected"]).sum())

            try:
                odds_ratio2, fisher_p2 = stats.fisher_exact([[a2, b2], [c2, d2]])
                results["conditions"][cond]["fisher_dismissal"] = {
                    "contingency": [[a2, b2], [c2, d2]],
                    "odds_ratio": float(odds_ratio2),
                    "p_value": float(fisher_p2),
                }
            except Exception:
                pass

        # Store full coherence data for detailed analysis
        results["conditions"][cond]["details"] = coherence_data

    return results


def anchoring_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze anchoring resistance across difficulty levels and conditions.

    For hard cases specifically designed with anchoring traps:
    - What % of models anchor on the 'obvious' wrong diagnosis at baseline?
    - Does reflection break the anchor?
    - Does anchoring resistance vary by model capability?
    """

    results = {"by_difficulty": {}, "by_model": {}}

    for difficulty in ["easy", "moderate", "hard"]:
        diff_df = df[df["difficulty"] == difficulty]
        if len(diff_df) == 0:
            continue

        diff_results = {}
        for cond in diff_df["condition"].unique():
            subset = diff_df[diff_df["condition"] == cond]
            acc = subset["top1_correct"].mean() if len(subset) > 0 else None
            diff_results[f"{cond}_accuracy"] = acc
            diff_results[f"{cond}_n"] = len(subset)

        # Anchoring escape rate: baseline wrong → reflection right
        for cond in [c for c in diff_df["condition"].unique() if c != "baseline"]:
            baseline = diff_df[diff_df["condition"] == "baseline"][["case_id", "model", "top1_correct"]].copy()
            baseline.columns = ["case_id", "model", "baseline_correct"]

            reflection = diff_df[diff_df["condition"] == cond][["case_id", "model", "top1_correct"]].copy()
            reflection.columns = ["case_id", "model", f"{cond}_correct"]

            merged = baseline.merge(reflection, on=["case_id", "model"])
            anchored = merged[~merged["baseline_correct"]]  # wrong at baseline

            if len(anchored) > 0:
                escaped = anchored[anchored[f"{cond}_correct"]].shape[0]
                diff_results[f"{cond}_escape_rate"] = escaped / len(anchored)
                diff_results[f"{cond}_escaped"] = escaped
                diff_results[f"{cond}_anchored_total"] = len(anchored)

        results["by_difficulty"][difficulty] = diff_results

    # By model capability: do stronger models resist anchoring better?
    for model in df["model"].unique():
        meta = MODEL_METADATA.get(model, {})
        model_df = df[df["model"] == model]

        model_results = {"capability_rank": meta.get("capability_rank", 0)}
        for cond in model_df["condition"].unique():
            hard = model_df[(model_df["condition"] == cond) & (model_df["difficulty"] == "hard")]
            model_results[f"{cond}_hard_accuracy"] = hard["top1_correct"].mean() if len(hard) > 0 else None

        results["by_model"][model] = model_results

    return results


def within_provider_analysis(df: pd.DataFrame) -> dict:
    """Within-provider pair analysis comparing standard vs frontier tiers."""

    results = {}

    for pair in PROVIDER_PAIRS:
        provider = pair["provider"]
        std_model = pair["standard"]
        frt_model = pair["frontier"]

        pair_results = {"provider": provider, "standard": std_model, "frontier": frt_model}

        for cond in df["condition"].unique():
            for label, model_id in [("standard", std_model), ("frontier", frt_model)]:
                subset = df[(df["model"] == model_id) & (df["condition"] == cond)]
                acc = subset["top1_correct"].mean() if len(subset) > 0 else None
                pair_results[f"{label}_{cond}_acc"] = acc

        # Delta for each condition vs baseline
        for label, model_id in [("standard", std_model), ("frontier", frt_model)]:
            bl = pair_results.get(f"{label}_baseline_acc", 0) or 0
            for cond in [c for c in df["condition"].unique() if c != "baseline"]:
                cond_acc = pair_results.get(f"{label}_{cond}_acc", 0) or 0
                pair_results[f"{label}_{cond}_delta"] = cond_acc - bl

        results[provider] = pair_results

    return results


def capability_scaling_analysis(df: pd.DataFrame) -> dict:
    """Tests whether reflection benefit scales with model capability."""

    results = {"models": [], "hypothesis": "negative_correlation"}

    for model in df["model"].unique():
        subset = df[df["model"] == model]
        meta = MODEL_METADATA.get(model, {})

        model_data = {
            "model": model,
            "provider": meta.get("provider", ""),
            "tier": meta.get("tier", ""),
            "capability_rank": meta.get("capability_rank", 0),
        }

        for cond in subset["condition"].unique():
            acc = subset[subset["condition"] == cond]["top1_correct"]
            model_data[f"{cond}_accuracy"] = acc.mean() if len(acc) > 0 else None

        bl = model_data.get("baseline_accuracy")
        for cond in [c for c in subset["condition"].unique() if c != "baseline"]:
            ca = model_data.get(f"{cond}_accuracy")
            if bl is not None and ca is not None:
                model_data[f"{cond}_delta"] = ca - bl

        results["models"].append(model_data)

    results["models"].sort(key=lambda x: x["capability_rank"])

    # Correlations for each reflection condition
    ranks = [m["capability_rank"] for m in results["models"]]

    for cond in [c for c in df["condition"].unique() if c != "baseline"]:
        deltas = [m.get(f"{cond}_delta") for m in results["models"]]
        valid = [(r, d) for r, d in zip(ranks, deltas) if d is not None]

        if len(valid) >= 3:
            r_vals, d_vals = zip(*valid)
            rho, rho_p = spearmanr(r_vals, d_vals)
            results[f"spearman_{cond}"] = {
                "rho": float(rho),
                "p_value": float(rho_p),
            }

    return results


def secondary_analysis(df: pd.DataFrame) -> dict:
    """Secondary analyses: diagnosis change rates, directions, and confidence shifts."""

    results = {}

    for cond in [c for c in df["condition"].unique() if c != "baseline"]:
        cond_df = df[df["condition"] == cond].copy()

        parsed_data = cond_df["response"].apply(parse_adversarial_trace)
        cond_df["diagnosis_changed"] = parsed_data.apply(lambda x: x.get("diagnosis_changed"))

        changed = cond_df["diagnosis_changed"].sum()
        total = cond_df["diagnosis_changed"].notna().sum()
        results[f"{cond}_change_rate"] = {
            "changed": int(changed),
            "total": int(total),
            "rate": changed / total if total > 0 else None
        }

        # Direction of changes
        if "top1_correct" in cond_df.columns:
            baseline = df[df["condition"] == "baseline"][["case_id", "model", "top1_correct"]].rename(
                columns={"top1_correct": "baseline_correct"}
            )
            merged = cond_df.merge(baseline, on=["case_id", "model"], how="left")
            changed_rows = merged[merged["diagnosis_changed"] == True]

            results[f"{cond}_change_direction"] = {
                "wrong_to_right": int((~changed_rows["baseline_correct"] & changed_rows["top1_correct"]).sum()),
                "right_to_wrong": int((changed_rows["baseline_correct"] & ~changed_rows["top1_correct"]).sum()),
                "wrong_to_wrong_different": int((~changed_rows["baseline_correct"] & ~changed_rows["top1_correct"]).sum()),
            }

        # ── Confidence shift: Phase 1 → Phase 3 ──
        has_both = cond_df[cond_df["phase1_confidence"].notna() & cond_df["numeric_confidence"].notna()]
        if len(has_both) > 0:
            shifts = has_both["numeric_confidence"] - has_both["phase1_confidence"]
            results[f"{cond}_confidence_shift"] = {
                "n": len(has_both),
                "mean_shift": float(shifts.mean()),
                "median_shift": float(shifts.median()),
                "increased": int((shifts > 0).sum()),
                "decreased": int((shifts < 0).sum()),
                "unchanged": int((shifts == 0).sum()),
            }

            # Confidence shift in RTW vs non-RTW
            if "top1_correct" in has_both.columns:
                merged_shift = has_both.merge(
                    df[df["condition"] == "baseline"][["case_id", "model", "top1_correct"]].rename(
                        columns={"top1_correct": "baseline_correct"}
                    ), on=["case_id", "model"], how="left"
                )
                rtw = merged_shift[merged_shift["baseline_correct"] & ~merged_shift["top1_correct"]]
                non_rtw = merged_shift[~(merged_shift["baseline_correct"] & ~merged_shift["top1_correct"])]

                if len(rtw) > 0:
                    rtw_shifts = rtw["numeric_confidence"] - rtw["phase1_confidence"]
                    results[f"{cond}_confidence_shift_rtw"] = {
                        "n": len(rtw),
                        "mean_shift": float(rtw_shifts.mean()),
                    }
                if len(non_rtw) > 0:
                    non_rtw_shifts = non_rtw["numeric_confidence"] - non_rtw["phase1_confidence"]
                    results[f"{cond}_confidence_shift_non_rtw"] = {
                        "n": len(non_rtw),
                        "mean_shift": float(non_rtw_shifts.mean()),
                    }

    return results


def generate_summary_tables(primary: dict, secondary: dict, df: pd.DataFrame,
                            confidence: dict = None, coherence: dict = None,
                            anchoring: dict = None, within_provider: dict = None,
                            capability: dict = None) -> str:
    """Generate markdown summary tables."""

    conditions = sorted(df["condition"].unique())
    md = "# Results Summary\n\n"

    # ── Table 1: Overall accuracy by condition ──
    md += "## Table 1: Diagnostic Accuracy by Condition\n\n"
    md += "| Metric |"
    for cond in conditions:
        md += f" {cond.title()} |"
    md += "\n|--------|" + "-------|" * len(conditions) + "\n"

    for metric in ["top1_correct", "top3_correct", "top5_correct"]:
        md += f"| {metric} |"
        for cond in conditions:
            r = primary.get(f"{cond}_{metric}", {})
            md += f" {r.get('mean', 0):.1%} |"
        md += "\n"

    # ── Table 2: McNemar's tests ──
    md += "\n## Table 2: McNemar's Tests (Paired Switches)\n\n"
    md += "| Comparison | Rescued | Sabotaged | Net | χ² | p-value |\n"
    md += "|------------|---------|-----------|-----|-----|----------|\n"

    for key, val in primary.items():
        if key.startswith("mcnemar_"):
            label = key.replace("mcnemar_", "").replace("_", " ").title()
            rescued = val.get("baseline_wrong_cond_right", val.get("adversarial_wrong_structured_right", 0))
            sabotaged = val.get("baseline_right_cond_wrong", val.get("adversarial_right_structured_wrong", 0))
            net = val.get("net_switches", rescued - sabotaged)
            md += f"| {label} | {rescued} | {sabotaged} | {net:+d} | {val.get('statistic', 0):.2f} | {val.get('p_value', 1):.4f} |\n"

    # ── Table 3: Accuracy by difficulty × condition ──
    md += "\n## Table 3: Top-1 Accuracy by Difficulty × Condition\n\n"
    md += "| Difficulty |"
    for cond in conditions:
        md += f" {cond.title()} |"
    md += "\n|------------|" + "-------|" * len(conditions) + "\n"

    for diff in ["easy", "moderate", "hard"]:
        md += f"| {diff.title()} |"
        for cond in conditions:
            r = primary.get(f"{diff}_{cond}_top1", {})
            val = r.get("mean")
            md += f" {val:.1%} |" if val is not None else " — |"
        md += "\n"

    # ── Table 4: Confidence Calibration ──
    if confidence and confidence.get("by_condition"):
        md += "\n## Table 4: Confidence Calibration\n\n"
        md += "| Condition | N | Mean Confidence | Mean Accuracy | Overconfidence Gap | r | p |\n"
        md += "|-----------|---|-----------------|---------------|-------------------|---|---|\n"
        for cond, val in confidence["by_condition"].items():
            md += f"| {cond.title()} | {val['n']} | {val['mean_confidence']:.1f}% | {val['mean_accuracy']:.1%} | {val['overconfidence_gap']:+.1%} | {val['correlation_r']:.3f} | {val['correlation_p']:.4f} |\n"

    # ── Table 5: Does Confidence Predict Reflection Benefit? ──
    if confidence:
        for key, val in confidence.items():
            if key.startswith("confidence_predicts_"):
                cond = key.replace("confidence_predicts_", "")
                md += f"\n## Table 5: Baseline Confidence vs {cond.title()} Benefit\n\n"
                md += "| Confidence | N | Helped | Hurt | Net Benefit |\n"
                md += "|------------|---|--------|------|-------------|\n"
                md += f"| Low (<60%) | {val['low_confidence_n']} | {val['low_confidence_helped']} | {val['low_confidence_hurt']} | {val.get('low_confidence_net_benefit', 0) or 0:+.3f} |\n"
                md += f"| High (≥60%) | {val['high_confidence_n']} | {val['high_confidence_helped']} | {val['high_confidence_hurt']} | {val.get('high_confidence_net_benefit', 0) or 0:+.3f} |\n"

    # ── Table 6: Coherence signals ──
    if coherence and coherence.get("conditions"):
        md += "\n## Table 6: Reasoning Coherence Signals\n\n"
        for cond, val in coherence["conditions"].items():
            md += f"### {cond.title()}\n"
            md += f"- Right-to-wrong switches: {val['n_right_to_wrong']}\n"
            md += f"- Wrong-to-right switches: {val['n_wrong_to_right']}\n"
            md += f"- Diagnosis changed: {val['n_diagnosis_changed']}\n"
            if "rtw_strong_counter_rate" in val:
                md += f"- Strong counter in RTW: {val['rtw_strong_counter_rate']:.1%} vs non-RTW: {val['non_rtw_strong_counter_rate']:.1%}\n"
            md += "\n"

    # ── Table 7: Anchoring resistance ──
    if anchoring and anchoring.get("by_difficulty"):
        md += "\n## Table 7: Anchoring Escape Rates\n\n"
        md += "| Difficulty |"
        for cond in [c for c in conditions if c != "baseline"]:
            md += f" {cond.title()} Escape Rate |"
        md += "\n|------------|" + "---------|" * (len(conditions) - 1) + "\n"
        for diff in ["easy", "moderate", "hard"]:
            val = anchoring["by_difficulty"].get(diff, {})
            md += f"| {diff.title()} |"
            for cond in [c for c in conditions if c != "baseline"]:
                rate = val.get(f"{cond}_escape_rate")
                md += f" {rate:.1%} |" if rate is not None else " — |"
            md += "\n"

    # ── Table 8: By model ──
    md += "\n## Table 8: Top-1 Accuracy by Model\n\n"
    md += "| Model |"
    for cond in conditions:
        md += f" {cond.title()} |"
    md += "\n|-------|" + "-------|" * len(conditions) + "\n"

    for key, val in primary.items():
        if key.startswith("model_"):
            model_name = key.replace("model_", "").split("/")[-1]
            md += f"| {model_name} |"
            for cond in conditions:
                acc = val.get(f"{cond}_mean")
                md += f" {acc:.1%} |" if acc is not None else " — |"
            md += "\n"

    return md


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "data/experiment/results_incremental.jsonl"
    df = load_results(results_path)

    print(f"Loaded {len(df)} results")
    print(f"Cases: {df['case_id'].nunique()}, Models: {df['model'].nunique()}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}\n")

    # Primary analysis
    print("Running primary analysis...")
    primary, df_scored = primary_analysis(df)

    # Confidence calibration
    print("Running confidence calibration analysis...")
    confidence = confidence_calibration_analysis(df_scored)

    # Coherence analysis
    print("Running coherence analysis...")
    coherence = coherence_analysis(df_scored)

    # Anchoring analysis
    print("Running anchoring analysis...")
    anchoring = anchoring_analysis(df_scored)

    # Secondary analysis
    print("Running secondary analysis...")
    secondary = secondary_analysis(df_scored)

    # Within-provider pair analysis
    print("Running within-provider pair analysis...")
    within_provider = within_provider_analysis(df_scored)

    # Capability-scaling analysis
    print("Running capability-scaling analysis...")
    capability = capability_scaling_analysis(df_scored)

    # Generate summary
    summary = generate_summary_tables(
        primary, secondary, df_scored,
        confidence=confidence,
        coherence=coherence,
        anchoring=anchoring,
        within_provider=within_provider,
        capability=capability,
    )
    print(summary)

    # Save everything
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)
    with open(output_dir / "primary_results.json", "w") as f:
        json.dump(primary, f, indent=2, default=str)
    with open(output_dir / "confidence_calibration.json", "w") as f:
        json.dump(confidence, f, indent=2, default=str)
    with open(output_dir / "coherence_analysis.json", "w") as f:
        json.dump(coherence, f, indent=2, default=str)
    with open(output_dir / "anchoring_analysis.json", "w") as f:
        json.dump(anchoring, f, indent=2, default=str)
    with open(output_dir / "secondary_results.json", "w") as f:
        json.dump(secondary, f, indent=2, default=str)
    with open(output_dir / "within_provider_results.json", "w") as f:
        json.dump(within_provider, f, indent=2, default=str)
    with open(output_dir / "capability_scaling_results.json", "w") as f:
        json.dump(capability, f, indent=2, default=str)
    df_scored.to_csv(output_dir / "scored_results.csv", index=False)

    print(f"\nAll results saved to {output_dir}/")