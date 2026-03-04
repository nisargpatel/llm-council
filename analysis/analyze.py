"""
Statistical analysis for adversarial self-critique experiment.
Outputs tables and figures for CPH poster and AMIA submission.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

# For the capability-scaling correlation
from scipy.stats import spearmanr, pearsonr

# Append parents for imports 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import MODEL_METADATA, PROVIDER_PAIRS
from backend.trace_parser import (
    parse_adversarial_trace,
    extract_leading_diagnosis,
    extract_confidence,
    extract_differential
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
    """
    gt = row["ground_truth"].lower().strip()
    response = row["response"]

    # For adversarial condition, use Phase 3 (revised) diagnosis
    if row["condition"] == "adversarial":
        parsed = parse_adversarial_trace(response)
        final_text = parsed.get("phase3", response)
    else:
        final_text = response

    leading = extract_leading_diagnosis(final_text)
    differential = extract_differential(final_text)

    # Fuzzy matching — ground truth substring in extracted diagnosis
    def matches(extracted: str, truth: str) -> bool:
        if not extracted:
            return False
        e, t = extracted.lower().strip(), truth.lower().strip()
        return t in e or e in t

    top1 = matches(leading, gt) if leading else False
    top3 = any(matches(d, gt) for d in differential[:3])
    top5 = any(matches(d, gt) for d in differential[:5])

    return {"top1_correct": top1, "top3_correct": top3 or top1, "top5_correct": top5 or top3 or top1}


def primary_analysis(df: pd.DataFrame) -> dict:
    """
    Primary analysis: paired comparison of accuracy across conditions.

    Statistical tests:
    - McNemar's test for paired binary outcomes (baseline vs adversarial, same case × model)
    - Stratified by difficulty
    - Bonferroni correction for multiple comparisons
    """

    # Score all results
    scores = df.apply(score_accuracy, axis=1, result_type="expand")
    df = pd.concat([df, scores], axis=1)

    results = {}

    # ── Overall accuracy by condition ──
    for metric in ["top1_correct", "top3_correct", "top5_correct"]:
        baseline = df[df["condition"] == "baseline"][metric]
        adversarial = df[df["condition"] == "adversarial"][metric]
        results[f"overall_{metric}"] = {
            "baseline_mean": baseline.mean(),
            "adversarial_mean": adversarial.mean(),
            "baseline_n": len(baseline),
            "adversarial_n": len(adversarial),
            "delta": adversarial.mean() - baseline.mean(),
        }

    # ── McNemar's test (paired by case_id × model) ──
    paired = df.pivot_table(
        index=["case_id", "model"],
        columns="condition",
        values="top1_correct",
        aggfunc="first"
    ).dropna()

    if len(paired) > 0:
        # Contingency: baseline_wrong→adv_right vs baseline_right→adv_wrong
        b_wrong_a_right = ((paired["baseline"] == False) & (paired["adversarial"] == True)).sum()
        b_right_a_wrong = ((paired["baseline"] == True) & (paired["adversarial"] == False)).sum()

        if b_wrong_a_right + b_right_a_wrong > 0:
            mcnemar_stat = (abs(b_wrong_a_right - b_right_a_wrong) - 1)**2 / (b_wrong_a_right + b_right_a_wrong)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat, mcnemar_p = 0, 1.0

        results["mcnemar_top1"] = {
            "baseline_wrong_adv_right": int(b_wrong_a_right),
            "baseline_right_adv_wrong": int(b_right_a_wrong),
            "both_right": int(((paired["baseline"] == True) & (paired["adversarial"] == True)).sum()),
            "both_wrong": int(((paired["baseline"] == False) & (paired["adversarial"] == False)).sum()),
            "statistic": mcnemar_stat,
            "p_value": mcnemar_p,
        }

    # ── Stratified by difficulty ──
    for difficulty in ["easy", "moderate", "hard"]:
        subset = df[df["difficulty"] == difficulty]
        for metric in ["top1_correct"]:
            bl = subset[subset["condition"] == "baseline"][metric]
            adv = subset[subset["condition"] == "adversarial"][metric]
            results[f"{difficulty}_{metric}"] = {
                "baseline_mean": bl.mean() if len(bl) > 0 else None,
                "adversarial_mean": adv.mean() if len(adv) > 0 else None,
                "n": len(bl),
                "delta": (adv.mean() - bl.mean()) if len(bl) > 0 and len(adv) > 0 else None,
            }

    # ── By model ──
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        bl = subset[subset["condition"] == "baseline"]["top1_correct"]
        adv = subset[subset["condition"] == "adversarial"]["top1_correct"]
        results[f"model_{model}"] = {
            "baseline_mean": bl.mean(),
            "adversarial_mean": adv.mean(),
            "delta": adv.mean() - bl.mean(),
        }

    return results, df

def within_provider_analysis(df: pd.DataFrame) -> dict:
    """
    Within-provider pair analysis.

    For each provider pair (OpenAI: 5.1 vs 5.2, Anthropic: Sonnet vs Opus),
    tests whether the adversarial self-critique benefit differs between the
    standard-tier and frontier-tier model from the same provider.

    This controls for architectural differences across providers —
    the only variable is capability tier within the same model family.

    Statistical approach:
    - For each pair, compute the adversarial delta (adv_accuracy - baseline_accuracy)
      for both the standard and frontier model
    - Test whether these deltas differ using a bootstrap confidence interval
      on the difference-of-differences
    - Also run McNemar's separately for each model in the pair
    """

    results = {}

    for pair in PROVIDER_PAIRS:
        provider = pair["provider"]
        std_model = pair["standard"]
        frt_model = pair["frontier"]

        pair_results = {"provider": provider, "standard": std_model, "frontier": frt_model}

        for label, model_id in [("standard", std_model), ("frontier", frt_model)]:
            subset = df[df["model"] == model_id]
            bl = subset[subset["condition"] == "baseline"]["top1_correct"]
            adv = subset[subset["condition"] == "adversarial"]["top1_correct"]

            pair_results[f"{label}_baseline_acc"] = bl.mean() if len(bl) > 0 else None
            pair_results[f"{label}_adversarial_acc"] = adv.mean() if len(adv) > 0 else None
            pair_results[f"{label}_delta"] = (adv.mean() - bl.mean()) if len(bl) > 0 and len(adv) > 0 else None
            pair_results[f"{label}_n"] = len(bl)

            # McNemar's for this specific model
            paired = subset.pivot_table(
                index="case_id", columns="condition",
                values="top1_correct", aggfunc="first"
            ).dropna()

            if len(paired) > 0:
                b_w_a_r = ((paired["baseline"] == False) & (paired["adversarial"] == True)).sum()
                b_r_a_w = ((paired["baseline"] == True) & (paired["adversarial"] == False)).sum()
                discordant = b_w_a_r + b_r_a_w

                if discordant > 0:
                    stat = (abs(b_w_a_r - b_r_a_w) - 1)**2 / discordant
                    p_val = 1 - stats.chi2.cdf(stat, df=1)
                else:
                    stat, p_val = 0, 1.0

                pair_results[f"{label}_mcnemar_stat"] = stat
                pair_results[f"{label}_mcnemar_p"] = p_val
                pair_results[f"{label}_rescued"] = int(b_w_a_r)
                pair_results[f"{label}_sabotaged"] = int(b_r_a_w)

        # Difference-of-differences
        std_delta = pair_results.get("standard_delta", 0) or 0
        frt_delta = pair_results.get("frontier_delta", 0) or 0
        pair_results["delta_of_deltas"] = std_delta - frt_delta
        # Positive means standard benefited MORE from self-critique than frontier

        # Bootstrap CI on difference-of-differences
        std_subset = df[df["model"] == std_model]
        frt_subset = df[df["model"] == frt_model]
        boot_diffs = _bootstrap_delta_of_deltas(std_subset, frt_subset, n_boot=2000)
        if boot_diffs is not None:
            pair_results["dod_ci_lower"] = float(np.percentile(boot_diffs, 2.5))
            pair_results["dod_ci_upper"] = float(np.percentile(boot_diffs, 97.5))
            pair_results["dod_p_value"] = float(np.mean(np.sign(boot_diffs) != np.sign(pair_results["delta_of_deltas"])) * 2)  # two-sided

        results[provider] = pair_results

    return results


def _bootstrap_delta_of_deltas(std_df: pd.DataFrame, frt_df: pd.DataFrame, n_boot: int = 2000) -> np.ndarray:
    """
    Bootstrap the difference-of-differences:
    (std_adv_acc - std_base_acc) - (frt_adv_acc - frt_base_acc)

    Resamples at the case level to preserve pairing.
    """
    try:
        case_ids = list(set(std_df["case_id"].unique()) & set(frt_df["case_id"].unique()))
        if len(case_ids) < 10:
            return None

        # Build lookup: (case_id, model, condition) -> correct
        def make_lookup(sub_df):
            lookup = {}
            for _, row in sub_df.iterrows():
                lookup[(row["case_id"], row["condition"])] = row.get("top1_correct", False)
            return lookup

        std_lookup = make_lookup(std_df)
        frt_lookup = make_lookup(frt_df)

        rng = np.random.default_rng(42)
        boot_diffs = []

        for _ in range(n_boot):
            sampled = rng.choice(case_ids, size=len(case_ids), replace=True)

            std_bl = np.mean([std_lookup.get((c, "baseline"), False) for c in sampled])
            std_adv = np.mean([std_lookup.get((c, "adversarial"), False) for c in sampled])
            frt_bl = np.mean([frt_lookup.get((c, "baseline"), False) for c in sampled])
            frt_adv = np.mean([frt_lookup.get((c, "adversarial"), False) for c in sampled])

            std_delta = std_adv - std_bl
            frt_delta = frt_adv - frt_bl
            boot_diffs.append(std_delta - frt_delta)

        return np.array(boot_diffs)

    except Exception:
        return None

def capability_scaling_analysis(df: pd.DataFrame) -> dict:
    """
    Tests whether the adversarial self-critique benefit scales inversely
    with baseline model capability.

    Approach:
    - For each model, compute: baseline_accuracy, adversarial_delta
    - Correlate capability_rank (pre-specified) with adversarial_delta
      using Spearman's rho (rank correlation, appropriate for ordinal predictor)
    - Also correlate baseline_accuracy with adversarial_delta
      using Pearson's r (tests the linear relationship directly)

    Hypothesis: negative correlation — stronger models benefit less from self-critique.
    """

    results = {"models": [], "hypothesis": "negative_correlation"}

    for model in df["model"].unique():
        subset = df[df["model"] == model]
        meta = MODEL_METADATA.get(model, {})

        bl = subset[subset["condition"] == "baseline"]["top1_correct"]
        adv = subset[subset["condition"] == "adversarial"]["top1_correct"]

        baseline_acc = bl.mean() if len(bl) > 0 else None
        adv_acc = adv.mean() if len(adv) > 0 else None
        delta = (adv_acc - baseline_acc) if baseline_acc is not None and adv_acc is not None else None

        results["models"].append({
            "model": model,
            "provider": meta.get("provider", ""),
            "tier": meta.get("tier", ""),
            "capability_rank": meta.get("capability_rank", 0),
            "baseline_accuracy": baseline_acc,
            "adversarial_accuracy": adv_acc,
            "adversarial_delta": delta,
            "n_cases": len(bl),
        })

    # Sort by capability rank
    results["models"].sort(key=lambda x: x["capability_rank"])

    # Extract arrays for correlation
    ranks = [m["capability_rank"] for m in results["models"] if m["adversarial_delta"] is not None]
    deltas = [m["adversarial_delta"] for m in results["models"] if m["adversarial_delta"] is not None]
    baselines = [m["baseline_accuracy"] for m in results["models"] if m["adversarial_delta"] is not None]

    if len(ranks) >= 3:
        # Spearman: capability_rank vs delta
        rho, rho_p = spearmanr(ranks, deltas)
        results["spearman_rank_vs_delta"] = {
            "rho": float(rho),
            "p_value": float(rho_p),
            "interpretation": "negative rho = stronger models benefit less"
        }

        # Pearson: baseline_accuracy vs delta
        r, r_p = pearsonr(baselines, deltas)
        results["pearson_baseline_vs_delta"] = {
            "r": float(r),
            "p_value": float(r_p),
            "interpretation": "negative r = models that are already accurate benefit less"
        }

    # Interaction test: logistic regression with capability_rank × condition
    if len(df) > 0:
        try:
            import statsmodels.api as sm

            reg_df = df[["model", "condition", "top1_correct"]].copy()
            reg_df["capability_rank"] = reg_df["model"].map(
                lambda m: MODEL_METADATA.get(m, {}).get("capability_rank", 0)
            )
            reg_df["is_adversarial"] = (reg_df["condition"] == "adversarial").astype(int)
            reg_df["interaction"] = reg_df["capability_rank"] * reg_df["is_adversarial"]

            X = sm.add_constant(reg_df[["is_adversarial", "capability_rank", "interaction"]])
            y = reg_df["top1_correct"].astype(int)

            logit = sm.Logit(y, X).fit(disp=0)

            results["logistic_interaction"] = {
                "interaction_coef": float(logit.params.get("interaction", 0)),
                "interaction_p": float(logit.pvalues.get("interaction", 1)),
                "adversarial_coef": float(logit.params.get("is_adversarial", 0)),
                "adversarial_p": float(logit.pvalues.get("is_adversarial", 1)),
                "interpretation": "negative interaction_coef = self-critique benefit diminishes with capability"
            }
        except Exception as e:
            results["logistic_interaction"] = {"error": str(e)}

    return results

def secondary_analysis(df: pd.DataFrame) -> dict:
    """
    Secondary analyses:
    - Diagnosis change rate in adversarial condition
    - Confidence calibration
    - Differential diversity (number of unique diagnoses across differential)
    - Direction of diagnosis changes (correct→wrong vs wrong→correct)
    """

    results = {}
    adv = df[df["condition"] == "adversarial"].copy()

    # Parse adversarial traces
    parsed_data = adv["response"].apply(parse_adversarial_trace)
    adv["diagnosis_changed"] = parsed_data.apply(lambda x: x.get("diagnosis_changed"))

    # Change rate
    changed = adv["diagnosis_changed"].sum()
    total = adv["diagnosis_changed"].notna().sum()
    results["change_rate"] = {
        "changed": int(changed),
        "total": int(total),
        "rate": changed / total if total > 0 else None
    }

    # Change rate by difficulty
    for diff in ["easy", "moderate", "hard"]:
        sub = adv[adv["difficulty"] == diff]
        ch = sub["diagnosis_changed"].sum()
        t = sub["diagnosis_changed"].notna().sum()
        results[f"change_rate_{diff}"] = {
            "changed": int(ch),
            "total": int(t),
            "rate": ch / t if t > 0 else None
        }

    # Direction of changes (requires top1 scoring from primary analysis)
    if "top1_correct" in adv.columns:
        # Merge with baseline scores for same case/model
        baseline = df[df["condition"] == "baseline"][["case_id", "model", "top1_correct"]].rename(
            columns={"top1_correct": "baseline_correct"}
        )
        merged = adv.merge(baseline, on=["case_id", "model"], how="left")
        changed_rows = merged[merged["diagnosis_changed"] == True]

        results["change_direction"] = {
            "wrong_to_right": int((~changed_rows["baseline_correct"] & changed_rows["top1_correct"]).sum()),
            "right_to_wrong": int((changed_rows["baseline_correct"] & ~changed_rows["top1_correct"]).sum()),
            "wrong_to_wrong_different": int((~changed_rows["baseline_correct"] & ~changed_rows["top1_correct"]).sum()),
            "right_to_right_different": int((changed_rows["baseline_correct"] & changed_rows["top1_correct"]).sum()),
        }

    return results


def generate_summary_tables(primary: dict, secondary: dict, df: pd.DataFrame, within_provider: dict = None, capability: dict = None) -> str:
    """Generate markdown summary tables for poster/paper."""

    md = "# Results Summary\n\n"

    # Table 1: Overall accuracy
    md += "## Table 1: Diagnostic Accuracy by Condition\n\n"
    md += "| Metric | Baseline | Adversarial | Δ | p-value |\n"
    md += "|--------|----------|-------------|---|----------|\n"
    for metric in ["top1_correct", "top3_correct", "top5_correct"]:
        r = primary.get(f"overall_{metric}", {})
        p = primary.get("mcnemar_top1", {}).get("p_value", "") if metric == "top1_correct" else ""
        md += f"| {metric} | {r.get('baseline_mean', 0):.1%} | {r.get('adversarial_mean', 0):.1%} | {r.get('delta', 0):+.1%} | {p} |\n"

    # Table 2: Accuracy by difficulty
    md += "\n## Table 2: Top-1 Accuracy by Case Difficulty\n\n"
    md += "| Difficulty | N | Baseline | Adversarial | Δ |\n"
    md += "|------------|---|----------|-------------|---|\n"
    for diff in ["easy", "moderate", "hard"]:
        r = primary.get(f"{diff}_top1_correct", {})
        md += f"| {diff.title()} | {r.get('n', 0)} | {r.get('baseline_mean', 0):.1%} | {r.get('adversarial_mean', 0):.1%} | {r.get('delta', 0):+.1%} |\n"

    # Table 3: By model
    md += "\n## Table 3: Top-1 Accuracy by Model\n\n"
    md += "| Model | Baseline | Adversarial | Δ |\n"
    md += "|-------|----------|-------------|---|\n"
    for key, val in primary.items():
        if key.startswith("model_"):
            model_name = key.replace("model_", "")
            md += f"| {model_name} | {val['baseline_mean']:.1%} | {val['adversarial_mean']:.1%} | {val['delta']:+.1%} |\n"

    # Table 4: Diagnosis change analysis
    md += "\n## Table 4: Adversarial Self-Critique — Diagnosis Changes\n\n"
    cr = secondary.get("change_rate", {})
    md += f"Overall change rate: {cr.get('rate', 0):.1%} ({cr.get('changed', 0)}/{cr.get('total', 0)})\n\n"

    if "change_direction" in secondary:
        cd = secondary["change_direction"]
        md += "| Direction | Count |\n|-----------|-------|\n"
        md += f"| Wrong → Right (rescued) | {cd.get('wrong_to_right', 0)} |\n"
        md += f"| Right → Wrong (sabotaged) | {cd.get('right_to_wrong', 0)} |\n"
        md += f"| Wrong → Wrong (lateral move) | {cd.get('wrong_to_wrong_different', 0)} |\n"

    # Table 5: McNemar's contingency
    if "mcnemar_top1" in primary:
        mc = primary["mcnemar_top1"]
        md += "\n## Table 5: McNemar's Test — Paired Outcome Changes\n\n"
        md += "| | Adversarial Correct | Adversarial Wrong |\n"
        md += "|---|---|---|\n"
        md += f"| **Baseline Correct** | {mc['both_right']} | {mc['baseline_right_adv_wrong']} |\n"
        md += f"| **Baseline Wrong** | {mc['baseline_wrong_adv_right']} | {mc['both_wrong']} |\n"
        md += f"\nMcNemar's χ² = {mc['statistic']:.2f}, p = {mc['p_value']:.4f}\n"

    # ── Table 6: Within-Provider Pairs ──
    if within_provider is not None:
        md += "\n## Table 6: Within-Provider Pair Comparison\n\n"
        md += "| Provider | Model | Tier | Baseline | Adversarial | Δ | Rescued | Sabotaged | McNemar p |\n"
        md += "|----------|-------|------|----------|-------------|---|---------|-----------|----------|\n"

        for provider, pr in within_provider.items():
            for tier_label in ["standard", "frontier"]:
                model = pr[tier_label]
                short_name = model.split("/")[-1]
                bl = pr.get(f"{tier_label}_baseline_acc", 0) or 0
                adv = pr.get(f"{tier_label}_adversarial_acc", 0) or 0
                delta = pr.get(f"{tier_label}_delta", 0) or 0
                rescued = pr.get(f"{tier_label}_rescued", 0)
                sabotaged = pr.get(f"{tier_label}_sabotaged", 0)
                p = pr.get(f"{tier_label}_mcnemar_p", "")
                p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
                md += f"| {provider} | {short_name} | {tier_label} | {bl:.1%} | {adv:.1%} | {delta:+.1%} | {rescued} | {sabotaged} | {p_str} |\n"

            dod = pr.get("delta_of_deltas", 0) or 0
            ci_lo = pr.get("dod_ci_lower", "")
            ci_hi = pr.get("dod_ci_upper", "")
            ci_str = f"[{ci_lo:.1%}, {ci_hi:.1%}]" if isinstance(ci_lo, float) else ""
            md += f"\n**{provider} Δ-of-Δ:** {dod:+.1%} (standard benefited {'more' if dod > 0 else 'less'} than frontier), 95% CI: {ci_str}\n\n"

    # ── Table 7: Capability Scaling ──
    if capability is not None:
        md += "\n## Table 7: Adversarial Benefit by Model Capability\n\n"
        md += "| Rank | Model | Baseline Acc | Adversarial Δ |\n"
        md += "|------|-------|-------------|---------------|\n"
        for m in capability.get("models", []):
            short = m["model"].split("/")[-1]
            bl = m.get("baseline_accuracy", 0) or 0
            delta = m.get("adversarial_delta", 0) or 0
            md += f"| {m['capability_rank']} | {short} | {bl:.1%} | {delta:+.1%} |\n"

        if "spearman_rank_vs_delta" in capability:
            sp = capability["spearman_rank_vs_delta"]
            md += f"\nSpearman ρ (capability rank vs Δ): {sp['rho']:.3f}, p = {sp['p_value']:.4f}\n"
        if "pearson_baseline_vs_delta" in capability:
            pr = capability["pearson_baseline_vs_delta"]
            md += f"Pearson r (baseline accuracy vs Δ): {pr['r']:.3f}, p = {pr['p_value']:.4f}\n"
        if "logistic_interaction" in capability and "error" not in capability["logistic_interaction"]:
            li = capability["logistic_interaction"]
            md += f"Logistic interaction (capability × condition): β = {li['interaction_coef']:.3f}, p = {li['interaction_p']:.4f}\n"

    return md


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    results_path = sys.argv[1] if len(sys.argv) > 1 else "data/experiment/results_incremental.jsonl"
    df = load_results(results_path)

    print(f"Loaded {len(df)} results")
    print(f"Cases: {df['case_id'].nunique()}, Models: {df['model'].nunique()}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}\n")
    
    # Primary analysis
    primary, df_scored = primary_analysis(df)

    # ── Interaction model + NNC ──
    import statsmodels.api as sm
    
    df_scored["condition_num"] = (df_scored["condition"] == "adversarial").astype(int)
    df_scored["hard"] = (df_scored["difficulty"] == "hard").astype(int)
    df_scored["interaction"] = df_scored["condition_num"] * df_scored["hard"]
    
    model = sm.Logit(df_scored["top1_correct"].astype(int),
                     sm.add_constant(df_scored[["condition_num", "hard", "interaction"]]))
    result = model.fit()
    print(result.summary())
    
    # NNC calculation
    bl_acc = primary["overall_top1_correct"]["baseline_mean"]
    adv_acc = primary["overall_top1_correct"]["adversarial_mean"]
    if adv_acc > bl_acc:
        nnc = 1 / (adv_acc - bl_acc)
        print(f"\nNumber Needed to Critique (NNC): {nnc:.1f}")
    
    #Secondary analysis
    secondary = secondary_analysis(df_scored)

    # Within-provider pair analysis
    print("Running within-provider pair analysis...")
    within_provider = within_provider_analysis(df_scored)

    # Capability-scaling analysis
    print("Running capability-scaling analysis...")
    capability = capability_scaling_analysis(df_scored)

    # Generate summary with all analyses
    summary = generate_summary_tables(primary, secondary, df_scored,
                                       within_provider=within_provider,
                                       capability=capability)
    print(summary)
    
    # Save everything
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)
    with open(output_dir / "primary_results.json", "w") as f:
        json.dump(primary, f, indent=2, default=str)
    with open(output_dir / "secondary_results.json", "w") as f:
        json.dump(secondary, f, indent=2, default=str)
    with open(output_dir / "within_provider_results.json", "w") as f:
        json.dump(within_provider, f, indent=2, default=str)
    with open(output_dir / "capability_scaling_results.json", "w") as f:
        json.dump(capability, f, indent=2, default=str)
    df_scored.to_csv(output_dir / "scored_results.csv", index=False)

    print(f"\nAll results saved to {output_dir}/")