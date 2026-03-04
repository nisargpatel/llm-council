"""
Chairman Scoring Module
Uses Opus 4.6 (via OpenRouter) to score whether each model's leading diagnosis
matches the ground truth. More robust than regex-based fuzzy matching for
diagnoses with multiple valid names.

Usage:
    python3 -m backend.chairman_scorer data/experiment/results_incremental.jsonl

Outputs:
    data/experiment/chairman_scores.jsonl — one score per result
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import CHAIRMAN_MODEL
from backend.openrouter import query_model
from backend.trace_parser import (
    parse_adversarial_trace,
    extract_leading_diagnosis,
    extract_differential,
)


SCORING_PROMPT = """You are an expert medical diagnostician serving as an impartial judge.
You will be given a model's diagnostic output and the known correct diagnosis.
Your task is to determine whether the model arrived at the correct diagnosis.

## Ground Truth Diagnosis
{ground_truth}

## Model's Response
Leading diagnosis: {leading_diagnosis}
Top 5 differential: {differential}

## Scoring Instructions
Score the model's performance using these criteria:

- **top1_correct**: Does the model's LEADING diagnosis match the ground truth?
  Accept synonyms, reasonable abbreviations, and equivalent diagnostic terms.
  For example, "Lyme carditis" and "cardiac Lyme disease" are equivalent.
  "Meningococcemia" and "disseminated meningococcal disease" are equivalent.
  Do NOT accept partial matches like "sepsis" for "meningococcemia" or
  "viral infection" for "dengue hemorrhagic fever."

- **top3_correct**: Does the ground truth appear anywhere in the model's top 3
  differential diagnoses (using the same synonym-matching logic)?

- **top5_correct**: Does the ground truth appear anywhere in the model's top 5
  differential diagnoses?

Respond with ONLY a JSON object, no other text:
{{"top1_correct": true/false, "top3_correct": true/false, "top5_correct": true/false, "reasoning": "brief explanation"}}"""


async def score_single_result(result: dict) -> dict:
    """Score one experiment result using the chairman model."""

    response = result["response"]
    condition = result["condition"]

    # Extract the final assessment
    if condition in ("adversarial", "structured"):
        parsed = parse_adversarial_trace(response)
        final_text = parsed.get("phase3", response)
    else:
        final_text = response

    leading = extract_leading_diagnosis(final_text) or "[not extracted]"
    differential = extract_differential(final_text)
    diff_str = "\n".join(f"{i+1}. {d}" for i, d in enumerate(differential)) or "[not extracted]"

    prompt = SCORING_PROMPT.format(
        ground_truth=result["ground_truth"],
        leading_diagnosis=leading,
        differential=diff_str,
    )

    messages = [
        {"role": "system", "content": "You are a precise medical scoring judge. Respond only with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = await query_model(CHAIRMAN_MODEL, messages)
        content = raw["content"] if isinstance(raw, dict) else raw

        # Parse JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        score = json.loads(content)

        return {
            "case_id": result["case_id"],
            "model": result["model"],
            "condition": result["condition"],
            "ground_truth": result["ground_truth"],
            "extracted_leading": leading,
            "chairman_top1": score.get("top1_correct", False),
            "chairman_top3": score.get("top3_correct", False),
            "chairman_top5": score.get("top5_correct", False),
            "chairman_reasoning": score.get("reasoning", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {
            "case_id": result["case_id"],
            "model": result["model"],
            "condition": result["condition"],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def score_all_results(results_path: str, output_path: str = None) -> list:
    """Score all experiment results using the chairman model."""

    if output_path is None:
        output_path = str(Path(results_path).parent / "chairman_scores.jsonl")

    # Load results
    with open(results_path) as f:
        if results_path.endswith(".jsonl"):
            results = [json.loads(line) for line in f if line.strip()]
        else:
            results = json.load(f)

    # Filter out errors
    results = [r for r in results if "error" not in r]

    print(f"Scoring {len(results)} results with {CHAIRMAN_MODEL}...")
    print(f"Estimated API calls: {len(results)}")
    print(f"Output: {output_path}\n")

    scores = []
    for i, result in enumerate(results):
        score = await score_single_result(result)
        scores.append(score)

        # Save incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(score) + "\n")

        status = "✓" if score.get("chairman_top1") else "✗"
        if "error" in score:
            status = "ERROR"

        print(f"[{i+1}/{len(results)}] {status} | {score.get('model', '?')} | {score.get('condition', '?')} | {score.get('case_id', '?')}")

        # Rate limit
        await asyncio.sleep(0.5)

    # Summary
    valid = [s for s in scores if "error" not in s]
    if valid:
        top1_acc = sum(1 for s in valid if s["chairman_top1"]) / len(valid)
        print(f"\nChairman scoring complete. {len(valid)} scored, {len(scores) - len(valid)} errors.")
        print(f"Overall top-1 accuracy (chairman-scored): {top1_acc:.1%}")

    return scores


def merge_chairman_scores(results_path: str, scores_path: str, output_path: str = None):
    """Merge chairman scores back into the main results DataFrame.

    Produces a CSV with both regex-based and chairman-based accuracy columns
    for comparison and validation.
    """
    import pandas as pd

    # Load results
    with open(results_path) as f:
        if results_path.endswith(".jsonl"):
            results = [json.loads(line) for line in f if line.strip()]
        else:
            results = json.load(f)
    df = pd.DataFrame(results)

    # Load chairman scores
    with open(scores_path) as f:
        scores = [json.loads(line) for line in f if line.strip()]
    scores_df = pd.DataFrame(scores)

    # Merge on case_id + model + condition
    merged = df.merge(
        scores_df[["case_id", "model", "condition", "chairman_top1", "chairman_top3", "chairman_top5", "chairman_reasoning"]],
        on=["case_id", "model", "condition"],
        how="left"
    )

    if output_path is None:
        output_path = str(Path(results_path).parent / "results_with_chairman.csv")

    merged.to_csv(output_path, index=False)
    print(f"Merged results saved to {output_path}")

    # Validation: compare regex vs chairman scoring
    if "top1_correct" in merged.columns and "chairman_top1" in merged.columns:
        both = merged[merged["chairman_top1"].notna() & merged["top1_correct"].notna()]
        if len(both) > 0:
            agreement = (both["top1_correct"] == both["chairman_top1"]).mean()
            print(f"Regex vs Chairman agreement: {agreement:.1%} ({len(both)} cases)")

            # Disagreements
            disagree = both[both["top1_correct"] != both["chairman_top1"]]
            if len(disagree) > 0:
                print(f"Disagreements ({len(disagree)}):")
                for _, row in disagree.iterrows():
                    print(f"  {row['case_id']} | {row['model']} | {row['condition']} | "
                          f"regex={row['top1_correct']} chairman={row['chairman_top1']} | "
                          f"{row.get('chairman_reasoning', '')[:80]}")

    return merged


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Score:  python3 -m backend.chairman_scorer data/experiment/results_incremental.jsonl")
        print("  Merge:  python3 -m backend.chairman_scorer merge results.jsonl scores.jsonl")
        sys.exit(1)

    if sys.argv[1] == "merge":
        merge_chairman_scores(sys.argv[2], sys.argv[3])
    else:
        asyncio.run(score_all_results(sys.argv[1]))