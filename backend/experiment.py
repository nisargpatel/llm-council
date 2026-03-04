"""
Adversarial Self-Critique Experiment Runner
Runs each case through two conditions:
  - BASELINE: standard diagnostic prompt
  - ADVERSARIAL: standard prompt → self-critique → revised answer
Records all outputs, reasoning traces, latency, and token usage.
"""

import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path

from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL, PROVIDER_PAIRS, MODEL_METADATA
from backend.openrouter import query_model

# ──────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────

BASELINE_SYSTEM = """You are an expert physician and diagnostician. You will be presented with a clinical
case. Provide:
1. Your differential diagnosis (ranked by probability, top 5)
2. Your leading diagnosis with confidence level (low/medium/high)
3. Your reasoning — explain step by step how you arrived at your differential,
   what findings support or argue against each diagnosis, and why you ranked them
   as you did.

Be thorough in your reasoning. Think through the case systematically."""

ADVERSARIAL_SYSTEM = """You are an expert diagnostician engaged in structured diagnostic
deliberation. You will work through this case in two phases.

PHASE 1 — INITIAL ASSESSMENT:
Provide your initial differential diagnosis (top 5, ranked by probability),
your leading diagnosis, and your reasoning.

PHASE 2 — ADVERSARIAL SELF-CRITIQUE:
Now act as a skeptical second opinion. For your TOP diagnosis:
- What is the strongest argument AGAINST this diagnosis?
- What findings in this case are NOT explained by your leading diagnosis?
- What alternative diagnosis could explain the full constellation of findings better?
- Is there a "can't miss" diagnosis you may be anchoring away from?

PHASE 3 — REVISED ASSESSMENT:
After completing your self-critique, provide your FINAL differential and leading
diagnosis. State explicitly whether your self-critique changed your assessment
and why or why not. Assign a final confidence level (low/medium/high).

Label each phase clearly with headers: ## Phase 1, ## Phase 2, ## Phase 3."""

def build_case_prompt(case: dict) -> str:
    """Format a clinical case for the model."""
    prompt = f"## Clinical Case: {case['case_id']}\n\n"
    prompt += case["presentation"]
    if case.get("labs"):
        prompt += f"\n\n**Laboratory/Imaging:**\n{case['labs']}"
    if case.get("additional"):
        prompt += f"\n\n**Additional Information:**\n{case['additional']}"
    prompt += "\n\nProvide your diagnostic assessment."
    return prompt

# ──────────────────────────────────────────────
# CORE EXPERIMENT LOGIC
# ──────────────────────────────────────────────

async def run_single_case(case: dict, model: str, condition: str) -> dict:
    """Run one case through one model under one condition."""

    system_prompt = BASELINE_SYSTEM if condition == "baseline" else ADVERSARIAL_SYSTEM
    case_prompt = build_case_prompt(case)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case_prompt}
    ]

    start_time = time.time()
    raw = await query_model(model, messages)
    latency_ms = int((time.time() - start_time) * 1000)

    # Pull metadata
    meta = MODEL_METADATA.get(model, {})

    return {
        "case_id": case["case_id"],
        "model": model,
        "provider": meta.get("provider", ""),
        "tier": meta.get("tier", ""),
        "family": meta.get("family", ""),
        "capability_rank": meta.get("capability_rank", 0),
        "condition": condition,
        "difficulty": case["difficulty"],
        "category": case.get("category", ""),
        "ground_truth": case["ground_truth"],
        "system_prompt": system_prompt,
        "case_prompt": case_prompt,
        "response": raw["content"] if isinstance(raw, dict) else raw,
        "input_tokens": raw.get("input_tokens", 0) if isinstance(raw, dict) else 0,
        "output_tokens": raw.get("output_tokens", 0) if isinstance(raw, dict) else 0,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }


async def run_experiment(cases: list[dict], output_dir: str = "data/experiment") -> None:
    """Run full experiment: all cases × all models × both conditions."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    total = len(cases) * len(COUNCIL_MODELS) * 2  # 2 conditions
    completed = 0

    for case in cases:
        for model in COUNCIL_MODELS:
            for condition in ["baseline", "adversarial"]:
                try:
                    result = await run_single_case(case, model, condition)
                    results.append(result)
                    completed += 1
                    print(f"[{completed}/{total}] {model} | {condition} | {case['case_id']}")

                    # Save incrementally (crash-safe)
                    with open(f"{output_dir}/results_incremental.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")

                except Exception as e:
                    print(f"ERROR: {model} | {condition} | {case['case_id']}: {e}")
                    results.append({
                        "case_id": case["case_id"],
                        "model": model,
                        "condition": condition,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                # Rate limiting — be polite to OpenRouter
                await asyncio.sleep(1.0)

    # Save complete results
    output_path = f"{output_dir}/results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment complete. {completed}/{total} runs saved to {output_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    case_file = sys.argv[1] if len(sys.argv) > 1 else "data/cases.json"
    with open(case_file) as f:
        cases = json.load(f)

    print(f"Running experiment with {len(cases)} cases × {len(COUNCIL_MODELS)} models × 2 conditions")
    print(f"Models: {MODEL_METADATA.keys()}")
    print(f"Chairman (eval only): {CHAIRMAN_MODEL}")
    print(f"Total API calls: {len(cases) * len(COUNCIL_MODELS) * 2}")
    print(f"Within-provider pairs: {PROVIDER_PAIRS}\n")

    asyncio.run(run_experiment(cases))