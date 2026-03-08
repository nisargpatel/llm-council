"""
NLP-based diagnosis extraction using a lightweight LLM.

Usage:
    # Extract diagnoses from existing results (no re-running the experiment)
    python3 -m backend.diagnosis_extractor data/experiment/results_incremental.jsonl

    # Output: data/experiment/extracted_diagnoses.jsonl
    # Each line: {"case_id", "model", "condition", "leading_diagnosis", "differential_top5", "numeric_confidence"}

Cost: ~$0.50–1.00 for 750 responses using gpt-5.1 via OpenRouter
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import OPENROUTER_API_URL, OPENROUTER_API_KEY
from backend.openrouter import query_model
from backend.trace_parser import parse_adversarial_trace


EXTRACTION_MODEL = "openai/gpt-5.1"

EXTRACTION_PROMPT = """Extract the following from this clinical diagnostic response. Return ONLY valid JSON with no other text.

{{
  "leading_diagnosis": "the single leading/top/most likely diagnosis (short name only, e.g. 'Severe dengue' or 'Meningococcemia')",
  "differential_top5": ["diagnosis 1", "diagnosis 2", "diagnosis 3", "diagnosis 4", "diagnosis 5"],
  "numeric_confidence": null or a number 0-100 representing the FINAL stated probability/confidence percentage for the leading diagnosis
}}

Rules:
- For leading_diagnosis: extract ONLY the disease name, not explanations or reasoning
- For differential_top5: extract disease names only, ranked as the author ranked them
- For numeric_confidence: extract the percentage if stated (e.g. "65%" → 65, "~50%" → 50), null if not stated. Look for phrases like "estimated probability", "confidence", or percentages near the leading diagnosis.
- If the response contains multiple phases, extract the diagnosis and differential from the FINAL/REVISED assessment, but search the ENTIRE response for the final confidence value.

Response to extract from:
---
{response_text}
---"""


PHASE1_CONFIDENCE_PROMPT = """Extract ONLY the initial (Phase 1) confidence/probability for the leading diagnosis from this text. Return ONLY valid JSON.

{{
  "phase1_confidence": null or a number 0-100
}}

Rules:
- Extract ONLY the FIRST/INITIAL probability stated, not any revised values.
- Look for phrases like "estimated probability", "confidence: ~50%", "(45%)", etc.
- Return null if no probability is stated.

Text:
---
{phase1_text}
---"""


async def extract_single(result: dict) -> dict:
    """Extract diagnosis info from one experiment result."""

    response = result["response"]
    condition = result["condition"]

    # For reflection conditions, send Phase 3 for diagnosis but full response
    # for confidence (in case confidence is only in Phase 1/2)
    if condition in ("adversarial", "structured"):
        parsed = parse_adversarial_trace(response)
        phase3_text = parsed.get("phase3", "")
        phase1_text = parsed.get("phase1", "")

        # If Phase 3 is very short, it might have parsed wrong — use full response
        if len(phase3_text) < 100:
            target_text = response[:3000]
        else:
            # Send Phase 3 + tail of Phase 2 + enough context for confidence
            target_text = phase3_text[:2000]
            # If no percentage visible in Phase 3, append Phase 1 confidence context
            if "%" not in phase3_text:
                target_text += "\n\n[Earlier in the response, the author stated:]\n"
                # Find lines with confidence/probability in full response
                for line in response.split("\n"):
                    if any(w in line.lower() for w in ["probability", "confidence", "%"]):
                        target_text += line + "\n"
                target_text = target_text[:3000]
    else:
        # Baseline: send first 2000 chars, then append all lines with probability info
        target_text = response[:2000]
        if "%" not in target_text:
            target_text += "\n\n[Additional context from the response:]\n"
            for line in response.split("\n"):
                if any(w in line.lower() for w in ["probability", "confidence", "%", "estimated"]):
                    target_text += line + "\n"
            target_text = target_text[:3500]
        phase1_text = None

    prompt = EXTRACTION_PROMPT.format(response_text=target_text)

    messages = [
        {"role": "system", "content": "You are a precise medical text extraction tool. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = await query_model(EXTRACTION_MODEL, messages)
        content = raw["content"] if isinstance(raw, dict) else raw

        # Clean up response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        extracted = json.loads(content)

        result_dict = {
            "case_id": result["case_id"],
            "model": result["model"],
            "condition": result["condition"],
            "leading_diagnosis": extracted.get("leading_diagnosis"),
            "differential_top5": extracted.get("differential_top5", []),
            "numeric_confidence": extracted.get("numeric_confidence"),
            "extraction_model": EXTRACTION_MODEL,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Extract Phase 1 confidence separately for reflection conditions
        if phase1_text and len(phase1_text) > 50:
            try:
                p1_prompt = PHASE1_CONFIDENCE_PROMPT.format(phase1_text=phase1_text[:1500])
                p1_messages = [
                    {"role": "system", "content": "You are a precise extraction tool. Return only valid JSON."},
                    {"role": "user", "content": p1_prompt},
                ]
                p1_raw = await query_model(EXTRACTION_MODEL, p1_messages)
                p1_content = p1_raw["content"] if isinstance(p1_raw, dict) else p1_raw
                p1_content = p1_content.strip()
                if p1_content.startswith("```"):
                    p1_content = p1_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                p1_extracted = json.loads(p1_content)
                result_dict["phase1_confidence"] = p1_extracted.get("phase1_confidence")
            except Exception:
                result_dict["phase1_confidence"] = None
        
        return result_dict

    except Exception as e:
        return {
            "case_id": result["case_id"],
            "model": result["model"],
            "condition": result["condition"],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def extract_all(results_path: str, output_path: str = None) -> list:
    """Extract diagnoses from all experiment results."""

    if output_path is None:
        output_path = str(Path(results_path).parent / "extracted_diagnoses.jsonl")

    with open(results_path) as f:
        if results_path.endswith(".jsonl"):
            results = [json.loads(line) for line in f if line.strip()]
        else:
            results = json.load(f)

    # Filter out errors
    results = [r for r in results if "error" not in r]

    print(f"Extracting diagnoses from {len(results)} results using {EXTRACTION_MODEL}...")
    print(f"Output: {output_path}\n")

    extractions = []

    for i, result in enumerate(results):
        ext = await extract_single(result)
        extractions.append(ext)

        # Save incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(ext) + "\n")

        dx = ext.get("leading_diagnosis", "?")
        conf = ext.get("numeric_confidence", "?")
        model = result["model"].split("/")[-1]
        status = "✓" if "error" not in ext else "ERROR"

        print(f"[{i+1}/{len(results)}] {status} | {model:20s} | {result['condition']:12s} | {dx} ({conf}%)")

        # Rate limit
        await asyncio.sleep(0.3)

    valid = [e for e in extractions if "error" not in e]
    print(f"\nExtraction complete. {len(valid)}/{len(extractions)} succeeded.")

    return extractions


def load_extractions(path: str) -> dict:
    """Load extractions into a lookup dict keyed by (case_id, model, condition)."""

    lookup = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ext = json.loads(line)
            key = (ext["case_id"], ext["model"], ext["condition"])
            lookup[key] = ext
    return lookup


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m backend.diagnosis_extractor data/experiment/results_incremental.jsonl")
        sys.exit(1)

    asyncio.run(extract_all(sys.argv[1]))