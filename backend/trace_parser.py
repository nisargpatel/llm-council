"""
Parse structured reasoning traces from model responses.
Extracts Phase 1 (initial), Phase 2 (self-critique/second look), Phase 3 (revised)
from adversarial and structured condition responses.

Also extracts:
- Leading diagnosis
- Numeric confidence (0-100%)
- Differential diagnosis list
- Diagnosis change detection
- Reasoning coherence signals
"""

import re
from typing import Optional


def parse_adversarial_trace(response: str) -> dict:
    """Parse a response from adversarial or structured condition into its three phases."""

    phases = {"phase1": "", "phase2": "", "phase3": "", "diagnosis_changed": None}

    # Try to split on ## Phase headers
    phase_pattern = r"##\s*Phase\s*(\d)"
    parts = re.split(phase_pattern, response, flags=re.IGNORECASE)

    if len(parts) >= 7:  # text-before, "1", phase1-text, "2", phase2-text, "3", phase3-text
        phases["phase1"] = parts[2].strip()
        phases["phase2"] = parts[4].strip()
        phases["phase3"] = parts[6].strip()
    else:
        # Fallback: try splitting on numbered headers or bold markers
        for pattern in [
            r"\*\*Phase\s*(\d)\*\*",
            r"Phase\s*(\d)\s*[-:—]",
            r"(\d)\.\s*(?:Initial|Adversarial|Structured|Revised)"
        ]:
            parts = re.split(pattern, response, flags=re.IGNORECASE)
            if len(parts) >= 7:
                phases["phase1"] = parts[2].strip()
                phases["phase2"] = parts[4].strip()
                phases["phase3"] = parts[6].strip()
                break
        else:
            # Could not parse — store full response as phase1
            phases["phase1"] = response
            phases["parse_error"] = True

    # Detect whether diagnosis changed
    phase3_lower = phases["phase3"].lower()
    if any(kw in phase3_lower for kw in ["changed", "revised", "updated", "different", "reconsidered", "has changed"]):
        phases["diagnosis_changed"] = True
    elif any(kw in phase3_lower for kw in ["unchanged", "maintain", "confirmed", "remains", "did not change", "does not change", "not change"]):
        phases["diagnosis_changed"] = False
    # else: None (ambiguous)

    return phases


def extract_leading_diagnosis(response: str) -> Optional[str]:
    """Extract the leading/top diagnosis from a response."""

    patterns = [
        # Markdown header followed by diagnosis on next line (GPT-5.1 style)
        r"#{1,3}\s*(?:final\s*)?(?:leading|top|primary|most likely)\s*diagnosis\s*\n+\s*\*{0,2}([^\n*]+)",
        # Inline bold label with bold value (Gemini style)
        r"\*{0,2}(?:final\s*)?(?:leading|top|primary|most likely)\s*diagnosis[:\s]*\*{0,2}\s*\*{0,2}([^\n*]+)",
        # Plain text label
        r"(?:final\s*)?(?:leading|top|primary|most likely)\s*diagnosis[:\s]*\*{0,2}([^\n*]+)",
        # Revised diagnosis
        r"(?:final|revised)\s*(?:leading)?\s*diagnosis[:\s]*\*{0,2}([^\n*]+)",
        # First item in numbered list as fallback
        r"1\.\s*\*{0,2}([^\n*—\-(]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")

    return None


def extract_confidence(response: str) -> Optional[str]:
    """Extract stated confidence level (categorical)."""

    match = re.search(
        r"confidence[:\s]*\*?\*?(low|medium|moderate|high|very high)",
        response, re.IGNORECASE
    )
    return match.group(1).lower() if match else None


def extract_numeric_confidence(response: str) -> Optional[float]:
    """Extract numeric confidence/probability estimate (0-100%).

    Looks for patterns like:
    - "estimated probability: 75%"
    - "probability (0-100%): 80%"
    - "confidence: 70%"
    - "65% probability"
    - "(85%)"
    """

    patterns = [
        # "estimated probability: 75%" or "probability (0-100%): 80%"
        r"(?:estimated\s+)?probability[^:]*?:\s*(\d{1,3})%",
        # "confidence: 70%" or "confidence level: 65%"
        r"confidence[^:]*?:\s*(\d{1,3})%",
        # "75% confident" or "75% probability" or "75% likely"
        r"(\d{1,3})%\s*(?:confident|probability|likely|chance|certain)",
        # Standalone percentage near diagnosis context — more targeted
        r"(?:leading|top|primary|final)\s*diagnosis[^.]*?(\d{1,3})%",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 100:
                return val

    return None


def extract_differential(response: str) -> list[str]:
    """Extract ranked differential diagnosis list."""

    differentials = []
    # Match numbered lists: "1. Diagnosis" or "1) Diagnosis"
    matches = re.findall(r"\d+[.)]\s*\*?\*?([^\n*—\-(]+)", response)
    for m in matches[:10]:  # cap at 10
        dx = m.strip().rstrip(".")
        if len(dx) > 3 and len(dx) < 200:  # sanity check
            differentials.append(dx)

    return differentials[:5]  # top 5


def extract_coherence_signals(phase2: str, phase3: str) -> dict:
    """Analyze reasoning trace coherence between Phase 2 critique and Phase 3 decision.

    Detects patterns that may predict right-to-wrong switches:
    - Strong counterargument identified but dismissed without adequate justification
    - Phase 2 raises a finding that Phase 3 doesn't address
    - Confidence increases despite acknowledging significant counterevidence

    Returns a dict of coherence metrics.
    """

    signals = {
        "counterargument_strength": None,
        "dismissal_detected": False,
        "unaddressed_findings": False,
        "confidence_direction": None,
    }

    p2_lower = phase2.lower()
    p3_lower = phase3.lower()

    # Detect strong counterargument language in Phase 2
    strong_counter_phrases = [
        "strongest argument against",
        "cannot explain",
        "does not explain",
        "inconsistent with",
        "argues strongly against",
        "pathognomonic for",  # when used for an alternative
        "classic for",        # when used for an alternative
        "highly atypical",
        "does not fit",
    ]

    counter_count = sum(1 for phrase in strong_counter_phrases if phrase in p2_lower)
    signals["counterargument_strength"] = "strong" if counter_count >= 3 else "moderate" if counter_count >= 1 else "weak"

    # Detect dismissal patterns in Phase 3
    dismissal_phrases = [
        "despite",
        "nevertheless",
        "however, the overall",
        "does not outweigh",
        "not sufficient to change",
        "still favors",
        "remains the most",
    ]
    signals["dismissal_detected"] = any(phrase in p3_lower for phrase in dismissal_phrases)

    # Check if Phase 2 mentions a specific finding that Phase 3 doesn't address
    # Look for clinical terms in Phase 2 that don't appear in Phase 3
    clinical_terms_p2 = set(re.findall(r'\b(?:hypoglycemia|thrombocytopenia|leukocytosis|tachycardia|bradycardia|petechiae|DIC|acidosis|hepatitis|AV block|myocarditis)\b', p2_lower))
    clinical_terms_p3 = set(re.findall(r'\b(?:hypoglycemia|thrombocytopenia|leukocytosis|tachycardia|bradycardia|petechiae|DIC|acidosis|hepatitis|AV block|myocarditis)\b', p3_lower))
    unaddressed = clinical_terms_p2 - clinical_terms_p3
    signals["unaddressed_findings"] = len(unaddressed) > 0
    signals["unaddressed_terms"] = list(unaddressed) if unaddressed else []

    # Compare confidence between Phase 1 and Phase 3
    p1_conf = extract_numeric_confidence(phase2)  # Phase 2 text often contains Phase 1 reference
    p3_conf = extract_numeric_confidence(phase3)
    if p1_conf is not None and p3_conf is not None:
        if p3_conf > p1_conf:
            signals["confidence_direction"] = "increased"
        elif p3_conf < p1_conf:
            signals["confidence_direction"] = "decreased"
        else:
            signals["confidence_direction"] = "unchanged"

    return signals


def extract_anchoring_features(baseline_response: str, case: dict) -> dict:
    """Identify potential anchoring patterns in baseline response.

    Checks whether the model's leading diagnosis aligns with the most
    'obvious' or surface-level interpretation of the case, which may
    differ from ground truth.

    Returns anchoring metrics for analysis.
    """

    leading = extract_leading_diagnosis(baseline_response)
    gt = case.get("ground_truth", "").lower()
    difficulty = case.get("difficulty", "")

    result = {
        "leading_diagnosis": leading,
        "ground_truth": case.get("ground_truth", ""),
        "baseline_correct": False,
        "difficulty": difficulty,
    }

    if leading:
        leading_lower = leading.lower()
        result["baseline_correct"] = gt in leading_lower or leading_lower in gt

    # Extract confidence for calibration analysis
    result["baseline_confidence"] = extract_numeric_confidence(baseline_response)

    return result