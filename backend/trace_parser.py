"""
Parse structured reasoning traces from model responses.
Extracts Phase 1 (initial), Phase 2 (self-critique), Phase 3 (revised)
from adversarial condition responses.
"""

import re
from typing import Optional


def parse_adversarial_trace(response: str) -> dict:
    """Parse a response from the adversarial condition into its three phases."""

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
            r"(\d)\.\s*(?:Initial|Adversarial|Revised)"
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
    if any(kw in phase3_lower for kw in ["changed", "revised", "updated", "different", "reconsidered"]):
        phases["diagnosis_changed"] = True
    elif any(kw in phase3_lower for kw in ["unchanged", "maintain", "confirmed", "remains", "did not change"]):
        phases["diagnosis_changed"] = False
    # else: None (ambiguous)

    return phases


def extract_leading_diagnosis(response: str) -> Optional[str]:
    """Extract the leading/top diagnosis from a response."""

    patterns = [
        r"(?:leading|top|primary|most likely)\s*diagnosis[:\s]*\*?\*?([^\n*]+)",
        r"1\.\s*\*?\*?([^\n*—\-(]+)",  # First item in numbered list
        r"(?:final|revised)\s*(?:leading)?\s*diagnosis[:\s]*\*?\*?([^\n*]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")

    return None


def extract_confidence(response: str) -> Optional[str]:
    """Extract stated confidence level."""

    match = re.search(
        r"confidence[:\s]*\*?\*?(low|medium|moderate|high|very high)",
        response, re.IGNORECASE
    )
    return match.group(1).lower() if match else None


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