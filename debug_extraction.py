"""
Dump raw Phase 3 (or full baseline) text for each response to debug dx extraction issues.
Run from repo root:
    python3 debug_extraction.py
"""

import json
from backend.trace_parser import parse_adversarial_trace, extract_leading_diagnosis

with open('data/experiment/results_incremental.jsonl') as f:
    results = [json.loads(line) for line in f if line.strip()]

# Show the first 600 chars of the text we're trying to extract from
for r in results:
    model = r['model'].split('/')[-1]
    cond = r['condition']

    if cond in ('adversarial', 'structured'):
        parsed = parse_adversarial_trace(r['response'])
        text = parsed.get('phase3', '')
    else:
        text = r['response']

    dx = extract_leading_diagnosis(text)

    # Only show problem cases
    is_problem = (
        dx is None or
        len(dx) > 80 or
        any(phrase in (dx or '').lower() for phrase in [
            'anchoring', 'reasoning', 'differential', 'unlikely',
            'underlying', 'revision', 'however', 'therefore'
        ])
    )

    if is_problem:
        print(f"\n{'='*80}")
        print(f"PROBLEM: {model} | {cond} | extracted: {dx}")
        print(f"{'='*80}")
        # Print first 800 chars of the target text
        print(text[:800])
        print(f"\n{'- '*40}")