"""
Rerun missing and empty results from the experiment.
Uses retry logic with exponential backoff and longer sleeps for Gemini.
"""
import asyncio
import json
from backend.experiment import run_single_case, CONDITIONS
from backend.config import COUNCIL_MODELS

with open('data/cases.json') as f:
    cases = {c['case_id']: c for c in json.load(f)}

# Find empty or missing results
needs_rerun = set()
seen = set()
with open('data/experiment/results_incremental.jsonl') as f:
    for line in f:
        r = json.loads(line)
        key = (r['case_id'], r['model'], r['condition'])
        seen.add(key)
        if not r.get('response'):
            needs_rerun.add(key)

# Also find completely missing
for cid in cases:
    for model in COUNCIL_MODELS:
        for cond in CONDITIONS:
            if (cid, model, cond) not in seen:
                needs_rerun.add((cid, model, cond))

tasks = [(cases[cid], model, cond) for cid, model, cond in needs_rerun if cid in cases]
# Sort: non-Gemini first, Gemini last (so rate limit doesn't slow everything)
tasks.sort(key=lambda x: (1 if 'gemini' in x[1] else 0, x[0]['case_id']))

print(f'{len(tasks)} results to re-run')
gemini_count = sum(1 for t in tasks if 'gemini' in t[1])
other_count = len(tasks) - gemini_count
print(f'  {other_count} non-Gemini, {gemini_count} Gemini\n')

async def rerun():
    succeeded = 0
    failed = 0
    for i, (case, model, cond) in enumerate(tasks):
        is_gemini = 'gemini' in model
        for attempt in range(3):
            try:
                result = await run_single_case(case, model, cond)
                if result.get('response'):
                    with open('data/experiment/results_incremental.jsonl', 'a') as f:
                        f.write(json.dumps(result) + '\n')
                    succeeded += 1
                    print(f'[{i+1}/{len(tasks)}] OK    {model:<35} | {cond:<12} | {case["case_id"]}')
                    break
                else:
                    print(f'  Attempt {attempt+1}/3 empty, retrying in {2 ** (attempt+1)}s...')
                    await asyncio.sleep(2 ** (attempt + 1))
            except Exception as e:
                wait = 2 ** (attempt + 1) * (3 if is_gemini else 1)
                print(f'  Attempt {attempt+1}/3 failed: {str(e)[:80]}')
                print(f'  Waiting {wait}s before retry...')
                await asyncio.sleep(wait)
        else:
            failed += 1
            print(f'[{i+1}/{len(tasks)}] FAIL  {model:<35} | {cond:<12} | {case["case_id"]}')

        # Sleep between calls
        await asyncio.sleep(5.0 if is_gemini else 1.0)

    print(f'\nRerun complete: {succeeded} succeeded, {failed} failed out of {len(tasks)}')

asyncio.run(rerun())