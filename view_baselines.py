import json

with open('data/experiment/results_incremental.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['condition'] == 'baseline':
            text = r['response']
            # Find the leading diagnosis section
            print(f"{'='*60}")
            print(f"MODEL: {r['model']}")
            print(f"TIER: {r.get('tier','')}")
            print(f"{'='*60}")
            print(text[:1500])
            print()
