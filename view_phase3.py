import json

with open('data/experiment/results_incremental.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['condition'] == 'adversarial':
            text = r['response']
            p3_start = text.lower().find('phase 3')
            if p3_start > -1:
                phase3 = text[p3_start:]
            else:
                phase3 = "[Phase 3 not found]"

            print(f"{'='*60}")
            print(f"MODEL: {r['model']}")
            print(f"PROVIDER: {r.get('provider','')}")
            print(f"TIER: {r.get('tier','')}")
            print(f"{'='*60}")
            print(phase3)
            print()
