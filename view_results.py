import json

baselines = {}
phase3s = {}

with open('data/experiment/results_incremental.jsonl') as f:
    for line in f:
        r = json.loads(line)
        model = r['model']
        tier = r.get('tier', '')

        if r['condition'] == 'baseline':
            baselines[model] = {'tier': tier, 'text': r['response']}

        elif r['condition'] == 'adversarial':
            text = r['response']
            p3_start = text.lower().find('phase 3')
            if p3_start > -1:
                phase3s[model] = {'tier': tier, 'text': text[p3_start:]}
            else:
                phase3s[model] = {'tier': tier, 'text': '[Phase 3 not found]'}

print("=" * 60)
print("BASELINES")
print("=" * 60)
for model in baselines:
    print(f"\n{'='*60}")
    print(f"MODEL: {model} | TIER: {baselines[model]['tier']}")
    print(f"{'='*60}")
    print(baselines[model]['text'])

print("\n\n")
print("=" * 60)
print("PHASE 3 (ADVERSARIAL)")
print("=" * 60)
for model in phase3s:
    print(f"\n{'='*60}")
    print(f"MODEL: {model} | TIER: {phase3s[model]['tier']}")
    print(f"{'='*60}")
    print(phase3s[model]['text'])
