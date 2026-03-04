import asyncio
from backend.openrouter import query_model
from backend.config import COUNCIL_MODELS

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one sentence."}
]

async def main():
    model = COUNCIL_MODELS[0]
    print(f"Testing {model}...")
    result = await query_model(model, messages)
    print(f"Response type: {type(result)}")
    print(f"Keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
    print(f"Content: {result['content'][:200]}")
    print(f"Tokens: {result.get('input_tokens')}/{result.get('output_tokens')}")

asyncio.run(main())
