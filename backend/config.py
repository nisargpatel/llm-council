"""Configuration for LLM Council — Medical Adversarial Self-Critique Study"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── COUNCIL MODELS (test subjects) ──
COUNCIL_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "google/gemini-3-pro-preview",
]

# ── CHAIRMAN (evaluator only — never a test subject) ──
CHAIRMAN_MODEL = "anthropic/claude-opus-4.6"

# ── MODEL METADATA (for analysis) ──
MODEL_METADATA = {
    "openai/gpt-5.4": {
        "provider": "openai",
        "tier": "frontier",
        "family": "gpt-5",
        "capability_rank": 5,       # 1=lowest in study, 5=highest
        "pair_with": "openai/gpt-5.2",
    },
    "openai/gpt-5.2": {
        "provider": "openai",
        "tier": "standard",
        "family": "gpt-5",
        "capability_rank": 2,
        "pair_with": "openai/gpt-5.4",
    },
    "anthropic/claude-sonnet-4.5": {
        "provider": "anthropic",
        "tier": "standard",
        "family": "claude-4",
        "capability_rank": 1,
        "pair_with": "anthropic/claude-opus-4.5",
    },
    "anthropic/claude-opus-4.5": {
        "provider": "anthropic",
        "tier": "frontier",
        "family": "claude-4",
        "capability_rank": 4,
        "pair_with": "anthropic/claude-sonnet-4.5",
    },
    "google/gemini-3-pro-preview": {
        "provider": "google",
        "tier": "frontier",
        "family": "gemini-3",
        "capability_rank": 3,
        "pair_with": None,          # no within-provider pair
    },
}

# Within-provider pairs for paired analysis
PROVIDER_PAIRS = [
    {
        "provider": "OpenAI",
        "standard": "openai/gpt-5.2",
        "frontier": "openai/gpt-5.4",
    },
    {
        "provider": "Anthropic",
        "standard": "anthropic/claude-sonnet-4.5",
        "frontier": "anthropic/claude-opus-4.5",
    },
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = "data/conversations"
