"""
Configuration for OpenRouter API with Claude Sonnet.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# OpenRouter settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model settings
MODEL_NAME = "anthropic/claude-3.5-sonnet"  # or "anthropic/claude-3-haiku" for faster/cheaper

# LLM parameters
LLM_CONFIG = {
    "model": MODEL_NAME,
    "temperature": 0.1,  # Low for structured extraction
    "max_tokens": 2048,
}

# Timeout settings
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

# Date handling
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_TIME_FORMAT = "%H:%M"

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
LOG_PROMPTS = DEBUG_MODE
LOG_RESPONSES = DEBUG_MODE


def get_api_key() -> str:
    """Get the OpenRouter API key."""
    return OPENROUTER_API_KEY


def get_api_url() -> str:
    """Get the OpenRouter API URL."""
    return OPENROUTER_BASE_URL


def get_model_config() -> dict:
    """Get the LLM configuration dict."""
    return LLM_CONFIG.copy()


def check_api_available() -> bool:
    """Check if OpenRouter API is accessible."""
    import requests
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print("=== OneClaim LLM Config ===")
    print(f"API: OpenRouter")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {LLM_CONFIG['temperature']}")
    print(f"Max tokens: {LLM_CONFIG['max_tokens']}")
    print(f"\nChecking API availability...")

    if check_api_available():
        print("OpenRouter API is accessible!")
    else:
        print("OpenRouter API not accessible - check your API key")
