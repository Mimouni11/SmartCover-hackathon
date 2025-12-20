"""
Configuration for Ollama and Mistral LLM.
Adjust these settings based on your hardware and needs.
"""

import os

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or "mistral:7b", "mixtral:8x7b"

# LLM parameters
LLM_CONFIG = {
    "model": OLLAMA_MODEL,
    "temperature": 0.1,  # Low for structured extraction (more deterministic)
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 2048,  # Max tokens to generate
    "stop": None,  # No stop sequences needed
}

# Timeout settings
REQUEST_TIMEOUT = 60  # seconds - increase if using large models
MAX_RETRIES = 3  # Retry on failure

# Date handling
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_TIME_FORMAT = "%H:%M"

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5  # Below this, ask for clarification
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Above this, proceed without followup

# Validation
STRICT_MODE = True  # Validate against templates strictly
ALLOW_UNKNOWN_FIELDS = False  # Reject fields not in templates

# Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_PROMPTS = DEBUG_MODE  # Log prompts to console for debugging
LOG_RESPONSES = DEBUG_MODE  # Log raw LLM responses


def get_ollama_url() -> str:
    """Get the full Ollama API URL."""
    return f"{OLLAMA_BASE_URL}/api/generate"


def get_model_config() -> dict:
    """Get the LLM configuration dict."""
    return LLM_CONFIG.copy()


# Hardware optimization hints
HARDWARE_PROFILES = {
    "cpu_only": {
        "model": "mistral:7b",
        "num_predict": 1024,
        "temperature": 0.1,
    },
    "gpu_available": {
        "model": "mixtral:8x7b",
        "num_predict": 2048,
        "temperature": 0.1,
    },
    "fast_demo": {
        "model": "mistral:7b",
        "num_predict": 512,
        "temperature": 0.0,  # Most deterministic
    }
}


def set_hardware_profile(profile: str):
    """
    Set hardware-optimized config.

    Args:
        profile: "cpu_only" | "gpu_available" | "fast_demo"
    """
    global LLM_CONFIG
    if profile in HARDWARE_PROFILES:
        LLM_CONFIG.update(HARDWARE_PROFILES[profile])
        print(f"✓ Hardware profile set to: {profile}")
        print(f"  Model: {LLM_CONFIG['model']}")
    else:
        print(f"✗ Unknown profile: {profile}")
        print(f"  Available: {list(HARDWARE_PROFILES.keys())}")


# Quick check function
def check_ollama_available() -> bool:
    """
    Check if Ollama is running and accessible.
    Returns True if available, False otherwise.
    """
    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    # Quick test when running config.py directly
    print("=== OneClaim LLM Config ===")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Temperature: {LLM_CONFIG['temperature']}")
    print(f"Max tokens: {LLM_CONFIG['num_predict']}")
    print(f"\nChecking Ollama availability...")

    if check_ollama_available():
        print("✓ Ollama is running!")
    else:
        print("✗ Ollama not accessible")
        print(f"  Make sure Ollama is running on {OLLAMA_BASE_URL}")
        print(f"  Run: ollama serve")