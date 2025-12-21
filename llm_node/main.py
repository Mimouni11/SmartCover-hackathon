"""
Main LLM node for insurance claim structuring.
Uses OpenRouter API with Claude Sonnet.

LangGraph Integration:
    graph.add_node("structure_claim", structure_claim)
    graph.add_edge("structure_claim", "validate")
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .templates import (
    TEMPLATES,
    get_required_fields,
    get_all_fields,
    is_valid_insurance_type
)
from .prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_followup_prompt
)
from .config import (
    get_api_key,
    get_api_url,
    get_model_config,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    LOG_PROMPTS,
    LOG_RESPONSES,
    MIN_CONFIDENCE_THRESHOLD
)


def call_claude(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Optional[Dict]:
    """
    Call OpenRouter API with Claude Sonnet.

    Args:
        prompt: User prompt
        system_prompt: System instructions

    Returns:
        Parsed JSON response or None on failure
    """
    url = get_api_url()
    api_key = get_api_key()
    config = get_model_config()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://oneclaim.local",  # Required by OpenRouter
        "X-Title": "OneClaim Insurance Processor"
    }

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }

    if LOG_PROMPTS:
        print("\n=== PROMPT SENT TO CLAUDE ===")
        print(f"System: {system_prompt[:300]}...")
        print(f"User: {prompt[:500]}...")
        print("=" * 50)

    # Try with retries
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()

            # Extract content from OpenRouter response
            raw_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            if LOG_RESPONSES:
                print("\n=== RAW CLAUDE RESPONSE ===")
                print(raw_response)
                print("=" * 50)

            # Parse JSON from response
            parsed = parse_json_response(raw_response)
            return parsed

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES - 1:
                continue
            return None

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text[:500]}")
            return None

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    return None


def parse_json_response(raw_response: str) -> Optional[Dict]:
    """
    Parse JSON from LLM response.
    Handles cases where LLM adds markdown code blocks or extra text.
    """
    import re

    # Remove markdown code blocks if present
    response = raw_response.strip()

    # Remove ```json and ``` if present
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # Try to find JSON object by looking for { and }
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_str = response[start:end + 1]

        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"  Attempted to parse: {json_str[:500]}...")
            return None

    # Try direct parse as fallback
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    print(f"Failed to parse JSON from response")
    print(f"  Raw response: {response[:500]}...")
    return None


def resolve_relative_dates(claim_data: Dict) -> Dict:
    """
    Convert relative dates ("hier", "yesterday") to absolute dates.
    """
    today = datetime.now()

    # Common date mappings
    date_map = {
        "aujourd'hui": today,
        "today": today,
        "hier": today - timedelta(days=1),
        "yesterday": today - timedelta(days=1),
        "avant-hier": today - timedelta(days=2),
        "ce matin": today,
        "this morning": today,
    }

    def resolve_date_field(value):
        if isinstance(value, str):
            lower_val = value.lower()
            for key, date in date_map.items():
                if key in lower_val:
                    return date.strftime("%Y-%m-%d")
        return value

    # Recursively process dict
    result = {}
    for key, value in claim_data.items():
        if isinstance(value, dict):
            result[key] = resolve_relative_dates(value)
        elif "_date" in key or key == "date":
            result[key] = resolve_date_field(value)
        else:
            result[key] = value

    return result


def validate_and_clean_response(llm_response: Dict) -> Dict:
    """
    Validate LLM response structure and clean up data.
    """
    # Ensure required keys exist
    if "insurance_types" not in llm_response:
        llm_response["insurance_types"] = []

    if "structured_claims" not in llm_response:
        llm_response["structured_claims"] = {}

    if "missing_required_fields" not in llm_response:
        llm_response["missing_required_fields"] = {}

    if "confidence" not in llm_response:
        llm_response["confidence"] = 0.5

    if "needs_followup" not in llm_response:
        llm_response["needs_followup"] = True

    # Resolve relative dates
    llm_response["structured_claims"] = resolve_relative_dates(
        llm_response["structured_claims"]
    )

    # Validate insurance types
    valid_types = [
        t for t in llm_response["insurance_types"]
        if is_valid_insurance_type(t)
    ]
    llm_response["insurance_types"] = valid_types

    return llm_response


def structure_claim(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Extract structured claim data from user input.

    This is the FIRST node in the pipeline.

    Input state keys (required):
        - user_input: str - Natural language description of incident
        - user_id: str - User identifier

    Input state keys (optional):
        - is_followup: bool - Is this a followup to get missing info?
        - previous_claim_data: dict - Previous structured data (for followup)
        - missing_fields: list - Fields we're asking about (for followup)

    Output state keys (added):
        - insurance_types: list[str] - Detected insurance type(s)
        - structured_claims: dict - Extracted fields per insurance type
        - missing_required_fields: dict - Missing required fields per type
        - confidence: float - Confidence score (0.0-1.0)
        - needs_followup: bool - Whether we need to ask for more info
        - followup_question: str - Question to ask user (if needs_followup)
        - extraction_status: str - "success" | "failed" | "low_confidence"
        - error_message: str - Error details if extraction failed

    LangGraph usage:
        graph.add_node("structure_claim", structure_claim)
    """
    # Extract input from state
    user_input = state.get("user_input", "")
    user_id = state.get("user_id", "unknown")
    is_followup = state.get("is_followup", False)

    # Validate input
    if not user_input or not user_input.strip():
        return {
            **state,
            "extraction_status": "failed",
            "error_message": "Empty user input",
            "needs_followup": True,
            "followup_question": "Pouvez-vous décrire ce qui s'est passé?"
        }

    # Build prompt
    if is_followup:
        previous_data = state.get("previous_claim_data", {})
        missing = state.get("missing_fields", [])
        prompt = build_followup_prompt(user_input, previous_data, missing)
    else:
        prompt = build_user_prompt(user_input)

    # Call LLM
    print(f"\n Processing claim for user: {user_id}")
    print(f" Input: {user_input[:100]}...")

    llm_response = call_claude(prompt)

    if llm_response is None:
        return {
            **state,
            "extraction_status": "failed",
            "error_message": "LLM call failed",
            "needs_followup": True,
            "followup_question": "Sorry, an error occurred. Please try again."
        }

    # Validate and clean response
    llm_response = validate_and_clean_response(llm_response)

    # Check confidence
    confidence = llm_response.get("confidence", 0.5)
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        extraction_status = "low_confidence"
    else:
        extraction_status = "success"

    # Build output state
    output_state = {
        **state,  # Keep all existing state
        "insurance_types": llm_response.get("insurance_types", []),
        "structured_claims": llm_response.get("structured_claims", {}),
        "missing_required_fields": llm_response.get("missing_required_fields", {}),
        "confidence": confidence,
        "needs_followup": llm_response.get("needs_followup", True),
        "followup_question": llm_response.get("followup_question", ""),
        "extraction_status": extraction_status,
        "timestamp": datetime.now().isoformat(),
    }

    # Log results
    print(f" Extraction: {extraction_status}")
    print(f"  Insurance types: {output_state['insurance_types']}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Needs followup: {output_state['needs_followup']}")

    return output_state


# Convenience function for testing without LangGraph
def test_structure_claim(user_input: str, user_id: str = "test_user") -> Dict:
    """
    Test the structure_claim function without LangGraph.
    """
    state = {
        "user_input": user_input,
        "user_id": user_id,
        "is_followup": False,
    }

    result = structure_claim(state)
    return result


if __name__ == "__main__":
    # Quick test
    print("=== Testing structure_claim with Claude ===\n")

    test_input = "Hier j'ai eu un accident sur l'autoroute. Un camion m'a percuté par derrière."

    result = test_structure_claim(test_input)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
