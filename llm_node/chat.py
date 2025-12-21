"""
Interactive Chat with OneClaim LLM Node

Run this to have a conversation with the LLM.
Type your claim description and keep adding more info as it asks followup questions.

Usage:
    python chat.py
"""

from .main import structure_claim
import json


def print_banner():
    print("\n" + "="*80)
    print("  OneClaim - Interactive Chat")
    print("  Type your insurance claim in French or English")
    print("  Type 'quit' or 'exit' to stop")
    print("="*80 + "\n")


def print_result(result):
    """Print the LLM's response nicely."""
    print("\n" + "-"*80)
    print("🤖 LLM Response:")
    print("-"*80)

    # Check if extraction failed
    if result.get('extraction_status') == 'failed':
        print(f"\n❌ Extraction failed: {result.get('error_message', 'Unknown error')}")
        if result.get('followup_question'):
            print(f"❓ {result['followup_question']}")
        print("-"*80 + "\n")
        return

    # Show detected insurance type
    types = result.get('insurance_types', [])
    if types:
        print(f"\n📋 Insurance Type: {', '.join(types)}")
    else:
        print("\n⚠️ No insurance type detected")

    # Show confidence
    confidence = result.get('confidence', 0)
    print(f"💯 Confidence: {confidence:.0%}")

    # Show extracted data
    claims = result.get('structured_claims', {})
    if claims:
        print("\n✅ Extracted Information:")
        for ins_type, data in claims.items():
            print(f"\n  [{ins_type.upper()}]")
            for key, value in data.items():
                if value is not None and value != "":
                    print(f"    • {key}: {value}")
    else:
        print("\n⚠️ No information extracted yet")

    # Show missing fields
    missing = result.get('missing_required_fields', {})
    has_missing = any(fields for fields in missing.values())
    if has_missing:
        print("\n⚠️  Missing Information:")
        for ins_type, fields in missing.items():
            if fields:
                print(f"  [{ins_type.upper()}]: {', '.join(fields)}")

    # Show followup question
    if result.get('needs_followup') and result.get('followup_question'):
        print(f"\n❓ {result['followup_question']}")
    elif not result.get('needs_followup'):
        print("\n✅ All required information collected!")

    print("-"*80 + "\n")


def chat():
    """Main chat loop."""
    print_banner()

    user_id = "chat_user"
    conversation_history = []
    state = None

    while True:
        # Get user input
        user_input = input("👤 You: ").strip()

        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break

        if not user_input:
            continue

        # Build state
        if state is None or state.get('extraction_status') == 'failed':
            # First message or retry after failure
            state = {
                "user_input": user_input,
                "user_id": user_id,
                "is_followup": False,
            }
        else:
            # Followup message
            state = {
                "user_input": user_input,
                "user_id": user_id,
                "is_followup": True,
                "previous_claim_data": state.get('structured_claims', {}),
                "missing_fields": state.get('missing_required_fields', {}),
            }

        # Call LLM
        print("\n⏳ Processing...")
        try:
            result = structure_claim(state)
        except Exception as e:
            print(f"\n❌ Error calling LLM: {e}")
            continue

        # Update state for next turn
        state = result

        # Add to conversation history
        conversation_history.append({
            "user": user_input,
            "result": result
        })

        # Print result
        print_result(result)

        # If extraction failed, continue to get new input
        if result.get('extraction_status') == 'failed':
            continue

        # If no followup needed, ask if they want to submit or continue
        if not result.get('needs_followup'):
            choice = input("✅ Claim ready! Type 'submit' to finish, or continue adding info: ").strip().lower()
            if choice == 'submit':
                print("\n📤 Claim submitted successfully!")
                print("\n📊 Final Claim Data:")
                print(json.dumps(result.get('structured_claims', {}), indent=2, ensure_ascii=False))
                print("\n")

                # Ask if they want to start a new claim
                restart = input("Start a new claim? (y/n): ").strip().lower()
                if restart == 'y':
                    state = None
                    conversation_history = []
                    print("\n" + "="*80 + "\n")
                else:
                    break


if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()