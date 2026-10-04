# SmartCOVER — AI Insurance Claim Assistant

> Hackathon Dauphine 2025

SmartCOVER is a conversational assistant that lets customers declare an insurance claim in plain language (**English, French or Arabic**). It turns the conversation into a structured claim, estimates its cost, scores its fraud risk and computes an acceptance probability. The results are then sent to an insurer dashboard for review.

> **Main branch:** active development and the most up-to-date code live on **`feature/llm`**. Please use this branch as the main branch (clone, pull requests, demos). The name comes from a naming mistake early in the hackathon, and we kept it because of time constraints.

---

## Features

- **Multilingual chat intake (EN / FR / AR):** customers describe their incident in their own words.
- **Smart routing:** a fast keyword router detects whether a message is a claim and its type (`auto`, `home`, `health`, `travel`).
- **LLM field extraction:** an LLM (Claude 3.5 Sonnet via OpenRouter) extracts structured claim fields and asks follow-up questions when required information is missing.
- **Cost estimation:** an ML regression model for health claims, plus rule-based estimates for the other claim types.
- **Fraud detection:** a hybrid agent that combines an ML classifier, an anomaly detector and business rules (night-time incident, new customer, no witnesses, unusually large amount).
- **Acceptance probability:** derived from the fraud score and shown to the customer.
- **Insurer dashboard API:** every completed claim is stored and exposed so insurers can review and approve or reject it.

## Architecture

The pipeline is a **LangGraph** state machine with conversation memory, so each chat thread keeps its own context:

```
START ──뿯▽ fast_router ──뿯▽ extract_fields ──뿯▽ run_cost_model ──뿯▽ run_fraud_model ──뿯▽ run_acceptance_model ──뿯▽ finalize_claim ──뿯▽ END
              │                  │
              │                  └──뿯▽ ask_followup ──뿯▽ END   (missing info 뿯↽ ask the user, resume on the next message)
              │
              └──뿯▽ end_chat ──뿯▽ END                          (not a claim)
```

```
┌──────────────────┐   HTTP    ┌──────────────────┐   invoke   ┌──────────────────────────┐
│ Streamlit chat   │ ────────뿯▽ │ FastAPI (api.py) │ ─────────뿯▽ │ LangGraph pipeline       │
│ (chatbot.py)     │ 뿯▽──────── │                  │ 뿯▽───────── │ (graph.py + nodes/)      │
└──────────────────┘           └────────┬─────────┘            └──────────────────────────┘
                                        │ claims_storage.json
                                        뿯▽
                               Insurer dashboard (GET /claims)
```

## Project Structure

```
.
├── api.py                  # FastAPI backend: /chat, /claims, claim status updates
├── chatbot.py              # Streamlit customer chat UI
├── graph.py                # LangGraph pipeline definition + interactive console
├── nodes/                  # LangGraph nodes
│   ├── state.py            #   shared ClaimState definition
│   ├── router_node.py      #   keyword-based claim / type detection
│   ├── extraction_node.py  #   bridge to the LLM extraction module
│   └── models_node.py      #   cost, fraud and acceptance model nodes
├── llm_node/               # LLM extraction module (OpenRouter)
│   ├── config.py           #   API and model configuration
│   ├── prompts.py          #   extraction prompts
│   ├── templates.py        #   claim templates per insurance type
│   └── schemas.py          #   structured output schemas
├── fraud/                  # Fraud detection agent
│   ├── agent.py            #   ML + anomaly + rules scoring
│   ├── rules.py            #   business rules
│   ├── features.py         #   feature list
│   └── train.py            #   model training script
├── models/                 # Pre-trained models (.pkl)
├── user_profiles.json      # Simulated customer profile database
├── claims_storage.json     # Stored claims for the insurer dashboard
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
git clone https://github.com/Mimouni11/Hackathon_dauphine_2025.git
cd Hackathon_dauphine_2025
git checkout feature/llm

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the project root (it is git-ignored):

```env
OPENROUTER_API_KEY=your_openrouter_api_key
DEBUG_MODE=true
```

> Never commit your API key.

## Usage

### 1. Start the backend API

```bash
uvicorn api:app --host 0.0.0.0 --port 5000 --reload
```

### 2. Start the customer chat UI

```bash
streamlit run chatbot.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501) and describe your claim, for example:

> *"J'ai eu un accident de voiture hier 뿯½ 14h, quelqu'un m'a percut뿯½ par l'arri뿯½re."*

### Alternative: console mode

To test the pipeline directly in the terminal:

```bash
python graph.py
```

Type `new` to start a new conversation or `quit` to exit.

## API Reference

| Method | Endpoint                                  | Description                                                    |
|--------|-------------------------------------------|----------------------------------------------------------------|
| POST   | `/chat`                                   | Send a user message. Body: `{ user_id, message, thread_id }`   |
| GET    | `/claims`                                 | List all submitted claims (insurer dashboard)                  |
| POST   | `/claims/{claim_id}/update?status=<s>`    | Update a claim status (e.g. `approved`, `rejected`)            |

Example:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "message": "My car was hit in a parking lot", "thread_id": "demo-1"}'
```

Customers only see their acceptance probability. Insurers get the full analysis: estimated cost, fraud score, risk level, fraud signals and the recommended decision.

## Tech Stack

- **Orchestration:** LangGraph, LangChain Core
- **LLM:** Claude 3.5 Sonnet via OpenRouter
- **ML:** scikit-learn, XGBoost, NumPy, pandas, joblib
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit

## Branches

| Branch                    | Purpose                                         |
|---------------------------|-------------------------------------------------|
| **`feature/llm`**         | **Main branch** (kept despite its name). Full integrated pipeline |
| `feature/fraud_detection` | Fraud detection model development               |
| `feature/structure`       | Initial project structure                       |
| `main`                    | Initial repository setup                        |
