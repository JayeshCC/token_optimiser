---
title: Token Optimiser Environment
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🔤 Prompt & Response Token Optimization Environment

> An OpenEnv-compatible RL environment that trains agents to minimize LLM API token usage while preserving semantic quality — reducing AI inference costs at scale.

---

## 📌 Introduction

Large Language Model APIs charge per token. Verbose prompts and unconstrained responses waste tokens and money. This environment trains an AI agent to **rewrite verbose prompts into concise, efficient versions** that:

- Use fewer input tokens
- Guide the LLM toward shorter, correctly-formatted responses
- Preserve the full semantic meaning and intent of the original request
- Respect output format constraints (free text, bullet points, JSON)

The agent learns real-world prompt engineering — a critical skill for production LLM systems where cost efficiency matters at scale.

---

## 🏗️ Architecture

```
token_optimiser/
├── inference.py                        # Hackathon evaluation script
├── models.py                           # Pydantic data models (Action / Observation / State)
├── client.py                           # Async WebSocket EnvClient
├── openenv.yaml                        # OpenEnv deployment config
├── Dockerfile                          # Root-level multi-stage build
├── pyproject.toml                      # Package config & dependencies
├── uv.lock                             # Locked dependencies
└── server/
    ├── app.py                          # FastAPI server (WebSocket + HTTP)
    ├── token_optimiser_environment.py  # Core RL environment logic
    └── requirements.txt                # Server dependencies
```

---

## 🧠 How It Works

### Action Space
The agent submits a **`TokenOptimiserAction`** containing its optimized version of the original prompt:

```python
class TokenOptimiserAction(Action):
    optimized_prompt: str   # Agent's rewritten, token-efficient prompt
```

### Observation Space
After each step, the agent receives a **`TokenOptimiserObservation`**:

```python
class TokenOptimiserObservation(Observation):
    llm_response: str    # Actual LLM response to the optimized prompt
    input_tokens: int    # Token count of the optimized prompt
    output_tokens: int   # Token count of the LLM response
    reward: float        # Step reward (0.0 – 1.0)
```

### State Space
```python
class TokenOptimiserState(State):
    original_prompt: str      # The verbose task prompt the agent must optimize
    task_difficulty: str      # "easy" | "medium" | "hard"
    task_index: int           # Index in task bank
```

---

## 📋 Tasks

### 🟢 Easy — Redundancy Stripping
**Original:** `"Could you possibly help me understand, if it's not too much trouble, what the word 'photosynthesis' means? I would really appreciate it if you could explain it to me in simple terms that are easy to understand."`  
**Goal:** Strip politeness filler and redundancy to a single direct question without formatting  
**Expected optimized:** `"What does photosynthesis mean? Be brief."`  
**Max output:** 30 tokens

### 🟡 Medium — Constraint Injection
**Original:** `"I'm looking for information about the main differences between Python and JavaScript programming languages. Could you give me a thorough breakdown covering things like typing, use cases, performance, syntax style, and ecosystem so I can decide which one to learn first?"`  
**Goal:** Compress input AND inject format + exactly 5 bullet point counts into prompt  
**Expected optimized:** `"Compare Python and JavaScript (typing, use cases, performance, syntax, ecosystem) in exactly 5 bullet points."`  
**Max output:** 120 tokens

### 🔴 Hard — Multi-Key JSON Extraction
**Original:** `"We need you to analyze our e-commerce platform data and provide strategic insights. Specifically: first identify which product categories are performing best by revenue, second tell us which geographic regions show the most growth potential, third identify which customer segments respond best to promotions, fourth suggest how we should allocate our Q3 marketing budget across channels, and fifth flag any market risks we should be watching. Please be thorough in your analysis and provide detailed reasoning for each point."`  
**Goal:** Compress 82-word multi-intent prompt and force structured JSON output with 5 exact required keys  
**Expected optimized:** `"Analyze e-commerce data based on revenue, growth regions, responsive segments, Q3 budget, and market risks. Output strictly as JSON with keys: top_categories, growth_regions, responsive_segments, budget_allocation, risks_watch."`  
**Max output:** 200 tokens

---

## 🏆 Reward Function

**Hybrid grading** — combines token efficiency + LLM-as-judge semantic scoring:

| Component | Weight | Description |
|-----------|--------|-------------|
| Token Efficiency | 0.0 – 0.40 | Tokens saved vs. reference (input + output combined) |
| Semantic Quality | 0.0 – 0.30 | LLM judge rates response quality 0–10 |
| Format Compliance | 0.0 – 0.20 | Response matches required format (bullets / JSON / brief) |
| Length Penalty | −0.10 | Output exceeds 2× max token budget |

```
reward = token_efficiency + (semantic_score × 0.3) + format_score + length_penalty
reward = clamp(reward, 0.0, 1.0)
```

| Score | Meaning |
|-------|---------|
| 0.0 | Meaning lost, system failure, or no optimization |
| 0.3 – 0.5 | Token reduction achieved but quality degraded |
| 0.6 – 0.8 | Good balance of compression and quality |
| 0.9 – 1.0 | Optimal: minimal tokens, correct format, meaning preserved |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- A Hugging Face account with a token that has **Inference Providers** permission

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd token_optimiser

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Set Environment Variables

```bash
# Required
export HF_TOKEN="hf_your_token_here"

# Optional (these are the defaults)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export SERVER_URL="http://localhost:8000"
```

On Windows (PowerShell):
```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

> **HF Token Permissions:** Your token must have **"Make calls to Inference Providers"** enabled.  
> Create/edit at → https://huggingface.co/settings/tokens

---

## 🚀 Running Locally

### Step 1 — Start the Environment Server

```bash
# Terminal 1
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2 — Run the Inference Script

```bash
# Terminal 2
uv run inference.py
```

Expected output:
```
[START] task=token_optimization env=token_optimiser model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='Analyze renewable energy trends...' reward=0.62 done=false error=null
[STEP] step=2 action='Summarize 2013-2023 renewable energy...' reward=0.81 done=false error=null
[STEP] step=3 action='Renewable energy trends 2013-2023...' reward=0.84 done=false error=null
[STEP] step=4 action='Renewable energy 2013-2023 trends...' reward=0.82 done=false error=null
[STEP] step=5 action='Energy trends 2013-2023: solar, wind...' reward=0.81 done=true error=null
[END] success=true steps=5 score=0.780 rewards=0.62,0.81,0.84,0.82,0.81
```

### Step 3 — (Optional) Run via Docker

```bash
# Build
docker build -t token-optimiser-env .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=$API_BASE_URL \
  token-optimiser-env
```

---

## 🐍 Using the Python Client

```python
import asyncio
from token_optimiser import TokenOptimiserEnv, TokenOptimiserAction

async def main():
    async with TokenOptimiserEnv(base_url="http://localhost:8000") as env:
        # Reset — get the task
        result = await env.reset()
        state = await env.state()
        print(f"Task: {state.original_prompt}")
        print(f"Difficulty: {state.task_difficulty}")

        # Agent submits an optimized prompt
        result = await env.step(
            TokenOptimiserAction(optimized_prompt="Explain machine learning briefly.")
        )
        print(f"LLM Response: {result.observation.llm_response}")
        print(f"Reward: {result.reward}")
        print(f"Tokens — in: {result.observation.input_tokens}, out: {result.observation.output_tokens}")

asyncio.run(main())
```

---

## ☁️ Deployment

Deploy to Hugging Face Spaces using the OpenEnv CLI:

```bash
openenv push
```

---

## 🔧 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face API key with Inference Providers permission |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model used for responses and judging |
| `SERVER_URL` | `http://localhost:8000` | Environment server URL (for inference.py) |
| `LOCAL_IMAGE_NAME` | *(optional)* | Docker image name — auto-spins container if set |

---

## 📦 Dependencies

- [`openenv-core`](https://github.com/meta-pytorch/OpenEnv) ≥ 0.2.2 — RL environment framework
- `openai` — LLM API client (routed through HF)
- `fastapi` + `uvicorn` — Environment server
- `pydantic` v2 — Data model validation

See [`pyproject.toml`](./pyproject.toml) for the full pinned dependency list.

---

## 📄 License

BSD License — see source files for details.