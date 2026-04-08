# Prompt & Response Token Optimization Environment

## Environment Description

A sandboxed LLM interaction environment where an AI agent is responsible for optimizing both input prompts and expected output responses to minimize total token usage. The environment simulates real-world API usage (e.g., LLM calls), including cost constraints, context limits, and response length control, ensuring efficient and scalable AI interactions.

## Motivation

Large Language Model (LLM) API usage can be expensive at scale, with costs directly proportional to token consumption. This environment trains agents to reduce AI inference costs by intelligently optimizing prompts while maintaining response quality - a valuable skill for production LLM applications where cost efficiency is paramount.

## Action and Observation Spaces

### Action Space
- **Type**: `TokenOptimiserAction`
- **Field**: `optimized_prompt: str`
- **Description**: The agent's optimized version of the user prompt, designed to:
  - Reduce unnecessary input tokens
  - Guide the LLM toward shorter, more precise responses
  - Specify output format constraints (bullets, JSON, etc.)
  - Include context/cost awareness instructions

### Observation Space
- **Type**: `TokenOptimiserObservation`
- **Fields**:
  - `llm_response: str` - The LLM's response to the optimized prompt
  - `input_tokens: int` - Token count of the optimized prompt
  - `output_tokens: int` - Token count of the LLM response
  - `reward: float` - Reward score for this step (0.0-1.0)

### State Space
- **Type**: `TokenOptimiserState`
- **Fields**:
  - `episode_id: str` - Unique identifier for the current episode
  - `step_count: int` - Number of steps taken in current episode
  - `original_prompt: str` - The initial user prompt/task
  - `task_difficulty: str` - Current difficulty level (easy/medium/hard)
  - `task_index: int` - Index of current task in task bank

## Tasks

The environment includes 3 progressively difficult tasks:

### Easy Task
- **Prompt**: "Can you please explain in a very detailed manner what machine learning is and how it works step by step?"
- **Goal**: Reduce input tokens + guide model to shorter, precise response
- **Example Optimization**: 
  - Input: "Can you please explain in a very detailed manner what machine learning is and how it works step by step?"
  - Optimized: "Explain machine learning briefly."
  - Expected Output: Concise definition under 50 tokens

### Medium Task
- **Prompt**: "I need a comprehensive analysis of the renewable energy market trends over the past decade, including solar, wind, and hydroelectric power growth rates, investment patterns, technological advancements, and policy impacts across different regions globally."
- **Goal**: Compress input + add output constraints + ensure structured response
- **Example Optimization**: 
  - Add: "Limit response to 5 bullet points"
  - Remove: Redundant temporal/geographic qualifiers
  - Preserve: Core request for trend analysis
  - Expected Output: 5 bullet points under 100 tokens

### Hard Task
- **Prompt**: "As a senior data scientist, I need you to analyze our Q3 sales performance dataset and provide actionable insights. The dataset contains: customer demographics, purchase history, product categories, regional sales data, marketing campaign ROI, seasonal trends, and competitor analysis. Please identify: 1) Our top 3 performing product categories and why, 2) Geographic regions with highest growth potential, 3) Customer segments most responsive to our email campaigns, 4) Optimal marketing budget allocation for Q4, and 5) Risks to watch based on economic indicators."
- **Goal**: Minimize total tokens (input+output) while maintaining correctness + adapting to context limits
- **Example Optimization**: 
  - Specify: JSON format with exact keys
  - Remove: Excessive background/methodology details
  - Preserve: All 5 requested analytical components
  - Expected Output: JSON with 5 specific keys under 200 tokens

## Reward Function

The reward function provides rich, multi-component feedback:

### Score Ranges
- **0.0**: Meaning lost, incorrect output, or system failure
- **0.3-0.5**: Token reduction achieved but output quality degraded
- **0.6-0.8**: Balanced optimization (good reduction + acceptable output)
- **0.9-1.0**: Optimal solution:
  - Minimal tokens (input + output)
  - Full semantic preservation
  - Correct and structured response

### Components
1. **Token Efficiency (0.0-0.4)**: Reduction in combined input+output tokens vs. baseline
2. **Semantic Preservation (0.0-0.3)**: Meaning retention vs. original intent
3. **Format Compliance (0.0-0.2)**: Adherence to required structure (bullets, JSON, etc.)
4. **Length Appropriateness (0.0-0.1): Response within expected length bounds**
5. **Cost Simulation Bonus (0.0-0.05)**: Reward for token efficiency
6. **Latency Penalty**: For excessively long outputs
7. **Context Penalty**: For exceeding simulated window limits

## Setup and Usage

### Prerequisites
- Python 3.8+
- openenv-core package
- Hugging Face API token (HF_TOKEN)
- Environment variables set:
  - `API_BASE_URL` (default: https://router.huggingface.co/v1)
  - `MODEL_NAME` (default: Qwen/Qwen2.5-72B-Instruct)
  - `HF_TOKEN` (your Hugging Face API key)

### Installation
```bash
# Install the environment in development mode
pip install -e .

# Or with UV (recommended)
uv pip install -e .
```

### Local Development
```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Or use the provided script
python -m server.app
```

### Docker Deployment
```bash
# Build the Docker image
docker build -t token_optimiser:latest .

# Run the container
docker run -p 8000:8000 token_optimiser:latest
```

### Hugging Face Spaces Deployment
```bash
# Push to Hugging Face Spaces (requires HF CLI login)
openenv push --repo-id your-username/token-optimiser-env
```

### Using the EnvClient
```python
from client import TokenOptimiserEnv

# Connect to local server
with TokenOptimiserEnv(base_url="http://localhost:8000") as client:
    obs = client.reset()
    print(f"Original prompt: {obs.original_prompt}")
    
    # Agent optimizes the prompt
    action = TokenOptimiserAction(optimized_prompt="Explain machine learning briefly")
    result = client.step(action)
    
    print(f"LLM Response: {result.observation.llm_response}")
    print(f"Reward: {result.reward}")
    print(f"Tokens - Input: {result.observation.input_tokens}, Output: {result.observation.output_tokens}")

# Or use Docker deployment
client = TokenOptimiserEnv.from_docker_image("token_optimiser-env:latest")
try:
    # ... interaction logic ...
finally:
    client.close()
```

## Baseline Inference Script

The environment includes a baseline inference script (`inference.py`) that:
- Uses OpenAI client routed through Hugging Face API
- Implements a basic prompt optimization strategy
- Emits structured logs in the required [START]/[STEP]/[END] format
- Runs against all task difficulties
- Produces reproducible baseline scores

Run the baseline:
```bash
python inference.py
```

Expected output format:
```
[START] task=token_optimisation env=token_optimiser model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=Explain machine learning briefly reward=0.75 done=false error=null
[END] success=true steps=1 score=0.75 rewards=0.75
```

## Validation

Validate OpenEnv specification compliance:
```bash
openenv validate
```

Expected output:
```
✓ OpenEnv specification validation passed
```

## Requirements

- Python 3.8+
- openenv-core >= 0.2.0
- huggingface_hub >= 0.20.0
- openai >= 2.7.2
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pydantic >= 2.0.0

See `pyproject.toml` for full dependency list.

## License

This environment is released under the MIT License.