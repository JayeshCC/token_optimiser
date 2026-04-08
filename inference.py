"""Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image() method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# Import the environment - this will need to be adjusted based on the actual environment name
# For now, we'll comment this out and use a placeholder since we don't have the actual env yet
# from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# Configuration from environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # For docker image usage

TASK_NAME = os.getenv("TOKEN_OPTIMISER_TASK", "token_optimization")
BENCHMARK = os.getenv("TOKEN_OPTIMISER_BENCHMARK", "token_optimiser")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a helpful AI assistant tasked with optimizing prompts and responses
    to minimize token usage while preserving semantic meaning and correctness.
    Your goal is to provide concise, accurate responses that fulfill the user's intent
    using as few tokens as possible.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# Placeholder functions - these would be replaced with actual environment interaction
def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"

async def main() -> None:
    # Check if HF_TOKEN is available
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable not set")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # For now, we'll simulate the environment interaction since we don't have the actual env class
    # In a real implementation, this would use: env = await MyEnvV4Env.from_docker_image(LOCAL_IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Simulate environment reset
        last_echoed = "Initial state"
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            # In real implementation, we would check if result.done from env.reset()
            # For simulation, we'll just run the steps

            message = get_model_message(client, step, last_echoed, last_reward, history)

            # Simulate environment step - in reality this would be:
            # result = await env.step(MyEnvV4Action(message=message))
            # For now, we'll simulate a simple response

            # Simulate observation, reward, done status
            obs = type('Observation', (), {'echoed_message': message})()
            reward = len(message) * 0.01  # Simple reward based on length
            done = step >= MAX_STEPS  # Done after max steps
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # In real implementation: await env.close()
        pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())