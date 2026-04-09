"""
Inference Script — Token Optimiser Environment
================================================
STDOUT FORMAT (mandatory):
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables required:
  HF_TOKEN       — Hugging Face API key
  API_BASE_URL   — LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — Model id       (default: Qwen/Qwen2.5-72B-Instruct)
  SERVER_URL     — Running env server (default: http://localhost:8000)
  LOCAL_IMAGE_NAME — Docker image name (optional; spins up container if set)
"""

import asyncio
import logging
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
try:
    from huggingface_hub import HfFolder
except Exception:  # pragma: no cover
    HfFolder = None

from token_optimiser import TokenOptimiserEnv, TokenOptimiserAction

logger = logging.getLogger("TokenOptimiserFrontend")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('\033[96m%(asctime)s\033[0m | \033[93m%(levelname)-7s\033[0m | \033[1mCLIENT\033[0m | %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL: str = os.getenv("SERVER_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
TASK_NAME: str = "token_optimization"
BENCHMARK: str = "token_optimiser"
MAX_STEPS: int = 5
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 200
SUCCESS_THRESHOLD: float = 0.6


def _resolve_hf_token() -> Optional[str]:
    """
    Resolve API token in this order:
    1) HF_TOKEN env var
    2) API_KEY env var
    3) huggingface-cli cached login token
    """
    token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if token:
        return token

    if HfFolder is not None:
        try:
            return HfFolder.get_token()
        except Exception:
            return None
    return None


HF_TOKEN: Optional[str] = _resolve_hf_token()

SYSTEM_PROMPT = textwrap.dedent("""
    You are a prompt optimization expert. Rewrite the given prompt to:
    1. Use the fewest possible tokens (concise language, no filler words)
    2. Preserve full semantic meaning and intent
    3. Add explicit output-format constraints (e.g., "in 5 bullet points", "as JSON with keys: …")
    4. Guide the responder toward a shorter, precise answer

    Reply with ONLY the optimized prompt — no explanations, no prefixes, no quotes.
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Truncate action for readability but keep it on one line
    action_short = action.replace("\n", " ")[:120]
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_short!r} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _build_user_message(original_prompt: str, step: int,
                         prev_reward: float, prev_response: str,
                         history: List[str]) -> str:
    if step == 1:
        return (
            f"Optimize this prompt to minimize tokens while preserving all meaning:\n\n"
            f"{original_prompt}"
        )
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(f"""
        Original prompt:
        {original_prompt}

        Your last optimized version got reward: {prev_reward:.2f}
        LLM responded with: {prev_response!r}

        Recent history:
        {history_block}

        Improve your optimization further. Reply with ONLY the new optimized prompt.
    """).strip()


def get_optimized_prompt(
    llm: OpenAI,
    original_prompt: str,
    step: int,
    prev_reward: float,
    prev_response: str,
    history: List[str],
) -> str:
    user_msg = _build_user_message(original_prompt, step, prev_reward, prev_response, history)
    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        result = (completion.choices[0].message.content or "").strip()
        return result if result else "Explain briefly."
    except Exception as exc:
        logger.warning(f"LLM call failed, falling back to basic rule compression: {exc}")
        return _rule_based_compress(original_prompt, step)


# Rule-based fallback compressor (used when LLM is unavailable)
_FILLER = {
    "please", "kindly", "could", "you", "can", "i", "need", "want", "would",
    "like", "very", "really", "just", "actually", "basically", "specifically",
    "a", "an", "the", "in", "of", "to", "and", "that", "is", "are", "be",
    "will", "should", "must", "have", "has", "do", "does", "for", "with",
    "as", "at", "by", "on", "or", "but", "it", "its", "this",
}
_BREVITY = [
    "",                                    # step 1 — just strip fillers
    " Be brief.",                           # step 2
    " Limit response to 3 sentences.",      # step 3
    " Reply in one sentence.",              # step 4+
]


def _rule_based_compress(original_prompt: str, step: int = 1) -> str:
    """Strip filler words and add a conciseness constraint."""
    words = original_prompt.split()
    compressed = [
        w for w in words
        if w.lower().rstrip(".,?!") not in _FILLER
    ]
    suffix = _BREVITY[min(step - 1, len(_BREVITY) - 1)]
    result = " ".join(compressed) + suffix
    return result if result.strip() else original_prompt


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def run_episode(llm: OpenAI) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Connect to environment
    if LOCAL_IMAGE_NAME:
        env = await TokenOptimiserEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = TokenOptimiserEnv(base_url=SERVER_URL)
        await env.connect()

    try:
        # Reset — get initial observation
        reset_result = await env.reset()

        # Fetch original prompt from server state
        env_state = await env.state()
        original_prompt: str = env_state.original_prompt or "Explain machine learning briefly."

        logger.info(f"Task connected. Difficulty: {env_state.task_difficulty.upper()}")
        logger.info(f"Original prompt ({len(original_prompt.split())} words): {original_prompt[:80]}...")

        prev_reward = 0.0
        prev_response = ""
        history: List[str] = []

        for step in range(1, MAX_STEPS + 1):
            # Ask LLM to optimize the prompt
            optimized = get_optimized_prompt(
                llm, original_prompt, step, prev_reward, prev_response, history
            )

            # Step the environment with the optimized prompt
            error_msg: Optional[str] = None
            reward = 0.0
            done = False
            try:
                result = await env.step(TokenOptimiserAction(optimized_prompt=optimized))
                obs = result.observation
                reward = result.reward          # server puts reward at top-level, not inside obs
                done = result.done or (step >= MAX_STEPS)
                prev_response = obs.llm_response
                logger.info(f"Step {step} Tokens => Input: {obs.input_tokens}, Output: {obs.output_tokens}")
            except Exception as exc:
                error_msg = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step
            prev_reward = reward
            history.append(f"step={step} prompt={optimized!r:.60} reward={reward:.2f}")

            log_step(step=step, action=optimized, reward=reward, done=done, error=error_msg)

            if done:
                break
                
            # Wait between steps to avoid rate limiting
            await asyncio.sleep(2.5)

        # Score = average reward across steps, clamped to [0, 1]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        logger.error(f"Episode error aborted run: {exc}")
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable not set. Exiting.")
        return

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    await run_episode(llm)


if __name__ == "__main__":
    asyncio.run(main())
