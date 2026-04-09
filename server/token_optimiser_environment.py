# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# source tree.

"""
Prompt & Response Token Optimization Environment Implementation.

A sandboxed LLM interaction environment where an AI agent optimizes both input prompts
and expected output responses to minimize total token usage while maintaining correctness.
"""

import json
import logging
import os
import re
from uuid import uuid4

logger = logging.getLogger("TokenOptimiserBackend")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    # Add clear ANSI color prefixes for visibility in backend terminal
    formatter = logging.Formatter('\033[94m%(asctime)s\033[0m | \033[92m%(levelname)-7s\033[0m | \033[1m%(message)s\033[0m', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TokenOptimiserAction, TokenOptimiserObservation, TokenOptimiserState
except ImportError:
    from models import TokenOptimiserAction, TokenOptimiserObservation, TokenOptimiserState


class TokenOptimiserEnvironment(Environment):
    """
    Prompt & Response Token Optimization Environment.

    The agent receives a user prompt/task and must optimize it to reduce token usage
    while guiding the LLM to produce correct, properly formatted responses.
    """

    # Enable concurrent WebSocket sessions - REQUIRED for RL training
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the token optimization environment."""
        self._state = TokenOptimiserState(episode_id=str(uuid4()), step_count=0)
        self._task_bank = self._load_task_bank()
        self._current_task = None
        self._reset_count = 0
        self._last_prompt_norm = ""
        self._best_reward = 0.0
        self._stagnation_steps = 0

        # Hybrid LLM client — reads credentials from env vars at startup
        api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self._model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        if OpenAI and api_key:
            self._llm = OpenAI(base_url=api_base, api_key=api_key)
            logger.info(f"LLM backend enabled (model={self._model})")
        else:
            self._llm = None
            logger.warning("LLM backend unavailable; using deterministic fallback simulation and keyword judge.")

    def _load_task_bank(self):
        """Load the bank of prompt optimization tasks."""
        return [
            # EASY TASK
            {
                "difficulty": "easy",
                "prompt": "Could you possibly help me understand, if it's not too much trouble, what the word 'photosynthesis' means? I would really appreciate it if you could explain it to me in simple terms that are easy to understand.",
                "expected_format": "plain_brief",
                "reference_response": "Photosynthesis is how plants convert sunlight into food using CO2 and water.",
                "max_output_tokens": 30,
                "description": "Strip politeness filler and redundancy to a single direct question"
            },
            # MEDIUM TASK
            {
                "difficulty": "medium",
                "prompt": "I'm looking for information about the main differences between Python and JavaScript programming languages. Could you give me a thorough breakdown covering things like typing, use cases, performance, syntax style, and ecosystem so I can decide which one to learn first?",
                "expected_format": "bullet_5",
                "reference_response": "• Python: dynamic typing, data/ML focus\n• JS: dynamic typing, web/frontend focus\n• Performance: JS V8 faster for runtime\n• Syntax: Python readable, JS C-like\n• Ecosystem: Python pip/sci libs, JS npm/frameworks",
                "max_output_tokens": 120,
                "description": "Compress input AND inject format + count constraint into prompt"
            },
            # HARD TASK
            {
                "difficulty": "hard",
                "prompt": "We need you to analyze our e-commerce platform data and provide strategic insights. Specifically: first identify which product categories are performing best by revenue, second tell us which geographic regions show the most growth potential, third identify which customer segments respond best to promotions, fourth suggest how we should allocate our Q3 marketing budget across channels, and fifth flag any market risks we should be watching. Please be thorough in your analysis and provide detailed reasoning for each point.",
                "expected_format": "json_5keys",
                "reference_response": '{"top_categories":"...","growth_regions":"...","responsive_segments":"...","budget_allocation":"...","risks_watch":"..."}',
                "required_json_keys": ["top_categories", "growth_regions", "responsive_segments", "budget_allocation", "risks_watch"],
                "max_output_tokens": 200,
                "description": "Compress 82-word multi-intent prompt and force structured JSON output with 5 exact keys"
            }
        ]

    def reset(self) -> TokenOptimiserObservation:
        """
        Reset the environment with a random task from the task bank.

        Returns:
            TokenOptimiserObservation with initial state
        """
        # Cycle tasks in a fixed order so baseline runs are reproducible.
        task_index = self._reset_count % len(self._task_bank)
        self._current_task = self._task_bank[task_index]
        self._state = TokenOptimiserState(
            episode_id=str(uuid4()),
            step_count=0,
            original_prompt=self._current_task["prompt"],
            task_difficulty=self._current_task["difficulty"],
            task_index=self._task_bank.index(self._current_task)
        )
        self._reset_count += 1
        self._last_prompt_norm = ""
        self._best_reward = 0.0
        self._stagnation_steps = 0

        logger.info(f"------ ENVIRONMENT RESET ------")
        logger.info(f"Loaded Task: [{self._current_task['difficulty'].upper()}] Index: {self._state.task_index}")
        logger.info(f"Requirements: Format='{self._current_task['expected_format']}', Max Tokens={self._current_task['max_output_tokens']}")
        logger.info(f"-------------------------------")

        return TokenOptimiserObservation(
            llm_response="",
            input_tokens=0,
            output_tokens=0,
            reward=0.0,
            done=False,
            done_reason=""
        )

    def step(self, action: TokenOptimiserAction) -> TokenOptimiserObservation:  # type: ignore[override]
        """
        Execute a step in the environment by evaluating the agent's optimized prompt.

        Args:
            action: TokenOptimiserAction containing the optimized prompt

        Returns:
            TokenOptimiserObservation with LLM response simulation and reward
        """
        self._state.step_count += 1
        optimized_prompt = action.optimized_prompt
        prompt_norm = " ".join(optimized_prompt.strip().lower().split())
        prompt_changed = bool(prompt_norm) and prompt_norm != self._last_prompt_norm
        original_prompt = self._current_task["prompt"]

        # 1. Call real LLM (or fallback) to get the actual response + token counts
        llm_response, input_tokens, output_tokens = self._call_llm(optimized_prompt)

        # 2. LLM-as-judge: semantic quality score (0.0-1.0)
        semantic_score = self._judge_semantic_quality(original_prompt, llm_response)

        # 3. Token efficiency: how much did we reduce vs the original prompt token count?
        original_tokens = len(original_prompt.split()) * 1.3
        ref_output_tokens = len(self._current_task["reference_response"].split()) * 1.3
        ref_total = original_tokens + ref_output_tokens
        actual_total = input_tokens + output_tokens
        token_efficiency = max(0.0, min(0.4, (ref_total - actual_total) / max(ref_total, 1)))

        # 4. Format compliance (0.0-0.2)
        expected_fmt = self._current_task["expected_format"]
        format_score = 0.0
        if expected_fmt == "bullet_5":
            bullet_count = llm_response.count('•')
            if bullet_count >= 5:
                format_score = 0.2
            elif 3 <= bullet_count <= 4:
                format_score = 0.1
        elif expected_fmt == "json_5keys":
            try:
                parsed = json.loads(llm_response.strip())
                keys_present = sum(1 for k in self._current_task["required_json_keys"] if k in parsed)
                format_score = 0.04 * keys_present
            except json.JSONDecodeError:
                format_score = 0.0
        elif expected_fmt == "plain_brief":
            sentences = len([s for s in re.split(r'[.!?]+', llm_response) if s.strip()])
            has_no_bullets = not any(c in llm_response for c in ("•", "-", "*"))
            if sentences <= 2 and has_no_bullets:
                format_score = 0.2

        # 5. Length penalty if output way too long
        max_out = self._current_task["max_output_tokens"]
        length_penalty = -0.1 if output_tokens > max_out * 2 else 0.0

        # Final reward: weighted hybrid
        reward = (
            token_efficiency            # 0.0 - 0.4   (token saving)
            + semantic_score * 0.3      # 0.0 - 0.3   (LLM judge quality)
            + format_score              # 0.0 - 0.2   (format compliance)
            + length_penalty            # 0.0 or -0.1 (penalty)
        )
        # Penalize no-op actions so each step requires substantive work.
        if self._state.step_count > 1 and not prompt_changed:
            reward -= 0.10

        reward = max(0.0, min(1.0, reward))

        # Done logic: terminate on strong convergence or repeated non-improving/no-op steps.
        done = False
        done_reason = ""

        improved = reward > (self._best_reward + 0.01)
        if improved:
            self._best_reward = reward
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1

        if reward >= 0.90:
            done = True
            done_reason = "converged_high_reward"
        elif self._state.step_count >= 2 and self._stagnation_steps >= 2:
            done = True
            done_reason = "stagnated_no_improvement"
        elif self._state.step_count >= 2 and not prompt_changed:
            done = True
            done_reason = "no_substantive_action_change"

        self._last_prompt_norm = prompt_norm

        logger.info(f"[STEP {self._state.step_count}] Optimized Prompt Length: {len(optimized_prompt.split())} words")
        logger.info(f"  └─ Tokens => In: {int(input_tokens)}, Out: {int(output_tokens)}")
        logger.info(
            f"  └─ Reward => Tok_Eff:{token_efficiency:.2f} | Semantic:{semantic_score*0.3:.2f} | "
            f"Fmt:{format_score:.2f} | Penalty:{length_penalty:.2f} || TOTAL: {reward:.3f}"
        )
        logger.info(
            f"  └─ Progress => PromptChanged:{str(prompt_changed).lower()} | "
            f"Stagnation:{self._stagnation_steps} | Done:{str(done).lower()} | Reason:{done_reason or 'null'}"
        )

        return TokenOptimiserObservation(
            llm_response=llm_response,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reward=reward,
            done=done,
            done_reason=done_reason
        )

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """
        Call the real LLM with retries.
        """
        if self._llm is not None:
            import time
            for attempt in range(2):  # Try twice
                try:
                    resp = self._llm.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.3,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    in_tok = resp.usage.prompt_tokens if resp.usage else len(prompt.split())
                    out_tok = resp.usage.completion_tokens if resp.usage else len(text.split())
                    return text, in_tok, out_tok
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        logger.warning(f"Rate limited, waiting 3s (attempt {attempt+1})...")
                        time.sleep(3)
                    else:
                        logger.error(f"LLM call failed: {e}")
                        break

        # Rule-based fallback
        logger.debug("Falling back to rule-based simulation.")
        return self._fallback_simulate(prompt)

    def _fallback_simulate(self, prompt: str) -> tuple[str, int, int]:
        """Fast deterministic fallback when LLM is unavailable."""
        if self._current_task is None:
            text = "No task loaded."
            return text, len(prompt.split()), len(text.split())

        expected_format = self._current_task["expected_format"]
        original_words = len(self._current_task["prompt"].split())
        compression_ratio = len(prompt.split()) / max(original_words, 1)

        if "brief explanation" in expected_format:
            text = ("Machine learning is AI that learns from data to make predictions."
                    if compression_ratio <= 0.6
                    else "Machine learning enables systems to learn from experience and improve without explicit programming.")
        elif "bullet" in expected_format:
            text = ("• Solar power growing rapidly\n• Wind energy investments increasing\n• Hydroelectric power stable\n• Battery storage advancing\n• Global renewable adoption rising"
                    if compression_ratio <= 0.7
                    else "Renewable energy sectors are growing, led by solar and wind with strong policy support.")
        elif "json" in expected_format.lower():
            text = ('{"top_categories": ["electronics", "software"], "growth_regions": ["Asia", "Africa"], "responsive_segments": ["professionals"], "budget_allocation": {"email": 0.4, "social": 0.3}, "risks_watch": ["inflation"]}'  # noqa
                    if compression_ratio <= 0.6
                    else "Key categories: electronics, software. Growth in Asia and Africa.")
        else:
            text = "I understand your request and will provide a helpful response."

        in_tok = int(len(prompt.split()) * 1.3)
        out_tok = int(len(text.split()) * 1.3)
        return text, in_tok, out_tok

    def _judge_semantic_quality(self, original_prompt: str, response: str) -> float:
        """
        LLM-as-judge: score how well the response answers the original prompt.
        Returns a float 0.0-1.0.
        """
        if self._llm is None:
            logger.warning("Semantic judge fallback: no LLM client available.")
            return self._keyword_fallback_score(original_prompt, response)

        judge_prompt = (
            f"Rate 0 to 10 how well the RESPONSE answers the ORIGINAL question. "
            f"Consider accuracy and completeness. Reply with a single integer only.\n\n"
            f"ORIGINAL: {original_prompt[:300]}\n\nRESPONSE: {response[:400]}"
        )
        try:
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "5").strip()
            score = int("".join(c for c in raw if c.isdigit())[:2] or "5")
            return min(max(score / 10.0, 0.0), 1.0)
        except Exception as exc:
            logger.warning(f"Semantic judge fallback: judge call failed ({exc}).")
            return self._keyword_fallback_score(original_prompt, response)

    def _keyword_fallback_score(self, original_prompt: str, response: str) -> float:
        """Simple keyword overlap as semantic score when judge is unavailable."""
        key_concepts = {
            "machine", "learning", "ai", "data", "predict", "solar", "wind",
            "energy", "renewable", "sales", "customer", "product", "market",
            "budget", "analysis", "trend", "growth", "json", "bullet",
            "python", "javascript", "typing", "performance", "syntax", "ecosystem",
        }
        orig = set(original_prompt.lower().split()) & key_concepts
        resp = set(response.lower().split()) & key_concepts
        if not orig:
            logger.warning("Keyword fallback is using neutral score because no tracked concepts were found in original prompt.")
        raw = (len(resp) / len(orig)) if orig else 0.5
        return min(raw, 1.0)

    def _calculate_reward(self, original_prompt: str, optimized_prompt: str,
                         llm_response: str, expected_format: str, reference_response: str,
                         input_tokens: int, output_tokens: int) -> float:
        """
        Calculate multi-component reward score (0.0-1.0).
        """
        # Component 1: Token Efficiency (0.0-0.4)
        original_tokens = len(original_prompt.split()) * 1.3
        optimized_input_tokens = input_tokens
        output_tokens_estimate = output_tokens

        # Reference token counts for comparison
        ref_input_tokens = len(self._current_task["prompt"].split()) * 1.3
        ref_output_tokens = len(self._current_task["reference_response"].split()) * 1.3
        ref_total_tokens = ref_input_tokens + ref_output_tokens

        actual_total_tokens = optimized_input_tokens + output_tokens_estimate
        token_efficiency = max(0, (ref_total_tokens - actual_total_tokens) / ref_total_tokens)
        token_efficiency = min(token_efficiency, 0.4)  # Cap at 0.4

        # Component 2: Semantic Preservation (0.0-0.3)
        # Simple keyword-based similarity (in practice, would use embeddings)
        original_keywords = set(original_prompt.lower().split())
        response_keywords = set(llm_response.lower().split())

        # Extract key concepts from original prompt
        key_concepts = {"machine", "learning", "AI", "data", "predict", "solar", "wind",
                       "energy", "renewable", "sales", "customer", "product", "market",
                       "budget", "analysis", "trend", "growth", "json", "bullet", "point"}

        original_key_concepts = original_keywords & key_concepts
        response_key_concepts = response_keywords & key_concepts

        if len(original_key_concepts) > 0:
            semantic_similarity = len(response_key_concepts) / len(original_key_concepts)
        else:
            semantic_similarity = 0.5  # Neutral if no key concepts found

        semantic_score = min(semantic_similarity, 0.3)  # Cap at 0.3

        # Component 3: Format Compliance (0.0-0.2)
        format_score = 0.0
        if "bullet point" in expected_format.lower() and ("•" in llm_response or "*" in llm_response or "-" in llm_response):
            format_score = 0.2
        elif "json" in expected_format.lower() and ("{" in llm_response and "}" in llm_response):
            format_score = 0.2
        elif "brief explanation" in expected_format.lower() and len(llm_response.split()) < 30:
            format_score = 0.2

        # Component 4: Length Appropriateness (0.0-0.1)
        length_score = 0.0
        max_expected = self._current_task["max_output_tokens"]
        if output_tokens <= max_expected:
            length_score = 0.1
        elif output_tokens <= max_expected * 1.5:  # Partial credit
            length_score = 0.05

        # Component 5: Cost Simulation Bonus (0.0-0.05)
        # Reward for being under reference token count
        cost_bonus = 0.0
        if actual_total_tokens < ref_total_tokens:
            cost_bonus = min(0.05, (ref_total_tokens - actual_total_tokens) / ref_total_tokens * 0.05)

        # Component 6: Latency Penalty (penalty)
        latency_penalty = 0.0
        if output_tokens > max_expected * 2:
            latency_penalty = -0.1

        # Component 7: Context Window Penalty (penalty)
        context_penalty = 0.0
        # Simulate context window limit (e.g., 4096 tokens)
        if input_tokens > 3000:  # Assuming prompt + context
            context_penalty = -0.1

        # Calculate final reward
        total_reward = (
            token_efficiency +
            semantic_score +
            format_score +
            length_score +
            cost_bonus +
            latency_penalty +
            context_penalty
        )

        # Clamp to valid range
        return max(0.0, min(1.0, total_reward))

    @property
    def state(self) -> TokenOptimiserState:
        """
        Get the current environment state.

        Returns:
            Current TokenOptimiserState
        """
        return self._state

def grade(*args, **kwargs) -> float:
    """
    Entry point for OpenEnv offline task validation. 
    Returns a unified float. Core RL grading is dynamically calculated in TokenOptimiserEnvironment.step().
    """
    return _grade_redundancy_stripping(*args, **kwargs)


def grade_redundancy_stripping(*args, **kwargs) -> float:
    """Task-specific grader for redundancy_stripping."""
    return _grade_redundancy_stripping(*args, **kwargs)


def grade_constraint_injection(*args, **kwargs) -> float:
    """Task-specific grader for constraint_injection."""
    return _grade_constraint_injection(*args, **kwargs)


def grade_multi_key_json_extraction(*args, **kwargs) -> float:
    """Task-specific grader for multi_key_json_extraction."""
    return _grade_multi_key_json_extraction(*args, **kwargs)


def _extract_action_observation(*args, **kwargs) -> tuple[str, str, float, str]:
    """
    Extract optimized prompt, llm response, and base reward from flexible grader args.

    Supports object-style and dict-style inputs since validation harnesses can vary.
    """
    action = kwargs.get("action")
    observation = kwargs.get("observation") or kwargs.get("obs")

    if action is None and len(args) >= 1:
        action = args[0]
    if observation is None and len(args) >= 2:
        observation = args[1]

    optimized_prompt = ""
    llm_response = ""
    reward = 0.0
    done_reason = ""

    if isinstance(action, dict):
        optimized_prompt = str(action.get("optimized_prompt", ""))
    elif action is not None:
        optimized_prompt = str(getattr(action, "optimized_prompt", "") or "")

    if isinstance(observation, dict):
        llm_response = str(observation.get("llm_response", ""))
        reward = float(observation.get("reward", 0.0) or 0.0)
        done_reason = str(observation.get("done_reason", "") or "")
    elif observation is not None:
        llm_response = str(getattr(observation, "llm_response", "") or "")
        reward = float(getattr(observation, "reward", 0.0) or 0.0)
        done_reason = str(getattr(observation, "done_reason", "") or "")

    return optimized_prompt, llm_response, max(0.0, min(1.0, reward)), done_reason


def _grade_redundancy_stripping(*args, **kwargs) -> float:
    optimized_prompt, llm_response, base_reward, done_reason = _extract_action_observation(*args, **kwargs)

    # Reward concise rewrites that still produce plain, brief answers.
    prompt_tokens = len(optimized_prompt.split()) if optimized_prompt else 0
    concision_bonus = 0.2 if 1 <= prompt_tokens <= 30 else 0.0

    sentences = len([s for s in re.split(r"[.!?]+", llm_response) if s.strip()])
    plain_text_bonus = 0.2 if sentences <= 2 and not any(c in llm_response for c in ("•", "-", "*")) else 0.0

    noop_penalty = 0.15 if done_reason in ("no_substantive_action_change", "stagnated_no_improvement") else 0.0

    if optimized_prompt or llm_response:
        return max(0.0, min(1.0, base_reward * 0.6 + concision_bonus + plain_text_bonus - noop_penalty))
    return base_reward


def _grade_constraint_injection(*args, **kwargs) -> float:
    optimized_prompt, llm_response, base_reward, done_reason = _extract_action_observation(*args, **kwargs)

    # Enforce the medium-task structure: exactly 5 bullet points.
    bullet_count = llm_response.count("•")
    if bullet_count == 0:
        # Fallback for hyphen or asterisk bullets
        lines = [line.strip() for line in llm_response.splitlines() if line.strip()]
        bullet_count = sum(1 for line in lines if line.startswith("-") or line.startswith("*"))

    format_bonus = 0.25 if bullet_count == 5 else (0.1 if 3 <= bullet_count <= 4 else 0.0)
    brevity_bonus = 0.15 if len(optimized_prompt.split()) <= 45 and optimized_prompt else 0.0

    noop_penalty = 0.15 if done_reason in ("no_substantive_action_change", "stagnated_no_improvement") else 0.0

    if optimized_prompt or llm_response:
        return max(0.0, min(1.0, base_reward * 0.6 + format_bonus + brevity_bonus - noop_penalty))
    return base_reward


def _grade_multi_key_json_extraction(*args, **kwargs) -> float:
    optimized_prompt, llm_response, base_reward, done_reason = _extract_action_observation(*args, **kwargs)

    required_keys = {
        "top_categories",
        "growth_regions",
        "responsive_segments",
        "budget_allocation",
        "risks_watch",
    }

    key_bonus = 0.0
    parse_bonus = 0.0
    if llm_response:
        try:
            parsed = json.loads(llm_response.strip())
            if isinstance(parsed, dict):
                present = len(required_keys & set(parsed.keys()))
                key_bonus = 0.3 * (present / len(required_keys))
                parse_bonus = 0.15
        except Exception:
            key_bonus = 0.0

    compression_bonus = 0.15 if len(optimized_prompt.split()) <= 65 and optimized_prompt else 0.0

    noop_penalty = 0.15 if done_reason in ("no_substantive_action_change", "stagnated_no_improvement") else 0.0

    if optimized_prompt or llm_response:
        return max(0.0, min(1.0, base_reward * 0.4 + parse_bonus + key_bonus + compression_bonus - noop_penalty))
    return base_reward