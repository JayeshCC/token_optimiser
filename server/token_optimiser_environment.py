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
import random
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

        # Hybrid LLM client — reads credentials from env vars at startup
        api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self._model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        if OpenAI and api_key:
            self._llm = OpenAI(base_url=api_base, api_key=api_key)
        else:
            self._llm = None

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
        # Select a random task
        self._current_task = random.choice(self._task_bank)
        self._state = TokenOptimiserState(
            episode_id=str(uuid4()),
            step_count=0,
            original_prompt=self._current_task["prompt"],
            task_difficulty=self._current_task["difficulty"],
            task_index=self._task_bank.index(self._current_task)
        )
        self._reset_count += 1

        logger.info(f"------ ENVIRONMENT RESET ------")
        logger.info(f"Loaded Task: [{self._current_task['difficulty'].upper()}] Index: {self._state.task_index}")
        logger.info(f"Requirements: Format='{self._current_task['expected_format']}', Max Tokens={self._current_task['max_output_tokens']}")
        logger.info(f"-------------------------------")

        return TokenOptimiserObservation(
            llm_response="",
            input_tokens=0,
            output_tokens=0,
            reward=0.0
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
        reward = max(0.0, min(1.0, reward))

        logger.info(f"[STEP {self._state.step_count}] Optimized Prompt Length: {len(optimized_prompt.split())} words")
        logger.info(f"  └─ Tokens => In: {int(input_tokens)}, Out: {int(output_tokens)}")
        logger.info(
            f"  └─ Reward => Tok_Eff:{token_efficiency:.2f} | Semantic:{semantic_score*0.3:.2f} | "
            f"Fmt:{format_score:.2f} | Penalty:{length_penalty:.2f} || TOTAL: {reward:.3f}"
        )

        return TokenOptimiserObservation(
            llm_response=llm_response,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reward=reward
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
        except Exception:
            return self._keyword_fallback_score(original_prompt, response)

    def _keyword_fallback_score(self, original_prompt: str, response: str) -> float:
        """Simple keyword overlap as semantic score when judge is unavailable."""
        key_concepts = {
            "machine", "learning", "ai", "data", "predict", "solar", "wind",
            "energy", "renewable", "sales", "customer", "product", "market",
            "budget", "analysis", "trend", "growth", "json", "bullet",
        }
        orig = set(original_prompt.lower().split()) & key_concepts
        resp = set(response.lower().split()) & key_concepts
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