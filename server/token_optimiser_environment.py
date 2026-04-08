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

import random
from uuid import uuid4

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

    def _load_task_bank(self):
        """Load the bank of prompt optimization tasks."""
        return [
            # EASY TASK
            {
                "difficulty": "easy",
                "prompt": "Can you please explain in a very detailed manner what machine learning is and how it works step by step?",
                "expected_format": "brief explanation",
                "reference_response": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming. It works by identifying patterns in training data to make predictions or decisions on new data.",
                "max_output_tokens": 50,
                "description": "Reduce verbosity while preserving core concept"
            },
            # MEDIUM TASK
            {
                "difficulty": "medium",
                "prompt": "I need a comprehensive analysis of the renewable energy market trends over the past decade, including solar, wind, and hydroelectric power growth rates, investment patterns, technological advancements, and policy impacts across different regions globally.",
                "expected_format": "5 bullet points summarizing key trends",
                "reference_response": "• Solar power capacity grew 22% annually avg.  • Wind energy investments reached $140B in 2020  • Hydroelectric remains largest renewable source  • Battery storage tech advancing rapidly  • Policy incentives driving global adoption",
                "max_output_tokens": 100,
                "description": "Compress input + specify bullet point format + length limit"
            },
            # HARD TASK
            {
                "difficulty": "hard",
                "prompt": "As a senior data scientist, I need you to analyze our Q3 sales performance dataset and provide actionable insights. The dataset contains: customer demographics, purchase history, product categories, regional sales data, marketing campaign ROI, seasonal trends, and competitor analysis. Please identify: 1) Our top 3 performing product categories and why, 2) Geographic regions with highest growth potential, 3) Customer segments most responsive to our email campaigns, 4) Optimal marketing budget allocation for Q4, and 5) Risks to watch based on economic indicators.",
                "expected_format": "JSON with 5 keys: top_categories, growth_regions, responsive_segments, budget_allocation, risks_watch",
                "reference_response": '{"top_categories": ["electronics", "software", "home_goods"], "growth_regions": ["SE Asia", "Latin America", "Africa"], "responsive_segments": ["young_professionals", "tech_enthusiasts"], "budget_allocation": {"email": 0.3, "social": 0.25, "search": 0.2, "tv": 0.15, "other": 0.1}, "risks_watch": ["inflation", "supply_chain", "labor_shortage"]}',
                "max_output_tokens": 200,
                "description": "Multi-intent optimization: compress complex request + specify JSON format + accuracy + length constraints"
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

        # Simulate LLM response based on optimized prompt
        simulated_response = self._simulate_llm_call(optimized_prompt)

        # Calculate token usage (simplified estimation)
        input_tokens = len(optimized_prompt.split()) * 1.3  # Rough token estimation
        output_tokens = len(simulated_response.split()) * 1.3

        # Calculate reward based on multiple factors
        reward = self._calculate_reward(
            original_prompt=self._current_task["prompt"],
            optimized_prompt=optimized_prompt,
            llm_response=simulated_response,
            expected_format=self._current_task["expected_format"],
            reference_response=self._current_task["reference_response"],
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens)
        )

        return TokenOptimiserObservation(
            llm_response=simulated_response,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reward=reward
        )

    def _simulate_llm_call(self, prompt: str) -> str:
        """
        Simulate an LLM response to the optimized prompt.
        In a real implementation, this would call an actual LLM API.
        """
        # Simple simulation based on prompt content
        if "machine learning" in prompt.lower() and ("brief" in prompt.lower() or "short" in prompt.lower()):
            return "Machine learning is AI that learns from data to make predictions."
        elif "bullet point" in prompt.lower():
            return "• Solar power growing rapidly\n• Wind energy investments increasing\n• Hydroelectric power stable\n• Battery storage advancing\n• Global renewable adoption rising"
        elif "json" in prompt.lower() and ("key" in prompt.lower() or ":" in prompt.lower()):
            return '{"top_categories": ["electronics", "software"], "growth_regions": ["Asia", "Africa"], "responsive_segments": ["professionals"], "budget_allocation": {"email": 0.4, "social": 0.3, "search": 0.2, "tv": 0.1}, "risks_watch": ["inflation", "supply_chain"]}'
        else:
            # Default response - would be improved with better prompting
            return "I understand your request and will provide a helpful response based on the information given."

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