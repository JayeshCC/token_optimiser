# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# source tree.

"""Token Optimiser Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TokenOptimiserAction, TokenOptimiserObservation, TokenOptimiserState


class TokenOptimiserEnv(
    EnvClient[TokenOptimiserAction, TokenOptimiserObservation, TokenOptimiserState]
):
    """
    Client for the Token Optimiser Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TokenOptimiserEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.llm_response)
        ...
        ...     result = client.step(TokenOptimiserAction(optimized_prompt="Explain ML briefly"))
        ...     print(result.observation.llm_response)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TokenOptimiserEnv.from_docker_image("token_optimiser-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TokenOptimiserAction(optimized_prompt="Test prompt"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TokenOptimiserAction) -> Dict:
        """
        Convert TokenOptimiserAction to JSON payload for step message.

        Args:
            action: TokenOptimiserAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "optimized_prompt": action.optimized_prompt,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TokenOptimiserObservation]:
        """
        Parse server response into StepResult[TokenOptimiserObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TokenOptimiserObservation
        """
        obs_data = payload.get("observation", {})
        observation = TokenOptimiserObservation(
            llm_response=obs_data.get("llm_response", ""),
            input_tokens=obs_data.get("input_tokens", 0),
            output_tokens=obs_data.get("output_tokens", 0),
            reward=obs_data.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TokenOptimiserState:
        """
        Parse server response into TokenOptimiserState object.

        Args:
            payload: JSON response from state request

        Returns:
            TokenOptimiserState object with episode information
        """
        return TokenOptimiserState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            original_prompt=payload.get("original_prompt", ""),
            task_difficulty=payload.get("task_difficulty", "easy"),
            task_index=payload.get("task_index", 0)
        )