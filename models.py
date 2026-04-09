# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# source tree.

"""
Data models for the Prompt & Response Token Optimization Environment.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class TokenOptimiserAction(Action):
    """Action for the Token Optimiser environment - the optimized prompt."""

    optimized_prompt: str = Field(..., description="The optimized prompt sent to the LLM")


class TokenOptimiserObservation(Observation):
    """Observation from the Token Optimiser environment - LLM response and token metrics."""

    llm_response: str = Field(default="", description="The response from the LLM")
    input_tokens: int = Field(default=0, description="Number of tokens in the optimized prompt")
    output_tokens: int = Field(default=0, description="Number of tokens in the LLM response")
    reward: float = Field(default=0.0, description="Reward score for this step (0.0-1.0)")
    done_reason: str = Field(default="", description="Why the episode terminated (if done=true)")


class TokenOptimiserState(State):
    """State for the Token Optimiser environment."""

    original_prompt: str = Field(default="", description="The original user prompt/task")
    task_difficulty: str = Field(default="easy", description="Current task difficulty: easy/medium/hard")
    task_index: int = Field(default=0, description="Index of current task in task bank")