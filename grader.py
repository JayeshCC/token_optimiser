"""
Grader functions for the Token Optimiser environment.

These are the entry points referenced in openenv.yaml under each task's
`grader.function` field. Each function accepts an action and observation
(or dicts thereof) and returns a float score in [0.0, 1.0].

The hackathon validator calls these functions by importing this module
and invoking the named function for each task.
"""

import sys
import os

# Ensure the project root is on sys.path so `server` is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.token_optimiser_environment import (  # noqa: E402
    grade_redundancy_stripping as _grade_redundancy_stripping,
    grade_constraint_injection as _grade_constraint_injection,
    grade_multi_key_json_extraction as _grade_multi_key_json_extraction,
)


def grade_redundancy_stripping(*args, **kwargs) -> float:
    """
    Grader for the easy 'redundancy_stripping' task.

    Rewards concise rewrites that still produce plain, brief answers.
    Returns a score in [0.0, 1.0].
    """
    return _grade_redundancy_stripping(*args, **kwargs)


def grade_constraint_injection(*args, **kwargs) -> float:
    """
    Grader for the medium 'constraint_injection' task.

    Enforces the medium-task structure: exactly 5 bullet points.
    Returns a score in [0.0, 1.0].
    """
    return _grade_constraint_injection(*args, **kwargs)


def grade_multi_key_json_extraction(*args, **kwargs) -> float:
    """
    Grader for the hard 'multi_key_json_extraction' task.

    Checks for all 5 required JSON keys and compression quality.
    Returns a score in [0.0, 1.0].
    """
    return _grade_multi_key_json_extraction(*args, **kwargs)
