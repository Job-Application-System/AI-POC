"""Custom verl reward function for resume GRPO training.

The collective training pipeline can use cover_letter_reward.py for both tasks
because it branches on ground_truth["task_type"]. This module exists as a
resume-specific entrypoint for single-task runs or external launch configs.
"""

from __future__ import annotations

import json

from cover_letter_reward import compute_score as compute_application_score


def compute_score(solution_str: str, ground_truth: str, **kwargs: object) -> dict[str, float]:
    """Score a generated tailored resume against the job application context."""
    try:
        truth = json.loads(ground_truth)
    except json.JSONDecodeError:
        truth = {}

    truth["task_type"] = "resume"
    return compute_application_score(
        solution_str=solution_str,
        ground_truth=json.dumps(truth, ensure_ascii=False),
        **kwargs,
    )
