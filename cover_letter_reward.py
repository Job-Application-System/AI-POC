"""Custom verl reward function for cover-letter GRPO training."""

from __future__ import annotations

import json
import re
from collections import Counter

from nltk.corpus import stopwords

from SimilarityModel import SimilarityModel


COMPONENT_KEYS = (
    "similarity_from_occurrences",
    "similarity_from_num_of_words",
    "similarity_ratio",
    "similarity_by_character",
    "similarity_by_commonality",
    "similarity_by_cosine",
)
DEFAULT_COMPONENT_WEIGHTS = {
    "similarity_from_occurrences": 1.0,
    "similarity_from_num_of_words": 1.0,
    "similarity_ratio": 1.0,
    "similarity_by_character": 0.25,
    "similarity_by_commonality": 1.0,
    "similarity_by_cosine": 1.0,
}
DEFAULT_REWARD_BLEND_WEIGHTS = {
    "weighted_similarity": 0.75,
    "length_reward": 0.10,
    "format_reward": 0.10,
    "candidate_match": 0.05,
}
FINAL_SCORE_WEIGHT = 1.0


def _english_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords corpus is missing. Install it once with: "
            "python -m nltk.downloader stopwords"
        ) from exc


def _tokens(text: str) -> list[str]:
    ignored = _english_stopwords()
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower()) if token not in ignored]


def _overlap(source: str, target: str) -> float:
    source_counts = Counter(_tokens(source))
    target_counts = Counter(_tokens(target))
    if not source_counts or not target_counts:
        return 0.0
    common = source_counts & target_counts
    return sum(common.values()) / max(1, sum(target_counts.values()))


def _length_reward(solution: str) -> float:
    words = solution.split()
    if 250 <= len(words) <= 450:
        return 1.0
    if 180 <= len(words) < 250 or 450 < len(words) <= 550:
        return 0.7
    return 0.3


def _format_reward(solution: str) -> float:
    lower = solution.lower()
    banned_prefixes = ("as an ai", "here is", "certainly", "cover letter:")
    if any(phrase in lower[:120] for phrase in banned_prefixes):
        return 0.0
    return 1.0


def _similarity_scores(job_description: str, resume: str, cover_letter: str) -> dict[str, float]:
    model = SimilarityModel(job_description, resume, cover_letter)
    scores = model.componentScores()
    scores["final_score"] = float(model.evaluateFinalScore())
    return {key: max(0.0, min(1.0, float(value))) for key, value in scores.items()}


def _weighted_similarity(scores: dict[str, float], weights: dict[str, float]) -> float:
    weighted_total = FINAL_SCORE_WEIGHT * scores.get("final_score", 0.0)
    total_weight = FINAL_SCORE_WEIGHT
    for key in COMPONENT_KEYS:
        weight = float(weights.get(key, DEFAULT_COMPONENT_WEIGHTS[key]))
        weighted_total += weight * scores.get(key, 0.0)
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_total / total_weight


def _reward_blend_weights(weights: object) -> dict[str, float]:
    if not isinstance(weights, dict):
        return DEFAULT_REWARD_BLEND_WEIGHTS

    blend = {
        key: max(0.0, float(weights.get(key, default)))
        for key, default in DEFAULT_REWARD_BLEND_WEIGHTS.items()
    }
    total = sum(blend.values())
    if total <= 0:
        return DEFAULT_REWARD_BLEND_WEIGHTS
    return {key: value / total for key, value in blend.items()}


def compute_score(solution_str: str, ground_truth: str, **_: object) -> dict[str, float]:
    """Return a scalar RL reward plus component rewards for an application artifact.

    The main signal is SimilarityModel.evaluateFinalScore(), and the individual
    SimilarityModel component scores are exposed as extra reward metrics. The
    Similarity component coefficients are read from
    ground_truth["reward_weights"] so the training pipeline can tune them with
    cross validation. The outer reward blend defaults to fixed coefficients, but
    can be overridden with ground_truth["reward_blend_weights"].

    For cover-letter rows, the rollout is the cover letter and the resume comes
    from ground truth. For resume rows, the rollout is the tailored resume and
    the cover letter comes from ground truth when available.
    """
    try:
        truth = json.loads(ground_truth)
    except json.JSONDecodeError:
        truth = {}

    job_description = truth.get("job_description", "")
    task_type = truth.get("task_type", "cover_letter")
    base_resume = truth.get("resume", "")
    paired_cover_letter = truth.get("cover_letter", "") or truth.get("best_cover_letter", "")
    best_cover_letter = truth.get("best_cover_letter", "")
    best_tailored_resume = truth.get("best_tailored_resume", "")
    weights = truth.get("reward_weights") or DEFAULT_COMPONENT_WEIGHTS
    blend_weights = _reward_blend_weights(truth.get("reward_blend_weights"))

    if task_type == "resume":
        resume = solution_str
        cover_letter = paired_cover_letter
        reference_text = best_tailored_resume
    else:
        resume = best_tailored_resume or base_resume
        cover_letter = solution_str
        reference_text = best_cover_letter

    similarity_scores = _similarity_scores(job_description, resume, cover_letter)
    weighted_similarity = _weighted_similarity(similarity_scores, weights)
    length_reward = _length_reward(solution_str)
    format_reward = _format_reward(solution_str)
    candidate_match = _overlap(solution_str, reference_text) if reference_text else 0.0
    score = (
        blend_weights["weighted_similarity"] * weighted_similarity
        + blend_weights["length_reward"] * length_reward
        + blend_weights["format_reward"] * format_reward
        + blend_weights["candidate_match"] * candidate_match
    )
    return {
        "score": max(0.0, min(1.0, score)),
        "weighted_similarity": weighted_similarity,
        "length_reward": length_reward,
        "format_reward": format_reward,
        "candidate_match": candidate_match,
        **similarity_scores,
    }
