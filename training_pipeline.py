"""Build and launch a Hy3 cover-letter RL training pipeline.

The Hy3 RL recipe expects verl parquet rows with five fields:
data_source, prompt, ability, reward_model, and extra_info. This module turns
scraped job/company/profile data plus one fixed resume into that format, with
optional candidate cover-letter generation for weak supervision and rewards.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from document_layout import conciseness_prompt
from TextReader import TextReader
from TextProcessor import TextProcessor

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    pd = None


CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent
DEFAULT_JOBS_DIR = CURR_DIR / "scraped_jobs"
DEFAULT_PROFILE_PATH = CURR_DIR / "scraped_profiles"
DEFAULT_RESUME_PATH = CURR_DIR / "sample_resumes"
DEFAULT_OUTPUT_DIR = CURR_DIR / "training_data" / "cover_letter_hy3"
DEFAULT_VERL_DIR = CURR_DIR / "verl"
DEFAULT_REWARD_PATH = CURR_DIR / "cover_letter_reward.py"
TEXT_READER_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".json"}
LLAMAPARSE_EXTENSIONS = {".pdf", ".doc", ".docx"}

NUM_GENERATIONS = 100
DATA_SOURCE = "job_application"
COVER_LETTER_ABILITY = "cover_letter"
RESUME_ABILITY = "resume"
CV_FOLDS = 5
REWARD_WEIGHT_TRIALS = 25
FINAL_SCORE_WEIGHT = 1.0
COMPONENT_KEYS = (
    "similarity_from_occurrences",
    "similarity_from_num_of_words",
    "similarity_ratio",
    "similarity_by_character",
    "similarity_by_commonality",
    "similarity_by_cosine",
)
DEFAULT_REWARD_WEIGHTS = {
    "similarity_from_occurrences": 1.0,
    "similarity_from_num_of_words": 1.0,
    "similarity_ratio": 1.0,
    "similarity_by_character": 0.25,
    "similarity_by_commonality": 1.0,
    "similarity_by_cosine": 1.0,
}
DEFAULT_GENERATION_PARAMS = {
    "cover_letter_temperature": 0.9,
    "cover_letter_top_p": 1.0,
    "resume_temperature": 0.8,
    "resume_top_p": 1.0,
}
DEFAULT_REWARD_BLEND_WEIGHTS = {
    "weighted_similarity": 0.60,
    "page_fit_reward": 0.10,
    "header_reward": 0.05,
    "length_reward": 0.10,
    "format_reward": 0.10,
    "candidate_match": 0.05,
}

BLEND_PARAM_PREFIX = "blend_"
DEFAULT_TUNING_PARAMS = {
    **DEFAULT_REWARD_WEIGHTS, **DEFAULT_GENERATION_PARAMS,
    "final_score_weight": FINAL_SCORE_WEIGHT,
    **{f"blend_{key}": value for key, value in DEFAULT_REWARD_BLEND_WEIGHTS.items()},
}


def tuning_blend_weights(params: dict[str, float]) -> dict[str, float]:
    from cover_letter_reward import _reward_blend_weights
    return _reward_blend_weights({
        key: params.get(f"blend_{key}", default)
        for key, default in DEFAULT_REWARD_BLEND_WEIGHTS.items()
    })


def candidate_reward(candidate: dict[str, Any], params: dict[str, float]) -> float:
    metrics = candidate["reward_metrics"]
    values = {**metrics, "weighted_similarity": weighted_similarity(candidate["component_scores"], params)}
    return sum(weight * values[key] for key, weight in tuning_blend_weights(params).items())


def candidate_reward_metrics(text: str, task: str, example: "JobExample") -> dict[str, float]:
    from cover_letter_reward import _length_reward, _format_reward
    from document_layout import layout_rewards
    return {
        "length_reward": _length_reward(text),
        "format_reward": _format_reward(text),
        # No independent accepted target exists at candidate-generation time.
        # Do not compare a candidate to itself or leak a held-out target.
        "candidate_match": 0.0,
        **layout_rewards(text, task, example.resume),
    }


@dataclass(frozen=True)
class JobExample:
    job_description: str
    resume: str
    user_profile: str
    company_profile: str
    title: str = ""
    company: str = ""
    location: str = ""
    source_file: str = ""
    source_index: int = 0


def load_dotenv_keys() -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def llamaparse_text(path: Path) -> str:
    """Extract text with LlamaParse when an API key is configured."""
    load_dotenv_keys()
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMA_PARSE_API_KEY")
    if not api_key:
        return ""

    try:
        from llama_parse import LlamaParse
    except ImportError:
        return ""

    parser = LlamaParse(api_key=api_key, result_type="markdown")
    documents = parser.load_data(str(path))
    return "\n\n".join(getattr(document, "text", "") for document in documents).strip()


def read_text(path: Path, prefer_llamaparse: bool = False) -> str:
    """Read text from txt/json/pdf/docx-ish sources."""
    if not path.exists():
        return ""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    if prefer_llamaparse and path.suffix.lower() in LLAMAPARSE_EXTENSIONS:
        parsed = llamaparse_text(path)
        if parsed:
            return parsed

    parsed = TextReader.parseText(str(path))
    if parsed:
        return parsed

    if path.suffix.lower() in LLAMAPARSE_EXTENSIONS:
        parsed = llamaparse_text(path)
        if parsed:
            return parsed
    return path.read_text(encoding="utf-8", errors="ignore")

def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing whitespace and removing non-printable characters."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    # Optional normalization including: lower, lemmatize, demojize
    text_processor = TextProcessor()
    text = text_processor.normalize(text)
    return text.strip()

def latest_supported_file(path: Path, label: str) -> Path:
    """Resolve a file or choose the latest supported text file in a directory."""
    if path.is_file():
        if path.suffix.lower() in TEXT_READER_EXTENSIONS:
            return path
        raise ValueError(f"{label} file has unsupported extension: {path}")

    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} path must be a file or directory: {path}")

    candidates = [
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in TEXT_READER_EXTENSIONS
    ]
    if not candidates:
        supported = ", ".join(sorted(TEXT_READER_EXTENSIONS))
        raise FileNotFoundError(f"No supported {label} file found in {path}. Supported: {supported}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def supported_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return [
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in TEXT_READER_EXTENSIONS
    ]


def extract_section(text: str, headings: tuple[str, ...]) -> str:
    pattern = "|".join(re.escape(heading) for heading in headings)
    match = re.search(rf"(?ims)^\s*(?:{pattern})\s*$([\s\S]*?)(?=^\s*[A-Z][A-Z /&-]{{2,}}\s*$|\Z)", text)
    if not match:
        return ""
    return "\n".join(line.strip() for line in match.group(1).splitlines() if line.strip())


def resume_profile_summary(resume_text: str) -> str:
    """Build compact applicant profile context directly from resume text."""
    resume_text = preprocess_text(resume_text)
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    contact = "\n".join(lines[:6])
    sections = {
        "summary": extract_section(resume_text, ("summary", "profile", "objective")),
        "skills": extract_section(resume_text, ("skills", "technical skills", "technologies")),
        "experience": extract_section(resume_text, ("experience", "work experience", "professional experience")),
        "education": extract_section(resume_text, ("education",)),
        "projects": extract_section(resume_text, ("projects", "selected projects")),
    }
    body = ["Derived user profile from attached resume.", "", "Contact/header:", contact]
    for label, value in sections.items():
        if value:
            body.extend(["", f"{label.title()}:", value[:4000]])
    if len(body) <= 3:
        body.extend(["", "Resume text:", resume_text[:8000]])
    return "\n".join(body).strip()


def load_user_profile(path: Path, resume_text: str) -> str:
    if path.is_file():
        return read_text(latest_supported_file(path, "user profile"))

    if supported_files(path):
        return read_text(latest_supported_file(path, "user profile"))

    return resume_profile_summary(resume_text)


def normalize_job_rows(raw: Any, source_file: Path) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("jobs", "data", "results"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [raw]
    return []


def extract_job_description(row: dict[str, Any]) -> str:
    fields = (
        "job_description",
        "job_description_formatted",
        "job_summary",
        "description",
        "summary",
    )
    for field in fields:
        value = row.get(field)
        if value:
            return str(value)
    return ""


def extract_company_profile(row: dict[str, Any], company_profiles: dict[str, str]) -> str:
    company = str(row.get("company") or row.get("company_name") or "").strip()
    if company and company.lower() in company_profiles:
        return company_profiles[company.lower()]

    snippets = []
    for field in ("company", "company_name", "company_description", "company_url", "industry"):
        value = row.get(field)
        if value:
            snippets.append(f"{field}: {value}")
    return "\n".join(snippets)


def load_company_profiles(paths: Iterable[Path]) -> dict[str, str]:
    profiles: dict[str, str] = {}
    for path in paths:
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = normalize_job_rows(raw, path)
        for row in rows:
            company = str(row.get("company") or row.get("name") or row.get("company_name") or "").strip()
            if company:
                profiles[company.lower()] = json.dumps(row, ensure_ascii=False)
    return profiles


def load_jobs(
    jobs_dir: Path,
    resume_path: Path,
    user_profile_path: Path,
    company_profile_paths: Iterable[Path],
    limit: int | None,
) -> list[JobExample]:
    resume_file = latest_supported_file(resume_path, "resume")
    resume = read_text(resume_file, prefer_llamaparse=True)
    resume = preprocess_text(resume)
    user_profile = load_user_profile(user_profile_path, resume)
    user_profile = preprocess_text(user_profile)
    company_profiles = load_company_profiles(company_profile_paths)
    company_profiles = {k: preprocess_text(v) for k, v in company_profiles.items()}

    examples: list[JobExample] = []
    for path in sorted(jobs_dir.glob("*.json")):
        rows = normalize_job_rows(json.loads(path.read_text(encoding="utf-8")), path)
        for index, row in enumerate(rows):
            job_description = extract_job_description(row).strip()
            if not job_description:
                continue
            examples.append(
                JobExample(
                    job_description=job_description,
                    resume=resume,
                    user_profile=user_profile,
                    company_profile=extract_company_profile(row, company_profiles),
                    title=str(row.get("title") or row.get("job_title") or ""),
                    company=str(row.get("company") or row.get("company_name") or ""),
                    location=str(row.get("location") or ""),
                    source_file=str(path),
                    source_index=index,
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def cover_letter_prompt(example: JobExample) -> list[dict[str, str]]:
    system = (
        "You are a professional cover letter writer. Use only facts supported by "
        "the provided resume, user profile, job description, and company profile. "
        "Write a concise, specific, ATS-aware cover letter."
    )
    user = f"""
        Fixed resume:
        {example.resume}

        User profile:
        {example.user_profile}

        Company profile:
        {example.company_profile}

        Job title: {example.title}
        Company: {example.company}
        Location: {example.location}

        Job description:
        {example.job_description}

        Return only the finished cover letter.
    """
    system += "\n" + conciseness_prompt("cover_letter")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def resume_prompt(example: JobExample) -> list[dict[str, str]]:
    system = (
        "You are a professional resume writer. Tailor the applicant's resume to "
        "the job while preserving factual accuracy. Use the user profile and "
        "company profile only as context; do not invent unsupported facts."
    )
    user = f"""
        Original resume:
        {example.resume}

        User profile:
        {example.user_profile}

        Company profile:
        {example.company_profile}

        Job title: {example.title}
        Company: {example.company}
        Location: {example.location}

        Job description:
        {example.job_description}

        Return only the tailored resume.
    """
    system += "\n" + conciseness_prompt("resume")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def similarity_component_scores(example: JobExample, resume: str, cover_letter: str) -> dict[str, float]:
    from SimilarityModel import SimilarityModel
    from TextProcessor import TextProcessor

    model = SimilarityModel(example.job_description, resume, cover_letter)
    scores = model.componentScores()
    scores["final_score"] = float(model.evaluateFinalScore())
    occurences_source = TextProcessor.getOccurences(model.tokenizedSource, model.tokenizedSource)
    occurences_target = TextProcessor.getOccurences(model.tokenizedCorpus, model.tokenizedCorpus)
    scores["simiilarityFromOccurences"] = scores["similarity_from_occurrences"]
    scores["simiilarityFromNumOfWords"] = float(
        model.simiilarityFromNumOfWords(occurences_source, occurences_target, model.tokenizedCorpus)
    )
    scores["similarityRatio"] = scores["similarity_ratio"]
    scores["similarityByCharacter"] = scores["similarity_by_character"]
    scores["similarityByCommonality"] = scores["similarity_by_commonality"]
    scores["similarityByCosineSimilarity"] = scores["similarity_by_cosine"]
    scores["evaluateFinalScore"] = scores["final_score"]
    return {key: max(0.0, min(1.0, float(value))) for key, value in scores.items()}


def weighted_similarity(scores: dict[str, float], weights: dict[str, float]) -> float:
    final_weight = weights.get("final_score_weight", FINAL_SCORE_WEIGHT)
    weighted_total = final_weight * scores.get("final_score", 0.0)
    total_weight = final_weight
    for key in COMPONENT_KEYS:
        weight = float(weights.get(key, DEFAULT_REWARD_WEIGHTS[key]))
        weighted_total += weight * scores.get(key, 0.0)
        total_weight += weight
    return weighted_total / total_weight if total_weight > 0 else 0.0


def task_generation_params(params: dict[str, float], task: str) -> dict[str, float]:
    prefix = "cover_letter" if task == COVER_LETTER_ABILITY else "resume"
    return {
        "temperature": float(params[f"{prefix}_temperature"]),
        "top_p": float(params[f"{prefix}_top_p"]),
    }


def candidate_matches_generation_params(candidate: dict[str, Any], params: dict[str, float], task: str) -> bool:
    candidate_params = candidate.get("generation_params")
    if not candidate_params:
        return True
    expected = task_generation_params(params, task)
    return all(abs(float(candidate_params[key]) - value) < 1e-9 for key, value in expected.items())


def score_application(
    example: JobExample,
    resume: str,
    cover_letter: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Local reward proxy used for generated labels and CV weight tuning."""
    try:
        score = weighted_similarity(
            similarity_component_scores(example, resume, cover_letter),
            weights or DEFAULT_REWARD_WEIGHTS,
        )
    except Exception:
        score = 0.0

    letter_words = cover_letter.split()
    length_score = 1.0 if 250 <= len(letter_words) <= 450 else 0.5
    unsupported_penalty = 0.0 if "As an AI" not in f"{resume}\n{cover_letter}" else 0.25
    return max(0.0, min(1.0, (0.8 * score) + (0.2 * length_score) - unsupported_penalty))


def generation_params_for_trial(
    trials: list[dict[str, float]],
    generation: int,
    task: str,
) -> dict[str, float]:
    params = trials[generation % len(trials)] if trials else DEFAULT_TUNING_PARAMS
    return task_generation_params(params, task)


def generate_cover_letter_candidates(
    examples: list[JobExample],
    num_generations: int,
    tuning_trials: list[dict[str, float]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    from CoverLetterModel import CoverLetterModel

    model = CoverLetterModel()
    candidates: dict[int, list[dict[str, Any]]] = {}
    for index, example in enumerate(examples):
        candidates[index] = []
        for generation in range(num_generations):
            generation_params = generation_params_for_trial(tuning_trials or [], generation, COVER_LETTER_ABILITY)
            cover_letter = model.generate_cover_letter(
                example.job_description,
                example.resume,
                example.user_profile,
                example.company_profile,
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
            )
            candidates[index].append(
                {
                    "generation": generation,
                    "generation_params": generation_params,
                    "cover_letter": cover_letter,
                    "reward_metrics": candidate_reward_metrics(cover_letter, COVER_LETTER_ABILITY, example),
                    "component_scores": similarity_component_scores(example, example.resume, cover_letter),
                    "score": score_application(example, example.resume, cover_letter),
                }
            )
    return candidates


def generate_resume_candidates(
    examples: list[JobExample],
    num_generations: int,
    tuning_trials: list[dict[str, float]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    from ResumeModel import ResumeModel

    model = ResumeModel()
    candidates: dict[int, list[dict[str, Any]]] = {}
    for index, example in enumerate(examples):
        candidates[index] = []
        for generation in range(num_generations):
            generation_params = generation_params_for_trial(tuning_trials or [], generation, RESUME_ABILITY)
            tailored_resume = model.generate_resume(
                example.job_description,
                example.resume,
                example.user_profile,
                example.company_profile,
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
            )
            candidates[index].append(
                {
                    "generation": generation,
                    "generation_params": generation_params,
                    "tailored_resume": tailored_resume,
                    "reward_metrics": candidate_reward_metrics(tailored_resume, RESUME_ABILITY, example),
                    "component_scores": similarity_component_scores(example, tailored_resume, ""),
                    "score": score_application(example, tailored_resume, ""),
                }
            )
    return candidates


class RandomSuggestionTrial:
    """Small Optuna-like fallback for numeric reward-weight suggestions."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def suggest_float(self, name: str, low: float, high: float, step: float | None = None) -> float:
        if step is None:
            return self.rng.uniform(low, high)
        steps = int(round((high - low) / step))
        return low + (self.rng.randint(0, steps) * step)


def suggest_tuning_params(trial: Any) -> dict[str, float]:
    return {
        "final_score_weight": trial.suggest_float("final_score_weight", 0.0, 2.0, step=0.05),
        **{f"blend_{key}": trial.suggest_float(f"blend_{key}", 0.0, 1.0, step=0.05)
           for key in DEFAULT_REWARD_BLEND_WEIGHTS},
        "similarity_from_occurrences": trial.suggest_float("similarity_from_occurrences", 0.0, 2.0, step=0.05),
        "similarity_from_num_of_words": trial.suggest_float("similarity_from_num_of_words", 0.0, 2.0, step=0.05),
        "similarity_ratio": trial.suggest_float("similarity_ratio", 0.0, 2.0, step=0.05),
        "similarity_by_character": trial.suggest_float("similarity_by_character", 0.0, 1.0, step=0.05),
        "similarity_by_commonality": trial.suggest_float("similarity_by_commonality", 0.0, 2.0, step=0.05),
        "similarity_by_cosine": trial.suggest_float("similarity_by_cosine", 0.0, 2.0, step=0.05),
        "cover_letter_temperature": trial.suggest_float("cover_letter_temperature", 0.2, 1.0, step=0.1),
        "cover_letter_top_p": trial.suggest_float("cover_letter_top_p", 0.7, 1.0, step=0.05),
        "resume_temperature": trial.suggest_float("resume_temperature", 0.2, 1.0, step=0.1),
        "resume_top_p": trial.suggest_float("resume_top_p", 0.7, 1.0, step=0.05),
    }


def reward_weight_trials(num_trials: int, seed: int) -> list[dict[str, float]]:
    try:
        import optuna
    except ImportError:
        rng = random.Random(seed)
        return [suggest_tuning_params(RandomSuggestionTrial(rng)) for _ in range(num_trials)]

    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trials = []
    for _ in range(num_trials):
        trial = study.ask()
        trials.append(suggest_tuning_params(trial))
        study.tell(trial, 0.0)
    return trials


def tune_reward_weights(
    candidates_by_task: dict[str, dict[int, list[dict[str, Any]]]],
    folds: int,
    seed: int,
    output_dir: Path,
    num_trials: int,
    trials: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    application_indexes = sorted(
        {
            index
            for candidates in candidates_by_task.values()
            for index, items in candidates.items()
            if items
        }
    )
    if len(application_indexes) < 2:
        return DEFAULT_TUNING_PARAMS

    shuffled = application_indexes[:]
    random.Random(seed).shuffle(shuffled)
    fold_count = min(folds, len(shuffled))
    fold_indexes = [shuffled[i::fold_count] for i in range(fold_count)]
    trials = trials or [DEFAULT_TUNING_PARAMS, *reward_weight_trials(num_trials, seed)]
    fold_results = []

    def mean_best_score(indexes: list[int], weights: dict[str, float]) -> float:
        if not indexes:
            return 0.0
        application_scores = []
        for index in indexes:
            task_scores = []
            for task, task_candidates in candidates_by_task.items():
                weighted_scores = [
                    candidate_reward(candidate, weights)
                    for candidate in task_candidates.get(index, [])
                    if candidate_matches_generation_params(candidate, weights, task)
                ]
                if weighted_scores:
                    eligible = [candidate for candidate in task_candidates.get(index, [])
                                if candidate_matches_generation_params(candidate, weights, task)]
                    selected = max(eligible, key=lambda candidate: candidate_reward(candidate, weights))
                    task_scores.append(candidate_reward(selected, DEFAULT_TUNING_PARAMS))
            if task_scores:
                application_scores.append(sum(task_scores) / len(task_scores))
        return sum(application_scores) / len(application_scores) if application_scores else 0.0

    selected_weights = []
    for fold_id, val_indexes in enumerate(fold_indexes):
        train_indexes = [index for index in shuffled if index not in set(val_indexes)]
        best_weights = max(trials, key=lambda weights: mean_best_score(train_indexes, weights))
        fold_results.append(
            {
                "fold": fold_id,
                "train_indexes": train_indexes,
                "val_indexes": val_indexes,
                "train_score": mean_best_score(train_indexes, best_weights),
                "val_score": mean_best_score(val_indexes, best_weights),
                "weights": best_weights,
            }
        )
        selected_weights.append(best_weights)

    averaged = {
        key: sum(weights[key] for weights in selected_weights) / len(selected_weights)
        for key in [*COMPONENT_KEYS, "final_score_weight",
                    *(f"blend_{key}" for key in DEFAULT_REWARD_BLEND_WEIGHTS)]
    }
    averaged_generation_params = {
        key: sum(weights[key] for weights in selected_weights) / len(selected_weights)
        for key in DEFAULT_GENERATION_PARAMS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_reward_weights.json").write_text(
        json.dumps(
            {
                "selected_weights": averaged,
                "selected_generation_params": averaged_generation_params,
                "selected_reward_blend_weights": tuning_blend_weights(averaged),
                "folds": fold_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {**averaged, **averaged_generation_params}


def best_candidate(
    candidates: list[dict[str, Any]] | None,
    weights: dict[str, float],
    task: str,
) -> dict[str, Any] | None:
    return max(
        [
            candidate
            for candidate in candidates or []
            if candidate_matches_generation_params(candidate, weights, task)
        ],
        key=lambda item: candidate_reward(item, weights),
        default=None,
    )


def to_cover_letter_row(
    example: JobExample,
    index: int,
    cover_letter_candidates: list[dict[str, Any]] | None = None,
    resume_candidates: list[dict[str, Any]] | None = None,
    reward_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = reward_weights or DEFAULT_TUNING_PARAMS
    generation_params = task_generation_params(weights, COVER_LETTER_ABILITY)
    component_weights = {key: weights[key] for key in COMPONENT_KEYS}
    best_cover_letter = best_candidate(cover_letter_candidates, weights, COVER_LETTER_ABILITY)
    best_resume = best_candidate(resume_candidates, weights, RESUME_ABILITY)
    ground_truth = {
        "task_type": COVER_LETTER_ABILITY,
        "resume": example.resume,
        "best_tailored_resume": best_resume["tailored_resume"] if best_resume else "",
        "user_profile": example.user_profile,
        "company_profile": example.company_profile,
        "job_description": example.job_description,
        "best_cover_letter": best_cover_letter["cover_letter"] if best_cover_letter else "",
        "best_score": candidate_reward(best_cover_letter, weights) if best_cover_letter else None,
        "reward_weights": component_weights,
        "generation_params": generation_params,
        "reward_blend_weights": tuning_blend_weights(weights),
        "final_score_weight": weights.get("final_score_weight", FINAL_SCORE_WEIGHT),
        "rubric": {
            "factuality": "Do not invent applicant experience or company facts.",
            "relevance": "Address job requirements and company context.",
            "format": "Return only a polished cover letter.",
        },
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": cover_letter_prompt(example),
        "ability": COVER_LETTER_ABILITY,
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "split": "train",
            "index": index,
            "application_id": f"{example.source_file}:{example.source_index}",
            "title": example.title,
            "company": example.company,
            "location": example.location,
            "source_file": example.source_file,
            "source_index": example.source_index,
            "task_type": COVER_LETTER_ABILITY,
        },
    }


def to_resume_row(
    example: JobExample,
    index: int,
    resume_candidates: list[dict[str, Any]] | None = None,
    cover_letter_candidates: list[dict[str, Any]] | None = None,
    reward_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = reward_weights or DEFAULT_TUNING_PARAMS
    generation_params = task_generation_params(weights, RESUME_ABILITY)
    component_weights = {key: weights[key] for key in COMPONENT_KEYS}
    best_resume = best_candidate(resume_candidates, weights, RESUME_ABILITY)
    best_cover_letter = best_candidate(cover_letter_candidates, weights, COVER_LETTER_ABILITY)
    ground_truth = {
        "task_type": RESUME_ABILITY,
        "resume": example.resume,
        "user_profile": example.user_profile,
        "company_profile": example.company_profile,
        "job_description": example.job_description,
        "cover_letter": best_cover_letter["cover_letter"] if best_cover_letter else "",
        "best_cover_letter": best_cover_letter["cover_letter"] if best_cover_letter else "",
        "best_tailored_resume": best_resume["tailored_resume"] if best_resume else "",
        "best_score": candidate_reward(best_resume, weights) if best_resume else None,
        "reward_weights": component_weights,
        "generation_params": generation_params,
        "reward_blend_weights": tuning_blend_weights(weights),
        "final_score_weight": weights.get("final_score_weight", FINAL_SCORE_WEIGHT),
        "rubric": {
            "factuality": "Do not invent applicant experience or company facts.",
            "relevance": "Address job requirements and optimize the resume for the job.",
            "format": "Return only a tailored resume.",
        },
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": resume_prompt(example),
        "ability": RESUME_ABILITY,
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "split": "train",
            "index": index,
            "application_id": f"{example.source_file}:{example.source_index}",
            "title": example.title,
            "company": example.company,
            "location": example.location,
            "source_file": example.source_file,
            "source_index": example.source_index,
            "task_type": RESUME_ABILITY,
        },
    }


def application_group_key(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    return str(
        extra_info.get("application_id")
        or f"{extra_info.get('source_file', '')}:{extra_info.get('source_index', extra_info.get('index', ''))}"
    )


def split_rows_by_application(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(application_group_key(row), []).append(row)

    group_keys = list(grouped)
    random.Random(seed).shuffle(group_keys)
    val_count = max(1, int(len(group_keys) * val_ratio)) if len(group_keys) > 1 else 0
    val_keys = set(group_keys[:val_count])
    train_keys = set(group_keys[val_count:]) or set(group_keys)

    train_rows = [row for key in group_keys if key in train_keys for row in grouped[key]]
    val_rows = [row for key in group_keys if key in val_keys for row in grouped[key]]
    return train_rows, val_rows or train_rows[:]


def cross_validation_row_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(application_group_key(row), []).append(row)

    group_keys = list(grouped)
    if len(group_keys) < 2:
        return []

    fold_count = min(folds, len(group_keys))
    random.Random(seed).shuffle(group_keys)
    fold_keys = [group_keys[i::fold_count] for i in range(fold_count)]
    folds_out = []
    for val_keys in fold_keys:
        val_key_set = set(val_keys)
        train_keys = [key for key in group_keys if key not in val_key_set]
        train_rows = [row for key in train_keys for row in grouped[key]]
        val_rows = [row for key in val_keys for row in grouped[key]]
        folds_out.append((train_rows, val_rows))
    return folds_out


def write_cross_validation_folds(rows: list[dict[str, Any]], output_dir: Path, folds: int, seed: int) -> None:
    if pd is None:
        raise RuntimeError("Install pandas and pyarrow to write Hy3/verl parquet files.")
    row_folds = cross_validation_row_folds(rows, folds, seed)
    if not row_folds:
        return

    for fold_id, (train_rows, val_rows) in enumerate(row_folds):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_rows).to_parquet(fold_dir / "train.parquet", index=False)
        pd.DataFrame(val_rows).to_parquet(fold_dir / "val.parquet", index=False)


def write_parquet(rows: list[dict[str, Any]], output_dir: Path, val_ratio: float, seed: int) -> tuple[Path, Path]:
    if pd is None:
        raise RuntimeError("Install pandas and pyarrow to write Hy3/verl parquet files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = split_rows_by_application(rows, val_ratio, seed)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows or train_rows[:1]).to_parquet(val_path, index=False)
    return train_path, val_path


def build_launch_command(
    train_path: Path,
    val_path: Path,
    verl_dir: Path,
    reward_path: Path,
    ray_address: str | None,
    checkpoint_dir: Path | None = None,
    resume_from_checkpoint: Path | None = None,
) -> str:
    env_parts = []
    if ray_address:
        env_parts.append(f"RAY_ADDRESS={shlex.quote(ray_address)}")
    env_parts.extend(
        [
            f"TRAIN_FILES={shlex.quote(str(train_path.resolve()))}",
            f"VAL_FILES={shlex.quote(str(val_path.resolve()))}",
        ]
    )
    script = verl_dir / "examples" / "grpo_trainer" / "run_hy_v3_megatron.sh"
    reward_args = [
        f"reward.custom_reward_function.path={shlex.quote(str(reward_path.resolve()))}",
        "reward.custom_reward_function.name=compute_score",
    ]
    checkpoint_args = []
    if checkpoint_dir:
        checkpoint_args.extend(
            [
                f"trainer.default_local_dir={shlex.quote(str(checkpoint_dir.resolve()))}",
                "trainer.resume_mode=auto",
            ]
        )
    if resume_from_checkpoint:
        checkpoint_args.append(f"trainer.resume_from_path={shlex.quote(str(resume_from_checkpoint.resolve()))}")
    job_parts = ["ray", "job", "submit", "--no-wait", "--runtime-env=runtime_env.yaml", "--"]
    return " ".join(env_parts + job_parts + ["bash", shlex.quote(str(script)), *reward_args, *checkpoint_args])


def write_training_state(output_dir: Path, status: str, **extra: Any) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    state_path = output_dir / "training_state.json"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state_path


def train_models_collectively(args: argparse.Namespace) -> tuple[Path, Path]:
    """Prepare one shared RL dataset containing both resume and cover-letter tasks."""
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    write_training_state(
        args.output_dir,
        "preparing",
        tasks=args.tasks,
        checkpoint_dir=str(checkpoint_dir.resolve()) if checkpoint_dir else "",
        resume_from_checkpoint=str(resume_from_checkpoint.resolve()) if resume_from_checkpoint else "",
    )
    examples = load_jobs(
        jobs_dir=args.jobs_dir,
        resume_path=args.resume,
        user_profile_path=args.user_profile,
        company_profile_paths=args.company_profiles,
        limit=args.max_jobs,
    )
    if not examples:
        raise RuntimeError(f"No job descriptions found in {args.jobs_dir}")

    tasks = set(args.tasks)
    write_training_state(
        args.output_dir,
        "generating_candidates" if args.generate_candidates else "building_rows",
        tasks=args.tasks,
        num_examples=len(examples),
        checkpoint_dir=str(checkpoint_dir.resolve()) if checkpoint_dir else "",
        resume_from_checkpoint=str(resume_from_checkpoint.resolve()) if resume_from_checkpoint else "",
    )
    tuning_trials = (
        [DEFAULT_TUNING_PARAMS, *reward_weight_trials(args.reward_weight_trials, args.seed)]
        if args.tune_reward_weights and args.generate_candidates
        else [DEFAULT_TUNING_PARAMS]
    )
    cover_letter_candidates = (
        generate_cover_letter_candidates(examples, args.num_generations, tuning_trials)
        if args.generate_candidates and COVER_LETTER_ABILITY in tasks
        else {}
    )
    resume_candidates = (
        generate_resume_candidates(examples, args.num_generations, tuning_trials)
        if args.generate_candidates and RESUME_ABILITY in tasks
        else {}
    )
    reward_weights = (
        tune_reward_weights(
            {
                COVER_LETTER_ABILITY: cover_letter_candidates,
                RESUME_ABILITY: resume_candidates,
            },
            args.cv_folds,
            args.seed,
            args.output_dir,
            args.reward_weight_trials,
            tuning_trials,
        )
        if args.tune_reward_weights
        else DEFAULT_TUNING_PARAMS
    )

    rows = []
    for index, example in enumerate(examples):
        if COVER_LETTER_ABILITY in tasks:
            rows.append(
                to_cover_letter_row(
                    example,
                    index,
                    cover_letter_candidates.get(index),
                    resume_candidates.get(index),
                    reward_weights,
                )
            )
        if RESUME_ABILITY in tasks:
            rows.append(
                to_resume_row(
                    example,
                    index,
                    resume_candidates.get(index),
                    cover_letter_candidates.get(index),
                    reward_weights,
                )
            )

    return write_training_outputs(rows, args)


def write_training_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    train_path, val_path = write_parquet(rows, args.output_dir, args.val_ratio, args.seed)
    if args.write_cv_folds:
        write_cross_validation_folds(rows, args.output_dir, args.cv_folds, args.seed)

    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    command = build_launch_command(
        train_path,
        val_path,
        args.verl_dir,
        args.reward_path,
        args.ray_address,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    (args.output_dir / "launch_hy3_grpo.sh").write_text(command + "\n", encoding="utf-8")

    if args.submit:
        write_training_state(
            args.output_dir,
            "submitted",
            checkpoint_dir=str(checkpoint_dir.resolve()) if checkpoint_dir else "",
            resume_from_checkpoint=str(resume_from_checkpoint.resolve()) if resume_from_checkpoint else "",
            train_parquet=str(train_path.resolve()),
            val_parquet=str(val_path.resolve()),
        )
        subprocess.run(command, cwd=args.verl_dir, shell=True, check=True)
    else:
        write_training_state(
            args.output_dir,
            "prepared",
            checkpoint_dir=str(checkpoint_dir.resolve()) if checkpoint_dir else "",
            resume_from_checkpoint=str(resume_from_checkpoint.resolve()) if resume_from_checkpoint else "",
            train_parquet=str(train_path.resolve()),
            val_parquet=str(val_path.resolve()),
        )

    print(f"Wrote {len(rows)} verl rows")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Launch command saved to: {args.output_dir / 'launch_hy3_grpo.sh'}")
    return train_path, val_path


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Path]:
    if not getattr(args, "_internal_model_call", False) and len(args.tasks) == 1:
        task = args.tasks[0]
        if task == COVER_LETTER_ABILITY:
            from CoverLetterModel import CoverLetterModel

            model = CoverLetterModel()
        elif task == RESUME_ABILITY:
            from ResumeModel import ResumeModel

            model = ResumeModel()
        else:
            raise ValueError(f"Unsupported training task: {task}")

        train_kwargs = {
            "jobs_dir": args.jobs_dir,
            "resume": args.resume,
            "user_profile": args.user_profile,
            "company_profiles": args.company_profiles,
            "output_dir": args.output_dir,
            "verl_dir": args.verl_dir,
            "reward_path": args.reward_path,
            "generate_candidates": args.generate_candidates,
            "write_cv_folds": args.write_cv_folds,
            "submit": args.submit,
            "num_generations": args.num_generations,
            "max_jobs": args.max_jobs,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "reward_weight_trials": args.reward_weight_trials,
            "ray_address": args.ray_address,
            "checkpoint_dir": getattr(args, "checkpoint_dir", None),
            "resume_from_checkpoint": getattr(args, "resume_from_checkpoint", None),
        }
        if args.tune_reward_weights:
            return model.tune(**train_kwargs)
        return model.train(tune_reward_weights=False, **train_kwargs)

    return train_models_collectively(args)


def train_model_task(
    task: str,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    resume: Path = DEFAULT_RESUME_PATH,
    user_profile: Path = DEFAULT_PROFILE_PATH,
    company_profiles: Iterable[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verl_dir: Path = DEFAULT_VERL_DIR,
    reward_path: Path = DEFAULT_REWARD_PATH,
    num_generations: int = NUM_GENERATIONS,
    max_jobs: int | None = None,
    val_ratio: float = 0.1,
    seed: int = 7,
    cv_folds: int = CV_FOLDS,
    reward_weight_trials: int = REWARD_WEIGHT_TRIALS,
    ray_address: str | None = None,
    checkpoint_dir: Path | None = None,
    resume_from_checkpoint: Path | None = None,
    generate_candidates: bool = False,
    tune_reward_weights: bool = False,
    write_cv_folds: bool = False,
    submit: bool = False,
) -> tuple[Path, Path]:
    args = argparse.Namespace(
        jobs_dir=jobs_dir,
        resume=resume,
        user_profile=user_profile,
        company_profiles=list(company_profiles or []),
        output_dir=output_dir,
        verl_dir=verl_dir,
        reward_path=reward_path,
        num_generations=num_generations,
        max_jobs=max_jobs,
        val_ratio=val_ratio,
        seed=seed,
        cv_folds=cv_folds,
        reward_weight_trials=reward_weight_trials,
        tasks=[task],
        ray_address=ray_address,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        generate_candidates=generate_candidates,
        tune_reward_weights=tune_reward_weights,
        write_cv_folds=write_cv_folds,
        submit=submit,
        _internal_model_call=True,
    )
    return run_pipeline(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Hy3/verl cover-letter training data.")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME_PATH)
    parser.add_argument("--user-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--company-profiles", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verl-dir", type=Path, default=DEFAULT_VERL_DIR)
    parser.add_argument("--reward-path", type=Path, default=DEFAULT_REWARD_PATH)
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--reward-weight-trials", type=int, default=REWARD_WEIGHT_TRIALS)
    parser.add_argument(
        "--tasks",
        choices=(COVER_LETTER_ABILITY, RESUME_ABILITY),
        nargs="+",
        default=[COVER_LETTER_ABILITY, RESUME_ABILITY],
    )
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Stable verl checkpoint directory for saving collective model weights.")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None, help="Existing verl checkpoint directory to resume from.")
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--tune-reward-weights", action="store_true")
    parser.add_argument("--write-cv-folds", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
