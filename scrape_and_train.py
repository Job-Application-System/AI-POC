"""Minimal scrape-then-train orchestration for the AI proof of concept.

This file intentionally stays thin: scraping and RL dataset/training behavior
remain in the existing modules so the same pieces can later feed a Google Sheet
evaluation loop for resume and cover-letter generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent
LINKEDIN_DIR = ROOT_DIR / "LinkedIn-Scraper"
FREE_SCRAPER_DIR = LINKEDIN_DIR / "free_scraper"
API_CODES_DIR = LINKEDIN_DIR / "linkedin_scraper_api_codes"

sys.path.extend([str(CURR_DIR), str(FREE_SCRAPER_DIR), str(API_CODES_DIR)])

import training_pipeline
import scrape_job_descriptions
from linkedin_jobs_by_keyword import LinkedInJobsDiscovery
from profile_scraper import save_result, scrape


DEFAULT_MODEL_DIR = CURR_DIR / "trained_models" / "job_application_hy3"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def selected_keywords(args: argparse.Namespace) -> list[str]:
    return args.keyword or list(scrape_job_descriptions.KEYWORDS)


def selected_locations(args: argparse.Namespace) -> list[str]:
    return args.location or list(scrape_job_descriptions.LOCATIONS)


def safe_filename_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def scrape_relevant_jobs(args: argparse.Namespace) -> list[Any]:
    args.jobs_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_scrape:
        return []

    # Try Free method first
    original_keywords = scrape_job_descriptions.KEYWORDS
    original_locations = scrape_job_descriptions.LOCATIONS
    original_saved_jobs_dir = scrape_job_descriptions.SAVED_JOBS_DIR
    scrape_job_descriptions.KEYWORDS = selected_keywords(args)
    scrape_job_descriptions.LOCATIONS = selected_locations(args)
    scrape_job_descriptions.SAVED_JOBS_DIR = args.jobs_dir
    try:
        jobs = scrape_job_descriptions.scrape_jobs(save_results=True, max_jobs=args.max_jobs)
        scrape_job_descriptions.extract_job_descriptions(jobs, save_results=True)
    finally:
        scrape_job_descriptions.KEYWORDS = original_keywords
        scrape_job_descriptions.LOCATIONS = original_locations
        scrape_job_descriptions.SAVED_JOBS_DIR = original_saved_jobs_dir
    if jobs:
        return jobs

    api_token = args.brightdata_token or os.environ.get("BRIGHTDATA_APIKEY")
    if not api_token:
        raise RuntimeError("BrightData scraping requires --brightdata-token or BRIGHTDATA_APIKEY.")

    jobs: list[Any] = []
    discoverer = LinkedInJobsDiscovery(api_token)
    original_saved_jobs_dir = scrape_job_descriptions.SAVED_JOBS_DIR
    scrape_job_descriptions.SAVED_JOBS_DIR = args.jobs_dir
    try:
        for keyword in selected_keywords(args):
            for location in selected_locations(args):
                output = args.jobs_dir / f"linkedin_jobs_{safe_filename_part(keyword)}_{safe_filename_part(location)}.json"
                search_criteria = [
                    {
                        "location": location,
                        "keyword": keyword,
                        "country": args.country,
                        "time_range": args.time_range,
                        "job_type": args.job_type,
                        "experience_level": args.experience_level,
                        "remote": args.remote,
                        "company": args.company,
                    }
                ]
                discovered_jobs = discoverer.discover_jobs(search_criteria, output=str(output)) or []
                jobs.extend(discovered_jobs)

        scrape_job_descriptions.extract_job_descriptions(jobs, save_results=True)
    finally:
        scrape_job_descriptions.SAVED_JOBS_DIR = original_saved_jobs_dir
    return jobs


async def scrape_profiles(args: argparse.Namespace) -> list[Path]:
    outputs: list[Path] = []
    args.profile_dir.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(args.profile_urls):
        output = args.profile_dir / f"profile_{index}_{timestamp()}.json"
        data = await scrape(url, profile_type="person", force_login=args.force_profile_login)
        save_result(data, str(output))
        outputs.append(output)

    for index, url in enumerate(args.company_profile_urls):
        output = args.profile_dir / f"company_{index}_{timestamp()}.json"
        data = await scrape(url, profile_type="company", force_login=args.force_profile_login)
        save_result(data, str(output))
        outputs.append(output)

    return outputs


def persist_training_manifest(
    train_path: Path,
    val_path: Path,
    args: argparse.Namespace,
    profile_outputs: list[Path],
) -> Path:
    args.model_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.model_dir / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    launch_path = args.output_dir / "launch_hy3_grpo.sh"
    if launch_path.exists():
        shutil.copy2(launch_path, run_dir / "launch_hy3_grpo.sh")

    copied_checkpoint_dir = ""
    if args.checkpoint_dir and args.checkpoint_dir.exists():
        copied_checkpoint_dir = str((run_dir / "checkpoints").resolve())
        shutil.copytree(args.checkpoint_dir, run_dir / "checkpoints", dirs_exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Final RL model checkpoints belong under this run directory. Use --checkpoint-dir to copy a completed Hy3/verl checkpoint here.",
        "submitted_training": args.submit,
        "train_parquet": str(train_path.resolve()),
        "val_parquet": str(val_path.resolve()),
        "training_output_dir": str(args.output_dir.resolve()),
        "model_dir": str(run_dir.resolve()),
        "copied_checkpoint_dir": copied_checkpoint_dir,
        "profile_outputs": [str(path.resolve()) for path in profile_outputs],
        "tasks": args.tasks,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def run(args: argparse.Namespace) -> Path:
    scrape_relevant_jobs(args)
    profile_outputs = asyncio.run(scrape_profiles(args)) if args.profile_urls or args.company_profile_urls else []

    pipeline_args = argparse.Namespace(
        jobs_dir=args.jobs_dir,
        resume=args.resume,
        user_profile=args.user_profile,
        company_profiles=[*args.company_profiles, *[path for path in profile_outputs if path.name.startswith("company_")]],
        output_dir=args.output_dir,
        verl_dir=args.verl_dir,
        reward_path=args.reward_path,
        num_generations=args.num_generations,
        max_jobs=args.training_max_jobs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cv_folds=args.cv_folds,
        reward_weight_trials=args.reward_weight_trials,
        tasks=args.tasks,
        ray_address=args.ray_address,
        generate_candidates=args.generate_candidates,
        tune_reward_weights=args.tune_reward_weights,
        write_cv_folds=args.write_cv_folds,
        submit=args.submit,
        _internal_model_call=False,
    )
    train_path, val_path = training_pipeline.run_pipeline(pipeline_args)
    return persist_training_manifest(train_path, val_path, args, profile_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape relevant jobs/profiles, then prepare or launch Hy3/verl RL training.")
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse existing files in --jobs-dir.")
    parser.add_argument("--jobs-dir", type=Path, default=training_pipeline.DEFAULT_JOBS_DIR)
    parser.add_argument("--profile-dir", type=Path, default=training_pipeline.DEFAULT_PROFILE_PATH)
    parser.add_argument("--profile-urls", nargs="*", default=[])
    parser.add_argument("--company-profile-urls", nargs="*", default=[])
    parser.add_argument("--force-profile-login", action="store_true")
    parser.add_argument("--resume", type=Path, default=training_pipeline.DEFAULT_RESUME_PATH)
    parser.add_argument("--user-profile", type=Path, default=training_pipeline.DEFAULT_PROFILE_PATH)
    parser.add_argument("--company-profiles", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=training_pipeline.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional completed Hy3/verl checkpoint directory to copy into --model-dir.")
    parser.add_argument("--verl-dir", type=Path, default=training_pipeline.DEFAULT_VERL_DIR)
    parser.add_argument("--reward-path", type=Path, default=training_pipeline.DEFAULT_REWARD_PATH)
    parser.add_argument("--max-jobs", type=int, default=100, help="Jobs to scrape per search in the scraper.")
    parser.add_argument("--training-max-jobs", type=int, default=None, help="Optional cap passed to training_pipeline.")
    parser.add_argument("--keyword", action="append", default=None, help="Job keyword to scrape. Repeat to pass several. Defaults to scrape_job_descriptions.KEYWORDS.")
    parser.add_argument("--location", action="append", default=None, help="Job location to scrape. Repeat to pass several. Defaults to scrape_job_descriptions.LOCATIONS.")
    parser.add_argument("--country", default="CA")
    parser.add_argument("--time-range", default="Any time")
    parser.add_argument("--job-type", default="Full-time")
    parser.add_argument("--experience-level", default="Entry level")
    parser.add_argument("--remote", default="")
    parser.add_argument("--company", default="")
    parser.add_argument("--brightdata-token", default=None)
    parser.add_argument("--num-generations", type=int, default=training_pipeline.NUM_GENERATIONS)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cv-folds", type=int, default=training_pipeline.CV_FOLDS)
    parser.add_argument("--reward-weight-trials", type=int, default=training_pipeline.REWARD_WEIGHT_TRIALS)
    parser.add_argument("--tasks", choices=(training_pipeline.COVER_LETTER_ABILITY, training_pipeline.RESUME_ABILITY), nargs="+", default=[training_pipeline.COVER_LETTER_ABILITY, training_pipeline.RESUME_ABILITY])
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--tune-reward-weights", action="store_true")
    parser.add_argument("--write-cv-folds", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    model_run_dir = run(parse_args())
    print(f"Training manifest/model handoff directory: {model_run_dir}")


if __name__ == "__main__":
    main()
