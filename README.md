# Scraping Jobs for Mass Applications

## Procedures
1. Scrape relevant job postings.
    - For the free LinkedIn UI flow, run `scrape_job_descriptions.py` after adjusting `KEYWORDS` and `LOCATIONS` near the top of the file. It uses [LinkedIn-Scraper](https://github.com/luminati-io/LinkedIn-Scraper) to fetch and persist basic job records:
    ```python
    class JobData:
        title: str
        company: str
        location: str
        job_link: str
        posted_date: str
        scraped_by: str = "UI (headless)"
        scraped_date: Optional[str] = None
    ```
    - For the paid BrightData flow, run the repository-level BrightData scraper by keyword when richer job metadata is needed. Keep country codes in two-letter form, such as `CA` or `US`.
    - Job JSON files should land under `AI-POC/scraped_jobs/`, generally named like `linkedin_jobs_<keywords>_<location>.json`.

2. Enrich jobs with full descriptions.
    - `scrape_job_descriptions.py` calls the LinkedIn guest job-posting endpoint for UI-scraped records and stores `job_description` back into the saved JSON.
    - BrightData records may already include `job_description_formatted` or `job_summary`; the pipeline normalizes those into training text.
    - Training ignores rows that do not contain a usable description field.

3. Prepare applicant and profile context.
    - Put the fixed applicant resume in `AI-POC/sample_resumes/`.
    - Put the user profile in `AI-POC/scraped_profiles/`, or pass a specific profile file with `--user-profile`.
    - Optionally scrape person/company profiles with `profile_scraper.py` or pass company profile JSON files with `--company-profiles`.
    - Directory inputs use the latest modified supported file, so repeated experiments can drop in newer context without changing commands.

4. Build the Hy3/verl training dataset.
    - Run `training_pipeline.py` against the scraped jobs directory to create train/validation parquet files.
    - By default, it creates both `cover_letter` and `resume` rows for each job application so both behaviors are trained together.
    - Add `--generate-candidates` to ask the local Hy3 server for weak-supervision candidates, and `--tune-reward-weights` to select similarity reward weights with cross validation.

5. Launch RL training or save the launcher.
    - The pipeline writes `train.parquet`, `val.parquet`, and `launch_hy3_grpo.sh` under `AI-POC/training_data/cover_letter_hy3/`.
    - Inspect the launcher before running it, or pass `--submit` to submit the Ray/verl job directly.
    - Collective training uses `cover_letter_reward.py`, which branches on each row's `task_type`. Resume-only runs can use `resume_reward.py`.

6. Use the scrape-and-train wrapper for minimal end-to-end experiments.
    - `scrape_and_train.py` can scrape jobs/profiles first, then delegate to `training_pipeline.py`.
    - If `--keyword` or `--location` are omitted, it loops over this file's scraper defaults from `scrape_job_descriptions.py`: `KEYWORDS` and `LOCATIONS`.
    - Run every configured keyword/location pair:
    ```powershell
    uv run python AI-POC\scrape_and_train.py --jobs-dir AI-POC\scraped_jobs
    ```
    - Run one keyword across all configured locations:
    ```powershell
    uv run python AI-POC\scrape_and_train.py --keyword "Data Scientist" --jobs-dir AI-POC\scraped_jobs
    ```
    - Run all configured keywords for one location:
    ```powershell
    uv run python AI-POC\scrape_and_train.py --location "Toronto CAN" --jobs-dir AI-POC\scraped_jobs
    ```
    - Repeat either option to run selected combinations:
    ```powershell
    uv run python AI-POC\scrape_and_train.py --keyword "AI/ML Engineer" --keyword "Data Scientist" --location "Toronto CAN" --location "Vancouver CAN"
    ```
    - Use `--skip-scrape` when reusing existing `AI-POC/scraped_jobs/*.json` files.
    - Training manifests and launcher copies are archived under `AI-POC/trained_models/job_application_hy3/`; after a Hy3/verl run finishes, pass `--checkpoint-dir <path>` to copy final checkpoints there.

7. Keep the workflow reusable for later evaluation.
    - The scraper/profile-loading pieces are intentionally separate from training so they can later feed a constantly updated Google Sheet of job postings.
    - That later evaluation path should reuse the same job/profile schema to generate actual resumes and cover letters without duplicating training-only code.

## Training our Models
`tencent/Hy3 reinforcement learning` uses **GRPO (Group Relative Policy Optimization)** implemented via the [verl](https://huggingface.co/tencent/Hy3/blob/main/rl/README.md#33) framework.

GRPO is an **on-policy RL** algorithm. In this setup, trajectories (rollouts) are generated directly by the current policy using [vLLM](https://huggingface.co/tencent/Hy3/blob/main/rl/README.md#36), and the advantages are computed using group-relative baselines across those generated samples before updating the actor model.

The use of an IcePop correction (`algorithm.rollout_correction.rollout_is`) specifically accounts for any numeric gap or log-prob mismatch between the rollout engine and the training engine, but the core training paradigm remains on-policy policy optimization.

We have 2 models trained collectively: 
1. [CoverLetterModel](./CoverLetterModel.py)
2. [ResumeModel](./ResumeModel.py)

### Goal
The training pipeline optimizes two application-writing behaviors:

1. `cover_letter`: generate a tailored cover letter from the fixed resume, user profile, company profile, and job description.
2. `resume`: generate a tailored resume from the same context.

The user profile and company profile are only generation context. The reward compares the job description against the final application materials:

```text
job_description VS tailored_resume + cover_letter
```

For cover-letter rollouts, the generated text is treated as the cover letter. For resume rollouts, the generated text is treated as the tailored resume.

### Important Files
- `CoverLetterModel.py`: local Hy3 client and prompts for cover-letter generation.
- `ResumeModel.py`: local Hy3 client and prompts for resume tailoring.
- `training_pipeline.py`: builds Hy3/verl-compatible datasets, optional candidate generations, 5-fold reward-weight tuning, and launch commands.
- `cover_letter_reward.py`: custom verl reward function used by collective training and cover-letter runs.
- `resume_reward.py`: resume-specific reward entrypoint for resume-only runs; it reuses the shared application scoring logic.
- `SimilarityModel.py`: similarity metrics used by the reward function.
- `runtime_env.yaml`: Ray runtime environment used when submitting Hy3/verl jobs.

### Dataset Inputs
Place or generate the following inputs before training:

- Scraped jobs: `scraped_jobs/*.json`
- Fixed resume: latest modified supported file in `sample_resumes/` by default
- User profile: latest modified supported file in `scraped_profiles/` by default
- Optional company profile JSON files passed with `--company-profiles`

Resume and profile paths can point either to a specific file or to a directory.
When a directory is provided, `training_pipeline.py` chooses the latest modified
supported file. Supported extensions are `.pdf`, `.doc`, `.docx`, `.txt`, and `.json`.

Scraped job rows should contain one of:

- `job_description`
- `job_description_formatted`
- `job_summary`
- `description`
- `summary`

### Prepare Training Data
From the repository root:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs
```

By default, this writes:

- `AI-POC/training_data/cover_letter_hy3/train.parquet`
- `AI-POC/training_data/cover_letter_hy3/val.parquet`
- `AI-POC/training_data/cover_letter_hy3/launch_hy3_grpo.sh`

The parquet rows follow Hy3/verl style:

- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

For a scrape-plus-training handoff in one command, use `scrape_and_train.py` as
described in the procedure list above.

### Train One or Both Tasks
Prepare both models, which is the default:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --tasks cover_letter resume
```

This calls `training_pipeline.train_models_collectively()` and creates one
`cover_letter` row plus one `resume` row per job application.

Prepare only cover-letter training:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --tasks cover_letter
```

Prepare only resume training:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --tasks resume
```

The model classes also expose task-specific `train()` wrappers for fine-tuning orchestration:

```python
from CoverLetterModel import CoverLetterModel
from ResumeModel import ResumeModel

CoverLetterModel().train(
    generate_candidates=True,
    tune_reward_weights=True,
    write_cv_folds=True,
)

ResumeModel().train(
    generate_candidates=True,
    tune_reward_weights=True,
    write_cv_folds=True,
)
```

Those methods call the shared `training_pipeline.train_model_task()` helper.

Each class also exposes a shorthand `tune()` method for candidate generation plus
5-fold reward-weight tuning:

```python
CoverLetterModel().tune(num_generations=100, cv_folds=5)
ResumeModel().tune(num_generations=100, cv_folds=5)
```

### Candidate Generation
To generate candidate outputs with the local Hy3 server before building the RL dataset:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --generate-candidates --num-generations 100
```

`NUM_GENERATIONS` controls how many candidate cover letters or resumes are generated for each job. The candidates are scored locally and used as weak supervision hints in `reward_model.ground_truth`.

Make sure Hy3 is being served at:

```text
http://127.0.0.1:8000/v1
```

### 5-Fold Reward-Weight Tuning
The pipeline supports a lightweight 5-fold cross validation step for reward coefficients:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --generate-candidates --tune-reward-weights --write-cv-folds
```

This tunes the weighted coefficients for the individual similarity components
using Optuna-style numeric `trial.suggest_float(...)` suggestions:

- `similarity_from_occurrences`
- `similarity_from_num_of_words`
- `similarity_ratio`
- `similarity_by_character`
- `similarity_by_commonality`
- `similarity_by_cosine`

`evaluateFinalScore()` remains the collective score and is included with a fixed coefficient. The selected component weights are saved to:

```text
AI-POC/training_data/cover_letter_hy3/cv_reward_weights.json
```

The outer reward blend defaults to `weighted_similarity=0.75`,
`length_reward=0.10`, `format_reward=0.10`, and `candidate_match=0.05`. Those
blend weights are recorded in each row's ground truth as `reward_blend_weights`
and can be overridden by a custom dataset, but they are not tuned by the current
cross-validation helper.

Control the number of numeric suggestions with:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --generate-candidates --tune-reward-weights --reward-weight-trials 50
```

The `--write-cv-folds` flag also writes fold datasets:

```text
AI-POC/training_data/cover_letter_hy3/fold_0/train.parquet
AI-POC/training_data/cover_letter_hy3/fold_0/val.parquet
...
AI-POC/training_data/cover_letter_hy3/fold_4/train.parquet
AI-POC/training_data/cover_letter_hy3/fold_4/val.parquet
```

Train/validation and CV splits are grouped by job application. The `cover_letter`
and `resume` rows for the same job, plus any candidates derived from that job,
stay in the same split to avoid prompt-level data leakage.

Full RL cross validation is possible by launching one training run per fold, but it is expensive because GRPO is on-policy and each fold requires separate rollouts and model updates.

### Launch Hy3/verl Training
After preparing data, inspect or run the generated launcher:

```powershell
Get-Content AI-POC\training_data\cover_letter_hy3\launch_hy3_grpo.sh
```

The command is generated in the Hy3 style:

```text
TRAIN_FILES=<train.parquet> VAL_FILES=<val.parquet> ray job submit --no-wait --runtime-env=runtime_env.yaml -- bash <run_hy_v3_megatron.sh> reward.custom_reward_function.path=<cover_letter_reward.py> reward.custom_reward_function.name=compute_score
```

To submit directly from the pipeline:

```powershell
uv run python AI-POC\training_pipeline.py --jobs-dir AI-POC\scraped_jobs --submit
```

### Reward Function
The reward compares each generated artifact against the job/application context.
Collective runs use `cover_letter_reward.py`; resume-only launch configs can use
`resume_reward.py`.

For `cover_letter` rows:

```text
solution_str = generated cover letter
reward input = job_description VS resume + generated cover letter
```

For `resume` rows:

```text
solution_str = generated tailored resume
reward input = job_description VS generated tailored resume + paired cover letter
```

The reward returns a dictionary so verl can log the scalar reward and component metrics:

- `score`
- `weighted_similarity`
- `final_score`
- all individual similarity components
- `length_reward`
- `format_reward`
- `candidate_match`
