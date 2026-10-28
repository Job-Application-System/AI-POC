from pathlib import Path
from typing import Iterable

from document_layout import conciseness_prompt
from openai import OpenAI
from prompt_tokens import load_tokenizer, log_prompt_tokens

class CoverLetterModel:

    def __init__(self):
        # Set up the LLM
        # https://huggingface.co/tencent/Hy3
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
        self.tokenizer = load_tokenizer()

    def setup_skills(self):
        self.skills: dict = {
            "Conciseness": conciseness_prompt("cover_letter"),
            "Cover Letter's Nature": """
                An advice to cover letter tailoring is that 
                a cover letter usually mentions how the job applicant applied soft skills 
                (like communication, problem-solving, and etc.) 
                in his/her previous experiences, 
                whereas a resume detailed technical skills and experiences. 
                Remember not to write anything that does not align with the facts 
                either stated on the resume or the linkedIn profile.            
            """,
            "Thought Process": """
                Follow this framework in your thought process:

                Match **keywords in the job description** provided. 
                Once you drafted the your response (e.g., the cover letter), check the followings:
                (1) validating grammatical errors, and 
                (2) two similarity scores (percentage %) based on a common technique used in ATS (Applicant Tracking System) systems (e.g., cosine similarity). 
                One for just the comparison between your generated document versus the job description (wherever applicable). 
                Another score for the comparison between the resume + cover letter versus the job description. 
                You may need to check all attached documents. 
                Try to optimize the percentage scores before returning a response. 
            """,
            "Structural Writing": """
                Try to outline a structured answer from the user's experiences based on this framework: 
                Situation-Target/Task-Achievment/Action-Results (STAR). 
            """
        }
    
    def generate_cover_letter(
        self,
        job_description,
        resume,
        user_profile,
        company_profile,
        temperature: float = 0.9,
        top_p: float = 1.0,
        verbose: bool = False,
    ):
        # Generate a cover letter using the LLM
        self.setup_skills()
        self.system_prompt = f"""
            You are a professional cover letter writer helping an applicant to submit 
            a job application. You have access to the applicant's resume and LinkedIn profile.

            Refer to a generalized version of resume:
            {resume}

            Also refer to the applicant's LinkedIn profile for additional context:
            {user_profile}
        """

        self.user_prompt = f"""
            Generate a tailored cover letter for the following job description:
            {job_description}

            Additionally, consider the company's profile and values:
            {company_profile}

            Return only the cover letter without any additional text or explanations.
        """

        self.skills_prompt = f"""
            Skills to consider:
            {[f"{key}: {value}" for key, value in self.skills.items()]}
        """

        messages = [
                {
                    "role": "system", 
                    "content": f"{self.system_prompt}\n\n{self.skills_prompt}"
                },
                {
                    "role": "user", 
                    "content": self.user_prompt
                }
            ]
        if verbose:
            log_prompt_tokens(self.tokenizer, messages, "Cover letter")
        response = self.client.chat.completions.create(
            model="hy3",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            # reasoning_effort: "no_think" (default, direct response), "low", "high" (deep chain-of-thought)
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
        )
        print(f"Number of Tokens used: {response.usage.total_tokens}")
        return response.choices[0].message.content

    def train(
        self,
        jobs_dir: Path | str = Path(__file__).resolve().parent / "scraped_jobs",
        resume: Path | str = Path(__file__).resolve().parent / "sample_resumes",
        user_profile: Path | str = Path(__file__).resolve().parent / "scraped_profiles",
        company_profiles: Iterable[Path | str] | None = None,
        output_dir: Path | str = Path(__file__).resolve().parent / "training_data" / "cover_letter_hy3",
        verl_dir: Path | str = Path(__file__).resolve().parent / "verl",
        reward_path: Path | str = Path(__file__).resolve().parent / "cover_letter_reward.py",
        generate_candidates: bool = False,
        tune_reward_weights: bool = False,
        write_cv_folds: bool = False,
        submit: bool = False,
        num_generations: int = 100,
        max_jobs: int | None = None,
        val_ratio: float = 0.1,
        seed: int = 7,
        cv_folds: int = 5,
        reward_weight_trials: int = 25,
        ray_address: str | None = None,
        checkpoint_dir: Path | str | None = None,
        resume_from_checkpoint: Path | str | None = None,
    ):
        from training_pipeline import COVER_LETTER_ABILITY, train_model_task

        return train_model_task(
            COVER_LETTER_ABILITY,
            jobs_dir=Path(jobs_dir),
            resume=Path(resume),
            user_profile=Path(user_profile),
            company_profiles=[Path(path) for path in company_profiles or []],
            output_dir=Path(output_dir),
            verl_dir=Path(verl_dir),
            reward_path=Path(reward_path),
            generate_candidates=generate_candidates,
            tune_reward_weights=tune_reward_weights,
            write_cv_folds=write_cv_folds,
            submit=submit,
            num_generations=num_generations,
            max_jobs=max_jobs,
            val_ratio=val_ratio,
            seed=seed,
            cv_folds=cv_folds,
            reward_weight_trials=reward_weight_trials,
            ray_address=ray_address,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            resume_from_checkpoint=Path(resume_from_checkpoint) if resume_from_checkpoint else None,
        )

    def tune(
        self,
        jobs_dir: Path | str = Path(__file__).resolve().parent / "scraped_jobs",
        resume: Path | str = Path(__file__).resolve().parent / "sample_resumes",
        user_profile: Path | str = Path(__file__).resolve().parent / "scraped_profiles",
        company_profiles: Iterable[Path | str] | None = None,
        output_dir: Path | str = Path(__file__).resolve().parent / "training_data" / "cover_letter_hy3",
        verl_dir: Path | str = Path(__file__).resolve().parent / "verl",
        reward_path: Path | str = Path(__file__).resolve().parent / "cover_letter_reward.py",
        generate_candidates: bool = True,
        tune_reward_weights: bool = True,
        write_cv_folds: bool = True,
        num_generations: int = 100,
        max_jobs: int | None = None,
        val_ratio: float = 0.1,
        seed: int = 7,
        cv_folds: int = 5,
        reward_weight_trials: int = 25,
        ray_address: str | None = None,
        checkpoint_dir: Path | str | None = None,
        resume_from_checkpoint: Path | str | None = None,
        submit: bool = False,
    ):
        from training_pipeline import COVER_LETTER_ABILITY, train_model_task

        return train_model_task(
            COVER_LETTER_ABILITY,
            jobs_dir=Path(jobs_dir),
            resume=Path(resume),
            user_profile=Path(user_profile),
            company_profiles=[Path(path) for path in company_profiles or []],
            output_dir=Path(output_dir),
            verl_dir=Path(verl_dir),
            reward_path=Path(reward_path),
            generate_candidates=generate_candidates,
            tune_reward_weights=tune_reward_weights,
            write_cv_folds=write_cv_folds,
            submit=submit,
            num_generations=num_generations,
            max_jobs=max_jobs,
            val_ratio=val_ratio,
            seed=seed,
            cv_folds=cv_folds,
            reward_weight_trials=reward_weight_trials,
            ray_address=ray_address,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            resume_from_checkpoint=Path(resume_from_checkpoint) if resume_from_checkpoint else None,
        )

def main():
    NUM_GENERATIONS = 1  # Number of cover letters to generate for each job
    import os, json
    scraped_jobs = [
        file for file in os.listdir("scraped_jobs") 
        if os.path.isfile(os.path.join("scraped_jobs", file))
        and file.endswith(".json")
    ]
    generated: int = 0
    for job_file in scraped_jobs:
        job_path = os.path.join("scraped_jobs", job_file)
        with open(job_path, "r") as f:
            job_data = json.load(f)
        
        job_description = job_data.get("job_description", "")
        resume = job_data.get("resume", "")
        user_profile = job_data.get("user_profile", "")
        company_profile = job_data.get("company_profile", "")

        cover_letter_model = CoverLetterModel()
        cover_letter = cover_letter_model.generate_cover_letter(
            job_description, resume, user_profile, company_profile
        )
        
        print(f"Generated Cover Letter for {job_file}:\n{cover_letter}\n")
        generated += 1
        if generated >= NUM_GENERATIONS:
            break
