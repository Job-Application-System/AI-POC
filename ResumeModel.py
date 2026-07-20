from pathlib import Path
from typing import Iterable

from openai import OpenAI


class ResumeModel:

    def __init__(self):
        # Set up the LLM
        # https://huggingface.co/tencent/Hy3
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

    def setup_skills(self):
        self.skills: dict = {
            "Resume Tailoring": """
                Tailor the resume to the job description while preserving the
                applicant's factual background. Emphasize relevant skills,
                responsibilities, outcomes, tools, and domain keywords already
                supported by the resume or LinkedIn profile.
            """,
            "Factuality": """
                Do not invent employers, degrees, certifications, dates, titles,
                metrics, technologies, or responsibilities. Rewrite, reorder, and
                emphasize existing facts instead of fabricating new ones.
            """,
            "ATS Optimization": """
                Match important keywords from the job description where they are
                truthful for the applicant. Keep the resume concise, scannable,
                and suitable for applicant tracking systems.
            """,
            "Resume's Nature": """
                An advice to resume tailoring is that 
                a resume usually mentions how the job applicant applied hard **technical** skills 
                from his/her responsibilities and experiences, 
                whereas a cover letter details soft skills (like communication, problem-solving, and etc.). 
                Remember not to write anything that does not align with the facts 
                either stated on the resume or the linkedIn profile.            
            """,
            "Thought Process": """
                Follow this framework in your thought process:

                Match **keywords in the job description** provided. 
                Once you drafted the your response (e.g., this resume), check the followings:
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

    def generate_resume(self, job_description, resume, user_profile, company_profile):
        # Generate a tailored resume using the LLM
        self.setup_skills()
        self.system_prompt = f"""
            You are a professional resume writer helping an applicant submit a
            job application. You have access to the applicant's original resume
            and LinkedIn profile.

            Original resume:
            {resume}

            Applicant LinkedIn profile:
            {user_profile}
        """

        self.user_prompt = f"""
            Customize the resume for the following job description:
            {job_description}

            Additionally, consider the company's profile and values:
            {company_profile}

            Return only the tailored resume without any additional text or explanations.
        """

        self.skills_prompt = f"""
            Skills to consider:
            {[f"{key}: {value}" for key, value in self.skills.items()]}
        """

        response = self.client.chat.completions.create(
            model="hy3",
            messages=[
                {
                    "role": "system",
                    "content": f"{self.system_prompt}\n\n{self.skills_prompt}",
                },
                {
                    "role": "user",
                    "content": self.user_prompt,
                },
            ],
            temperature=0.8,
            top_p=1.0,
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
        output_dir: Path | str = Path(__file__).resolve().parent / "training_data" / "resume_hy3",
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
    ):
        from training_pipeline import RESUME_ABILITY, train_model_task

        return train_model_task(
            RESUME_ABILITY,
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
        )

    def tune(
        self,
        jobs_dir: Path | str = Path(__file__).resolve().parent / "scraped_jobs",
        resume: Path | str = Path(__file__).resolve().parent / "sample_resumes",
        user_profile: Path | str = Path(__file__).resolve().parent / "scraped_profiles",
        company_profiles: Iterable[Path | str] | None = None,
        output_dir: Path | str = Path(__file__).resolve().parent / "training_data" / "resume_hy3",
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
        submit: bool = False,
    ):
        from training_pipeline import RESUME_ABILITY, train_model_task

        return train_model_task(
            RESUME_ABILITY,
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
        )
