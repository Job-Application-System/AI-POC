from openai import OpenAI

class CoverLetterModel:

    def __init__(self):
        # Set up the LLM
        # https://huggingface.co/tencent/Hy3
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

    def setup_skills(self):
        self.skills: dict = {
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
    
    def generate_cover_letter(self, job_description, resume, user_profile, company_profile):
        # Generate a cover letter using the LLM
        self.setup_skills()
        self.base_prompt = f"""
            You are a professional cover letter writer helping an applicant to submit 
            a job application. You have access to the applicant's resume and LinkedIn profile.

            Refer to a generalized version of resume:
            {resume}

            Also refer to the applicant's LinkedIn profile for additional context:
            {user_profile}

            Generate a tailored cover letter for the following job description:
            {job_description}

            Additionally, consider the company's profile and values:
            {company_profile}

            Return only the cover letter without any additional text or explanations.

            Skills to consider:
            {[f"{key}: {value}" for key, value in self.skills.items()]}
        """

        response = self.client.chat.completions.create(
            model="hy3",
            messages=[
                {
                    "role": "system", 
                    "content": self.base_prompt
                },
            ],
            temperature=0.9,
            top_p=1.0,
            # reasoning_effort: "no_think" (default, direct response), "low", "high" (deep chain-of-thought)
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
        )
        print(f"Number of Tokens used: {response.usage.total_tokens}")
        return response.choices[0].message.content

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