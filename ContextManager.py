from pathlib import Path
import sys, os
import json
from datetime import datetime
from typing import Any

CURR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURR_DIR.parent / "LinkedIn-Scraper" / "free_scraper"))
sys.path.append(str(CURR_DIR.parent / "LinkedIn-Scraper" / "linkedin_scraper_api_codes"))
from scrape_job_descriptions import scrape_jobs, extract_job_descriptions
from profile_scraper import scrape # free method
# Paid methods
from linkedin_profile_by_url import LinkedInProfileInfo
from linkedin_profile_by_name import LinkedInProfileDiscovery
from linkedin_company_info_by_url import LinkedInCompanyInfo
from dotenv import load_dotenv
from linkedin_scraper import RateLimitError, ProfileNotFoundError


class ContextManager:
    def __init__(self):
        load_dotenv()

    async def set_company_context(self) -> list[bool]:
        self.scraped_jobs = scrape_jobs(save_results=True, max_jobs=100)
        self.job_descriptions = extract_job_descriptions(self.scraped_jobs, save_results=True)
        self.company_profiles = []
        rets = []

        for job in self.scraped_jobs:
            company = self._get_job_value(job, "company")
            company_url = f"https://www.linkedin.com/company/{company}"
            company_profile = None
            ret = False

            try:
                company_profile = await scrape(
                    profile_url=company_url,
                    profile_type="company",
                    force_login=False,
                )
                ret = True
            except Exception:
                company_profile = self._collect_paid_company_profile(company_url)
                ret = company_profile is not None

            if company_profile is None:
                company_profile = ""

            self._set_job_value(job, "company_profile", company_profile)
            self.company_profiles.append(company_profile)
            rets.append(ret)

        return rets

    def _collect_paid_company_profile(self, company_url: str) -> dict | list | None:
        try:
            self.collector = LinkedInCompanyInfo(
                api_token=os.getenv("BRIGHTDATA_APIKEY")
            )
            ret = self.collector.collect_company_info(
                company_urls=[{"url": company_url}],
                output_dir=CURR_DIR.parent / "scraper_profiles"
            )
        except Exception:
            self.collector = None
            return None

        if not ret:
            self.collector = None
            return None

        latest_profile_file = None
        latest_profile_time = None
        output_dir = CURR_DIR.parent / "scraper_profiles"
        for profile_file in output_dir.glob("linkedin_company_info_*.json"):
            time_text = profile_file.stem.replace("linkedin_company_info_", "")
            try:
                file_time = datetime.strptime(time_text, "%H:%M:%S").time()
            except ValueError:
                continue
            if latest_profile_time is None or file_time > latest_profile_time:
                latest_profile_time = file_time
                latest_profile_file = profile_file

        if latest_profile_file:
            with latest_profile_file.open("r", encoding="utf-8") as f:
                company_info = json.load(f)
            return company_info[0] if company_info else {}

        return None

    def _get_job_value(self, job: Any, key: str):
        return job.get(key) if isinstance(job, dict) else getattr(job, key)

    def _set_job_value(self, job: Any, key: str, value) -> None:
        if isinstance(job, dict):
            job[key] = value
        else:
            setattr(job, key, value)

    async def set_user_context(self) -> bool:
        ret = False
        output_dir = CURR_DIR.parent / "scraper_profiles"
        try:
            self.user_profile = await scrape(
                profile_url = os.getenv("PROFILE_URL"),
                profile_type = "person",
                force_login = False,
            )
            ret = True
        except (RuntimeError, RateLimitError, ProfileNotFoundError):
            # Try paid BrightData API
            if os.getenv("PROFILE_URL"):
                try:
                    self.collector = LinkedInProfileInfo(
                        api_token=os.getenv("BRIGHTDATA_APIKEY")
                    )
                    ret = self.collector.collect_profile_info(
                        profile_urls=[{"url": os.getenv("PROFILE_URL")}],
                        output_dir=CURR_DIR.parent / "scraper_profiles"
                    )
                except Exception:
                    self.collector = None
                    ret = False
                # Parse the saved JSON file
                if ret:
                    latest_profile_file = None
                    latest_profile_time = None
                    for profile_file in output_dir.glob("profiles_by_url_*.json"):
                        time_text = profile_file.stem.replace("profiles_by_url_", "")
                        try:
                            file_time = datetime.strptime(time_text, "%H:%M:%S").time()
                        except ValueError:
                            continue # Not the latest file, skip this file
                        if latest_profile_time is None or file_time > latest_profile_time:
                            latest_profile_time = file_time
                            latest_profile_file = profile_file

                    if latest_profile_file:
                        with latest_profile_file.open("r", encoding="utf-8") as f:
                            self.user_profile = json.load(f)
                        return ret
                    ret = False
                    self.collector = None

            if not ret and (os.getenv("PROFILE_USER_FIRSTNAME") and os.getenv("PROFILE_USER_LASTNAME")):
                try:
                    self.discoverer = LinkedInProfileDiscovery(
                        api_token=os.getenv("BRIGHTDATA_APIKEY")
                    )
                    ret = self.discoverer.discover_profiles(
                        people=[{
                            "first_name": os.getenv("PROFILE_USER_FIRSTNAME"),
                            "last_name": os.getenv("PROFILE_USER_LASTNAME"),
                        }],
                        output_dir=CURR_DIR.parent / "scraper_profiles"
                    )
                except Exception:
                    self.discoverer = None
                    ret = False
                # Parse the saved JSON file
                if ret:
                    latest_profile_file = None
                    latest_profile_time = None
                    for profile_file in output_dir.glob("profiles_by_name_*.json"):
                        time_text = profile_file.stem.replace("profiles_by_name_", "")
                        try:
                            file_time = datetime.strptime(time_text, "%H:%M:%S").time()
                        except ValueError:
                            continue # Not the latest file, skip it
                        if latest_profile_time is None or file_time > latest_profile_time:
                            latest_profile_time = file_time
                            latest_profile_file = profile_file

                    if latest_profile_file:
                        with latest_profile_file.open("r", encoding="utf-8") as f:
                            self.user_profile = json.load(f)
                        return ret
                    ret = False
                    self.discoverer = None

            if not ret:
                self.user_profile = None
                self.collector = None
                self.discoverer = None

        return ret

    def get_scraped_jobs(self):
        for index, job in enumerate(self.scraped_jobs):
            if index < len(self.company_profiles):
                self._set_job_value(job, "company_profile", self.company_profiles[index])
        return self.scraped_jobs

    def get_user_profile(self):
        return self.user_profile
