"""Scrape Job Descriptions"""
import sys
from pathlib import Path
import re
import json
from html import unescape
CURR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURR_DIR.parent / "LinkedIn-Scraper" / "free_scraper"))
from jobs_scraper import LinkedInJobsScraper, JobData, ScraperConfig
from typing import Any, List
import requests
from bs4 import BeautifulSoup

KEYWORDS = [
    "AI/ML Engineer",
    "Data Scientist",
    "Software Engineer",
    "Machine Learning Engineer",
    "Data Analyst", 
    "Full Stack Developer",
]
LOCATIONS = [
    "Ottawa CAN",
    "Toronto CAN",
    "Vancouver CAN",
    "London UK",
    "Hong Kong"
]
SAVED_JOBS_DIR = CURR_DIR / "scraped_jobs"

JOB_ID_PATTERN = re.compile(r"/jobs/view/(?:[^/?]+-)?(\d+)")

def scrape_jobs(save_results: bool = True, max_jobs: int = 100) -> List[JobData]:
    """
    Scrape job descriptions from LinkedIn.
    """
    scraper = LinkedInJobsScraper()
    if save_results:
        SAVED_JOBS_DIR.mkdir(exist_ok=True)

    jobs_list: List[JobData] = []
    for keyword in KEYWORDS:
        for location in LOCATIONS:
            print(f"\nScraping jobs for '{keyword}' in '{location}'...")
            try:
                jobs: List[JobData] = scraper.scrape_jobs(
                    keywords=keyword, 
                    location=location, 
                    max_jobs=max_jobs
                )
            except Exception:
                print(f"Error scraping jobs for '{keyword}' in '{location}'. Skipping...")
                jobs = []
            jobs_list.extend(jobs)
            if save_results:
                # Persist
                output_filename = f"{str(SAVED_JOBS_DIR)}/linkedin_jobs_{keyword.replace('/', '_')}_{location.replace(' ', '_')}.json"
                scraper.save_results(jobs, filename=output_filename)
    return jobs_list

def extract_job_descriptions(jobs: List[JobData | dict[str, Any]], save_results: bool = True) -> List[str]:
    """
    Extract job descriptions from scraped jobs.

    BrightData rows already include job_summary/job_description_formatted. UI-scraped
    JobData rows only include job_link, so those need a follow-up request.
    """
    descriptions: list[str] = []
    descriptions_by_url: dict[str, str] = {}
    for job in jobs:
        url = _job_url(job)
        saved_description = _saved_job_description(job)
        if saved_description:
            descriptions.append(saved_description)
            if url:
                descriptions_by_url[url] = saved_description
                _set_job_description(job, saved_description)
            continue

        if not url:
            continue
        print(f"Accessing job description URL: {url}")
        description = fetch_job_description(url)
        descriptions.append(description)
        descriptions_by_url[url] = description
        _set_job_description(job, description)

    if save_results and descriptions_by_url:
        _save_job_descriptions(descriptions_by_url)

    return descriptions

def _job_url(job: JobData | dict[str, Any]) -> str | None:
    if isinstance(job, dict):
        return job.get("job_link") or job.get("url") or job.get("input", {}).get("url")
    return job.job_link

def _saved_job_description(job: JobData | dict[str, Any]) -> str | None:
    if not isinstance(job, dict):
        return None

    if job.get("job_description"):
        return job["job_description"]

    if job.get("job_summary"):
        return job["job_summary"]

    formatted_description = job.get("job_description_formatted")
    if formatted_description:
        soup = BeautifulSoup(unescape(formatted_description), "html.parser")
        return soup.get_text(separator="\n", strip=True)

    return None

def _set_job_description(job: JobData | dict[str, Any], description: str) -> None:
    if isinstance(job, dict):
        job["job_description"] = description
    else:
        setattr(job, "job_description", description)

def _save_job_descriptions(descriptions_by_url: dict[str, str]) -> None:
    """
    Add job_description to the saved JSON records that match the scraped job URLs.
    """
    SAVED_JOBS_DIR.mkdir(exist_ok=True)

    for path in SAVED_JOBS_DIR.glob("*.json"):
        with path.open("r", encoding="utf-8") as file:
            saved_jobs = json.load(file)

        if not isinstance(saved_jobs, list):
            continue

        updated = False
        for saved_job in saved_jobs:
            if not isinstance(saved_job, dict):
                continue

            url = _job_url(saved_job)
            if url in descriptions_by_url:
                saved_job["job_description"] = descriptions_by_url[url]
                updated = True

        if updated:
            with path.open("w", encoding="utf-8") as file:
                json.dump(saved_jobs, file, indent=2, ensure_ascii=False)
                file.write("\n")

def _job_posting_api_url(url: str) -> str:
    """
    Convert a public LinkedIn job URL to the guest job-posting endpoint.
    """
    match = JOB_ID_PATTERN.search(url)
    if not match:
        raise ValueError(f"Could not find a LinkedIn job id in URL: {url}")
    return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{match.group(1)}"

def fetch_job_description(url: str) -> str:
    """
    Fetch and parse the relevant job description text from a LinkedIn job URL.
    """
    api_url = _job_posting_api_url(url)
    response = requests.get(api_url, headers=ScraperConfig.HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    description = soup.find("div", class_="description__text")
    if description is None:
        description = soup.find("section", class_="description")
    if description is None:
        raise RuntimeError(f"Could not find a job description in: {url}")

    return description.get_text(separator="\n", strip=True)

def main(save_results: bool = True):
    # Scrape jobs
    jobs = scrape_jobs(save_results=save_results)
    # Extract job descriptions
    descriptions = extract_job_descriptions(jobs, save_results=save_results)
    return descriptions

if __name__ == "__main__":
    # Run a sample fetch for a specific job description
    jd:str = fetch_job_description("https://ca.linkedin.com/jobs/view/qa-engineer-automation-ai-manual-testing-at-pillway-4366186476")
    print(jd)
    # Now run the full scrape and description extraction from `main`
    main(save_results=True)
