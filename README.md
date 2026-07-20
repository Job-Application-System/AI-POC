# Scraping Jobs for Mass Applications

## Procedures
1. Run `scrape_job_descriptions.py` to scrape a massive list of jobs from LinkdIn. Adjust `KEYWORDS` and `LOCATIONS` lists at the start of the file. 
    - It utilizes [LinkedIn-Scraper](https://github.com/luminati-io/LinkedIn-Scraper) to fetch and persist JSON files containing a job's basic information, as follows: 
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
    - Now we only choose the free scrapping method from the UI in headless mode. Alternatively, you can also choose to apply an API-based approach which scrapes jobs more robustly and with enriched information from BrightData's endpoints. That method needs to be paid.
    - Results are saved in JSON files under `scraped_jobs/` directory, annotated by the following convention: `linkedin_jobs_<keywords>_<location>.json`. 
    - A string of the corresponding job description from the link is also fetched and persisted into the corresponding JSON file.
