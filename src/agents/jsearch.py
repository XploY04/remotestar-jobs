import asyncio
from typing import Any, Dict, List

import aiohttp

from src.agents import BaseFetcher
from src.services.query_planner import QueryPlan, QueryPlannerService
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JSearchFetcher(BaseFetcher):
    """Fetcher for JSearch via RapidAPI."""

    BASE_URL = "https://jsearch.p.rapidapi.com/search"
    
    # Broad IT/tech queries to capture ALL tech jobs
    QUERIES = [
        "software engineer",
        "software developer",
        "backend engineer",
        "backend developer",
        "frontend engineer",
        "frontend developer",
        "full stack developer",
        "devops engineer",
        "site reliability engineer",
        "cloud engineer",
        "platform engineer",
        "data engineer",
        "data scientist",
        "machine learning engineer",
        "AI engineer",
        "mobile developer",
        "iOS developer",
        "android developer",
        "QA engineer",
        "test engineer",
        "security engineer",
        "cybersecurity",
        "network engineer",
        "systems engineer",
        "database administrator",
        "solutions architect",
        "cloud architect",
        "technical lead",
        "engineering manager",
        "product engineer",
        "embedded engineer",
        "firmware engineer",
        "golang developer",
        "python developer",
        "java developer",
        "rust developer",
        "react developer",
        "node.js developer",
    ]
    
    MAX_PAGES = 5  # Pages per query
    COUNTRIES = ["IN", "US", "GB"]

    def __init__(self) -> None:
        super().__init__("jsearch")
        self.api_key = settings.rapidapi_key
        self.query_plan: QueryPlan | None = None
        if not self.api_key:
            logger.warning("[%s] RAPIDAPI_KEY missing; fetcher will return zero jobs", self.source_name)

    def set_query_plan(self, query_plan: QueryPlan) -> None:
        self.query_plan = query_plan

    async def fetch_jobs(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        source_queries = self._source_queries()
        logger.info("[%s] Fetching IT/tech jobs (%d planned queries)", self.source_name, len(source_queries))
        results: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            for source_query in source_queries:
                query_jobs = await self._fetch_query(session, source_query)
                results.extend(query_jobs)
                await asyncio.sleep(1)  # stay within rate limits

        unique_results = self._dedupe_jobs(results)
        logger.info("[%s] Fetched %d unique raw jobs (%d before dedupe)",
                    self.source_name, len(unique_results), len(results))
        return unique_results

    def _source_queries(self) -> List:
        if self.query_plan:
            return self.query_plan.queries

        planner = QueryPlannerService()
        return planner._fallback_plan(self.source_name).queries

    def _dedupe_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        unique_jobs: List[Dict[str, Any]] = []

        for job in jobs:
            job_id = job.get("job_id")
            if job_id:
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)
            unique_jobs.append(job)

        return unique_jobs

    async def _fetch_query(self, session: aiohttp.ClientSession, source_query) -> List[Dict[str, Any]]:
        all_jobs: List[Dict[str, Any]] = []

        for page in range(1, source_query.max_pages + 1):
            params = {
                "query": source_query.query,
                "page": str(page),
                "num_pages": "1",
                "date_posted": "week",
                "country": source_query.country,
            }
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
            }

            try:
                async with session.get(self.BASE_URL, params=params, headers=headers, timeout=30) as response:
                    if response.status == 429:
                        logger.warning("[%s] Rate limited on query '%s' page %d, stopping query", 
                                      self.source_name, source_query.query, page)
                        break
                    if response.status != 200:
                        logger.error("[%s] HTTP %s for query '%s' page %d", 
                                    self.source_name, response.status, source_query.query, page)
                        break

                    payload = await response.json()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("[%s] Request error for query '%s' page %d: %s", 
                            self.source_name, source_query.query, page, exc, exc_info=True)
                break

            data = payload.get("data", [])
            if not data:
                break
            
            # Return raw data with query context
            for job in data:
                job['_query_plan_id'] = self.query_plan.plan_id if self.query_plan else None
                job['_query'] = source_query.query
                job['_country'] = source_query.canonical_country
                job['_source_priority'] = source_query.priority
                job['_jsearch_query'] = source_query.query
                job['_jsearch_page'] = page
                all_jobs.append(job)
            
            logger.debug("[%s] Fetched %d jobs for '%s' page %d", self.source_name, len(data), source_query.query, page)
            await asyncio.sleep(0.5)  # Delay between pages

        return all_jobs
