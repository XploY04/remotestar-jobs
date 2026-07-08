import asyncio
import re
from typing import Any, Dict, List

import aiohttp

from src.agents import BaseFetcher
from src.services.query_planner import QueryPlan, QueryPlannerService
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdzunaFetcher(BaseFetcher):
    """Fetcher for Adzuna API."""

    BASE_URL_TEMPLATE = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    COUNTRIES = ["in", "us", "gb"]
    CATEGORY = "it-jobs"
    MAX_PAGES = 20  # 20 pages × 100 results = 2000 jobs per country
    DETAIL_CONCURRENCY = 10
    PAGE_TEXT_CAP = 30_000  # chars of page text kept for the AI
    PAGE_FETCH_HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    }

    def __init__(self) -> None:
        super().__init__("adzuna")
        self.app_id = settings.adzuna_app_id
        self.app_key = settings.adzuna_api_key
        self.query_plan: QueryPlan | None = None
        if not (self.app_id and self.app_key):
            logger.warning("[%s] Adzuna credentials missing; fetcher will skip", self.source_name)

    def set_query_plan(self, query_plan: QueryPlan) -> None:
        self.query_plan = query_plan

    async def fetch_jobs(self) -> List[Dict[str, Any]]:
        if not (self.app_id and self.app_key):
            return []

        countries = self._countries()
        source_queries = self._source_queries()
        logger.info("[%s] Fetching category jobs from %d countries plus %d planned keyword queries",
                    self.source_name, len(countries), len(source_queries))
        collected: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            for country in countries:
                source_query = self._category_source_query(country)
                country_jobs = await self._fetch_country(session, country)
                for job in country_jobs:
                    job['_query_plan_id'] = self.query_plan.plan_id if self.query_plan else None
                    job['_query'] = source_query.query
                    job['_country'] = source_query.canonical_country
                    job['_source_priority'] = source_query.priority
                collected.extend(country_jobs)
                logger.info("[%s] Collected %d jobs from %s", self.source_name, len(country_jobs), country.upper())
                await asyncio.sleep(0.5)

            for source_query in source_queries:
                query_jobs = await self._fetch_source_query(session, source_query)
                collected.extend(query_jobs)
                await asyncio.sleep(0.2)

            unique_jobs = self._dedupe_jobs(collected)
            await self._fetch_full_descriptions(session, unique_jobs)

        logger.info("[%s] Total unique jobs collected: %d (NO FILTERING - %d before dedupe)",
                    self.source_name, len(unique_jobs), len(collected))
        return unique_jobs

    def _countries(self) -> List[str]:
        if not self.query_plan:
            return self.COUNTRIES

        countries = []
        for query in self.query_plan.queries:
            if query.country not in countries:
                countries.append(query.country)
        return countries or self.COUNTRIES

    def _source_queries(self) -> List:
        if self.query_plan:
            return self.query_plan.queries

        planner = QueryPlannerService()
        return planner._fallback_plan(self.source_name).queries

    def _category_source_query(self, country: str):
        for source_query in self._source_queries():
            if source_query.country == country:
                return source_query
        return self._source_queries()[0]

    async def _fetch_country(
        self,
        session: aiohttp.ClientSession,
        country: str,
        query: str | None = None,
        max_pages: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from a country with pagination - returns RAW data with all fields"""
        all_jobs: List[Dict[str, Any]] = []
        pages = max_pages or self.MAX_PAGES
        
        for page in range(1, pages + 1):
            url = self.BASE_URL_TEMPLATE.format(country=country, page=page)
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": 100,  # Maximum allowed by Adzuna API
                "category": self.CATEGORY,
                "sort_by": "date",
            }
            if query:
                params["what"] = query

            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.error("[%s] HTTP %s for %s page %d", self.source_name, response.status, country, page)
                        break
                    payload = await response.json()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("[%s] Request error for %s page %d: %s", self.source_name, country, page, exc)
                break

            results = payload.get("results", [])
            if not results:
                logger.info("[%s] No more results for %s at page %d", self.source_name, country, page)
                break
            
            # Return RAW data with ALL fields - no normalization here
            # Add metadata for context
            for job in results:
                job['_adzuna_country'] = country  # Add country context
                job['_adzuna_page'] = page  # Add page context for debugging
                if query:
                    job['_adzuna_query'] = query
                all_jobs.append(job)
            
            logger.debug("[%s] Fetched %d jobs from %s page %d", self.source_name, len(results), country, page)
            
            # Small delay between pages to avoid rate limiting
            await asyncio.sleep(0.2)
        
        return all_jobs

    async def _fetch_source_query(self, session: aiohttp.ClientSession, source_query) -> List[Dict[str, Any]]:
        jobs = await self._fetch_country(
            session,
            source_query.country,
            query=source_query.query,
            max_pages=source_query.max_pages,
        )
        for job in jobs:
            job['_query_plan_id'] = self.query_plan.plan_id if self.query_plan else None
            job['_query'] = source_query.query
            job['_country'] = source_query.canonical_country
            job['_source_priority'] = source_query.priority
        return jobs

    async def _fetch_full_descriptions(self, session: aiohttp.ClientSession,
                                       jobs: List[Dict[str, Any]]) -> None:
        """Adzuna's API returns only a description snippet. Fetch each job's
        redirect_url page and attach its text as full_description so the AI
        extraction sees the real listing. Best-effort: redirect targets are
        arbitrary partner sites, so failures keep the snippet."""

        if not settings.enable_adzuna_detail_fetch or not jobs:
            return

        min_len = settings.adzuna_detail_min_description
        targets = [j for j in jobs
                   if j.get("redirect_url") and len(j.get("description") or "") < min_len]
        if not targets:
            return

        logger.info("[%s] Fetching full descriptions for %d snippet-only jobs",
                    self.source_name, len(targets))
        sem = asyncio.Semaphore(self.DETAIL_CONCURRENCY)
        fetched = 0

        async def fill(job: Dict[str, Any]) -> None:
            nonlocal fetched
            async with sem:
                text = await self._fetch_page_text(session, job["redirect_url"])
                if text:
                    job["full_description"] = text
                    fetched += 1
                await asyncio.sleep(0.1)

        await asyncio.gather(*(fill(j) for j in targets))
        logger.info("[%s] Full descriptions fetched: %d/%d", self.source_name, fetched, len(targets))

    async def _fetch_page_text(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20),
                                   headers=self.PAGE_FETCH_HEADERS) as response:
                if response.status != 200:
                    return ""
                html = await response.text(errors="ignore")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("[%s] Page fetch failed for %s: %s", self.source_name, url, exc)
            return ""
        return self._page_to_text(html)

    @classmethod
    def _page_to_text(cls, html: str) -> str:
        if not html:
            return ""
        html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:cls.PAGE_TEXT_CAP]

    def _dedupe_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        unique_jobs: List[Dict[str, Any]] = []

        for job in jobs:
            job_id = job.get("id")
            if job_id:
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)
            unique_jobs.append(job)

        return unique_jobs
