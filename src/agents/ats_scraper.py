"""
ATS Scraper — fetches live jobs from 5 ATS platform APIs using
company slugs stored in the discovered_companies DB table.

Supported platforms:
  1. Greenhouse  — boards-api.greenhouse.io (JSON API)
  2. Lever       — api.lever.co (JSON API)
  3. Ashby       — jobs.ashbyhq.com (API via POST)
  4. Workable    — apply.workable.com/api/v3 (JSON API)
  5. SmartRecruiters — api.smartrecruiters.com (JSON API)
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from src.agents import BaseFetcher
from src.services.query_planner import QueryPlan
from src.services.company_discovery import CompanyDiscoveryService
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Max jobs per company (avoid huge boards like Google blowing up AI cost)
MAX_JOBS_PER_COMPANY = 50
# Concurrent requests per platform
CONCURRENCY = 5
# Concurrent detail fetches per company
DETAIL_CONCURRENCY = 5


class ATSScraperFetcher(BaseFetcher):
    """
    Reads company slugs from discovered_companies DB table,
    then hits each ATS platform's public API for live job data.
    """

    def __init__(self) -> None:
        super().__init__("ats_scraper")
        self.discovery = CompanyDiscoveryService()

    def set_query_plan(self, query_plan: QueryPlan) -> None:
        self.discovery.set_query_plan(query_plan)

    async def fetch_jobs(self) -> List[Dict[str, Any]]:
        """Fetch jobs from all 5 ATS platforms."""
        logger.info("[%s] Starting ATS scraper run", self.source_name)

        # Step 1: Run discovery (seeds + Google Search)
        await self.discovery.run_discovery()

        all_jobs: List[Dict[str, Any]] = []

        # Step 2: Scrape each platform
        platform_scrapers = {
            "greenhouse": self._scrape_greenhouse,
            "lever": self._scrape_lever,
            "ashby": self._scrape_ashby,
            "workable": self._scrape_workable,
            "smartrecruiters": self._scrape_smartrecruiters,
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "JobsAI/1.0 (job aggregator)"},
        ) as session:
            for platform, scraper_fn in platform_scrapers.items():
                companies = await self.discovery.get_companies_for_platform(platform)

                if not companies:
                    logger.info("[%s] No companies for %s", self.source_name, platform)
                    continue

                logger.info(
                    "[%s] Scraping %d %s companies",
                    self.source_name, len(companies), platform,
                )

                # Scrape with concurrency limit
                semaphore = asyncio.Semaphore(CONCURRENCY)
                tasks = [
                    self._scrape_with_semaphore(
                        semaphore, scraper_fn, session, c["slug"], c.get("company_name", c["slug"])
                    )
                    for c in companies
                ]
                results = await asyncio.gather(*tasks)

                platform_total = 0
                for slug, jobs in zip([c["slug"] for c in companies], results):
                    platform_total += len(jobs)
                    all_jobs.extend(jobs)
                    # Update company metadata
                    await self.discovery.mark_company_fetched(platform, slug, len(jobs))

                logger.info(
                    "[%s] %s: %d jobs from %d companies",
                    self.source_name, platform, platform_total, len(companies),
                )

        logger.info(
            "[%s] Total: %d jobs from all ATS platforms",
            self.source_name, len(all_jobs),
        )
        return all_jobs

    async def _scrape_with_semaphore(self, sem, fn, session, slug, company_name):
        """Wrap scraper call with semaphore + error handling."""
        async with sem:
            try:
                jobs = await fn(session, slug, company_name)
                await asyncio.sleep(0.3)  # Rate limit between companies
                return jobs
            except Exception as exc:
                logger.error("[%s] Error scraping %s: %s", self.source_name, slug, exc)
                return []

    async def _enrich_descriptions(self, session, jobs, fetch_one):
        """For jobs whose list response omitted the description, hit the
        platform-specific detail endpoint to grab the full text.

        Called per-company so we can cap total detail calls and bail out
        cleanly if a single company has hundreds of postings.
        """
        if not settings.enable_ats_detail_fetch or not jobs:
            return

        min_len = settings.ats_detail_min_description
        targets = [j for j in jobs if len(j.get("description") or "") < min_len]
        if not targets:
            return

        sem = asyncio.Semaphore(DETAIL_CONCURRENCY)

        async def fill(job):
            async with sem:
                try:
                    full = await fetch_one(session, job)
                    if full:
                        job["description"] = self._strip_html(full)[:8000]
                except Exception as exc:
                    logger.debug("[%s] Detail fetch failed for %s: %s",
                                 self.source_name, job.get("source_id"), exc)
                await asyncio.sleep(0.1)

        await asyncio.gather(*(fill(j) for j in targets))

    # ------------------------------------------------------------------
    # Greenhouse  (boards-api.greenhouse.io)
    # ------------------------------------------------------------------

    async def _scrape_greenhouse(
        self, session: aiohttp.ClientSession, slug: str, company_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from Greenhouse public API."""
        url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"

        async with session.get(url) as resp:
            if resp.status == 404:
                await self.discovery.mark_company_inactive("greenhouse", slug)
                return []
            if resp.status != 200:
                return []
            data = await resp.json()

        jobs = []
        for item in data.get("jobs", [])[:MAX_JOBS_PER_COMPANY]:
            location_name = item.get("location", {}).get("name", "") if isinstance(item.get("location"), dict) else ""
            description = item.get("content", "") or ""

            jobs.append({
                "source": self.source_name,
                "source_id": f"gh_{slug}_{item['id']}",
                "title": item.get("title", ""),
                "company": company_name,
                "description": self._strip_html(description)[:8000],
                "location": {
                    "city": location_name if location_name else None,
                    "country": None,
                    "remote": bool(re.search(r"\bremote\b", location_name, re.I)),
                },
                "employment_type": "FULLTIME",
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "apply_url": item.get("absolute_url", f"https://boards.greenhouse.io/{slug}/jobs/{item['id']}"),
                "posted_at": self._parse_iso_date(item.get("updated_at")),
                "raw_data": {"ats": "greenhouse", "slug": slug, "job_id": item["id"]},
                "_gh_id": item["id"],
                "_gh_slug": slug,
            })

        async def fetch_gh_detail(s, job):
            url = f"https://boards-api.greenhouse.io/v1/boards/{job['_gh_slug']}/jobs/{job['_gh_id']}"
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                d = await r.json()
                return d.get("content", "")

        await self._enrich_descriptions(session, jobs, fetch_gh_detail)
        for j in jobs:
            j.pop("_gh_id", None)
            j.pop("_gh_slug", None)
        return jobs

    # ------------------------------------------------------------------
    # Lever  (api.lever.co)
    # ------------------------------------------------------------------

    async def _scrape_lever(
        self, session: aiohttp.ClientSession, slug: str, company_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from Lever public API."""
        url = f"https://api.lever.co/v0/postings/{slug}?mode=json&limit={MAX_JOBS_PER_COMPANY}"

        async with session.get(url) as resp:
            if resp.status == 404:
                await self.discovery.mark_company_inactive("lever", slug)
                return []
            if resp.status != 200:
                return []
            data = await resp.json()

        if not isinstance(data, list):
            return []

        jobs = []
        for item in data[:MAX_JOBS_PER_COMPANY]:
            categories = item.get("categories", {}) or {}
            location = categories.get("location", "") or ""
            description_parts = []
            for section in item.get("lists", []):
                description_parts.append(section.get("text", ""))
                description_parts.append(section.get("content", ""))
            description = item.get("descriptionPlain", "") or " ".join(description_parts)

            created_at = item.get("createdAt")
            posted_at = (
                datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
                if created_at
                else datetime.now(timezone.utc)
            )

            jobs.append({
                "source": self.source_name,
                "source_id": f"lv_{slug}_{item['id']}",
                "title": item.get("text", ""),
                "company": company_name,
                "description": self._strip_html(description)[:8000],
                "location": {
                    "city": location if location else None,
                    "country": None,
                    "remote": bool(re.search(r"\bremote\b", location, re.I)),
                },
                "employment_type": self._map_lever_commitment(categories.get("commitment", "")),
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "apply_url": item.get("hostedUrl", f"https://jobs.lever.co/{slug}/{item['id']}"),
                "posted_at": posted_at,
                "raw_data": {"ats": "lever", "slug": slug, "job_id": item["id"]},
                "_lv_id": item["id"],
                "_lv_slug": slug,
            })

        async def fetch_lv_detail(s, job):
            url = f"https://api.lever.co/v0/postings/{job['_lv_slug']}/{job['_lv_id']}?mode=json"
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                d = await r.json()
                parts = []
                for section in d.get("lists", []) or []:
                    parts.append(section.get("text", ""))
                    parts.append(section.get("content", ""))
                return d.get("descriptionPlain") or d.get("description") or " ".join(parts)

        await self._enrich_descriptions(session, jobs, fetch_lv_detail)
        for j in jobs:
            j.pop("_lv_id", None)
            j.pop("_lv_slug", None)
        return jobs

    # ------------------------------------------------------------------
    # Ashby  (jobs.ashbyhq.com API)
    # ------------------------------------------------------------------

    async def _scrape_ashby(
        self, session: aiohttp.ClientSession, slug: str, company_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from Ashby's public posting API (GET endpoint)."""
        url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"

        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    await self.discovery.mark_company_inactive("ashby", slug)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        # Ashby returns {jobs: [{id, title, location, ...}]}
        jobs_data = data.get("jobs", [])
        if not jobs_data:
            return []

        jobs = []
        for item in jobs_data[:MAX_JOBS_PER_COMPANY]:
            location = item.get("location", "") or ""
            is_remote = item.get("isRemote", False) or bool(
                re.search(r"\bremote\b", location, re.I)
            )
            description = item.get("descriptionHtml", "") or item.get("descriptionPlain", "") or ""

            jobs.append({
                "source": self.source_name,
                "source_id": f"ab_{slug}_{item['id']}",
                "title": item.get("title", ""),
                "company": company_name,
                "description": self._strip_html(description)[:8000],
                "location": {
                    "city": location if location else None,
                    "country": None,
                    "remote": is_remote,
                },
                "employment_type": self._map_ashby_type(item.get("employmentType", "")),
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "apply_url": item.get("jobUrl", f"https://jobs.ashbyhq.com/{slug}/{item['id']}"),
                "posted_at": self._parse_iso_date(item.get("publishedAt")),
                "raw_data": {"ats": "ashby", "slug": slug, "job_id": item["id"]},
            })
        return jobs

    # ------------------------------------------------------------------
    # Workable  (apply.workable.com/api/v3)
    # ------------------------------------------------------------------

    async def _scrape_workable(
        self, session: aiohttp.ClientSession, slug: str, company_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from Workable's public API."""
        url = f"https://apply.workable.com/api/v3/accounts/{slug}/jobs"

        try:
            async with session.post(url, json={}) as resp:
                if resp.status == 404:
                    await self.discovery.mark_company_inactive("workable", slug)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        results = data.get("results", [])
        jobs = []
        for item in results[:MAX_JOBS_PER_COMPANY]:
            location = item.get("location", {}) or {}
            city = location.get("city", "") or ""
            country = location.get("country", "") or ""
            is_remote = item.get("telecommuting", False) or bool(
                re.search(r"\bremote\b", f"{city} {country}", re.I)
            )

            shortcode = item.get("shortcode", item.get("id", ""))

            jobs.append({
                "source": self.source_name,
                "source_id": f"wk_{slug}_{shortcode}",
                "title": item.get("title", ""),
                "company": company_name,
                "description": item.get("description", "")[:8000],
                "location": {
                    "city": city if city else None,
                    "country": country if country else None,
                    "remote": is_remote,
                },
                "employment_type": self._map_workable_type(item.get("employment_type", "")),
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "apply_url": item.get("url", f"https://apply.workable.com/{slug}/j/{shortcode}/"),
                "posted_at": self._parse_iso_date(item.get("published_on")),
                "raw_data": {"ats": "workable", "slug": slug, "shortcode": shortcode},
                "_wk_shortcode": shortcode,
                "_wk_slug": slug,
            })

        async def fetch_wk_detail(s, job):
            url = f"https://apply.workable.com/api/v3/accounts/{job['_wk_slug']}/jobs/{job['_wk_shortcode']}"
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                d = await r.json()
                return d.get("description") or d.get("full_description") or ""

        await self._enrich_descriptions(session, jobs, fetch_wk_detail)
        for j in jobs:
            j.pop("_wk_shortcode", None)
            j.pop("_wk_slug", None)
        return jobs

    # ------------------------------------------------------------------
    # SmartRecruiters  (api.smartrecruiters.com)
    # ------------------------------------------------------------------

    async def _scrape_smartrecruiters(
        self, session: aiohttp.ClientSession, slug: str, company_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch jobs from SmartRecruiters public API."""
        url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
        params = {"limit": MAX_JOBS_PER_COMPANY}

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 404:
                    await self.discovery.mark_company_inactive("smartrecruiters", slug)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []

        content = data.get("content", [])
        jobs = []
        for item in content[:MAX_JOBS_PER_COMPANY]:
            location = item.get("location", {}) or {}
            city = location.get("city", "") or ""
            country = location.get("country", "") or ""
            remote_status = location.get("remote", False)

            jobs.append({
                "source": self.source_name,
                "source_id": f"sr_{slug}_{item.get('id', '')}",
                "title": item.get("name", ""),
                "company": company_name or item.get("company", {}).get("name", slug),
                "description": item.get("jobAd", {}).get("sections", {}).get("jobDescription", {}).get("text", "")[:8000] if isinstance(item.get("jobAd"), dict) else "",
                "location": {
                    "city": city if city else None,
                    "country": country if country else None,
                    "remote": bool(remote_status),
                },
                "employment_type": "FULLTIME",
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "apply_url": f"https://jobs.smartrecruiters.com/{slug}/{item.get('id', '')}",
                "posted_at": self._parse_iso_date(item.get("releasedDate")),
                "raw_data": {"ats": "smartrecruiters", "slug": slug, "job_id": item.get("id")},
                "_sr_id": item.get("id"),
                "_sr_slug": slug,
            })

        async def fetch_sr_detail(s, job):
            url = f"https://api.smartrecruiters.com/v1/companies/{job['_sr_slug']}/postings/{job['_sr_id']}"
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                d = await r.json()
                sections = (d.get("jobAd") or {}).get("sections") or {}
                parts = []
                for key in ("jobDescription", "qualifications", "responsibilities", "additionalInformation"):
                    text = (sections.get(key) or {}).get("text") or ""
                    if text:
                        parts.append(text)
                return "\n\n".join(parts)

        await self._enrich_descriptions(session, jobs, fetch_sr_detail)
        for j in jobs:
            j.pop("_sr_id", None)
            j.pop("_sr_slug", None)
        return jobs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags, keep text content."""
        if not html:
            return ""
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _parse_iso_date(value: Optional[str]) -> datetime:
        """Parse ISO date string, fallback to now."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            from dateutil import parser
            return parser.isoparse(value)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _map_lever_commitment(commitment: str) -> str:
        """Map Lever commitment field to our employment_type."""
        if not commitment:
            return "FULLTIME"
        c = commitment.lower()
        if "intern" in c:
            return "INTERN"
        if "part" in c:
            return "PARTTIME"
        if "contract" in c or "temp" in c:
            return "CONTRACT"
        return "FULLTIME"

    @staticmethod
    def _map_ashby_type(emp_type: str) -> str:
        """Map Ashby employment type."""
        if not emp_type:
            return "FULLTIME"
        t = emp_type.lower()
        if "intern" in t:
            return "INTERN"
        if "part" in t:
            return "PARTTIME"
        if "contract" in t:
            return "CONTRACT"
        return "FULLTIME"

    @staticmethod
    def _map_workable_type(emp_type: str) -> str:
        """Map Workable employment type."""
        if not emp_type:
            return "FULLTIME"
        t = emp_type.lower()
        if "intern" in t:
            return "INTERN"
        if "part" in t:
            return "PARTTIME"
        if "contract" in t or "temp" in t:
            return "CONTRACT"
        return "FULLTIME"
