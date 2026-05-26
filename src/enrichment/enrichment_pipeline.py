"""Job processing pipeline: Raw API data → Gemini AI → Structured schema.

NO normalizer. Gemini sees the full raw data and extracts everything.
For jobs where AI is disabled or fails, a lightweight fallback extractor runs.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Any, List, Optional

from src.enrichment.ai_processor import AIProcessor, PROMPT_VERSION
from src.enrichment.skills_extractor import SkillsExtractor
from src.enrichment.quality_scorer import QualityScorer
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = setup_logger(__name__)

# Jobs older than this are dropped during ingestion
MAX_JOB_AGE_DAYS = 15


class EnrichmentPipeline:
    """
    The single transformation layer for all job data.
    
    Flow: Raw API data → AI extraction → structured job dict → DB
    
    When AI is enabled:  Raw → Gemini (extracts all 40 fields) → quality score → done
    When AI is disabled: Raw → fallback extractor (basic field mapping) → done
    """

    def __init__(self, use_ai: bool = None):
        self.use_ai = use_ai if use_ai is not None else settings.enable_ai_enrichment
        self.ai_processor = AIProcessor() if self.use_ai else None
        self.skills_extractor = SkillsExtractor()
        self.quality_scorer = QualityScorer()
        self.metrics_by_source: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Pipeline initialized (AI: {'enabled' if self.use_ai else 'disabled'})")

    # ------------------------------------------------------------------
    # Main entry point: process raw jobs from a source
    # ------------------------------------------------------------------

    async def process_source(
        self,
        source_name: str,
        raw_jobs: List[Dict[str, Any]],
        batch_size: int = 5,
        max_concurrent: int = 10,
        on_batch_ready: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process ALL raw jobs from a single source into final structured format.
        
        Uses BATCH AI processing: sends `batch_size` jobs per Gemini call,
        with up to `max_concurrent` batch calls in parallel.
        
        If `on_batch_ready` is provided, each finished batch is passed to it
        immediately (e.g. for saving to DB) instead of accumulating in memory.
        """
        if not raw_jobs:
            return []

        logger.info(f"[{source_name}] Processing {len(raw_jobs)} raw jobs...")

        if self.ai_processor:
            self.ai_processor.reset_metrics(source_name)

        if self.use_ai and self.ai_processor and self.ai_processor.enabled:
            jobs = await self._process_with_ai(source_name, raw_jobs, batch_size, max_concurrent, on_batch_ready)
        else:
            jobs = self._process_with_fallback(source_name, raw_jobs)
            if on_batch_ready and jobs:
                await on_batch_ready(jobs)

        if self.ai_processor and self.ai_processor.enabled:
            metrics = self.ai_processor.get_metrics(source_name)
            self.metrics_by_source[source_name] = metrics
            if metrics["fallback_rate"] > 0.05 and metrics["batch_ok"] + metrics["batch_fallback"] >= 5:
                logger.warning(
                    f"[{source_name}] HIGH fallback_rate={metrics['fallback_rate']:.2%} "
                    f"(ok={metrics['batch_ok']}, fallback={metrics['batch_fallback']})"
                )
        else:
            self.metrics_by_source[source_name] = {"batch_ok": 0, "batch_fallback": 0, "batch_failed": 0, "fallback_rate": 0.0}

        logger.info(f"[{source_name}] Processed: {len(jobs)}/{len(raw_jobs)} jobs")
        return jobs

    # ------------------------------------------------------------------
    # AI-powered processing (primary path — batched)
    # ------------------------------------------------------------------

    async def _process_with_ai(self, source: str, raw_jobs: List[Dict[str, Any]],
                                batch_size: int, max_concurrent: int,
                                on_batch_ready: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Send raw jobs to Gemini in batches for structured extraction.
        
        Chunks raw_jobs into groups of `batch_size`, then fires up to
        `max_concurrent` batch API calls in parallel using a dedicated thread pool.
        
        If `on_batch_ready` is provided, each batch is saved immediately after
        processing — no accumulation in memory.
        """
        from concurrent.futures import ThreadPoolExecutor

        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(raw_jobs)
        executor = ThreadPoolExecutor(max_workers=max_concurrent)

        # Split into chunks
        chunks = [raw_jobs[i:i + batch_size] for i in range(0, total, batch_size)]
        logger.info(f"[{source}] {total} jobs → {len(chunks)} batches of ≤{batch_size} (up to {max_concurrent} parallel)")

        async def process_chunk(chunk_idx: int, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    loop = asyncio.get_event_loop()
                    # Call _process_chunk directly — no double-chunking
                    ai_results = await loop.run_in_executor(
                        executor, self.ai_processor._process_chunk, source, chunk
                    )

                    finalized = []
                    for raw_job, ai_result in zip(chunk, ai_results):
                        if ai_result:
                            result = self._finalize_job(source, raw_job, ai_result)
                            if result:
                                finalized.append(result)
                        else:
                            fb = self._fallback_extract(source, raw_job)
                            if fb:
                                finalized.append(fb)

                    # Save this batch to DB immediately if callback provided
                    if on_batch_ready and finalized:
                        await on_batch_ready(finalized)

                    done = (chunk_idx + 1) * batch_size
                    logger.info(f"[{source}] Batch {chunk_idx + 1}/{len(chunks)} done ({min(done, total)}/{total} jobs)")
                    return finalized

                except Exception as e:
                    logger.error(f"[{source}] Batch {chunk_idx + 1} error: {e} — using fallback")
                    fallbacks = [fb for raw_job in chunk if (fb := self._fallback_extract(source, raw_job))]
                    if on_batch_ready and fallbacks:
                        await on_batch_ready(fallbacks)
                    return fallbacks

        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        batch_results = await asyncio.gather(*tasks)
        executor.shutdown(wait=False)

        # Flatten list of lists
        all_jobs = []
        for batch in batch_results:
            all_jobs.extend(batch)
        return all_jobs

    # ------------------------------------------------------------------
    # Fallback processing (no AI)
    # ------------------------------------------------------------------

    def _process_with_fallback(self, source: str, raw_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process jobs using rule-based extraction when AI is disabled."""
        jobs = []
        for raw_job in raw_jobs:
            job = self._fallback_extract(source, raw_job)
            if job:
                jobs.append(job)
        return jobs

    def _fallback_extract(self, source: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Lightweight field extraction for when AI is unavailable.
        Pulls fields from known locations in each source's raw data.
        """
        try:
            extracted = {}

            if source == "remoteok":
                extracted = {
                    "title": raw.get("position") or raw.get("title", ""),
                    "company": raw.get("company", "Unknown"),
                    "company_logo": raw.get("company_logo") or raw.get("logo"),
                    "description": raw.get("description", ""),
                    "is_remote": True,
                    "work_arrangement": "remote",
                    "country": raw.get("location"),
                    "employment_type": (raw.get("type") or "FULLTIME").upper(),
                    "salary_min": self._to_str(raw.get("salary_min")),
                    "salary_max": self._to_str(raw.get("salary_max")),
                    "salary_currency": raw.get("salary_currency", "USD"),
                    "apply_url": raw.get("url", ""),
                    "posted_at": self._parse_epoch(raw.get("epoch")),
                    "tags": raw.get("tags", []),
                    "skills": raw.get("tags", []),
                }

            elif source == "jsearch":
                loc_city = raw.get("job_city")
                loc_country = raw.get("job_country")
                extracted = {
                    "title": raw.get("job_title", ""),
                    "company": raw.get("employer_name", "Unknown"),
                    "company_logo": raw.get("employer_logo"),
                    "company_website": raw.get("employer_website"),
                    "description": raw.get("job_description", ""),
                    "city": loc_city,
                    "country": loc_country,
                    "is_remote": raw.get("job_is_remote", False),
                    "work_arrangement": "remote" if raw.get("job_is_remote") else "onsite",
                    "employment_type": (raw.get("job_employment_type") or "FULLTIME").upper(),
                    "salary_min": self._to_str(raw.get("job_min_salary")),
                    "salary_max": self._to_str(raw.get("job_max_salary")),
                    "salary_currency": raw.get("job_salary_currency", "USD"),
                    "salary_period": raw.get("job_salary_period", "").lower() or None,
                    "apply_url": raw.get("job_apply_link", ""),
                    "apply_options": raw.get("apply_options"),
                    "posted_at": self._parse_iso(raw.get("job_posted_at_datetime_utc")),
                    "application_deadline": self._parse_iso(raw.get("job_offer_expiration_datetime_utc")),
                    "required_experience_years": self._extract_years(raw.get("job_required_experience")),
                    "required_education": self._extract_education(raw.get("job_required_education")),
                    "skills": raw.get("job_required_skills") or [],
                    "benefits": self._extract_highlights_benefits(raw.get("job_highlights")),
                    "key_responsibilities": self._extract_highlights_responsibilities(raw.get("job_highlights")),
                }

            elif source == "adzuna":
                company_obj = raw.get("company") or {}
                location_obj = raw.get("location") or {}
                extracted = {
                    "title": raw.get("title", ""),
                    "company": company_obj.get("display_name", "Unknown") if isinstance(company_obj, dict) else str(company_obj),
                    "description": raw.get("description", ""),
                    "city": location_obj.get("display_name") if isinstance(location_obj, dict) else None,
                    "country": raw.get("_adzuna_country", ""),
                    "is_remote": self._detect_remote_from_text(raw.get("title", "") + " " + raw.get("description", "")),
                    "work_arrangement": "remote" if self._detect_remote_from_text(raw.get("title", "") + " " + raw.get("description", "")) else "onsite",
                    "employment_type": (raw.get("contract_type") or "FULLTIME").upper(),
                    "salary_min": self._to_str(raw.get("salary_min")),
                    "salary_max": self._to_str(raw.get("salary_max")),
                    "salary_currency": raw.get("salary_currency") or ("GBP" if raw.get("_adzuna_country") == "gb" else "USD"),
                    "apply_url": raw.get("redirect_url", ""),
                    "posted_at": self._parse_iso(raw.get("created")),
                    "latitude": raw.get("latitude"),
                    "longitude": raw.get("longitude"),
                    "tags": [raw.get("category", {}).get("label")] if isinstance(raw.get("category"), dict) else [],
                }

            elif source == "hackernews":
                extracted = {
                    "title": raw.get("title", ""),
                    "company": raw.get("company", "Unknown"),
                    "description": raw.get("description") or raw.get("_raw_text", ""),
                    "country": raw.get("location_raw"),
                    "is_remote": raw.get("remote", False),
                    "work_arrangement": "remote" if raw.get("remote") else "onsite",
                    "apply_url": raw.get("apply_url", ""),
                    "posted_at": self._parse_epoch(raw.get("hn_time")) or self._parse_iso(raw.get("posted_at")),
                    "source_url": raw.get("apply_url"),
                    "tags": [],
                }

            elif source == "rss_feed":
                extracted = {
                    "title": raw.get("title", ""),
                    "company": raw.get("company", "Unknown"),
                    "description": raw.get("description") or raw.get("_description_html", ""),
                    "country": raw.get("location_raw"),
                    "is_remote": raw.get("remote", True),
                    "work_arrangement": "remote" if raw.get("remote", True) else "onsite",
                    "apply_url": raw.get("apply_url") or raw.get("link", ""),
                    "posted_at": self._parse_posted_at(raw.get("posted_at")),
                    "tags": raw.get("_tags") or [],
                }

            elif source == "ats_scraper":
                # ATS scraper already outputs semi-structured data
                location = raw.get("location") or {}
                extracted = {
                    "title": raw.get("title", ""),
                    "company": raw.get("company", "Unknown"),
                    "description": raw.get("description", ""),
                    "city": location.get("city") if isinstance(location, dict) else None,
                    "country": location.get("country") if isinstance(location, dict) else None,
                    "is_remote": location.get("remote", False) if isinstance(location, dict) else False,
                    "employment_type": raw.get("employment_type"),
                    "apply_url": raw.get("apply_url", ""),
                    "posted_at": self._parse_iso(raw.get("posted_at")),
                }

            else:
                logger.warning(f"Unknown source: {source}")
                return None

            # Validate minimum required fields
            if not extracted.get("title") or not extracted.get("description"):
                return None

            # Run rule-based skill extraction
            title = extracted.get("title", "")
            desc = extracted.get("description", "")
            if not extracted.get("skills"):
                extracted["skills"] = self.skills_extractor.extract(title, desc)
            if not extracted.get("category"):
                extracted["category"] = self.skills_extractor.categorize_role(title, desc, extracted.get("skills", []))

            return self._finalize_job(source, raw, extracted)

        except Exception as e:
            logger.error(f"[{source}] Fallback extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Finalize: add system fields (id, source, raw_data, hashes, etc.)
    # ------------------------------------------------------------------

    def _finalize_job(self, source: str, raw_job: Dict[str, Any],
                      extracted: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add system-level fields that aren't part of AI extraction."""
        
        job = extracted.copy()

        # Source identity
        job["source"] = source
        job["source_id"] = self._derive_source_id(source, raw_job)
        job["id"] = f"{source}_{job['source_id']}"

        # Store full raw data as backup (serialize datetimes for JSON column)
        job["raw_data"] = json.loads(json.dumps(raw_job, default=str))

        # Older extraction schemas did not return the full description. Keep a
        # source-derived description as a safety net so downstream detail views
        # are not left with only the short summary.
        if not job.get("description"):
            job["description"] = self._raw_description(raw_job)

        # Carry over source_url from raw data if AI didn't extract it
        if not job.get("source_url"):
            job["source_url"] = (
                raw_job.get("apply_url")
                or raw_job.get("url")
                or raw_job.get("link")
                or raw_job.get("job_apply_link")
                or raw_job.get("redirect_url")
            )

        # Ensure posted_at has a value
        if not job.get("posted_at"):
            job["posted_at"] = datetime.now(timezone.utc)

        # Drop jobs older than MAX_JOB_AGE_DAYS
        cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_JOB_AGE_DAYS)
        posted = job["posted_at"]
        if isinstance(posted, datetime) and posted < cutoff:
            logger.debug("[%s] Dropping old job '%s' (posted %s)", source, job.get('title', ''), posted.date())
            return None

        # Parse application_deadline if it's a string
        if isinstance(job.get("application_deadline"), str):
            job["application_deadline"] = self._parse_iso(job["application_deadline"])

        # Ensure company has a value
        if not job.get("company"):
            job["company"] = "Unknown"

        # Coerce salary fields to strings to match the Go service's job model
        # (it deserializes them as *string, not *float). save_jobs._build_job_doc
        # already does this for the ingest path; doing it here covers the
        # reenrich path too, which writes the dict directly via update_one.
        for key in ("salary_min", "salary_max"):
            value = job.get(key)
            if value is not None and not isinstance(value, str):
                job[key] = str(value)

        # Quality score (rule-based, fast)
        job["quality_score"] = self.quality_scorer.score(job)

        # Track which prompt version produced this enrichment (only set on AI path)
        if self.use_ai and self.ai_processor and self.ai_processor.enabled:
            job["prompt_version"] = PROMPT_VERSION

        # Seed ranking_score from quality_score until popularity aggregation runs
        qs = job["quality_score"]
        if qs >= 60:
            job["ranking_score"] = min(50, qs)
        elif qs >= 40:
            job["ranking_score"] = 40
        else:
            job["ranking_score"] = 30

        # Title+company hash for dedup
        title = job.get("title", "")
        company = job.get("company", "")
        job["title_company_hash"] = hashlib.sha256(
            f"{title.lower().strip()}_{company.lower().strip()}".encode()
        ).hexdigest()[:16]

        # Ensure apply_url has a value, prefer raw data's URL over AI guess
        if not job.get("apply_url"):
            job["apply_url"] = (
                raw_job.get("apply_url")
                or raw_job.get("url")
                or raw_job.get("link")
                or raw_job.get("job_apply_link")
                or raw_job.get("redirect_url")
                or job.get("source_url")
                or "https://unknown"
            )

        return job

    # ------------------------------------------------------------------
    # Source ID derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_source_id(source: str, raw: Dict[str, Any]) -> str:
        """Extract or generate a unique source_id from raw data."""
        # Try common ID field names
        for key in ("id", "job_id", "source_id", "hn_comment_id", "entry_id"):
            val = raw.get(key)
            if val:
                return str(val)
        
        # Fallback: hash some identifying fields
        text = json.dumps(raw.get("title", "") + raw.get("company", ""), default=str)
        return hashlib.md5(text.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Helper: date parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_epoch(value) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    @staticmethod
    def _parse_iso(value) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.isoparse(str(value))
        except Exception:
            return None

    @staticmethod
    def _parse_posted_at(value) -> Optional[datetime]:
        """Parse posted_at which could be datetime, ISO string, or epoch."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                return None
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.isoparse(str(value))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helper: field extraction utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_str(value) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _detect_remote_from_text(text: str) -> bool:
        return bool(re.search(r'\b(remote|work from home|wfh|telecommut|anywhere)\b', text.lower()))

    @staticmethod
    def _raw_description(raw: Dict[str, Any]) -> str:
        for key in ("description", "job_description", "_description_html", "_raw_text", "summary"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _extract_years(exp_obj) -> Optional[int]:
        """Extract min years from JSearch job_required_experience object."""
        if not exp_obj or not isinstance(exp_obj, dict):
            return None
        min_years = exp_obj.get("required_experience_in_months")
        if min_years:
            return max(1, int(min_years) // 12)
        no_exp = exp_obj.get("no_experience_required")
        if no_exp:
            return 0
        return None

    @staticmethod
    def _extract_education(edu_obj) -> Optional[str]:
        """Extract education from JSearch job_required_education object."""
        if not edu_obj or not isinstance(edu_obj, dict):
            return None
        for key in ("degree_preferred", "degree_mentioned"):
            if edu_obj.get(key):
                return "Bachelor's"  # JSearch only flags presence, not level
        return None

    @staticmethod
    def _extract_highlights_benefits(highlights) -> Optional[List[str]]:
        if not highlights or not isinstance(highlights, dict):
            return None
        return highlights.get("Benefits") or highlights.get("benefits")

    @staticmethod
    def _extract_highlights_responsibilities(highlights) -> Optional[List[str]]:
        if not highlights or not isinstance(highlights, dict):
            return None
        return highlights.get("Responsibilities") or highlights.get("responsibilities")
