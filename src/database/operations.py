"""MongoDB database operations using Motor (async driver)."""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import DESCENDING
from pymongo.uri_parser import parse_uri

from src.database.models import ensure_indexes, normalize_doc
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Database:
    """MongoDB database helper for connections and CRUD operations."""

    def __init__(self) -> None:
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None

    @property
    def jobs(self):
        return self.db["jobs"]

    @property
    def companies(self):
        return self.db["discovered_companies"]

    @property
    def query_metrics(self):
        return self.db["query_metrics"]

    async def connect(self) -> None:
        """Initialize MongoDB connection and ensure indexes exist."""

        uri = settings.database_url
        self.client = AsyncIOMotorClient(uri)

        parsed = parse_uri(uri)
        db_name = parsed.get("database") or "jobs_db"
        self.db = self.client[db_name]

        await self.client.admin.command("ping")
        await ensure_indexes(self.db)
        logger.info("MongoDB connected (database: %s)", db_name)

    async def disconnect(self) -> None:
        """Close MongoDB connection."""

        if self.client:
            self.client.close()
            logger.info("MongoDB disconnected")

    async def save_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Insert jobs, restore expired duplicates, and skip active/archive hits."""

        stats = {"new": 0, "skipped": 0, "restored": 0}

        if self.db is None:
            raise RuntimeError("Database not connected")

        for job_data in jobs:
            try:
                job_id = job_data.get("id") or f"{job_data['source']}_{job_data['source_id']}"
                title_company_hash = job_data.get("title_company_hash") or self._hash_title_company(
                    job_data.get("title", ""), job_data.get("company", "")
                )
                doc = self._build_job_doc(job_id, title_company_hash, job_data)

                duplicate = await self.jobs.find_one({"_id": job_id})
                if duplicate is None:
                    duplicate = await self.jobs.find_one({"title_company_hash": title_company_hash})

                if duplicate:
                    if duplicate.get("is_archived") is True:
                        stats["skipped"] += 1
                        continue
                    if duplicate.get("is_deleted") is True:
                        update_doc = doc.copy()
                        update_doc.pop("_id", None)
                        update_doc["is_deleted"] = False
                        await self.jobs.update_one(
                            {"_id": duplicate["_id"]},
                            {
                                "$set": update_doc,
                                "$unset": {
                                    "deleted_at": "",
                                    "delete_reason": "",
                                    "pinecone_embedded_at": "",
                                },
                            },
                        )
                        stats["restored"] += 1
                        continue
                    stats["skipped"] += 1
                    continue

                await self.jobs.insert_one(doc)
                stats["new"] += 1

            except Exception as exc:
                logger.error("Error saving job: %s", exc, exc_info=True)
                stats["skipped"] += 1

        return stats

    def _build_job_doc(
        self,
        job_id: str,
        title_company_hash: str,
        job_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "_id": job_id,
            "source": job_data.get("source", ""),
            "source_id": str(job_data.get("source_id", "")),
            "source_url": job_data.get("source_url"),
            "title": job_data.get("title", ""),
            "company": job_data.get("company", "Unknown"),
            "company_logo": job_data.get("company_logo"),
            "company_website": job_data.get("company_website"),
            "description": job_data.get("description", ""),
            "short_description": job_data.get("short_description"),
            "country": job_data.get("country"),
            "city": job_data.get("city"),
            "state": job_data.get("state"),
            "is_remote": job_data.get("is_remote"),
            "work_arrangement": job_data.get("work_arrangement"),
            "latitude": job_data.get("latitude"),
            "longitude": job_data.get("longitude"),
            "employment_type": job_data.get("employment_type"),
            "seniority_level": job_data.get("seniority_level"),
            "department": job_data.get("department"),
            "category": job_data.get("category"),
            "salary_min": self._to_str(job_data.get("salary_min")),
            "salary_max": self._to_str(job_data.get("salary_max")),
            "salary_currency": job_data.get("salary_currency"),
            "salary_period": job_data.get("salary_period"),
            "skills": job_data.get("skills"),
            "required_experience_years": job_data.get("required_experience_years"),
            "required_education": job_data.get("required_education"),
            "key_responsibilities": job_data.get("key_responsibilities"),
            "nice_to_have_skills": job_data.get("nice_to_have_skills"),
            "benefits": job_data.get("benefits"),
            "visa_sponsorship": job_data.get("visa_sponsorship"),
            "posted_at": job_data.get("posted_at"),
            "application_deadline": job_data.get("application_deadline"),
            "fetched_at": datetime.now(timezone.utc),
            "apply_url": job_data.get("apply_url", ""),
            "apply_options": job_data.get("apply_options"),
            "tags": job_data.get("tags"),
            "quality_score": job_data.get("quality_score"),
            "company_tier": job_data.get("company_tier"),
            "board_rank": job_data.get("board_rank"),
            "raw_data": job_data.get("raw_data"),
            "prompt_version": job_data.get("prompt_version"),
            "title_company_hash": title_company_hash,
        }

    @staticmethod
    def _hash_title_company(title: str, company: str) -> str:
        text = f"{title.lower().strip()}_{company.lower().strip()}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _to_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    def _build_filter(
        self,
        *,
        search: Optional[str] = None,
        sources: Optional[List[str]] = None,
        employment_type: Optional[str] = None,
        remote_only: bool = False,
        seniority: Optional[List[str]] = None,
        category: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a MongoDB filter dict from query parameters."""

        query: Dict[str, Any] = self.active_job_filter()

        if search:
            query["$text"] = {"$search": search}

        if sources:
            query["source"] = {"$in": sources}

        if employment_type:
            query["employment_type"] = re.compile(f"^{re.escape(employment_type)}$", re.IGNORECASE)

        if remote_only:
            query["is_remote"] = True

        if seniority:
            query["seniority_level"] = {
                "$in": [re.compile(f"^{re.escape(s)}$", re.IGNORECASE) for s in seniority]
            }

        if category:
            query["category"] = {
                "$in": [re.compile(f"^{re.escape(c)}$", re.IGNORECASE) for c in category]
            }

        return query

    @staticmethod
    def active_job_filter() -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "is_archived": {"$ne": True},
            "is_deleted": {"$ne": True},
            "is_unlisted": {"$ne": True},
            "is_user_private": {"$ne": True},
            "$or": [
                {"application_deadline": None},
                {"application_deadline": {"$gte": now}},
            ],
        }

    async def count_jobs(
        self,
        *,
        search: Optional[str] = None,
        sources: Optional[List[str]] = None,
        employment_type: Optional[str] = None,
        remote_only: bool = False,
        seniority: Optional[List[str]] = None,
        category: Optional[List[str]] = None,
    ) -> int:
        """Count total jobs matching the filters."""

        if self.db is None:
            raise RuntimeError("Database not connected")

        query = self._build_filter(
            search=search, sources=sources, employment_type=employment_type,
            remote_only=remote_only, seniority=seniority, category=category,
        )
        return await self.jobs.count_documents(query)

    async def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        sources: Optional[List[str]] = None,
        employment_type: Optional[str] = None,
        remote_only: bool = False,
        seniority: Optional[List[str]] = None,
        category: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return paginated jobs with filtering and full-text search."""

        if self.db is None:
            raise RuntimeError("Database not connected")

        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        query = self._build_filter(
            search=search, sources=sources, employment_type=employment_type,
            remote_only=remote_only, seniority=seniority, category=category,
        )

        projection = {"raw_data": 0}

        if search:
            projection["score"] = {"$meta": "textScore"}
            sort = [("score", {"$meta": "textScore"}), ("posted_at", DESCENDING)]
        else:
            sort = [("board_rank", DESCENDING), ("posted_at", DESCENDING)]

        cursor = self.jobs.find(query, projection).sort(sort).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)

        for doc in docs:
            normalize_doc(doc)
            doc.pop("score", None)

        return docs

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single job by identifier."""

        if self.db is None:
            raise RuntimeError("Database not connected")

        doc = await self.jobs.find_one(
            {"_id": job_id, **self.active_job_filter()},
            {"raw_data": 0},
        )
        return normalize_doc(doc)

    async def get_filter_options(self) -> Dict[str, Any]:
        """Get available filter options with job counts."""

        if self.db is None:
            raise RuntimeError("Database not connected")

        seniority_pipeline = [
            {"$match": {**self.active_job_filter(), "seniority_level": {"$ne": None}}},
            {"$group": {"_id": "$seniority_level", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        seniority = [
            {"value": doc["_id"], "count": doc["count"]}
            async for doc in self.jobs.aggregate(seniority_pipeline)
        ]

        category_pipeline = [
            {"$match": {**self.active_job_filter(), "category": {"$ne": None}}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        categories = [
            {"value": doc["_id"], "count": doc["count"]}
            async for doc in self.jobs.aggregate(category_pipeline)
        ]

        source_pipeline = [
            {"$match": self.active_job_filter()},
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        sources = [
            {"value": doc["_id"], "count": doc["count"]}
            async for doc in self.jobs.aggregate(source_pipeline)
        ]

        remote_count = await self.jobs.count_documents({**self.active_job_filter(), "is_remote": True})

        return {
            "seniority": seniority,
            "category": categories,
            "sources": sources,
            "remote_count": remote_count,
        }

    async def get_recent_query_metrics(self, source: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent query-planner metrics for feedback."""

        if self.db is None:
            return []

        cursor = (
            self.query_metrics.find({"source": source}, {"_id": 0})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def save_query_metrics(
        self,
        source: str,
        query_plan,
        raw_jobs: List[Dict[str, Any]],
        processed_jobs: List[Dict[str, Any]],
        save_stats: Dict[str, int],
    ) -> None:
        """Store lightweight query metrics for future query-planner feedback."""

        if self.db is None or query_plan is None:
            return

        raw_by_query: Dict[str, int] = {}
        for job in raw_jobs:
            key = job.get("_query") or job.get("_jsearch_query") or "unknown"
            raw_by_query[key] = raw_by_query.get(key, 0) + 1

        country_counts: Dict[str, int] = {}
        seniority_counts: Dict[str, int] = {}
        for job in processed_jobs:
            country = job.get("country") or "unknown"
            seniority = job.get("seniority_level") or "unknown"
            country_counts[country] = country_counts.get(country, 0) + 1
            seniority_counts[seniority] = seniority_counts.get(seniority, 0) + 1

        await self.query_metrics.insert_one({
            "source": source,
            "plan_id": query_plan.plan_id,
            "created_at": datetime.now(timezone.utc),
            "fallback_used": query_plan.fallback_used,
            "query_count": len(query_plan.queries),
            "raw_jobs": len(raw_jobs),
            "processed_jobs": len(processed_jobs),
            "new_jobs": save_stats.get("new", 0),
            "skipped_jobs": save_stats.get("skipped", 0),
            "restored_jobs": save_stats.get("restored", 0),
            "raw_by_query": raw_by_query,
            "country_counts": country_counts,
            "seniority_counts": seniority_counts,
        })

    async def cleanup_expired_jobs(self, default_expiry_days: int = 15) -> Dict[str, int]:
        """Soft-delete jobs past their application_deadline or default age."""

        if self.db is None:
            raise RuntimeError("Database not connected")

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=default_expiry_days)

        condition = {
            "is_archived": {"$ne": True},
            "is_deleted": {"$ne": True},
            "$or": [
                {"application_deadline": {"$ne": None, "$lt": now}},
                {"application_deadline": None, "posted_at": {"$lt": cutoff}},
            ],
        }

        result = await self.jobs.update_many(
            condition,
            {"$set": {"is_deleted": True, "deleted_at": now, "delete_reason": "expired"}},
        )
        return {"deleted": result.modified_count}


db = Database()
