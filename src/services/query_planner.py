"""AI-assisted query planning for source fetchers."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


CANONICAL_COUNTRIES = ["IN", "GB", "US"]

COUNTRY_TERMS = {
    "IN": "India",
    "GB": "UK",
    "US": "United States",
}

SOURCE_COUNTRY_MAP = {
    "jsearch": {"IN": "IN", "GB": "GB", "US": "US"},
    "adzuna": {"IN": "in", "GB": "gb", "US": "us"},
    "ats_discovery": {"IN": "India", "GB": "UK", "US": "United States"},
}

BROAD_TECH_QUERIES = [
    "software engineer",
    "software developer",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "data engineer",
    "machine learning engineer",
    "devops engineer",
    "qa engineer",
    "security engineer",
    "mobile developer",
    "cloud engineer",
]

ATS_DISCOVERY_QUERIES = [
    "software engineer",
    "engineering jobs",
    "software developer",
    "technology careers",
    "developer jobs",
]


@dataclass
class TargetConfig:
    """High-level job coverage target for query planning."""

    countries: List[str] = field(default_factory=lambda: CANONICAL_COUNTRIES.copy())
    role_scope: str = "tech"
    seniority: str = "all"
    include_internships: bool = True
    posted_within_days: int = settings.query_planner_posted_within_days
    max_queries_per_source: int = settings.query_planner_max_queries_per_source
    max_pages_per_query: int = settings.query_planner_max_pages_per_query


@dataclass
class SourceQuery:
    """One source-specific query instruction."""

    query: str
    country: str
    canonical_country: str
    priority: int = 5
    max_pages: int = settings.query_planner_max_pages_per_query

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    """Query instructions for one source."""

    plan_id: str
    source: str
    queries: List[SourceQuery]
    generated_at: str
    fallback_used: bool = False

    def to_summary(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source": self.source,
            "query_count": len(self.queries),
            "fallback_used": self.fallback_used,
            "generated_at": self.generated_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.to_summary()
        data["queries"] = [query.to_dict() for query in self.queries]
        return data


class QueryPlannerService:
    """Generate source query plans with OpenAI, or deterministic fallback plans."""

    PLANNED_SOURCES = {"jsearch", "adzuna", "ats_scraper"}
    SOURCE_ALIASES = {"ats_scraper": "ats_discovery"}

    def __init__(self, target_config: Optional[TargetConfig] = None) -> None:
        self.target_config = target_config or TargetConfig()
        self.enabled = settings.enable_query_planner
        self.openai_key = settings.openai_api_key
        self.model = settings.query_planner_model

    async def generate_plans(self, sources: List[str]) -> Dict[str, QueryPlan]:
        """Generate plans for requested sources that support query planning."""
        plans: Dict[str, QueryPlan] = {}

        for source in sources:
            if source not in self.PLANNED_SOURCES:
                continue
            plans[source] = await self.generate_query_plan(source)

        return plans

    async def generate_query_plan(self, source: str) -> QueryPlan:
        """Generate a query plan for one source."""
        planner_source = self.SOURCE_ALIASES.get(source, source)
        if not self.enabled or not self.openai_key:
            return self._fallback_plan(source, planner_source=planner_source)

        try:
            recent_metrics = await self._recent_metrics(source)
            raw_plan = await self._call_openai(planner_source, recent_metrics)
            queries = self._parse_queries(planner_source, raw_plan)
            if not queries:
                logger.warning("Query planner returned no valid queries for %s; using fallback", source)
                return self._fallback_plan(source, planner_source=planner_source)

            return QueryPlan(
                plan_id=self._new_plan_id(source),
                source=source,
                queries=queries,
                generated_at=datetime.now(timezone.utc).isoformat(),
                fallback_used=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Query planner failed for %s: %s", source, exc, exc_info=True)
            return self._fallback_plan(source, planner_source=planner_source)

    async def _call_openai(self, source: str, recent_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_key)
        response = await client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise job-search query plans. Return only JSON. "
                        "Generate broad tech coverage queries across all seniority levels, including internships, "
                        "but do not overfit to city names or junior-only phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source": source,
                            "target": asdict(self.target_config),
                            "allowed_countries": self.target_config.countries,
                            "country_terms": COUNTRY_TERMS,
                            "recent_metrics": recent_metrics,
                            "output_schema": {
                                "queries": [
                                    {
                                        "query": "string",
                                        "canonical_country": "IN|GB|US",
                                        "priority": "integer 1-10",
                                        "max_pages": "integer",
                                    }
                                ]
                            },
                        },
                        default=str,
                        ensure_ascii=True,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _parse_queries(self, source: str, raw_plan: Dict[str, Any]) -> List[SourceQuery]:
        raw_queries = raw_plan.get("queries", [])
        if not isinstance(raw_queries, list):
            return []

        queries: List[SourceQuery] = []
        seen = set()
        for item in raw_queries:
            if not isinstance(item, dict):
                continue

            query_text = str(item.get("query", "")).strip()
            canonical_country = str(item.get("canonical_country", "")).upper()
            if not query_text or canonical_country not in self.target_config.countries:
                continue

            mapped_country = self.map_country(source, canonical_country)
            if not mapped_country:
                continue

            key = (query_text.lower(), mapped_country)
            if key in seen:
                continue
            seen.add(key)

            priority = self._bounded_int(item.get("priority"), default=5, minimum=1, maximum=10)
            max_pages = self._bounded_int(
                item.get("max_pages"),
                default=self.target_config.max_pages_per_query,
                minimum=1,
                maximum=self.target_config.max_pages_per_query,
            )
            queries.append(SourceQuery(
                query=query_text,
                country=mapped_country,
                canonical_country=canonical_country,
                priority=priority,
                max_pages=max_pages,
            ))

            if len(queries) >= self.target_config.max_queries_per_source:
                break

        return self._ensure_country_coverage(source, queries)

    def _fallback_plan(self, source: str, planner_source: str | None = None) -> QueryPlan:
        planner_source = planner_source or self.SOURCE_ALIASES.get(source, source)
        if planner_source == "ats_discovery":
            base_queries = ATS_DISCOVERY_QUERIES
        else:
            base_queries = BROAD_TECH_QUERIES

        queries: List[SourceQuery] = []
        for query in base_queries:
            for canonical_country in self.target_config.countries:
                mapped_country = self.map_country(planner_source, canonical_country)
                if not mapped_country:
                    continue

                country_term = COUNTRY_TERMS.get(canonical_country, canonical_country)
                query_text = f"{country_term} {query}" if planner_source == "ats_discovery" else query
                queries.append(SourceQuery(
                    query=query_text,
                    country=mapped_country,
                    canonical_country=canonical_country,
                    priority=5,
                    max_pages=self.target_config.max_pages_per_query,
                ))
                if len(queries) >= self.target_config.max_queries_per_source:
                    break
            if len(queries) >= self.target_config.max_queries_per_source:
                break

        return QueryPlan(
            plan_id=self._new_plan_id(source),
            source=source,
            queries=queries,
            generated_at=datetime.now(timezone.utc).isoformat(),
            fallback_used=True,
        )

    @staticmethod
    def map_country(source: str, canonical_country: str) -> Optional[str]:
        return SOURCE_COUNTRY_MAP.get(source, {}).get(canonical_country)

    async def _recent_metrics(self, source: str) -> List[Dict[str, Any]]:
        try:
            from src.database.operations import db
            metrics = await db.get_recent_query_metrics(source)
            return json.loads(json.dumps(metrics, default=str))
        except Exception:
            return []

    def _ensure_country_coverage(self, source: str, queries: List[SourceQuery]) -> List[SourceQuery]:
        present = {query.canonical_country for query in queries}
        missing = [country for country in self.target_config.countries if country not in present]
        if not missing:
            return queries

        for canonical_country in missing:
            mapped_country = self.map_country(source, canonical_country)
            if not mapped_country:
                continue

            if source == "ats_discovery":
                query_text = f"{COUNTRY_TERMS.get(canonical_country, canonical_country)} software engineer"
            else:
                query_text = BROAD_TECH_QUERIES[0]

            fallback_query = SourceQuery(
                query=query_text,
                country=mapped_country,
                canonical_country=canonical_country,
                priority=5,
                max_pages=self.target_config.max_pages_per_query,
            )
            if len(queries) >= self.target_config.max_queries_per_source:
                queries[-1] = fallback_query
            else:
                queries.append(fallback_query)

        return queries

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(number, maximum))

    @staticmethod
    def _new_plan_id(source: str) -> str:
        return f"{source}_{uuid.uuid4().hex[:12]}"
