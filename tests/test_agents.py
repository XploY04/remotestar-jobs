from datetime import datetime, timezone

import pytest

from src.agents.jsearch import JSearchFetcher
from src.agents.remoteok import RemoteOKFetcher
from src.services.query_planner import QueryPlan, SourceQuery


def _plan(source: str) -> QueryPlan:
    return QueryPlan(
        plan_id=f"{source}_test",
        source=source,
        generated_at=datetime.now(timezone.utc).isoformat(),
        queries=[
            SourceQuery(
                query="software engineer",
                country="GB" if source == "jsearch" else "gb",
                canonical_country="GB",
                priority=9,
                max_pages=2,
            )
        ],
    )


@pytest.mark.asyncio
async def test_remoteok_fetcher_initialization():
    fetcher = RemoteOKFetcher()

    assert fetcher.source_name == "remoteok"
    assert hasattr(fetcher, "fetch_jobs")


@pytest.mark.asyncio
async def test_jsearch_fetcher_initialization():
    fetcher = JSearchFetcher()

    assert fetcher.source_name == "jsearch"
    assert hasattr(fetcher, "fetch_jobs")


@pytest.mark.asyncio
async def test_base_fetcher_keyword_filtering():
    from src.agents import BaseFetcher

    class TestFetcher(BaseFetcher):
        async def fetch_jobs(self):
            return []

    fetcher = TestFetcher("test")

    assert fetcher.is_backend_devops_job("Senior Backend Engineer", "")
    assert fetcher.is_backend_devops_job("DevOps Engineer", "")
    assert fetcher.is_backend_devops_job("", "Experience with kubernetes and docker")
    assert fetcher.is_backend_devops_job("SRE", "Site reliability engineering")
    assert fetcher.is_backend_devops_job("Cloud Engineer", "AWS and terraform")

    assert not fetcher.is_backend_devops_job("Frontend Developer", "React and Vue.js")
    assert not fetcher.is_backend_devops_job("Marketing Manager", "")
    assert not fetcher.is_backend_devops_job("Data Analyst", "Excel and SQL")


def test_jsearch_accepts_query_plan():
    fetcher = JSearchFetcher()
    plan = _plan("jsearch")

    fetcher.set_query_plan(plan)

    assert fetcher._source_queries() == plan.queries


def test_jsearch_dedupes_raw_jobs():
    fetcher = JSearchFetcher()
    jobs = [{"job_id": "1"}, {"job_id": "1"}, {"job_id": "2"}]

    assert fetcher._dedupe_jobs(jobs) == [{"job_id": "1"}, {"job_id": "2"}]


def test_import_all_agents():
    from src.agents import BaseFetcher
    from src.agents.jsearch import JSearchFetcher
    from src.agents.remoteok import RemoteOKFetcher

    assert RemoteOKFetcher is not None
    assert JSearchFetcher is not None
    assert BaseFetcher is not None
