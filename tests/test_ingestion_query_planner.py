from datetime import datetime, timezone

import pytest

from src.services.ingestion import _collect_jobs, _query_planner_summary
from src.services.query_planner import QueryPlan, SourceQuery


class PlannedFetcher:
    source_name = "planned"

    def __init__(self):
        self.query_plan = None

    def set_query_plan(self, query_plan):
        self.query_plan = query_plan

    async def fetch_jobs(self):
        return [{"_query_plan_id": self.query_plan.plan_id, "_query": self.query_plan.queries[0].query}]


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="planned_test",
        source="planned",
        generated_at=datetime.now(timezone.utc).isoformat(),
        queries=[
            SourceQuery(
                query="software engineer",
                country="GB",
                canonical_country="GB",
                priority=5,
                max_pages=1,
            )
        ],
        fallback_used=True,
    )


@pytest.mark.asyncio
async def test_collect_jobs_passes_query_plan_to_fetcher():
    fetcher = PlannedFetcher()
    plan = _plan()

    source_name, jobs = await _collect_jobs(fetcher, plan)

    assert source_name == "planned"
    assert fetcher.query_plan == plan
    assert jobs == [{"_query_plan_id": "planned_test", "_query": "software engineer"}]


def test_query_planner_summary_reports_counts():
    plan = _plan()

    summary = _query_planner_summary({"planned": plan})

    assert summary["generated_queries"] == 1
    assert summary["fallback_used"] is True
    assert summary["sources"]["planned"]["query_count"] == 1
