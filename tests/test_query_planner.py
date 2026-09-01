import pytest

from src.services.query_planner import QueryPlannerService, TargetConfig


@pytest.mark.asyncio
async def test_fallback_plan_covers_all_target_countries():
    planner = QueryPlannerService(TargetConfig(max_queries_per_source=9, max_pages_per_query=3))

    plan = await planner.generate_query_plan("jsearch")

    assert plan.fallback_used is True
    assert {query.canonical_country for query in plan.queries} == {"IN", "GB", "US"}
    assert {query.country for query in plan.queries} == {"IN", "GB", "US"}
    assert all(query.max_pages == 3 for query in plan.queries)


def test_country_mapping_is_source_specific():
    assert QueryPlannerService.map_country("jsearch", "GB") == "GB"
    assert QueryPlannerService.map_country("ats_discovery", "GB") == "UK"


def test_parse_queries_caps_and_validates_model_output():
    planner = QueryPlannerService(TargetConfig(max_queries_per_source=3, max_pages_per_query=3))
    raw_plan = {
        "queries": [
            {"query": "software engineer", "canonical_country": "GB", "priority": 20, "max_pages": 9},
            {"query": "data engineer", "canonical_country": "US", "priority": 4, "max_pages": 2},
            {"query": "ignored", "canonical_country": "CA", "priority": 4, "max_pages": 2},
            {"query": "ignored too", "canonical_country": "IN", "priority": 4, "max_pages": 2},
        ]
    }

    queries = planner._parse_queries("jsearch", raw_plan)

    assert len(queries) == 3
    assert queries[0].country == "GB"
    assert queries[0].priority == 10
    assert queries[0].max_pages == 3
    assert queries[1].country == "US"
    assert queries[2].country == "IN"


def test_ats_fallback_uses_location_terms():
    planner = QueryPlannerService(TargetConfig(max_queries_per_source=3))

    plan = planner._fallback_plan("ats_scraper", planner_source="ats_discovery")

    assert plan.source == "ats_scraper"
    assert {query.country for query in plan.queries} == {"India", "UK", "United States"}
    assert all("software engineer" in query.query for query in plan.queries)
