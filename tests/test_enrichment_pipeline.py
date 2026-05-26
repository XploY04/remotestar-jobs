from datetime import datetime, timezone

from src.enrichment.enrichment_pipeline import EnrichmentPipeline


def test_finalize_job_preserves_raw_description_when_ai_result_has_only_summary():
    pipeline = EnrichmentPipeline(use_ai=False)
    raw = {
        "id": "job-123",
        "title": "Backend Engineer",
        "company": "RemoteStar",
        "description": "Build APIs, own service reliability, and work with product teams.",
        "url": "https://example.com/jobs/job-123",
    }
    extracted = {
        "title": "Backend Engineer",
        "company": "RemoteStar",
        "short_description": "Build backend APIs for RemoteStar.",
        "is_remote": True,
        "work_arrangement": "remote",
        "employment_type": "FULLTIME",
        "category": "backend",
        "skills": ["Python"],
        "posted_at": datetime.now(timezone.utc),
    }

    job = pipeline._finalize_job("remoteok", raw, extracted)

    assert job["description"] == raw["description"]
    assert job["short_description"] == extracted["short_description"]

