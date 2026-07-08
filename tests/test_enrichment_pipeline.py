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



def test_compute_board_rank_orders_by_company_tier_and_seniority():
    from src.enrichment.enrichment_pipeline import compute_board_rank

    google_intern = compute_board_rank(
        {"company": "Google India", "company_tier": "top_tech", "seniority_level": "intern"}
    )
    startup_junior = compute_board_rank(
        {"company": "Notion", "company_tier": "hot_startup", "seniority_level": "junior"}
    )
    unknown_intern = compute_board_rank(
        {"company": "Acme Corp", "company_tier": "other", "seniority_level": "intern"}
    )
    google_senior = compute_board_rank(
        {"company": "Google", "company_tier": "top_tech", "seniority_level": "senior"}
    )
    unknown_no_seniority = compute_board_rank({"company": "Acme Corp"})

    assert google_intern == 100
    assert google_intern > startup_junior > unknown_intern > google_senior
    assert unknown_no_seniority == 25


def test_finalize_job_sets_board_rank():
    pipeline = EnrichmentPipeline(use_ai=False)
    raw = {"id": "job-1", "title": "SWE Intern", "company": "Acme Corp"}
    extracted = {
        "title": "SWE Intern",
        "company": "Acme Corp",
        "seniority_level": "intern",
        "posted_at": datetime.now(timezone.utc),
    }

    job = pipeline._finalize_job("remoteok", raw, extracted)

    assert job["board_rank"] == 60


def test_prompt_version_stamped_only_on_ai_extracted_jobs():
    pipeline = EnrichmentPipeline(use_ai=False)
    raw = {"id": "job-2", "title": "QA Intern", "company": "Acme Corp"}
    extracted = {
        "title": "QA Intern",
        "company": "Acme Corp",
        "posted_at": datetime.now(timezone.utc),
    }

    fallback_job = pipeline._finalize_job("remoteok", raw, extracted)
    ai_job = pipeline._finalize_job("remoteok", raw, dict(extracted), ai_extracted=True)

    assert "prompt_version" not in fallback_job
    assert ai_job["prompt_version"]


def test_age_cutoff_skipped_when_disabled():
    from datetime import timedelta

    old_posted = datetime.now(timezone.utc) - timedelta(days=60)
    raw = {"id": "job-3", "title": "Dev", "company": "Acme Corp"}
    extracted = {"title": "Dev", "company": "Acme Corp", "posted_at": old_posted}

    ingest_pipeline = EnrichmentPipeline(use_ai=False)
    reenrich_pipeline = EnrichmentPipeline(use_ai=False, enforce_age_cutoff=False)

    assert ingest_pipeline._finalize_job("remoteok", raw, dict(extracted)) is None
    assert reenrich_pipeline._finalize_job("remoteok", raw, dict(extracted)) is not None


def test_process_with_ai_slims_results_after_batch_save():
    import asyncio

    pipeline = EnrichmentPipeline(use_ai=False)

    class _FakeProcessor:
        enabled = True

        def _process_chunk(self, source, chunk):
            return [{
                "title": r["title"], "company": "Acme Corp",
                "seniority_level": "junior", "country": "in",
                "posted_at": datetime.now(timezone.utc),
            } for r in chunk]

    pipeline.ai_processor = _FakeProcessor()
    saved = []

    async def on_batch_ready(batch):
        saved.extend(batch)

    raw = [{"id": f"j{i}", "title": f"Job {i}"} for i in range(3)]
    result = asyncio.run(pipeline._process_with_ai("remoteok", raw, 5, 2, on_batch_ready))

    # Callback received full docs; the retained result is slim
    assert len(saved) == 3 and saved[0].get("raw_data") is not None
    assert len(result) == 3
    assert set(result[0].keys()) == {"country", "seniority_level"}
