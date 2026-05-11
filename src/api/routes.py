from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import JobResponse, JobsListResponse
from src.database.operations import db
from src.services.ingestion import run_ingestion_cycle

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/health")
async def health_check() -> dict:
    """Basic health endpoint for uptime monitoring."""

    return {"status": "ok"}


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(default=None, min_length=2),
    source: Optional[List[str]] = Query(default=None, alias="source"),
    employment_type: Optional[str] = Query(default=None),
    remote_only: bool = Query(default=False),
    seniority: Optional[List[str]] = Query(default=None, alias="seniority"),
    category: Optional[List[str]] = Query(default=None, alias="category"),
) -> JobsListResponse:
    """Return paginated backend/devops jobs."""

    # Get total count and paginated jobs
    total = await db.count_jobs(
        search=search,
        sources=source,
        employment_type=employment_type,
        remote_only=remote_only,
        seniority=seniority,
        category=category,
    )
    
    jobs = await db.list_jobs(
        limit=limit,
        offset=offset,
        search=search,
        sources=source,
        employment_type=employment_type,
        remote_only=remote_only,
        seniority=seniority,
        category=category,
    )

    return JobsListResponse(total=total, jobs=jobs)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Return a single job by identifier."""

    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/filters")
async def get_filters() -> dict:
    """Return available filter options with job counts."""

    return await db.get_filter_options()


@router.get("/companies/popular")
async def get_popular_companies(limit: int = Query(20, ge=1, le=100)) -> dict:
    """Top companies by active-job count. Used for the popular-companies UI carousel."""

    companies = await db.get_popular_companies(limit=limit)
    return {"companies": companies}


@router.post("/jobs/ingest")
async def trigger_ingestion() -> dict:
    """Fire off an ad-hoc ingestion cycle."""

    return await run_ingestion_cycle()
