from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    """Job representation returned via the public API."""

    model_config = ConfigDict(from_attributes=True)

    # ── Identity ──
    id: str
    source: str
    source_id: str
    source_url: Optional[str] = None

    # ── Core Job Info ──
    title: str
    company: str
    company_logo: Optional[str] = None
    company_website: Optional[str] = None
    description: str
    short_description: Optional[str] = None

    # ── Location ──
    location: Optional[Union[Dict, List, str]] = None  # legacy blob
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    is_remote: Optional[bool] = None
    work_arrangement: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # ── Employment Details ──
    employment_type: Optional[str] = None
    seniority_level: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None

    # ── Compensation ──
    salary_min: Optional[str] = None
    salary_max: Optional[str] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None

    # ── Skills & Requirements ──
    skills: Optional[List[str]] = None
    required_experience_years: Optional[int] = None
    required_education: Optional[str] = None
    key_responsibilities: Optional[List[str]] = None
    nice_to_have_skills: Optional[List[str]] = None

    # ── Benefits & Perks ──
    benefits: Optional[List[str]] = None
    visa_sponsorship: Optional[str] = None

    # ── Dates ──
    posted_at: Optional[datetime] = None
    application_deadline: Optional[datetime] = None
    fetched_at: Optional[datetime] = None

    # ── Apply ──
    apply_url: Optional[str] = None
    apply_options: Optional[List[Dict]] = None

    # ── Quality / Meta ──
    tags: Optional[List[str]] = None
    quality_score: Optional[int] = None


class JobsListResponse(BaseModel):
    """Paginated response wrapper for job listings."""

    total: int
    jobs: List[JobResponse]
