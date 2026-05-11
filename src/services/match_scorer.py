"""Structured scoring functions for job-user matching.

Each function returns a float between 0.0 and 1.0.
The orchestrator (matcher.py) combines them with weights into a 0-100 score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

# Weights for the final score
WEIGHTS = {
    "skills_match": 0.35,
    "title_similarity": 0.20,
    "seniority_fit": 0.15,
    "location_match": 0.15,
    "experience_fit": 0.10,
    "salary_fit": 0.05,
}

# Seniority levels in order (index = rank)
SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager"]

# Normalized role-to-category mapping for title matching. Keys MUST match the
# job-side Category Literal in src/enrichment/ai_processor.py.
ROLE_CATEGORY_MAP = {
    "backend": ["backend", "back-end", "server", "api", "microservices", "systems engineer"],
    "frontend": ["frontend", "front-end", "ui engineer", "react developer", "angular", "vue"],
    "fullstack": ["fullstack", "full-stack", "full stack"],
    "mobile": ["mobile", "ios", "android", "flutter", "react native"],
    "devops": ["devops", "infrastructure", "cloud engineer", "kubernetes"],
    "sre": ["sre", "site reliability", "platform engineer", "reliability"],
    "data": ["data engineer", "data pipeline", "etl", "data platform", "analytics engineer"],
    "ml": ["machine learning", "ml engineer", "ai engineer", "deep learning", "nlp", "mlops"],
    "security": ["security", "cybersecurity", "infosec", "appsec", "pentest", "soc analyst"],
    "qa": ["qa", "quality assurance", "test engineer", "sdet", "test automation"],
    "design": ["designer", "ux", "ui designer", "product designer", "design lead"],
    "product": ["product manager", "product owner", "product analyst", " pm "],
    "general": [],   # catch-all, matched only via the no-keyword fallback path
}

# Legacy free-form country variants for jobs ingested before v3 (when country
# became a Literal of ISO alpha-2 codes). New jobs already arrive normalized.
_LEGACY_COUNTRY_ALIASES = {
    "us": ("usa", "united states", "america", "u.s.", "u.s.a."),
    "gb": ("uk", "united kingdom", "england", "britain", "great britain"),
    "in": ("india",),
    "ae": ("uae", "united arab emirates"),
    "de": ("germany", "deutschland"),
    "fr": ("france",),
    "nl": ("netherlands", "holland"),
    "ca": ("canada",),
    "au": ("australia",),
    "sg": ("singapore",),
}


def compute_idf(jobs: List[dict]) -> Dict[str, float]:
    """Compute inverse document frequency for skills across the job corpus."""
    n = len(jobs)
    if n == 0:
        return {}

    doc_freq: Counter = Counter()
    for job in jobs:
        skills = job.get("skills") or []
        unique_skills = set(s.lower().strip() for s in skills if s)
        for skill in unique_skills:
            doc_freq[skill] += 1

    idf = {}
    for skill, freq in doc_freq.items():
        idf[skill] = math.log(n / (1 + freq))

    return idf


def skills_match_score(
    user_skills: List[str],
    job_skills: List[str],
    idf_weights: Dict[str, float],
) -> float:
    """TF-IDF weighted skill overlap. Rare skills count more."""
    if not user_skills or not job_skills:
        return 0.0

    user_set = set(s.lower().strip() for s in user_skills)
    job_set = set(s.lower().strip() for s in job_skills)

    overlap = user_set & job_set
    if not overlap:
        return 0.0

    default_idf = 1.0
    overlap_weight = sum(idf_weights.get(s, default_idf) for s in overlap)
    job_weight = sum(idf_weights.get(s, default_idf) for s in job_set)

    if job_weight == 0:
        return 0.0

    return min(1.0, overlap_weight / job_weight)


def title_similarity_score(
    user_titles: List[str],
    job_title: str,
) -> float:
    """Fuzzy match user's role focus + past job titles against job title."""
    if not user_titles or not job_title:
        return 0.0

    job_lower = job_title.lower().strip()
    best = 0.0

    for title in user_titles:
        if not title:
            continue
        title_lower = title.lower().strip()

        ratio = SequenceMatcher(None, title_lower, job_lower).ratio()

        if _same_role_category(title_lower, job_lower):
            ratio = max(ratio, 0.7)

        best = max(best, ratio)

    return best


def _same_role_category(title_a: str, title_b: str) -> bool:
    """Check if two titles belong to the same role category."""
    cat_a = _get_role_category(title_a)
    cat_b = _get_role_category(title_b)
    return cat_a is not None and cat_a == cat_b


def _get_role_category(title: str) -> Optional[str]:
    title = title.lower()
    for category, keywords in ROLE_CATEGORY_MAP.items():
        if any(kw in title for kw in keywords):
            return category
    return None


def seniority_fit_score(user_level: Optional[str], job_level: Optional[str]) -> float:
    """Score based on seniority distance. Exact=1.0, adjacent=0.6, far=0."""
    if not user_level or not job_level:
        return 0.5  # unknown = neutral

    user_idx = _seniority_index(user_level)
    job_idx = _seniority_index(job_level)

    if user_idx is None or job_idx is None:
        return 0.5

    distance = abs(user_idx - job_idx)
    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.6
    elif distance == 2:
        return 0.3
    return 0.0


def _seniority_index(level: str) -> Optional[int]:
    level = level.lower().strip()
    for i, s in enumerate(SENIORITY_ORDER):
        if s in level or level in s:
            return i
    if "mid" in level or "middle" in level:
        return 2
    return None


def location_match_score(
    user_location: Optional[str],
    job_country: Optional[str],
    job_is_remote: Optional[bool],
) -> float:
    """Score location fit. Remote jobs get 0.8 baseline."""
    if job_is_remote:
        return 0.8

    if not user_location or not job_country:
        return 0.3  # unknown = low

    user_country = _extract_country(user_location)
    job_country_norm = _normalize_country(job_country)

    if user_country and job_country_norm and user_country == job_country_norm:
        return 1.0

    return 0.0


def _extract_country(location_str: str) -> Optional[str]:
    """Extract ISO alpha-2 country code from a free-form location string.

    Used for user.location which is typed by humans (e.g. "Bangalore, India").
    """
    location_lower = location_str.lower().strip()
    if len(location_lower) == 2 and location_lower.isalpha():
        return location_lower
    for code, aliases in _LEGACY_COUNTRY_ALIASES.items():
        if code in location_lower:
            return code
        for alias in aliases:
            if alias in location_lower:
                return code
    return None


def _normalize_country(country: str) -> Optional[str]:
    """Normalize a job.country value to lowercase ISO alpha-2.

    Post-v3 jobs already store the canonical form, so this is mostly a
    lowercase. Pre-v3 jobs may have full names like "United States"; the
    legacy alias table covers the most common ones.
    """
    country_lower = country.lower().strip()
    if len(country_lower) == 2 and country_lower.isalpha():
        return country_lower
    for code, aliases in _LEGACY_COUNTRY_ALIASES.items():
        if country_lower == code or country_lower in aliases:
            return code
    return country_lower


def experience_fit_score(
    user_years: Optional[int],
    job_required_years: Optional[int],
) -> float:
    """Score experience fit. Within range=1.0, over-qualified=0.7, under=0.3."""
    if user_years is None or job_required_years is None:
        return 0.5  # unknown = neutral

    diff = user_years - job_required_years
    if -1 <= diff <= 3:
        return 1.0
    elif diff > 3:
        return 0.7  # over-qualified
    elif diff >= -3:
        return 0.3  # slightly under
    return 0.1  # significantly under


def salary_fit_score(
    user_salary: Optional[int],
    job_salary_min: Optional[str],
    job_salary_max: Optional[str],
) -> float:
    """Score salary fit if both sides have data."""
    if user_salary is None:
        return 0.5  # unknown = neutral

    try:
        jmin = int(float(job_salary_min)) if job_salary_min else None
        jmax = int(float(job_salary_max)) if job_salary_max else None
    except (ValueError, TypeError):
        return 0.5

    if jmin is None and jmax is None:
        return 0.5

    job_mid = ((jmin or 0) + (jmax or jmin or 0)) / 2
    if job_mid == 0:
        return 0.5

    ratio = user_salary / job_mid
    if 0.8 <= ratio <= 1.2:
        return 1.0
    elif 0.6 <= ratio <= 1.5:
        return 0.5
    return 0.1


def compute_total(signals: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> int:
    """Combine signal scores into a 0-100 total using weights."""
    w = weights or WEIGHTS
    total = sum(signals.get(k, 0.0) * v for k, v in w.items())
    return max(0, min(100, round(total * 100)))


def is_stretch_match(
    user_years: Optional[int],
    job_required_years: Optional[int],
    user_education: Optional[str],
    job_required_education: Optional[str],
) -> bool:
    """Check if user clearly doesn't meet hard requirements."""
    if user_years is not None and job_required_years is not None:
        if job_required_years - user_years >= 4:
            return True

    if job_required_education and user_education:
        edu_order = ["bachelor", "master", "phd", "doctorate"]
        job_idx = next((i for i, e in enumerate(edu_order) if e in job_required_education.lower()), -1)
        user_idx = next((i for i, e in enumerate(edu_order) if e in user_education.lower()), -1)
        if job_idx > user_idx >= 0:
            return True

    return False
