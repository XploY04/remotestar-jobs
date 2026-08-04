"""Structured scoring for job-user matching.

Three hard gates: seniority, country, years. A job that fails any gate is
dropped (the gate function returns None). A job that clears every gate gets a
graded 0-1 score per gate; those three grades are averaged with the Pinecone
cosine (semantic relevance) into the final 0-100 match percentage.

The blended score doubles as the ranking key, so a same-level / home-country
job naturally outranks a one-level-up / remote one without a separate sort.
"""

from __future__ import annotations

from typing import Optional

# Canonical ladder, low to high. Both the candidate (resume-parse) and the job
# (enrichment) emit one of these. Rank = list index.
SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager"]

# Below this cosine a job is dropped even if it clears every gate, so a job
# that fits on seniority/country/years but is semantically unrelated can't ride
# the gate scores to a high blended %.
# ponytail: floor tuned by eye; retune from the live cosine distribution seen
# during backfill (Phase 3).
MIN_RELEVANCE = 0.25


def _seniority_rank(level: Optional[str]) -> Optional[int]:
    if not level:
        return None
    level = level.lower().strip()
    for i, s in enumerate(SENIORITY_ORDER):
        if s in level or level in s:
            return i
    if "middle" in level:
        return 2
    return None


def seniority_score(cand_level: Optional[str], job_level: Optional[str]) -> Optional[float]:
    """Up-only gate: a candidate sees their own level and exactly one level up.
    Same level = 1.0, one up = 0.7. Below the candidate, or 2+ up, is excluded
    (None). Unknown level on either side can't be judged, so it passes neutral."""
    cand = _seniority_rank(cand_level)
    job = _seniority_rank(job_level)
    if cand is None or job is None:
        return 0.5
    delta = job - cand
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.7
    return None


def country_score(
    cand_iso: Optional[str], job_iso: Optional[str], job_is_remote: Optional[bool]
) -> Optional[float]:
    """Eligibility gate + home-vs-remote grade.
    Same country (remote or onsite) = 1.0. Remote in another country = 0.85
    (the candidate can still apply). Onsite in a different country is excluded
    (None) — they can't take it. Unknown candidate country passes neutral."""
    if not cand_iso:
        return 0.5
    cand = cand_iso.lower().strip()
    job = (job_iso or "").lower().strip()
    if job and cand == job:
        return 1.0
    if job_is_remote:
        return 0.85
    if not job:
        return 0.5
    return None


def years_score(cand_years: Optional[int], job_min_years: Optional[int]) -> Optional[float]:
    """Minimum-experience gate. Meets or exceeds the job minimum = 1.0, one
    year short = 0.7 (still worth showing), 2+ short is excluded (None).
    Missing data on either side passes neutral.
    ponytail: 1-year grace is a knob; drop it for a strict minimum."""
    if job_min_years is None or cand_years is None:
        return 0.5
    gap = cand_years - job_min_years
    if gap >= 0:
        return 1.0
    if gap == -1:
        return 0.7
    return None


def match_percent(seniority: float, country: float, years: float, cosine: float) -> int:
    """Average the three gate grades with the cosine, scaled to 0-100.
    Callers pass only jobs that cleared the gates (no None here)."""
    blended = (seniority + country + years + cosine) / 4.0
    return max(0, min(100, round(blended * 100)))
