"""Compute ranking_score from job_matches aggregation.

Recomputes per-job popularity based on how often each job appears in users'
match sets and the mean match score, then writes the result back to the
jobs collection. Runs after each weekly match cycle.

Only paying users trigger matches (credit-gated in matcher.py), so this
signal reflects paying-candidate demand, not all candidates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from pymongo import UpdateOne

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def compute_job_popularity(db) -> Dict[str, Any]:
    """Aggregate job_matches by job_id and update each job's ranking_score.

    Returns {scored_count, max_appearances, mean_appearances}.
    """

    if db.db is None:
        raise RuntimeError("Database not connected")

    pipeline = [
        {"$group": {
            "_id": "$job_id",
            "appearances": {"$sum": 1},
            "mean_score": {"$avg": "$score"},
        }},
    ]
    rows = await db.db["job_matches"].aggregate(pipeline).to_list(length=None)

    if not rows:
        logger.info("No job_matches yet — skipping popularity aggregation")
        return {"scored_count": 0, "max_appearances": 0, "mean_appearances": 0.0}

    max_appearances = max(r["appearances"] for r in rows)
    if max_appearances <= 0:
        logger.warning("max_appearances is zero — skipping")
        return {"scored_count": 0, "max_appearances": 0, "mean_appearances": 0.0}

    now = datetime.now(timezone.utc)
    operations = []
    for r in rows:
        appearance_pct = (r["appearances"] / max_appearances) * 100
        mean_score = float(r["mean_score"] or 0.0)
        popularity = round(0.6 * appearance_pct + 0.4 * mean_score)
        popularity = max(20, min(100, popularity))
        operations.append(UpdateOne(
            {"_id": r["_id"]},
            {"$set": {"ranking_score": popularity, "ranking_updated_at": now}},
        ))

    result = await db.jobs.bulk_write(operations, ordered=False)

    total_appearances = sum(r["appearances"] for r in rows)
    summary = {
        "scored_count": result.modified_count,
        "max_appearances": max_appearances,
        "mean_appearances": round(total_appearances / len(rows), 2),
    }
    logger.info("Popularity recomputed: %s", summary)
    return summary
