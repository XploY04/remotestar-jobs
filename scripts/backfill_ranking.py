"""Backfill ranking_score on all active jobs.

Two passes:
  1. Run compute_job_popularity() to score every job that appears in the
     current 14-day job_matches window.
  2. Seed remaining jobs from quality_score using the same tiering used by
     the ingest pipeline.

Idempotent: safe to re-run. Run from the repo root:
  ./venv/bin/python -m scripts.backfill_ranking
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root on path when invoked as `python scripts/backfill_ranking.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.operations import db
from src.services.popularity import compute_job_popularity


async def _seed_from_quality(db) -> int:
    """Set ranking_score for active jobs that still lack it."""

    filter_ = {
        "is_archived": {"$ne": True},
        "is_deleted": {"$ne": True},
        "ranking_score": {"$exists": False},
    }
    pending = await db.jobs.count_documents(filter_)
    if pending == 0:
        return 0

    print(f"Seeding ranking_score on {pending:,} active jobs without one")

    pipeline = [
        {"$match": filter_},
        {"$set": {
            "ranking_score": {
                "$switch": {
                    "branches": [
                        {"case": {"$gte": ["$quality_score", 60]},
                         "then": {"$min": [50, "$quality_score"]}},
                        {"case": {"$gte": ["$quality_score", 40]}, "then": 40},
                    ],
                    "default": 30,
                },
            },
        }},
        {"$merge": {"into": "jobs", "on": "_id", "whenMatched": "merge"}},
    ]
    await db.jobs.aggregate(pipeline).to_list(length=None)
    remaining = await db.jobs.count_documents(filter_)
    return pending - remaining


async def main() -> None:
    await db.connect()
    try:
        print("Pass 1: aggregate job_matches → ranking_score")
        pop = await compute_job_popularity(db)
        print(f"  scored from matches: {pop}")

        print("Pass 2: seed remaining jobs from quality_score")
        seeded = await _seed_from_quality(db)
        print(f"  seeded {seeded:,} jobs")

        missing = await db.jobs.count_documents({
            "is_archived": {"$ne": True},
            "is_deleted": {"$ne": True},
            "ranking_score": {"$exists": False},
        })
        print(f"Active jobs still missing ranking_score: {missing}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
