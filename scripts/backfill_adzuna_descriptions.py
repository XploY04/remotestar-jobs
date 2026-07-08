"""Backfill full descriptions for existing adzuna jobs.

Adzuna's API stores only a description snippet in raw_data, so re-enriching
those jobs can't produce skills/seniority. This script fetches each job's
redirect_url page and writes the text into raw_data.full_description; the
next --reenrich-stale run then extracts from the full listing.

Expired postings return errors or dead pages — those jobs keep the snippet
and are reported as failed.

Usage:
  python -m scripts.backfill_adzuna_descriptions [--limit N]
"""

import argparse
import asyncio

import aiohttp

from src.agents.adzuna import AdzunaFetcher
from src.database.operations import db
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CONCURRENCY = 10


async def backfill(limit: int | None) -> dict:
    await db.connect()
    try:
        criteria = {
            "source": "adzuna",
            "is_archived": {"$ne": True},
            "is_deleted": {"$ne": True},
            "raw_data.redirect_url": {"$exists": True, "$ne": None},
            "raw_data.full_description": {"$exists": False},
        }
        cursor = db.jobs.find(criteria, {"_id": 1, "raw_data.redirect_url": 1})
        if limit:
            cursor = cursor.limit(limit)
        targets = await cursor.to_list(length=None)
        logger.info("Backfilling full descriptions for %d adzuna jobs", len(targets))

        fetcher = AdzunaFetcher()
        sem = asyncio.Semaphore(CONCURRENCY)
        stats = {"targets": len(targets), "fetched": 0, "failed": 0}

        async with aiohttp.ClientSession() as session:
            async def fill(doc):
                async with sem:
                    text = await fetcher._fetch_page_text(session, doc["raw_data"]["redirect_url"])
                    if text:
                        await db.jobs.update_one(
                            {"_id": doc["_id"]},
                            {"$set": {"raw_data.full_description": text}},
                        )
                        stats["fetched"] += 1
                    else:
                        stats["failed"] += 1
                    done = stats["fetched"] + stats["failed"]
                    if done % 500 == 0:
                        logger.info("Progress: %d/%d (fetched=%d failed=%d)",
                                    done, len(targets), stats["fetched"], stats["failed"])
                    await asyncio.sleep(0.1)

            await asyncio.gather(*(fill(doc) for doc in targets))

        logger.info("Backfill complete: %s", stats)
        return stats
    finally:
        await db.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill adzuna full descriptions")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of jobs processed")
    args = parser.parse_args()
    asyncio.run(backfill(args.limit))


if __name__ == "__main__":
    main()
