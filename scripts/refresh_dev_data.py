"""Mirror prod Class A collections into the dev database.

Class A = canonical public data the prod ingest cron writes:
  - jobs (filtered to the last JOBS_WINDOW_DAYS to keep dev small)
  - discovered_companies
  - companies_popularity
  - query_metrics
  - ingest_metrics

Class B (users, resumes, interviews, applied_jobs, ...) and Class C (ledger,
stripe-related) are NOT touched. Each dev env keeps its own test accounts
and per-user state. job_matches is dev-managed too; the matcher will
populate it against dev's own users + the copied jobs.

Env vars:
  DATABASE_URL       prod cluster connection string (source)
  DATABASE_URL_DEV   dev cluster connection string (target). Typically the
                     same Atlas cluster as prod with a different db name in
                     the URI path, e.g. .../remotestar_candidate_dev.

Idempotent. Drops the listed Class A collections on dev, then bulk-copies
from prod. Safe to re-run; built to be a weekly cron job:

  0 2 * * 0 (cd /root/jobs.ai-prod && /root/jobs.ai-prod/venv/bin/python \
            scripts/refresh_dev_data.py) >> /var/log/jobs-ai-refresh.log 2>&1
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.uri_parser import parse_uri

from src.database.models import ensure_indexes


JOBS_WINDOW_DAYS = 14
BATCH_SIZE = 1000

# Collections copied wholesale (no time filter).
SIMPLE_COPY_COLLECTIONS = [
    "discovered_companies",
    "companies_popularity",
    "query_metrics",
    "ingest_metrics",
]


async def _copy_collection(src_db, dst_db, name, filter_=None) -> int:
    """Drop dst[name] then bulk-copy matching docs from src[name]."""

    await dst_db[name].drop()
    cursor = src_db[name].find(filter_ or {})
    batch: list = []
    total = 0
    async for doc in cursor:
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            await dst_db[name].insert_many(batch, ordered=False)
            total += len(batch)
            batch = []
    if batch:
        await dst_db[name].insert_many(batch, ordered=False)
        total += len(batch)
    return total


async def main() -> None:
    prod_uri = os.environ.get("DATABASE_URL")
    dev_uri = os.environ.get("DATABASE_URL_DEV")

    if not prod_uri or not dev_uri:
        print("DATABASE_URL and DATABASE_URL_DEV must both be set", file=sys.stderr)
        sys.exit(1)
    if prod_uri == dev_uri:
        print("DATABASE_URL and DATABASE_URL_DEV must point at different databases", file=sys.stderr)
        sys.exit(1)

    prod_client = AsyncIOMotorClient(prod_uri)
    dev_client = AsyncIOMotorClient(dev_uri)

    prod_db_name = parse_uri(prod_uri).get("database") or "jobs_db"
    dev_db_name = parse_uri(dev_uri).get("database") or "jobs_db"
    prod_db = prod_client[prod_db_name]
    dev_db = dev_client[dev_db_name]

    print(f"prod: {prod_db_name}")
    print(f"dev:  {dev_db_name}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=JOBS_WINDOW_DAYS)
    print(f"\nCopying jobs with posted_at >= {cutoff.isoformat()}")
    try:
        n = await _copy_collection(prod_db, dev_db, "jobs", filter_={"posted_at": {"$gte": cutoff}})
        print(f"  jobs: {n:,} copied")
    except Exception as exc:
        print(f"  jobs: FAILED ({exc})", file=sys.stderr)
        raise

    for name in SIMPLE_COPY_COLLECTIONS:
        try:
            n = await _copy_collection(prod_db, dev_db, name)
            print(f"  {name}: {n:,} copied")
        except Exception as exc:
            # Allow individual collections to fail without aborting the rest.
            print(f"  {name}: FAILED ({exc})", file=sys.stderr)

    print("\nRebuilding indexes on dev")
    await ensure_indexes(dev_db)

    prod_client.close()
    dev_client.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
