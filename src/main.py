"""Application entrypoint and lightweight CLI."""

import argparse
import asyncio
import json

import uvicorn

from src.api.main import app  # noqa: F401  (exposed for uvicorn)
from src.database.operations import db
from src.services.ingestion import run_ingestion_cycle
from src.utils.config import settings


async def _run_ingestion_once() -> dict:
    await db.connect()
    summary = await run_ingestion_cycle()
    await db.disconnect()
    return summary


async def _run_matching(user_id: str | None = None) -> dict:
    from src.services.matcher import run_matching_for_all, run_matching_for_user
    await db.connect()
    if user_id:
        result = await run_matching_for_user(user_id)
    else:
        result = await run_matching_for_all()
    await db.disconnect()
    return result


async def _run_embed_jobs() -> dict:
    from src.services.matcher import embed_jobs
    await db.connect()
    result = await embed_jobs()
    await db.disconnect()
    return result


async def _run_cleanup() -> dict:
    await db.connect()
    result = await db.cleanup_expired_jobs()
    await db.disconnect()
    return result


async def _run_reenrich_stale(limit: int | None) -> dict:
    from src.services.reenrich import reenrich_stale
    await db.connect()
    result = await reenrich_stale(db, limit=limit)
    await db.disconnect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend & DevOps job aggregator")
    parser.add_argument(
        "--ingest-once",
        action="store_true",
        help="Run a single ingestion cycle then exit",
    )
    parser.add_argument(
        "--match",
        action="store_true",
        help="Run job matching for all enabled users then exit",
    )
    parser.add_argument(
        "--match-user",
        type=str,
        default=None,
        help="Run job matching for a specific user ID then exit",
    )
    parser.add_argument(
        "--embed-jobs",
        action="store_true",
        help="Embed all jobs into Pinecone then exit",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Start Redis pub/sub worker for on-demand matching",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove expired jobs then exit",
    )
    parser.add_argument(
        "--reenrich-stale",
        action="store_true",
        help="Re-run AI enrichment on jobs with outdated prompt_version then exit",
    )
    parser.add_argument(
        "--reenrich-limit",
        type=int,
        default=None,
        help="Cap the number of jobs re-enriched (default: all stale)",
    )

    args = parser.parse_args()

    if args.worker:
        from src.services.match_worker import start_worker
        asyncio.run(start_worker())
        return

    if args.embed_jobs:
        result = asyncio.run(_run_embed_jobs())
        print(json.dumps(result, indent=2, default=str))  # noqa: T201
        return

    if args.match or args.match_user:
        result = asyncio.run(_run_matching(args.match_user))
        print(json.dumps(result, indent=2, default=str))  # noqa: T201
        return

    if args.cleanup:
        result = asyncio.run(_run_cleanup())
        print(json.dumps(result, indent=2))  # noqa: T201
        return

    if args.reenrich_stale:
        result = asyncio.run(_run_reenrich_stale(args.reenrich_limit))
        print(json.dumps(result, indent=2, default=str))  # noqa: T201
        return

    if args.ingest_once:
        summary = asyncio.run(_run_ingestion_once())
        print(json.dumps(summary, indent=2))  # noqa: T201
        return

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.port or settings.api_port,
        reload=settings.environment != "production",
    )


if __name__ == "__main__":
    main()
