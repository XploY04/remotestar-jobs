"""Re-enrich jobs with a stale or missing prompt_version.

Reuses the existing EnrichmentPipeline so newly-enriched fields match what
fresh ingestion would produce. Updates jobs in place (does not go through
save_jobs, which would treat them as duplicates and skip).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.enrichment.ai_processor import PROMPT_VERSION
from src.enrichment.enrichment_pipeline import EnrichmentPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Fields that should never be overwritten by re-enrichment (system metadata).
PROTECTED = {
    "_id", "source", "source_id", "id", "raw_data", "title_company_hash",
    "fetched_at", "is_archived", "is_deleted", "deleted_at", "archived_at",
    "archive_reason", "delete_reason", "pinecone_embedded_at",
}


async def reenrich_stale(
    db,
    limit: Optional[int] = None,
    batch_size: int = 100,
    mode: str = "stale",
) -> Dict[str, Any]:
    """Re-run AI enrichment on a target subset of jobs.

    mode:
      "stale"    — jobs where prompt_version is missing or != current
      "low_info" — jobs with empty skills or null seniority_level
                   (raw_data must still be present so we can re-run)

    `limit` caps the number of jobs processed (None = all).
    Returns {scanned, reenriched, errors, per_source, mode}.
    """

    if db.db is None:
        raise RuntimeError("Database not connected")

    base = {
        "is_archived": {"$ne": True},
        "is_deleted": {"$ne": True},
        "raw_data": {"$exists": True, "$ne": None},
    }
    if mode == "stale":
        criteria = {
            **base,
            "$or": [
                {"prompt_version": {"$exists": False}},
                {"prompt_version": {"$ne": PROMPT_VERSION}},
            ],
        }
    elif mode == "low_info":
        criteria = {
            **base,
            "$or": [
                {"skills": {"$exists": False}},
                {"skills": {"$size": 0}},
                {"seniority_level": {"$in": [None, ""]}},
            ],
        }
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    total = await db.jobs.count_documents(criteria)
    if total == 0:
        logger.info("No stale jobs to re-enrich")
        return {"scanned": 0, "reenriched": 0, "errors": 0, "per_source": {}}

    if limit:
        total = min(total, limit)
    logger.info("Re-enriching %d jobs to prompt_version=%s (batch=%d)", total, PROMPT_VERSION, batch_size)

    pipeline = EnrichmentPipeline(use_ai=True)
    if not (pipeline.ai_processor and pipeline.ai_processor.enabled):
        raise RuntimeError("AI processor not enabled — cannot re-enrich")

    reenriched = 0
    errors = 0
    per_source: Dict[str, int] = {}

    # Fetch all matching ids up front (Atlas shared tier disallows no_cursor_timeout
    # cursors and our pipeline can run >10min between fetches).
    id_cursor = db.jobs.find(criteria, {"_id": 1}).limit(limit or 0)
    target_ids = [doc["_id"] for doc in await id_cursor.to_list(length=None)]

    for i in range(0, len(target_ids), batch_size):
        chunk_ids = target_ids[i:i + batch_size]
        batch_docs = await db.jobs.find({"_id": {"$in": chunk_ids}}).to_list(length=None)
        if not batch_docs:
            continue

        await _reenrich_batch(db, pipeline, batch_docs, per_source)
        reenriched += len(batch_docs)
        logger.info("Re-enriched %d/%d", reenriched, total)

    summary = {
        "mode": mode,
        "scanned": total,
        "reenriched": reenriched,
        "errors": errors,
        "per_source": per_source,
    }
    logger.info("Re-enrich complete: %s", summary)
    return summary


async def _reenrich_batch(db, pipeline: EnrichmentPipeline,
                          docs: List[Dict[str, Any]], per_source: Dict[str, int]) -> None:
    """Group docs by source, re-run through AI, update in place."""

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for doc in docs:
        src = doc.get("source") or "unknown"
        by_source.setdefault(src, []).append(doc)

    now = datetime.now(timezone.utc)

    for source, source_docs in by_source.items():
        raw_jobs = [d.get("raw_data") for d in source_docs if d.get("raw_data")]
        if not raw_jobs:
            continue

        results = await pipeline.process_source(
            source_name=source,
            raw_jobs=raw_jobs,
            batch_size=5,
            max_concurrent=10,
            on_batch_ready=None,
        )

        for original, finalized in zip(source_docs, results):
            if not finalized:
                continue
            update_doc = {k: v for k, v in finalized.items() if k not in PROTECTED}
            update_doc["reenriched_at"] = now
            await db.jobs.update_one(
                {"_id": original["_id"]},
                {"$set": update_doc},
            )
            per_source[source] = per_source.get(source, 0) + 1
