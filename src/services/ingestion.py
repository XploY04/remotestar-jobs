"""Job ingestion orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.database.operations import db
from src.enrichment.enrichment_pipeline import EnrichmentPipeline
from src.services.orchestrator import get_todays_fetchers, FETCHER_MAP
from src.services.query_planner import QueryPlan, QueryPlannerService
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ALL_FETCHER_CLASSES = list(FETCHER_MAP.values())

# Single pipeline: Raw → AI → Structured (no normalizer)
pipeline = EnrichmentPipeline(use_ai=settings.enable_ai_enrichment)


async def run_ingestion_cycle(fetcher_classes: List | None = None) -> Dict[str, Any]:
    """Fetch raw → AI process → Save per batch. Each batch of ~5 jobs
    hits the DB as soon as Gemini finishes processing it.

    When fetcher_classes is None, the orchestrator picks today's scheduled sources.
    Pass ALL_FETCHER_CLASSES explicitly to run everything."""

    if fetcher_classes is None:
        fetcher_classes = get_todays_fetchers()
    fetchers = [cls() for cls in fetcher_classes]
    query_plans = await _generate_query_plans(fetchers)
    results = await asyncio.gather(*(_collect_jobs(fetcher, query_plans.get(fetcher.source_name)) for fetcher in fetchers))

    # Thread-safe counters (only mutated inside async tasks, one event loop)
    per_source: Dict[str, Dict[str, Any]] = {}
    total_new = 0
    total_skipped = 0
    total_restored = 0
    total_processed = 0

    async def _process_source(source_name: str, raw_jobs: List[Dict[str, Any]]) -> None:
        nonlocal total_new, total_skipped, total_restored, total_processed

        if not raw_jobs:
            per_source[source_name] = {"raw": 0, "processed": 0, "new": 0, "skipped": 0, "restored": 0}
            return

        source_stats = {"new": 0, "skipped": 0, "restored": 0}

        async def _save_batch(batch: List[Dict[str, Any]]) -> None:
            """Called by the pipeline after each batch (~5 jobs) is processed."""
            stats = await db.save_jobs(batch)
            source_stats["new"] += stats["new"]
            source_stats["skipped"] += stats["skipped"]
            source_stats["restored"] += stats["restored"]
            logger.info(
                "[%s] Batch saved: new=%d skipped=%d restored=%d",
                source_name, stats["new"], stats["skipped"], stats["restored"],
            )

        try:
            started_at = datetime.now(timezone.utc)
            processed = await pipeline.process_source(
                source_name, raw_jobs, on_batch_ready=_save_batch
            )
            completed_at = datetime.now(timezone.utc)
            try:
                await db.save_query_metrics(source_name, query_plans.get(source_name), raw_jobs, processed, source_stats)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[%s] Query metrics save failed: %s", source_name, exc)

            ai_metrics = pipeline.metrics_by_source.get(source_name, {})
            try:
                await db.db["ingest_metrics"].insert_one({
                    "source": source_name,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": (completed_at - started_at).total_seconds(),
                    "raw_count": len(raw_jobs),
                    "processed_count": len(processed),
                    "new_count": source_stats["new"],
                    "skipped_count": source_stats["skipped"],
                    "restored_count": source_stats["restored"],
                    "batch_ok": ai_metrics.get("batch_ok", 0),
                    "batch_fallback": ai_metrics.get("batch_fallback", 0),
                    "batch_failed": ai_metrics.get("batch_failed", 0),
                    "fallback_rate": ai_metrics.get("fallback_rate", 0.0),
                })
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[%s] Ingest metrics save failed: %s", source_name, exc)

            per_source[source_name] = {
                "raw": len(raw_jobs),
                "processed": len(processed),
                "new": source_stats["new"],
                "skipped": source_stats["skipped"],
                "restored": source_stats["restored"],
                "fallback_rate": ai_metrics.get("fallback_rate", 0.0),
            }
            total_new += source_stats["new"]
            total_skipped += source_stats["skipped"]
            total_restored += source_stats["restored"]
            total_processed += len(processed)

            logger.info("[%s] Done: raw=%d processed=%d new=%d skipped=%d restored=%d fallback_rate=%.2f",
                        source_name, len(raw_jobs), len(processed),
                        source_stats["new"], source_stats["skipped"], source_stats["restored"],
                        ai_metrics.get("fallback_rate", 0.0))

        except Exception as exc:
            logger.error("[%s] Failed: %s", source_name, exc, exc_info=True)
            per_source[source_name] = {
                "raw": len(raw_jobs), "processed": 0, "new": 0,
                "skipped": 0, "restored": 0, "error": str(exc),
            }

    # Process ALL sources concurrently; each batch saves independently
    await asyncio.gather(
        *(_process_source(name, jobs) for name, jobs in results)
    )

    cleanup_stats = await db.cleanup_expired_jobs()
    logger.info("Cleanup: deleted %d expired jobs", cleanup_stats["deleted"])

    summary = {
        "sources": per_source,
        "db": {"new": total_new, "skipped": total_skipped, "restored": total_restored},
        "cleanup": cleanup_stats,
        "total_jobs": total_processed,
        "query_planner": _query_planner_summary(query_plans),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Ingestion finished | total=%d new=%d skipped=%d restored=%d",
        total_processed, total_new, total_skipped, total_restored,
    )
    return summary


async def _generate_query_plans(fetchers: List) -> Dict[str, QueryPlan]:
    planner = QueryPlannerService()
    source_names = [fetcher.source_name for fetcher in fetchers]
    return await planner.generate_plans(source_names)


def _query_planner_summary(query_plans: Dict[str, QueryPlan]) -> Dict[str, Any]:
    plans = {source: plan.to_summary() for source, plan in query_plans.items()}
    return {
        "enabled": settings.enable_query_planner,
        "generated_queries": sum(len(plan.queries) for plan in query_plans.values()),
        "fallback_used": any(plan.fallback_used for plan in query_plans.values()) if query_plans else False,
        "sources": plans,
    }


async def _collect_jobs(fetcher, query_plan: QueryPlan | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        if query_plan and hasattr(fetcher, "set_query_plan"):
            fetcher.set_query_plan(query_plan)
        jobs = await fetcher.fetch_jobs()
        return fetcher.source_name, jobs
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("[%s] Fetch failure: %s", fetcher.source_name, exc, exc_info=True)
        return fetcher.source_name, []
