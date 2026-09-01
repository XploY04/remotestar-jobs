"""Config-driven orchestrator — picks which sources run today."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Type

import yaml

from src.agents import BaseFetcher
from src.agents.remoteok import RemoteOKFetcher
from src.agents.jsearch import JSearchFetcher
from src.agents.hackernews import HackerNewsFetcher
from src.agents.rss_feed import RSSFeedFetcher
from src.agents.ats_scraper import ATSScraperFetcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FETCHER_MAP: dict[str, Type[BaseFetcher]] = {
    "remoteok": RemoteOKFetcher,
    "jsearch": JSearchFetcher,
    "hackernews": HackerNewsFetcher,
    "rss_feed": RSSFeedFetcher,
    "ats_scraper": ATSScraperFetcher,
}

SCHEDULE_PATH = Path(__file__).parent / "schedule.yaml"


def load_schedule(path: Path = SCHEDULE_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_todays_fetchers(today: date | None = None) -> List[Type[BaseFetcher]]:
    today = today or date.today()
    schedule = load_schedule()
    fetchers = []

    for source_name, config in schedule["sources"].items():
        if source_name not in FETCHER_MAP:
            logger.warning("Unknown source in schedule: %s", source_name)
            continue

        schedule_type = config.get("type", "weekly")
        days = config.get("days", [])

        if schedule_type == "monthly":
            should_run = today.day in days
        else:
            should_run = today.weekday() in days

        if should_run:
            fetchers.append(FETCHER_MAP[source_name])
            logger.info("Scheduled today: %s", source_name)
        else:
            logger.debug("Skipped today: %s", source_name)

    return fetchers
