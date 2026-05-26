import asyncio
from datetime import datetime, timezone

import feedparser

from src.agents.ats_scraper import ATSScraperFetcher
from src.agents.hackernews import HackerNewsFetcher
from src.agents.rss_feed import RSSFeedFetcher


def test_rss_entry_keeps_full_description_text_and_html():
    fetcher = RSSFeedFetcher()
    long_text = "Backend engineer " + ("full jd " * 2000)
    entry = feedparser.FeedParserDict({
        "title": "Backend Engineer",
        "summary": f"<p>{long_text}</p>",
        "link": "https://example.com/job",
        "id": "entry-1",
    })

    job = fetcher._parse_entry(entry, "https://example.com/feed")

    assert job["description"] == long_text.strip()
    assert job["_description_html"] == f"<p>{long_text}</p>"
    assert job["_raw_entry"]["summary"] == f"<p>{long_text}</p>"


def test_hackernews_entry_keeps_full_description_text():
    fetcher = HackerNewsFetcher()
    long_text = "RemoteStar | Backend Engineer | Remote\n" + ("full jd " * 2000)
    comment = {
        "id": 123,
        "text": long_text,
        "time": int(datetime.now(timezone.utc).timestamp()),
    }

    job = fetcher._parse_comment_to_job(comment)

    assert "full jd" in job["description"]
    assert len(job["description"]) > 8000
    assert job["_raw_comment"] == comment


def test_ats_description_enrichment_keeps_full_detail_payload():
    fetcher = ATSScraperFetcher()
    long_html = "<p>" + ("full jd " * 2000) + "</p>"
    job = {
        "source_id": "ats_1",
        "description": "",
        "raw_data": {"ats": "greenhouse", "item": {"id": 1}},
    }

    async def fetch_one(_session, target):
        target["raw_data"]["detail"] = {"content": long_html}
        return long_html

    asyncio.run(fetcher._enrich_descriptions(None, [job], fetch_one))

    assert len(job["description"]) > 8000
    assert job["raw_data"]["detail"]["content"] == long_html
