"""RSS feed reader for job boards"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
import feedparser

from src.agents import BaseFetcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RAW_TEXT_CHAR_LIMIT = 1_000_000


class RSSFeedFetcher(BaseFetcher):
    """
    Generic RSS feed fetcher for job boards.
    Returns ALL raw entries — no filtering, no normalization.
    """

    # Comprehensive list of tech job RSS feeds (ALL tech roles, not just backend)
    DEFAULT_FEEDS = [
        # We Work Remotely — ALL tech categories
        "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-front-end-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-product-jobs.rss",
        "https://weworkremotely.com/categories/remote-design-jobs.rss",
        # RemoteOK RSS — various categories
        "https://remoteok.com/remote-dev-jobs.rss",
        "https://remoteok.com/remote-backend-jobs.rss",
        "https://remoteok.com/remote-frontend-jobs.rss",
        "https://remoteok.com/remote-devops-jobs.rss",
        "https://remoteok.com/remote-data-jobs.rss",
        "https://remoteok.com/remote-machine-learning-jobs.rss",
        "https://remoteok.com/remote-security-jobs.rss",
        "https://remoteok.com/remote-golang-jobs.rss",
        "https://remoteok.com/remote-python-jobs.rss",
        "https://remoteok.com/remote-react-jobs.rss",
        "https://remoteok.com/remote-rust-jobs.rss",
    ]

    def __init__(self, feed_urls: Optional[List[str]] = None) -> None:
        super().__init__("rss_feed")
        self.feed_urls = feed_urls or self.DEFAULT_FEEDS

    async def fetch_jobs(self) -> List[Dict[str, Any]]:
        """Fetch ALL jobs from all configured RSS feeds — no filtering"""
        logger.info("[%s] Fetching from %d RSS feeds (ALL jobs, no filtering)", self.source_name, len(self.feed_urls))
        
        all_jobs = []
        async with aiohttp.ClientSession() as session:
            for feed_url in self.feed_urls:
                jobs = await self._fetch_feed(session, feed_url)
                all_jobs.extend(jobs)
        
        # Deduplicate by source_id
        seen_ids = set()
        unique_jobs = []
        for job in all_jobs:
            sid = job.get("source_id") or job.get("_entry_id", "")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                unique_jobs.append(job)
            elif not sid:
                unique_jobs.append(job)
        
        logger.info("[%s] Fetched %d unique jobs from %d feeds (ALL - no filtering)", 
                    self.source_name, len(unique_jobs), len(self.feed_urls))
        return unique_jobs

    async def _fetch_feed(self, session: aiohttp.ClientSession, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed — return ALL entries raw"""
        try:
            async with session.get(feed_url, timeout=30) as response:
                if response.status != 200:
                    logger.warning("[%s] HTTP %s for feed: %s", self.source_name, response.status, feed_url)
                    return []
                
                content = await response.text()
            
            # Parse with feedparser
            feed = feedparser.parse(content)
            
            if feed.bozo:  # Feed parsing error
                logger.warning("[%s] Parse error for feed: %s", self.source_name, feed_url)
                return []
            
            jobs = []
            for entry in feed.entries[:200]:  # Increased limit to 200 entries per feed
                job = self._parse_entry(entry, feed_url)
                if job:
                    jobs.append(job)  # NO filtering — return ALL entries
            
            logger.info("[%s] Fetched %d entries from: %s", self.source_name, len(jobs), feed_url)
            return jobs
            
        except Exception as exc:
            logger.error("[%s] Error fetching feed %s: %s", self.source_name, feed_url, exc, exc_info=True)
            return []

    def _parse_entry(self, entry: Any, feed_url: str) -> Optional[Dict[str, Any]]:
        """Parse an RSS entry into raw job data with ALL available fields"""
        try:
            # Extract basic fields
            title = entry.get("title", "").strip()
            if not title:
                return None
            
            # Get description from summary or content
            description = ""
            if hasattr(entry, "summary"):
                description = entry.summary
            elif hasattr(entry, "content") and entry.content:
                description = entry.content[0].value
            
            # Keep both raw HTML and cleaned text
            description_html = description
            description_clean = self._strip_html(description)
            
            # Extract URL
            apply_url = entry.get("link", "")
            
            # Generate unique ID from URL or content
            if apply_url:
                source_id = hashlib.md5(apply_url.encode()).hexdigest()[:16]
            else:
                source_id = hashlib.md5((title + description_clean[:100]).encode()).hexdigest()[:16]
            
            # Extract published date
            posted_at = self._parse_date(entry)
            
            # Try to extract company from title or description
            company = self._extract_company(title, description_clean)
            
            # Extract location
            location_raw = self._extract_location_from_entry(entry, title, description_clean)
            
            # Extract all tags/categories
            tags = []
            if hasattr(entry, "tags"):
                tags = [tag.get("term", "") for tag in entry.tags if tag.get("term")]
            
            # Return RAW data with ALL fields
            return {
                "_source": self.source_name,
                "_feed_url": feed_url,
                "_entry_id": source_id,
                "_description_html": self._cap_text(description_html),
                "_raw_entry": self._cap_raw_value(dict(entry)),
                "_tags": tags,
                "source_id": source_id,
                "title": title[:500],
                "company": company[:200],
                "description": self._cap_text(description_clean),
                "location_raw": location_raw,
                "remote": "remote" in location_raw.lower() if location_raw else False,
                "apply_url": apply_url or "",
                "posted_at": posted_at,
                # Pass through any extra fields from the entry
                "author": getattr(entry, "author", None),
                "entry_id": getattr(entry, "id", None),
            }
            
        except Exception as exc:
            logger.warning("[%s] Error parsing entry: %s", self.source_name, exc)
            return None

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _cap_text(value: str) -> str:
        return value[:RAW_TEXT_CHAR_LIMIT]

    @classmethod
    def _cap_raw_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._cap_text(value)
        if isinstance(value, dict):
            return {k: cls._cap_raw_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._cap_raw_value(v) for v in value]
        return value

    def _parse_date(self, entry: Any) -> datetime:
        """Parse published date from entry"""
        # Try different date fields
        for date_field in ["published_parsed", "updated_parsed", "created_parsed"]:
            if hasattr(entry, date_field):
                date_tuple = getattr(entry, date_field)
                if date_tuple:
                    try:
                        import time
                        timestamp = time.mktime(date_tuple)
                        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    except Exception:
                        pass
        
        # Fallback to current time
        return datetime.now(timezone.utc)

    def _extract_company(self, title: str, description: str) -> str:
        """Try to extract company name from title or description"""
        import re
        
        # Pattern: "Company Name - Job Title" or "Job Title at Company Name"
        match = re.match(r"^([^-@]+?)\s*[-–—]\s*", title)
        if match:
            potential_company = match.group(1).strip()
            if 2 < len(potential_company) < 50 and not any(word in potential_company.lower() for word in ["engineer", "developer", "job", "senior", "junior"]):
                return potential_company
        
        match = re.search(r"\bat\s+([A-Z][a-zA-Z0-9\s&.,]+?)(?:\s*[-|]|\s*$)", title)
        if match:
            return match.group(1).strip()
        
        # Look in description for "Company:" or similar
        match = re.search(r"(?:Company|Organization|Employer):\s*([^\n]+)", description, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            if 2 < len(company) < 100:
                return company
        
        return "See Description"

    def _extract_location_from_entry(self, entry: Any, title: str, description: str) -> str:
        """Extract location from entry"""
        import re
        
        # Check for REMOTE keyword
        text = f"{title} {description}"
        if re.search(r"\b(remote|work from home|wfh|distributed)\b", text, re.IGNORECASE):
            return "Remote"
        
        # Look for location in tags
        if hasattr(entry, "tags"):
            for tag in entry.tags:
                if "location" in tag.get("term", "").lower():
                    return tag.get("label", "See Description")
        
        # Look for location patterns in description
        match = re.search(r"(?:Location|Based in|Office in):\s*([^\n]+)", description, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:100]
        
        # Look for city, state/country patterns
        match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2}|[A-Z][a-z]+)\b", description)
        if match:
            return f"{match.group(1)}, {match.group(2)}"
        
        return "See Description"
