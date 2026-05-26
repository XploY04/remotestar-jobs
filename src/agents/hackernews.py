"""HackerNews 'Who is Hiring?' scraper"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from src.agents import BaseFetcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RAW_TEXT_CHAR_LIMIT = 1_000_000


class HackerNewsFetcher(BaseFetcher):
    """
    Fetcher for HackerNews 'Who is Hiring?' monthly threads.
    
    HN posts a monthly thread where companies post job listings in comments.
    Returns ALL job comments as raw data — no filtering.
    """

    HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
    ALGOLIA_API = "https://hn.algolia.com/api/v1"
    MAX_COMMENTS = 1000  # Fetch more comments from the thread

    def __init__(self) -> None:
        super().__init__("hackernews")

    async def fetch_jobs(self) -> List[Dict[str, Any]]:
        """Fetch ALL jobs from latest 'Who is Hiring?' thread — no filtering"""
        logger.info("[%s] Fetching ALL jobs from HackerNews (no filtering)", self.source_name)

        async with aiohttp.ClientSession() as session:
            # Find the latest "Who is Hiring?" thread
            thread_id = await self._find_latest_thread(session)
            if not thread_id:
                logger.warning("[%s] Could not find 'Who is Hiring?' thread", self.source_name)
                return []

            # Fetch all comments from the thread
            comments = await self._fetch_thread_comments(session, thread_id)
            logger.info("[%s] Found %d comments in thread", self.source_name, len(comments))

            # Parse ALL job postings from comments — no backend filter
            jobs = []
            for comment in comments:
                job = self._parse_comment_to_job(comment)
                if job:
                    jobs.append(job)

        logger.info("[%s] Extracted %d total jobs (ALL - no filtering)", self.source_name, len(jobs))
        return jobs

    async def _find_latest_thread(self, session: aiohttp.ClientSession) -> Optional[int]:
        """Find the latest 'Who is Hiring?' thread using Algolia HN Search API"""
        try:
            # Search for recent "Who is hiring?" threads
            url = f"{self.ALGOLIA_API}/search"
            params = {
                "query": "Ask HN: Who is hiring?",
                "tags": "story",
                "numericFilters": f"created_at_i>{int((datetime.now().timestamp() - 60*60*24*45))}",  # Last 45 days
            }

            async with session.get(url, params=params, timeout=15) as response:
                if response.status != 200:
                    logger.error("[%s] Algolia API returned %s", self.source_name, response.status)
                    return None

                data = await response.json()
                hits = data.get("hits", [])

                # Find the most recent thread from whoishiring user
                for hit in hits:
                    title = hit.get("title", "").lower()
                    author = hit.get("author", "")
                    if "who is hiring" in title and author == "whoishiring":
                        thread_id = hit.get("objectID")
                        logger.info("[%s] Found thread: '%s' (ID: %s)", self.source_name, hit.get("title"), thread_id)
                        return int(thread_id)

                # Fallback: any "who is hiring" thread
                if hits:
                    thread_id = hits[0].get("objectID")
                    logger.info("[%s] Using thread: '%s' (ID: %s)", self.source_name, hits[0].get("title"), thread_id)
                    return int(thread_id)

        except Exception as exc:
            logger.error("[%s] Error finding thread: %s", self.source_name, exc, exc_info=True)

        return None

    async def _fetch_thread_comments(self, session: aiohttp.ClientSession, thread_id: int) -> List[Dict[str, Any]]:
        """Fetch all top-level comments from a thread"""
        try:
            url = f"{self.HN_API_BASE}/item/{thread_id}.json"
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return []

                thread_data = await response.json()
                comment_ids = thread_data.get("kids", [])[:1000]  # Fetch up to 1000 comments

                # Fetch all comments in parallel
                tasks = [self._fetch_comment(session, cid) for cid in comment_ids]
                comments = await asyncio.gather(*tasks)

                # Filter out None results
                return [c for c in comments if c]

        except Exception as exc:
            logger.error("[%s] Error fetching comments: %s", self.source_name, exc, exc_info=True)
            return []

    async def _fetch_comment(self, session: aiohttp.ClientSession, comment_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single comment"""
        try:
            url = f"{self.HN_API_BASE}/item/{comment_id}.json"
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                return await response.json()
        except Exception:
            return None

    def _parse_comment_to_job(self, comment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a HN comment into a raw job posting.
        Returns ALL data — no filtering, no normalization.
        """
        text = comment.get("text", "")
        if not text or len(text) < 50:  # Too short to be a real job posting
            return None

        # Remove HTML tags
        clean_text = re.sub(r"<[^>]+>", " ", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Skip if it looks like a job-seeker comment, not a job posting
        if any(phrase in clean_text.lower() for phrase in ["looking for", "seeking employment", "available for hire", "open to opportunities"]):
            return None

        # Extract components
        lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
        if not lines:
            return None

        # First line often has: Company | Title | Location
        first_line = lines[0]
        company = self._extract_company(first_line)
        title = self._extract_title(first_line, clean_text)
        location = self._extract_location(clean_text)
        remote = self._is_remote(clean_text)

        # Get description (everything)
        description = self._clean_description(clean_text)

        # Extract apply URL or email
        apply_url = self._extract_apply_url(text)  # Use original HTML text for URLs

        # Must have at least some content
        if not title and not description:
            return None

        # Return raw data with ALL extracted info + original comment data
        return {
            '_source': self.source_name,
            '_raw_html': self._cap_text(text),  # Original HTML content
            '_raw_text': self._cap_text(clean_text),  # Cleaned text
            '_raw_comment': self._cap_raw_value(comment),
            '_first_line': first_line,
            'id': str(comment.get('id', 0)),
            'title': title[:200] if title else first_line[:200],
            'company': company[:100] if company else "See Description",
            'description': self._cap_text(description),
            'location_raw': location,
            'remote': remote,
            'apply_url': apply_url or f"https://news.ycombinator.com/item?id={comment.get('id')}",
            'hn_comment_id': comment.get('id'),
            'hn_parent_id': comment.get('parent'),
            'hn_author': comment.get('by', ''),
            'hn_time': comment.get('time'),
            'posted_at': datetime.fromtimestamp(comment.get("time", 0), tz=timezone.utc) if comment.get("time") else datetime.now(timezone.utc),
        }

    def _extract_company(self, first_line: str) -> Optional[str]:
        """Extract company name from first line (often before | or :)"""
        # Pattern: "CompanyName |" or "CompanyName:"
        match = re.match(r"^([^|:\n]+?)(?:\s*[|:])", first_line)
        if match:
            company = match.group(1).strip()
            # Remove common prefixes
            company = re.sub(r"^(at|@)\s+", "", company, flags=re.IGNORECASE)
            if len(company) > 3 and not company.lower().startswith("http"):
                return company
        return None

    def _extract_title(self, first_line: str, full_text: str) -> Optional[str]:
        """Extract job title"""
        # Look for common patterns
        patterns = [
            r"(?:senior|sr\.?|junior|jr\.?|staff|principal|lead)?\s*(?:software|backend|devops|platform|site reliability|cloud|infrastructure)\s*engineer",
            r"(?:senior|sr\.?|junior|jr\.?|staff|principal|lead)?\s*(?:backend|devops|full[- ]?stack)\s*developer",
            r"(?:senior|sr\.?|junior|jr\.?)\s*sre",
            r"(?:senior|sr\.?|junior|jr\.?)\s*(?:golang|python|java|rust|node\.?js)\s*(?:engineer|developer)",
        ]

        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # Fallback: look in full text for job titles
        for pattern in patterns:
            match = re.search(pattern, full_text[:300], re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # Last resort: use first line if it looks like a title
        if "|" in first_line:
            parts = first_line.split("|")
            if len(parts) >= 2:
                potential_title = parts[1].strip()
                if 5 < len(potential_title) < 100:
                    return potential_title

        return "Backend/DevOps Engineer"

    def _extract_location(self, text: str) -> str:
        """Extract location"""
        # Check for REMOTE
        if re.search(r"\bREMOTE\b", text, re.IGNORECASE):
            return "Remote"

        # Check for specific cities/countries
        location_match = re.search(r"(?:ONSITE|Location|Based in)[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\||$|\n)", text)
        if location_match:
            return location_match.group(1).strip()[:100]

        return "See Description"

    def _is_remote(self, text: str) -> bool:
        """Check if job is remote"""
        return bool(re.search(r"\bREMOTE\b", text, re.IGNORECASE))

    def _clean_description(self, text: str) -> str:
        """Clean description text"""
        # Remove URLs and emails for cleaner description
        text = re.sub(r"https?://[^\s]+", "", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

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

    def _extract_apply_url(self, text: str) -> Optional[str]:
        """Extract application URL"""
        # Look for apply URLs (often after "Apply:" or standalone)
        url_match = re.search(r"(https?://[^\s]+)", text)
        if url_match:
            url = url_match.group(1)
            # Clean up trailing punctuation
            url = re.sub(r"[,.\)]+$", "", url)
            return url
        return None


# Missing import
import asyncio
