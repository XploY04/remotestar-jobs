"""AI processor — the ONLY data transformation layer.

Takes raw API data from any source and outputs the final structured job schema.
No normalizer needed. The AI handles schema mapping for ALL sources.

Supports Gemini and OpenAI. Uses whichever API key is available in env
(GEMINI_API_KEY or OPENAI_API_KEY). Gemini is preferred if both are set.

OpenAI path uses structured outputs (json_schema) bound to a Pydantic schema —
the model cannot omit required fields, cannot wrap the response, cannot return
wrong types. Gemini still uses json_object mode (no structured-output equivalent).
"""

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BATCH_SIZE = 5

# Bump when SYSTEM_INSTRUCTION or extraction schema changes. Used by
# `--reenrich-stale` to identify jobs that need to be re-processed.
# v1: original 27-field schema, json_object mode.
# v2: slimmed schema (~20 fields), OpenAI structured outputs.
# v3: country tightened to ISO 3166 alpha-2 lowercase via Literal enum.
# v4: AI now produces a long-form description, not only a short summary.
# v5: added company_tier classification (drives board_rank ordering).
PROMPT_VERSION = "v5"


# ---------------------------------------------------------------------------
# Pydantic schema bound to OpenAI structured outputs (gpt-4o-mini json_schema)
# ---------------------------------------------------------------------------

WorkArrangement = Literal["remote", "hybrid", "onsite"]
EmploymentType = Literal["FULLTIME", "PARTTIME", "CONTRACT", "INTERN", "TEMPORARY"]
SeniorityLevel = Literal["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager"]
Category = Literal[
    "backend", "frontend", "fullstack", "mobile", "devops", "sre",
    "data", "ml", "security", "qa", "design", "product", "general",
]
VisaSponsorship = Literal["yes", "no", "unknown"]
CompanyTier = Literal["top_tech", "hot_startup", "established", "other"]
CountryCode = Literal[
    "ad", "ae", "af", "ag", "ai", "al", "am", "ao", "aq", "ar", "as", "at", "au", "aw", "ax", "az",
    "ba", "bb", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bl", "bm", "bn", "bo", "bq", "br", "bs",
    "bt", "bv", "bw", "by", "bz", "ca", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn",
    "co", "cr", "cu", "cv", "cw", "cx", "cy", "cz", "de", "dj", "dk", "dm", "do", "dz", "ec", "ee",
    "eg", "eh", "er", "es", "et", "fi", "fj", "fk", "fm", "fo", "fr", "ga", "gb", "gd", "ge", "gf",
    "gg", "gh", "gi", "gl", "gm", "gn", "gp", "gq", "gr", "gs", "gt", "gu", "gw", "gy", "hk", "hm",
    "hn", "hr", "ht", "hu", "id", "ie", "il", "im", "in", "io", "iq", "ir", "is", "it", "je", "jm",
    "jo", "jp", "ke", "kg", "kh", "ki", "km", "kn", "kp", "kr", "kw", "ky", "kz", "la", "lb", "lc",
    "li", "lk", "lr", "ls", "lt", "lu", "lv", "ly", "ma", "mc", "md", "me", "mf", "mg", "mh", "mk",
    "ml", "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv", "mw", "mx", "my", "mz", "na",
    "nc", "ne", "nf", "ng", "ni", "nl", "no", "np", "nr", "nu", "nz", "om", "pa", "pe", "pf", "pg",
    "ph", "pk", "pl", "pm", "pn", "pr", "ps", "pt", "pw", "py", "qa", "re", "ro", "rs", "ru", "rw",
    "sa", "sb", "sc", "sd", "se", "sg", "sh", "si", "sj", "sk", "sl", "sm", "sn", "so", "sr", "ss",
    "st", "sv", "sx", "sy", "sz", "tc", "td", "tf", "tg", "th", "tj", "tk", "tl", "tm", "tn", "to",
    "tr", "tt", "tv", "tw", "tz", "ua", "ug", "um", "us", "uy", "uz", "va", "vc", "ve", "vg", "vi",
    "vn", "vu", "wf", "ws", "ye", "yt", "za", "zm", "zw",
]


class JobExtraction(BaseModel):
    title: str
    company: str
    company_logo: Optional[str]
    company_website: Optional[str]
    description: str
    short_description: str
    country: Optional[CountryCode]
    city: Optional[str]
    is_remote: bool
    work_arrangement: WorkArrangement
    employment_type: EmploymentType
    seniority_level: Optional[SeniorityLevel]
    category: Category
    salary_min: Optional[float]
    salary_max: Optional[float]
    salary_currency: Optional[str]
    skills: List[str]
    required_experience_years: Optional[int]
    key_responsibilities: List[str]
    benefits: List[str]
    visa_sponsorship: VisaSponsorship
    application_deadline: Optional[str]
    company_tier: CompanyTier


class JobsBatch(BaseModel):
    jobs: List[JobExtraction]


# ---------------------------------------------------------------------------
# System instruction — extraction rules only; the schema is enforced by
# the structured-outputs response_format on OpenAI, and by the schema block
# we still include for Gemini's json_object mode.
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are a job data extraction engine. You extract structured fields from raw job listing data.

OUTPUT SCHEMA — every job object must match this structure:
{
  "title": "string — job title",
  "company": "string — company/employer name",
  "company_logo": "string or null — URL to company logo image",
  "company_website": "string or null — company website URL",
  "description": "string — detailed long-form job description synthesized from the raw listing",
  "short_description": "string — 2-3 sentence summary of the role",
  "country": "ISO 3166 alpha-2 lowercase code (e.g. us, gb, in, de), or null if unclear",
  "city": "city name or null",
  "is_remote": true or false,
  "work_arrangement": "remote|hybrid|onsite",
  "employment_type": "FULLTIME|PARTTIME|CONTRACT|INTERN|TEMPORARY",
  "seniority_level": "intern|junior|mid|senior|staff|principal|lead|manager or null",
  "category": "backend|frontend|fullstack|mobile|devops|sre|data|ml|security|qa|design|product|general",
  "salary_min": number or null,
  "salary_max": number or null,
  "salary_currency": "USD|EUR|GBP|etc or null",
  "skills": ["skill1", "skill2"],
  "required_experience_years": number or null,
  "key_responsibilities": ["resp1", "resp2"],
  "benefits": ["benefit1", "benefit2"],
  "visa_sponsorship": "yes|no|unknown",
  "application_deadline": "YYYY-MM-DD or null",
  "company_tier": "top_tech|hot_startup|established|other"
}

EXTRACTION RULES:
1. Extract EVERYTHING available from the raw data. Do NOT leave fields null if the data is present.
2. For "title": Use the actual job title. Clean up any formatting artifacts.
3. For "company": Extract company/employer name. Use "Unknown" only if truly absent.
4. For "company_logo": Look for logo/image URLs (e.g. employer_logo, company_logo, logo fields).
5. For "company_website": Look for employer/company website URLs.
6. For "description": Generate a long, candidate-facing description from the full raw listing. Preserve the real role details. Include responsibilities, requirements, tools, benefits, compensation, location, and application context when present. Do not invent facts, do not add generic filler, and do not shorten this into a summary.
7. For "short_description": Generate a concise 2-3 sentence summary from the long description.
8. For location fields: Parse location strings intelligently. "San Francisco, CA" → city="San Francisco", country="US". "Remote" → is_remote=true.
9. For "is_remote": true if remote work mentioned, OR source is "remoteok", OR job_is_remote=true.
10. For "work_arrangement": Determine from context. Default "onsite" if unclear, "remote" for remoteok source.
11. For "category": Classify based on ACTUAL role responsibilities. Sales/marketing/HR = "general". Only tech categories for actual tech roles.
12. For "salary_min"/"salary_max": NUMERIC values only (assume annual). If single salary mentioned, use for both.
13. For "skills": ALL technical skills, languages, frameworks, tools mentioned. Max 20.
14. For "required_experience_years": From "3+ years", "5-7 years" etc. Use the MINIMUM.
15. For "application_deadline": ONLY explicit deadlines. null if not mentioned.
16. For "benefits": Health insurance, 401k, PTO, equity, etc. Max 10.
17. For "visa_sponsorship": "yes" ONLY if explicitly mentioned. "unknown" if not discussed.
18. Max 20 skills, 8 responsibilities, 10 benefits.
19. For "company_tier": Classify the COMPANY (not the role). "top_tech" = globally famous large tech companies (Google, Meta, Apple, Amazon, Microsoft, Netflix, Nvidia and peers). "hot_startup" = well-known high-profile startups/scaleups (OpenAI, Anthropic, Notion, Stripe, Figma and similar). "established" = other recognizable mid/large companies (IT services firms, banks, older unicorns). "other" = everything else or unknown companies.
20. For SINGLE job requests: return a JSON object. For BATCH requests: return {"jobs": [...]} with the array in the same order as the input."""


class AIProcessor:
    """AI processor supporting Gemini and OpenAI.

    Uses whichever API key is available. Gemini is preferred if both are set.
    """

    def __init__(self):
        self.provider = None
        self._gemini_model = None
        self._openai_client = None

        # Per-source counters consumed by the enrichment pipeline for metrics.
        # Keyed by source because multiple sources process in parallel.
        self.batch_ok: Dict[str, int] = {}
        self.batch_fallback: Dict[str, int] = {}
        self.batch_failed: Dict[str, int] = {}

        if settings.gemini_api_key:
            self._init_gemini()
        elif settings.openai_api_key:
            self._init_openai()
        else:
            logger.warning("No AI API key set (GEMINI_API_KEY or OPENAI_API_KEY) - AI processing disabled")

        self.enabled = self.provider is not None

    def reset_metrics(self, source: str) -> None:
        self.batch_ok[source] = 0
        self.batch_fallback[source] = 0
        self.batch_failed[source] = 0

    def get_metrics(self, source: str) -> Dict[str, Any]:
        ok = self.batch_ok.get(source, 0)
        fb = self.batch_fallback.get(source, 0)
        failed = self.batch_failed.get(source, 0)
        total = ok + fb + failed
        fallback_rate = round(fb / total, 4) if total else 0.0
        return {
            "batch_ok": ok,
            "batch_fallback": fb,
            "batch_failed": failed,
            "fallback_rate": fallback_rate,
        }

    def _init_gemini(self):
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=settings.gemini_api_key)
        self._gemini_model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        self.provider = "gemini"
        logger.info("AI processor initialized (provider: Gemini, model: gemini-2.5-flash-lite)")

    def _init_openai(self):
        from openai import OpenAI

        self._openai_client = OpenAI(api_key=settings.openai_api_key)
        self.provider = "openai"
        logger.info("AI processor initialized (provider: OpenAI, model: gpt-4o-mini, structured outputs)")

    # ------------------------------------------------------------------
    # Single job processing (fallback for failed batch items)
    # ------------------------------------------------------------------

    def process_raw_job(self, source: str, raw_job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            raw_str = json.dumps(raw_job, default=str, ensure_ascii=False)
            prompt = f'Extract this job from source "{source}" into the schema.\n\n{raw_str}'

            if self.provider == "openai":
                result = self._call_openai_single(prompt)
            else:
                result = self._call_gemini(prompt)
                if isinstance(result, list):
                    result = result[0] if result else None
                elif isinstance(result, dict) and "jobs" in result:
                    jobs = result["jobs"]
                    result = jobs[0] if jobs else None

            if result:
                logger.info(f"AI processed: {result.get('title', '?')[:50]} @ {result.get('company', '?')}")
            return result

        except Exception as e:
            logger.error(f"AI single processing failed: {e}")
            return None

    # ------------------------------------------------------------------
    # BATCH processing (primary path — multiple jobs per API call)
    # ------------------------------------------------------------------

    def process_batch(self, source: str, raw_jobs: List[Dict[str, Any]],
                      batch_size: int = BATCH_SIZE) -> List[Optional[Dict[str, Any]]]:
        if not self.enabled:
            return [None] * len(raw_jobs)

        all_results: List[Optional[Dict[str, Any]]] = []

        for i in range(0, len(raw_jobs), batch_size):
            chunk = raw_jobs[i:i + batch_size]
            chunk_results = self._process_chunk(source, chunk)
            all_results.extend(chunk_results)

        return all_results

    def _process_chunk(self, source: str, chunk: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        n = len(chunk)

        try:
            jobs_block = []
            for idx, raw_job in enumerate(chunk):
                raw_str = json.dumps(raw_job, default=str, ensure_ascii=False)
                jobs_block.append(f"=== JOB {idx + 1} of {n} ===\n{raw_str}")

            joined = "\n\n".join(jobs_block)
            prompt = (
                f'Extract {n} jobs from source "{source}". Return JSON shaped as '
                f'{{"jobs": [...]}} where "jobs" is an array of exactly {n} objects '
                f"in the same order as the input.\n\n{joined}"
            )

            if self.provider == "openai":
                result = self._call_openai_batch(prompt)
            else:
                raw = self._call_gemini(prompt)
                result = self._unwrap_array(raw, n)

            if isinstance(result, list) and len(result) == n:
                logger.info(f"[{source}] Batch OK: {n} jobs in 1 API call")
                self.batch_ok[source] = self.batch_ok.get(source, 0) + 1
                return result
            elif isinstance(result, list):
                logger.warning(f"[{source}] Batch returned {len(result)} for {n} jobs — padding/trimming")
                self.batch_failed[source] = self.batch_failed.get(source, 0) + 1
                return (result + [None] * n)[:n]
            elif isinstance(result, dict) and n == 1:
                self.batch_ok[source] = self.batch_ok.get(source, 0) + 1
                return [result]
            else:
                logger.error(f"[{source}] Batch returned unexpected type: {type(result)}")
                return self._fallback_to_single(source, chunk)

        except Exception as e:
            logger.error(f"[{source}] Batch call failed: {e} — falling back to single")
            return self._fallback_to_single(source, chunk)

    def _fallback_to_single(self, source: str, chunk: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        self.batch_fallback[source] = self.batch_fallback.get(source, 0) + 1
        return [self.process_raw_job(source, raw_job) for raw_job in chunk]

    @staticmethod
    def _unwrap_array(result: Any, n: int) -> Any:
        """Gemini json_object mode can wrap arrays in an object. Unwrap to the inner list."""
        if not isinstance(result, dict):
            return result
        for key in ("jobs", "results", "items", "data", "output", "extracted"):
            value = result.get(key)
            if isinstance(value, list):
                return value
        list_values = [v for v in result.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        return result

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> Any:
        response = self._gemini_model.generate_content(prompt)
        return json.loads(response.text)

    def _call_openai_batch(self, prompt: str) -> List[Dict[str, Any]]:
        """Structured-outputs batch call. Returns list of dicts, never raises on shape."""
        response = self._openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            response_format=JobsBatch,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        return [job.model_dump() for job in parsed.jobs]

    def _call_openai_single(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Structured-outputs single call. Returns dict or None."""
        response = self._openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            response_format=JobExtraction,
        )
        parsed = response.choices[0].message.parsed
        return parsed.model_dump() if parsed else None
