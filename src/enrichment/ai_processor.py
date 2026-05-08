"""AI processor — the ONLY data transformation layer.

Takes raw API data from any source and outputs the final structured job schema.
No normalizer needed. The AI handles schema mapping for ALL sources.

Supports Gemini and OpenAI. Uses whichever API key is available in env
(GEMINI_API_KEY or OPENAI_API_KEY). Gemini is preferred if both are set.

Optimizations:
  - Batch processing: 5 jobs per API call (80% fewer calls)
  - JSON mode: guarantees valid JSON output
  - temperature=0: deterministic extraction
  - System instruction sent once, not repeated every call
"""

import json
from typing import Dict, Any, Optional, List
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BATCH_SIZE = 5

SYSTEM_INSTRUCTION = """You are a job data extraction engine. You extract structured fields from raw job listing data.

OUTPUT SCHEMA — every job object must match this structure:
{
  "title": "string — job title",
  "company": "string — company/employer name",
  "company_logo": "string or null — URL to company logo image",
  "company_website": "string or null — company website URL",
  "short_description": "string — 2-3 sentence summary of the role",
  "country": "ISO country code or full name, or null",
  "city": "city name or null",
  "state": "state/region or null",
  "is_remote": true or false,
  "work_arrangement": "remote|hybrid|onsite",
  "employment_type": "FULLTIME|PARTTIME|CONTRACT|INTERN|TEMPORARY",
  "seniority_level": "intern|junior|mid|senior|staff|principal|lead|manager",
  "department": "department name or null",
  "category": "backend|frontend|fullstack|mobile|devops|sre|data|ml|security|qa|design|product|general",
  "salary_min": number or null,
  "salary_max": number or null,
  "salary_currency": "USD|EUR|GBP|etc or null",
  "salary_period": "year|month|week|hour or null",
  "skills": ["skill1", "skill2"],
  "required_experience_years": number or null,
  "required_education": "Bachelor's|Master's|PhD|etc or null",
  "key_responsibilities": ["resp1", "resp2"],
  "nice_to_have_skills": ["skill1", "skill2"],
  "benefits": ["benefit1", "benefit2"],
  "visa_sponsorship": "yes|no|unknown",
  "application_deadline": "YYYY-MM-DD or null",
  "tags": ["tag1", "tag2"]
}

EXTRACTION RULES:
1. Extract EVERYTHING available from the raw data. Do NOT leave fields null if the data is present.
2. For "title": Use the actual job title. Clean up any formatting artifacts.
3. For "company": Extract company/employer name. Use "Unknown" only if truly absent.
4. For "company_logo": Look for logo/image URLs (e.g. employer_logo, company_logo, logo fields).
5. For "company_website": Look for employer/company website URLs.
6. For "short_description": Generate a concise 2-3 sentence summary from the full description.
7. For location fields: Parse location strings intelligently. "San Francisco, CA" → city="San Francisco", state="CA", country="US". "Remote" → is_remote=true.
8. For "is_remote": true if remote work mentioned, OR source is "remoteok", OR job_is_remote=true.
9. For "work_arrangement": Determine from context. Default "onsite" if unclear, "remote" for remoteok source.
10. For "category": Classify based on ACTUAL role responsibilities. Sales/marketing/HR = "general". Only tech categories for actual tech roles.
11. For "salary_min"/"salary_max": NUMERIC values only. If single salary mentioned, use for both.
12. For "salary_period": Is salary per year, month, week, or hour? Clues: "/yr", "annual", "per hour", range size (>$30k likely annual, <$100 likely hourly).
13. For "skills": ALL technical skills, languages, frameworks, tools mentioned. Max 20.
14. For "required_experience_years": From "3+ years", "5-7 years" etc. Use the MINIMUM.
15. For "application_deadline": ONLY explicit deadlines. null if not mentioned.
16. For "tags": Categories, tags, labels from the source data.
17. For "benefits": Health insurance, 401k, PTO, equity, etc.
18. For "visa_sponsorship": "yes" ONLY if explicitly mentioned. "unknown" if not discussed.
19. Max 20 skills, 8 responsibilities, 10 benefits.
20. For SINGLE job requests: return a JSON object. For BATCH requests: return a JSON array."""


class AIProcessor:
    """AI processor supporting Gemini and OpenAI.

    Uses whichever API key is available. Gemini is preferred if both are set.
    """

    def __init__(self):
        self.provider = None
        self._gemini_model = None
        self._openai_client = None

        if settings.gemini_api_key:
            self._init_gemini()
        elif settings.openai_api_key:
            self._init_openai()
        else:
            logger.warning("No AI API key set (GEMINI_API_KEY or OPENAI_API_KEY) - AI processing disabled")

        self.enabled = self.provider is not None

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
        logger.info("AI processor initialized (provider: OpenAI, model: gpt-4o-mini)")

    # ------------------------------------------------------------------
    # Single job processing (fallback for failed batch items)
    # ------------------------------------------------------------------

    def process_raw_job(self, source: str, raw_job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            raw_str = json.dumps(raw_job, default=str, ensure_ascii=False)
            prompt = f'Extract this job from source "{source}" into the schema. Return a single JSON object.\n\n{raw_str}'

            result = self._call_ai(prompt)
            if isinstance(result, list):
                result = result[0] if result else None
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

            result = self._call_ai(prompt)
            result = self._unwrap_array(result, n)

            if isinstance(result, list) and len(result) == n:
                logger.info(f"[{source}] Batch OK: {n} jobs in 1 API call")
                return result
            elif isinstance(result, list):
                logger.warning(f"[{source}] Batch returned {len(result)} for {n} jobs — padding/trimming")
                padded = (result + [None] * n)[:n]
                return padded
            elif isinstance(result, dict) and n == 1:
                return [result]
            else:
                logger.error(f"[{source}] Batch returned unexpected type: {type(result)}")
                return self._fallback_to_single(source, chunk)

        except Exception as e:
            logger.error(f"[{source}] Batch call failed: {e} — falling back to single")
            return self._fallback_to_single(source, chunk)

    def _fallback_to_single(self, source: str, chunk: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        return [self.process_raw_job(source, raw_job) for raw_job in chunk]

    @staticmethod
    def _unwrap_array(result: Any, n: int) -> Any:
        """OpenAI's json_object mode forces an object wrapper, so the array often
        comes back as {"jobs": [...]} or similar. Unwrap to the inner list."""
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
    # Provider dispatch
    # ------------------------------------------------------------------

    def _call_ai(self, prompt: str) -> Any:
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)

    def _call_gemini(self, prompt: str) -> Any:
        response = self._gemini_model.generate_content(prompt)
        return json.loads(response.text)

    def _call_openai(self, prompt: str) -> Any:
        response = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
