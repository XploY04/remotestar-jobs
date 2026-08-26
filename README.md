# jobs.ai

Job ingestion, enrichment, search, embedding, and candidate matching for the
Rumero jobs product.

The service uses Python 3.12, FastAPI, MongoDB, OpenAI, Pinecone, and Redis.
It shares its MongoDB database with the candidate API. The candidate frontend
does not call this service directly; it reads jobs and matches through the Go
candidate API.

## What it does

- Fetches jobs from RemoteOK, JSearch, Adzuna, Hacker News, RSS feeds, and
  public ATS APIs.
- Uses OpenAI structured output to map source payloads into one job schema.
- Saves each completed enrichment batch immediately and skips known jobs before
  making AI calls.
- Soft-deletes expired jobs and removes their stale Pinecone vectors.
- Embeds job titles and skills in Pinecone.
- Matches candidates using vector relevance plus seniority, country, and
  experience gates.
- Writes match explanations, strengths, and skill gaps for the top results.
- Serves a small FastAPI surface for health checks, job search, filters, and
  manual ingestion.

## Architecture

```text
External job sources
  -> source fetchers
  -> OpenAI enrichment or rule-based fallback
  -> MongoDB jobs collection
  -> Pinecone embedding

Candidate API match request
  -> Redis match:start channel
  -> jobs worker
  -> Pinecone candidate search
  -> structured scoring and OpenAI explanation
  -> MongoDB job_matches collection
  -> candidate API and frontend
```

The MongoDB collections this repo owns directly are `jobs`, `job_matches`,
`discovered_companies`, `query_metrics`, and `ingest_metrics`. Matching also
reads `users` and `resumes`, which are written by the candidate API.

## Local setup

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

At minimum, set `DATABASE_URL`. Ingestion needs the keys for whichever sources
you want to run. AI enrichment needs `OPENAI_API_KEY`. Matching needs OpenAI,
Pinecone, and Redis.

Start the API on port 8000:

```bash
python -m src.main
```

This starts only the API. It does not schedule or automatically run ingestion.
Use a command mode or call the ingestion endpoint when you want work to run.

## Command modes

```bash
python -m src.main --ingest-once
python -m src.main --embed-jobs
python -m src.main --match
python -m src.main --match-user USER_ID
python -m src.main --worker
python -m src.main --cleanup
python -m src.main --reenrich-stale
python -m src.main --reenrich-low-info
python -m src.main --reenrich-stale --reenrich-limit 100
```

`--worker` is the long-running Redis consumer used for on-demand matching.
`--match` refreshes every user who has `job_matching_enabled=true`. Matching is
free and does not read or write credit fields.

The source schedule in `src/services/schedule.yaml` controls which fetchers an
ingestion pass selects on a given day. Pass the fetcher classes explicitly in
code when you need to run every source regardless of the schedule.

## API

The FastAPI app exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Process and MongoDB health |
| `GET` | `/api/jobs` | Paginated job search and filters |
| `GET` | `/api/jobs/{job_id}` | One active job |
| `GET` | `/api/filters` | Current source, category, and seniority facets |
| `POST` | `/api/jobs/ingest` | Run one ingestion pass |

Swagger UI is available at `http://localhost:8000/docs` while the API is
running.

Example search:

```bash
curl 'http://localhost:8000/api/jobs?limit=20&search=kubernetes&remote_only=true&category=devops'
```

## Ingestion details

The pipeline batches five raw jobs per OpenAI request and runs up to ten batches
concurrently. A source failure does not stop the other sources. Jobs older than
15 days are rejected during normal ingestion.

Deduplication happens before enrichment using the source identifier. MongoDB
also enforces unique source identifiers and a sparse title-company hash. An
active or archived duplicate is skipped. A soft-deleted duplicate is restored
without another enrichment call.

The current extraction prompt version is recorded on AI-enriched jobs. Use the
re-enrichment modes after changing the extraction schema or when existing jobs
have missing skills or seniority.

## Matching details

Job vectors contain title and skills. Candidate vectors contain role and skills,
including technologies found in parsed resume experience.

Pinecone returns the top 100 relevant jobs. The matcher then applies these
rules:

- Cosine relevance must be at least 0.25.
- Seniority must be the candidate's level or one level higher.
- An onsite job must be in the candidate's country. Remote jobs may be in
  another country.
- The candidate may meet the requested experience or be one year short.

The three gate scores and cosine relevance are averaged into the final match
percentage. OpenAI adds explanations to the top 20 results but does not change
their scores or ordering. A user keeps at most 50 current matches, and MongoDB
expires match documents after 14 days.

## Deployment

This repository runs on the `plane.remotestar.io` droplet rather than in the
candidate Kubernetes namespace.

- `/root/jobs.ai` is the dev checkout. It runs `jobs-ai-worker.service` from
  `main` as a long-running Redis worker.
- `/root/jobs.ai-prod` is the production checkout. Root cron runs ingestion and
  embedding daily, then matching weekly.

Dev and production use separate MongoDB databases and separate Pinecone indexes.
The database names can look similar, so check the connection target before any
write or maintenance command.

The production checkout also runs `scripts/refresh_dev_data.py` weekly. It
replaces dev public job collections with recent production data while leaving
dev users, resumes, interviews, applications, matches, and payment state alone.

## Tests

```bash
pytest
pytest tests/test_matcher.py
```

Tests use `asyncio_mode=auto`. Several database tests connect to the MongoDB URI
in `.env`, so they need an isolated test database when run outside CI.

## Adding a source

1. Add a `BaseFetcher` implementation under `src/agents/`.
2. Return raw source dictionaries without mapping them to the final schema.
3. Register the fetcher in `FETCHER_MAP` in `src/services/orchestrator.py`.
4. Add its schedule to `src/services/schedule.yaml`.
5. Add fetch, failure, deduplication, and query-plan tests.

The enrichment pipeline owns the final schema mapping.
