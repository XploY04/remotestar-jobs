"""Job-user matching orchestrator using Pinecone vector search.

Pipeline:
  1. Embed jobs at ingestion time → store in Pinecone (namespace: jobs-pool)
  2. At match time: embed user profile → query Pinecone for top similar jobs
  3. Apply structured scoring on Pinecone results (seniority, location, experience)
  4. AI refinement via OpenAI on the top matches
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

from src.database.operations import db
from src.services.match_scorer import (
    compute_total,
    experience_fit_score,
    is_stretch_match,
    location_match_score,
    salary_fit_score,
    seniority_fit_score,
    title_similarity_score,
)
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MATCH_CREDIT_COST = 50
PINECONE_INDEX = settings.pinecone_index or "remotestar"
PINECONE_NAMESPACE = settings.pinecone_namespace or "jobs-pool"
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_N_VECTOR = 100       # Top results from Pinecone
TOP_N_STRUCTURED = 50    # Keep after structured scoring
TOP_N_AI = 20            # Send to AI refinement

# Skills match is computed via vector similarity (semantic match between user profile and job)
MATCH_WEIGHTS = {
    "skills_match": 0.40,
    "title_similarity": 0.20,
    "seniority_fit": 0.15,
    "location_match": 0.15,
    "experience_fit": 0.05,
    "salary_fit": 0.05,
}

_openai_client: Optional[OpenAI] = None
_pinecone_index = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def _embed_text(text: str) -> List[float]:
    """Generate embedding using OpenAI text-embedding-3-large."""
    response = _get_openai().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_job_embed_text(job: dict) -> str:
    """Build the text to embed for a job."""
    parts = []
    if job.get("title"):
        parts.append(f"Title: {job['title']}")
    if job.get("skills"):
        parts.append(f"Skills: {', '.join(job['skills'][:20])}")
    if job.get("seniority_level"):
        parts.append(f"Seniority: {job['seniority_level']}")
    if job.get("short_description"):
        parts.append(f"Description: {job['short_description'][:500]}")
    elif job.get("description"):
        parts.append(f"Description: {job['description'][:500]}")
    if job.get("category"):
        parts.append(f"Category: {job['category']}")
    return "\n".join(parts)


def _build_user_embed_text(user: dict, resume_doc: Optional[dict] = None) -> str:
    """Build the text to embed for a user profile."""
    parts = []
    if user.get("role_focus"):
        parts.append(f"Role: {user['role_focus']}")
    if user.get("skills"):
        parts.append(f"Skills: {', '.join(user['skills'][:20])}")
    if user.get("seniority_level"):
        parts.append(f"Seniority: {user['seniority_level']}")
    if user.get("years_of_experience") is not None:
        parts.append(f"Experience: {user['years_of_experience']} years")

    if resume_doc:
        profile = resume_doc.get("editable_profile") or {}
        # Add technologies from experiences
        techs = set()
        for exp in (profile.get("experiences") or []):
            for tech in (exp.get("technologies") or []):
                techs.add(tech)
        if techs:
            existing = set(user.get("skills") or [])
            new_techs = techs - existing
            if new_techs:
                parts.append(f"Additional technologies: {', '.join(list(new_techs)[:10])}")
        # Add summary
        if profile.get("summary"):
            parts.append(f"Summary: {profile['summary'][:300]}")

    return "\n".join(parts)


# ------------------------------------------------------------------
# Job embedding (run at ingestion time)
# ------------------------------------------------------------------

async def embed_jobs(job_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Embed jobs and upsert to Pinecone. Only embeds jobs not already embedded."""

    if not settings.openai_api_key or not settings.pinecone_api_key:
        logger.warning("OpenAI or Pinecone API key not set, skipping embedding")
        return {"embedded": 0}

    index = _get_pinecone_index()

    if job_ids:
        query = {"_id": {"$in": job_ids}, **db.active_job_filter()}
    else:
        query = {"pinecone_embedded_at": {"$exists": False}, **db.active_job_filter()}

    jobs = await db.jobs.find(query, {"raw_data": 0}).to_list(length=None)
    logger.info("Embedding %d new jobs for Pinecone", len(jobs))

    if not jobs:
        return {"embedded": 0}

    batch_size = 50
    total_embedded = 0

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        vectors = []

        for job in batch:
            text = build_job_embed_text(job)
            if not text.strip():
                continue

            try:
                embedding = _embed_text(text)
                metadata = {
                    "title": job.get("title") or "",
                    "company": job.get("company") or "",
                    "category": job.get("category") or "",
                    "seniority_level": job.get("seniority_level") or "",
                    "country": job.get("country") or "",
                    "is_remote": bool(job.get("is_remote")),
                    "source": job.get("source") or "",
                }
                vectors.append({
                    "id": job["_id"],
                    "values": embedding,
                    "metadata": metadata,
                })
            except Exception as e:
                logger.error("Failed to embed job %s: %s", job["_id"], e)

        if vectors:
            index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
            embedded_ids = [v["id"] for v in vectors]
            await db.jobs.update_many(
                {"_id": {"$in": embedded_ids}},
                {"$set": {"pinecone_embedded_at": datetime.now(timezone.utc)}},
            )
            total_embedded += len(vectors)
            logger.info("Upserted batch %d-%d (%d vectors)", i, i + len(batch), len(vectors))

    logger.info("Embedding complete: %d new jobs embedded", total_embedded)
    return {"embedded": total_embedded}


# ------------------------------------------------------------------
# User matching
# ------------------------------------------------------------------

async def run_matching_for_all() -> Dict[str, Any]:
    """Weekly batch: match jobs for all users with job_matching_enabled=true."""

    users = await _get_matching_users()
    logger.info("Found %d users with job_matching_enabled=true", len(users))

    if not users:
        return {"users_processed": 0, "total_matches": 0}

    total_matches = 0
    users_processed = 0
    users_skipped = 0

    for user in users:
        credits = user.get("credits", 0)
        if credits < MATCH_CREDIT_COST:
            logger.info("Skipping user %s: insufficient credits (%d), pausing matching", user["_id"], credits)
            await _pause_matching(user)
            users_skipped += 1
            continue

        deducted = await _deduct_credits(user["_id"], user.get("email", ""))
        if not deducted:
            users_skipped += 1
            continue

        matches = await _compute_matches(user, run_ai=True)
        await _save_matches(user["_id"], matches)

        total_matches += len(matches)
        users_processed += 1
        logger.info("User %s: %d matches (credits deducted)", user["_id"], len(matches))

    summary = {
        "users_processed": users_processed,
        "users_skipped": users_skipped,
        "total_matches": total_matches,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from src.services.popularity import compute_job_popularity
        summary["popularity"] = await compute_job_popularity(db)
    except Exception as e:
        logger.error("Popularity recompute failed: %s", e, exc_info=True)
        summary["popularity"] = {"error": str(e)}

    logger.info("Matching complete: %s", summary)
    return summary


async def run_matching_for_user(user_id: str, run_ai: bool = True, redis_client=None) -> Dict[str, Any]:
    """Single user matching. If redis_client is provided, publishes progress events."""

    def publish(stage: str, message: str, progress: int = 0, **extra):
        if redis_client:
            event = json.dumps({"stage": stage, "message": message, "progress": progress, **extra})
            redis_client.publish(f"match:progress:{user_id}", event)

    user = await db.db["users"].find_one({"_id": user_id})
    if not user:
        logger.error("User %s not found", user_id)
        publish("error", "User not found")
        return {"error": "user not found"}

    publish("embedding", "Analyzing your profile...", 0)
    matches = await _compute_matches(user, run_ai=run_ai, progress_fn=publish)

    publish("saving", f"Saving {len(matches)} matches...", 95)
    await _save_matches(user_id, matches)

    publish("done", f"Found {len(matches)} matches!", 100, matches=len(matches))
    logger.info("User %s: %d matches computed", user_id, len(matches))
    return {"matches": len(matches), "user_id": user_id}


async def _compute_matches(user: dict, run_ai: bool = False, progress_fn=None) -> List[dict]:
    """Vector search + structured scoring + optional AI refinement."""

    def _progress(stage, message, progress=0, **extra):
        if progress_fn:
            progress_fn(stage, message, progress, **extra)

    if not settings.openai_api_key or not settings.pinecone_api_key:
        logger.warning("OpenAI or Pinecone not configured, cannot match")
        return []

    # Build user embedding
    resume_doc = await db.db["resumes"].find_one({"user_id": user["_id"]})
    user_text = _build_user_embed_text(user, resume_doc)
    if not user_text.strip():
        logger.warning("User %s has empty profile, cannot match", user["_id"])
        return []

    _progress("embedding", "Embedding your profile...", 5)
    user_embedding = _embed_text(user_text)
    user_titles = _collect_user_titles(user, resume_doc)
    user_level = user.get("seniority_level")
    user_location = user.get("location")
    user_years = user.get("years_of_experience")

    user_education = None
    if resume_doc:
        profile = resume_doc.get("editable_profile") or {}
        education = profile.get("education") or []
        if education:
            user_education = education[0].get("degree")

    # Query Pinecone for top similar jobs
    _progress("searching", "Searching jobs in Pinecone...", 15)
    index = _get_pinecone_index()
    results = index.query(
        vector=user_embedding,
        top_k=TOP_N_VECTOR,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )

    if not results.matches:
        logger.info("User %s: no Pinecone matches found", user["_id"])
        return []

    logger.info("User %s: %d vector matches from Pinecone", user["_id"], len(results.matches))

    # Fetch full job docs from MongoDB for the matched IDs
    job_ids = [m.id for m in results.matches]
    job_docs = await db.jobs.find(
        {"_id": {"$in": job_ids}, **db.active_job_filter()},
        {"raw_data": 0, "description": 0},
    ).to_list(length=None)
    job_lookup = {j["_id"]: j for j in job_docs}

    # Build score map from Pinecone similarity
    similarity_map = {m.id: m.score for m in results.matches}

    _progress("scoring", f"Scoring {len(job_docs)} candidate jobs...", 35)

    # Stage 2: Structured scoring on Pinecone results
    scored = []
    for job_id, vector_score in similarity_map.items():
        job = job_lookup.get(job_id)
        if not job:
            continue

        signals = {
            "skills_match": vector_score,
            "title_similarity": title_similarity_score(user_titles, job.get("title", "")),
            "seniority_fit": seniority_fit_score(user_level, job.get("seniority_level")),
            "location_match": location_match_score(user_location, job.get("country"), job.get("is_remote")),
            "experience_fit": experience_fit_score(user_years, job.get("required_experience_years")),
            "salary_fit": salary_fit_score(None, job.get("salary_min"), job.get("salary_max")),
        }

        score = _compute_weighted_score(signals)
        if score < 25:
            continue

        stretch = is_stretch_match(
            user_years, job.get("required_experience_years"),
            user_education, job.get("required_education"),
        )
        if stretch:
            score = min(score, 40)

        scored.append({
            "_id": f"{user['_id']}_{job['_id']}",
            "user_id": user["_id"],
            "job_id": job["_id"],
            "score": score,
            "is_stretch": stretch,
            "signals": signals,
            "computed_at": datetime.now(timezone.utc),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_matches = scored[:TOP_N_STRUCTURED]

    # Stage 3: AI refinement on top 20
    _progress("ai_refining", f"AI analyzing top {min(len(top_matches), TOP_N_AI)} matches...", 60)
    if run_ai and top_matches:
        ai_results = await _ai_refine(user, job_lookup, top_matches[:TOP_N_AI])
        for match in top_matches:
            _merge_ai_result(match, ai_results.get(match["job_id"]))

        top_matches.sort(key=lambda x: x.get("ai_score") or x["score"], reverse=True)

    return top_matches


def _compute_weighted_score(signals: Dict[str, float]) -> int:
    """Combine signals into 0-100 score using match weights."""
    total = sum(signals.get(k, 0.0) * v for k, v in MATCH_WEIGHTS.items())
    return max(0, min(100, round(total * 100)))


def _merge_ai_result(match: dict, ai: Optional[dict]) -> None:
    """Apply an AI refinement result onto a match.

    Only a positive ai_score is persisted. A 0 or missing score is left unset
    so every consumer (frontend `ai_score ?? score`, the analyze-cache
    freshness check) falls back to the structured score instead of rendering a
    misleading 0% match.
    """
    if not ai:
        return
    score = ai.get("ai_score")
    if isinstance(score, (int, float)) and score > 0:
        match["ai_score"] = int(score)
    match["ai_reason"] = ai.get("reason")
    match["skills_gap"] = ai.get("skills_gap", [])
    match["strengths"] = ai.get("strengths", [])


def _collect_user_titles(user: dict, resume_doc: Optional[dict] = None) -> List[str]:
    titles = []
    if user.get("role_focus"):
        titles.append(user["role_focus"])

    if resume_doc:
        profile = resume_doc.get("editable_profile") or {}
        for exp in (profile.get("experiences") or []):
            position = (exp.get("position") or "").strip()
            if position and not _is_intern_position(position):
                titles.append(position)

    deduped = []
    seen = set()
    for title in titles:
        key = title.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(title)
    return deduped


def _is_intern_position(position: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", position.lower())
    return any(token in tokens for token in ("intern", "internship", "trainee", "apprentice"))


async def _ai_refine(
    user: dict, job_lookup: Dict[str, dict], top_matches: List[dict]
) -> Dict[str, dict]:
    """Stage 3: Send top matches to OpenAI for scoring + reasoning."""

    user_summary = (
        f"Role: {user.get('role_focus', 'Unknown')}. "
        f"Skills: {', '.join((user.get('skills') or [])[:15])}. "
        f"Experience: {user.get('years_of_experience', '?')} years. "
        f"Seniority: {user.get('seniority_level', 'Unknown')}. "
        f"Location: {user.get('location', 'Unknown')}."
    )

    jobs_block = []
    for match in top_matches:
        job = job_lookup.get(match["job_id"])
        if not job:
            continue
        jobs_block.append({
            "job_id": match["job_id"],
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "skills": (job.get("skills") or [])[:10],
            "seniority": job.get("seniority_level", ""),
            "country": job.get("country", ""),
            "is_remote": job.get("is_remote", False),
            "required_years": job.get("required_experience_years"),
        })

    if not jobs_block:
        return {}

    prompt = (
        f"Candidate profile: {user_summary}\n\n"
        f"Jobs to evaluate:\n{json.dumps(jobs_block, default=str)}\n\n"
        "For each job, return a JSON array with objects containing:\n"
        '- "job_id": the job_id\n'
        '- "ai_score": 0-100 match score\n'
        '- "reason": one sentence explaining the match\n'
        '- "skills_gap": list of skills the candidate is missing\n'
        '- "strengths": list of candidate skills that are relevant\n'
        "Return ONLY the JSON array."
    )

    try:
        client = _get_openai()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a job matching assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        if isinstance(parsed, dict) and "matches" in parsed:
            return {item["job_id"]: item for item in parsed["matches"] if "job_id" in item}
        if isinstance(parsed, list):
            return {item["job_id"]: item for item in parsed if "job_id" in item}
    except Exception as e:
        logger.error("AI refinement failed: %s", e)

    return {}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _get_matching_users() -> List[dict]:
    cursor = db.db["users"].find(
        {"job_matching_enabled": True},
        {"skills": 1, "role_focus": 1, "seniority_level": 1, "location": 1,
         "years_of_experience": 1, "email": 1, "credits": 1, "first_name": 1},
    )
    return await cursor.to_list(length=None)


async def _pause_matching(user: dict) -> None:
    """Auto-disable matching and send notification email."""
    user_id = user["_id"]
    email = user.get("email", "")
    first_name = user.get("first_name", "there")
    credits = user.get("credits", 0)

    # Disable toggle and set paused reason
    await db.db["users"].update_one(
        {"_id": user_id},
        {"$set": {
            "job_matching_enabled": False,
            "job_matching_paused_reason": "insufficient_credits",
        }},
    )

    # Get previous match count for the email
    match_count = await db.db["job_matches"].count_documents({"user_id": user_id})

    # Send email
    from src.services.email_service import send_matching_paused_email
    send_matching_paused_email(email, first_name, credits, match_count)

    logger.info("Paused matching for user %s, email sent to %s", user_id, email)


async def _deduct_credits(user_id: str, email: str) -> bool:
    user = await db.db["users"].find_one({"_id": user_id})
    if not user or user.get("credits", 0) < MATCH_CREDIT_COST:
        return False

    new_balance = user["credits"] - MATCH_CREDIT_COST
    result = await db.db["users"].update_one(
        {"_id": user_id, "credits": {"$gte": MATCH_CREDIT_COST}},
        {"$inc": {"credits": -MATCH_CREDIT_COST}},
    )
    if result.modified_count == 0:
        return False

    await db.db["ledger"].insert_one({
        "user_id": user_id,
        "email": email,
        "spent_for": "job_matching",
        "credit_spent": -MATCH_CREDIT_COST,
        "running_balance": new_balance,
        "created_at": datetime.now(timezone.utc),
    })
    return True


async def _save_matches(user_id: str, matches: List[dict]) -> None:
    await db.db["job_matches"].delete_many({"user_id": user_id})
    if matches:
        await db.db["job_matches"].insert_many(matches)
