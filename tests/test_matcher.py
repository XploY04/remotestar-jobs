import pytest

from src.services import matcher
from src.services.matcher import (
    _build_candidate_summary,
    _collect_user_titles,
    _merge_ai_result,
)


@pytest.mark.asyncio
async def test_matching_for_all_does_not_require_or_deduct_credits(monkeypatch):
    users = [
        {"_id": "user-with-zero", "credits": 0},
        {"_id": "user-without-credit-field"},
    ]
    computed = []
    saved = []

    async def get_users():
        return users

    async def compute(user, run_ai=False, progress_fn=None):
        computed.append((user["_id"], run_ai))
        return [{"job_id": f"job-for-{user['_id']}"}]

    async def save(user_id, matches):
        saved.append((user_id, matches))

    monkeypatch.setattr(matcher, "_get_matching_users", get_users)
    monkeypatch.setattr(matcher, "_compute_matches", compute)
    monkeypatch.setattr(matcher, "_save_matches", save)

    result = await matcher.run_matching_for_all()

    assert computed == [
        ("user-with-zero", True),
        ("user-without-credit-field", True),
    ]
    assert [user_id for user_id, _ in saved] == [
        "user-with-zero",
        "user-without-credit-field",
    ]
    assert result["users_processed"] == 2
    assert result["users_skipped"] == 0
    assert result["total_matches"] == 2


@pytest.mark.asyncio
async def test_compute_matches_fails_when_providers_are_unavailable(monkeypatch):
    monkeypatch.setattr(matcher.settings, "openai_api_key", None)
    monkeypatch.setattr(matcher.settings, "pinecone_api_key", "configured")

    with pytest.raises(matcher.MatchingUnavailableError):
        await matcher._compute_matches({"_id": "user-1"})


@pytest.mark.asyncio
async def test_matching_for_all_does_not_replace_matches_when_unavailable(monkeypatch):
    saved = []

    async def get_users():
        return [{"_id": "user-1"}]

    async def unavailable(user, run_ai=False, progress_fn=None):
        raise matcher.MatchingUnavailableError("unavailable")

    async def save(user_id, matches):
        saved.append((user_id, matches))

    monkeypatch.setattr(matcher, "_get_matching_users", get_users)
    monkeypatch.setattr(matcher, "_compute_matches", unavailable)
    monkeypatch.setattr(matcher, "_save_matches", save)

    with pytest.raises(matcher.MatchingUnavailableError):
        await matcher.run_matching_for_all()

    assert saved == []


def test_candidate_summary_uses_resume_when_users_doc_is_empty():
    # The real-world case: users doc is blank, all data lives in the resume.
    user = {"role_focus": "Senior Software Engineer", "skills": [],
            "seniority_level": "", "years_of_experience": 0, "location": ""}
    resume_doc = {"editable_profile": {
        "skills": [{"name": "Go"}, {"name": "Python"}, {"name": "Kubernetes"}],
        "experiences": [{"position": "Senior Software Engineer", "technologies": ["AWS", "Terraform"]}],
        "summary": "Backend engineer with 7 years building distributed systems.",
    }}
    summary = _build_candidate_summary(user, resume_doc)
    assert "Go, Python, Kubernetes" in summary       # resume skills, not blank
    assert "AWS" in summary and "Terraform" in summary
    assert "7 years" in summary
    assert "Skills: ." not in summary                # the old empty-profile bug


def test_candidate_summary_falls_back_to_users_doc_skills():
    user = {"role_focus": "Data Engineer", "skills": ["Spark", "SQL"]}
    summary = _build_candidate_summary(user, None)
    assert "Spark, SQL" in summary


def test_merge_ai_result_attaches_explanation_only():
    # The LLM is explanation-only now: it attaches reason/strengths/skills_gap
    # and never a score. The blended structured score stands untouched.
    match = {"score": 60}
    _merge_ai_result(match, {"reason": "great fit", "skills_gap": ["Rust"], "strengths": ["Go"]})
    assert "ai_score" not in match
    assert match["score"] == 60
    assert match["ai_reason"] == "great fit"
    assert match["skills_gap"] == ["Rust"]
    assert match["strengths"] == ["Go"]


def test_merge_ai_result_handles_missing_result():
    match = {"score": 60}
    _merge_ai_result(match, None)
    assert match == {"score": 60}


def test_collect_user_titles_includes_resume_positions_and_skips_interns():
    user = {"role_focus": "Backend Engineer"}
    resume_doc = {
        "editable_profile": {
            "experiences": [
                {"position": "Software Engineer Intern"},
                {"position": "Junior Backend Developer"},
                {"position": "Backend Engineer"},
                {"position": "Platform Engineer"},
            ]
        }
    }

    assert _collect_user_titles(user, resume_doc) == [
        "Backend Engineer",
        "Junior Backend Developer",
        "Platform Engineer",
    ]
