from src.services.matcher import _collect_user_titles, _merge_ai_result
from src.services.match_scorer import seniority_fit_score


def test_merge_ai_result_drops_zero_score_so_ui_falls_back_to_structured():
    # AI returned 0 (e.g. empty profile). ai_score must NOT be persisted, so the
    # frontend's `ai_score ?? score` shows the real structured score, not 0%.
    match = {"score": 60}
    _merge_ai_result(match, {"ai_score": 0, "reason": "n/a", "skills_gap": [], "strengths": []})
    assert "ai_score" not in match
    assert match["ai_reason"] == "n/a"


def test_merge_ai_result_keeps_positive_score():
    match = {"score": 60}
    _merge_ai_result(match, {"ai_score": 85, "reason": "great fit", "skills_gap": ["Rust"], "strengths": ["Go"]})
    assert match["ai_score"] == 85
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


def test_seniority_fit_supports_legacy_and_canonical_values():
    assert seniority_fit_score("Mid-Level", "mid") == 1.0
    assert seniority_fit_score("Junior", "junior") == 1.0
    assert seniority_fit_score("senior", "Senior") == 1.0
