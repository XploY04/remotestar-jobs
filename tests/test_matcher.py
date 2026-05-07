from src.services.matcher import _collect_user_titles
from src.services.match_scorer import seniority_fit_score


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
