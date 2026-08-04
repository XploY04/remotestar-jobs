from src.services.match_scorer import (
    country_score,
    match_percent,
    seniority_score,
    years_score,
)


def test_seniority_up_only_gate():
    # Same level and exactly one up are allowed (graded); below or 2+ up excluded.
    assert seniority_score("junior", "junior") == 1.0
    assert seniority_score("junior", "mid") == 0.7
    assert seniority_score("intern", "intern") == 1.0
    assert seniority_score("intern", "junior") == 0.7
    assert seniority_score("junior", "senior") is None      # 2 levels up -> excluded
    assert seniority_score("mid", "junior") is None         # below the candidate -> excluded
    # Unknown on either side can't be judged -> neutral pass, never excluded.
    assert seniority_score("junior", None) == 0.5
    assert seniority_score(None, "senior") == 0.5
    assert seniority_score("junior", "Mid-Level") == 0.7    # substring-tolerant


def test_country_eligibility_and_grade():
    assert country_score("in", "in", False) == 1.0          # home onsite
    assert country_score("in", "in", True) == 1.0           # home remote
    assert country_score("in", "us", True) == 0.85          # remote from abroad -> eligible
    assert country_score("in", "us", False) is None         # foreign onsite -> excluded
    assert country_score(None, "us", False) == 0.5          # unknown candidate country -> neutral
    assert country_score("in", None, False) == 0.5          # unknown job country -> neutral


def test_years_minimum_gate():
    assert years_score(3, 2) == 1.0                          # exceeds min
    assert years_score(2, 2) == 1.0                          # meets min
    assert years_score(1, 2) == 0.7                          # one year short -> still shown
    assert years_score(0, 3) is None                         # 2+ short -> excluded
    assert years_score(None, 3) == 0.5                       # unknown candidate years -> neutral
    assert years_score(2, None) == 0.5                       # job states no minimum -> neutral


def test_match_percent_blends_four_signals():
    # (1.0 + 1.0 + 1.0 + 0.52) / 4 = 0.88
    assert match_percent(1.0, 1.0, 1.0, 0.52) == 88
    # A one-up remote match ranks lower: (0.7 + 0.85 + 1.0 + 0.48) / 4 = 0.7575
    assert match_percent(0.7, 0.85, 1.0, 0.48) == 76
