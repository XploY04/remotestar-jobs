from src.services.match_scorer import (
    WEIGHTS,
    compute_idf,
    compute_total,
    seniority_gate,
    skills_match_score,
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    assert "salary_fit" not in WEIGHTS
    assert WEIGHTS["seniority_fit"] > WEIGHTS["skills_match"]


def test_seniority_gate_blocks_two_or_more_levels():
    assert seniority_gate("junior", "senior") is False
    assert seniority_gate("intern", "mid") is False
    assert seniority_gate("junior", "mid") is True
    assert seniority_gate("junior", "intern") is True
    assert seniority_gate("junior", None) is True
    assert seniority_gate(None, "senior") is True
    assert seniority_gate("junior", "unrecognized level") is True


def test_skills_match_uses_idf_overlap():
    jobs = [
        {"skills": ["Python", "SQL"]},
        {"skills": ["Python", "Rust"]},
        {"skills": ["Python"]},
    ]
    idf = compute_idf(jobs)
    # Rare skill (Rust) counts more than ubiquitous one (Python)
    rust_match = skills_match_score(["Rust"], ["Python", "Rust"], idf)
    python_match = skills_match_score(["Python"], ["Python", "Rust"], idf)
    assert rust_match > python_match
    assert skills_match_score([], ["Python"], idf) == 0.0


def test_compute_total_gives_seniority_more_weight_than_skills():
    seniority_only = compute_total({"seniority_fit": 1.0})
    skills_only = compute_total({"skills_match": 1.0})
    assert seniority_only == 30
    assert skills_only == 25
