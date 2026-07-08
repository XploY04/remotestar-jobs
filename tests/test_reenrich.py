import asyncio
from typing import Any, Dict, List

from src.services import reenrich


class _FakeJobs:
    def __init__(self):
        self.updates: List[Dict[str, Any]] = []

    async def update_one(self, filt, update):
        self.updates.append((filt, update))


class _FakeDB:
    def __init__(self):
        self.jobs = _FakeJobs()


class _FakePipeline:
    """Returns results with the middle job dropped, as process_source does."""

    async def process_source(self, source_name, raw_jobs, **kwargs):
        results = []
        for raw in raw_jobs:
            if raw["n"] == 2:
                continue  # dropped job
            results.append({
                "id": f"{source_name}_{raw['n']}",
                "title_company_hash": f"hash{raw['n']}",
                "title": f"Job {raw['n']}",
            })
        return results


def test_reenrich_batch_pairs_by_identity_when_jobs_drop():
    db = _FakeDB()
    docs = [
        {"_id": "src_1", "title_company_hash": "hash1", "source": "src", "raw_data": {"n": 1}},
        {"_id": "src_2", "title_company_hash": "hash2", "source": "src", "raw_data": {"n": 2}},
        {"_id": "src_3", "title_company_hash": "hash3", "source": "src", "raw_data": {"n": 3}},
    ]
    per_source: Dict[str, int] = {}

    asyncio.run(reenrich._reenrich_batch(db, _FakePipeline(), docs, per_source))

    updated_ids = [filt["_id"] for filt, _ in db.jobs.updates]
    assert updated_ids == ["src_1", "src_3"]
    for filt, update in db.jobs.updates:
        assert update["$set"]["title"] == f"Job {filt['_id'].split('_')[1]}"
        assert update["$unset"] == {"pinecone_embedded_at": ""}
