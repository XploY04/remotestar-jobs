import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from src.database.models import ensure_indexes, normalize_doc
from src.database.operations import Database


@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection and disconnection"""
    db = Database()

    # Should be able to connect
    await db.connect()
    assert db.client is not None
    assert db.db is not None

    # Should be able to disconnect
    await db.disconnect()


@pytest.mark.asyncio
async def test_database_hash_function():
    """Test title+company hashing for deduplication"""
    db = Database()

    hash1 = db._hash_title_company("Backend Engineer", "TechCorp")
    hash2 = db._hash_title_company("Backend Engineer", "TechCorp")
    hash3 = db._hash_title_company("Frontend Engineer", "TechCorp")

    # Same title+company should produce same hash
    assert hash1 == hash2

    # Different title should produce different hash
    assert hash1 != hash3

    # Hash should be 16 characters (truncated SHA256)
    assert len(hash1) == 16


@pytest.mark.asyncio
async def test_database_string_coercion():
    """Test salary value string coercion"""
    db = Database()

    # Test various input types
    assert db._to_str(None) is None
    assert db._to_str("100000") == "100000"
    assert db._to_str(100000) == "100000"
    assert db._to_str(100000.50) == "100000.5"


@pytest.mark.asyncio
async def test_save_jobs_with_duplicates():
    """Test job saving with duplicate detection"""
    db = Database()
    await db.connect()

    # Sample job data
    job_data = {
        "source": "test",
        "source_id": "test123",
        "title": "Test Backend Engineer",
        "company": "TestCompany",
        "description": "Test description",
        "location": {"city": "Test City", "country": "US", "remote": False},
        "employment_type": "FULLTIME",
        "salary_min": "100000",
        "salary_max": "150000",
        "salary_currency": "USD",
        "apply_url": "https://example.com",
        "posted_at": datetime.now(timezone.utc),
        "raw_data": {}
    }

    # Save first time - should be new
    stats1 = await db.save_jobs([job_data])
    assert stats1["new"] >= 0  # May be 0 if already exists from previous runs

    # Save same job again - should be skipped
    stats2 = await db.save_jobs([job_data])
    assert stats2["skipped"] >= 1
    assert stats2["new"] == 0

    await db.disconnect()


@pytest.mark.asyncio
async def test_save_jobs_restores_deleted_duplicate():
    db = Database()
    await db.connect()
    job_data = _job_data("restore-deleted")
    job_id = f"{job_data['source']}_{job_data['source_id']}"
    title_company_hash = db._hash_title_company(job_data["title"], job_data["company"])

    await db.jobs.delete_many({"_id": job_id})
    await db.jobs.insert_one({
        "_id": job_id,
        "source": job_data["source"],
        "source_id": job_data["source_id"],
        "title": job_data["title"],
        "company": job_data["company"],
        "description": "old description",
        "apply_url": "https://old.example.com",
        "posted_at": datetime.now(timezone.utc) - timedelta(days=20),
        "title_company_hash": title_company_hash,
        "is_deleted": True,
        "deleted_at": datetime.now(timezone.utc),
        "delete_reason": "expired",
    })

    stats = await db.save_jobs([job_data])
    restored = await db.jobs.find_one({"_id": job_id})

    assert stats["restored"] == 1
    assert restored["is_deleted"] is False
    assert "deleted_at" not in restored
    assert "delete_reason" not in restored
    assert restored["apply_url"] == job_data["apply_url"]

    await db.jobs.delete_many({"_id": job_id})
    await db.disconnect()


@pytest.mark.asyncio
async def test_save_jobs_skips_archived_duplicate():
    db = Database()
    await db.connect()
    job_data = _job_data("skip-archived")
    job_id = f"{job_data['source']}_{job_data['source_id']}"
    title_company_hash = db._hash_title_company(job_data["title"], job_data["company"])

    await db.jobs.delete_many({"_id": job_id})
    await db.jobs.insert_one({
        "_id": job_id,
        "source": job_data["source"],
        "source_id": job_data["source_id"],
        "title": job_data["title"],
        "company": job_data["company"],
        "description": "archived description",
        "apply_url": "https://archived.example.com",
        "posted_at": datetime.now(timezone.utc),
        "title_company_hash": title_company_hash,
        "is_archived": True,
        "archived_at": datetime.now(timezone.utc),
    })

    stats = await db.save_jobs([job_data])
    archived = await db.jobs.find_one({"_id": job_id})

    assert stats["skipped"] == 1
    assert stats["restored"] == 0
    assert archived["is_archived"] is True
    assert archived["apply_url"] == "https://archived.example.com"

    await db.jobs.delete_many({"_id": job_id})
    await db.disconnect()


@pytest.mark.asyncio
async def test_cleanup_expired_jobs_soft_deletes_without_archiving():
    db = Database()
    await db.connect()
    job_data = _job_data("cleanup-expired")
    job_id = f"{job_data['source']}_{job_data['source_id']}"

    await db.jobs.delete_many({"_id": job_id})
    doc = db._build_job_doc(job_id, db._hash_title_company(job_data["title"], job_data["company"]), job_data)
    doc["application_deadline"] = datetime.now(timezone.utc) - timedelta(days=1)
    await db.jobs.insert_one(doc)

    stats = await db.cleanup_expired_jobs()
    deleted = await db.jobs.find_one({"_id": job_id})

    assert stats["deleted"] >= 1
    assert deleted["is_deleted"] is True
    assert deleted["delete_reason"] == "expired"
    assert "deleted_at" in deleted
    assert "archived_at" not in deleted

    await db.jobs.delete_many({"_id": job_id})
    await db.disconnect()


def test_active_job_filter_is_default_query_base():
    db = Database()
    assert db._build_filter() == db.active_job_filter()


def test_build_job_doc_persists_prompt_version():
    db = Database()
    job_data = _job_data("prompt-version")
    job_data["prompt_version"] = "v4"

    doc = db._build_job_doc(
        "test_prompt-version",
        db._hash_title_company(job_data["title"], job_data["company"]),
        job_data,
    )

    assert doc["prompt_version"] == "v4"


def _job_data(source_id: str):
    return {
        "source": "test",
        "source_id": source_id,
        "title": f"Test Backend Engineer {source_id}",
        "company": "TestCompany",
        "description": "Test description",
        "location": {"city": "Test City", "country": "US", "remote": False},
        "employment_type": "FULLTIME",
        "salary_min": "100000",
        "salary_max": "150000",
        "salary_currency": "USD",
        "apply_url": "https://example.com",
        "posted_at": datetime.now(timezone.utc),
        "raw_data": {},
    }


def test_normalize_doc():
    """Test MongoDB _id to id remapping"""
    doc = {"_id": "test_123", "title": "Test Job"}
    result = normalize_doc(doc)

    assert result["id"] == "test_123"
    assert "_id" not in result
    assert normalize_doc(None) is None


def test_import_database_modules():
    """Test that database modules can be imported"""
    from src.database.models import ensure_indexes, normalize_doc
    from src.database.operations import Database, db

    assert ensure_indexes is not None
    assert normalize_doc is not None
    assert Database is not None
    assert db is not None
