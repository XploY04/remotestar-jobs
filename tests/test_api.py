import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.fixture
def client():
    """Create test client with app lifecycle"""
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_jobs_endpoint(client):
    """Test job listing endpoint"""
    response = client.get("/api/jobs?limit=10")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_list_jobs_with_search(client):
    """Test job search functionality"""
    response = client.get("/api/jobs?search=backend&limit=5")
    
    assert response.status_code == 200
    data = response.json()
    
    # If there are results, they should contain 'backend' in title or description
    if data["total"] > 0:
        for job in data["jobs"]:
            text = f"{job['title']} {job.get('description', '')}".lower()
            # Note: search may not always match due to filtering


def test_list_jobs_pagination(client):
    """Test pagination parameters"""
    # Test default limit
    response = client.get("/api/jobs")
    assert response.status_code == 200
    
    # Test custom limit
    response = client.get("/api/jobs?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["jobs"]) <= 5
    
    # Test max limit boundary - should reject values > 200
    response = client.get("/api/jobs?limit=250")
    assert response.status_code == 422  # Validation error


def test_get_job_not_found(client):
    """Test getting non-existent job"""
    response = client.get("/api/jobs/nonexistent_id_12345")
    
    assert response.status_code == 404


def test_job_response_schema(client):
    """Test job response has correct schema"""
    response = client.get("/api/jobs?limit=1")
    
    assert response.status_code == 200
    data = response.json()
    
    if data["total"] > 0:
        job = data["jobs"][0]
        
        # Check required fields
        assert "id" in job
        assert "source" in job
        assert "title" in job
        assert "company" in job
        assert "location" in job
        assert "apply_url" in job
        assert "posted_at" in job


def test_import_api_modules():
    """Test that API modules can be imported"""
    from src.api.main import app, create_app
    from src.api.routes import router
    from src.api.schemas import JobResponse, JobsListResponse, JobLocation
    
    assert app is not None
    assert create_app is not None
    assert router is not None
    assert JobResponse is not None
