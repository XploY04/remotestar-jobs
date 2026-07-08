from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str
    # Optional second target for the prod->dev refresh script. Set in
    # /root/jobs.ai-prod/.env on the rs server; unused elsewhere.
    database_url_dev: Optional[str] = None

    # API Keys
    rapidapi_key: Optional[str] = None
    adzuna_app_id: Optional[str] = None
    adzuna_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    redis_url: Optional[str] = None
    serpapi_key: Optional[str] = None

    # Email (Gmail SMTP)
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    candidate_app_url: str = "https://candidate.remotestar.io"

    # AI Enrichment
    enable_ai_enrichment: bool = True

    # ATS scraper: fetch per-job detail URLs when the list response omits
    # the description (Workable, SmartRecruiters mostly).
    enable_ats_detail_fetch: bool = True
    ats_detail_min_description: int = 200

    # Query planning
    enable_query_planner: bool = True
    query_planner_model: str = "gpt-4o-mini"
    query_planner_max_queries_per_source: int = 25
    query_planner_posted_within_days: int = 15
    query_planner_max_pages_per_query: int = 5

    # App settings
    environment: str = "development"
    log_level: str = "INFO"
    port: int = 8000  # Railway/Render inject PORT
    api_port: int = 8000  # Legacy alias
    ingestion_interval_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Leftover keys in a .env (e.g. GEMINI_API_KEY after the Gemini
        # removal) must not crash startup.
        extra = "ignore"


settings = Settings()
