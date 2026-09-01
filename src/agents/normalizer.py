"""Centralized job normalization agent."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dateutil import parser
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LocationSchema(BaseModel):
    """Standardized location schema"""
    city: Optional[str] = None
    country: Optional[str] = None
    remote: bool = False


class NormalizedJob(BaseModel):
    """Standardized job schema with validation"""
    source: str
    source_id: str
    title: str
    company: str
    description: str
    location: LocationSchema
    employment_type: str = "FULLTIME"
    salary_min: Optional[str] = None
    salary_max: Optional[str] = None
    salary_currency: Optional[str] = None
    apply_url: str
    posted_at: datetime
    application_deadline: Optional[datetime] = None  # Last date to apply
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('salary_min', 'salary_max', mode='before')
    @classmethod
    def validate_salary(cls, v):
        """Convert salary to string if needed"""
        if v is None:
            return None
        return str(v)

    @field_validator('employment_type')
    @classmethod
    def validate_employment_type(cls, v):
        """Ensure employment_type is uppercase and valid"""
        valid_types = ['FULLTIME', 'PARTTIME', 'CONTRACT', 'INTERN', 'TEMPORARY']
        v_upper = v.upper() if v else 'FULLTIME'
        return v_upper if v_upper in valid_types else 'FULLTIME'

    @field_validator('title', 'description')
    @classmethod
    def validate_required_strings(cls, v, info):
        """Ensure required strings are not empty"""
        if not v or not str(v).strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return str(v).strip()
    
    @field_validator('apply_url')
    @classmethod
    def validate_apply_url(cls, v):
        """Ensure apply_url has a value, use placeholder if empty"""
        if not v or not str(v).strip():
            return "https://unknown"
        return str(v).strip()
    
    @field_validator('company')
    @classmethod
    def validate_company(cls, v):
        """Ensure company is not empty, use default if None"""
        if not v or not str(v).strip():
            return "Unknown Company"
        return str(v).strip()


class NormalizerAgent:
    """
    Centralized normalization agent that converts jobs from any source
    to a standardized schema.
    
    Benefits:
    - Single source of truth for job schema
    - Consistent validation across all sources
    - Easy to update schema (one place)
    - Better error handling and logging
    """

    # Source-specific field mappings
    FIELD_MAPPINGS = {
        'jsearch': {
            'source_id': 'job_id',
            'title': 'job_title',
            'company': 'employer_name',
            'description': 'job_description',
            'employment_type': 'job_employment_type',
            'salary_min': 'job_min_salary',
            'salary_max': 'job_max_salary',
            'salary_currency': 'job_salary_currency',
            'apply_url': 'job_apply_link',
            'posted_at': 'job_posted_at_datetime_utc',
            'application_deadline': 'job_offer_expiration_datetime_utc',
            'location': {
                'city': 'job_city',
                'country': 'job_country',
                'remote': 'job_is_remote',
            }
        },
        'remoteok': {
            'source_id': 'id',
            'title': 'position',
            'company': 'company',
            'description': 'description',
            'employment_type': 'type',
            'salary_min': 'salary_min',
            'salary_max': 'salary_max',
            'salary_currency': 'salary_currency',
            'apply_url': 'url',
            'posted_at': 'epoch',  # Unix timestamp
            'location': {
                'city': None,
                'country': 'location',
                'remote': True,  # Always remote (RemoteOK)
            }
        },
        'hackernews': {
            'source_id': 'id',
            'title': 'title',
            'company': 'company',
            'description': 'description',
            'apply_url': 'apply_url',
            'posted_at': 'posted_at',
            'location': {
                'city': None,
                'country': 'location_raw',
                'remote': 'remote',
            }
        },
        'rss_feed': {
            'source_id': 'source_id',
            'title': 'title',
            'company': 'company',
            'description': 'description',
            'apply_url': 'apply_url',
            'posted_at': 'posted_at',
            'location': {
                'city': None,
                'country': 'location_raw',
                'remote': 'remote',
            }
        },
        # ATS scraper already outputs normalized format — passthrough mapping
        'ats_scraper': {
            'source_id': 'source_id',
            'title': 'title',
            'company': 'company',
            'description': 'description',
            'employment_type': 'employment_type',
            'salary_min': 'salary_min',
            'salary_max': 'salary_max',
            'salary_currency': 'salary_currency',
            'apply_url': 'apply_url',
            'posted_at': 'posted_at',
            'location': {
                'city': '_location_city',
                'country': '_location_country',
                'remote': '_location_remote',
            }
        },
    }

    def normalize(self, source: str, raw_job: Dict[str, Any], 
                  context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Normalize a job from any source to standard schema.
        
        Args:
            source: Source name (jsearch, remoteok, hackernews, rss)
            raw_job: Raw job data from source
            context: Additional source context
            
        Returns:
            Normalized job dict or None if validation fails
        """
        try:
            # Get field mapping for this source
            mapping = self.FIELD_MAPPINGS.get(source)
            if not mapping:
                logger.error(f"No field mapping for source: {source}")
                return None

            # Build normalized job using mapping
            normalized = {
                'source': source,
                'raw_data': raw_job,
            }

            # Map simple fields
            for std_field, source_field in mapping.items():
                if std_field == 'location' or isinstance(source_field, dict):
                    continue
                    
                value = self._extract_value(raw_job, source_field, context)
                
                # Special handling for dates
                if std_field in ('posted_at', 'application_deadline'):
                    value = self._parse_date(value, source) if value else None
                
                # Special handling for employment type
                if std_field == 'employment_type':
                    value = (value or 'FULLTIME').upper()
                
                # Special handling for salary currency (default based on source)
                if std_field == 'salary_currency' and not value:
                    value = "USD"
                
                normalized[std_field] = value

            # Map location (nested structure)
            location_mapping = mapping.get('location', {})
            normalized['location'] = {
                'city': self._extract_value(raw_job, location_mapping.get('city'), context),
                'country': self._extract_value(raw_job, location_mapping.get('country'), context),
                'remote': self._extract_value(raw_job, location_mapping.get('remote'), context) or False,
            }

            # Validate with Pydantic
            validated = NormalizedJob(**normalized)
            return validated.model_dump()

        except Exception as exc:
            logger.error(f"[{source}] Normalization failed: {exc}", exc_info=True)
            return None

    def _extract_value(self, data: Dict, path: Optional[str], 
                      context: Optional[Dict] = None) -> Any:
        """
        Extract value from nested dict using dot notation path.
        
        Examples:
            path='company.display_name' -> data['company']['display_name']
            path='_country' -> context['country'] (from context)
        """
        if path is None:
            return None
        
        # Handle boolean literals (e.g., True for always-remote sources)
        if isinstance(path, bool):
            return path
            
        # Check context first (fields starting with _)
        if isinstance(path, str) and path.startswith('_') and context:
            return context.get(path[1:])
        
        # Handle nested paths (e.g., 'company.display_name')
        if '.' in path:
            keys = path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            return value
        
        # Simple key lookup
        return data.get(path)

    def _parse_date(self, value: Any, source: str) -> datetime:
        """Parse date from various formats"""
        if not value:
            return datetime.now(timezone.utc)
        
        try:
            # Unix timestamp (RemoteOK)
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
            
            # ISO string (JSearch and ATS feeds)
            if isinstance(value, str):
                return parser.isoparse(value)
            
            # Datetime object
            if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
                
        except Exception as exc:
            logger.warning(f"[{source}] Date parsing failed for '{value}': {exc}")
        
        return datetime.now(timezone.utc)


# Global instance
normalizer = NormalizerAgent()
