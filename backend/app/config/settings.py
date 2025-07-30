"""
Configuration settings for different environments.
"""
import os
from datetime import timedelta


class Config:
    """Base configuration."""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get('AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME')
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME')
    AZURE_OPENAI_API_VERSION = '2024-02-01'
    
    # Azure Search
    AZURE_SEARCH_ENDPOINT = os.environ.get('AZURE_SEARCH_SERVICE_ENDPOINT')
    AZURE_SEARCH_KEY = os.environ.get('AZURE_SEARCH_ADMIN_KEY')
    AZURE_SEARCH_INDEX_NAME = os.environ.get('AZURE_SEARCH_INDEX_NAME', 'azure-multimodal-search-new')
    
    # Azure Storage - Updated to use access key instead of SAS token
    AZURE_STORAGE_ACCOUNT_NAME = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
    AZURE_STORAGE_ACCESS_KEY = os.environ.get('AZURE_STORAGE_ACCESS_KEY')  # Changed from SAS_TOKEN
    AZURE_BLOB_CONTAINER_NAME = os.environ.get('BLOB_CONTAINER_NAME', 'rag-demo-images')
    
    # SAS Token Settings
    SAS_TOKEN_EXPIRY_MINUTES = int(os.environ.get('SAS_TOKEN_EXPIRY_MINUTES', '30'))
    
    # CORS
    CORS_ORIGINS = ['http://localhost:5173', 'http://localhost:3000']
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    # Request limits
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    REQUEST_TIMEOUT = 300  # 5 minutes
    IMAGE_PROXY_TIMEOUT = 10  # 10 seconds for image fetching
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = '100/hour'
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CORS_ORIGINS = ['*']  # Allow all origins in development


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Stricter CORS in production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }
    
    # Session security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    LOG_LEVEL = 'ERROR'
    
    # Use test database/index
    AZURE_SEARCH_INDEX_NAME = 'test-index'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration object based on environment."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(config_name, DevelopmentConfig)