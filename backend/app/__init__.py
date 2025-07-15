"""
Application factory for the Flask backend.
"""
import logging
from flask import Flask
from flask_cors import CORS

from app.config import get_config
from app.api import register_blueprints
from app.core.exceptions import register_error_handlers


def create_app(config_name='development'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Initialize logging
    setup_logging(app)
    
    # Initialize CORS
    setup_cors(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize LlamaIndex components
    with app.app_context():
        from app.config.llama_config import initialize_llama_components
        initialize_llama_components(app)
    
    return app


def setup_logging(app):
    """Configure application logging."""
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure app logger
    app.logger.setLevel(getattr(logging, log_level))
    
    # Reduce noise from libraries
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def setup_cors(app):
    """Configure CORS settings."""
    CORS(
        app,
        origins=app.config.get('CORS_ORIGINS', ['*']),
        supports_credentials=True,
        allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
        expose_headers=['Content-Type', 'X-Content-Type-Options', 'X-Frame-Options']
    )