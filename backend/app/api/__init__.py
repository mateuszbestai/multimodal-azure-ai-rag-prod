"""API package - Blueprint registration."""
from flask import Flask


def register_blueprints(app: Flask):
    """Register all API blueprints."""
    from .chat import chat_bp
    from .health import health_bp
    from .images import images_bp
    
    # Register blueprints with URL prefixes
    app.register_blueprint(health_bp, url_prefix='/api/health')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(images_bp, url_prefix='/api/image')
    
    # Register static file serving for production
    if app.config.get('SERVE_STATIC_FILES', False):
        from .static import static_bp
        app.register_blueprint(static_bp)