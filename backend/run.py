"""
Development server entry point.
"""
import os
from app import create_app

# Create app with environment-specific config
config_name = os.environ.get('FLASK_ENV', 'development')
app = create_app(config_name)

if __name__ == '__main__':
    # Get port from environment or default to 5001
    port = int(os.environ.get('PORT', 8000))
    
    # Run the development server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config.get('DEBUG', False)
    )