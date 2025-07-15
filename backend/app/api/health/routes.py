"""
Health check API routes.
"""
import logging
from flask import Blueprint, jsonify, current_app
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__)


@health_bp.route('', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'knowledge-assistant-api',
        'version': current_app.config.get('APP_VERSION', '1.0.0')
    })


@health_bp.route('/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with dependency status."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Check Azure OpenAI
    try:
        llm = current_app.extensions.get('llm')
        if llm:
            health_status['checks']['azure_openai'] = {
                'status': 'healthy',
                'deployment': current_app.config['AZURE_OPENAI_CHAT_DEPLOYMENT']
            }
        else:
            raise Exception('LLM not initialized')
    except Exception as e:
        health_status['checks']['azure_openai'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        health_status['status'] = 'degraded'
    
    # Check Azure Search
    try:
        search_client = current_app.extensions.get('search_client')
        if search_client:
            health_status['checks']['azure_search'] = {
                'status': 'healthy',
                'index': current_app.config['AZURE_SEARCH_INDEX_NAME']
            }
        else:
            raise Exception('Search client not initialized')
    except Exception as e:
        health_status['checks']['azure_search'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        health_status['status'] = 'degraded'
    
    # Check configuration
    required_config = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_KEY'
    ]
    
    missing_config = [key for key in required_config if not current_app.config.get(key)]
    
    if missing_config:
        health_status['checks']['configuration'] = {
            'status': 'unhealthy',
            'missing': missing_config
        }
        health_status['status'] = 'unhealthy'
    else:
        health_status['checks']['configuration'] = {
            'status': 'healthy'
        }
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        # Check if all components are initialized
        required_components = ['llm', 'multimodal_llm', 'index', 'search_client']
        
        for component in required_components:
            if not current_app.extensions.get(component):
                return jsonify({
                    'ready': False,
                    'reason': f'{component} not initialized'
                }), 503
        
        return jsonify({'ready': True}), 200
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 503


@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return jsonify({'alive': True}), 200