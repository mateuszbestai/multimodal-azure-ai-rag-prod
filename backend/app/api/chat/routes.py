"""
Chat API routes.
"""
import json
import logging
from flask import Blueprint, request, jsonify, Response, stream_with_context
from werkzeug.exceptions import BadRequest

from app.models.schemas import ChatRequest, ChatResponse
from .services import ChatService

logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Initialize service
chat_service = ChatService()


@chat_bp.route('', methods=['POST'])
def handle_chat():
    """Process chat messages and return formatted response."""
    try:
        # Validate request
        data = request.get_json()
        if not data:
            raise BadRequest('Invalid JSON payload')
        
        chat_request = ChatRequest(**data)
        
        # Process request
        response = chat_service.process_chat(chat_request.message)
        
        return jsonify(ChatResponse(**response).dict())
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process your request.'
        }), 500


@chat_bp.route('/stream', methods=['POST'])
def handle_chat_stream():
    """Process chat messages and return streaming response."""
    try:
        # Validate request
        data = request.get_json()
        if not data:
            raise BadRequest('Invalid JSON payload')
        
        chat_request = ChatRequest(**data)
        
        def generate():
            """Generate streaming response."""
            try:
                for chunk in chat_service.stream_chat(chat_request.message):
                    yield json.dumps(chunk) + '\n'
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield json.dumps({
                    'type': 'error',
                    'message': str(e)
                }) + '\n'
        
        return Response(
            stream_with_context(generate()),
            mimetype='application/x-ndjson',
            headers={
                'X-Content-Type-Options': 'nosniff',
                'Transfer-Encoding': 'chunked'
            }
        )
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Stream processing error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process your request.'
        }), 500


@chat_bp.route('/debug/metadata', methods=['POST'])
def debug_metadata():
    """Debug endpoint to inspect retrieval metadata."""
    if not current_app.debug:
        return jsonify({'error': 'Debug endpoint disabled'}), 403
    
    try:
        data = request.get_json()
        query = data.get('message', 'test query')
        
        metadata = chat_service.get_debug_metadata(query)
        
        return jsonify(metadata)
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return jsonify({'error': str(e)}), 500