"""
Custom exceptions and error handlers.
"""
import logging
from flask import jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class APIException(Exception):
    """Base API exception."""
    status_code = 500
    message = "Internal server error"
    
    def __init__(self, message=None, status_code=None, payload=None):
        super().__init__()
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        """Convert exception to dictionary."""
        rv = dict(self.payload or ())
        rv['error'] = self.message
        return rv


class ValidationError(APIException):
    """Validation error exception."""
    status_code = 400
    message = "Validation error"


class NotFoundError(APIException):
    """Resource not found exception."""
    status_code = 404
    message = "Resource not found"


class AuthenticationError(APIException):
    """Authentication error exception."""
    status_code = 401
    message = "Authentication required"


class AuthorizationError(APIException):
    """Authorization error exception."""
    status_code = 403
    message = "Access forbidden"


class RateLimitError(APIException):
    """Rate limit exceeded exception."""
    status_code = 429
    message = "Rate limit exceeded"


class ExternalServiceError(APIException):
    """External service error exception."""
    status_code = 502
    message = "External service error"


def register_error_handlers(app):
    """Register error handlers with the Flask app."""
    
    @app.errorhandler(APIException)
    def handle_api_exception(error):
        """Handle custom API exceptions."""
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions."""
        response = jsonify({
            'error': error.description,
            'status_code': error.code
        })
        response.status_code = error.code
        return response
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        return jsonify({
            'error': 'Endpoint not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors."""
        return jsonify({
            'error': 'Method not allowed',
            'status_code': 405
        }), 405
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {str(error)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status_code': 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'status_code': 500
        }), 500