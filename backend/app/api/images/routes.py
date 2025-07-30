"""
Image proxy API routes with dynamic SAS token generation.
"""
import logging
import requests
from urllib.parse import unquote, urlparse
from flask import Blueprint, request, jsonify, Response, current_app

from app.utils.blob_storage import get_blob_service

logger = logging.getLogger(__name__)

# Create blueprint
images_bp = Blueprint('images', __name__)


@images_bp.route('/proxy', methods=['GET'])
def proxy_image():
    """Proxy images from private blob storage with dynamically generated SAS tokens."""
    try:
        # Get the blob path from query parameter
        encoded_path = request.args.get('path')
        if not encoded_path:
            return jsonify({'error': 'Missing image path'}), 400
        
        # Decode the URL-encoded path
        blob_path = unquote(encoded_path)
        logger.debug(f"Proxy received path: {blob_path}")
        
        # Get blob storage service
        blob_service = get_blob_service()
        
        # Extract blob name
        if blob_path.startswith('http://') or blob_path.startswith('https://'):
            # It's a full URL, extract blob name
            blob_name = blob_service.extract_blob_name_from_url(blob_path)
        else:
            # It's already just a blob name
            blob_name = blob_path
        
        if not blob_name:
            logger.error(f"Could not extract blob name from: {blob_path}")
            return jsonify({'error': 'Invalid blob path'}), 400
        
        # Generate SAS token URL
        try:
            sas_url = blob_service.generate_sas_token(
                blob_name, 
                expiry_minutes=current_app.config.get('SAS_TOKEN_EXPIRY_MINUTES', 30)
            )
        except Exception as e:
            logger.error(f"Failed to generate SAS token: {str(e)}")
            return jsonify({'error': 'Failed to generate access token'}), 500
        
        logger.debug(f"Generated SAS URL for blob: {blob_name}")
        
        # Fetch the image with timeout
        try:
            response = requests.get(
                sas_url, 
                timeout=current_app.config.get('IMAGE_PROXY_TIMEOUT', 10),
                stream=True
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching image: {e}")
            return jsonify({
                'error': 'Image fetch failed',
                'status_code': e.response.status_code if e.response else None
            }), 404
        except requests.exceptions.Timeout:
            logger.error("Image fetch timeout")
            return jsonify({'error': 'Image fetch timeout'}), 504
        
        # Determine content type
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Stream the image back
        return Response(
            response.iter_content(chunk_size=8192),
            content_type=content_type,
            headers={
                'Cache-Control': 'public, max-age=1800',  # Cache for 30 minutes
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Image proxy error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@images_bp.route('/generate-sas-url', methods=['POST'])
def generate_sas_url():
    """Generate a SAS URL for a specific blob."""
    try:
        data = request.get_json()
        if not data or 'blob_name' not in data:
            return jsonify({'error': 'Missing blob_name in request'}), 400
        
        blob_name = data['blob_name']
        expiry_minutes = data.get('expiry_minutes', current_app.config.get('SAS_TOKEN_EXPIRY_MINUTES', 30))
        
        # Get blob storage service
        blob_service = get_blob_service()
        
        # Generate SAS URL
        sas_url = blob_service.generate_sas_token(blob_name, expiry_minutes)
        
        return jsonify({
            'sasurl': sas_url,
            'blob_name': blob_name,
            'expires_in_minutes': expiry_minutes
        })
        
    except Exception as e:
        logger.error(f"SAS generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate SAS URL'}), 500


@images_bp.route('/validate-sas-url', methods=['POST'])
def validate_sas_url():
    """Check if a SAS URL is still valid."""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing url in request'}), 400
        
        url = data['url']
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return jsonify({'error': 'Invalid URL format'}), 400
        except Exception:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Check if URL has SAS token
        if '?' not in url:
            return jsonify({
                'is_valid': False,
                'reason': 'No SAS token found in URL'
            })
        
        # Extract and validate SAS token
        _, sas_token = url.split('?', 1)
        blob_service = get_blob_service()
        is_valid = blob_service.is_sas_token_valid(sas_token)
        
        return jsonify({
            'is_valid': is_valid,
            'url': url
        })
        
    except Exception as e:
        logger.error(f"SAS validation error: {str(e)}")
        return jsonify({'error': 'Failed to validate SAS URL'}), 500


@images_bp.route('/regenerate-sas-token', methods=['POST'])
def regenerate_sas_token():
    """Regenerate SAS token if expired, or return existing if still valid."""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing url in request'}), 400
        
        url = data['url']
        expiry_minutes = data.get('expiry_minutes', current_app.config.get('SAS_TOKEN_EXPIRY_MINUTES', 30))
        
        # Get blob storage service
        blob_service = get_blob_service()
        
        # Regenerate if needed
        new_url = blob_service.regenerate_sas_token_if_needed(url, expiry_minutes)
        
        return jsonify({
            'sasurl': new_url,
            'regenerated': new_url != url
        })
        
    except Exception as e:
        logger.error(f"SAS regeneration error: {str(e)}")
        return jsonify({'error': 'Failed to regenerate SAS token'}), 500