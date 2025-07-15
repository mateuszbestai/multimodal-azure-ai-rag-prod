"""
Image proxy API routes.
"""
import logging
import requests
from urllib.parse import unquote, urlparse
from flask import Blueprint, request, jsonify, Response, current_app

from app.utils.image_utils import extract_blob_name_from_url, build_image_url

logger = logging.getLogger(__name__)

# Create blueprint
images_bp = Blueprint('images', __name__)


@images_bp.route('/proxy', methods=['GET'])
def proxy_image():
    """Proxy images from private blob storage for frontend display."""
    try:
        # Get the blob path from query parameter
        encoded_path = request.args.get('path')
        if not encoded_path:
            return jsonify({'error': 'Missing image path'}), 400
        
        # Decode the URL-encoded path
        blob_path = unquote(encoded_path)
        logger.debug(f"Proxy received path: {blob_path}")
        
        # Extract just the blob name if a full URL was passed
        if blob_path.startswith('http://') or blob_path.startswith('https://'):
            # Parse the URL to extract just the blob name
            parsed = urlparse(blob_path)
            path_parts = parsed.path.strip('/').split('/')
            
            # For Azure blob URLs, the format is: /container/blob_name
            # We want just the blob_name (last part)
            if len(path_parts) >= 2:
                blob_name = path_parts[-1]
                logger.debug(f"Extracted blob name: {blob_name}")
            else:
                logger.error(f"Could not extract blob name from URL: {blob_path}")
                return jsonify({'error': 'Invalid blob URL format'}), 400
        else:
            # It's already just a blob name
            blob_name = blob_path
        
        # Build the full image URL with SAS token
        image_url = build_image_url(
            blob_name,
            current_app.config['AZURE_STORAGE_ACCOUNT_NAME'],
            current_app.config['AZURE_STORAGE_SAS_TOKEN'],
            current_app.config['AZURE_BLOB_CONTAINER_NAME']
        )
        
        if not image_url:
            logger.error(f"Failed to build image URL for blob: {blob_name}")
            return jsonify({'error': 'Failed to build image URL'}), 500
        
        logger.debug(f"Fetching image from: {image_url}")
        
        # Fetch the image with timeout
        response = requests.get(
            image_url, 
            timeout=current_app.config.get('IMAGE_PROXY_TIMEOUT', 10),
            stream=True
        )
        response.raise_for_status()
        
        # Determine content type
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Stream the image back
        return Response(
            response.iter_content(chunk_size=8192),
            content_type=content_type,
            headers={
                'Cache-Control': 'public, max-age=3600',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching image: {e}")
        logger.error(f"Response status: {e.response.status_code if e.response else 'No response'}")
        logger.error(f"Response content: {e.response.text[:500] if e.response else 'No response'}")
        return jsonify({
            'error': 'Image fetch failed',
            'status_code': e.response.status_code if e.response else None
        }), 404
    except requests.exceptions.Timeout:
        logger.error("Image fetch timeout")
        return jsonify({'error': 'Image fetch timeout'}), 504
    except Exception as e:
        logger.error(f"Image proxy error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500