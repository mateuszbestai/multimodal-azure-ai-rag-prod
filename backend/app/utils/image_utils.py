"""
Image utility functions.
"""
import logging
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)


def extract_blob_name_from_url(url: str) -> str:
    """Extract just the blob name (e.g., 'page_74.jpg') from a full blob URL."""
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Get the path and remove leading slash
        path = parsed.path.lstrip('/')
        
        # Split by '/' and get the last part (the filename)
        parts = path.split('/')
        if parts:
            return parts[-1]  # Return just the filename
        return ""
    except Exception as e:
        logger.error(f"Error extracting blob name from {url}: {str(e)}")
        return ""


def build_image_url(blob_path: str, storage_account: str, sas_token: str, container: str) -> str:
    """Build a complete blob URL with SAS token."""
    if not blob_path:
        return ""
    
    # If it's already a full URL, return it
    if blob_path.startswith("http://") or blob_path.startswith("https://"):
        return blob_path
    
    # Extract just the filename
    blob_name = blob_path.split('/')[-1].split('?')[0]
    
    # Build URL with SAS token
    if storage_account and sas_token:
        sas = sas_token if sas_token.startswith('?') else f'?{sas_token}'
        return f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}{sas}"
    
    logger.error("Missing storage configuration!")
    return ""


def validate_image_url(url: str) -> bool:
    """Verify image URL is accessible - lenient for private endpoints."""
    # Always return True for blob storage URLs since we'll proxy them
    if "blob.core.windows.net" in url:
        return True
    
    # For other URLs, you might want to actually check them
    # This is a simplified version
    return bool(url and url.startswith(('http://', 'https://')))