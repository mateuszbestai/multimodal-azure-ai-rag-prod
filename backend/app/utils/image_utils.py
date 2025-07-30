"""
Image utility functions without static SAS token dependency.
"""
import logging
from urllib.parse import urlparse

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


def build_blob_url_without_sas(blob_path: str, storage_account: str, container: str) -> str:
    """Build a blob URL without SAS token (for internal use)."""
    if not blob_path:
        return ""
    
    # If it's already a full URL, return it
    if blob_path.startswith("http://") or blob_path.startswith("https://"):
        # Remove any existing SAS token
        base_url = blob_path.split('?')[0]
        return base_url
    
    # Extract just the filename
    blob_name = blob_path.split('/')[-1].split('?')[0]
    
    # Build URL without SAS token
    if storage_account and container:
        return f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}"
    
    logger.error("Missing storage configuration!")
    return ""


def validate_blob_url(url: str) -> bool:
    """Verify blob URL format is valid."""
    # Check if it's a valid Azure blob storage URL
    if "blob.core.windows.net" in url:
        return True
    
    # For other URLs, basic validation
    return bool(url and url.startswith(('http://', 'https://')))