"""
Azure Blob Storage utilities with automatic SAS token generation.
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from typing import Optional

from azure.storage.blob import generate_blob_sas, BlobSasPermissions

logger = logging.getLogger(__name__)


class BlobStorageService:
    """Service for handling blob storage operations with dynamic SAS tokens."""
    
    def __init__(self, account_name: str, account_key: str, container_name: str):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
    
    def generate_sas_token(self, blob_name: str, expiry_minutes: int = 30) -> str:
        """
        Generate a SAS token for a specific blob.
        
        Args:
            blob_name: Name of the blob
            expiry_minutes: Token expiry time in minutes (default: 30)
            
        Returns:
            Complete SAS URL for the blob
        """
        try:
            start_time = datetime.now(timezone.utc)
            expiry_time = start_time + timedelta(minutes=expiry_minutes)
            
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry_time,
                start=start_time
            )
            
            # Construct the full URL
            url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            
            logger.debug(f"Generated SAS token for blob: {blob_name}, expires in {expiry_minutes} minutes")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate SAS token for blob {blob_name}: {str(e)}")
            raise
    
    def is_sas_token_valid(self, sas_token: str) -> bool:
        """
        Check if a SAS token is still valid.
        
        Args:
            sas_token: The SAS token query string
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Parse the SAS token to extract expiry time
            params = {}
            for param in sas_token.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key] = value
            
            # Get the expiry time (se parameter)
            if 'se' not in params:
                return False
            
            # Parse the expiry time
            expiry_str = params['se'].replace('%3A', ':')
            expiry_time = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
            
            # Check if current time is before expiry
            current_time = datetime.now(timezone.utc)
            return expiry_time > current_time
            
        except Exception as e:
            logger.error(f"Error validating SAS token: {str(e)}")
            return False
    
    def extract_blob_name_from_url(self, url: str) -> str:
        """
        Extract blob name from a full blob URL.
        
        Args:
            url: Full blob URL (with or without SAS token)
            
        Returns:
            Blob name
        """
        try:
            parsed_url = urlparse(url)
            # Remove leading slash and container name
            path_parts = parsed_url.path.strip('/').split('/', 1)
            
            if len(path_parts) > 1:
                # Path includes container name, return everything after it
                return path_parts[1]
            else:
                # Path might be just the blob name
                return path_parts[0] if path_parts[0] else ""
                
        except Exception as e:
            logger.error(f"Error extracting blob name from URL {url}: {str(e)}")
            return ""
    
    def regenerate_sas_token_if_needed(self, url: str, expiry_minutes: int = 30) -> str:
        """
        Check if SAS token in URL is valid, regenerate if needed.
        
        Args:
            url: Blob URL with SAS token
            expiry_minutes: Token expiry time for new token
            
        Returns:
            URL with valid SAS token
        """
        try:
            # Check if URL has SAS token
            if '?' not in url:
                # No SAS token, generate new one
                blob_name = self.extract_blob_name_from_url(url)
                return self.generate_sas_token(blob_name, expiry_minutes)
            
            # Extract and validate existing token
            base_url, sas_token = url.split('?', 1)
            
            if self.is_sas_token_valid(sas_token):
                # Token is still valid
                return url
            else:
                # Token expired, generate new one
                blob_name = self.extract_blob_name_from_url(base_url)
                return self.generate_sas_token(blob_name, expiry_minutes)
                
        except Exception as e:
            logger.error(f"Error regenerating SAS token for URL {url}: {str(e)}")
            raise


# Singleton instance
_blob_service: Optional[BlobStorageService] = None


def get_blob_service() -> BlobStorageService:
    """Get or create blob storage service instance."""
    global _blob_service
    
    if _blob_service is None:
        account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
        account_key = os.environ.get('AZURE_STORAGE_ACCESS_KEY')
        container_name = os.environ.get('BLOB_CONTAINER_NAME', 'rag-demo-images')
        
        if not all([account_name, account_key]):
            raise ValueError("Missing required Azure Storage credentials")
        
        _blob_service = BlobStorageService(account_name, account_key, container_name)
    
    return _blob_service