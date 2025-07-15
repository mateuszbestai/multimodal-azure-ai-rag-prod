"""
Request and response schemas.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ChatRequest:
    """Chat request schema."""
    message: str
    
    def __post_init__(self):
        """Validate the request."""
        if not self.message or not self.message.strip():
            raise ValueError("Message cannot be empty")
        
        # Limit message length
        if len(self.message) > 4000:
            raise ValueError("Message too long (max 4000 characters)")


@dataclass
class SourcePreview:
    """Source preview schema."""
    page: str
    content: str
    imageUrl: Optional[str] = None
    category: Optional[str] = None
    title: Optional[str] = None
    
    def dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Sources:
    """Sources schema."""
    pages: List[int]
    images: List[str]
    
    def dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ChatResponse:
    """Chat response schema."""
    response: str
    sources: Optional[Sources] = None
    sourcePreviews: Optional[List[SourcePreview]] = None
    
    def dict(self):
        """Convert to dictionary."""
        data = {'response': self.response}
        
        if self.sources:
            data['sources'] = self.sources.dict() if hasattr(self.sources, 'dict') else self.sources
        
        if self.sourcePreviews:
            data['sourcePreviews'] = [
                preview.dict() if hasattr(preview, 'dict') else preview 
                for preview in self.sourcePreviews
            ]
        
        return data


@dataclass
class StreamChunk:
    """Stream chunk schema."""
    type: str  # 'metadata', 'chunk', 'done', 'error'
    data: Optional[Any] = None
    message: Optional[str] = None
    
    def dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HealthStatus:
    """Health status schema."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: str
    checks: Optional[Dict[str, Any]] = None
    
    def dict(self):
        """Convert to dictionary."""
        return asdict(self)