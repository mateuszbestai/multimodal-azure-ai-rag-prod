"""
Chat service - business logic for chat functionality.
"""
import logging
import requests
from typing import Dict, List, Generator, Any
from flask import current_app

from app.config.llama_config import get_index, get_multimodal_llm
from app.core.query_engine import VisionQueryEngine

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations."""
    
    def __init__(self):
        self._query_engine = None
    
    @property
    def query_engine(self):
        """Lazy initialization of query engine."""
        if self._query_engine is None:
            index = get_index()
            multimodal_llm = get_multimodal_llm()
            
            self._query_engine = VisionQueryEngine(
                retriever=index.as_retriever(similarity_top_k=3),
                multi_modal_llm=multimodal_llm
            )
        
        return self._query_engine
    
    def process_chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message and return response."""
        try:
            response = self.query_engine.custom_query(message)
            
            # Extract pages and convert image URLs to proxy URLs
            pages = list(response.metadata.get('pages', []))
            valid_images = []
            
            for node in response.source_nodes:
                image_path = node.metadata.get('image_path')
                if image_path:
                    proxy_url = self._create_proxy_url(image_path)
                    valid_images.append(proxy_url)
            
            # Build source previews
            source_previews = self._build_source_previews(response.source_nodes)
            
            return {
                'response': response.response,
                'sources': {
                    'pages': pages,
                    'images': valid_images
                },
                'sourcePreviews': source_previews
            }
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}", exc_info=True)
            raise
    
    def stream_chat(self, message: str) -> Generator[Dict[str, Any], None, None]:
        """Stream chat response."""
        try:
            response_gen, metadata = self.query_engine.stream_query(message)
            
            # Send metadata first
            yield {
                'type': 'metadata',
                'data': metadata
            }
            
            # Stream response chunks
            for chunk in response_gen:
                if chunk.delta:
                    yield {
                        'type': 'chunk',
                        'data': chunk.delta
                    }
            
            # Send completion signal
            yield {'type': 'done'}
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            raise
    
    def get_debug_metadata(self, query: str) -> Dict[str, Any]:
        """Get debug metadata for a query."""
        nodes = self.query_engine.retriever.retrieve(query)
        
        metadata_info = []
        for i, node in enumerate(nodes):
            node_info = {
                'index': i,
                'page_num': node.metadata.get('page_num'),
                'image_path': node.metadata.get('image_path'),
                'doc_id': node.metadata.get('doc_id'),
                'content_preview': node.get_content()[:100] + '...'
            }
            metadata_info.append(node_info)
        
        return {
            'query': query,
            'nodes_found': len(nodes),
            'metadata': metadata_info
        }
    
    def _create_proxy_url(self, image_path: str) -> str:
        """Create proxy URL for image."""
        encoded_path = requests.utils.quote(image_path)
        return f"/api/image/proxy?path={encoded_path}"
    
    def _build_source_previews(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Build source preview data."""
        previews = []
        
        for node in source_nodes:
            image_path = node.metadata.get('image_path')
            image_url = None
            
            if image_path:
                image_url = self._create_proxy_url(image_path)
            
            preview = {
                'page': node.metadata.get('page_num', 'N/A'),
                'content': node.get_content(metadata_mode='LLM')[:250] + "...",
                'imageUrl': image_url
            }
            
            previews.append(preview)
        
        return previews