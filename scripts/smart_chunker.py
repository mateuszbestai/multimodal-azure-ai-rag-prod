import os
import json
import time
import logging
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv
import nest_asyncio
import io
from PIL import Image
from dataclasses import dataclass
from enum import Enum

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.storage.blob import BlobServiceClient, ContainerClient

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)

# Async imports
import asyncio
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
import aiofiles

nest_asyncio.apply()
load_dotenv()

# ================== Configuration ==================
# Environment Variables (same as before)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "aiopsassistant-2")
INDEX_NAME = "aiops-assistant-2"

# Container folder structure
DOCS_FOLDER = "DOCS"
IMAGES_FOLDER = "IMAGES"
PROCESSING_STATUS_FILE = "processing_status.json"

# Local temp directory for processing
TEMP_DOWNLOAD_PATH = "temp_processing"

# Optimization settings
IMAGE_DPI = 100
IMAGE_FORMAT = "JPEG"
IMAGE_QUALITY = 85
MAX_CONCURRENT_UPLOADS = 15

# ================== NEW: Chunking Configuration ==================
class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    PAGE_LEVEL = "page_level"  # Original: one chunk per page
    SENTENCE_BASED = "sentence_based"  # Split by sentences with size limits
    SEMANTIC = "semantic"  # Split by semantic similarity
    HIERARCHICAL = "hierarchical"  # Multi-level chunking
    HYBRID = "hybrid"  # Combine page boundaries with smart splitting

@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 512  # Target chunk size in tokens
    chunk_overlap: int = 50  # Overlap between chunks in tokens
    respect_page_boundaries: bool = True  # Don't split chunks across pages
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1024  # Maximum chunk size
    semantic_breakpoint_threshold: int =70  # For semantic splitting
    include_prev_next_chunks: bool = True  # Add references to adjacent chunks

# Default chunking configuration
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()

# Initialize Azure OpenAI settings (same as before)
Settings.llm = AzureOpenAI(
    engine=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",
    api_type="azure"
)

Settings.embed_model = AzureOpenAIEmbedding(
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",
    api_type="azure"
)

# Initialize Azure Clients (same as before)
document_analysis_client = DocumentAnalysisClient(
    endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
)

search_credential = AzureKeyCredential(SEARCH_SERVICE_API_KEY)
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=search_credential)
search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=search_credential)

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Enhanced metadata fields for the index
metadata_fields = {
    "page_num": ("page_num", MetadataIndexFieldType.INT64),
    "chunk_id": ("chunk_id", MetadataIndexFieldType.STRING),  # NEW: Unique chunk identifier
    "chunk_index": ("chunk_index", MetadataIndexFieldType.INT64),  # NEW: Chunk position within page
    "total_chunks_in_page": ("total_chunks_in_page", MetadataIndexFieldType.INT64),  # NEW
    "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
    "document_name": ("document_name", MetadataIndexFieldType.STRING),
    "image_path": ("image_path", MetadataIndexFieldType.STRING),
    "full_text": ("full_text", MetadataIndexFieldType.STRING),
    "chunk_text": ("chunk_text", MetadataIndexFieldType.STRING),  # NEW: Actual chunk text
    "prev_chunk_id": ("prev_chunk_id", MetadataIndexFieldType.STRING),  # NEW: Reference to previous chunk
    "next_chunk_id": ("next_chunk_id", MetadataIndexFieldType.STRING),  # NEW: Reference to next chunk
    "source_document": ("source_document", MetadataIndexFieldType.STRING),
    "ingestion_date": ("ingestion_date", MetadataIndexFieldType.STRING),
    "chunk_strategy": ("chunk_strategy", MetadataIndexFieldType.STRING),  # NEW: Strategy used
    "headings": ("headings", MetadataIndexFieldType.STRING),  # NEW: Detected headings in chunk
}

# ================== Smart Text Processing ==================
class SmartTextProcessor:
    """Advanced text processing for intelligent chunking."""
    
    @staticmethod
    def detect_structure(text: str) -> Dict[str, Any]:
        """Detect document structure like headings, paragraphs, lists."""
        structure = {
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "sections": []
        }
        
        lines = text.split('\n')
        current_section = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_section:
                    structure["paragraphs"].append('\n'.join(current_section))
                    current_section = []
                continue
            
            # Detect headings (simple heuristic - can be enhanced)
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^\d+\.?\s+[A-Z]', line) or
                 re.match(r'^[A-Z][^.!?]*$', line) and len(line.split()) < 10)):
                structure["headings"].append({
                    "text": line,
                    "line_num": i,
                    "level": 1 if line.isupper() else 2
                })
                if current_section:
                    structure["paragraphs"].append('\n'.join(current_section))
                    current_section = []
            # Detect list items
            elif re.match(r'^[\-\*\•]\s+', line) or re.match(r'^\d+\.\s+', line):
                structure["lists"].append(line)
                current_section.append(line)
            else:
                current_section.append(line)
        
        if current_section:
            structure["paragraphs"].append('\n'.join(current_section))
        
        return structure
    
    @staticmethod
    def split_by_structure(text: str, max_chunk_size: int) -> List[str]:
        """Split text respecting document structure."""
        structure = SmartTextProcessor.detect_structure(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Process paragraphs and maintain heading context
        last_heading = ""
        for para in structure["paragraphs"]:
            # Check if there's a heading before this paragraph
            for heading in structure["headings"]:
                if heading["line_num"] < len(text.split('\n')):
                    last_heading = heading["text"]
                    break
            
            para_size = len(para.split())
            
            if current_size + para_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
                
                # Add heading context to new chunk if exists
                if last_heading and len(last_heading.split()) < 20:
                    current_chunk.append(f"[Context: {last_heading}]")
                    current_size = len(last_heading.split())
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]

# ================== Enhanced Chunking Strategies ==================
class EnhancedChunker:
    """Implements various chunking strategies with image association preservation."""
    
    def __init__(self, config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG):
        self.config = config
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize LlamaIndex parsers for different strategies."""
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Semantic splitter if embedding model is available
        if Settings.embed_model:
            self.semantic_splitter = SemanticSplitterNodeParser(
                embed_model=Settings.embed_model,
                breakpoint_percentile_threshold=self.config.semantic_breakpoint_threshold,
                max_chunk_size=self.config.max_chunk_size
            )
    
    def chunk_document(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Main method to chunk document based on selected strategy."""
        if self.config.strategy == ChunkingStrategy.PAGE_LEVEL:
            return self._page_level_chunking(document_data, image_blob_paths)
        elif self.config.strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._sentence_based_chunking(document_data, image_blob_paths)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(document_data, image_blob_paths)
        elif self.config.strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunking(document_data, image_blob_paths)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunking(document_data, image_blob_paths)
        else:
            logging.warning(f"Unknown strategy {self.config.strategy}, using page-level")
            return self._page_level_chunking(document_data, image_blob_paths)
    
    def _page_level_chunking(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Original page-level chunking for compatibility."""
        nodes = []
        document_name = document_data["document_name"]
        
        page_image_map = self._create_page_image_map(image_blob_paths)
        
        for page_num, page_text in enumerate(document_data["pages"], start=1):
            image_blob_path = page_image_map.get(page_num)
            if not image_blob_path:
                continue
            
            chunk_id = f"{document_name}_p{page_num}_c1"
            
            node = TextNode(
                text=page_text["text"],
                metadata={
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": 1,
                    "total_chunks_in_page": 1,
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "full_text": page_text["text"],
                    "chunk_text": page_text["text"],
                    "source_document": f"{DOCS_FOLDER}/{document_name}.pdf",
                    "ingestion_date": datetime.utcnow().isoformat(),
                    "chunk_strategy": "page_level"
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _sentence_based_chunking(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Split pages into sentence-based chunks with size limits."""
        nodes = []
        document_name = document_data["document_name"]
        page_image_map = self._create_page_image_map(image_blob_paths)
        
        all_chunks = []  # Track all chunks for prev/next references
        
        for page_num, page_data in enumerate(document_data["pages"], start=1):
            page_text = page_data["text"]
            image_blob_path = page_image_map.get(page_num)
            
            if not image_blob_path or not page_text.strip():
                continue
            
            # Create temporary nodes for splitting
            temp_node = TextNode(text=page_text)
            page_chunks = self.sentence_splitter.get_nodes_from_documents([temp_node])
            
            for chunk_idx, chunk_node in enumerate(page_chunks, start=1):
                chunk_id = f"{document_name}_p{page_num}_c{chunk_idx}"
                
                metadata = {
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks_in_page": len(page_chunks),
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "full_text": page_text,  # Keep full page text for reference
                    "chunk_text": chunk_node.text,
                    "source_document": f"{DOCS_FOLDER}/{document_name}.pdf",
                    "ingestion_date": datetime.utcnow().isoformat(),
                    "chunk_strategy": "sentence_based"
                }
                
                node = TextNode(text=chunk_node.text, metadata=metadata)
                all_chunks.append(node)
        
        # Add prev/next chunk references
        nodes = self._add_chunk_references(all_chunks)
        return nodes
    
    def _semantic_chunking(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Split based on semantic similarity."""
        if not hasattr(self, 'semantic_splitter'):
            logging.warning("Semantic splitter not available, falling back to sentence-based")
            return self._sentence_based_chunking(document_data, image_blob_paths)
        
        nodes = []
        document_name = document_data["document_name"]
        page_image_map = self._create_page_image_map(image_blob_paths)
        all_chunks = []
        
        for page_num, page_data in enumerate(document_data["pages"], start=1):
            page_text = page_data["text"]
            image_blob_path = page_image_map.get(page_num)
            
            if not image_blob_path or not page_text.strip():
                continue
            
            # Create temporary nodes for semantic splitting
            temp_node = TextNode(text=page_text)
            try:
                page_chunks = self.semantic_splitter.get_nodes_from_documents([temp_node])
            except Exception as e:
                logging.warning(f"Semantic splitting failed for page {page_num}: {e}")
                # Fallback to sentence splitting
                page_chunks = self.sentence_splitter.get_nodes_from_documents([temp_node])
            
            for chunk_idx, chunk_node in enumerate(page_chunks, start=1):
                chunk_id = f"{document_name}_p{page_num}_c{chunk_idx}"
                
                # Detect any headings in this chunk
                structure = SmartTextProcessor.detect_structure(chunk_node.text)
                headings = [h["text"] for h in structure["headings"]]
                
                metadata = {
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks_in_page": len(page_chunks),
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "full_text": page_text,
                    "chunk_text": chunk_node.text,
                    "headings": json.dumps(headings) if headings else "",
                    "source_document": f"{DOCS_FOLDER}/{document_name}.pdf",
                    "ingestion_date": datetime.utcnow().isoformat(),
                    "chunk_strategy": "semantic"
                }
                
                node = TextNode(text=chunk_node.text, metadata=metadata)
                all_chunks.append(node)
        
        nodes = self._add_chunk_references(all_chunks)
        return nodes
    
    def _hierarchical_chunking(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Create hierarchical chunks - both large and small chunks."""
        nodes = []
        document_name = document_data["document_name"]
        page_image_map = self._create_page_image_map(image_blob_paths)
        
        for page_num, page_data in enumerate(document_data["pages"], start=1):
            page_text = page_data["text"]
            image_blob_path = page_image_map.get(page_num)
            
            if not image_blob_path or not page_text.strip():
                continue
            
            # Create parent chunk (full page)
            parent_chunk_id = f"{document_name}_p{page_num}_parent"
            parent_node = TextNode(
                text=page_text,
                metadata={
                    "page_num": page_num,
                    "chunk_id": parent_chunk_id,
                    "chunk_index": 0,
                    "chunk_level": "parent",
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "chunk_strategy": "hierarchical_parent"
                }
            )
            nodes.append(parent_node)
            
            # Create child chunks (smaller segments)
            temp_node = TextNode(text=page_text)
            child_chunks = self.sentence_splitter.get_nodes_from_documents([temp_node])
            
            for chunk_idx, chunk_node in enumerate(child_chunks, start=1):
                child_chunk_id = f"{document_name}_p{page_num}_c{chunk_idx}"
                
                child_metadata = {
                    "page_num": page_num,
                    "chunk_id": child_chunk_id,
                    "chunk_index": chunk_idx,
                    "chunk_level": "child",
                    "parent_chunk_id": parent_chunk_id,
                    "total_chunks_in_page": len(child_chunks),
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "chunk_text": chunk_node.text,
                    "source_document": f"{DOCS_FOLDER}/{document_name}.pdf",
                    "ingestion_date": datetime.utcnow().isoformat(),
                    "chunk_strategy": "hierarchical_child"
                }
                
                child_node = TextNode(text=chunk_node.text, metadata=child_metadata)
                nodes.append(child_node)
        
        return nodes
    
    def _hybrid_chunking(self, document_data: dict, image_blob_paths: Dict[str, str]) -> List[TextNode]:
        """Hybrid approach: smart splitting within page boundaries."""
        nodes = []
        document_name = document_data["document_name"]
        page_image_map = self._create_page_image_map(image_blob_paths)
        all_chunks = []
        
        for page_num, page_data in enumerate(document_data["pages"], start=1):
            page_text = page_data["text"]
            image_blob_path = page_image_map.get(page_num)
            
            if not image_blob_path or not page_text.strip():
                continue
            
            # Use smart text processor to split by structure
            text_chunks = SmartTextProcessor.split_by_structure(
                page_text, 
                self.config.chunk_size
            )
            
            # If chunks are too large, further split them
            final_chunks = []
            for chunk_text in text_chunks:
                if len(chunk_text.split()) > self.config.max_chunk_size:
                    # Further split large chunks
                    temp_node = TextNode(text=chunk_text)
                    sub_chunks = self.sentence_splitter.get_nodes_from_documents([temp_node])
                    final_chunks.extend([sc.text for sc in sub_chunks])
                elif len(chunk_text.split()) < self.config.min_chunk_size and final_chunks:
                    # Merge small chunks with previous
                    final_chunks[-1] += "\n" + chunk_text
                else:
                    final_chunks.append(chunk_text)
            
            # Create nodes for final chunks
            for chunk_idx, chunk_text in enumerate(final_chunks, start=1):
                chunk_id = f"{document_name}_p{page_num}_c{chunk_idx}"
                
                # Extract structural information
                structure = SmartTextProcessor.detect_structure(chunk_text)
                headings = [h["text"] for h in structure["headings"]]
                
                metadata = {
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks_in_page": len(final_chunks),
                    "image_path": image_blob_path,
                    "doc_id": document_name,
                    "document_name": document_name,
                    "full_text": page_text,
                    "chunk_text": chunk_text,
                    "headings": json.dumps(headings) if headings else "",
                    "source_document": f"{DOCS_FOLDER}/{document_name}.pdf",
                    "ingestion_date": datetime.utcnow().isoformat(),
                    "chunk_strategy": "hybrid"
                }
                
                node = TextNode(text=chunk_text, metadata=metadata)
                all_chunks.append(node)
        
        # Add chunk references for navigation
        nodes = self._add_chunk_references(all_chunks)
        return nodes
    
    def _create_page_image_map(self, image_blob_paths: Dict[str, str]) -> Dict[int, str]:
        """Create mapping from page number to blob path."""
        page_image_map = {}
        for img_name, blob_path in image_blob_paths.items():
            match = re.search(r"page_(\d+)", img_name)
            if match:
                page_num = int(match.group(1))
                page_image_map[page_num] = blob_path
        return page_image_map
    
    def _add_chunk_references(self, chunks: List[TextNode]) -> List[TextNode]:
        """Add previous and next chunk references for better navigation."""
        if not self.config.include_prev_next_chunks:
            return chunks
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["prev_chunk_id"] = chunks[i-1].metadata["chunk_id"]
            else:
                chunk.metadata["prev_chunk_id"] = ""
            
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = chunks[i+1].metadata["chunk_id"]
            else:
                chunk.metadata["next_chunk_id"] = ""
        
        return chunks

# ================== Processing Status Management (same as before) ==================
class ProcessingStatusManager:
    """Manages tracking of processed documents."""
    
    def __init__(self, container_client: ContainerClient):
        self.container_client = container_client
        self.status_blob_name = f"{PROCESSING_STATUS_FILE}"
        self.status_data = self._load_status()
    
    def _load_status(self) -> Dict:
        """Load processing status from blob storage."""
        try:
            blob_client = self.container_client.get_blob_client(self.status_blob_name)
            if blob_client.exists():
                data = blob_client.download_blob().readall()
                return json.loads(data)
            else:
                return {"processed_documents": {}, "last_scan": None}
        except Exception as e:
            logging.warning(f"Could not load processing status: {e}")
            return {"processed_documents": {}, "last_scan": None}
    
    def _save_status(self):
        """Save processing status to blob storage."""
        try:
            blob_client = self.container_client.get_blob_client(self.status_blob_name)
            blob_client.upload_blob(
                json.dumps(self.status_data, indent=2), 
                overwrite=True
            )
        except Exception as e:
            logging.error(f"Failed to save processing status: {e}")
    
    def is_processed(self, doc_name: str, strategy: str = None) -> bool:
        """Check if a document has been processed with specific strategy."""
        if doc_name not in self.status_data["processed_documents"]:
            return False
        
        if strategy:
            doc_info = self.status_data["processed_documents"][doc_name]
            return doc_info.get("chunk_strategy") == strategy
        
        return True
    
    def mark_processed(self, doc_name: str, metadata: Dict):
        """Mark a document as processed."""
        self.status_data["processed_documents"][doc_name] = {
            "processed_at": datetime.utcnow().isoformat(),
            "pages": metadata.get("pages", 0),
            "chunks": metadata.get("chunks", 0),
            "chunk_strategy": metadata.get("chunk_strategy", "unknown"),
            "status": "completed"
        }
        self.status_data["last_scan"] = datetime.utcnow().isoformat()
        self._save_status()
    
    def mark_failed(self, doc_name: str, error: str):
        """Mark a document as failed."""
        self.status_data["processed_documents"][doc_name] = {
            "processed_at": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": error
        }
        self._save_status()

# ================== Document Discovery (same as before) ==================
def discover_new_documents(container_client: ContainerClient, status_manager: ProcessingStatusManager, chunking_config: ChunkingConfig) -> List[str]:
    """Discover new documents in the DOCS folder that haven't been processed."""
    new_documents = []
    docs_prefix = f"{DOCS_FOLDER}/"
    
    logging.info(f"Scanning for documents in {docs_prefix}")
    
    try:
        blob_list = container_client.list_blobs(name_starts_with=docs_prefix)
        
        for blob in blob_list:
            if blob.name == docs_prefix:
                continue
                
            doc_name = blob.name.replace(docs_prefix, "")
            
            # Check if it's a PDF and hasn't been processed with current strategy
            if doc_name.lower().endswith('.pdf'):
                if not status_manager.is_processed(doc_name, chunking_config.strategy.value):
                    new_documents.append(blob.name)
                    logging.info(f"Found document to process: {doc_name}")
        
        logging.info(f"Found {len(new_documents)} documents to process with {chunking_config.strategy.value} strategy")
        return new_documents
        
    except Exception as e:
        logging.error(f"Error discovering documents: {e}")
        return []

# ================== Keep existing helper functions (download, pdf_to_images, etc.) ==================
def download_document(container_client: ContainerClient, blob_name: str, local_path: str) -> str:
    """Download a document from blob storage to local temp directory."""
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.basename(blob_name)
    local_file_path = os.path.join(local_path, filename)
    
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_file_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    
    logging.info(f"Downloaded {blob_name} to {local_file_path}")
    return local_file_path

def pdf_to_images_optimized(pdf_path: str, output_base: str, doc_name: str) -> List[dict]:
    """Convert PDF to optimized images with document-specific naming."""
    image_dicts = []
    folder_path = os.path.join(output_base, Path(doc_name).stem)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise ValueError("Encrypted PDF - password required")

        total_pages = len(doc)
        logging.info(f"Converting {total_pages} pages to images for {doc_name}")
        
        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csRGB, alpha=False)
                
                image_name = f"{Path(doc_name).stem}_page_{page_num+1}.jpg"
                image_path = os.path.join(folder_path, image_name)
                
                img_data = pix.pil_tobytes(format="JPEG", optimize=True)
                img = Image.open(io.BytesIO(img_data))
                img.save(image_path, "JPEG", quality=IMAGE_QUALITY, optimize=True)
                
                image_dicts.append({
                    "name": image_name,
                    "local_path": image_path,
                    "page_num": page_num + 1,
                    "document_name": Path(doc_name).stem
                })
                
                if (page_num + 1) % 10 == 0:
                    logging.info(f"Converted {page_num + 1}/{total_pages} pages")
                
            except Exception as e:
                logging.error(f"Page {page_num+1} processing failed: {str(e)}")
                continue
                
        logging.info(f"Successfully converted {len(image_dicts)} pages to images")
        return image_dicts

    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def extract_document_data(pdf_path: str, doc_name: str) -> dict:
    """Extract text and images from PDF."""
    start_time = time.time()
    
    with open(pdf_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()
    
    text_extraction_time = time.time() - start_time
    logging.info(f"Text extraction took {text_extraction_time:.2f}s")
    
    image_start = time.time()
    image_dicts = pdf_to_images_optimized(pdf_path, TEMP_DOWNLOAD_PATH, doc_name)
    image_conversion_time = time.time() - image_start
    logging.info(f"Image conversion took {image_conversion_time:.2f}s")

    return {
        "text_content": result.content,
        "pages": [{"text": "\n".join(line.content for line in page.lines)} for page in result.pages],
        "images": image_dicts,
        "source_path": pdf_path,
        "document_name": Path(doc_name).stem
    }

# ================== Optimized Blob Upload (same as before) ==================
class OptimizedBlobUploader:
    """Upload images to organized structure in IMAGES folder."""
    
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = None
        
    async def __aenter__(self):
        self.blob_service_client = AsyncBlobServiceClient.from_connection_string(
            self.connection_string
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.blob_service_client:
            await self.blob_service_client.close()
    
    async def upload_images_batch(self, image_dicts: List[dict], document_name: str) -> Dict[str, str]:
        """Upload images to IMAGES/document_name/ structure."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
        tasks = []
        
        logging.info(f"Starting upload of {len(image_dicts)} images for {document_name}")
        
        for img in image_dicts:
            blob_path = f"{IMAGES_FOLDER}/{document_name}/{img['name']}"
            task = self._upload_single_optimized(img, blob_path, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        uploaded_urls = {}
        failed_count = 0
        
        for img, result in zip(image_dicts, results):
            if isinstance(result, Exception):
                logging.error(f"Upload failed for {img['name']}: {str(result)}")
                failed_count += 1
            elif result:
                blob_path = f"{IMAGES_FOLDER}/{document_name}/{img['name']}"
                uploaded_urls[img["name"]] = blob_path
        
        success_count = len(uploaded_urls)
        logging.info(f"Upload complete: {success_count} succeeded, {failed_count} failed")
        
        return uploaded_urls
    
    async def _upload_single_optimized(self, image: dict, blob_path: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Upload single image to specified blob path."""
        async with semaphore:
            try:
                async with aiofiles.open(image["local_path"], "rb") as f:
                    image_data = await f.read()
                
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_path
                )
                
                await blob_client.upload_blob(
                    image_data, 
                    overwrite=True,
                    max_concurrency=4,
                    length=len(image_data)
                )
                
                return blob_path
                
            except Exception as e:
                logging.error(f"Upload failed for {blob_path}: {str(e)}")
                return None

async def upload_images_concurrently(image_dicts: List[dict], document_name: str) -> Dict[str, str]:
    """Upload images using optimized uploader."""
    async with OptimizedBlobUploader(BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME) as uploader:
        return await uploader.upload_images_batch(image_dicts, document_name)

# ================== Enhanced Search Index Integration ==================
def create_search_nodes(document_data: dict, image_blob_paths: Dict[str, str], chunking_config: ChunkingConfig) -> List[TextNode]:
    """Create search nodes using enhanced chunking strategies."""
    chunker = EnhancedChunker(chunking_config)
    nodes = chunker.chunk_document(document_data, image_blob_paths)
    
    logging.info(f"Created {len(nodes)} search nodes using {chunking_config.strategy.value} strategy")
    return nodes

# ================== Keep existing index functions (ensure_index_exists, create_vector_store, add_nodes_to_index) ==================
def ensure_index_exists():
    """Ensure the search index exists with proper configuration."""
    try:
        existing_indexes = [index.name for index in index_client.list_indexes()]
        
        if INDEX_NAME in existing_indexes:
            logging.info(f"Index '{INDEX_NAME}' already exists")
            return True
            
        logging.info(f"Index '{INDEX_NAME}' not found. Creating new index...")
        
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=index_client,
            index_name=INDEX_NAME,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="chunk_text",  # Changed to use chunk_text
            embedding_field_key="embedding",
            embedding_dimensionality=3072,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            filterable_metadata_field_keys=metadata_fields,
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        dummy_node = TextNode(
            text="Initial index creation",
            metadata={
                "page_num": 0,
                "chunk_id": "init",
                "chunk_index": 0,
                "total_chunks_in_page": 0,
                "image_path": "",
                "doc_id": "init",
                "document_name": "init",
                "chunk_text": "Initial index creation",
                "full_text": "Initial index creation",
                "source_document": "",
                "ingestion_date": datetime.utcnow().isoformat(),
                "chunk_strategy": "init"
            }
        )
        
        index = VectorStoreIndex(
            nodes=[dummy_node],
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=False,
        )
        
        logging.info(f"Successfully created index '{INDEX_NAME}'")
        return True
        
    except Exception as e:
        logging.error(f"Failed to ensure index exists: {e}")
        return False

def create_vector_store(index_client, force_create: bool = False) -> AzureAISearchVectorStore:
    """Create or get existing Azure AI Search vector store."""
    try:
        existing_indexes = [index.name for index in index_client.list_indexes()]
        index_exists = INDEX_NAME in existing_indexes
        
        if not index_exists:
            logging.info(f"Index '{INDEX_NAME}' does not exist. Creating new index...")
            index_management = IndexManagement.CREATE_IF_NOT_EXISTS
        elif force_create:
            logging.info(f"Force creating index '{INDEX_NAME}'...")
            index_management = IndexManagement.CREATE_OR_UPDATE
        else:
            logging.info(f"Using existing index '{INDEX_NAME}'")
            index_management = IndexManagement.NO_VALIDATION
            
    except Exception as e:
        logging.warning(f"Could not list indexes: {e}. Will attempt to create if not exists.")
        index_management = IndexManagement.CREATE_IF_NOT_EXISTS
    
    return AzureAISearchVectorStore(
        search_or_index_client=index_client,
        index_name=INDEX_NAME,
        index_management=index_management,
        id_field_key="id",
        chunk_field_key="chunk_text",  # Changed to use chunk_text
        embedding_field_key="embedding",
        embedding_dimensionality=3072,
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        filterable_metadata_field_keys=metadata_fields,
        language_analyzer="en.lucene",
        vector_algorithm_type="exhaustiveKnn",
    )

def add_nodes_to_index(text_nodes: List[TextNode]) -> VectorStoreIndex:
    """Add new nodes to existing index or create new one."""
    try:
        existing_indexes = [index.name for index in index_client.list_indexes()]
        index_exists = INDEX_NAME in existing_indexes
    except Exception as e:
        logging.warning(f"Could not check if index exists: {e}")
        index_exists = False
    
    vector_store = create_vector_store(index_client, force_create=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if not index_exists:
        logging.info(f"Creating new index '{INDEX_NAME}' with {len(text_nodes)} nodes...")
        index = VectorStoreIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            show_progress=True,
        )
    else:
        logging.info(f"Loading existing index '{INDEX_NAME}'...")
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )
        if text_nodes:
            logging.info(f"Adding {len(text_nodes)} new nodes to existing index...")
            index.insert_nodes(text_nodes)
    
    return index

# ================== Enhanced Processing Pipeline ==================
async def process_single_document(blob_name: str, status_manager: ProcessingStatusManager, chunking_config: ChunkingConfig) -> bool:
    """Process a single document with configurable chunking strategy."""
    doc_name = os.path.basename(blob_name)
    temp_dir = os.path.join(TEMP_DOWNLOAD_PATH, Path(doc_name).stem)
    
    try:
        total_start_time = time.time()
        logging.info(f"Processing document: {doc_name} with {chunking_config.strategy.value} strategy")
        
        # Download document
        local_pdf_path = download_document(container_client, blob_name, temp_dir)
        
        # Extract document data
        document_data = extract_document_data(local_pdf_path, doc_name)
        
        # Upload images
        upload_start = time.time()
        image_blob_paths = await upload_images_concurrently(
            document_data["images"], 
            document_data["document_name"]
        )
        upload_time = time.time() - upload_start
        logging.info(f"Uploaded {len(image_blob_paths)} images in {upload_time:.2f}s")
        
        # Create search nodes with enhanced chunking
        nodes = create_search_nodes(document_data, image_blob_paths, chunking_config)
        
        # Add to index
        index_start = time.time()
        index = add_nodes_to_index(text_nodes=nodes)
        index_time = time.time() - index_start
        logging.info(f"Index update took {index_time:.2f}s")
        
        # Mark as processed with strategy info
        status_manager.mark_processed(doc_name, {
            "pages": len(document_data["pages"]),
            "images": len(image_blob_paths),
            "chunks": len(nodes),
            "chunk_strategy": chunking_config.strategy.value
        })
        
        # Clean up temp files
        cleanup_temp_files(temp_dir)
        
        # Log performance
        total_time = time.time() - total_start_time
        logging.info(f"Successfully processed {doc_name} in {total_time:.2f}s")
        logging.info(f"Created {len(nodes)} chunks from {len(document_data['pages'])} pages")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to process {doc_name}: {str(e)}")
        status_manager.mark_failed(doc_name, str(e))
        cleanup_temp_files(temp_dir)
        return False

def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files after processing."""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Failed to clean up temp files: {e}")

async def process_all_new_documents(chunking_config: ChunkingConfig = None):
    """Main function to process all new documents with specified chunking strategy."""
    if chunking_config is None:
        chunking_config = DEFAULT_CHUNKING_CONFIG
    
    # Ensure index exists
    if not ensure_index_exists():
        logging.error("Failed to ensure index exists. Aborting.")
        return
    
    status_manager = ProcessingStatusManager(container_client)
    
    # Discover new documents
    new_documents = discover_new_documents(container_client, status_manager, chunking_config)
    
    if not new_documents:
        logging.info("No new documents to process")
        return
    
    logging.info(f"Starting processing of {len(new_documents)} documents with {chunking_config.strategy.value} strategy")
    
    # Process documents
    success_count = 0
    for doc_blob_name in new_documents:
        success = await process_single_document(doc_blob_name, status_manager, chunking_config)
        if success:
            success_count += 1
    
    logging.info(f"Processing complete: {success_count}/{len(new_documents)} documents processed successfully")

# ================== Main Entry Point ==================
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('document_ingestion.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create temp directory
    Path(TEMP_DOWNLOAD_PATH).mkdir(parents=True, exist_ok=True)
    
    # Configure chunking strategy (can be modified via environment variables or config file)
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy.HYBRID,  # Change this to try different strategies
        chunk_size=512,
        chunk_overlap=50,
        respect_page_boundaries=True,
        min_chunk_size=100,
        max_chunk_size=1024
    )
    
    logging.info("=== Starting Enhanced Document Ingestion Process ===")
    logging.info(f"Container: {BLOB_CONTAINER_NAME}")
    logging.info(f"Documents folder: {DOCS_FOLDER}")
    logging.info(f"Images folder: {IMAGES_FOLDER}")
    logging.info(f"Chunking Strategy: {chunking_config.strategy.value}")
    logging.info(f"Chunk Size: {chunking_config.chunk_size}")
    logging.info(f"Chunk Overlap: {chunking_config.chunk_overlap}")
    logging.info("=" * 50)
    
    try:
        asyncio.run(process_all_new_documents(chunking_config))
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        # Final cleanup
        if os.path.exists(TEMP_DOWNLOAD_PATH):
            try:
                import shutil
                shutil.rmtree(TEMP_DOWNLOAD_PATH)
                logging.info("Cleaned up all temp files")
            except:
                pass