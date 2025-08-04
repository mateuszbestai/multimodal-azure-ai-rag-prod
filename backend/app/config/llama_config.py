"""
LlamaIndex configuration and initialization.
"""

import os
import logging
from flask import current_app
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    MetadataIndexFieldType
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration for frontend components"""
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
    SEARCH_SERVICE_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
    INDEX_NAME = "azure-multimodal-search-3"
    BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER_NAME", "rag-demo-images-2")
    STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    # Removed SAS_TOKEN - now using dynamic generation with access key


def initialize_llama_components(app):
    """Initialize LlamaIndex components with app configuration."""
    config = app.config
    
    try:
        # Initialize LLM
        llm = AzureOpenAI(
            engine="gpt-4.1",
            deployment_name=Config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",
            streaming=True
        )
        
        # Initialize embeddings
        embed_model = AzureOpenAIEmbedding(
            engine="text-embedding-3-large",
            deployment_name=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01"
        )
        
        # Initialize multimodal LLM
        multimodal_llm = AzureOpenAIMultiModal(
            deployment_name=Config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            engine="gpt-4.1",
            max_new_tokens=4096,
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01"
        )
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Initialize search client
        search_client = SearchClient(
            endpoint=Config.SEARCH_SERVICE_ENDPOINT,
            index_name=Config.INDEX_NAME,
            credential=AzureKeyCredential(Config.SEARCH_SERVICE_KEY)
        )
        
        # Initialize vector store
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=search_client,
            id_field_key="id",
            chunk_field_key="full_text",
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            embedding_field_key="embedding",
            embedding_dimensionality=3072,
            filterable_metadata_field_keys={
                "page_num": ("page_num", MetadataIndexFieldType.INT64),
                "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
                "image_path": ("image_path", MetadataIndexFieldType.STRING),
                "full_text": ("full_text", MetadataIndexFieldType.STRING),
            },
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        
        # Create storage context and index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=[], 
            storage_context=storage_context
        )
        
        # Store components in app extensions
        app.extensions['llm'] = llm
        app.extensions['embed_model'] = embed_model
        app.extensions['multimodal_llm'] = multimodal_llm
        app.extensions['search_client'] = search_client
        app.extensions['vector_store'] = vector_store
        app.extensions['index'] = index
        
        logger.info("Successfully initialized LlamaIndex components")
        
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex components: {str(e)}")
        raise


def get_llm():
    """Get LLM instance from current app context."""
    return current_app.extensions.get('llm')


def get_multimodal_llm():
    """Get multimodal LLM instance from current app context."""
    return current_app.extensions.get('multimodal_llm')


def get_index():
    """Get index instance from current app context."""
    return current_app.extensions.get('index')