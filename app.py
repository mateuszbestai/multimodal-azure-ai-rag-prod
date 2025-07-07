import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from typing import Optional, List
import logging
import requests
import json
import urllib.parse
from urllib.parse import urlparse, unquote

import base64
from PIL import Image
from io import BytesIO

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)

# ================== Load Environment ==================
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ================== Initialize Flask FIRST ==================
app = Flask(__name__)

# ================== Configure CORS after creating app ==================
# Enhanced CORS configuration for streaming support
CORS(app, 
     origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5001", "http://localhost:5001"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Content-Type", "X-Content-Type-Options", "X-Frame-Options"])

# ================== Configuration Class ==================
class FrontendConfig:
    """Centralized configuration for frontend components"""
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
    SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    SEARCH_SERVICE_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    INDEX_NAME = "azure-multimodal-search-new"  # Matches ingest.py
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER_NAME", "rag-demo-images")
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

    @classmethod
    def build_image_url(cls, blob_path: str) -> str:
        """Build a complete blob URL with SAS token"""
        if not blob_path:
            return ""
        
        # If it's already a full URL, return it
        if blob_path.startswith("http://") or blob_path.startswith("https://"):
            return blob_path
        
        # Extract just the filename
        blob_name = blob_path.split('/')[-1].split('?')[0]
        
        # Build URL with SAS token
        if cls.STORAGE_ACCOUNT_NAME and cls.SAS_TOKEN:
            sas = cls.SAS_TOKEN if cls.SAS_TOKEN.startswith('?') else f'?{cls.SAS_TOKEN}'
            return f"https://{cls.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{cls.BLOB_CONTAINER}/{blob_name}{sas}"
        
        logger.error("Missing storage configuration!")
        return ""

def extract_blob_name_from_url(url: str) -> str:
    """Extract just the blob name (e.g., 'page_74.jpg') from a full blob URL"""
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

def validate_env():
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_SEARCH_SERVICE_ENDPOINT',
        'AZURE_SEARCH_ADMIN_KEY',
        'AZURE_STORAGE_ACCOUNT_NAME',
        'AZURE_STORAGE_SAS_TOKEN',
        'AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME',
        'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Log the deployment names for debugging
    logger.info(f"Azure OpenAI Endpoint: {FrontendConfig.AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Chat Deployment Name: {FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT}")
    logger.info(f"Embedding Deployment Name: {FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")

validate_env()

# ================== Multimodal LLM ==================
try:
    # Use deployment_name instead of engine, and ensure model matches
    azure_openai_mm_llm = AzureOpenAIMultiModal(
        deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,  # Changed from engine
        api_version="2024-02-01",  # Use stable API version
        model="gpt-4.1",  # Specify the actual model
        max_new_tokens=4096,
        api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
        azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,  # Changed from api_base
    )
    logger.info("Successfully initialized AzureOpenAIMultiModal")
except Exception as e:
    logger.error(f"Failed to initialize AzureOpenAIMultiModal: {str(e)}")
    logger.error("Please check your deployment name in Azure Portal")
    raise

# ================== Enhanced Prompt Template ==================
QA_PROMPT_TMPL = """\
You are a helpful AI assistant with access to both text and images. 
Use the document text and any associated images to provide the best possible answer.
Do not use knowledge outside of the provided documents.

DOCUMENT CONTEXT:
{context_str}

INSTRUCTIONS:
1. If using image information, clearly state which page(s) you are referencing.
2. Integrate text and image details to form a coherent answer.
3. If there are contradictions or missing information, explain them.
4. Give a concise yet thorough answer, and cite relevant pages or images.

USER QUERY:
{query_str}

Now craft your final answer:
"""
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

# ================== Enhanced Query Engine ==================
class VisionQueryEngine(CustomQueryEngine):
    """Updated query engine with image proxy support for private blob storage"""
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: AzureOpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs):
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def fetch_and_convert_image(self, image_url: str) -> Optional[str]:
        """Fetch image from private blob storage and convert to base64"""
        try:
            # Use requests to fetch the image (your backend has network access)
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Open image to validate and potentially resize
            img = Image.open(BytesIO(response.content))
            
            # Resize if too large (Azure OpenAI has limits)
            max_size = (2048, 2048)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output = BytesIO()
            img_format = img.format if img.format else 'JPEG'
            img.save(output, format=img_format)
            output.seek(0)
            
            # Convert to base64
            base64_image = base64.b64encode(output.read()).decode('utf-8')
            return f"data:image/{img_format.lower()};base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Failed to fetch/convert image from {image_url}: {str(e)}")
            return None

    def custom_query(self, query_str: str) -> LlamaResponse:
        nodes = self.retriever.retrieve(query_str)
        
        # Build image nodes with base64 data
        image_nodes = []
        failed_images = []
        
        for n in nodes:
            blob_path = n.metadata.get("image_path")
            if blob_path:
                try:
                    full_url = FrontendConfig.build_image_url(blob_path)
                    # Fetch and convert to base64
                    base64_data = self.fetch_and_convert_image(full_url)
                    
                    if base64_data:
                        # Create image node with base64 data
                        img_node = ImageNode()
                        img_node.image = base64_data  # Use base64 data instead of URL
                        img_node.metadata = {"page_num": n.metadata.get("page_num", "N/A")}
                        image_nodes.append(NodeWithScore(node=img_node))
                    else:
                        failed_images.append(blob_path)
                except Exception as e:
                    logger.error(f"Image node error: {str(e)}")
                    failed_images.append(blob_path)
        
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
        
        # Build the textual context
        context_str = "\n".join([
            f"Page {n.metadata.get('page_num', '?')}: {n.get_content(metadata_mode=MetadataMode.LLM)}"
            for n in nodes
        ])
        
        try:
            formatted_prompt = self.qa_prompt.format(
                context_str=context_str,
                query_str=query_str
            )
            
            # Try with images first
            try:
                response = self.multi_modal_llm.complete(
                    prompt=formatted_prompt,
                    image_documents=[n.node for n in image_nodes],
                )
            except Exception as img_error:
                if "image" in str(img_error).lower():
                    logger.warning(f"Image processing failed, falling back to text-only: {str(img_error)}")
                    # Fallback to text-only response
                    response = self.multi_modal_llm.complete(
                        prompt=formatted_prompt,
                        image_documents=[],  # No images
                    )
                else:
                    raise
            
            if not response or not str(response).strip():
                raise ValueError("Empty response from OpenAI")

            # Build references
            references = []
            for n in nodes:
                ref_text = f"Page {n.metadata.get('page_num', 'N/A')}: {n.get_content(metadata_mode=MetadataMode.LLM)[:100]}..."
                if n.metadata.get("image_path"):
                    ref_text += " [Image available]"
                references.append(ref_text)
            
            return LlamaResponse(
                response=str(response),
                source_nodes=nodes,
                metadata={
                    "references": references,
                    "pages": list({int(n.metadata.get("page_num", 0)) for n in nodes if n.metadata.get("page_num")}),
                    "images": [FrontendConfig.build_image_url(n.metadata.get("image_path")) 
                              for n in nodes if n.metadata.get("image_path")]
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def stream_query(self, query_str: str):
        """Stream the response with image proxy support"""
        nodes = self.retriever.retrieve(query_str)
        
        # Build image nodes with base64 data
        image_nodes = []
        public_image_urls = []  # Keep track of URLs for frontend display
        
        for n in nodes:
            blob_path = n.metadata.get("image_path")
            if blob_path:
                try:
                    full_url = FrontendConfig.build_image_url(blob_path)
                    public_image_urls.append(full_url)  # Store for frontend
                    
                    # Fetch and convert to base64 for Azure OpenAI
                    base64_data = self.fetch_and_convert_image(full_url)
                    
                    if base64_data:
                        img_node = ImageNode()
                        img_node.image = base64_data
                        img_node.metadata = {"page_num": n.metadata.get("page_num", "N/A")}
                        image_nodes.append(NodeWithScore(node=img_node))
                except Exception as e:
                    logger.error(f"Image processing error: {str(e)}")
        
        # Build context
        context_str = "\n".join([
            f"Page {n.metadata.get('page_num', '?')}: {n.get_content(metadata_mode=MetadataMode.LLM)}"
            for n in nodes
        ])
        
        formatted_prompt = self.qa_prompt.format(
            context_str=context_str,
            query_str=query_str
        )
        
        # Stream with fallback
        try:
            response_gen = self.multi_modal_llm.stream_complete(
                prompt=formatted_prompt,
                image_documents=[n.node for n in image_nodes],
            )
        except Exception as e:
            if "image" in str(e).lower():
                logger.warning("Falling back to text-only streaming")
                response_gen = self.multi_modal_llm.stream_complete(
                    prompt=formatted_prompt,
                    image_documents=[],
                )
            else:
                raise
        
        # Extract metadata
        pages = list({int(n.metadata.get("page_num", 0)) for n in nodes if n.metadata.get("page_num")})
        
        # Build source previews (URLs are for frontend display only)
        source_previews = []
        for node in nodes:
            image_path = node.metadata.get('image_path')
            image_url = None
            if image_path:
                # Encode the path for URL safety
                encoded_path = requests.utils.quote(image_path)
                image_url = f"/api/image/proxy?path={encoded_path}"
            
            source_previews.append({
                'page': node.metadata.get('page_num', 'N/A'),
                'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                'imageUrl': image_url  # This now points to our proxy
            })

        proxy_image_urls = []
        for node in nodes:
            image_path = node.metadata.get('image_path')
            if image_path:
                encoded_path = requests.utils.quote(image_path)
                proxy_url = f"/api/image/proxy?path={encoded_path}"
                proxy_image_urls.append(proxy_url)
        
        return response_gen, {
            'pages': pages,
            'images': proxy_image_urls,  # Send URLs for frontend reference
            'sourcePreviews': source_previews
        }

# ================== Initialize Query Engine ==================
def initialize_engine():
    """Initialize Azure components and query engine"""
    try:
        # Test the deployment first
        llm = AzureOpenAI(
            model="gpt-4.1",  # Specify the base model
            deployment_name=FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",  # Use stable API version
            streaming=True
        )

        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-3-large",  # Specify the base model
            deployment_name=FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=FrontendConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=FrontendConfig.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",  # Use stable API version
        )

        # Test the deployments
        logger.info("Testing LLM deployment...")
        test_response = llm.complete("Hello")
        logger.info(f"LLM test successful: {test_response}")

        # Tie these to the global Settings (matching ingest.py)
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Here we use a SearchClient to READ from the existing index
        search_client = SearchClient(
            endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
            index_name=FrontendConfig.INDEX_NAME,
            credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
        )

        # Vector store reading from the existing index
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=SearchClient(
                endpoint=FrontendConfig.SEARCH_SERVICE_ENDPOINT,
                index_name=FrontendConfig.INDEX_NAME,  # index name is already here
                credential=AzureKeyCredential(FrontendConfig.SEARCH_SERVICE_KEY)
            ),
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
        # Load existing index (which ingest.py already populated)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents=[], storage_context=storage_context)

        return VisionQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            multi_modal_llm=azure_openai_mm_llm,
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        logger.error(f"Current deployment names - Chat: {FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT}, Embedding: {FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
        logger.error("Please verify these deployment names exist in your Azure OpenAI resource")
        raise

query_engine = initialize_engine()

# ================== Validate URL ==================
def validate_image_url(url: str) -> bool:
    """Verify image URL is accessible - lenient for private endpoints"""
    # Always return True for blob storage URLs since we'll proxy them
    if "blob.core.windows.net" in url:
        return True
    
    try:
        response = requests.head(url, timeout=3)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Image validation failed for {url}: {str(e)}")
        # Return True anyway to allow proxy attempt
        return True
    
@app.route('/')
def serve_file():
    return send_from_directory("./frontend/dist/", "index.html")

@app.route('/assets/<filename>')
def serve_asset(filename: str):
    return send_from_directory("./frontend/dist/assets/", filename)

# ================== API Endpoints ==================
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Process chat messages and return formatted response"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
        
        response = query_engine.custom_query(query)
        
        # Extract and validate response components
        pages = list(response.metadata.get('pages', []))
        
        # Convert to proxy URLs
        valid_images = []
        for node in response.source_nodes:
            image_path = node.metadata.get('image_path')
            if image_path:
                encoded_path = requests.utils.quote(image_path)
                proxy_url = f"/api/image/proxy?path={encoded_path}"
                valid_images.append(proxy_url)

        # Build source previews with proxy URLs
        source_previews = []
        for node in response.source_nodes:
            image_path = node.metadata.get('image_path')
            
            image_url = None
            if image_path:
                encoded_path = requests.utils.quote(image_path)
                image_url = f"/api/image/proxy?path={encoded_path}"

            source_previews.append({
                'page': node.metadata.get('page_num', 'N/A'),
                'content': node.get_content(metadata_mode=MetadataMode.LLM)[:250] + "...",
                'imageUrl': image_url
            })
        
        return jsonify({
            'response': response.response,
            'sources': {
                'pages': pages,
                'images': valid_images
            },
            'sourcePreviews': source_previews
        })
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request.'
        }), 500

@app.route('/api/chat/stream', methods=['POST', 'GET'])
def handle_chat_stream():
    """Process chat messages and return streaming response using chunked transfer encoding"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Empty query received'}), 400
        
        def generate():
            try:
                response_gen, metadata = query_engine.stream_query(query)
                
                # Send metadata first as a JSON line
                yield json.dumps({
                    'type': 'metadata',
                    'data': metadata
                }) + '\n'
                
                # Stream the response chunks
                for chunk in response_gen:
                    if chunk.delta:
                        yield json.dumps({
                            'type': 'chunk',
                            'data': chunk.delta
                        }) + '\n'
                
                # Send done signal
                yield json.dumps({'type': 'done'}) + '\n'
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield json.dumps({
                    'type': 'error',
                    'message': str(e)
                }) + '\n'
        
        return Response(
            stream_with_context(generate()),
            mimetype='application/x-ndjson',  # Using newline-delimited JSON
            headers={
                'X-Content-Type-Options': 'nosniff',
                'Transfer-Encoding': 'chunked'
            }
        )
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to process your request.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with deployment info"""
    return jsonify({
        'status': 'healthy',
        'deployments': {
            'chat': FrontendConfig.AZURE_OPENAI_CHAT_DEPLOYMENT,
            'embedding': FrontendConfig.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        },
        'endpoint': FrontendConfig.AZURE_OPENAI_ENDPOINT
    })

@app.route('/api/image/proxy', methods=['GET'])
def proxy_image():
    """Proxy images from private blob storage for frontend display"""
    try:
        # Get the blob path from query parameter
        encoded_path = request.args.get('path')
        if not encoded_path:
            return jsonify({'error': 'Missing image path'}), 400
        
        # Decode the URL-encoded path
        blob_path = unquote(encoded_path)
        logger.debug(f"Proxy received path: {blob_path}")
        
        # Determine the actual blob name/path
        if 'blob.core.windows.net' in blob_path:
            # It's a full URL - extract just the blob name
            blob_name = extract_blob_name_from_url(blob_path)
            if not blob_name:
                logger.error(f"Could not extract blob name from URL: {blob_path}")
                return jsonify({'error': 'Invalid blob URL'}), 400
            
            # Rebuild the URL with our credentials
            image_url = FrontendConfig.build_image_url(blob_name)
            logger.debug(f"Extracted blob name: {blob_name}, rebuilt URL: {image_url}")
        else:
            # It's already just a blob name/path
            image_url = FrontendConfig.build_image_url(blob_path)
            logger.debug(f"Using blob path directly: {blob_path} -> {image_url}")
        
        # Fetch the image
        response = requests.get(image_url, timeout=10, stream=True)
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
        logger.error(f"Status code: {e.response.status_code if e.response else 'No response'}")
        logger.error(f"URL attempted: {image_url if 'image_url' in locals() else 'URL not built'}")
        return jsonify({'error': f'Image fetch failed: {str(e)}'}), 404
    except Exception as e:
        logger.error(f"Image proxy error: {str(e)}")
        logger.error(f"Path received: {encoded_path}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Proxy error: {str(e)}'}), 500

@app.route('/api/debug/metadata', methods=['POST'])
def debug_metadata():
    """Debug endpoint to see what's in the metadata"""
    try:
        data = request.get_json()
        query = data.get('message', 'test query')
        
        # Retrieve nodes
        nodes = query_engine.retriever.retrieve(query)
        
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
            
            # Log the information
            logger.info(f"Node {i} metadata:")
            logger.info(f"  - page_num: {node.metadata.get('page_num')}")
            logger.info(f"  - image_path: {node.metadata.get('image_path')}")
            logger.info(f"  - doc_id: {node.metadata.get('doc_id')}")
        
        return jsonify({
            'query': query,
            'nodes_found': len(nodes),
            'metadata': metadata_info
        })
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)