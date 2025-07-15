import os
import json
import time
import logging
import re
import fitz  # PyMuPDF
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, List
from dotenv import load_dotenv
import nest_asyncio
import io
from PIL import Image

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.formrecognizer import DocumentAnalysisClient

# LlamaIndex imports
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType
)

# Async imports
import asyncio
from azure.storage.blob.aio import BlobServiceClient
import aiofiles

nest_asyncio.apply()
load_dotenv()

# ================== Configuration ==================
# Set Azure OpenAI environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"  # Updated to stable version

# Environment Variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "rag-demo-images"
INDEX_NAME = "azure-multimodal-search-new"
DOWNLOAD_PATH = "pdf-files"

# Optimization settings
IMAGE_DPI = 100  # Reduced from 150 for faster processing
IMAGE_FORMAT = "JPEG"  # Changed from PNG for smaller files
IMAGE_QUALITY = 85  # JPEG quality (1-100)
MAX_CONCURRENT_UPLOADS = 15  # Increased from 5

# Initialize Azure OpenAI settings first
Settings.llm = AzureOpenAI(
    engine=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",  # Match API version
    api_type="azure"
)

Settings.embed_model = AzureOpenAIEmbedding(
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,  # Correct parameter
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15",  # Correct version for embeddings
    api_type="azure"
)

# Initialize Azure Clients
document_analysis_client = DocumentAnalysisClient(
    endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
)

# Initialize Search Clients
search_credential = AzureKeyCredential(SEARCH_SERVICE_API_KEY)
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=search_credential)
search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=search_credential)

# Define metadata fields for the index
metadata_fields = {
    "page_num": ("page_num", MetadataIndexFieldType.INT64),
    "doc_id": ("doc_id", MetadataIndexFieldType.STRING),
    "image_path": ("image_path", MetadataIndexFieldType.STRING),
    "full_text": ("full_text", MetadataIndexFieldType.STRING),
}

# ================== Document Processing ==================
def create_folder_structure(base_path: str, pdf_path: str) -> str:
    """Create organized folder structure for output files."""
    folder_name = Path(pdf_path).stem
    folder_path = Path(base_path) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return str(folder_path)

def pdf_to_images_optimized(pdf_path: str, output_base: str) -> List[dict]:
    """Convert PDF to optimized images with lower DPI and JPEG format."""
    image_dicts = []
    folder_path = create_folder_structure(output_base, pdf_path)
    
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise ValueError("Encrypted PDF - password required")

        total_pages = len(doc)
        logging.info(f"Converting {total_pages} pages to images...")
        
        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)
                # Lower DPI for smaller files
                pix = page.get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csRGB, alpha=False)
                
                # Save as JPEG instead of PNG
                image_name = f"page_{page_num+1}.jpg"
                image_path = str(Path(folder_path) / image_name)
                
                # Convert to PIL Image for JPEG saving with quality control
                img_data = pix.pil_tobytes(format="JPEG", optimize=True)
                img = Image.open(io.BytesIO(img_data))
                img.save(image_path, "JPEG", quality=IMAGE_QUALITY, optimize=True)
                
                image_dicts.append({
                    "name": image_name,
                    "path": image_path,
                    "page_num": page_num + 1
                })
                
                if (page_num + 1) % 10 == 0:
                    logging.info(f"Converted {page_num + 1}/{total_pages} pages")
                
            except Exception as e:
                logging.error(f"Page {page_num+1} processing failed: {str(e)}", exc_info=True)
                continue
                
        logging.info(f"Successfully converted {len(image_dicts)} pages to images")
        return image_dicts

    except Exception as e:
        logging.error(f"PDF processing failed: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def extract_document_data(pdf_path: str) -> dict:
    """Extract text and images from PDF."""
    start_time = time.time()
    
    # Extract text using Document Intelligence
    with open(pdf_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()
    
    text_extraction_time = time.time() - start_time
    logging.info(f"Text extraction took {text_extraction_time:.2f}s")
    
    # Convert to optimized images
    image_start = time.time()
    image_dicts = pdf_to_images_optimized(pdf_path, DOWNLOAD_PATH)
    image_conversion_time = time.time() - image_start
    logging.info(f"Image conversion took {image_conversion_time:.2f}s")

    return {
        "text_content": result.content,
        "pages": [{"text": "\n".join(line.content for line in page.lines)} for page in result.pages],
        "images": image_dicts,
        "source_path": pdf_path
    }

# ================== Optimized Azure Blob Storage ==================
class OptimizedBlobUploader:
    """Optimized blob uploader with connection reuse and better concurrency."""
    
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = None
        
    async def __aenter__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        # Ensure container exists
        container_client = self.blob_service_client.get_container_client(self.container_name)
        if not await container_client.exists():
            await container_client.create_container()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.blob_service_client:
            await self.blob_service_client.close()
    
    async def upload_images_batch(self, image_dicts: List[dict]) -> Dict[str, str]:
        """Upload images in batch with high concurrency."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
        tasks = []
        
        logging.info(f"Starting upload of {len(image_dicts)} images...")
        
        for img in image_dicts:
            task = self._upload_single_optimized(img, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        uploaded_urls = {}
        failed_count = 0
        
        for img, result in zip(image_dicts, results):
            if isinstance(result, Exception):
                logging.error(f"Upload failed for {img['name']}: {str(result)}")
                failed_count += 1
            elif result:
                uploaded_urls[img["name"]] = result
        
        success_count = len(uploaded_urls)
        logging.info(f"Upload complete: {success_count} succeeded, {failed_count} failed")
        
        return uploaded_urls
    
    async def _upload_single_optimized(self, image: dict, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Upload single image with optimization."""
        async with semaphore:
            try:
                blob_name = image["name"]
                
                # Read file asynchronously
                async with aiofiles.open(image["path"], "rb") as f:
                    image_data = await f.read()
                
                # Get blob client
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                # Upload with optimized settings
                await blob_client.upload_blob(
                    image_data, 
                    overwrite=True,
                    max_concurrency=4,  # Parallel chunk upload
                    length=len(image_data)
                )
                
                return blob_client.url
                
            except Exception as e:
                logging.error(f"Upload failed for {image['name']}: {str(e)}")
                return None

async def upload_images_concurrently(image_dicts: List[dict]) -> Dict[str, str]:
    """Upload images using optimized uploader."""
    async with OptimizedBlobUploader(BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME) as uploader:
        return await uploader.upload_images_batch(image_dicts)

# ================== Search Index Integration ==================
def create_search_nodes(document_data: dict, image_urls: Dict[str, str]) -> List[TextNode]:
    """Create search nodes with linked text and images."""
    nodes = []
    page_image_map = {
        int(re.search(r"page_(\d+)", name).group(1)): url
        for name, url in image_urls.items()
    }

    for page_num, page_text in enumerate(document_data["pages"], start=1):
        image_url = page_image_map.get(page_num)
        if not image_url:
            logging.warning(f"No image found for page {page_num}")
            continue

        node = TextNode(
            text=page_text["text"],
            metadata={
                "page_num": page_num,
                "image_path": image_url,
                "doc_id": Path(document_data["source_path"]).stem,
                "full_text": page_text["text"]
            }
        )
        nodes.append(node)
    
    logging.info(f"Created {len(nodes)} search nodes")
    return nodes

def create_vector_store(
    index_client,  # now using the SearchIndexClient
    use_existing_index: bool = False
) -> AzureAISearchVectorStore:
    """Create or get existing Azure AI Search vector store."""
    # IMPORTANT: Must specify index_name if passing SearchIndexClient
    return AzureAISearchVectorStore(
        search_or_index_client=index_client,
        index_name=INDEX_NAME,  # <-- The fix
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="full_text",
        embedding_field_key="embedding",
        embedding_dimensionality=3072,
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        filterable_metadata_field_keys=metadata_fields,
        language_analyzer="en.lucene",
        vector_algorithm_type="exhaustiveKnn",
    )

def create_or_load_index(
    text_nodes,
    index_client,  # now expect the index_client
    embed_model,
    llm,
    use_existing_index: bool = False
) -> VectorStoreIndex:
    """Create new index or load existing one."""
    vector_store = create_vector_store(index_client, use_existing_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if use_existing_index:
        return VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
        )
    else:
        return VectorStoreIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
            show_progress=True,
        )

# ================== Main Processing Pipeline ==================
async def process_document(pdf_path: str) -> RetrieverQueryEngine:
    """End-to-end document processing pipeline with performance monitoring."""
    try:
        total_start_time = time.time()
        
        # Extract document data
        logging.info(f"Processing document: {pdf_path}")
        document_data = extract_document_data(pdf_path)
        
        # Upload images with optimized settings
        upload_start = time.time()
        image_urls = await upload_images_concurrently(document_data["images"])
        upload_time = time.time() - upload_start
        logging.info(f"Uploaded {len(image_urls)} images in {upload_time:.2f}s")
        
        # Calculate average upload speed
        if image_urls:
            avg_upload_time = upload_time / len(image_urls)
            logging.info(f"Average upload time per image: {avg_upload_time:.2f}s")
        
        # Create search nodes
        nodes = create_search_nodes(document_data, image_urls)
        
        # Create index
        index_start = time.time()
        index = create_or_load_index(
            text_nodes=nodes,
            index_client=index_client,
            embed_model=Settings.embed_model,
            llm=Settings.llm,
            use_existing_index=False
        )
        index_time = time.time() - index_start
        logging.info(f"Index creation took {index_time:.2f}s")

        # Create query engine
        response_synthesizer = get_response_synthesizer(
            llm=Settings.llm,
            response_mode="compact"
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            response_synthesizer=response_synthesizer
        )
        
        # Log total processing time
        total_time = time.time() - total_start_time
        logging.info(f"Total processing time: {total_time:.2f}s")
        
        # Log performance summary
        logging.info("Performance Summary:")
        logging.info(f"  - Document extraction: {document_data.get('extraction_time', 0):.2f}s")
        logging.info(f"  - Image upload: {upload_time:.2f}s")
        logging.info(f"  - Index creation: {index_time:.2f}s")
        logging.info(f"  - Total time: {total_time:.2f}s")
        
        return query_engine

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    # Setup logging with more detailed format
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    pdf_path = "data/pdfs/new-relic-2024-observability-forecast-report.pdf"
    
    # Log optimization settings
    logging.info("=== Starting PDF Processing with Optimized Settings ===")
    logging.info(f"Image DPI: {IMAGE_DPI}")
    logging.info(f"Image Format: {IMAGE_FORMAT}")
    logging.info(f"Image Quality: {IMAGE_QUALITY}")
    logging.info(f"Max Concurrent Uploads: {MAX_CONCURRENT_UPLOADS}")
    logging.info("=" * 50)
    
    try:
        query_engine = asyncio.run(process_document(pdf_path))
        logging.info("Processing completed successfully")
        
        # You can now use the query engine
        # Example: response = query_engine.query("What is this document about?")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)