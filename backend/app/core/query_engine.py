"""
Vision Query Engine for multimodal RAG.
"""
import base64
import logging
from typing import Optional, List, Dict, Any
from io import BytesIO
import requests
from PIL import Image

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

logger = logging.getLogger(__name__)

# Default prompt template
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


class VisionQueryEngine(CustomQueryEngine):
    """Query engine with image proxy support for private blob storage."""
    
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: AzureOpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs):
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def fetch_and_convert_image(self, image_url: str) -> Optional[str]:
        """Fetch image from private blob storage and convert to base64."""
        try:
            # Use requests to fetch the image
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
        """Process query with multimodal context."""
        nodes = self.retriever.retrieve(query_str)
        
        # Build image nodes with base64 data
        image_nodes = []
        failed_images = []
        
        for n in nodes:
            blob_path = n.metadata.get("image_path")
            if blob_path:
                try:
                    # Build full URL (this should be handled by configuration)
                    from flask import current_app
                    from app.utils.image_utils import build_image_url
                    
                    full_url = build_image_url(
                        blob_path,
                        current_app.config['AZURE_STORAGE_ACCOUNT_NAME'],
                        current_app.config['AZURE_STORAGE_SAS_TOKEN'],
                        current_app.config['AZURE_BLOB_CONTAINER_NAME']
                    )
                    
                    # Fetch and convert to base64
                    base64_data = self.fetch_and_convert_image(full_url)
                    
                    if base64_data:
                        # Create image node with base64 data
                        img_node = ImageNode()
                        img_node.image = base64_data
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
                        image_documents=[],
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
                    "images": [n.metadata.get("image_path") for n in nodes if n.metadata.get("image_path")]
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def stream_query(self, query_str: str):
        """Stream the response with image proxy support."""
        nodes = self.retriever.retrieve(query_str)
        
        # Build image nodes with base64 data
        image_nodes = []
        public_image_urls = []
        
        for n in nodes:
            blob_path = n.metadata.get("image_path")
            if blob_path:
                try:
                    from flask import current_app
                    from app.utils.image_utils import build_image_url
                    
                    full_url = build_image_url(
                        blob_path,
                        current_app.config['AZURE_STORAGE_ACCOUNT_NAME'],
                        current_app.config['AZURE_STORAGE_SAS_TOKEN'],
                        current_app.config['AZURE_BLOB_CONTAINER_NAME']
                    )
                    public_image_urls.append(full_url)
                    
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
        
        # Build source previews
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
                'imageUrl': image_url
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
            'images': proxy_image_urls,
            'sourcePreviews': source_previews
        }