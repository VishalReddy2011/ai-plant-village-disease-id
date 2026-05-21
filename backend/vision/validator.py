import base64
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LeafValidationResult(BaseModel):
    is_leaf: bool = Field(description="True if the image displays a plant leaf or foliage, False otherwise")

def validate_leaf(image_bytes: bytes, api_key: str, model_name: str) -> bool:
    """Validates whether the uploaded image contains a leaf using ChatOpenAI."""
    if not api_key:
        return True  # Fallback if no API key is provided
        
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Configure the LangChain OpenAI client
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.0,
            max_tokens=512
        )
        
        # Attempt structured JSON validation
        try:
            structured_llm = llm.with_structured_output(LeafValidationResult)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Verify if this image contains a plant leaf. Respond with is_leaf as true or false."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            result = structured_llm.invoke([message])
            return result.is_leaf
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to text parsing: {e}")
            # Text fallback
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Does this image show a plant leaf? Answer with exactly 'true' or 'false'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            response = llm.invoke([message])
            text = response.content.lower().strip()
            return "true" in text or "yes" in text
            
    except Exception as e:
        logger.error(f"Leaf validation exception occurred: {e}")
        return True  # Fail open to allow proceeding if LLM is unreachable
