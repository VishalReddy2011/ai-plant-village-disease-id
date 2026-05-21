import os
import re
import shutil
import logging
import chromadb
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def parse_md_file(file_path: str) -> tuple[dict, str]:
    """Parses simple frontmatter and body from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    metadata = {}
    body = content
    
    # Split frontmatter if it exists
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            front_text = parts[1]
            body = parts[2]
            for line in front_text.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip('"').strip("'")
                    
    # Clean crop name and disease name for filtering
    if "crop" in metadata:
        metadata["crop"] = metadata["crop"].lower()
    if "disease_name" in metadata:
        metadata["disease_name"] = metadata["disease_name"].lower()
        
    return metadata, body.strip()

def run_ingestion(raw_docs_dir: str, persist_dir: str, api_key: str) -> bool:
    """Ingests all raw markdown documents into the local Chroma vector store using native chromadb."""
    if not os.path.exists(raw_docs_dir):
        logger.warning(f"Raw documents directory '{raw_docs_dir}' does not exist.")
        return False
        
    logger.info("Starting ingestion of RAG documentation...")
    
    # Read files
    texts = []
    metadatas = []
    ids = []
    
    idx = 0
    for filename in os.listdir(raw_docs_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(raw_docs_dir, filename)
            meta, body = parse_md_file(file_path)
            meta["source"] = filename
            
            texts.append(body)
            metadatas.append(meta)
            ids.append(f"doc_{idx}")
            idx += 1
            
    if not texts:
        logger.warning("No markdown documents found to ingest.")
        return False
        
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    logger.info(f"Generating embeddings for {len(texts)} documents using OpenAI Embeddings...")
    vectors = embeddings.embed_documents(texts)
    
    # Recreate the persist dir to ensure fresh state
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            logger.warning(f"Could not clear vector store dir: {e}")
            
    # Load into Chroma vector store using native client
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="plant_handbook")
    
    collection.add(
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"Successfully ingested {len(texts)} documents into local Chroma database collection 'plant_handbook'.")
    return True

if __name__ == "__main__":
    # Allow running directly as a script
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    raw_dir = "./rag/raw_docs"
    db_dir = "./data/vector_store"
    
    if api_key:
        run_ingestion(raw_dir, db_dir, api_key)
    else:
        print("Error: OPENAI_API_KEY not found in environment.")
