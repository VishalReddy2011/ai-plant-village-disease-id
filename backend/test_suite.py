import os
import sys
import logging
import traceback
import uuid
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add current folder to path to allow importing modules correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_comprehensive_test_suite() -> bool:
    print("=" * 60)
    print("      PLANT DISEASE IDENTIFIER BACKEND INTEGRATION SUITE      ")
    print("=" * 60)
    
    # 1. Verify Imports
    logger.info("Step 1: Verifying backend custom module imports...")
    try:
        from vision.classifier import DiseaseClassifier
        from vision.gradcam import GradCAM
        from vision.validator import validate_leaf
        from rag.ingest import run_ingestion
        from rag.advisor import TreatmentAdvisor
        from stt_tts.stt_service import SpeechToTextService
        from stt_tts.tts_service import TextToSpeechService
        logger.info("SUCCESS: All custom modules imported successfully.")
    except Exception as e:
        logger.error(f"FAILURE: Module imports failed: {e}")
        traceback.print_exc()
        return False

    # 2. Verify Dotenv Loading
    logger.info("Step 2: Loading environment configurations...")
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        model_name = os.getenv("LLM_MODEL", "gpt-5.4-nano")
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store")
        
        logger.info("SUCCESS: .env configuration variables loaded.")
        logger.info(f" -> LLM_MODEL: {model_name}")
        logger.info(f" -> Vector DB Persistence: {persist_dir}")
        logger.info(f" -> API Key Present: {'Yes' if api_key else 'No'}")
        
        # Auto-heal base URL based on API key format to ensure smooth user experience
        if api_key and api_key.startswith("sk-or-"):
            if not api_base:
                logger.warning("Auto-Correction: Detected OpenRouter key (sk-or-...) but no base URL. Configuring endpoint to OpenRouter API to prevent 401.")
                api_base = "https://openrouter.ai/api/v1"
            # Clean model name prefix if it's OpenRouter
            if not model_name.startswith("openai/"):
                logger.warning(f"Auto-Correction: Adding openrouter model namespace path. Model updated: 'openai/{model_name}'")
                model_name = f"openai/{model_name}"
        else:
            # Clean prefix if using OpenAI directly
            if model_name.startswith("openai/"):
                model_name = model_name.replace("openai/", "")
                logger.info(f"Model cleaned for OpenAI: {model_name}")
                
    except Exception as e:
        logger.error(f"FAILURE: Dotenv loading failed: {e}")
        return False

    # 3. Test database CRUD functionality
    logger.info("Step 3: Verifying SQLite database schemas and CRUD operations...")
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from main import DiagnosisRecord, FeedbackRecord, Base
        
        db_path = "./test_suite_db.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Test creation of a diagnosis record
        diag_id = str(uuid.uuid4())
        diag_record = DiagnosisRecord(
            id=diag_id,
            disease_name="Tomato Early Blight",
            confidence=0.95,
            language="en"
        )
        session.add(diag_record)
        
        # Test feedback record creation
        feedback_record = FeedbackRecord(
            id=str(uuid.uuid4()),
            diagnosis_id=diag_id,
            helpful="yes",
            notes="Excellent prediction accuracy"
        )
        session.add(feedback_record)
        session.commit()
        
        # Test reading the records
        retrieved_diag = session.query(DiagnosisRecord).filter_by(id=diag_id).first()
        retrieved_feedback = session.query(FeedbackRecord).filter_by(diagnosis_id=diag_id).first()
        
        assert retrieved_diag is not None, "Failed to retrieve diagnosis record from DB."
        assert retrieved_diag.disease_name == "Tomato Early Blight", "Data fields are corrupted."
        assert retrieved_feedback is not None, "Failed to retrieve feedback record from DB."
        
        # Cleanup
        session.close()
        engine.dispose()
        if os.path.exists(db_path):
            os.remove(db_path)
            
        logger.info("SUCCESS: Database models created and CRUD operations verified successfully.")
    except Exception as e:
        logger.error(f"FAILURE: Database CRUD validation failed: {e}")
        return False

    # 4. Local Vision Model Validation (if weights file is present)
    logger.info("Step 4: Checking Vision Classifier and Grad-CAM layers...")
    weights_path = "./models/plant_disease.pt"
    if os.path.exists(weights_path):
        try:
            import torch
            classifier = DiseaseClassifier(weights_path)
            gradcam = GradCAM(classifier)
            logger.info("SUCCESS: Loaded PyTorch model and initialized Grad-CAM successfully.")
            
            # Run inference verification with dummy tensor
            dummy_tensor = torch.zeros(1, 3, 224, 224).to(classifier.device)
            with torch.no_grad():
                logits = classifier.model(dummy_tensor)
            logger.info(f"SUCCESS: Vision classification runs on dummy tensor. Output logits shape: {logits.shape}")
        except Exception as e:
            logger.error(f"FAILURE: Vision model load/inference check failed: {e}")
            return False
    else:
        logger.warning("VISION MODEL NOT DETECTED: Skipping PyTorch weights check. Server will launch in mock mode.")

    # 5. Live LLM Reachability & Schema Validation
    if not api_key:
        logger.warning("OPENAI_API_KEY NOT FOUND: Skipping live LLM and RAG index validations.")
        print("=" * 60)
        print("RESULT: SUCCESS (Partially verified - Offline components passed)")
        print("=" * 60)
        return True
        
    logger.info("Step 5: Verifying live LLM connectivity and token constraints...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from pydantic import BaseModel, Field
        
        # Simple Schema Check
        class ConnectionSchema(BaseModel):
            success: bool = Field(description="Set to true if request is processed successfully")
            
        # Initialize client using resolved base configuration
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=1024
        )
        
        # 5a. Check raw text connection
        test_msg = "Ping"
        logger.info(f"Sending prompt to '{model_name}' (Raw completion test)...")
        raw_res = llm.invoke([HumanMessage(content=test_msg)])
        logger.info(f"Raw connection returned response content: '{raw_res.content.strip()}'")
        
        # 5b. Check structured output (Test schema)
        logger.info("Verifying structured output parser and model parsing schema...")
        structured_llm = llm.with_structured_output(ConnectionSchema)
        structured_res = structured_llm.invoke([HumanMessage(content="Return success as true")])
        assert structured_res.success is True, "Model returned unexpected structured values."
        
        logger.info("SUCCESS: LLM connectivity, token limits, and schema serialization verified.")
    except Exception as e:
        logger.error(f"FAILURE: LLM validation checks failed: {e}")
        traceback.print_exc()
        return False

    # 6. Live RAG Vector Search & Ingestion Validation
    logger.info("Step 6: Verifying local RAG vector store pipeline...")
    try:
        import chromadb
        from langchain_openai import OpenAIEmbeddings
        from rag.advisor import TreatmentAdvisor
        from rag.advisor import TreatmentPlan
        
        # Ephemeral index check
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(name="test_suite_collection")
        
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=api_base,
            model="text-embedding-3-small"
        )
        
        sample_doc = "Early Blight on Tomatoes is caused by the fungus Alternaria solani. Spray copper fungicide."
        logger.info("Generating embedding vectors using OpenAI Embeddings...")
        vector = embeddings.embed_query(sample_doc)
        
        collection.add(
            embeddings=[vector],
            documents=[sample_doc],
            metadatas=[{"crop": "tomato", "disease": "early blight"}],
            ids=["doc_test_99"]
        )
        
        # Query check
        query_vec = embeddings.embed_query("tomato early blight treatment")
        results = collection.query(query_embeddings=[query_vec], n_results=1)
        assert len(results["documents"][0]) > 0, "No documents returned from vector database check."
        assert "Alternaria solani" in results["documents"][0][0], "Vector search retrieved incorrect context."
        
        logger.info("SUCCESS: Embedding generation and ephemeral RAG lookup working correctly.")
    except Exception as e:
        logger.error(f"FAILURE: Vector database check failed: {e}")
        traceback.print_exc()
        return False

    # 7. Complete TreatmentAdvisor integration test
    logger.info("Step 7: Verifying TreatmentAdvisor full schema RAG advisory output...")
    try:
        from rag.advisor import TreatmentAdvisor
        
        # In-memory mock workspace for advisor checks
        advisor_inst = TreatmentAdvisor(persist_dir, api_key, model_name=model_name)
        if api_base:
            advisor_inst.embeddings.base_url = api_base
            advisor_inst.llm.base_url = api_base
            
        local_conditions = {
            "growth_stage": "seedling",
            "severity": "few leaves",
            "irrigation": "drip",
            "fungicide_access": "organic only",
            "weather": "warm & humid"
        }
        
        logger.info("Invoking advisor.get_treatment_plan()...")
        advisory_result = advisor_inst.get_treatment_plan(
            crop="tomato",
            disease_name="early blight",
            local_conditions=local_conditions,
            language="en"
        )
        
        print("\n" + "-"*40)
        print("TREATMENT ADVISORY PLAN OUTPUT RECEIVED:")
        print(f"Generated by LLM: {advisory_result.get('generated_by_llm')}")
        print(f"Immediate Actions: {advisory_result.get('immediate_actions')[:2]}...")
        print("-"*40 + "\n")
        
        assert advisory_result.get("generated_by_llm") is True, "Treatment plan execution fell back to static mock data."
        logger.info("SUCCESS: Full RAG TreatmentAdvisor system offline/online checks passed.")
    except Exception as e:
        logger.error(f"FAILURE: Full advisory flow check failed: {e}")
        traceback.print_exc()
        return False

    # 8. Verify STT & TTS services
    logger.info("Step 8: Verifying STT and TTS services...")
    try:
        stt_service = SpeechToTextService(api_key)
        tts_service = TextToSpeechService(api_key)
        
        # Test TTS synthesis with a small test string
        test_text = "Hello farmer, this is a test audio report."
        logger.info("Testing OpenAI TTS synthesis...")
        audio_data = tts_service.synthesize_speech(test_text, "en")
        assert len(audio_data) > 0, "TTS synthesis returned empty bytes."
        logger.info(f"SUCCESS: Synthesized {len(audio_data)} bytes of test speech.")
        
        # Test STT transcription interface with a mock tiny wav header
        logger.info("Testing OpenAI STT transcription structure...")
        mock_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x11\x2b\x00\x00\x22\x56\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        res_text = stt_service.transcribe_audio(mock_audio, "en")
        logger.info(f"STT structure test returned response: {res_text}")
        logger.info("SUCCESS: STT & TTS service interfaces verified successfully.")
    except Exception as e:
        logger.error(f"FAILURE: STT/TTS service validation failed: {e}")
        traceback.print_exc()
        return False

    print("=" * 60)
    print("RESULT: ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)
