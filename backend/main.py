import os
import io
import uuid
import logging
import threading
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Database imports
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Custom modules
from vision.classifier import DiseaseClassifier
from vision.gradcam import GradCAM
from vision.validator import validate_leaf
from rag.ingest import run_ingestion
from rag.advisor import TreatmentAdvisor
from stt_tts.stt_service import SpeechToTextService
from stt_tts.tts_service import TextToSpeechService

# Setup logger and configurations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config from .env
load_dotenv()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-nano")
VISION_MODEL_PATH = os.getenv("VISION_MODEL_PATH", "./models/plant_disease.pt")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./plant_disease.db")

# Initialize SQLite database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DiagnosisRecord(Base):
    """Database record for logging plant disease diagnoses."""
    __tablename__ = "diagnoses"
    id = Column(String, primary_key=True, index=True)
    disease_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    language = Column(String, default="en")
    timestamp = Column(DateTime, default=datetime.utcnow)

class FeedbackRecord(Base):
    """Database record for farmer feedback (RLHF)."""
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, index=True)
    diagnosis_id = Column(String, nullable=False)
    helpful = Column(String, nullable=True)
    notes = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# Setup FastAPI App
app = FastAPI(
    title="Plant Disease Identification & Treatment Advisor API",
    version="1.0.0"
)

# Enable CORS for local frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons initialized on startup
classifier = None
gradcam = None
advisor = None
stt_service = None
tts_service = None

# Threading lock for model loading
init_lock = threading.Lock()

def download_model_from_drive(file_id: str, dest_path: str):
    import urllib.request
    import urllib.parse
    import re
    import http.cookiejar

    logger.info(f"Downloading model from Google Drive ID: {file_id} to {dest_path}...")
    
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    try:
        # First request to get warning page
        with opener.open(url) as response:
            content = response.read()
            
        # Parse confirmation elements if warning page is returned
        if b"confirm=" in content or b"download-form" in content:
            html = content.decode('utf-8', errors='ignore')
            
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
            
            confirm_val = confirm_match.group(1) if confirm_match else "t"
            uuid_val = uuid_match.group(1) if uuid_match else ""
            
            params = {
                "id": file_id,
                "export": "download",
                "confirm": confirm_val
            }
            if uuid_val:
                params["uuid"] = uuid_val
                
            confirm_url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(params)
            logger.info("Large file warning bypassed. Redirecting to confirmation URL...")
            with opener.open(confirm_url) as response:
                content = response.read()
                
        # Save file
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Successfully downloaded model to {dest_path} ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to download model from Google Drive: {e}")
        return False

def ensure_classifier_initialized():
    global classifier, gradcam
    
    if classifier is not None and gradcam is not None:
        return True
        
    with init_lock:
        # Double check after obtaining lock
        if classifier is not None and gradcam is not None:
            return True
            
        if os.path.exists(VISION_MODEL_PATH):
            file_size = os.path.getsize(VISION_MODEL_PATH)
            if file_size < 300 * 1024 * 1024:
                logger.info(f"Vision model file exists but size ({file_size} bytes) is less than 300MB. Deleting and redownloading...")
                try:
                    os.remove(VISION_MODEL_PATH)
                except Exception as e:
                    logger.error(f"Failed to remove small/corrupted model file: {e}")
            
        if not os.path.exists(VISION_MODEL_PATH):
            logger.info(f"Vision model file not found at '{VISION_MODEL_PATH}'. Attempting download...")
            success = download_model_from_drive("1tXkTBoWx8ykzHCsFyf27MIwtTaEioBqi", VISION_MODEL_PATH)
            if not success:
                logger.error("Failed to download model from Google Drive.")
                return False
                
        try:
            logger.info(f"Loading vision model from {VISION_MODEL_PATH}...")
            classifier = DiseaseClassifier(VISION_MODEL_PATH)
            gradcam = GradCAM(classifier)
            logger.info("Vision model and Grad-CAM layers successfully initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            return False

@app.on_event("startup")
async def startup_event():
    """Starts all model resources and does automatic RAG document index validation."""
    global classifier, gradcam, advisor, stt_service, tts_service
    
    logger.info("Initializing backend services...")
    
    # 1. Init Vision Pipeline (will download if missing)
    ensure_classifier_initialized()

    # 2. Init RAG advisor & Auto Ingestion if store is empty
    if OPENAI_API_KEY:
        try:
            # Trigger automatic ingestion if the database directory doesn't exist
            if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
                logger.info("Local ChromaDB store not found or empty. Performing automatic ingestion of handbook articles...")
                run_ingestion("./rag/raw_docs", CHROMA_PERSIST_DIR, OPENAI_API_KEY)
                
            advisor = TreatmentAdvisor(CHROMA_PERSIST_DIR, OPENAI_API_KEY, model_name=LLM_MODEL)
            logger.info("RAG Advisor system online.")
        except Exception as e:
            logger.error(f"Failed to start RAG advisor system: {e}")
    else:
        logger.warning("OPENAI_API_KEY missing from environment. RAG systems will run with static fallbacks.")

    # 3. Init Audio Skeletons
    stt_service = SpeechToTextService(OPENAI_API_KEY)
    tts_service = TextToSpeechService(OPENAI_API_KEY)

# --- Pydantic Schemas ---
class TreatmentRequest(BaseModel):
    crop: str
    disease_name: str
    language: str = "en"
    growth_stage: str = "seedling"
    severity: str = "few leaves"
    irrigation: str = "drip"
    fungicide_access: str = "organic only"
    weather: str = "warm & humid"

class FeedbackRequest(BaseModel):
    diagnosis_id: str
    helpful: str
    notes: str = ""

# --- REST Endpoints ---

@app.get("/api/v1/health")
def health_check():
    """Simple status check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/test")
def run_tests_post():
    """Runs the comprehensive integration test suite and returns stdout and success status."""
    return execute_test_suite()

@app.get("/api/v1/test")
def run_tests_get():
    """Runs the comprehensive integration test suite and returns stdout and success status."""
    return execute_test_suite()

def execute_test_suite():
    import io
    import contextlib
    import traceback
    from test_suite import run_comprehensive_test_suite
    
    output_capture = io.StringIO()
    success = False
    try:
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            success = run_comprehensive_test_suite()
    except Exception as e:
        success = False
        traceback.print_exc(file=output_capture)
        
    return {
        "success": success,
        "logs": output_capture.getvalue()
    }

@app.post("/api/v1/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """Processes leaf image uploads: runs validation -> prediction -> Grad-CAM overlay."""
    image_bytes = await file.read()
    diagnosis_id = str(uuid.uuid4())
    
    # 1. Leaf check using LLM
    is_leaf = True
    if OPENAI_API_KEY:
        is_leaf = validate_leaf(image_bytes, OPENAI_API_KEY, LLM_MODEL)
        if not is_leaf:
            return {
                "diagnosis_id": diagnosis_id,
                "is_leaf": False,
                "error_message": "This image does not appear to contain a plant leaf. Please upload a clear photo of an infected leaf."
            }

    # 2. Vision prediction (top 3)
    if ensure_classifier_initialized():
        try:
            predictions = classifier.predict_top_3(image_bytes)
            top_prediction = predictions[0]
            
            # Save top diagnosis details in SQLite
            db = SessionLocal()
            record = DiagnosisRecord(
                id=diagnosis_id,
                disease_name=top_prediction["disease_name_en"],
                confidence=top_prediction["confidence"],
                language=language
            )
            db.add(record)
            db.commit()
            db.close()
            
            # 3. Compute Grad-CAM for the top predicted class
            # Map top prediction raw label to class index
            top_class_idx = classifier.class_names.index(top_prediction["disease_raw"])
            overlay_b64, explainable_regions = gradcam.generate_heatmap_overlay(image_bytes, top_class_idx)
            
            return {
                "diagnosis_id": diagnosis_id,
                "is_leaf": True,
                "top_predictions": predictions,
                "gradcam_overlay": f"data:image/jpeg;base64,{overlay_b64}",
                "explainable_regions": explainable_regions,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Vision inference execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Vision model classification failed: {str(e)}")
    else:
        logger.warning("Classifier initialization failed (model download or loading failed). Falling back to mock vision response.")
        # Mock vision response if no Pt model is present
        mock_preds = [
            {"disease_name_en": "Tomato Early Blight", "confidence": 0.94, "disease_raw": "Tomato___Early_blight"},
            {"disease_name_en": "Tomato Late Blight", "confidence": 0.04, "disease_raw": "Tomato___Late_blight"},
            {"disease_name_en": "Tomato healthy", "confidence": 0.02, "disease_raw": "Tomato___healthy"}
        ]
        return {
            "diagnosis_id": diagnosis_id,
            "is_leaf": True,
            "top_predictions": mock_preds,
            "gradcam_overlay": "",
            "explainable_regions": {
                "primary_region": "Leaf margin",
                "feature_detected": "Concentrated necrotic spots",
                "activation_level": "High"
            },
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/v1/treatment")
def get_treatment(req: TreatmentRequest):
    """Generates RAG-grounded treatment plans tailored to localized conditions."""
    local_conditions = {
        "growth_stage": req.growth_stage,
        "severity": req.severity,
        "irrigation": req.irrigation,
        "fungicide_access": req.fungicide_access,
        "weather": req.weather
    }
    
    if advisor is not None:
        try:
            plan = advisor.get_treatment_plan(
                crop=req.crop,
                disease_name=req.disease_name,
                local_conditions=local_conditions,
                language=req.language
            )
            return plan
        except Exception as e:
            logger.error(f"Advisory generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Fallback to hardcoded mock data if advisor is offline
        from rag.advisor import MOCK_TREATMENT_PLANS
        fallback = MOCK_TREATMENT_PLANS.get(req.language, MOCK_TREATMENT_PLANS["en"])
        return {
            **fallback,
            "generated_by_llm": False
        }

@app.post("/api/v1/voice/stt")
async def voice_speech_to_text(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """Transcription endpoint (stub placeholder for speech-to-text integration)."""
    audio_bytes = await file.read()
    if stt_service is not None:
        transcription = stt_service.transcribe_audio(audio_bytes, language)
        return {"text": transcription, "language": language}
    return {"text": "[STT Offline]", "language": language}

@app.post("/api/v1/voice/tts")
def voice_text_to_speech(
    text: str = Form(...),
    language: str = Form("en")
):
    """Synthesis endpoint (stub placeholder for text-to-speech integration)."""
    if tts_service is not None:
        audio_data = tts_service.synthesize_speech(text, language)
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mp3")
    return Response(content=b"", media_type="audio/mp3")

@app.post("/api/v1/feedback")
def submit_feedback(req: FeedbackRequest):
    """Records feedback regarding diagnosis accuracy."""
    db = SessionLocal()
    feedback_id = str(uuid.uuid4())
    record = FeedbackRecord(
        id=feedback_id,
        diagnosis_id=req.diagnosis_id,
        helpful=req.helpful,
        notes=req.notes
    )
    db.add(record)
    db.commit()
    db.close()
    return {"status": "success", "feedback_id": feedback_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
