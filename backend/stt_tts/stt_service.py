from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
import base64
import io
import logging

logger = logging.getLogger(__name__)

class SpeechToTextService:
    """Transcribes farmer audio messages using OpenAI models (gpt-4o-mini-transcribe/whisper-1)."""
    def __init__(self, api_key: str = None):
        if not api_key:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        # Initialize ChatOpenAI with the specific transcription model
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini-transcribe",
            temperature=0.0
        ) if api_key else None

    def transcribe_audio(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe uploaded audio bytes in the given language.
        
        Args:
            audio_bytes: Raw audio binary data (e.g. WAV, MP3, M4A).
            language: The target spoken language (e.g. 'en', 'hi', 'mr').
            
        Returns:
            str: Transcribed natural language text.
        """
        if not self.llm:
            return f"[STT Offline - No API Key: Spoken text query in {language}]"
            
        # Try audio modalities interface using ChatOpenAI first
        try:
            logger.info("Attempting transcription using ChatOpenAI multimodal input...")
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            messages = [
                SystemMessage(content=f"You are a precise transcription system. Transcribe the audio in target language: {language}. Return ONLY transcription."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Transcribe this audio."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        }
                    ]
                )
            ]
            response = self.llm.invoke(messages)
            transcription = response.content.strip()
            if transcription:
                logger.info("Successfully transcribed audio via ChatOpenAI multimodal input.")
                return transcription
        except Exception as e:
            logger.warning(f"ChatOpenAI multimodal transcription failed: {e}. Trying direct API translation...")
            
        # Fallback 1: Direct OpenAI client transcription using "gpt-4o-mini-transcribe"
        try:
            client = OpenAI(api_key=self.api_key)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            # Check for standard language ISO code conversions
            lang_code = language.split("-")[0].lower() if language else "en"
            
            logger.info("Attempting transcription using direct client with model 'gpt-4o-mini-transcribe'...")
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language=lang_code
            )
            transcription = transcript.text.strip()
            if transcription:
                logger.info("Successfully transcribed audio via direct client 'gpt-4o-mini-transcribe'.")
                return transcription
        except Exception as e:
            logger.warning(f"Direct client 'gpt-4o-mini-transcribe' failed: {e}. Trying standard 'whisper-1' model...")

        # Fallback 2: Direct OpenAI client transcription using standard "whisper-1" model
        try:
            client = OpenAI(api_key=self.api_key)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            lang_code = language.split("-")[0].lower() if language else "en"
            
            logger.info("Attempting transcription using direct client with standard 'whisper-1'...")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=lang_code
            )
            transcription = transcript.text.strip()
            if transcription:
                logger.info("Successfully transcribed audio via standard 'whisper-1' model.")
                return transcription
        except Exception as e:
            logger.error(f"All Speech-to-Text transcription methods failed: {e}")
            return f"[Transcription failed: {str(e)}]"
