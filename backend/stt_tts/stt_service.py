from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class SpeechToTextService:
    """Transcribes farmer audio messages using gpt-4o-transcribe."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        # Initialize ChatOpenAI with the specific transcription model
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-transcribe",
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
            
        import base64
        import io
        
        try:
            # Try audio modalities interface using ChatOpenAI
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
            return response.content.strip()
        except Exception:
            # Fallback to direct OpenAI client transcription endpoint
            try:
                client = self.llm.client
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"
                
                # Check for standard language ISO code conversions
                lang_code = language.split("-")[0].lower() if language else "en"
                
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language=lang_code
                )
                return transcript.text.strip()
            except Exception as e:
                return f"[Transcription failed: {str(e)}]"

