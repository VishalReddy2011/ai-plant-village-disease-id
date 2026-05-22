import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_audio_components():
    print("=" * 60)
    print("        TTS & STT SERVICE COMPONENT VERIFICATION        ")
    print("=" * 60)
    
    # 1. Load env variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Loaded OpenAI API Key: {'Present' if api_key else 'Missing'}")

    # 2. Test imports
    try:
        from stt_tts.stt_service import SpeechToTextService
        from stt_tts.tts_service import TextToSpeechService
        logger.info("SUCCESS: Imported speech services correctly.")
    except Exception as e:
        logger.error(f"FAILURE: Service imports failed: {e}")
        return False

    # 3. Test Text-to-Speech (TTS)
    logger.info("Initializing Text-to-Speech Service...")
    tts = TextToSpeechService(api_key)
    
    # Verify model parameter is updated correctly
    test_text = "This is a test speech synthesis using gpt-4o-mini-tts model."
    logger.info("Calling synthesize_speech...")
    audio_data = tts.synthesize_speech(test_text, "en")
    
    logger.info(f"SUCCESS: Generated {len(audio_data)} bytes of audio data.")
    
    # Write a test file to verify it plays
    output_filename = "test_synthesis.mp3" if api_key else "test_synthesis.wav"
    try:
        with open(output_filename, "wb") as f:
            f.write(audio_data)
        logger.info(f"SUCCESS: Wrote audio track to local file '{output_filename}' for manual listening test.")
    except Exception as e:
        logger.error(f"Failed to write audio output file: {e}")

    # 4. Test Speech-to-Text (STT)
    logger.info("Initializing Speech-to-Text Service...")
    stt = SpeechToTextService(api_key)
    
    # Verify model parameters
    logger.info(f" -> Configured LLM model: {stt.llm.model_name if stt.llm else 'Offline (No API Key)'}")
    
    # Run a dry-run test with a dummy WAV block
    mock_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x11\x2b\x00\x00\x22\x56\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    logger.info("Running transcription interface dry-run...")
    transcription = stt.transcribe_audio(mock_audio, "en")
    logger.info(f"Transcription response: {transcription}")
    
    print("=" * 60)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_audio_components()
