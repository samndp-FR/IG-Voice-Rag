from dataclasses import dataclass
from typing import Union
import logging
from openai import AzureOpenAI
import requests
from src.config.settings import get_settings

@dataclass
class AudioResult:
    success: bool
    content: Union[str, bytes]  # str for text, bytes for audio
    error: str = None

class VoiceService:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Whisper
        self.whisper_client = AzureOpenAI(
            api_key=self.settings.whisper_api_key,
            api_version=self.settings.azure_api_version,
            azure_endpoint=self.settings.whisper_endpoint
        )
        
        # ElevenLabs settings
        self.elevenlabs_api_key = self.settings.elevenlabs_api_key
        self.elevenlabs_voice_id = self.settings.elevenlabs_voice_id

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def text_to_speech(self, text: str) -> AudioResult:
        """Convert text to speech using ElevenLabs"""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(url, json=data, headers=headers)
            if response.status_code != 200:
                return AudioResult(False, bytes(), f"ElevenLabs API error: {response.status_code}")
            
            return AudioResult(True, response.content)

        except Exception as e:
            self.logger.error(f"Text-to-speech error: {str(e)}")
            return AudioResult(False, bytes(), str(e))

    def speech_to_text(self, audio_data: bytes) -> AudioResult:
        """Convert speech to text using Azure Whisper"""
        try:
            # Transcribe with Whisper
            transcript = self.whisper_client.audio.transcriptions.create(
                model=self.settings.whisper_deployment_name,
                file=('audio.wav', audio_data)
            )

            if not transcript or not transcript.text:
                return AudioResult(False, "", "No speech could be recognized")

            return AudioResult(True, transcript.text)

        except Exception as e:
            self.logger.error(f"Speech-to-text error: {str(e)}")
            return AudioResult(False, "", str(e))