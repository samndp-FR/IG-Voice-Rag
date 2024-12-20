from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Azure OpenAI Services
    gpt4_api_key: str = Field(..., env='GPT4_API_KEY')
    gpt4_endpoint: str = Field(..., env='GPT4_ENDPOINT')
    gpt4_deployment_name: str = Field(..., env='GPT4_DEPLOYMENT_NAME')
    
    # Azure Whisper
    whisper_api_key: str = Field(..., env='WHISPER_API_KEY')
    whisper_endpoint: str = Field(..., env='WHISPER_ENDPOINT')
    whisper_deployment_name: str = Field(..., env='WHISPER_DEPLOYMENT_NAME')
    
    # Azure Ada
    ada_api_key: str = Field(..., env='ADA_API_KEY')
    ada_endpoint: str = Field(..., env='ADA_ENDPOINT')
    ada_deployment_name: str = Field(..., env='ADA_DEPLOYMENT_NAME')
    
    # Azure Version
    azure_api_version: str = Field(..., env='AZURE_API_VERSION')
    
    # Supabase
    supabase_url: str = Field(..., env='SUPABASE_URL')
    supabase_key: str = Field(..., env='SUPABASE_KEY')
    
    # Paths
    video_folder_path: str = Field(..., env='VIDEO_FOLDER_PATH')
    
    # ElevenLabs
    elevenlabs_api_key: str = Field(..., env='ELEVENLABS_API_KEY')
    elevenlabs_voice_id: str = Field(..., env='ELEVENLABS_VOICE_ID')

    class Config:
        env_file = ".env"
        
@lru_cache
def get_settings() -> Settings:
    return Settings()