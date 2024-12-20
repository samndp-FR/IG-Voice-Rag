from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import asyncio
from moviepy.editor import VideoFileClip
from openai import AzureOpenAI
from supabase import create_client
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio

@dataclass
class ProcessingResult:
    success: bool
    video_path: str
    title: Optional[str] = None
    error: Optional[str] = None

class VideoProcessor:
    def __init__(self, settings):
        self.settings = settings
        self._setup_clients()
        self._setup_logging()
        
    def _setup_clients(self):
        """Initialize API clients"""
        self.gpt4_client = AzureOpenAI(
            api_key=self.settings.gpt4_api_key,
            api_version=self.settings.azure_api_version,
            azure_endpoint=self.settings.gpt4_endpoint
        )
        
        self.whisper_client = AzureOpenAI(
            api_key=self.settings.whisper_api_key,
            api_version=self.settings.azure_api_version,
            azure_endpoint=self.settings.whisper_endpoint
        )
        
        self.ada_client = AzureOpenAI(
            api_key=self.settings.ada_api_key,
            api_version=self.settings.azure_api_version,
            azure_endpoint=self.settings.ada_endpoint
        )
        
        self.supabase = create_client(
            self.settings.supabase_url,
            self.settings.supabase_key
        )

    def _setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(f'logs/video_processing_{datetime.now():%Y%m%d_%H%M%S}.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    async def extract_audio(self, video_path: Path, output_path: Path) -> None:
        """Extract audio from video file"""
        try:
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
            video.close()
        except Exception as e:
            self.logger.error(f"Audio extraction failed for {video_path}: {str(e)}")
            raise

    async def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file using Whisper"""
        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = await asyncio.to_thread(
                    self.whisper_client.audio.transcriptions.create,
                    model=self.settings.whisper_deployment_name,
                    file=audio_file
                )
            return transcript.text
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path}: {str(e)}")
            raise

    async def generate_title(self, transcript: str) -> str:
        """Generate title using GPT-4"""
        try:
            response = await asyncio.to_thread(
                self.gpt4_client.chat.completions.create,
                model=self.settings.gpt4_deployment_name,
                messages=[
                    {"role": "system", "content": "Generate a concise title starting with 'How To' for a video based on its transcript."},
                    {"role": "user", "content": f"Generate a title for: {transcript[:500]}..."}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Title generation failed: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ada"""
        try:
            response = await asyncio.to_thread(
                self.ada_client.embeddings.create,
                model=self.settings.ada_deployment_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def store_in_supabase(self, data: Dict[str, Any]) -> None:
        """Store processed data in Supabase"""
        try:
            await asyncio.to_thread(
                self.supabase.table('video_content').insert(data).execute
            )
        except Exception as e:
            self.logger.error(f"Database storage failed: {str(e)}")
            raise

    async def process_video(self, video_path: Path, temp_dir: Path) -> ProcessingResult:
        """Process a single video file"""
        self.logger.info(f"Processing video: {video_path}")
        
        try:
            # Create temporary directory for this video
            video_temp_dir = temp_dir / video_path.stem
            video_temp_dir.mkdir(exist_ok=True)
            
            # Extract and process audio
            audio_path = video_temp_dir / f"{video_path.stem}.wav"
            await self.extract_audio(video_path, audio_path)
            
            # Generate transcript
            transcript = await self.transcribe_audio(audio_path)
            
            # Generate title and embedding in parallel
            title, embedding = await asyncio.gather(
                self.generate_title(transcript),
                self.generate_embedding(transcript)
            )
            
            # Prepare data for storage
            data = {
                "title": title,
                "script": transcript,
                "embedding": embedding,
                "url": str(video_path),
                "metadata": {
                    "processed_date": datetime.utcnow().isoformat(),
                    "original_filename": video_path.name,
                    "file_path": str(video_path)
                }
            }
            
            # Store in database
            await self.store_in_supabase(data)
            
            # Cleanup
            audio_path.unlink()
            video_temp_dir.rmdir()
            
            self.logger.info(f"Successfully processed: {title}")
            return ProcessingResult(success=True, video_path=str(video_path), title=title)
            
        except Exception as e:
            self.logger.error(f"Failed to process {video_path}: {str(e)}")
            return ProcessingResult(success=False, video_path=str(video_path), error=str(e))

    async def process_all_videos(self):
        """Process all videos in the specified folder"""
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        video_folder = Path(self.settings.video_folder_path)
        videos = list(video_folder.glob("*.mp4"))
        
        self.logger.info(f"Found {len(videos)} videos to process")
        
        results = []
        async for video in tqdm_asyncio(videos, desc="Processing videos"):
            result = await self.process_video(video, temp_dir)
            results.append(result)
            
            # Save progress
            with open('processing_progress.json', 'w') as f:
                json.dump({
                    'successful': [r.video_path for r in results if r.success],
                    'failed': [r.video_path for r in results if not r.success],
                    'total': len(videos),
                    'completed': len(results)
                }, f, indent=2)
        
        # Cleanup main temp directory
        temp_dir.rmdir()
        
        return results

def main():
    from config.settings import get_settings
    
    async def run():
        processor = VideoProcessor(get_settings())
        results = await processor.process_all_videos()
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print("\nProcessing Complete!")
        print(f"Successfully processed: {len(successful)} videos")
        print(f"Failed to process: {len(failed)} videos")
        
        if failed:
            print("\nFailed videos:")
            for result in failed:
                print(f"- {result.video_path}: {result.error}")
    
    asyncio.run(run())

if __name__ == "__main__":
    main()