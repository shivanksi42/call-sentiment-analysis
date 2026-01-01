import os
from openai import OpenAI
from typing import Dict, Any


class TranscriptionService:
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    async def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription text, duration, and detected language
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                # Use Whisper API for transcription
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            return {
                "text": transcript.text,
                "duration": transcript.duration if hasattr(transcript, 'duration') else 0,
                "language": transcript.language if hasattr(transcript, 'language') else "unknown",
                "segments": transcript.segments if hasattr(transcript, 'segments') else []
            }
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    async def transcribe_with_timestamps(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            return {
                "text": transcript.text,
                "duration": transcript.duration if hasattr(transcript, 'duration') else 0,
                "language": transcript.language if hasattr(transcript, 'language') else "unknown",
                "words": transcript.words if hasattr(transcript, 'words') else [],
                "segments": transcript.segments if hasattr(transcript, 'segments') else []
            }
        except Exception as e:
            raise Exception(f"Transcription with timestamps failed: {str(e)}")


transcription_service = TranscriptionService()

