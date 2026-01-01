"""
Supabase Storage Service for handling audio file uploads, downloads, and deletions.
Files are automatically set to expire after 3 days.
"""
import os
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Tuple


class StorageService:
    """Service for managing audio files in Supabase Storage"""
    
    BUCKET_NAME = "call-recordings"
    RETENTION_DAYS = 3
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Supabase client"""
        if self._client is None:
            from supabase import create_client
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required. "
                    "Please set them in your .env file."
                )
            
            self._client = create_client(supabase_url, supabase_key)
        return self._client
    
    async def ensure_bucket_exists(self) -> None:
        """Create the storage bucket if it doesn't exist"""
        try:
            # Try to get bucket info
            self.client.storage.get_bucket(self.BUCKET_NAME)
        except Exception:
            # Bucket doesn't exist, create it
            try:
                self.client.storage.create_bucket(
                    self.BUCKET_NAME,
                    options={
                        "public": False,
                        "file_size_limit": 26214400,  # 25MB limit for Whisper API
                    }
                )
            except Exception as e:
                # Bucket might already exist (race condition) or other error
                if "already exists" not in str(e).lower():
                    raise
    
    async def upload_audio(self, file_content: bytes, filename: str, call_id: str) -> Tuple[str, datetime]:
        """
        Upload an audio file to Supabase Storage.
        
        Args:
            file_content: The audio file content as bytes
            filename: Original filename
            call_id: Unique identifier for the call
            
        Returns:
            Tuple of (storage_path, expiration_datetime)
        """
        await self.ensure_bucket_exists()
        
        # Generate storage path with call_id
        file_extension = os.path.splitext(filename)[1] or ".mp3"
        storage_path = f"{call_id}{file_extension}"
        
        # Upload to Supabase Storage
        self.client.storage.from_(self.BUCKET_NAME).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": self._get_content_type(file_extension)}
        )
        
        # Calculate expiration date
        expires_at = datetime.utcnow() + timedelta(days=self.RETENTION_DAYS)
        
        return storage_path, expires_at
    
    async def get_audio_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """
        Get a signed URL for accessing the audio file.
        
        Args:
            storage_path: Path to the file in storage
            expires_in: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Signed URL for file access
        """
        result = self.client.storage.from_(self.BUCKET_NAME).create_signed_url(
            path=storage_path,
            expires_in=expires_in
        )
        # Handle both dict and object responses
        if isinstance(result, dict):
            return result.get("signedURL") or result.get("signedUrl", "")
        return result.signed_url if hasattr(result, 'signed_url') else str(result)
    
    async def download_audio(self, storage_path: str) -> bytes:
        """
        Download an audio file from Supabase Storage.
        
        Args:
            storage_path: Path to the file in storage
            
        Returns:
            File content as bytes
        """
        result = self.client.storage.from_(self.BUCKET_NAME).download(storage_path)
        return result
    
    async def download_to_temp_file(self, storage_path: str) -> str:
        """
        Download audio file to a temporary file for processing.
        
        Args:
            storage_path: Path to the file in storage
            
        Returns:
            Path to the temporary file
        """
        content = await self.download_audio(storage_path)
        
        # Get file extension
        file_extension = os.path.splitext(storage_path)[1] or ".mp3"
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(content)
            return temp_path
        except Exception:
            os.close(temp_fd)
            raise
    
    async def delete_audio(self, storage_path: str) -> bool:
        """
        Delete an audio file from Supabase Storage.
        
        Args:
            storage_path: Path to the file in storage
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.storage.from_(self.BUCKET_NAME).remove([storage_path])
            return True
        except Exception as e:
            print(f"Error deleting file {storage_path}: {e}")
            return False
    
    async def delete_expired_recordings(self) -> dict:
        """
        Delete all recordings that have passed their expiration date.
        This should be called by a scheduled cron job.
        
        Returns:
            Dictionary with deletion statistics
        """
        # This method works with the database to find and delete expired recordings
        # The actual database query will be done in the router/endpoint
        pass
    
    def _get_content_type(self, file_extension: str) -> str:
        """Get the content type for a file extension"""
        content_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/m4a",
            ".webm": "audio/webm",
            ".ogg": "audio/ogg",
        }
        return content_types.get(file_extension.lower(), "audio/mpeg")
    
    @staticmethod
    def cleanup_temp_file(temp_path: str) -> None:
        """Clean up a temporary file"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_path}: {e}")


# Singleton instance
storage_service = StorageService()
