import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration management for the RAG system"""
    
    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Admin Authentication
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    
    # Storage Configuration
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "./storage/index")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./storage/documents")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "./storage/metadata.json")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    @classmethod
    def validate_required_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            return False
        return True
    
    @classmethod
    def get_model_options(cls) -> list[str]:
        """Get available Gemini model options"""
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    @classmethod
    def ensure_storage_paths(cls) -> None:
        """Ensure all storage directories exist"""
        os.makedirs(cls.STORAGE_PATH, exist_ok=True)
        os.makedirs(cls.INDEX_PATH, exist_ok=True)
        os.makedirs(cls.DOCUMENTS_PATH, exist_ok=True)