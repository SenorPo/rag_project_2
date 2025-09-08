import os
import json
import shutil
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import tempfile
import hashlib
from config import Config

class StorageManager:
    """Handles persistent storage of vector indices and document metadata"""
    
    def __init__(self):
        self.config = Config()
        self.config.ensure_storage_paths()
        self.metadata_file = self.config.METADATA_PATH
        
    def _load_metadata(self) -> Dict:
        """Load document metadata from file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"documents": {}, "last_updated": None, "index_exists": False}
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save document metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_stored_documents(self) -> Dict:
        """Get information about stored documents"""
        metadata = self._load_metadata()
        return metadata.get("documents", {})
    
    def has_stored_index(self) -> bool:
        """Check if a vector index exists in storage"""
        metadata = self._load_metadata()
        return metadata.get("index_exists", False) and os.path.exists(self.config.INDEX_PATH)
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load existing vector index from storage"""
        if not self.has_stored_index():
            return None
            
        try:
            # Configure embeddings
            embed_model = HuggingFaceEmbedding(model_name=self.config.EMBEDDING_MODEL)
            Settings.embed_model = embed_model
            
            # Load index from storage
            storage_context = StorageContext.from_defaults(persist_dir=self.config.INDEX_PATH)
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Error loading index: {e}")
            return None
    
    def save_documents_and_build_index(self, uploaded_files: List, model_config: Dict) -> Tuple[Optional[VectorStoreIndex], int]:
        """Save documents and build/save vector index"""
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            document_info = {}
            
            # Save uploaded files and calculate metadata
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                permanent_path = os.path.join(self.config.DOCUMENTS_PATH, uploaded_file.name)
                
                # Write temporary file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Copy to permanent storage
                shutil.copy2(file_path, permanent_path)
                
                # Calculate file hash and metadata
                file_hash = self._calculate_file_hash(permanent_path)
                document_info[uploaded_file.name] = {
                    "hash": file_hash,
                    "size": os.path.getsize(permanent_path),
                    "uploaded_at": datetime.now().isoformat(),
                    "path": permanent_path
                }
            
            # Configure LLM and embeddings
            llm = Gemini(
                model=model_config.get("model", self.config.DEFAULT_MODEL),
                temperature=model_config.get("temperature", self.config.DEFAULT_TEMPERATURE),
                api_key=self.config.GOOGLE_API_KEY
            )
            
            embed_model = HuggingFaceEmbedding(model_name=self.config.EMBEDDING_MODEL)
            
            node_parser = SentenceSplitter(
                chunk_size=model_config.get("chunk_size", self.config.DEFAULT_CHUNK_SIZE),
                chunk_overlap=model_config.get("chunk_overlap", self.config.DEFAULT_CHUNK_OVERLAP)
            )
            
            # Configure global settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.node_parser = node_parser
            
            # Load documents from temporary directory
            loader = SimpleDirectoryReader(temp_dir)
            documents = loader.load_data()
            
            # Create vector store index
            index = VectorStoreIndex.from_documents(documents)
            
            # Persist index to storage
            index.storage_context.persist(persist_dir=self.config.INDEX_PATH)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata["documents"].update(document_info)
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["index_exists"] = True
            metadata["model_config"] = model_config
            self._save_metadata(metadata)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            return index, len(documents)
            
        except Exception as e:
            # Clean up on error
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document and rebuild index"""
        try:
            metadata = self._load_metadata()
            if filename not in metadata["documents"]:
                return False
            
            # Remove file from storage
            file_path = os.path.join(self.config.DOCUMENTS_PATH, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from metadata
            del metadata["documents"][filename]
            
            # If no documents left, clear index
            if not metadata["documents"]:
                self._clear_index()
                metadata["index_exists"] = False
            else:
                # Rebuild index with remaining documents
                self._rebuild_index_from_storage(metadata)
            
            metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata(metadata)
            
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def _clear_index(self) -> None:
        """Clear the vector index storage"""
        if os.path.exists(self.config.INDEX_PATH):
            shutil.rmtree(self.config.INDEX_PATH)
        os.makedirs(self.config.INDEX_PATH, exist_ok=True)
    
    def _rebuild_index_from_storage(self, metadata: Dict) -> None:
        """Rebuild index from documents in storage"""
        # Configure embeddings
        embed_model = HuggingFaceEmbedding(model_name=self.config.EMBEDDING_MODEL)
        Settings.embed_model = embed_model
        
        # Load documents from storage
        loader = SimpleDirectoryReader(self.config.DOCUMENTS_PATH)
        documents = loader.load_data()
        
        if documents:
            # Create new index
            index = VectorStoreIndex.from_documents(documents)
            
            # Clear old index and save new one
            self._clear_index()
            index.storage_context.persist(persist_dir=self.config.INDEX_PATH)
    
    def get_document_count(self) -> int:
        """Get total number of stored documents"""
        metadata = self._load_metadata()
        return len(metadata.get("documents", {}))
    
    def get_storage_info(self) -> Dict:
        """Get comprehensive storage information"""
        metadata = self._load_metadata()
        return {
            "document_count": len(metadata.get("documents", {})),
            "has_index": self.has_stored_index(),
            "last_updated": metadata.get("last_updated"),
            "documents": metadata.get("documents", {}),
            "storage_size": self._calculate_storage_size()
        }
    
    def _calculate_storage_size(self) -> Dict[str, int]:
        """Calculate storage size for different components"""
        sizes = {"documents": 0, "index": 0, "total": 0}
        
        # Calculate documents size
        if os.path.exists(self.config.DOCUMENTS_PATH):
            for filename in os.listdir(self.config.DOCUMENTS_PATH):
                filepath = os.path.join(self.config.DOCUMENTS_PATH, filename)
                if os.path.isfile(filepath):
                    sizes["documents"] += os.path.getsize(filepath)
        
        # Calculate index size
        if os.path.exists(self.config.INDEX_PATH):
            for root, dirs, files in os.walk(self.config.INDEX_PATH):
                for file in files:
                    filepath = os.path.join(root, file)
                    if os.path.isfile(filepath):
                        sizes["index"] += os.path.getsize(filepath)
        
        sizes["total"] = sizes["documents"] + sizes["index"]
        return sizes