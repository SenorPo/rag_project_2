from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from storage_manager import StorageManager
from config import Config
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import tempfile
import os
import shutil

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentInfo(BaseModel):
    filename: str
    size: int
    uploaded_at: str
    hash: str

class SystemStatus(BaseModel):
    document_count: int
    index_exists: bool
    last_updated: Optional[str]
    storage_size: Dict[str, int]

class UploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="REST API for the RAG (Retrieval-Augmented Generation) system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
storage_manager = StorageManager()
security = HTTPBearer()

# Global query engine cache
query_engine = None

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication token"""
    if credentials.credentials != config.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

def get_or_load_query_engine():
    """Get cached query engine or load from storage"""
    global query_engine
    
    if query_engine is None:
        if not config.validate_required_config():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google API key not configured"
            )
        
        if not storage_manager.has_stored_index():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge base available. Upload documents first."
            )
        
        try:
            # Configure Gemini
            genai.configure(api_key=config.GOOGLE_API_KEY)
            
            # Load index
            index = storage_manager.load_index()
            if not index:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to load vector index"
                )
            
            # Configure LLM
            llm = Gemini(
                model=config.DEFAULT_MODEL,
                temperature=config.DEFAULT_TEMPERATURE,
                api_key=config.GOOGLE_API_KEY
            )
            Settings.llm = llm
            
            # Create query engine
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                verbose=True,
                streaming=False
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading query engine: {str(e)}"
            )
    
    return query_engine

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "status": "/status",
            "documents": "/admin/documents",
            "upload": "/admin/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_configured": config.validate_required_config(),
        "index_available": storage_manager.has_stored_index(),
        "document_count": storage_manager.get_document_count()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the knowledge base"""
    try:
        engine = get_or_load_query_engine()
        
        # Override model settings if provided
        if request.model or request.temperature is not None:
            llm = Gemini(
                model=request.model or config.DEFAULT_MODEL,
                temperature=request.temperature if request.temperature is not None else config.DEFAULT_TEMPERATURE,
                api_key=config.GOOGLE_API_KEY
            )
            Settings.llm = llm
        
        # Query the engine
        response = engine.query(request.message)
        response_text = str(response)
        
        # Extract source information
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                source_info = {
                    "source_id": i + 1,
                    "text_preview": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                }
                if hasattr(node, 'score'):
                    source_info["relevance_score"] = float(node.score)
                sources.append(source_info)
        
        return ChatResponse(
            response=response_text,
            sources=sources if sources else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and information"""
    storage_info = storage_manager.get_storage_info()
    return SystemStatus(**storage_info)

@app.get("/admin/documents", response_model=List[DocumentInfo])
async def list_documents(credentials: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """List all stored documents (Admin only)"""
    docs = storage_manager.get_stored_documents()
    return [
        DocumentInfo(
            filename=filename,
            size=info["size"],
            uploaded_at=info["uploaded_at"],
            hash=info["hash"]
        )
        for filename, info in docs.items()
    ]

@app.post("/admin/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    credentials: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Upload and process documents (Admin only)"""
    global query_engine
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed."
            )
    
    try:
        model_config = {
            "model": model or config.DEFAULT_MODEL,
            "temperature": temperature if temperature is not None else config.DEFAULT_TEMPERATURE,
            "chunk_size": chunk_size or config.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
        }
        
        # Process documents
        index, doc_count = storage_manager.save_documents_and_build_index(files, model_config)
        
        if index:
            # Clear cached query engine to force reload
            query_engine = None
            
            return UploadResponse(
                success=True,
                message=f"Successfully processed {doc_count} documents",
                documents_processed=doc_count
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process documents"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing documents: {str(e)}"
        )

@app.delete("/admin/documents/{filename}")
async def delete_document(
    filename: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Delete a specific document (Admin only)"""
    global query_engine
    
    success = storage_manager.delete_document(filename)
    if success:
        # Clear cached query engine to force reload
        query_engine = None
        return {"success": True, "message": f"Document {filename} deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {filename} not found"
        )

@app.post("/admin/rebuild-index")
async def rebuild_index(credentials: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """Rebuild the vector index from existing documents (Admin only)"""
    global query_engine
    
    if storage_manager.get_document_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents available to rebuild index"
        )
    
    try:
        metadata = storage_manager._load_metadata()
        storage_manager._rebuild_index_from_storage(metadata)
        metadata["index_exists"] = True
        storage_manager._save_metadata(metadata)
        
        # Clear cached query engine to force reload
        query_engine = None
        
        return {"success": True, "message": "Index rebuilt successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rebuilding index: {str(e)}"
        )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    config.ensure_storage_paths()
    
    if not config.validate_required_config():
        print("WARNING: Google API key not configured. Some features will not work.")

if __name__ == "__main__":
    uvicorn.run(
        "api_backend:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )