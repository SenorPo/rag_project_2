"""
Lightweight RAG API - Hybrid Approach
- Uses Sentence Transformers (all-mpnet-base-v2) for query embeddings
- Uses Gemini only for text generation (LLM responses)
- Optimized for Render deployment
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_db = None
documents = []
conversations: Dict[str, List[Dict]] = {}
embedding_model = None

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

def load_vector_database():
    """Load the exported vector database"""
    global vector_db, documents

    try:
        if not os.path.exists("vector_database.json"):
            print("No vector_database.json found.")
            return False

        with open("vector_database.json", "r", encoding="utf-8") as f:
            vector_db = json.load(f)

        documents = vector_db.get("documents", [])

        # Check if we have embeddings
        docs_with_embeddings = sum(1 for doc in documents if doc.get("embedding") is not None)

        print(f"Vector database loaded: {len(documents)} documents")
        print(f"Documents with embeddings: {docs_with_embeddings}")

        return True

    except Exception as e:
        print(f"Error loading vector database: {e}")
        return False

def vector_search(query_text: str, top_k: int = 3) -> List[Dict]:
    """Vector-based search using pre-computed embeddings"""
    if not documents:
        return []

    # Filter documents that have embeddings
    docs_with_embeddings = [doc for doc in documents if doc.get("embedding") is not None]

    if not docs_with_embeddings:
        print("No embeddings found, falling back to keyword search")
        return keyword_search(query_text, top_k)

    # For now, use hybrid approach since we don't have query embeddings yet
    # This is where we'd implement proper vector similarity if we had query embeddings
    print(f"Found {len(docs_with_embeddings)} documents with embeddings")

    # Fallback to keyword search for now
    # TODO: Implement proper vector similarity when we have query embeddings
    return keyword_search(query_text, top_k)

def vector_similarity_search(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """Actual vector similarity search (for when we have query embeddings)"""
    if not documents:
        return []

    docs_with_embeddings = [doc for doc in documents if doc.get("embedding") is not None]

    if not docs_with_embeddings or not query_embedding:
        return []

    try:
        # Extract embeddings
        doc_embeddings = [doc["embedding"] for doc in docs_with_embeddings]

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = docs_with_embeddings[idx]
            results.append({
                "text": doc["text"],
                "score": float(similarities[idx]),
                "metadata": doc.get("metadata", {}),
                "search_type": "vector"
            })

        return results

    except Exception as e:
        print(f"Vector search error: {e}")
        return keyword_search(query_text, top_k)

def keyword_search(query: str, top_k: int = 3) -> List[Dict]:
    """Fallback keyword search"""
    if not documents:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    if not query_words:
        return []

    scored_docs = []
    for doc in documents:
        text_lower = doc["text"].lower()
        text_words = set(re.findall(r'\w+', text_lower))

        # Calculate simple overlap score
        overlap = len(query_words.intersection(text_words))
        if overlap > 0:
            score = overlap / len(query_words)
            scored_docs.append({
                "text": doc["text"],
                "score": score,
                "metadata": doc.get("metadata", {}),
                "search_type": "keyword"
            })

    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]

def get_sentence_transformer_embedding(text: str) -> Optional[List[float]]:
    """Get embedding using Sentence Transformers"""
    global embedding_model
    try:
        if embedding_model is None:
            print("Loading Sentence Transformer model...")
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
            print("Model loaded successfully!")
        
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        print(f"Sentence Transformer embedding error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Load vector database
    load_vector_database()
    
    # Initialize Sentence Transformer model
    print("Initializing Sentence Transformer model...")
    get_sentence_transformer_embedding("initialization test")

@app.get("/health")
async def health_check():
    docs_with_embeddings = sum(1 for doc in documents if doc.get("embedding") is not None) if documents else 0

    return {
        "status": "healthy",
        "database_loaded": vector_db is not None,
        "document_count": len(documents) if documents else 0,
        "documents_with_embeddings": docs_with_embeddings,
        "search_capability": "vector" if docs_with_embeddings > 0 else "keyword",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "llm_model": "gemini-1.5-flash",
        "deployment": "render"
    }

@app.post("/chat")
async def chat_endpoint(chat_request: ChatMessage):
    try:
        if not vector_db:
            return {"error": "Vector database not loaded"}

        session_id = chat_request.session_id
        user_message = chat_request.message

        # Get conversation history for this session
        if session_id not in conversations:
            conversations[session_id] = []

        conversation = conversations[session_id]

        # Try to get query embedding for vector search
        query_embedding = get_sentence_transformer_embedding(user_message)

        if query_embedding:
            # Use vector search if we have embeddings
            relevant_docs = vector_similarity_search(query_embedding, top_k=3)
            search_method = "vector"
        else:
            # Fallback to our existing search
            relevant_docs = vector_search(user_message, top_k=3)
            search_method = "keyword"

        # Build context from relevant documents
        context = "\n".join([d["text"][:300] for d in relevant_docs])

        # Build conversation context (last 5 messages)
        conversation_context = ""
        for msg in conversation[-5:]:
            conversation_context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

        # Create enhanced prompt with context
        import google.generativeai as genai
        
        # Configure Gemini for text generation only
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY not set for text generation"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

        if conversation_context:
            prompt = f"""Previous conversation:
{conversation_context}

Context from documents:
{context}

Current question: {user_message}

Answer the current question considering both the document context and previous conversation:"""
        else:
            prompt = f"""Context from documents:
{context}

Question: {user_message}

Answer based on the provided context:"""

        response = model.generate_content(prompt)
        assistant_response = response.text

        # Store this exchange
        conversation.append({
            "user": user_message,
            "assistant": assistant_response
        })

        # Keep only last 10 exchanges to manage memory
        if len(conversation) > 10:
            conversation = conversation[-10:]
            conversations[session_id] = conversation

        return {
            "response": assistant_response,
            "session_id": session_id,
            "sources_found": len(relevant_docs),
            "search_method": search_method,
            "document_scores": [{"score": doc.get("score", 0), "type": doc.get("search_type", "unknown")} for doc in relevant_docs]
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "messages": conversations.get(session_id, []),
        "message_count": len(conversations.get(session_id, []))
    }

@app.get("/database-info")
async def database_info():
    """Get information about the loaded database"""
    if not vector_db:
        return {"error": "No database loaded"}

    docs_with_embeddings = sum(1 for doc in documents if doc.get("embedding") is not None)

    return {
        "total_documents": len(documents),
        "documents_with_embeddings": docs_with_embeddings,
        "export_version": vector_db.get("export_version", "unknown"),
        "embedding_model": vector_db.get("embedding_model", "unknown"),
        "sample_document": documents[0]["text"][:200] + "..." if documents else None,
        "has_vector_search": docs_with_embeddings > 0
    }

@app.get("/search-test")
async def test_search(query: str = "tea benefits"):
    """Test endpoint to compare search methods"""
    keyword_results = keyword_search(query, top_k=3)

    # Try vector search if embeddings available
    query_embedding = get_sentence_transformer_embedding(query)
    if query_embedding:
        vector_results = vector_similarity_search(query_embedding, top_k=3)
    else:
        vector_results = []

    return {
        "query": query,
        "keyword_results": keyword_results,
        "vector_results": vector_results,
        "has_embeddings": len(vector_results) > 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)