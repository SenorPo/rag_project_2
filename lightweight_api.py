# lightweight_api.py - For Replit deployment
from fastapi import FastAPI, HTTPException
import json
import os
from typing import List, Dict
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot - Hybrid API", version="1.0.0")

# Global variables
vector_db = None
documents = []


def load_vector_database():
    """Load the exported vector database"""
    global vector_db, documents

    try:
        if not os.path.exists("vector_database.json"):
            print("No vector_database.json found. Please upload the exported database.")
            return False

        with open("vector_database.json", "r", encoding="utf-8") as f:
            vector_db = json.load(f)

        documents = vector_db.get("documents", [])
        print(f"Vector database loaded: {len(documents)} documents")
        return True

    except Exception as e:
        print(f"Error loading vector database: {e}")
        return False


def simple_text_search(query: str, top_k: int = 3) -> List[Dict]:
    """Simple keyword-based search (replace with proper vector search if needed)"""
    if not documents:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    # Score documents based on keyword overlap
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
                "metadata": doc.get("metadata", {})
            })

    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set")
    else:
        genai.configure(api_key=api_key)

    # Load vector database
    load_vector_database()


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API - Hybrid Mode", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": vector_db is not None,
        "document_count": len(documents) if documents else 0
    }


@app.post("/chat")
async def chat_endpoint(message: dict):
    """Chat endpoint using exported vector database"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="Vector database not loaded")

        query = message.get("message", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty message")

        # Search for relevant documents
        relevant_docs = simple_text_search(query, top_k=3)

        if not relevant_docs:
            # No relevant documents found, use general response
            context = "No specific documents found for this query."
        else:
            # Build context from relevant documents
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"Document {i} (Score: {doc['score']:.2f}):\n{doc['text'][:500]}...")
            context = "\n\n".join(context_parts)

        # Generate response with Gemini
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

        prompt = f"""You are a helpful assistant that answers questions based on provided documents.

Context from documents:
{context}

User question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise and helpful
- If no documents were found, provide a general helpful response

Answer:"""

        response = model.generate_content(prompt)

        return {
            "response": response.text,
            "sources_found": len(relevant_docs),
            "context_used": bool(relevant_docs)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/database-info")
async def database_info():
    """Get information about the loaded database"""
    if not vector_db:
        return {"error": "No database loaded"}

    return {
        "total_documents": len(documents),
        "export_version": vector_db.get("export_version", "unknown"),
        "sample_document": documents[0]["text"][:200] + "..." if documents else None
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)