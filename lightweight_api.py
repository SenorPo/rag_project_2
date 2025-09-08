from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from typing import List, Dict, Optional
import google.generativeai as genai

app = FastAPI()

# In-memory storage for conversation context
conversations: Dict[str, List[Dict]] = {}


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@app.post("/chat")
async def chat_endpoint(chat_request: ChatMessage):
    try:
        session_id = chat_request.session_id
        user_message = chat_request.message

        # Get conversation history for this session
        if session_id not in conversations:
            conversations[session_id] = []

        conversation = conversations[session_id]

        # Search vector database
        relevant_docs = search_docs(user_message)
        context = "\n".join([d["text"][:300] for d in relevant_docs])

        # Build conversation context (last 5 messages)
        conversation_context = ""
        for msg in conversation[-5:]:  # Last 5 exchanges
            conversation_context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

        # Create enhanced prompt with context
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Previous conversation:
{conversation_context}

Context from documents:
{context}

Current question: {user_message}

Answer the current question considering both the document context and previous conversation:"""

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
            "session_id": session_id
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