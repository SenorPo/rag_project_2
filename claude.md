# RAG System Transformation Project

## Project Overview
Transform the existing Streamlit RAG application into a production-ready system with separated admin/user interfaces and persistent storage capabilities.

## Current Architecture
- Single Streamlit app where users upload PDFs and chat
- Real-time document processing for each session
- No persistent storage of vector indices
- Uses Gemini + Sentence Transformers (all-mpnet-base-v2)

## Target Architecture
- **Admin Interface**: Document management and index building
- **User Interface**: Clean chat interface with pre-built knowledge base
- **Persistent Storage**: Save/load vector indices from disk
- **API Backend**: FastAPI endpoints for mobile integration
**IMPORTANT: API Key Management**
- Remove API key input from UI - use environment variables instead
- Create .env file for configuration (GOOGLE_API_KEY, ADMIN_PASSWORD, etc.)
- Add .env.example template for deployment
- Create config.py to handle environment variable loading
- Ensure API keys are never exposed to frontend users

## Key Files
- `rag_3.py` - Current Streamlit application
- `admin_interface.py` - To be created for document management
- `user_interface.py` - To be created for user chat
- `api_backend.py` - To be created for FastAPI endpoints
- `storage_manager.py` - To be created for persistent storage handling

## Technical Requirements
- Maintain existing hybrid model (Sentence Transformers + Gemini)
- Implement persistent vector storage using LlamaIndex
- Create separate admin/user authentication flows
- Add document versioning and management capabilities
- Build REST API endpoints for mobile integration

## Environment Setup
```bash
# Existing dependencies (keep these)
pip install streamlit llama-index google-generativeai sentence-transformers

# New dependencies (add these)
pip install fastapi uvicorn python-multipart
```

## Storage Structure
```
project_root/
├── storage/
│   ├── index/ (vector index storage)
│   ├── documents/ (uploaded documents)
│   └── metadata.json (document tracking)
├── admin_interface.py
├── user_interface.py
├── api_backend.py
├── storage_manager.py
└── main.py (entry point)
```

## Key Implementation Notes
1. **Environment Variables**: Use .env file for API keys and configuration
2. **Security**: Never commit .env file, provide .env.example template
3. **Preserve existing functionality**: Keep all current RAG capabilities
4. **Backward compatibility**: Ensure smooth transition from current system
5. **Local deployment**: Optimize for local server hosting
6. **Admin authentication**: Use environment-based password
7. **Performance**: Implement efficient index loading/saving
8. **Configuration management**: Centralized config handling

## Success Criteria
- [ ] Admin can upload and manage documents
- [ ] Vector index persists between sessions
- [ ] Users get instant chat access (no upload needed)
- [ ] FastAPI provides clean REST endpoints
- [ ] System maintains current chat quality
- [ ] Easy deployment on local server