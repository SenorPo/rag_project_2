# api_main.py - FastAPI entry point for Replit
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Entry point for FastAPI backend on Replit"""

    # Import your FastAPI app
    from api_backend import app

    # Replit configuration
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"🚀 Starting RAG API server on {host}:{port}")
    print(f"📋 API Documentation will be available at: http://{host}:{port}/docs")
    print(f"❤️  Health check available at: http://{host}:{port}/health")

    # Start the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False  # Disable reload in production
    )


if __name__ == "__main__":
    main()