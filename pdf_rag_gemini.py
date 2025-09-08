import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import tempfile
import shutil
import google.generativeai as genai

# Configure page
st.set_page_config(page_title="PDF Chat with RAG (Hybrid)", page_icon="📚", layout="wide")
st.title("📚 Chat with Your PDFs using RAG + Hybrid Model")
st.markdown("Upload PDF documents and ask questions using Sentence Transformers embeddings + Gemini generation!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API Key input - check environment variable first
    default_api_key = os.getenv("GOOGLE_API_KEY", "")
    gemini_api_key = st.text_input("Google Gemini API Key",
                                   value=default_api_key,
                                   type="password",
                                   help="Enter your Google Gemini API key or set GOOGLE_API_KEY environment variable")

    # Model selection
    model_name = st.selectbox(
        "Select Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        index=0,
        help="Gemini 1.5 Flash is fastest and most cost-effective"
    )

    # Chunk size configuration
    chunk_size = st.slider("Chunk Size", 200, 1000, 512,
                           help="Size of text chunks for processing")

    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50,
                              help="Overlap between consecutive chunks")

    # Temperature setting
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.1,
                            help="Higher values make responses more creative")


def setup_rag_system(uploaded_files, api_key, model, chunk_size, chunk_overlap, temperature):
    """Set up the RAG system with uploaded documents using Hybrid approach"""

    if not api_key:
        st.error("Please provide a Google Gemini API key")
        return None

    # Configure Gemini API
    genai.configure(api_key=api_key)

    try:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()

        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Load documents
        with st.spinner("Loading documents..."):
            loader = SimpleDirectoryReader(temp_dir)
            documents = loader.load_data()

        # Configure Gemini LLM
        llm = Gemini(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

        # Configure Sentence Transformers embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Configure node parser
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser

        # Create vector store index
        with st.spinner("Creating vector index with Sentence Transformers embeddings..."):
            index = VectorStoreIndex.from_documents(documents)

        # Create query engine with custom prompt
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
            streaming=False
        )

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return query_engine, len(documents)

    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        # Clean up temporary directory in case of error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None


# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with"
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        for file in uploaded_files:
            st.write(f"• {file.name}")

        if st.button("Process Documents", type="primary"):
            result = setup_rag_system(
                uploaded_files,
                gemini_api_key,
                model_name,
                chunk_size,
                chunk_overlap,
                temperature
            )

            if result:
                st.session_state.query_engine, doc_count = result
                st.session_state.documents_loaded = True
                st.session_state.messages = []  # Clear previous messages
                st.success(f"✅ Successfully processed {doc_count} documents with Sentence Transformers + Gemini!")
                st.rerun()

with col2:
    st.header("Chat Interface")

    if not st.session_state.documents_loaded:
        st.info("👈 Upload and process PDF documents to start chatting!")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.query_engine.query(prompt)
                        response_text = str(response)
                        st.markdown(response_text)

                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })

                        # Show source information if available
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            with st.expander("📄 Source Information"):
                                for i, node in enumerate(response.source_nodes):
                                    st.write(f"**Source {i + 1}:**")
                                    if hasattr(node, 'score'):
                                        st.write(f"Relevance Score: {node.score:.3f}")
                                    st.write(f"Text Preview: {node.text[:300]}...")
                                    st.write("---")

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

# Clear chat history button
if st.session_state.documents_loaded:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Instructions and tips
with st.expander("ℹ️ How to Use - Hybrid Model"):
    st.markdown("""
    ### Getting Started:
    1. **Get your Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Enter your API Key** in the sidebar
    3. **Upload PDF files** using the file uploader
    4. **Click "Process Documents"** to create the RAG index with Sentence Transformers embeddings
    5. **Start chatting** with your documents powered by the hybrid model!

    ### Hybrid Model Benefits:
    - **🎯 Superior Retrieval**: Uses proven all-mpnet-base-v2 embeddings for better document matching
    - **⚡ Fast Generation**: Gemini 1.5 Flash for quick, high-quality responses
    - **💰 Cost Effective**: No API costs for embeddings, only for generation
    - **🔍 Better Accuracy**: Combines best-in-class retrieval with modern generation

    ### Model Architecture:
    - **Embeddings**: sentence-transformers/all-mpnet-base-v2 (local, specialized for semantic search)
    - **Generation**: Gemini 1.5 Flash (fast, capable, cost-effective)
    - **Best of both worlds**: Proven retrieval + modern generation

    ### Tips for Better Results:
    - **Adjust chunk size**: Smaller chunks (200-400) for specific facts, larger chunks (600-1000) for broader context
    - **Use specific questions**: Instead of "What is this about?", try "What are the main conclusions about X?"
    - **Reference specific topics**: Mention key terms or concepts from your documents
    - **Check sources**: Expand the "Source Information" to see which parts of your documents were used
    """)

# Model comparison info
with st.expander("🔍 Embedding Model Comparison"):
    st.markdown("""
    | Model | Type | Best For | Speed | Quality |
    |-------|------|----------|-------|---------|
    | **all-mpnet-base-v2** | Local Sentence Transformers | Document retrieval, semantic search | ⚡⚡ Fast | 🎯🎯🎯 Excellent |
    | **Gemini Embeddings** | API-based | General purpose tasks | ⚡ Variable | 🎯🎯 Good |
    | **OpenAI Embeddings** | API-based | General purpose, large context | ⚡ Variable | 🎯🎯 Good |

    **Why all-mpnet-base-v2?**
    - Specifically trained for semantic similarity tasks
    - Proven performance on retrieval benchmarks
    - Runs locally (no API costs or latency)
    - Optimized for finding relevant document chunks
    """)

# Footer
st.markdown("---")
st.markdown(
    "🤖 **Hybrid Architecture**: This app combines Sentence Transformers (embeddings) + Google Gemini (generation) "
    "for optimal performance. Embeddings run locally while generation uses Gemini API."
)