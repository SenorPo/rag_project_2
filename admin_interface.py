import streamlit as st
import os
from storage_manager import StorageManager
from config import Config
import google.generativeai as genai

def admin_interface():
    """Admin interface for document management"""
    
    # Configure page
    st.set_page_config(page_title="RAG Admin Panel", page_icon="🔧", layout="wide")
    st.title("🔧 RAG System Admin Panel")
    
    config = Config()
    storage_manager = StorageManager()
    
    # Check authentication
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.markdown("### Admin Authentication Required")
        password = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if password == config.ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
        return
    
    # Logout button
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Logout", type="secondary"):
            st.session_state.admin_authenticated = False
            st.rerun()
    
    # Check API key configuration
    if not config.validate_required_config():
        st.error("⚠️ Google API Key not configured. Please check your .env file or environment variables.")
        st.info("Add GOOGLE_API_KEY to your .env file or set it as an environment variable.")
        return
    
    # Configure Gemini API
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    # Sidebar for system configuration
    with st.sidebar:
        st.header("System Configuration")
        
        # Model configuration
        model_name = st.selectbox(
            "Gemini Model",
            config.get_model_options(),
            index=0
        )
        
        chunk_size = st.slider("Chunk Size", 200, 1000, config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, config.DEFAULT_CHUNK_OVERLAP)
        temperature = st.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE)
        
        st.markdown("---")
        
        # Storage information
        storage_info = storage_manager.get_storage_info()
        st.subheader("📊 Storage Status")
        st.metric("Documents", storage_info["document_count"])
        st.metric("Index Status", "✅ Built" if storage_info["has_index"] else "❌ Missing")
        
        if storage_info["last_updated"]:
            st.write(f"Last Updated: {storage_info['last_updated'][:19].replace('T', ' ')}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Document Management", "🔄 Index Management", "📊 System Status"])
    
    with tab1:
        st.header("Document Upload & Management")
        
        # Upload new documents
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to add to the knowledge base"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                st.write(f"• {file.name}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📤 Process & Add Documents", type="primary"):
                    model_config = {
                        "model": model_name,
                        "temperature": temperature,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap
                    }
                    
                    with st.spinner("Processing documents and building index..."):
                        try:
                            index, doc_count = storage_manager.save_documents_and_build_index(
                                uploaded_files, model_config
                            )
                            
                            if index:
                                st.success(f"✅ Successfully processed {doc_count} documents!")
                                st.rerun()
                            else:
                                st.error("Failed to process documents")
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
        
        # Existing documents management
        st.subheader("Existing Documents")
        stored_docs = storage_manager.get_stored_documents()
        
        if stored_docs:
            for filename, info in stored_docs.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 **{filename}**")
                    st.caption(f"Size: {info.get('size', 0):,} bytes | "
                              f"Added: {info.get('uploaded_at', 'Unknown')[:19].replace('T', ' ')}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{filename}"):
                        if storage_manager.delete_document(filename):
                            st.success(f"Deleted {filename}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {filename}")
        else:
            st.info("No documents in storage. Upload some PDFs to get started!")
    
    with tab2:
        st.header("Vector Index Management")
        
        if storage_manager.has_stored_index():
            st.success("✅ Vector index is available")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔄 Rebuild Index", type="secondary"):
                    if storage_manager.get_document_count() > 0:
                        with st.spinner("Rebuilding vector index..."):
                            try:
                                metadata = storage_manager._load_metadata()
                                storage_manager._rebuild_index_from_storage(metadata)
                                st.success("✅ Index rebuilt successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error rebuilding index: {str(e)}")
                    else:
                        st.warning("No documents available to rebuild index")
            
            with col2:
                if st.button("🗑️ Clear Index", type="secondary"):
                    try:
                        storage_manager._clear_index()
                        metadata = storage_manager._load_metadata()
                        metadata["index_exists"] = False
                        storage_manager._save_metadata(metadata)
                        st.success("✅ Index cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing index: {str(e)}")
        else:
            st.warning("❌ No vector index found")
            if storage_manager.get_document_count() > 0:
                if st.button("🔨 Build Index from Existing Documents"):
                    with st.spinner("Building vector index..."):
                        try:
                            metadata = storage_manager._load_metadata()
                            storage_manager._rebuild_index_from_storage(metadata)
                            metadata["index_exists"] = True
                            storage_manager._save_metadata(metadata)
                            st.success("✅ Index built successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error building index: {str(e)}")
            else:
                st.info("Upload documents first to build an index")
    
    with tab3:
        st.header("System Status & Information")
        
        # Storage metrics
        storage_info = storage_manager.get_storage_info()
        sizes = storage_info["storage_size"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", storage_info["document_count"])
            st.metric("Documents Size", f"{sizes['documents'] / 1024 / 1024:.1f} MB")
        
        with col2:
            st.metric("Index Status", "✅ Ready" if storage_info["has_index"] else "❌ Missing")
            st.metric("Index Size", f"{sizes['index'] / 1024 / 1024:.1f} MB")
        
        with col3:
            st.metric("Total Storage", f"{sizes['total'] / 1024 / 1024:.1f} MB")
        
        # Configuration display
        st.subheader("Current Configuration")
        config_data = {
            "Model": model_name,
            "Chunk Size": chunk_size,
            "Chunk Overlap": chunk_overlap,
            "Temperature": temperature,
            "Embedding Model": config.EMBEDDING_MODEL,
            "Storage Path": config.STORAGE_PATH
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}:** {value}")
        
        # Document details
        if stored_docs:
            st.subheader("Document Details")
            for filename, info in stored_docs.items():
                with st.expander(f"📄 {filename}"):
                    st.write(f"**File Hash:** `{info.get('hash', 'N/A')}`")
                    st.write(f"**Size:** {info.get('size', 0):,} bytes")
                    st.write(f"**Upload Date:** {info.get('uploaded_at', 'Unknown')}")
                    st.write(f"**Path:** `{info.get('path', 'N/A')}`")

if __name__ == "__main__":
    admin_interface()