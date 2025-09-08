import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def main():
    """Main entry point for the RAG system"""
    
    # Initialize query parameters
    query_params = st.query_params
    
    # Check if specific interface is requested
    interface = query_params.get("interface", ["user"])[0] if isinstance(query_params.get("interface"), list) else query_params.get("interface", "user")
    
    if interface == "admin":
        from admin_interface import admin_interface
        admin_interface()
    elif interface == "user":
        from user_interface import user_interface
        user_interface()
    else:
        # Show interface selection
        st.set_page_config(page_title="RAG System", page_icon="🤖", layout="wide")
        
        st.title("🤖 RAG System")
        st.markdown("### Choose your interface:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 👤 User Interface
            - Clean chat interface
            - Access to pre-built knowledge base
            - No document management
            """)
            if st.button("👤 Open User Interface", type="primary", use_container_width=True):
                st.query_params["interface"] = "user"
                st.rerun()
        
        with col2:
            st.markdown("""
            #### 🔧 Admin Interface  
            - Document upload and management
            - Vector index building
            - System configuration
            """)
            if st.button("🔧 Open Admin Interface", type="secondary", use_container_width=True):
                st.query_params["interface"] = "admin"
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### Quick Access URLs:
        - **User Interface:** `?interface=user`
        - **Admin Interface:** `?interface=admin`
        
        ### API Backend:
        Run the API server separately with: `python api_backend.py`
        - **API Documentation:** Available at `http://localhost:8000/docs`
        - **Health Check:** `http://localhost:8000/health`
        """)
        
        # Show system status
        from storage_manager import StorageManager
        storage_manager = StorageManager()
        storage_info = storage_manager.get_storage_info()
        
        st.markdown("### 📊 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", storage_info["document_count"])
        
        with col2:
            st.metric("Index Status", "✅ Ready" if storage_info["has_index"] else "❌ Missing")
        
        with col3:
            total_size_mb = storage_info["storage_size"]["total"] / 1024 / 1024
            st.metric("Storage", f"{total_size_mb:.1f} MB")

if __name__ == "__main__":
    main()