# hybrid_admin.py - Enhanced admin for hybrid approach
import streamlit as st
import sys
from pathlib import Path
from export_vector_db import export_with_embeddings

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def hybrid_admin_interface():
    """Admin interface with export functionality"""

    st.set_page_config(page_title="RAG Admin - Hybrid Mode", page_icon="🔧", layout="wide")
    st.title("🔧 RAG System - Hybrid Admin Interface")
    st.markdown("**Laptop Processing → Cloud Deployment**")

    # Import your existing admin interface
    from admin_interface import admin_interface

    # Show regular admin interface
    admin_interface()

    # Add export section
    st.markdown("---")
    st.header("📤 Export for Cloud Deployment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Vector Database")
        st.write("After processing documents, export the vector database for cloud deployment.")

        if st.button("🚀 Export Vector Database", type="primary"):
            with st.spinner("Exporting vector database..."):
                if export_with_embeddings():
                    st.success("✅ Export successful! Upload 'vector_database.json' to Replit")
                else:
                    st.error("❌ Export failed. Check console for details.")

    with col2:
        st.subheader("Export Instructions")
        st.markdown("""
        **Steps:**
        1. Upload and process documents above
        2. Click "Export Vector Database"
        3. Upload `vector_database.json` to Replit
        4. Your cloud API will use the exported data

        **File Location:**
        `vector_database.json` will be created in your project folder
        """)


if __name__ == "__main__":
    hybrid_admin_interface()