import streamlit as st
from storage_manager import StorageManager
from config import Config
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import google.generativeai as genai

def user_interface():
    """Clean user interface for chatting with pre-built knowledge base"""
    
    # Configure page
    st.set_page_config(page_title="Chat with Knowledge Base", page_icon="💬", layout="wide")
    st.title("💬 Chat with Knowledge Base")
    st.markdown("Ask questions about the uploaded documents!")
    
    config = Config()
    storage_manager = StorageManager()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False
    
    # Check if system is configured
    if not config.validate_required_config():
        st.error("⚠️ System not configured properly.")
        st.info("Please contact your administrator to configure the Google API key.")
        return
    
    # Configure Gemini API
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    # Load index if not already loaded
    if not st.session_state.index_loaded:
        if storage_manager.has_stored_index():
            with st.spinner("Loading knowledge base..."):
                try:
                    index = storage_manager.load_index()
                    if index:
                        # Configure LLM
                        llm = Gemini(
                            model=config.DEFAULT_MODEL,
                            temperature=config.DEFAULT_TEMPERATURE,
                            api_key=config.GOOGLE_API_KEY
                        )
                        Settings.llm = llm
                        
                        # Create query engine
                        st.session_state.query_engine = index.as_query_engine(
                            response_mode="tree_summarize",
                            verbose=True,
                            streaming=False
                        )
                        st.session_state.index_loaded = True
                    else:
                        st.error("Failed to load knowledge base")
                except Exception as e:
                    st.error(f"Error loading knowledge base: {str(e)}")
        else:
            st.warning("📚 No knowledge base available.")
            st.info("Contact your administrator to upload documents and build the knowledge base.")
            return
    
    # System status in sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        storage_info = storage_manager.get_storage_info()
        st.metric("Available Documents", storage_info["document_count"])
        
        if storage_info["last_updated"]:
            st.write(f"**Last Updated:** {storage_info['last_updated'][:19].replace('T', ' ')}")
        
        # Show available documents
        if storage_info["document_count"] > 0:
            with st.expander("📄 Available Documents"):
                docs = storage_manager.get_stored_documents()
                for filename in docs.keys():
                    st.write(f"• {filename}")
        
        st.markdown("---")
        st.write("💡 **Tip:** Ask specific questions about the documents for best results!")
    
    # Main chat interface
    if st.session_state.index_loaded and st.session_state.query_engine:
        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
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
                                    if i < len(response.source_nodes) - 1:
                                        st.write("---")
                    
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Clear chat history button
        if st.session_state.messages:
            col1, col2 = st.columns([1, 1])
            with col2:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
    
    # Footer with usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        ### How to get better responses:
        
        **✅ Good Questions:**
        - "What are the main findings about [specific topic]?"
        - "How does [concept A] relate to [concept B]?"
        - "What recommendations are made regarding [specific issue]?"
        
        **❌ Avoid:**
        - Very vague questions like "What is this about?"
        - Questions about information not in the documents
        
        **🎯 Pro Tips:**
        - Be specific with your questions
        - Reference key terms from the documents
        - Check the source information to understand where answers come from
        - Use follow-up questions to dive deeper into topics
        """)

if __name__ == "__main__":
    user_interface()