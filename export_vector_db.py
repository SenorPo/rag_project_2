# export_vector_db.py - Enhanced version
import json
import os
import numpy as np
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage


def export_with_embeddings():
    """Export both text and vector embeddings"""

    storage_path = Path("./storage/index")
    if not storage_path.exists():
        print("No vector index found.")
        return False

    try:
        # Load the full index
        storage_context = StorageContext.from_defaults(persist_dir="./storage/index")
        index = load_index_from_storage(storage_context)

        # Get docstore and vector store
        docstore = index.storage_context.docstore
        vector_store = index.storage_context.vector_store

        documents_data = []

        for doc_id, doc in docstore.docs.items():
            text_content = doc.text

            # Try to get the embedding for this document
            try:
                # Different vector stores have different methods
                if hasattr(vector_store, 'get'):
                    embedding = vector_store.get(doc_id)
                elif hasattr(vector_store, '_data') and doc_id in vector_store._data.embedding_dict:
                    embedding = vector_store._data.embedding_dict[doc_id]
                else:
                    embedding = None

                # Convert to list if it's a numpy array
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
            except:
                embedding = None

            doc_data = {
                "id": doc_id,
                "text": text_content,
                "embedding": embedding,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            documents_data.append(doc_data)

        # Export data
        export_data = {
            "documents": documents_data,
            "total_docs": len(documents_data),
            "export_version": "2.0-with-embeddings",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }

        # Save to JSON
        with open("vector_database.json", "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        embeddings_count = sum(1 for doc in documents_data if doc["embedding"] is not None)
        file_size = os.path.getsize("vector_database.json") / 1024

        print(f"Export successful!")
        print(f"Documents: {len(documents_data)}")
        print(f"With embeddings: {embeddings_count}")
        print(f"File size: {file_size:.1f} KB")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    export_with_embeddings()