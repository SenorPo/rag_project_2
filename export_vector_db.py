# export_vector_db.py - Run on your laptop
import json
import os
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage
import numpy as np


def export_vector_database():
    """Export your vector database to a JSON file for upload"""

    # Check if index exists
    index_path = Path("./storage/index")
    if not index_path.exists():
        print("No vector index found. Please build the index first using admin interface.")
        return False

    try:
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage/index")
        index = load_index_from_storage(storage_context)

        # Get document store
        docstore = index.storage_context.docstore

        # Extract documents and their embeddings
        documents_data = []

        for doc_id, doc in docstore.docs.items():
            doc_data = {
                "id": doc_id,
                "text": doc.text,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            documents_data.append(doc_data)

        # Create exportable database
        export_data = {
            "documents": documents_data,
            "total_docs": len(documents_data),
            "export_version": "1.0"
        }

        # Save to JSON file
        with open("vector_database.json", "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"Vector database exported successfully!")
        print(f"Documents exported: {len(documents_data)}")
        print(f"File size: {os.path.getsize('vector_database.json') / 1024:.1f} KB")
        print("Upload 'vector_database.json' to Replit")

        return True

    except Exception as e:
        print(f"Error exporting vector database: {e}")
        return False


if __name__ == "__main__":
    export_vector_database()